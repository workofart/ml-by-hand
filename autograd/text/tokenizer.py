import logging
import os
import pickle
from collections import Counter
from multiprocessing import Pool, cpu_count
from typing import ByteString, Dict, List, Optional, Tuple

try:
    # drop-in replacement for numpy for GPU acceleration
    import cupy as np  # type: ignore

    _ = np.cuda.runtime.getDeviceCount()  # Check if a CUDA device is available
except Exception:
    import numpy as np

import regex
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BytePairEncoder:
    """Byte Pair Encoder (BPE) for tokenizing text.

    This class implements BPE, which merges the most frequent pairs of tokens
    iteratively to learn subword units. It allows encoding raw text into
    a list of integer token IDs and decoding token IDs back into text.

    Examples:
        >>> raw_text = "Hello world! This is a test."
        >>> bpe = BytePairEncoder(num_merges=50)
        >>> encoded_array = bpe.prepare_data(raw_text) # Outputs an array of token IDs
    """

    SPECIAL_TOKENS = ["<|endoftext|>", "<PAD>", "<SOS>", "<UNK>"]

    def __init__(
        self,
        num_merges: int = 500,
        vocab_file_path: str = "vocab.pkl",
        encoded_data_path: str = "bpe_encoded_data.npz",
        n_workers: Optional[int] = None,
    ) -> None:
        """Initializes the Byte Pair Encoder.

        Args:
            num_merges (int): Number of BPE merges to learn during training.
            vocab_file_path (str): Path to store or load the pickled vocabulary.
            encoded_data_path (str): Path to store or load the encoded data as an NPZ file.
            n_workers (Optional[int]): Number of processes for parallel operations. If None,
                defaults to the number of CPU cores minus one.

        Examples:
            >>> bpe = BytePairEncoder(num_merges=100, vocab_file_path="vocab.pkl", encoded_data_path="encoded.npz")
            >>> print(bpe.num_merges)  # Expected output: 100
        """
        self.num_merges = num_merges
        self.vocab_file_path = vocab_file_path
        self.encoded_data_path = encoded_data_path

        # For storing vocab: char -> int
        self._unicode_to_int_vocab: Dict[ByteString, int] = {}
        # For storing reverse vocab: int -> char
        self._int_to_unicode_vocab: Dict[int, ByteString] = {}
        # Store merges: list of ((tokenA, tokenB), new_id)
        self.learned_merges: List[Tuple[Tuple[int, int], int]] = []

        # Load existing dictionary if available
        self._load_dictionary()

        # Start merged token IDs from the first unused index after the base vocabulary
        self.new_idx = max(self._unicode_to_int_vocab.values()) + 1

        # Parallelism
        if n_workers is None:
            self.n_workers = max(1, cpu_count() - 1)
        else:
            self.n_workers = n_workers

    @property
    def n_vocab(self) -> int:
        """int: Number of tokens (including special tokens) in the vocabulary.

        Examples:
            >>> bpe = BytePairEncoder(num_merges=50)
            >>> print(bpe.n_vocab)  # Outputs the size of the vocabulary
        """
        return len(self._unicode_to_int_vocab)

    def _load_dictionary(self) -> None:
        """Loads the dictionary (vocab + merges) from disk if it exists.

        If the vocabulary file does not exist or fails to load, creates a new base
        dictionary (all single-byte chars plus special tokens).

        Examples:
            >>> bpe = BytePairEncoder(num_merges=50, vocab_file_path="nonexistent.pkl")
            >>> bpe._load_dictionary()  # Will create a new dictionary since the file does not exist.
        """
        if not os.path.exists(self.vocab_file_path):
            logger.info(
                "Vocabulary file does not exist. Creating new dictionary from scratch."
            )
            self._unicode_to_int_vocab = self._construct_unicode_to_int_vocab()
            self._int_to_unicode_vocab = {
                v: k for k, v in self._unicode_to_int_vocab.items()
            }
            self.learned_merges = []
            return

        # If the file exists, attempt to load; fallback to new dictionary if there's an error.
        try:
            with open(self.vocab_file_path, "rb") as f:
                logger.info("Loading the vocabulary from disk.")
                data = pickle.load(f)
                (
                    self._unicode_to_int_vocab,
                    self._int_to_unicode_vocab,
                    self.learned_merges,
                ) = data
        except (pickle.UnpicklingError, EOFError, ValueError) as e:
            logger.warning(
                f"Failed to load the vocabulary from {self.vocab_file_path}. "
                f"Reason: {e}. Creating new dictionary."
            )
            self._unicode_to_int_vocab = self._construct_unicode_to_int_vocab()
            self._int_to_unicode_vocab = {
                v: k for k, v in self._unicode_to_int_vocab.items()
            }
            self.learned_merges = []

    def _construct_unicode_to_int_vocab(self) -> Dict[ByteString, int]:
        """Constructs a base vocabulary for all single-byte values plus special tokens.

        Returns:
            Dict[ByteString, int]: A dictionary mapping each single-byte (0..255) and special tokens
            to unique integer IDs.

        Examples:
            >>> bpe = BytePairEncoder(num_merges=50)
            >>> vocab = bpe._construct_unicode_to_int_vocab()
            >>> print(list(vocab.items())[:5])  # Show first 5 items in the vocabulary
        """
        unicode_to_int_vocab: Dict[ByteString, int] = {}
        for i in range(256):
            unicode_to_int_vocab[bytes([i])] = i
        # Add special tokens
        for i, special_char in enumerate(self.SPECIAL_TOKENS):
            unicode_to_int_vocab[special_char.encode("utf-8")] = 256 + i
        return unicode_to_int_vocab

    def prepare_data(
        self,
        raw_text: str,
        overwrite_vocabulary_file: bool = False,
        overwrite_encoded_data: bool = False,
        split_token: str = "<|endoftext|>",
    ) -> np.ndarray:
        """Trains and applies BPE on raw_text, returning encoded token IDs as a NumPy array.

        This method:
          1) Trains (or loads) the BPE vocabulary on the given raw_text.
          2) Encodes the text into a NumPy array of token IDs (in parallel).
          3) Caches the result to an .npz file unless already existing and not overwritten.

        Args:
            raw_text (str): The raw text from which to train or apply BPE.
            overwrite_vocabulary_file (bool): If True, re-trains and overwrites the BPE vocabulary.
            overwrite_encoded_data (bool): If True, overwrites an existing encoded .npz file.
            split_token (str): A token (usually in SPECIAL_TOKENS) used as a delimiter.

        Returns:
            np.ndarray: An array of token IDs representing the BPE-encoded text.

        Example:
            >>> raw_text = "Hello world! This is a test."
            >>> bpe = BytePairEncoder(num_merges=50)
            >>> encoded_array = bpe.prepare_data(raw_text)
            >>> print(encoded_array)
            [ ...some token IDs... ]
        """
        # 1) Train the vocabulary if needed
        self.train_vocabulary(raw_text, overwrite_saved_file=overwrite_vocabulary_file)

        # 2) Check if we already have an encoded .npz file
        if os.path.exists(self.encoded_data_path) and not overwrite_encoded_data:
            logger.info(
                f"Found existing encoded data at '{self.encoded_data_path}', "
                f"loading it instead of re-encoding."
            )
            with np.load(self.encoded_data_path, allow_pickle=True) as npz_data:
                encoded_data = npz_data["arr_0"]
        else:
            # 3) Parallel encoding
            chunk_size = max(1, len(raw_text) // self.n_workers)
            text_chunks = [
                raw_text[i : i + chunk_size]
                for i in range(0, len(raw_text), chunk_size)
            ]

            with Pool(self.n_workers) as pool:
                partial_encoded = pool.map(self.encode, text_chunks)

            encoded_data = np.array([], dtype=np.int32)
            for part in partial_encoded:
                encoded_data = np.concatenate(
                    (encoded_data, np.array(part, dtype=np.int32))
                )

            # 4) Save to disk
            np.savez_compressed(self.encoded_data_path, encoded_data)

        logger.info(f"Vocabulary size: {len(self._unicode_to_int_vocab)}")
        logger.info(f"Encoded data length: {len(encoded_data)}")
        logger.debug(f"Sample encoded data (first 50 tokens): {encoded_data[:50]}")

        return encoded_data

    def train_vocabulary(
        self, input_text: str, overwrite_saved_file: bool = False
    ) -> Tuple[Dict[ByteString, int], Dict[int, ByteString]]:
        """Trains the BPE vocabulary on the given input_text.

        The learned merges and vocabulary are saved to `vocab_file_path`.

        Args:
            input_text (str): The text to train on.
            overwrite_saved_file (bool): If True, re-trains and overwrites any existing vocabulary file.

        Returns:
            Tuple[Dict[ByteString, int], Dict[int, ByteString]]:
                A tuple containing:
                - A dictionary mapping byte sequences or special tokens to integer IDs.
                - A dictionary mapping integer IDs back to byte sequences.

        Examples:
            >>> bpe = BytePairEncoder(num_merges=50)
            >>> vocab, rev_vocab = bpe.train_vocabulary("Hello world! Hello again!")
            >>> print(vocab)  # Prints the vocabulary mapping
        """
        # If we already have a loaded vocab and don't want to overwrite, skip training
        if self._unicode_to_int_vocab and not overwrite_saved_file:
            return self._unicode_to_int_vocab, self._int_to_unicode_vocab

        text_chunks = self._pretokenize(input_text)
        logger.debug(f"Text chunks: {text_chunks[:10]}")

        word_freq = Counter()
        for chunk in text_chunks:
            if chunk in self.SPECIAL_TOKENS:
                special_id = self._unicode_to_int_vocab[chunk.encode("utf-8")]
                word_freq[(special_id,)] += 1
            else:
                base_ids = tuple(
                    self._unicode_to_int_vocab[bytes([b])]
                    for b in chunk.encode("utf-8")
                )
                word_freq[base_ids] += 1

        # Build initial pair counts
        pair_counts = self._get_initial_pair_counts(word_freq)

        # Remove pairs involving special tokens so we don't merge them
        pair_counts = {
            p: c for p, c in pair_counts.items() if p[0] < 256 and p[1] < 256
        }

        # Merge loop
        for i in tqdm(range(self.num_merges), desc="Merging pairs"):
            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            best_pair_count = pair_counts[best_pair]

            # If best pair is not frequent enough, stop merging
            if best_pair_count < 2:
                break

            new_id = self.new_idx
            self.new_idx += 1

            # Create the merged token
            merged_bytes = (
                self._int_to_unicode_vocab[best_pair[0]]
                + self._int_to_unicode_vocab[best_pair[1]]
            )
            self._int_to_unicode_vocab[new_id] = merged_bytes
            self._unicode_to_int_vocab[merged_bytes] = new_id

            # Apply merges throughout the corpus
            self._apply_merges_to_corpus(best_pair, new_id, word_freq, pair_counts)
            self.learned_merges.append((best_pair, new_id))

            if (i + 1) % 100 == 0:
                logger.info(
                    f"[{i + 1}/{self.num_merges} merge] Best Pair Merged "
                    f"({best_pair_count} occurrences). "
                    f"Vocab size: {len(self._unicode_to_int_vocab)}"
                )

        # Save the newly learned vocab
        with open(self.vocab_file_path, "wb") as f:
            logger.info(f"Saving the vocabulary to {self.vocab_file_path}")
            pickle.dump(
                (
                    self._unicode_to_int_vocab,
                    self._int_to_unicode_vocab,
                    self.learned_merges,
                ),
                f,
            )

        return self._unicode_to_int_vocab, self._int_to_unicode_vocab

    def encode(self, input_text: str, **kwargs) -> List[int]:
        """Encodes a raw input string into a list of BPE token IDs.

        The process is:
          1) Pre-tokenize the input into chunks (words, punctuation, special tokens).
          2) Convert each chunk to base (byte-level) token IDs.
          3) Apply merges in the learned order to combine tokens.

        Args:
            input_text (str): The text to encode.
            **kwargs: Additional keyword arguments (unused in the default implementation).

        Returns:
            List[int]: The list of integer token IDs representing the encoded text.

        Example:
            >>> bpe = BytePairEncoder(num_merges=50)
            >>> bpe.train_vocabulary("Hello world!")
            >>> token_ids = bpe.encode("Hello world!")
            >>> print(token_ids)  # Outputs a list of token IDs
        """
        text_chunks = self._pretokenize(input_text)
        logger.debug(f"Text chunks: {text_chunks[:20]}")

        if not text_chunks:
            return []

        # Convert chunks to base token IDs
        byte_encoded_chars: List[int] = []
        for chunk in text_chunks:
            if chunk in self.SPECIAL_TOKENS:
                special_id = self._unicode_to_int_vocab[chunk.encode("utf-8")]
                byte_encoded_chars.append(special_id)
            else:
                for b_int in chunk.encode("utf-8"):
                    byte_encoded_chars.append(
                        self._unicode_to_int_vocab[bytes([b_int])]
                    )

        # Apply learned merges in a naive pass
        for pair, new_id in tqdm(
            self.learned_merges, desc="Applying merges to encode", leave=False
        ):
            byte_encoded_chars = list(
                self._merge_pairs(pair, new_id, byte_encoded_chars)
            )

        return byte_encoded_chars

    def decode(self, encoded_tokens: List[int]) -> str:
        """Decodes a sequence of BPE token IDs back into a string.

        Args:
            encoded_tokens (List[int]): The list of integer token IDs to decode.

        Returns:
            str: The decoded string.

        Example:
            >>> bpe = BytePairEncoder(num_merges=50)
            >>> bpe.train_vocabulary("Hello world!")
            >>> token_ids = bpe.encode("Hello world!")
            >>> text = bpe.decode(token_ids) # Expected: "Hello world!" (or similar)
        """
        result_bytes = []
        for t in encoded_tokens:
            if isinstance(t, np.ndarray):  # handle NumPy scalar
                t = t.item()
            if t in self._int_to_unicode_vocab:
                result_bytes.append(self._int_to_unicode_vocab[t])
            else:
                result_bytes.append(b"<UNK>")
        return b"".join(result_bytes).decode("utf-8", errors="replace")

    def _pretokenize(self, input_text: str) -> List[str]:
        r"""Splits the input text into smaller chunks (tokens).

        GPT-2 style regex:
          - Matches special tokens explicitly.
          - Uses a regex pattern to split into words, punctuation, digits, and whitespace.

        Args:
            input_text (str): The text to tokenize.

        Returns:
            List[str]: A list of tokens or special symbols extracted from the text.

        Example:
            >>> bpe = BytePairEncoder(num_merges=50)
            >>> tokens = bpe._pretokenize("Hello <|endoftext|> world!")
            >>> print(tokens)
            ['Hello', '<|endoftext|>', ' ', 'world', '!']
        """
        special_pattern = (
            "(" + "|".join(regex.escape(k) for k in self.SPECIAL_TOKENS) + ")"
        )
        splitted_input = regex.split(special_pattern, input_text)

        general_pattern = regex.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""
        )

        final_chunks: List[str] = []
        for segment in splitted_input:
            if not segment:
                continue
            if segment in self.SPECIAL_TOKENS:
                final_chunks.append(segment)
            else:
                final_chunks.extend(general_pattern.findall(segment))

        return [tok for tok in final_chunks if tok]

    def _get_initial_pair_counts(
        self, word_freq: Counter
    ) -> Dict[Tuple[int, int], int]:
        """Builds a global frequency dictionary of adjacent token pairs across the `word_freq` corpus.

        Args:
            word_freq (Counter): A counter mapping word tuples (of token IDs) to their frequencies.

        Returns:
            Dict[Tuple[int, int], int]: A dictionary mapping each bigram (two adjacent token IDs)
            to its total frequency across the corpus.

        Examples:
            >>> from collections import Counter
            >>> word_freq = Counter({(65, 66, 67): 10, (66, 67, 68): 5})
            >>> bpe = BytePairEncoder(num_merges=50)
            >>> counts = bpe._get_initial_pair_counts(word_freq) # Displays the frequency counts for each bigram
        """
        items = list(word_freq.items())
        chunk_size = max(1, len(items) // self.n_workers)

        with Pool(self.n_workers) as pool:
            chunked_items = [
                items[i : i + chunk_size] for i in range(0, len(items), chunk_size)
            ]
            partial_results = pool.map(
                BytePairEncoder._local_pair_counts, chunked_items
            )

        total_counts = Counter()
        for c in partial_results:
            total_counts.update(c)
        return dict(total_counts)

    def _apply_merges_to_corpus(
        self,
        pair: Tuple[int, int],
        new_id: int,
        corpus_word_freq: Counter,
        pair_counts: Dict[Tuple[int, int], int],
    ) -> None:
        """Merges all occurrences of `pair` across the entire corpus and updates pair_counts.

        Args:
            pair (Tuple[int, int]): The bigram pair of token IDs to merge.
            new_id (int): The integer ID assigned to the merged token.
            corpus_word_freq (Counter): A counter of (tuple_of_tokens -> frequency).
            pair_counts (Dict[Tuple[int, int], int]): Global dictionary of bigram frequencies.

        Examples:
            >>> corpus_word_freq = Counter({(65, 66, 67): 3})
            >>> pair_counts = {(65, 66): 3, (66, 67): 3}
            >>> bpe = BytePairEncoder(num_merges=50)
            >>> bpe._apply_merges_to_corpus((65, 66), 300, corpus_word_freq, pair_counts)
            >>> print(corpus_word_freq)
            >>> print(pair_counts)
        """
        # Gather all items (word tuples) that contain the pair
        items_to_update = []
        for w_tuple, freq in corpus_word_freq.items():
            if pair in zip(w_tuple, w_tuple[1:]):
                items_to_update.append((w_tuple, freq))

        if not items_to_update:
            return

        # Decrement old bigram counts in pair_counts
        for old_tuple, freq in items_to_update:
            for bg in zip(old_tuple, old_tuple[1:]):
                pair_counts[bg] -= freq
                if pair_counts[bg] <= 0:
                    pair_counts.pop(bg, None)

        # Remove old tuples from corpus_word_freq
        for w_tuple, _ in items_to_update:
            del corpus_word_freq[w_tuple]

        # Parallel merge
        chunk_size = max(1, len(items_to_update) // self.n_workers)
        with Pool(self.n_workers) as pool:
            chunked_items = [
                items_to_update[i : i + chunk_size]
                for i in range(0, len(items_to_update), chunk_size)
            ]
            partial_results = pool.map(
                BytePairEncoder._local_merge_chunk,
                [(chunk, pair, new_id) for chunk in chunked_items],
            )

        # Aggregate results
        for local_new_word_freq, local_pair_counts in partial_results:
            for tup, freq in local_new_word_freq.items():
                corpus_word_freq[tup] += freq
            for bg, freq in local_pair_counts.items():
                pair_counts[bg] = pair_counts.get(bg, 0) + freq

    # ------------- Some helper functions for building vocabulary and encoding data ----------
    @staticmethod
    def _merge_pairs(
        pair: Tuple[int, int], new_idx: int, corpus: List[int]
    ) -> Tuple[int, ...]:
        """Merges occurrences of `pair` in a list of token IDs with the token ID `new_idx`.

        Args:
            pair (Tuple[int, int]): The bigram pair of token IDs to merge.
            new_idx (int): The ID of the newly created merged token.
            corpus (List[int]): The list of token IDs where merges need to be applied.

        Returns:
            Tuple[int, ...]: A new sequence of token IDs with the merges replaced by new_idx.

        Examples:
            >>> corpus = [65, 66, 67, 65, 66]
            >>> merged = BytePairEncoder._merge_pairs((65, 66), 300, corpus) # Expected: (300, 67, 300) or similar depending on merge implementation
        """
        merged_tokens: List[int] = []
        i = 0
        while i < len(corpus):
            # If we see the pair, merge them
            if i < len(corpus) - 1 and (corpus[i], corpus[i + 1]) == pair:
                merged_tokens.append(new_idx)
                i += 2
            else:
                merged_tokens.append(corpus[i])
                i += 1
        return tuple(merged_tokens)

    @staticmethod
    def _local_pair_counts(chunk: List[Tuple[Tuple[int, ...], int]]) -> Counter:
        """Computes local bigram counts for a chunk of (word_tuple, frequency) items.

        Args:
            chunk (List[Tuple[Tuple[int, ...], int]]): A subset of corpus word-frequency items.

        Returns:
            Counter: A Counter object mapping each bigram to its frequency within this chunk.

        Examples:
            >>> chunk = [((65, 66, 67), 2)]
            >>> counts = BytePairEncoder._local_pair_counts(chunk)  # Expected output: Counter({(65, 66): 2, (66, 67): 2})
        """
        partial_counts = Counter()
        for w_tuple, freq in chunk:
            for pair_ in zip(w_tuple, w_tuple[1:]):
                partial_counts[pair_] += freq
        return partial_counts

    @staticmethod
    def _local_merge_chunk(
        args: Tuple[List[Tuple[Tuple[int, ...], int]], Tuple[int, int], int],
    ):
        """Merges a bigram pair locally within a chunk of word tuples and frequencies.

        Args:
            args: A tuple containing:
                - The chunk of items to update.
                - The bigram pair to merge.
                - The new token ID.

        Returns:
            Tuple[Dict[Tuple[int, ...], int], Dict[Tuple[int, int], int]]:
                A dictionary of updated word tuples to frequency counts,
                and a dictionary of updated bigram pair frequencies.

        Examples:
            >>> chunk = [((65, 66, 67), 2)]
            >>> result = BytePairEncoder._local_merge_chunk((chunk, (65, 66), 300))  # Expected: a tuple with updated frequencies
        """
        chunk, pair, new_id = args
        local_new_word_freq = Counter()
        local_pair_counts = Counter()
        for old_tuple, freq in chunk:
            new_tuple = BytePairEncoder._merge_pairs(pair, new_id, old_tuple)
            local_new_word_freq[new_tuple] += freq
            for bg in zip(new_tuple, new_tuple[1:]):
                local_pair_counts[bg] += freq
        return dict(local_new_word_freq), dict(local_pair_counts)
