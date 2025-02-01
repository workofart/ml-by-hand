import logging
import os
import pickle
from collections import Counter
from typing import ByteString, Dict, List, Tuple

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
    SPECIAL_TOKENS = ["<|endoftext|>", "<PAD>", "<SOS>", "<UNK>"]

    def __init__(
        self,
        num_merges: int = 500,
        vocab_file_path: str = "vocab.pkl",
    ) -> None:
        self.num_merges = num_merges
        self.vocab_file_path = vocab_file_path

        # For storing vocab: char -> int
        self._unicode_to_int_vocab: Dict[ByteString, int] = {}
        # For storing reverse vocab: int -> char
        self._int_to_unicode_vocab: Dict[int, ByteString] = {}
        # Store merges: list of ((tokenA, tokenB), new_id)
        self.learned_merges: List[Tuple[Tuple[int, int], int]] = []

        self._load_dictionary()

        # start merged token ids from the first unused index after the base vocabulary
        self.new_idx = max(self._unicode_to_int_vocab.values()) + 1

    def prepare_data(
        self,
        raw_text_list: List[str],
        npz_file_path: str = "bpe_encoded.npz",
        overwrite_saved_file: bool = False,
        split_token: str = "<|endoftext|>",
    ) -> Tuple[np.ndarray]:
        """
        High-level method that:
          1) Trains (or loads) the BPE vocabulary on the given raw_text_list.
          2) Encodes the text into a NumPy array of token IDs.
          3) Caches the result to an npz file (unless it already exists and we're not overwriting).

        You can choose to use this or the individual methods below for fine-grain control:
        - train_vocabulary()
        - encode_text()

        Args:
        - raw_text_list: List[str]
            The list of texts from which to train or apply BPE. If there is only one string, wrap it with [raw_text] before calling this method.
        - npz_file_path: str
            File path to store (or load) the encoded .npz data.
        - overwrite_saved_file: bool
            Whether to overwrite an existing .npz file with newly encoded data.
        - split_token: str, optional
            Delimiter to insert between data blocks.

        Returns:
        - encoded_data : np.ndarray
            The encoded tokens as a NumPy array.
        """
        joined_text = split_token.join(raw_text_list)

        # 1) Train the vocabulary if needed
        self.train_vocabulary(joined_text, overwrite_saved_file=overwrite_saved_file)

        # 2) Check if we already have an encoded .npz file
        if os.path.exists(npz_file_path) and not overwrite_saved_file:
            logger.info(
                f"Found existing encoded data at '{npz_file_path}', "
                "loading it instead of re-encoding."
            )
            with np.load(npz_file_path, allow_pickle=True) as npz_data:
                encoded_data = npz_data["arr_0"]
        else:
            # Re-encode from raw blocks
            encoded_data = np.array(self.encode(joined_text), dtype=np.int32)
            # Save to disk
            np.savez_compressed(npz_file_path, encoded_data)
            logger.info(f"Saved newly encoded data to {npz_file_path}")

        logger.info(f"Vocabulary size: {len(self._unicode_to_int_vocab)}")
        logger.info(f"Encoded data length: {len(encoded_data)}")
        logger.debug(f"Sample encoded data (first 50 tokens): {encoded_data[:50]}")
        return encoded_data

    def _load_dictionary(self) -> None:
        if not os.path.exists(self.vocab_file_path):
            logger.info(
                "Vocabulary file does not exist. Creating new dictionary from scratch."
            )
            self._unicode_to_int_vocab = self._construct_unicode_to_int_vocab()
            self._int_to_unicode_vocab = {
                v: k for k, v in self._unicode_to_int_vocab.items()
            }
            # We need to store the merges we've learned
            # so we can apply them to new text during the encode step
            self.learned_merges = []
            return

        # If the file exists, attempt to load. Fallback to a new dictionary if there's an error.
        try:
            with open(self.vocab_file_path, "rb") as f:
                logger.info("Loading the vocabulary from disk.")
                data = pickle.load(f)
                (
                    self._unicode_to_int_vocab,
                    self._int_to_unicode_vocab,
                    self.learned_merges,
                ) = data
        except (pickle.UnpicklingError, EOFError) as e:
            logger.warning(
                f"Failed to load the vocabulary from {self.vocab_file_path}. "
                f"Reason: {e}. Creating new dictionary."
            )
            self._unicode_to_int_vocab = self._construct_unicode_to_int_vocab()
            self._int_to_unicode_vocab = {
                v: k for k, v in self._unicode_to_int_vocab.items()
            }
            # We need to store the merges we've learned
            # so we can apply them to new text during the encode step
            self.learned_merges = []

    def train_vocabulary(
        self, input_text: str, overwrite_saved_file: bool = False
    ) -> Tuple[Dict[ByteString, int], Dict[int, ByteString]]:
        """
        Train the BPE vocabulary on `input_text`. Saves the results to disk
        unless the vocab is already loaded and `overwrite_saved_file` is False.
        """
        if self._unicode_to_int_vocab and not overwrite_saved_file:
            # Already loaded a vocab from file
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

        pair_counts = self._get_initial_pair_counts(word_freq)
        # remove pairs involving special tokens so we don't merge them
        pair_counts = {
            p: c for p, c in pair_counts.items() if p[0] < 256 and p[1] < 256
        }

        for i in tqdm(range(self.num_merges), desc="Merging pairs"):
            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            best_pair_count = pair_counts[best_pair]

            # if best pair is not frequent enough, stop merging
            if pair_counts[best_pair] < 2:
                break

            new_id = self.new_idx
            self.new_idx += 1

            # Store the merged pair into the vocab
            merged_bytes = (
                self._int_to_unicode_vocab[best_pair[0]]
                + self._int_to_unicode_vocab[best_pair[1]]
            )
            self._int_to_unicode_vocab[new_id] = merged_bytes
            self._unicode_to_int_vocab[merged_bytes] = new_id

            # Updates word_freq in-place
            self._apply_merges_to_corpus(best_pair, new_id, word_freq, pair_counts)
            # Record this for encoding step later
            self.learned_merges.append((best_pair, new_id))

            if (i + 1) % 100 == 0:
                logger.info(
                    f"[{i+1}/{self.num_merges} merge] Best Pair Merged "
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

    def encode(self, input_text: str) -> List[int]:
        """
        We need to perform the merges that appear in the vocabulary
        Merge the highest-frequency pair of tokens in the text, then repeat

        Break `input_text` into tokens (characters plus merges), then return
        a list of token IDs.
        """
        text_chunks = self._pretokenize(input_text)
        logger.debug(f"Text chunks: {text_chunks[:20]}")

        if not text_chunks:
            return []

        # Convert a list of strings to a list of integers,
        # where each integer is the index of the character in the vocabulary
        byte_encoded_chars: List[int] = []
        for chunk in text_chunks:
            if chunk in self.SPECIAL_TOKENS:
                # convert to single ID
                special_id = self._unicode_to_int_vocab[chunk.encode("utf-8")]
                byte_encoded_chars.append(special_id)
            else:
                for b_int in chunk.encode("utf-8"):
                    # e.g. b"\x46"
                    # Convert single_byte -> int ID from base vocab
                    byte_encoded_chars.append(
                        self._unicode_to_int_vocab[bytes([b_int])]
                    )

        # Apply merges in order (naive pass)
        for pair, new_id in tqdm(self.learned_merges, desc="Applying merges to encode"):
            byte_encoded_chars = list(
                self._merge_pairs(pair, new_id, byte_encoded_chars)
            )

        return byte_encoded_chars

    def decode(self, encoded_tokens: List[int]) -> str:
        result: List[ByteString] = []
        for t in encoded_tokens:
            if t in self._int_to_unicode_vocab:
                result.append(self._int_to_unicode_vocab[t])
            else:
                result.append(b"<UNK>")
        # Remember our result contains bytes, so we need to join them and decode them
        return b"".join(result).decode("utf-8", errors="replace")

    def _construct_unicode_to_int_vocab(self) -> Dict[ByteString, int]:
        """
        Returns a dict: char -> int, for 0..255, plus expansions for non-printable control characters
        """
        unicode_to_int_vocab: Dict[ByteString, int] = {}
        # Base 256
        for i in range(256):
            # Each i is mapped to the single byte b"\x00" ... b"\xff"
            unicode_to_int_vocab[bytes([i])] = i
        # Add special tokens
        for i, special_char in enumerate(self.SPECIAL_TOKENS):
            unicode_to_int_vocab[special_char.encode("utf-8")] = 256 + i
        return unicode_to_int_vocab

    def _pretokenize(self, input_text: str) -> List[str]:
        r"""
        From OpenAI GPT-2 regex pattern
        https://github.com/openai/tiktoken/blob/63527649963def8c759b0f91f2eb69a40934e468/tiktoken_ext/openai_public.py#L9-L14

        - `?\p{L}+` grabs runs of letters (including Unicode letters) optionally preceded by a space.
        - `?\p{N}+` similarly for digits (e.g. `" 2022"` becomes a token) optionally preceded by a space.
        - `?[^\s\p{L}\p{N}]+` captures punctuation or other symbols optionally preceded by a space.
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
        """
        Builds a global frequency dictionary of adjacent token pairs (bigrams)
        across the entire corpus, which is represented by `word_freq`.

        Args:
            word_freq (Counter[Tuple[int, ...]]):
                A mapping of distinct word tuples (each a sequence of token IDs)
                to their frequency in the corpus.

        Returns:
            Dict[Tuple[int, int], int]:
                A dictionary mapping each distinct bigram (pair of token IDs)
                to the total number of times it appears across all word tuples
                in `word_freq`.

        Example:
            Suppose word_freq = {
                (10, 11, 12): 3,     # appears 3 times
                (11, 12, 12): 2      # appears 2 times
            }
            Then _get_initial_pair_counts(...) might return:
            {
                (10, 11): 3,
                (11, 12): 5,  # (3 + 2) across both words
                (12, 12): 2
            }
        """
        pair_counts = Counter()
        for w_tuple, freq in word_freq.items():
            for tup in zip(w_tuple[:-1], w_tuple[1:]):
                pair_counts[tup] += freq
            # for i in range(len(w_tuple) - 1):
            #     pair_counts[(w_tuple[i], w_tuple[i + 1])] += freq
        return dict(pair_counts)

    def _merge_pairs(
        self, pair: Tuple[int, int], new_idx: int, corpus: List[int]
    ) -> Tuple[int, ...]:
        """
        Merges a specified pair of token IDs in a single pass over a list of token IDs.

        This function is primarily used at encoding time or in a naive approach,
        rather than the frequency-based approach used during BPE training.

        Args:
            pair (Tuple[int, int]):
                The token pair to be merged (e.g., (10, 11)).
            new_idx (int):
                The new token ID that replaces each occurrence of `pair`.
            corpus (List[int]):
                A list of token IDs representing a single sequence of text.

        Returns:
            List[int]:
                A new list of token IDs, where each adjacent occurrence of `pair`
                has been replaced by `new_idx`.

        Example:
            corpus = [10, 11, 11, 12]
            pair = (10, 11)
            new_idx = 256

            => Output: [256, 11, 12]
        """
        merged_tokens: List[int] = []
        i = 0

        # Use while loop instead of for loop
        # to avoid skipping merges
        while i < len(corpus):
            if i < len(corpus) - 1 and (corpus[i], corpus[i + 1]) == pair:
                merged_tokens.append(new_idx)
                i += 2  # merged so we skip next token
            else:
                merged_tokens.append(corpus[i])
                i += 1
        return tuple(merged_tokens)

    def _apply_merges_to_corpus(
        self,
        pair: Tuple[int, int],
        new_id: int,
        corpus_word_freq: Counter,
        pair_counts: Dict[Tuple[int, int], int],
    ) -> None:
        """
        Merges a given bigram pair in-place throughout the entire corpus
        (represented by `corpus_word_freq`) and updates the global `pair_counts` accordingly.

        Args:
            pair (Tuple[int, int]):
                The token pair to merge (e.g., (32, 101)).
            new_id (int):
                The new token ID that replaces each occurrence of `pair`.
            corpus_word_freq (Counter[Tuple[int, ...]]):
                A mapping of word tuples -> frequency for the entire corpus.
                This will be modified in-place to reflect the merged pair.
            pair_counts (Dict[Tuple[int, int], int]):
                A global dictionary mapping pairs -> frequency.
                Will be updated to remove old bigrams and add new ones formed by the merge.

        Returns:
            None. (Modifies `corpus_word_freq` and `pair_counts` in-place.)

        Example:
            If `corpus_word_freq` has (10, 11, 11, 12) occurring 2 times,
            and `pair` is (10, 11) -> new_id=256,
            then all (10,11) occurrences in that word tuple will be replaced by [256],
            resulting in (256, 11, 12).
            Frequencies in `pair_counts` are updated to reflect the removal
            of (10,11) and the addition of (256,11).
        """
        items_to_update = []
        # Step 1: Gather a list of tuples that contain this pair so we can update them.
        for w_tuple, freq in corpus_word_freq.items():
            # Check if the target pair occurs anywhere in w_tuple.
            if pair in zip(w_tuple, w_tuple[1:]):
                items_to_update.append((w_tuple, freq))

        # Step 2: For each word tuple that needs merging, remove it from `corpus_word_freq`,
        # decrement the old bigrams in `pair_counts`, merge into a new tuple,
        # and then update `corpus_word_freq` and `pair_counts` for the new tuple.
        for old_tuple, freq in items_to_update:
            # Remove the old tuple from the frequency dictionary
            del corpus_word_freq[old_tuple]

            # Decrement the bigram counts for each adjacent pair in the old tuple
            for bg in zip(old_tuple, old_tuple[1:]):
                pair_counts[bg] -= freq
                if pair_counts[bg] <= 0:
                    pair_counts.pop(bg, None)

            # Merge the specified pair within this tuple
            new_tuple = self._merge_pairs(pair, new_id, old_tuple)
            corpus_word_freq[new_tuple] += freq

            # Add/increment bigrams formed by the new tuple
            for bg in zip(new_tuple, new_tuple[1:]):
                pair_counts[bg] = pair_counts.get(bg, 0) + freq


if __name__ == "__main__":
    # Example usage
    from autograd.tools.data import (
        load_data,
    )  # your custom data loader, adjust as needed

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filename = "examples/tinyshakespeare.txt"

    # 1) Load raw text
    data = load_data(url, filename)[:50]

    # 2) Create BPE instance
    bpe = BytePairEncoder(num_merges=50, vocab_file_path="tiny_shakespeare_vocab.pkl")

    # 3) Prepare data:
    #    - This trains the vocab if not already loaded or if overwrite=True
    #    - Encodes the text
    #    - Saves (or loads) from 'bpe_mini_shakespeare.npz'
    encoded_data = bpe.prepare_data(
        raw_text_list=data.split("\n\n"),
        npz_file_path="bpe_tokenizer_demo.npz",
        overwrite_saved_file=False,
        split_token="<|endoftext|>",
    )

    # 4) Try a small decode test
    logger.info(f"Size of final vocabulary: {len(bpe._int_to_unicode_vocab)}")
    logger.info(f"Encoded data length: {len(encoded_data)}")

    decoded_subset = bpe.decode(encoded_data)
    logger.info(f"Original text: {data}")
    logger.info(f"Encoded text: {encoded_data}")
    logger.info(f"Decoded text: {decoded_subset}")
