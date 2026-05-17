import heapq
import logging
import os
import pickle
from collections import Counter, OrderedDict
from itertools import batched
from multiprocessing import Pool, cpu_count
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
        >>> encoded_array = bpe.prepare_data([raw_text]) # Outputs an array of token IDs
    """

    SPECIAL_TOKENS = [
        "<|endoftext|>",
        "<PAD>",
        "<SOS>",
        "<UNK>",
        "<|USER|>",
        "<|ASSISTANT|>",
        "<|SYSTEM|>",
        "<|END_OF_TURN|>",
        "<|TOOL|>",
        "<|TOOL_CALL|>",
        "<|TOOL_RESULT|>",
    ]
    ENCODE_CHUNK_CACHE_MAX_SIZE = 50_000
    WORD_FREQ_BATCH_SIZE = 10_000
    WORD_FREQ_LOG_INTERVAL = 500_000

    # Worker-side handle for the encode Pool. Set by ``_init_encode_worker``
    # in each spawned worker process so subsequent ``imap`` calls can reach
    # the BPE without re-pickling ``self`` per batch.
    _WORKER_BPE: Optional["BytePairEncoder"] = None

    def __init__(
        self,
        num_merges: int = 500,
        vocab_file_path: str = "vocab.pkl",
        encoded_data_path: str = "bpe_encoded_data.npz",
        n_workers: Optional[int] = None,
        min_word_freq: int = 10,
    ) -> None:
        """Initializes the Byte Pair Encoder.

        Args:
            num_merges (int): Number of BPE merges to learn during training.
            vocab_file_path (str): Path to store or load the pickled vocabulary.
            encoded_data_path (str): Path to store or load the encoded data. The extension is replaced with ``.bin`` (the stored file is a raw little-endian int32 stream); see :meth:`load_encoded`.
            n_workers (Optional[int]): Number of processes for parallel operations. If None,
                defaults to the number of CPU cores minus one.
            min_word_freq (int): Minimum frequency for a word form to be included in
                the merge loop. Forms below this threshold are pruned before merging
                to avoid spending time on rare/singleton entries that don't influence
                merge decisions. Set to 1 to disable pruning.

        Examples:
            >>> bpe = BytePairEncoder(num_merges=100, vocab_file_path="vocab.pkl", encoded_data_path="encoded.npz")
            >>> print(bpe.num_merges)  # Expected output: 100
        """
        self.num_merges = num_merges
        self.min_word_freq = min_word_freq
        self.vocab_file_path = vocab_file_path
        self.encoded_data_path = encoded_data_path
        # We store the encoded corpus as a raw little-endian int32 stream.
        # No header — the count is recovered at load time from the file size.
        self.mmap_path = os.path.splitext(encoded_data_path)[0] + ".bin"
        base, ext = os.path.splitext(vocab_file_path)
        self.word_freq_cache_path = base + ".word_freq" + ext

        # For storing vocab: char -> int
        self._unicode_to_int_vocab: Dict[bytes, int] = {}
        # For storing reverse vocab: int -> char
        self._int_to_unicode_vocab: Dict[int, bytes] = {}
        # Store merges: list of ((tokenA, tokenB), new_id)
        self.learned_merges: List[Tuple[Tuple[int, int], int]] = []

        # Load existing dictionary if available
        self._load_dictionary()

        # Start merged token IDs from the first unused index after the base vocabulary
        self.new_idx = max(self._unicode_to_int_vocab.values()) + 1

        # Parallelism
        if n_workers is None:
            self.n_workers = min(32, max(1, cpu_count() - 1))
        else:
            self.n_workers = n_workers

        # Pre-compiled regex patterns for _pretokenize
        self._special_pattern = regex.compile(
            "(" + "|".join(regex.escape(k) for k in self.SPECIAL_TOKENS) + ")"
        )
        self._general_pattern = regex.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""
        )
        # Two encode-time caches with different lifetimes/types:
        # chunk text -> encoded ids (LRU), and learned merge pair -> priority/new id.
        self._encoded_chunk_cache: OrderedDict[str, Tuple[int, ...]] = OrderedDict()
        self._merge_priority_cache: Optional[Dict[Tuple[int, int], Tuple[int, int]]] = (
            None
        )

    @property
    def n_vocab(self) -> int:
        """int: Number of tokens (including special tokens) in the vocabulary.

        Examples:
            >>> bpe = BytePairEncoder(num_merges=50)
            >>> print(bpe.n_vocab)  # Outputs the size of the vocabulary
        """
        return len(self._unicode_to_int_vocab)

    def prepare_data(
        self,
        texts: Iterable[str],
        overwrite_vocabulary_file: bool = False,
        overwrite_encoded_data: bool = False,
    ) -> np.ndarray:
        """Trains and applies BPE on the given texts, returning encoded token IDs.

        Convenience wrapper that trains the vocabulary then writes token IDs
        to the configured memory-mapped file. ``texts`` is consumed twice
        when both passes run — once to build word frequencies during training,
        once to encode — so it must be re-iterable (e.g. a list or an object
        whose ``__iter__`` returns a fresh iterator each time). A bare
        generator will be exhausted before encoding starts.

        Args:
            texts (Iterable[str]): Documents to train on and encode.
            overwrite_vocabulary_file (bool): If True, re-trains and overwrites the BPE vocabulary.
            overwrite_encoded_data (bool): If True, overwrites an existing encoded file.

        Returns:
            np.ndarray: A memory-mapped array of token IDs.

        Example:
            >>> bpe = BytePairEncoder(num_merges=50)
            >>> encoded_array = bpe.prepare_data(["Hello world! This is a test."])
            >>> print(encoded_array)
            [ ...some token IDs... ]
        """
        self.train_vocabulary(texts, overwrite_saved_file=overwrite_vocabulary_file)
        self._encode_to_mmap(texts, overwrite_encoded_data=overwrite_encoded_data)
        return self.load_encoded(self.mmap_path)

    def train_vocabulary(
        self, texts: Iterable[str], overwrite_saved_file: bool = False
    ) -> Tuple[Dict[bytes, int], Dict[int, bytes]]:
        """Trains the BPE vocabulary on the given text documents.

        Streams text one document at a time, so the full corpus never needs to
        be held in memory. The learned merges and vocabulary are saved to
        vocab_file_path.

        Args:
            texts (Iterable[str]): Documents to train on (e.g. ["doc1", "doc2"]
                or a lazy generator). Pass a single document as [raw_text].
                A bare str is rejected to prevent silent per-character iteration.
            overwrite_saved_file (bool): If True, re-trains and overwrites any
                existing vocabulary file.

        Returns:
            Tuple[Dict[bytes, int], Dict[int, bytes]]:
                A tuple containing:
                - A dictionary mapping byte sequences or special tokens to integer IDs.
                - A dictionary mapping integer IDs back to byte sequences.

        Examples:
            >>> bpe = BytePairEncoder(num_merges=50)
            >>> vocab, rev_vocab = bpe.train_vocabulary(["Hello world! Hello again!"])
            >>> print(vocab)  # Prints the vocabulary mapping
        """
        if isinstance(texts, str):
            raise TypeError(
                "train_vocabulary expects an iterable of strings, not a bare str. "
                "Wrap a single document as [text]."
            )

        if os.path.exists(self.vocab_file_path) and not overwrite_saved_file:
            return self._unicode_to_int_vocab, self._int_to_unicode_vocab

        self._encoded_chunk_cache.clear()
        self._merge_priority_cache = None

        if not overwrite_saved_file and os.path.exists(self.word_freq_cache_path):
            logger.info(
                "Loading cached word frequencies from %s", self.word_freq_cache_path
            )
            with open(self.word_freq_cache_path, "rb") as f:
                word_freq = pickle.load(f)
        else:
            word_freq = self._build_word_freq(texts)
            logger.info("Caching word frequencies to %s", self.word_freq_cache_path)
            with open(self.word_freq_cache_path, "wb") as f:
                pickle.dump(word_freq, f)
        return self._learn_vocabulary_from_word_freq(word_freq)

    def encode(self, input_text: str) -> List[int]:
        """Encodes a raw input string into a list of BPE token IDs.

        The process is:
          1) Pre-tokenize the input into chunks (words, punctuation, special tokens).
          2) Convert each chunk to base (byte-level) token IDs.
          3) Greedily apply the highest-priority merge until no more merges apply.

        Args:
            input_text (str): The text to encode.

        Returns:
            List[int]: The list of integer token IDs representing the encoded text.

        Example:
            >>> bpe = BytePairEncoder(num_merges=50)
            >>> bpe.train_vocabulary(["Hello world!"])
            >>> token_ids = bpe.encode("Hello world!")
            >>> print(token_ids)  # Outputs a list of token IDs
        """
        text_chunks = self._pretokenize(input_text)
        logger.debug("Text chunks: %s", text_chunks[:20])

        if not text_chunks:
            return []

        merge_priority = self._get_merge_priority()
        result: List[int] = []
        for chunk in text_chunks:
            if chunk in self.SPECIAL_TOKENS:
                result.append(self._unicode_to_int_vocab[chunk.encode("utf-8")])
            else:
                cached = self._encoded_chunk_cache.get(chunk)
                if cached is not None:
                    self._encoded_chunk_cache.move_to_end(chunk)
                    result.extend(cached)
                else:
                    # Greedily apply the earliest-learned merge until convergence.
                    token_ids = list(chunk.encode("utf-8"))
                    while len(token_ids) >= 2:
                        next_pair = None
                        next_priority = float("inf")
                        for i in range(len(token_ids) - 1):
                            pair = (token_ids[i], token_ids[i + 1])
                            info = merge_priority.get(pair)
                            if info is not None and info[0] < next_priority:
                                next_priority = info[0]
                                next_pair = pair
                        if next_pair is None:
                            break
                        new_id = merge_priority[next_pair][1]
                        token_ids = list(
                            BytePairEncoder._merge_pairs(next_pair, new_id, token_ids)
                        )

                    encoded_chunk = tuple(token_ids)
                    if (
                        len(self._encoded_chunk_cache)
                        >= self.ENCODE_CHUNK_CACHE_MAX_SIZE
                    ):
                        self._encoded_chunk_cache.popitem(last=False)
                    self._encoded_chunk_cache[chunk] = encoded_chunk
                    result.extend(encoded_chunk)

        return result

    def decode(self, encoded_tokens: List[int]) -> str:
        """Decodes a sequence of BPE token IDs back into a string.

        Args:
            encoded_tokens (List[int]): The list of integer token IDs to decode.
                Numpy arrays must be converted with ``.tolist()`` first; this
                keeps the dict lookup on the hot path free of scalar coercion.

        Returns:
            str: The decoded string.

        Example:
            >>> bpe = BytePairEncoder(num_merges=50)
            >>> bpe.train_vocabulary(["Hello world!"])
            >>> token_ids = bpe.encode("Hello world!")
            >>> text = bpe.decode(token_ids) # Expected: "Hello world!" (or similar)
        """
        result_bytes = []
        for t in encoded_tokens:
            if t in self._int_to_unicode_vocab:
                result_bytes.append(self._int_to_unicode_vocab[t])
            else:
                result_bytes.append(b"<UNK>")
        return b"".join(result_bytes).decode("utf-8", errors="replace")

    def _load_dictionary(self) -> None:
        """Loads the dictionary (vocab + merges) from disk if it exists.

        If the file does not exist, creates a fresh base dictionary (all
        single-byte chars plus special tokens). If the file exists but fails
        to load, raises RuntimeError — silently rebuilding would discard the
        learned merges the caller asked us to load. Delete the file (or point
        ``vocab_file_path`` elsewhere) to force a clean retrain.

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
            raise RuntimeError(
                f"Failed to load vocabulary from {self.vocab_file_path}: {e}. "
                "Delete the file (or pass a different vocab_file_path) "
                "to retrain from scratch."
            ) from e

    def _construct_unicode_to_int_vocab(self) -> Dict[bytes, int]:
        """Constructs a base vocabulary for all single-byte values plus special tokens.

        Returns:
            Dict[bytes, int]: A dictionary mapping each single-byte (0..255) and special tokens
            to unique integer IDs.

        Examples:
            >>> bpe = BytePairEncoder(num_merges=50)
            >>> vocab = bpe._construct_unicode_to_int_vocab()
            >>> print(list(vocab.items())[:5])  # Show first 5 items in the vocabulary
        """
        unicode_to_int_vocab: Dict[bytes, int] = {}
        for i in range(256):
            unicode_to_int_vocab[bytes([i])] = i
        # Add special tokens
        for i, special_char in enumerate(self.SPECIAL_TOKENS):
            unicode_to_int_vocab[special_char.encode("utf-8")] = 256 + i
        return unicode_to_int_vocab

    def _encode_to_mmap(
        self,
        text_iter: Iterable[str],
        overwrite_encoded_data: bool = False,
        *,
        text_batch_size: int = 256,
    ) -> str:
        """Encodes streamed text to a raw int32 binary file.

        Writes each encoded batch's tokens directly with ``ndarray.tofile``,
        then atomically renames the temporary file to ``self.mmap_path``.
        The file has no header — readers recover the token count from the
        file size via :meth:`load_encoded`.

        Args:
            text_iter (Iterable[str]): Documents to encode (e.g. a list of strings or
                a lazy generator). The vocabulary must already be trained.
            overwrite_encoded_data (bool): If True, overwrites an existing file.
            text_batch_size (int): Number of documents to encode per batch.

        Returns:
            str: The path to the encoded binary file.
        """
        if text_batch_size < 1:
            raise ValueError(f"text_batch_size must be >= 1, got {text_batch_size}")

        if os.path.exists(self.mmap_path) and not overwrite_encoded_data:
            logger.info(
                f"Found existing memory-mapped encoded data at '{self.mmap_path}', "
                "reusing it instead of re-encoding."
            )
            return self.mmap_path
        parent_dir = os.path.dirname(self.mmap_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        int32_dtype = np.dtype("<i4")
        tmp_path = f"{self.mmap_path}.tmp"

        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

            with open(tmp_path, "wb") as out_file:
                text_batches = batched(text_iter, text_batch_size)
                if self.n_workers > 1:
                    # Pool initializer pickles `self` once per worker process;
                    # imap then only ships text batches across the boundary
                    # instead of re-pickling the BPE state per batch.
                    with Pool(
                        self.n_workers,
                        initializer=BytePairEncoder._init_encode_worker,
                        initargs=(self,),
                    ) as pool:
                        encoded_batches = pool.imap(
                            BytePairEncoder._encode_text_batch_array_worker,
                            text_batches,
                        )
                        for flat_batch in tqdm(
                            encoded_batches,
                            desc="Encoding text to tokens",
                            unit="batch",
                        ):
                            flat_batch.tofile(out_file)
                else:
                    for text_batch in tqdm(
                        text_batches,
                        desc="Encoding text to tokens",
                        unit="batch",
                    ):
                        flat_batch = np.fromiter(
                            (
                                token
                                for text in text_batch
                                for token in self.encode(text)
                            ),
                            dtype=int32_dtype,
                        )
                        flat_batch.tofile(out_file)

            os.replace(tmp_path, self.mmap_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return self.mmap_path

    @staticmethod
    def load_encoded(path: str) -> np.ndarray:
        """Memory-maps a raw int32 token stream written by ``_encode_to_mmap``.

        The file has no header, so the token count is the file size divided
        by ``int32.itemsize`` (4 bytes per token).
        """
        int32_dtype = np.dtype("<i4")
        token_count = os.path.getsize(path) // int32_dtype.itemsize
        return np.memmap(path, dtype=int32_dtype, mode="r", shape=(token_count,))

    @staticmethod
    def _init_encode_worker(bpe: "BytePairEncoder") -> None:
        # Pool ``initializer`` hook: runs once per spawned worker process so
        # the heavy BPE state is pickled only at startup, not per batch.
        BytePairEncoder._WORKER_BPE = bpe

    @staticmethod
    def _encode_text_batch_array_worker(text_batch: Sequence[str]) -> np.ndarray:
        bpe = BytePairEncoder._WORKER_BPE
        if bpe is None:
            raise RuntimeError("Tokenizer worker was not initialized")
        return np.fromiter(
            (token for text in text_batch for token in bpe.encode(text)),
            dtype=np.dtype("<i4"),
        )

    def _build_word_freq(self, texts: Iterable[str]) -> Counter:
        """Build word frequency table from documents, parallelized when possible.

        Counts pre-tokenized word forms in document batches, then converts byte
        keys to token-id tuples used by the merge loop.
        """
        pattern_str = self._general_pattern.pattern
        special_tokens_set = set(self.SPECIAL_TOKENS)
        special_token_ids = {
            tok: self._unicode_to_int_vocab[tok.encode("utf-8")]
            for tok in self.SPECIAL_TOKENS
        }

        args_iter = (
            (batch, pattern_str, special_tokens_set, special_token_ids)
            for batch in batched(texts, self.WORD_FREQ_BATCH_SIZE)
        )

        word_freq: Counter = Counter()
        doc_count = 0
        with Pool(max(1, self.n_workers)) as pool:
            for partial_wf, batch_doc_count in pool.imap_unordered(
                BytePairEncoder._word_freq_worker, args_iter
            ):
                word_freq.update(partial_wf)
                doc_count += batch_doc_count
                if doc_count % self.WORD_FREQ_LOG_INTERVAL < self.WORD_FREQ_BATCH_SIZE:
                    logger.info(
                        "Processed %d documents (%d unique word forms so far)",
                        doc_count,
                        len(word_freq),
                    )

        logger.info(
            "Word frequency table complete: %d documents, %d unique word forms",
            doc_count,
            len(word_freq),
        )

        return Counter(
            {tuple(k) if isinstance(k, bytes) else k: v for k, v in word_freq.items()}
        )

    @staticmethod
    def _word_freq_worker(args):
        """Pretokenize a document batch and count word forms."""
        docs, pattern_str, special_tokens, special_token_ids = args
        general_pat = regex.compile(pattern_str)
        special_pat = regex.compile(
            "(" + "|".join(regex.escape(k) for k in special_tokens) + ")"
        )
        wf: Counter = Counter()
        for doc in docs:
            for segment in special_pat.split(doc):
                if not segment:
                    continue
                if segment in special_tokens:
                    wf[(special_token_ids[segment],)] += 1
                else:
                    for chunk in general_pat.findall(segment):
                        wf[chunk.encode("utf-8")] += 1
        return wf, len(docs)

    def _learn_vocabulary_from_word_freq(
        self,
        word_freq: Counter,
    ) -> Tuple[Dict[bytes, int], Dict[int, bytes]]:
        # Prune rare word forms that don't meaningfully influence merge decisions
        if self.min_word_freq > 1:
            before = len(word_freq)
            word_freq = Counter(
                {k: v for k, v in word_freq.items() if v >= self.min_word_freq}
            )
            logger.info(
                "Pruned word forms with freq < %d: %d -> %d (%d removed)",
                self.min_word_freq,
                before,
                len(word_freq),
                before - len(word_freq),
            )

        # Build initial pair counts and an inverted index: pair -> set of word tuples
        pair_counts: Dict[Tuple[int, int], int] = {}
        pair_to_words: Dict[Tuple[int, int], set] = {}
        for w_tuple, freq in word_freq.items():
            for p in zip(w_tuple, w_tuple[1:]):
                pair_counts[p] = pair_counts.get(p, 0) + freq
                if p not in pair_to_words:
                    pair_to_words[p] = set()
                pair_to_words[p].add(w_tuple)

        # Remove pairs involving special tokens so we don't merge them
        for p in list(pair_counts):
            if p[0] >= 256 or p[1] >= 256:
                del pair_counts[p]
                pair_to_words.pop(p, None)

        # Max-heap for finding the best pair in O(log n).
        # Entries: (-count, pair). Stale entries are skipped via pair_counts lookup.
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        # Merge loop
        for i in tqdm(range(self.num_merges), desc="Merging pairs"):
            # Pop stale/exhausted entries to find the current best pair.
            # Lazy correction: if a popped entry's count is stale but still positive,
            # re-push with the corrected count instead of discarding.
            most_frequent_pair = None
            most_frequent_count = 0
            while heap:
                neg_count, candidate = heap[0]
                actual = pair_counts.get(candidate, 0)
                if actual <= 0:
                    heapq.heappop(heap)
                    continue
                if actual != -neg_count:
                    heapq.heapreplace(heap, (-actual, candidate))
                    continue
                most_frequent_pair = candidate
                most_frequent_count = actual
                heapq.heappop(heap)
                break

            if most_frequent_pair is None or most_frequent_count < 2:
                break

            new_id = self.new_idx
            self.new_idx += 1

            # Create the merged token
            merged_bytes = (
                self._int_to_unicode_vocab[most_frequent_pair[0]]
                + self._int_to_unicode_vocab[most_frequent_pair[1]]
            )
            self._int_to_unicode_vocab[new_id] = merged_bytes
            self._unicode_to_int_vocab[merged_bytes] = new_id

            # Apply merge using the inverted index for fast lookup
            self._apply_merges_to_corpus_indexed(
                most_frequent_pair,
                new_id,
                word_freq,
                pair_counts,
                pair_to_words,
                heap,
            )
            self.learned_merges.append((most_frequent_pair, new_id))

            if (i + 1) % 5_000 == 0:
                logger.info(
                    f"[{i + 1}/{self.num_merges} merge] Best Pair Merged "
                    f"({most_frequent_count} occurrences). "
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

    def _get_merge_priority(self) -> Dict[Tuple[int, int], Tuple[int, int]]:
        """Lazily build and cache a merge priority lookup: pair -> (priority, new_id).
        Lower priority = earlier merge = applied first during greedy encoding.
        """
        if self._merge_priority_cache is None:
            cache: Dict[Tuple[int, int], Tuple[int, int]] = {}
            for priority, (pair, new_id) in enumerate(self.learned_merges):
                if pair not in cache:
                    cache[pair] = (priority, new_id)
            self._merge_priority_cache = cache
        return self._merge_priority_cache

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
        splitted_input = self._special_pattern.split(input_text)

        final_chunks: List[str] = []
        for segment in splitted_input:
            if not segment:
                continue
            if segment in self.SPECIAL_TOKENS:
                final_chunks.append(segment)
            else:
                final_chunks.extend(self._general_pattern.findall(segment))

        return [tok for tok in final_chunks if tok]

    def _apply_merges_to_corpus_indexed(
        self,
        pair: Tuple[int, int],
        new_id: int,
        corpus_word_freq: Counter,
        pair_counts: Dict[Tuple[int, int], int],
        pair_to_words: Dict[Tuple[int, int], set],
        heap: list,
    ) -> None:
        """Apply a single merge to affected words using an inverted index,
        updating pair_counts, pair_to_words, and the heap in place.
        """
        affected = pair_to_words.pop(pair, set())
        pair_counts.pop(pair, None)
        if not affected:
            return

        # Collect (old_tuple, freq) for affected words that are still in the corpus
        items_to_update = []
        for w_tuple in affected:
            if w_tuple in corpus_word_freq:
                items_to_update.append((w_tuple, corpus_word_freq[w_tuple]))
        if not items_to_update:
            return

        # Decrement old pair counts and remove from inverted index.
        # No heap pushes here — stale entries are lazily corrected at pop time.
        for old_tuple, freq in items_to_update:
            for bg in zip(old_tuple, old_tuple[1:]):
                if bg in pair_counts:
                    pair_counts[bg] -= freq
                    if pair_counts[bg] <= 0:
                        pair_counts.pop(bg, None)
                        pair_to_words.pop(bg, None)
                    elif bg in pair_to_words:
                        pair_to_words[bg].discard(old_tuple)
            del corpus_word_freq[old_tuple]

        # Merge and update. Track which pairs changed so we push each only once.
        changed_pairs: set = set()
        for old_tuple, freq in items_to_update:
            new_tuple = BytePairEncoder._merge_pairs(pair, new_id, old_tuple)
            corpus_word_freq[new_tuple] += freq

            for bg in zip(new_tuple, new_tuple[1:]):
                pair_counts[bg] = pair_counts.get(bg, 0) + freq
                if bg not in pair_to_words:
                    pair_to_words[bg] = set()
                pair_to_words[bg].add(new_tuple)
                changed_pairs.add(bg)

        # Single heap push per changed pair (not per word form)
        for bg in changed_pairs:
            heapq.heappush(heap, (-pair_counts[bg], bg))

    # ------------- Some helper functions for building vocabulary and encoding data ----------
    @staticmethod
    def _merge_pairs(
        pair: Tuple[int, int], new_idx: int, corpus: Sequence[int]
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
        first, second = pair
        i = 0
        last = len(corpus) - 1
        while i <= last:
            # If we see the pair, merge them
            if i < last and corpus[i] == first and corpus[i + 1] == second:
                merged_tokens.append(new_idx)
                i += 2
            else:
                merged_tokens.append(corpus[i])
                i += 1
        return tuple(merged_tokens)
