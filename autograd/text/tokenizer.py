from collections import Counter
import os
import regex
from typing import ByteString, Dict, List, Tuple
import logging
import pickle
import numpy as np

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
            encoded_data = np.array(self.encode(joined_text))

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
            self._int_to_unicode_vocab = dict(
                zip(
                    self._unicode_to_int_vocab.values(),
                    self._unicode_to_int_vocab.keys(),
                )
            )
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
            self._int_to_unicode_vocab = dict(
                zip(
                    self._unicode_to_int_vocab.values(),
                    self._unicode_to_int_vocab.keys(),
                )
            )
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

        # Convert a list of strings to a list of integers,
        # where each integer is the index of the character in the vocabulary
        byte_encoded_chars: List[int] = []
        for chunk in text_chunks:
            if chunk in self.SPECIAL_TOKENS:
                # convert to single ID
                special_id = self._unicode_to_int_vocab[chunk.encode("utf-8")]
                byte_encoded_chars.append(special_id)
            else:
                byte_encoded_chars.extend(list(chunk.encode("utf-8")))
        logger.debug(f"Byte encoded chars: {byte_encoded_chars[:10]}")

        for i in range(self.num_merges):
            pair_counts = self._get_bigrams_to_count(byte_encoded_chars)

            # Check if there are still pairs to merge
            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)

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

            # Update the original encoded chars with the merged one
            byte_encoded_chars = self._merge_pairs(
                best_pair, new_id, byte_encoded_chars
            )
            # Record this for encoding step later
            self.learned_merges.append((best_pair, new_id))

            if (i + 1) % 100 == 0:
                logger.info(
                    f"[{i+1}/{self.num_merges} merge] Best Pair Merged "
                    f"({pair_counts[best_pair]} occurrences). "
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

        # Convert a list of strings to a list of integers,
        # where each integer is the index of the character in the vocabulary
        byte_encoded_chars: List[int] = []
        for chunk in text_chunks:
            if chunk in self.SPECIAL_TOKENS:
                # convert to single ID
                special_id = self._unicode_to_int_vocab[chunk.encode("utf-8")]
                byte_encoded_chars.append(special_id)
            else:
                for b_int in list(chunk.encode("utf-8")):
                    single_byte = bytes([b_int])  # e.g. b"\x46"
                    # Convert single_byte -> int ID from base vocab
                    byte_encoded_chars.append(self._unicode_to_int_vocab[single_byte])
        logger.info(f"Byte encoded chars: {byte_encoded_chars[:10]}")

        # Re-apply merges on the input_text
        for pair, new_id in self.learned_merges:
            byte_encoded_chars = self._merge_pairs(pair, new_id, byte_encoded_chars)

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

        return final_chunks

    def _get_bigrams_to_count(self, tokens: List[int]) -> Dict[Tuple[int, int], int]:
        """
        Count how frequently each adjacent pair of tokens occurs in the entire corpus.
        Note: This function should be called before merging any bigrams/pairs.

        Args:
            tokens (list[int]): The list of integers representing the characters in the original corpus

        Returns:
            dict[tuple[int,int], int]: A dictionary mapping each pair of tokens to the number of times it occurs
        """
        counter = Counter()
        for tup in zip(tokens[:-1], tokens[1:]):
            counter[tup] += 1
        return dict(counter)

    def _merge_pairs(
        self, pair: Tuple[int, int], new_idx: int, corpus: List[int]
    ) -> List[int]:
        """
        Given a pair (a, b) and its new_id, replace all occurrences of (a, b) in each line
        with [new_id].

        Example:
            corpus = [10, 11, 11, 12]
            pair = (10, 11), new_id=256
            result -> [256, 11, 12]
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
        return merged_tokens


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
        overwrite_saved_file=True,
        split_token="<|endoftext|>",
    )

    # 4) Try a small decode test
    logger.info(f"Size of final vocabulary: {len(bpe._int_to_unicode_vocab)}")
    logger.info(f"Encoded data length: {len(encoded_data)}")

    decoded_subset = bpe.decode(encoded_data)
    logger.info(f"Original text: {data}")
    logger.info(f"Encoded text: {encoded_data}")
    logger.info(f"Decoded text: {decoded_subset}")
