from collections import Counter
import os
import regex
from typing import ByteString
import logging
import pickle

logger = logging.getLogger(__name__)


class BytePairEncoder:
    SPECIAL_TOKENS = ["<|endoftext|>", "<PAD>", "<SOS>", "<UNK>"]

    def __init__(self, num_merges=500, vocab_file_path="vocab.pkl") -> None:
        self.num_merges = num_merges
        self.vocab_file_path = vocab_file_path
        self._unicode_to_int_vocab = self._construct_unicode_to_int_vocab()
        self._int_to_unicode_vocab = dict(
            zip(self._unicode_to_int_vocab.values(), self._unicode_to_int_vocab.keys())
        )
        self.learned_merges = []
        # start merged token ids from the first unused index after the base vocabulary
        self.new_idx = max(self._unicode_to_int_vocab.values()) + 1

    def train_vocabulary(
        self, input_text: str, overwrite_saved_file: bool = False
    ) -> tuple[dict[ByteString, int], dict[int, ByteString]]:
        if os.path.exists(self.vocab_file_path) and not overwrite_saved_file:
            with open(self.vocab_file_path, "rb") as f:
                logger.info("Loading the vocabulary from disk")
                self._unicode_to_int_vocab, self._int_to_unicode_vocab = pickle.load(f)
                return self._unicode_to_int_vocab, self._int_to_unicode_vocab

        text_chunks = self._pretokenize(input_text)
        logger.debug(f"Text chunks: {text_chunks[:10]}")

        # Convert a list of strings to a list of integers,
        # where each integer is the index of the character in the vocabulary
        byte_encoded_chars = []
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
                    f"[{i+1}/{self.num_merges} merge] Best Pair Merged with {pair_counts[best_pair]} occurrences"
                )

        with open(self.vocab_file_path, "wb") as f:
            logger.info("Saving the vocabulary from disk")
            pickle.dump((self._unicode_to_int_vocab, self._int_to_unicode_vocab), f)

        return self._unicode_to_int_vocab, self._int_to_unicode_vocab

    def encode(self, input_text: str) -> list[int]:
        """
        We need to perform the merges that appear in the vocabulary
        Merge the highest-frequency pair of tokens in the text, then repeat
        """
        text_chunks = self._pretokenize(input_text)
        logger.info(f"Text chunks: {text_chunks[:20]}")

        # Convert a list of strings to a list of integers,
        # where each integer is the index of the character in the vocabulary
        byte_encoded_chars = []
        for chunk in text_chunks:
            if chunk in self.SPECIAL_TOKENS:
                # convert to single ID
                special_id = self._unicode_to_int_vocab[chunk.encode("utf-8")]
                byte_encoded_chars.append(special_id)
            else:
                for b_int in list(chunk.encode("utf-8")):
                    single_byte = bytes([b_int])  # e.g. b"\x46"
                    byte_encoded_chars.append(self._unicode_to_int_vocab[single_byte])
        logger.info(f"Byte encoded chars: {byte_encoded_chars[:10]}")

        # Re-apply merges on the input_text
        for pair, new_id in self.learned_merges:
            byte_encoded_chars = self._merge_pairs(pair, new_id, byte_encoded_chars)

        return byte_encoded_chars

    def decode(self, encoded_tokens: list[int]):
        result = []
        for t in encoded_tokens:
            if t in self._int_to_unicode_vocab:
                result.append(self._int_to_unicode_vocab[t])
            else:
                result.append(b"<UNK>")
        # Remember our result contains bytes, so we need to join them and decode them
        return b"".join(result).decode("utf-8", errors="replace")

    def _construct_unicode_to_int_vocab(self) -> dict[ByteString, int]:
        """
        Returns a dict: char -> int, for 0..255, plus expansions for non-printable control characters
        """
        unicode_to_int_vocab = {}
        for i in range(256):
            # Each i is mapped to the single byte b"\x00" ... b"\xff"
            unicode_to_int_vocab[bytes([i])] = i
        for i, special_char in enumerate(self.SPECIAL_TOKENS):
            unicode_to_int_vocab[special_char.encode("utf-8")] = 256 + i
        return unicode_to_int_vocab

    def _pretokenize(self, input_text: str) -> list[str]:
        r"""
        From OpenAI GPT-2 regex pattern
        https://github.com/openai/tiktoken/blob/63527649963def8c759b0f91f2eb69a40934e468/tiktoken_ext/openai_public.py#L9-L14

        - `?\p{L}+` grabs runs of letters (including Unicode letters) optionally preceded by a space.
        - `?\p{N}+` similarly for digits (e.g. `" 2022"` becomes a token) optionally preceded by a space.
        - `?[^\s\p{L}\p{N}]+` captures punctuation or other symbols optionally preceded by a space.
        """
        # We handle the special tokens first
        special_pattern = (
            "(" + "|".join(regex.escape(k) for k in self.SPECIAL_TOKENS) + ")"
        )
        splitted_input = regex.split(special_pattern, input_text)

        general_pattern = regex.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""
        )

        final_chunks = []
        for segment in splitted_input:
            if not segment:  # Skip empty splits
                continue
            if segment in self.SPECIAL_TOKENS:
                # Preserve special tokens as-is
                final_chunks.append(segment)
            else:
                # Apply secondary regex for normal text
                final_chunks.extend(general_pattern.findall(segment))

        return final_chunks

    def _get_bigrams_to_count(self, tokens: list[int]) -> dict[list[tuple[int]], int]:
        """
        Count how frequently each adjacent pair of tokens occurs in the entire corpus.
        Note: This function should be called before merging any bigrams/pairs.

        Args:
            tokens (list[int]): The list of integers representing the characters in the original corpus

        Returns:
            dict[list[tuple[int]], int]: A dictionary mapping each pair of tokens to the number of times it occurs
        """
        counter = Counter()
        for tup in list(zip(tokens[:-1], tokens[1:])):
            counter[tup] += 1
        return counter

    def _merge_pairs(
        self, pair: tuple[int, int], new_idx: int, corpus: list[int]
    ) -> list[int]:
        """
        Given a pair (a, b) and its new_id, replace all occurrences of (a, b) in each line
        with [new_id].

        Example:
            corpus = [10, 11, 11, 12]
            pair = (10, 11), new_id=256
            result -> [256, 11, 12]
        """
        merged_tokens = []
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
    bpe = BytePairEncoder(num_merges=50)
    with open("autograd/text/taylorswift.txt", "r", encoding="utf-8") as f:
        original_text = f.read()
    encoded_tokens = bpe.encode(original_text)
    decoded_tokens = bpe.decode(encoded_tokens)
    logger.info(f"Size of vocabulary: {len(bpe._int_to_unicode_vocab)}")
    logger.info(original_text[:50])
    logger.info(encoded_tokens[:50])
    logger.info(decoded_tokens[:50])
