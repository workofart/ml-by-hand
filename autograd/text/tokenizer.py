from collections import Counter
import regex


class BytePairEncoder:
    SPECIAL_TOKENS = ["<|endoftext|>"]

    def __init__(self, num_merges=500) -> None:
        self.num_merges = num_merges
        self._unicode_to_int_vocab = self._construct_unicode_to_int_vocab()
        print(self._unicode_to_int_vocab)
        self._int_to_unicode_vocab = dict(
            zip(self._unicode_to_int_vocab.values(), self._unicode_to_int_vocab.keys())
        )

        # start merged token ids from the first unused index after the base vocabulary
        self.new_idx = max(self._unicode_to_int_vocab.values()) + 1

    def encode(self, input_text: str) -> list[int]:
        text_chunks = self._pretokenize(input_text)
        print(f"Text chunks: {text_chunks[:10]}")

        # Convert a list of strings to a list of integers,
        # where each integer is the index of the character in the vocabulary
        byte_encoded_chars = [
            char for string in text_chunks for char in list(string.encode("utf-8"))
        ]
        print(f"Byte encoded chars: {byte_encoded_chars[:10]}")

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
            self._int_to_unicode_vocab[new_id] = (
                self._int_to_unicode_vocab[best_pair[0]]
                + self._int_to_unicode_vocab[best_pair[1]]
            )

            # Update the original encoded chars with the merged one
            byte_encoded_chars = self._merge_pairs(
                best_pair, new_id, byte_encoded_chars
            )

            print(
                f"Merge {i+1}/{self.num_merges}: {best_pair} -> {new_id} ({self._int_to_unicode_vocab[new_id]}) had {pair_counts[best_pair]} occurrences"
            )
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

    def _construct_unicode_to_int_vocab(self) -> dict[str, int]:
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
        pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""
        pattern = regex.compile(pattern)
        chunks = pattern.findall(input_text)

        return chunks

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
        for tuple in list(zip(tokens[:-1], tokens[1:])):
            counter[tuple] += 1
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
    print(f"Size of vocabulary: {len(bpe._int_to_unicode_vocab)}")
    print(original_text[:50])
    print(encoded_tokens[:50])
    print(decoded_tokens[:50])
