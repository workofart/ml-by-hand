from unittest import TestCase
import os

from autograd.text.tokenizer import BytePairEncoder


class TestTokenizer(TestCase):
    def setUp(self):
        self.bpe = BytePairEncoder(num_merges=50, vocab_file_path="test_vocab.pkl")
        with open("test/autograd/text/test_text.txt", "r", encoding="utf-8") as f:
            self.original_text = f.read()

    def tearDown(self) -> None:
        os.remove("test_vocab.pkl")

    def test_construct_unicode_to_int_vocab(self):
        assert len(self.bpe._construct_unicode_to_int_vocab()) == 256 + len(
            self.bpe.SPECIAL_TOKENS
        )

    def test_get_bigrams_to_count(self):
        list_of_encoded_ints = [10, 11, 11, 11, 12]
        bigrams = self.bpe._get_bigrams_to_count(list_of_encoded_ints)
        assert sorted(bigrams) == sorted({(10, 11): 1, (11, 11): 2, (11, 12): 1})

    def test_merge_pairs(self):
        corpus = [10, 11, 11, 12]
        pair = (10, 11)
        new_idx = 256
        merged_tokens = self.bpe._merge_pairs(pair, new_idx, corpus)
        assert merged_tokens == [256, 11, 12]

    def test_encode_decode(self):
        input_text = (
            self.original_text + self.bpe.SPECIAL_TOKENS[0] + self.original_text
        )
        encoded_tokens = self.bpe.encode(input_text)

        # Check if special token is correctly encoded as a single integer, not broken down
        assert (
            self.bpe._unicode_to_int_vocab[self.bpe.SPECIAL_TOKENS[0].encode("utf-8")]
            in encoded_tokens
        )

        decoded_tokens = self.bpe.decode(encoded_tokens)
        assert input_text == decoded_tokens

        # Test empty input
        input_text = ""
        encoded_tokens = self.bpe.encode(input_text)
        decoded_text = self.bpe.decode(encoded_tokens)

        # Both encoded and decoded outputs should be empty
        assert encoded_tokens == [], f"Expected empty list, got {encoded_tokens}"
        assert decoded_text == "", f"Expected empty string, got {decoded_text}"

    def test_vocab_loading(self):
        input_text = "Hello world <PAD>"
        self.bpe.train_vocabulary(input_text, overwrite_saved_file=True)

        # Re-instantiate and check if vocab loads correctly
        new_bpe = BytePairEncoder(num_merges=50)
        new_bpe.train_vocabulary(
            "", overwrite_saved_file=False
        )  # Should load from disk

        assert self.bpe._unicode_to_int_vocab == new_bpe._unicode_to_int_vocab
        assert self.bpe._int_to_unicode_vocab == new_bpe._int_to_unicode_vocab

    def test_pretokenize(self):
        input_text = f"Hello hello {self.bpe.SPECIAL_TOKENS[0]} world! {self.bpe.SPECIAL_TOKENS[1]}"

        # Make sure the special tokens are not broken down
        expected_output = [
            "Hello",
            " hello",
            " ",
            self.bpe.SPECIAL_TOKENS[0],
            " world",
            "!",
            " ",
            self.bpe.SPECIAL_TOKENS[1],
        ]

        result = self.bpe._pretokenize(input_text)

        # Check if the result matches the expected tokens
        assert result == expected_output, f"Expected {expected_output}, got {result}"

    def test_encode_handles_special_tokens(self):
        input_text = (
            f"{self.bpe.SPECIAL_TOKENS[0]}Hello world{self.bpe.SPECIAL_TOKENS[1]}"
        )
        encoded_tokens = self.bpe.encode(input_text)

        assert (
            self.bpe._unicode_to_int_vocab[self.bpe.SPECIAL_TOKENS[0].encode("utf-8")]
            in encoded_tokens
        )
        assert (
            self.bpe._unicode_to_int_vocab[self.bpe.SPECIAL_TOKENS[1].encode("utf-8")]
            in encoded_tokens
        )

    def test_decode_handles_special_tokens(self):
        special_token_ids = [
            self.bpe._unicode_to_int_vocab[token.encode("utf-8")]
            for token in self.bpe.SPECIAL_TOKENS
        ]
        input_tokens = [
            special_token_ids[0],
            72,
            101,
            108,
            108,
            111,
            special_token_ids[1],
        ]
        expected_output = (
            f"{self.bpe.SPECIAL_TOKENS[0]}Hello{self.bpe.SPECIAL_TOKENS[1]}"
        )

        decoded_text = self.bpe.decode(input_tokens)

        # Check if decoding produces the correct text
        assert (
            decoded_text == expected_output
        ), f"Expected {expected_output}, got {decoded_text}"

    def test_no_merges_with_special_tokens(self):
        input_text = (
            f"Hello {self.bpe.SPECIAL_TOKENS[0]} world {self.bpe.SPECIAL_TOKENS[1]}"
        )
        self.bpe.train_vocabulary(input_text, overwrite_saved_file=True)

        # Ensure no merges involve special tokens
        for pair, _ in self.bpe.learned_merges:
            token_1, token_2 = pair
            assert token_1 < 256, "Special token merged incorrectly"
            assert token_2 < 256, "Special token merged incorrectly"

    def test_learned_merges(self):
        # Need to have at least one repetition to be merged, otherwise the resulting output will be the same as the input
        input_text = (
            f"{self.bpe.SPECIAL_TOKENS[0]}Hello Hello world{self.bpe.SPECIAL_TOKENS[1]}"
        )
        self.bpe.train_vocabulary(input_text, overwrite_saved_file=True)

        # Check if some merges were learned
        assert len(self.bpe.learned_merges) > 0, "No merges were learned"

        encoded_tokens = self.bpe.encode(input_text)
        decoded_text = self.bpe.decode(encoded_tokens)

        # Ensure the encoded and decoded text matches the input
        assert decoded_text == input_text, f"Expected {input_text}, got {decoded_text}"

    def test_non_ascii_characters(self):
        input_text = "🤖🤖🤖 <|endoftext|>"
        encoded_tokens = self.bpe.encode(input_text)
        decoded_text = self.bpe.decode(encoded_tokens)

        assert decoded_text == input_text, f"Expected {input_text}, got {decoded_text}"
