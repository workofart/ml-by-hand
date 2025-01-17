from unittest import TestCase
from autograd.text.tokenizer import BytePairEncoder


class TestTokenizer(TestCase):
    def setUp(self):
        self.bpe = BytePairEncoder(num_merges=50)
        with open("test/autograd/text/test_text.txt", "r", encoding="utf-8") as f:
            self.original_text = f.read()

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
        encoded_tokens = self.bpe.encode(self.original_text)
        decoded_tokens = self.bpe.decode(encoded_tokens)
        assert self.original_text == decoded_tokens
