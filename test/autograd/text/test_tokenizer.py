import os
from collections import Counter
from unittest import TestCase

from autograd.text.tokenizer import BytePairEncoder


class TestTokenizer(TestCase):
    def setUp(self):
        self.bpe = BytePairEncoder(num_merges=50, vocab_file_path="test_vocab.pkl")
        with open("test/autograd/text/test_text.txt", "r", encoding="utf-8") as f:
            self.original_text = f.read()

    def tearDown(self) -> None:
        if os.path.exists("test_vocab.pkl"):
            os.remove("test_vocab.pkl")

    def test_construct_unicode_to_int_vocab(self):
        vocab = self.bpe._construct_unicode_to_int_vocab()
        # 256 + number of special tokens
        self.assertEqual(len(vocab), 256 + len(self.bpe.SPECIAL_TOKENS))

    def test_pair_counting(self):
        # Instead of test_get_bigrams_to_count, we test _get_initial_pair_counts directly
        # We'll create a small word_freq manually
        word_freq = Counter(
            {
                (10, 11, 12): 3,
                (11, 12, 12): 2,
            }
        )
        pair_counts = self.bpe._get_initial_pair_counts(word_freq)

        # (10,11) occurs in the first tuple, 3 times
        # (11,12) occurs in the first tuple (3 times) and second tuple (2 times) -> total 5
        # (12,12) occurs in second tuple (2 times)
        expected = {
            (10, 11): 3,
            (11, 12): 5,
            (12, 12): 2,
        }
        self.assertEqual(pair_counts, expected)

    def test_apply_merges_to_corpus(self):
        # Instead of test_merge_pairs, we test _apply_merges_to_corpus
        word_freq = Counter(
            {
                (10, 11, 11, 12): 2,
            }
        )
        pair_counts = {
            (10, 11): 2,
            (11, 11): 2,
            (11, 12): 2,
        }
        pair = (10, 11)
        new_id = 256

        self.bpe._apply_merges_to_corpus(pair, new_id, word_freq, pair_counts)

        # The old tuple (10, 11, 11, 12) should be removed
        # The new tuple is (256, 11, 12)
        # And word_freq should have updated counts
        self.assertFalse((10, 11, 11, 12) in word_freq)
        self.assertEqual(word_freq[(256, 11, 12)], 2)

        # Also check that pair_counts updated
        # old bigrams: (10,11), (11,11), (11,12)
        # new bigrams: (256,11), (11,12)
        self.assertFalse((10, 11) in pair_counts)
        self.assertFalse((11, 11) in pair_counts)
        self.assertTrue((11, 12) in pair_counts)  # still remain in the pair_counts
        self.assertIn((256, 11), pair_counts)
        self.assertIn((11, 12), pair_counts)

    def test_encode_decode(self):
        input_text = self.original_text + "<|endoftext|>" + self.original_text
        encoded = self.bpe.encode(input_text)
        decoded = self.bpe.decode(encoded)
        self.assertEqual(input_text, decoded)

    def test_special_tokens_encoded_as_single_id(self):
        # This checks special tokens remain single ID
        st_id = self.bpe._unicode_to_int_vocab[
            self.bpe.SPECIAL_TOKENS[0].encode("utf-8")
        ]
        self.assertIsNotNone(st_id)

        input_text = self.bpe.SPECIAL_TOKENS[0]
        encoded = self.bpe.encode(input_text)
        self.assertIn(st_id, encoded)
        self.assertEqual(self.bpe.decode(encoded), input_text)

    # ... etc. (include or adapt other tests similarly)
