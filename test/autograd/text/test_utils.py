from unittest import TestCase
import numpy as np
import re

from autograd.text.utils import (
    create_vocabulary,
    text_to_one_hot_and_sparse,
    create_padding_mask,
    create_causal_mask,
    clean_and_tokenize,
)


class TestTextUtils(TestCase):
    def test_create_vocabulary_basic(self):
        texts = ["I love apples", "I love to eat apples every day", "Apples are great"]
        vocab = create_vocabulary(texts, max_features=5)

        # Check that we have <PAD> and <UNK>
        self.assertIn("<PAD>", vocab)
        self.assertIn("<UNK>", vocab)

        # Because max_features=5, we expect exactly 5 tokens in the vocab
        self.assertEqual(len(vocab), 5)

        # Check that "apples" is definitely in the vocabulary
        self.assertIn("apples", vocab)

    def test_create_vocabulary_custom_tokenizer(self):
        # Custom tokenizer that splits on punctuation
        def custom_tok(text):
            return re.split(r"[,\s]+", text.lower())

        texts = ["Hello,world", "Hello, Universe!"]
        vocab = create_vocabulary(texts, max_features=6, custom_tokenizer=custom_tok)

        self.assertIn("hello", vocab)
        self.assertIn("world", vocab)
        self.assertIn("universe!", vocab)
        self.assertIn("<PAD>", vocab)
        self.assertIn("<UNK>", vocab)

    def test_text_to_one_hot_and_sparse(self):
        texts = ["I love apples", "I love apples too"]
        vocab = create_vocabulary(texts, max_features=10)  # large enough
        max_seq_len = 4

        one_hot, matrix = text_to_one_hot_and_sparse(
            texts, vocab, max_seq_len, pad_str="<PAD>"
        )

        # Check shape
        self.assertEqual(one_hot.shape, (2, max_seq_len, len(vocab)))
        self.assertEqual(matrix.shape, (2, max_seq_len))

        # Check that the same positions in matrix have 1 in one_hot
        for i in range(2):
            for j in range(max_seq_len):
                idx = matrix[i, j]
                self.assertTrue(one_hot[i, j, idx] == 1.0)

        pad_idx = vocab["<PAD>"]
        # "I love apples" => 3 words => last index should be pad_idx
        self.assertEqual(matrix[0, 3], pad_idx)

    def test_create_padding_mask_default_dims(self):
        token_indices = np.array(
            [
                [1, 2, 0, 0],  # 0 => pad
                [3, 4, 5, 0],
            ],
            dtype=np.int32,
        )

        mask = create_padding_mask(token_indices, pad_idx=0, dims=None)
        # shape => (batch_size=2, 1, 1, seq_len=4)
        self.assertEqual(mask.shape, (2, 1, 1, 4))

        # 1.0 where token_indices==0
        # first row => positions 2,3 are 1
        np.testing.assert_array_equal(mask[0, 0, 0], [0, 0, 1, 1])
        np.testing.assert_array_equal(mask[1, 0, 0], [0, 0, 0, 1])

    def test_create_padding_mask_custom_dims(self):
        token_indices = np.array([[1, 0, 0], [2, 2, 0]], dtype=np.int32)

        # Suppose we want dims = (batch_size, 1, 3)
        # i.e. (2, 1, 3) => effectively shape for broadcasting
        mask = create_padding_mask(token_indices, pad_idx=0, dims=(2, 1, 3))

        self.assertEqual(mask.shape, (2, 1, 3))
        # For first row => [1,0,0] => pad=0 => positions 1,2 => mask=1 => [0,1,1]
        np.testing.assert_array_equal(mask[0, 0], [0, 1, 1])
        # For second row => [2,2,0] => only last is 0 => [0,0,1]
        np.testing.assert_array_equal(mask[1, 0], [0, 0, 1])

    def test_create_causal_mask_lookforward(self):
        seq_len = 4
        batch_size = 2
        mask = create_causal_mask(
            seq_len, batch_size, lookback=False, mask_diagonal=True
        )

        # shape => (batch_size, 1, seq_len, seq_len)
        self.assertEqual(mask.shape, (2, 1, 4, 4))
        # For lookforward, upper triangle is masked.
        # Because mask_diagonal=True, diagonal is included in the mask.
        # So row0 => [1,1,1,1]
        #            [0,1,1,1]
        #            [0,0,1,1]
        #            [0,0,0,1]
        expected_single = np.array(
            [[1, 1, 1, 1], [0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=np.float32
        )

        for b in range(batch_size):
            np.testing.assert_array_equal(mask[b, 0], expected_single)

    def test_create_causal_mask_lookback_no_diag(self):
        seq_len = 3
        batch_size = 1
        mask = create_causal_mask(
            seq_len, batch_size, lookback=True, mask_diagonal=False
        )
        # shape => (1, 1, 3, 3)
        # For lookback, we mask the lower triangle.  mask_diagonal=False => diagonal is not masked.
        # So the result should mask strictly below diagonal
        # row0 => [0,0,0]
        #         [1,0,0]
        #         [1,1,0]
        expected = np.array([[[[0, 0, 0], [1, 0, 0], [1, 1, 0]]]], dtype=np.float32)
        self.assertEqual(mask.shape, (1, 1, 3, 3))
        np.testing.assert_array_equal(mask, expected)

    def test_clean_and_tokenize_default(self):
        text = "Hello, World! (Testing) \n new-lines?"
        tokens = clean_and_tokenize(text)
        # default pattern => r"\w+|[^\w\s]|[\n\s]"
        # expect punctuation kept, lowercased => "hello", ",", "world", "!", "(", "testing", ")", "new", "-", "lines", "?"
        self.assertTrue("hello" in tokens)
        self.assertTrue("," in tokens)
        self.assertTrue("world" in tokens)
        self.assertTrue("!" in tokens)
        self.assertTrue("(" in tokens)
        self.assertTrue("testing" in tokens)
        self.assertTrue(")" in tokens)
        self.assertTrue("-" in tokens)
        self.assertTrue("lines" in tokens)
        self.assertTrue("?" in tokens)

    def test_clean_and_tokenize_custom_pattern(self):
        text = "Hello, world! This is 2025"
        # let's split only on spaces and punctuation ignoring digits
        pattern = r"[a-zA-Z]+|[^\w\s]"  # just alpha words + punctuation
        tokens = clean_and_tokenize(text, pattern=pattern, lowercase=False)
        # Because lowercase=False, "Hello" should remain "Hello"
        self.assertIn("Hello", tokens)
        # "2025" should be excluded by pattern
        self.assertNotIn("2025", tokens)
        # punctuation like "," and "!" is included
        self.assertIn(",", tokens)
        self.assertIn("!", tokens)
