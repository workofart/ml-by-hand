import re
from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np

from autograd.text.utils import (
    clean_and_tokenize,
    create_causal_mask,
    create_padding_mask,
    create_vocabulary,
    inference,
    text_to_one_hot_and_sparse,
)


class MockedBPE:
    def encode(self, text: str):
        if text == "<SOS>":
            return [0]
        # For simplicity, convert each uppercase letter to an integer (A=0, B=1, …)
        return [ord(ch) - 65 for ch in text if ch.isupper()]

    def decode(self, tokens: list) -> str:
        return "".join(chr(t + 65) for t in tokens)


class TestTextUtils(TestCase):
    def setUp(self) -> None:
        self.bpe = MockedBPE()

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
        expected_single = np.array(
            [[1, 1, 1, 1], [0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=np.float32
        )

        for b in range(batch_size):
            np.testing.assert_array_equal(mask[b, 0], expected_single)

        mask = create_causal_mask(
            seq_len, batch_size, lookback=False, mask_diagonal=False
        )

        # shape => (batch_size, 1, seq_len, seq_len)
        self.assertEqual(mask.shape, (2, 1, 4, 4))
        # For lookforward, upper triangle is not masked.
        # Because mask_diagonal=False, diagonal is not included in the mask.
        expected_single = np.array(
            [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=np.float32
        )

        for b in range(batch_size):
            np.testing.assert_array_equal(mask[b, 0], expected_single)

    def test_create_causal_mask_lookback(self):
        seq_len = 3
        batch_size = 1
        mask = create_causal_mask(
            seq_len, batch_size, lookback=True, mask_diagonal=False
        )
        # shape => (1, 1, 3, 3)
        # For lookback, we mask the lower triangle.  mask_diagonal=False => diagonal is not masked.
        # So the result should mask strictly below diagonal
        expected = np.array([[[[0, 0, 0], [1, 0, 0], [1, 1, 0]]]], dtype=np.float32)
        self.assertEqual(mask.shape, (1, 1, 3, 3))
        np.testing.assert_array_equal(mask, expected)

        mask = create_causal_mask(
            seq_len, batch_size, lookback=True, mask_diagonal=True
        )
        # For lookback, we mask the lower triangle.  mask_diagonal=True => diagonal is masked.
        # So the result should mask including diagonal
        expected = np.array([[[[1, 0, 0], [1, 1, 0], [1, 1, 1]]]], dtype=np.float32)
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

    @patch("numpy.random.choice")
    def test_normal_inference(self, mock_choice):
        """
        In normal (auto-regressive) mode, we expect the inference loop to use sampling.
        We patch np.random.choice to always return 1. Also, we simulate the prediction_func
        with a MagicMock that returns a dummy object with a proper 'data' attribute.
        """

        def fake_prediction(model, batch_data, mode):
            # Determine the current sequence length.
            seq_len = batch_data.shape[1]
            # Build a dummy array of shape (1, seq_len, vocab_size); the values don't matter
            # because np.random.choice is patched.
            dummy_arr = np.zeros((1, seq_len, 10))
            dummy_obj = MagicMock()
            dummy_obj.data = dummy_arr
            return dummy_obj

        # Always return 1 when sampling (so our token will be 1 → 'B').
        mock_choice.return_value = 1
        prediction_func = MagicMock(side_effect=fake_prediction)

        # For predictability, use a small max_length.
        max_length = 3
        result = inference(
            model=MagicMock(),
            prediction_func=prediction_func,
            bpe=self.bpe,  # type: ignore
            start_tokens="<SOS>",
            groundtruth_data=None,
            max_length=max_length,
            temperature=1.0,
            top_k=5,
        )
        # bpe.encode("<SOS>") returns [0] → 'A'
        # Then we append token 1 for each iteration → 'B'
        # Final output: [0] + [1, 1, 1] decodes to "A" followed by "B" repeated 3 times: "ABBB"
        expected = self.bpe.decode([0] + [1] * max_length)
        self.assertEqual(result, expected)
        # Ensure prediction_func was called exactly max_length times.
        self.assertEqual(prediction_func.call_count, max_length)

    def test_teacher_forcing_inference(self):
        """
        In teacher forcing mode, the inference function should use argmax (temperature=0).
        We simulate a prediction function that returns dummy logits such that np.argmax
        returns the next groundtruth token.
        For groundtruth [0, 1, 2, 3], we expect the output tokens to be 0 then 1, 2, and 3.
        """
        groundtruth = np.array([0, 1, 2, 3])

        def fake_prediction_teacher(model, batch_data, mode):
            # x is an array of shape (1, seq_len). The current seq_len tells us which token to predict.
            seq_len = batch_data.shape[1]
            # In teacher forcing, for iteration i (where i = seq_len - 1), we want to predict groundtruth[i+1].
            token = groundtruth[
                seq_len
            ]  # e.g., for seq_len=1, predict groundtruth[1] which is 1.
            # Create logits with a high value at the desired token.
            logits = np.full(10, -100.0)
            logits[token] = 100.0
            # Build an array of shape (1, seq_len, 10) and place logits at the last position.
            pred_arr = np.zeros((1, seq_len, 10))
            pred_arr[0, -1] = logits
            mock_tensor = MagicMock()
            mock_tensor.data = pred_arr
            return mock_tensor, None

        prediction_func = MagicMock(side_effect=fake_prediction_teacher)

        # Even if max_length is larger, teacher forcing uses groundtruth length.
        result = inference(
            model=MagicMock(),
            prediction_func=prediction_func,
            bpe=self.bpe,  # type: ignore
            groundtruth_data=groundtruth,
            max_length=10,
            temperature=1.0,  # overridden to 0 in teacher forcing
            top_k=5,  # overridden to 1 in teacher forcing
        )
        # The output tokens should be: [groundtruth[0], groundtruth[1], groundtruth[2], groundtruth[3]]
        expected = self.bpe.decode([0, 1, 2, 3])
        self.assertEqual(result, expected)
        # Teacher forcing should have made 3 calls (len(groundtruth) - 1)
        self.assertEqual(prediction_func.call_count, 3)

    def test_teacher_forcing_single_token(self):
        """
        If groundtruth_data contains only one token, the inference loop should not be entered.
        The output should exactly match the decoded single token, and the prediction function should not be called.
        """
        groundtruth = np.array([65])
        prediction_func = MagicMock()
        result = inference(
            model=MagicMock(),
            prediction_func=prediction_func,
            bpe=self.bpe,  # type: ignore
            groundtruth_data=groundtruth,
            max_length=10,
        )
        expected = self.bpe.decode([65])
        self.assertEqual(result, expected)
        # Verify that the prediction function was never called.
        prediction_func.assert_not_called()
