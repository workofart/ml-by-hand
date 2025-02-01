import unittest
from unittest.mock import patch

import numpy as np

from autograd.text import utils as text_utils
from autograd.tools.data import LLMDataLoader, SimpleDataLoader


def mock_causal_mask(seq_len, batch_size):
    # Create a lower-triangular mask of ones with shape (seq_len, seq_len)
    mask = np.tril(np.ones((seq_len, seq_len)))
    # Broadcast to (batch_size, 1, seq_len, seq_len)
    return np.broadcast_to(mask, (batch_size, 1, seq_len, seq_len))


def mock_padding_mask(X_chunk, pad_idx):
    # For testing, assume no actual padding occurs.
    # Return a zero mask of shape (batch_size, 1, 1, seq_len)
    batch_size, seq_len = X_chunk.shape
    return np.zeros((batch_size, 1, 1, seq_len))


class MockBPE:
    def encode(self, token, allowed_special=set()):
        if token in allowed_special:
            if token == "<PAD>":
                return [0]
            elif token == "<SOS>":
                return [1]
        # Fallback: use the ASCII code of the first character.
        return [ord(token[0])]


class TestDataLoaders(unittest.TestCase):
    def setUp(self):
        self.X = np.arange(20).reshape(10, 2)
        self.y = np.arange(10)
        self.data = np.arange(200)
        self.seq_len = 10
        self.batch_size_simple = 3
        self.batch_size_llm = 4
        self.steps = 5
        self.bpe = MockBPE()

    def test_simple_dataloader_no_shuffle(self):
        loader = SimpleDataLoader(
            self.X, self.y, batch_size=self.batch_size_simple, shuffle=False
        )
        expected_indices = np.arange(len(self.X))
        self.assertTrue(np.array_equal(loader.indices, expected_indices))
        batches = list(loader)
        expected_batches = (
            len(self.X) + self.batch_size_simple - 1
        ) // self.batch_size_simple
        self.assertEqual(len(batches), expected_batches)
        reconstructed_y = np.concatenate([batch[1] for batch in batches])
        self.assertTrue(np.array_equal(reconstructed_y, self.y))

    def test_simple_dataloader_shuffle(self):
        loader = SimpleDataLoader(
            self.X, self.y, batch_size=self.batch_size_simple, shuffle=True
        )
        loader.on_epoch_start()
        self.assertTrue(np.array_equal(np.sort(loader.indices), np.arange(len(self.X))))

    def test_simple_dataloader_length(self):
        loader = SimpleDataLoader(
            self.X, self.y, batch_size=self.batch_size_simple, shuffle=False
        )
        expected_batches = (
            len(self.X) + self.batch_size_simple - 1
        ) // self.batch_size_simple
        self.assertEqual(len(loader), expected_batches)

    def test_simple_dataloader_preprocess(self):
        X_orig = np.array([[1, 2], [3, 4]])
        y_orig = np.array([10, 20])
        loader = SimpleDataLoader(
            X_orig.copy(), y_orig.copy(), batch_size=1, shuffle=False
        )

        def preprocess_func(X_in, y_in):
            return X_in * 2, y_in * 3

        loader.preprocess(preprocess_func)
        np.testing.assert_array_equal(loader.X, X_orig * 2)
        np.testing.assert_array_equal(loader.y, y_orig * 3)

    # Use decorators to patch the text_utils functions for LLMDataLoader tests.
    @patch.object(text_utils, "mock_causal_mask", side_effect=mock_causal_mask)
    @patch.object(text_utils, "mock_padding_mask", side_effect=mock_padding_mask)
    def test_llm_dataloader_length(self, mock_padding, mock_causal):
        loader = LLMDataLoader(
            self.data,
            self.bpe,
            batch_size=self.batch_size_llm,
            seq_len=self.seq_len,
            steps_per_epoch=self.steps,
        )
        self.assertEqual(len(loader), self.steps)
        loader_infinite = LLMDataLoader(
            self.data,
            self.bpe,
            batch_size=self.batch_size_llm,
            seq_len=self.seq_len,
            steps_per_epoch=None,
        )
        with self.assertRaises(NotImplementedError):
            _ = len(loader_infinite)

    @patch.object(text_utils, "mock_causal_mask", side_effect=mock_causal_mask)
    @patch.object(text_utils, "mock_padding_mask", side_effect=mock_padding_mask)
    def test_llm_dataloader_small_data(self, mock_padding, mock_causal):
        data_small = np.arange(5)
        loader = LLMDataLoader(
            data_small, self.bpe, batch_size=2, seq_len=self.seq_len, steps_per_epoch=1
        )
        it = iter(loader)
        with self.assertRaises(ValueError):
            next(it)

    @patch.object(text_utils, "mock_causal_mask", side_effect=mock_causal_mask)
    @patch.object(text_utils, "mock_padding_mask", side_effect=mock_padding_mask)
    def test_llm_dataloader_output(self, mock_padding, mock_causal):
        loader = LLMDataLoader(
            self.data,
            self.bpe,
            batch_size=self.batch_size_llm,
            seq_len=self.seq_len,
            steps_per_epoch=1,
        )
        batch = next(iter(loader))
        self.assertEqual(len(batch), 6)
        X_chunk, dec_inp, Y_chunk, smask, tmask, causal_mask = batch
        self.assertEqual(X_chunk.shape, (self.batch_size_llm, self.seq_len))
        self.assertEqual(Y_chunk.shape, (self.batch_size_llm, self.seq_len))
        if loader.include_decoder_input and loader.create_decoder_inp:
            self.assertEqual(dec_inp.shape, (self.batch_size_llm, self.seq_len))
            self.assertTrue(np.all(dec_inp[:, 0] == loader.sos_idx))
        else:
            self.assertIsNone(dec_inp)
        if loader.create_masks:
            self.assertEqual(smask.shape, (self.batch_size_llm, 1, 1, self.seq_len))
            self.assertEqual(
                tmask.shape, (self.batch_size_llm, 1, self.seq_len, self.seq_len)
            )
            self.assertEqual(
                causal_mask.shape, (self.batch_size_llm, 1, self.seq_len, self.seq_len)
            )
        else:
            self.assertIsNone(smask)
            self.assertIsNone(tmask)
            self.assertIsNone(causal_mask)

    @patch.object(text_utils, "mock_causal_mask", side_effect=mock_causal_mask)
    @patch.object(text_utils, "mock_padding_mask", side_effect=mock_padding_mask)
    def test_llm_dataloader_no_decoder_input(self, mock_padding, mock_causal):
        loader = LLMDataLoader(
            self.data,
            self.bpe,
            batch_size=self.batch_size_llm,
            seq_len=self.seq_len,
            steps_per_epoch=1,
            include_decoder_input=False,
        )
        batch = next(iter(loader))
        self.assertIsNone(batch[1])
