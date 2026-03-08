import os
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from autograd.backend import xp
from autograd.text import utils as text_utils
from autograd.tools.data import (
    LLMDataLoader,
    SimpleDataLoader,
    load_data,
)


def mock_causal_mask(seq_len, batch_size):
    # Create a lower-triangular mask of ones with shape (seq_len, seq_len)
    mask = xp.tril(xp.ones((seq_len, seq_len)))
    # Broadcast to (batch_size, 1, seq_len, seq_len)
    return xp.broadcast_to(mask, (batch_size, 1, seq_len, seq_len))


def mock_padding_mask(X_chunk, pad_idx):
    # For testing, assume no actual padding occurs.
    # Return a zero mask of shape (batch_size, 1, 1, seq_len)
    batch_size, seq_len = X_chunk.shape
    return xp.zeros((batch_size, 1, 1, seq_len))


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
        self.X = xp.arange(20).reshape(10, 2)
        self.y = xp.arange(10)
        self.data = xp.arange(200)
        self.seq_len = 10
        self.batch_size_simple = 3
        self.batch_size_llm = 4
        self.steps = 5
        self.bpe = MockBPE()

    def test_simple_dataloader_no_shuffle(self):
        loader = SimpleDataLoader(
            self.X, self.y, batch_size=self.batch_size_simple, shuffle=False
        )
        expected_indices = xp.arange(len(self.X))
        self.assertTrue(xp.array_equal(loader.indices, expected_indices))
        batches = list(loader)
        expected_batches = (
            len(self.X) + self.batch_size_simple - 1
        ) // self.batch_size_simple
        self.assertEqual(len(batches), expected_batches)
        reconstructed_y = xp.concatenate([batch[1] for batch in batches])
        self.assertTrue(xp.array_equal(reconstructed_y, self.y))

    def test_simple_dataloader_shuffle(self):
        loader = SimpleDataLoader(
            self.X, self.y, batch_size=self.batch_size_simple, shuffle=True
        )
        loader.on_epoch_start()
        self.assertTrue(xp.array_equal(xp.sort(loader.indices), xp.arange(len(self.X))))

    def test_simple_dataloader_length(self):
        loader = SimpleDataLoader(
            self.X, self.y, batch_size=self.batch_size_simple, shuffle=False
        )
        expected_batches = (
            len(self.X) + self.batch_size_simple - 1
        ) // self.batch_size_simple
        self.assertEqual(len(loader), expected_batches)

    def test_llm_dataloader_on_epoch_start_reseeds_without_crashing(self):
        loader = LLMDataLoader(
            self.data,
            self.bpe,
            batch_size=self.batch_size_llm,
            seq_len=self.seq_len,
            steps_per_epoch=1,
            shuffle=True,
        )

        loader.on_epoch_start()

    def test_simple_dataloader_preprocess(self):
        X_orig = xp.array([[1, 2], [3, 4]])
        y_orig = xp.array([10, 20])
        loader = SimpleDataLoader(
            xp.array(X_orig), xp.array(y_orig), batch_size=1, shuffle=False
        )

        def preprocess_func(X_in, y_in):
            return X_in * 2, y_in * 3

        loader.preprocess(preprocess_func)
        assert xp.array_equal(loader.X, X_orig * 2)
        assert xp.array_equal(loader.y, y_orig * 3)

    def test_load_data_reads_parquet_without_pandas(self):
        rows = [
            {
                "url": "u1",
                "title": "t1",
                "summary": "s1",
                "article": "a1",
                "step_headers": "h1",
            },
            {
                "url": "u2",
                "title": "t2",
                "summary": "s2",
                "article": "a2",
                "step_headers": "h2",
            },
        ]
        table = pa.Table.from_pylist(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = os.path.join(tmpdir, "sample.parquet")
            pq.write_table(table, parquet_path)

            data = load_data("unused", parquet_path, max_rows=1)

        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["summary"], "s1")
        self.assertEqual(data[0]["article"], "a1")

    def test_load_data_reads_csv_without_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "sample.csv")
            with open(csv_path, "w", encoding="utf-8") as handle:
                handle.write("review,sentiment\n")
                handle.write('"great movie, would watch again",positive\n')
                handle.write('"bad ending",negative\n')

            rows = load_data(csv_path, csv_path)

        self.assertEqual(
            rows,
            [
                ["great movie, would watch again", "positive"],
                ["bad ending", "negative"],
            ],
        )

    # Use decorators to patch the text_utils functions for LLMDataLoader tests.
    @patch.object(text_utils, "create_causal_mask", side_effect=mock_causal_mask)
    @patch.object(text_utils, "create_padding_mask", side_effect=mock_padding_mask)
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

    @patch.object(text_utils, "create_causal_mask", side_effect=mock_causal_mask)
    @patch.object(text_utils, "create_padding_mask", side_effect=mock_padding_mask)
    def test_llm_dataloader_small_data(self, mock_padding, mock_causal):
        data_small = xp.arange(5)
        loader = LLMDataLoader(
            data_small, self.bpe, batch_size=2, seq_len=self.seq_len, steps_per_epoch=1
        )
        it = iter(loader)
        with self.assertRaises(ValueError):
            next(it)

    @patch.object(text_utils, "create_causal_mask", side_effect=mock_causal_mask)
    @patch.object(text_utils, "create_padding_mask", side_effect=mock_padding_mask)
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
        if loader.include_decoder_input:
            self.assertEqual(dec_inp.shape, (self.batch_size_llm, self.seq_len))
            self.assertTrue(xp.all(xp.asarray(dec_inp[:, 0] == loader.sos_idx)))
        else:
            self.assertIsNone(dec_inp)
        if loader.create_padding_masks:
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

    @patch.object(text_utils, "create_causal_mask", side_effect=mock_causal_mask)
    @patch.object(text_utils, "create_padding_mask", side_effect=mock_padding_mask)
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
