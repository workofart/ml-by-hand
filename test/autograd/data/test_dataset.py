import unittest

import pytest

from autograd.backend import xp
from autograd.data.dataset import (
    MapDataset,
    PairedMapDataset,
    TokenWindowMapDataset,
)


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.X = xp.arange(20).reshape(10, 2)
        self.y = xp.arange(10)

    def test_paired_dataset_yields_single_example(self):
        dataset = PairedMapDataset(self.X, self.y)

        example = next(iter(dataset))

        self.assertEqual(tuple(example.keys()), ("inputs", "targets"))
        self.assertTrue(xp.array_equal(example["inputs"], self.X[0]))
        self.assertEqual(int(example["targets"]), int(self.y[0]))

    def test_map_dataset_iteration_stays_in_stored_order_after_epoch_start(self):
        dataset = PairedMapDataset(self.X, self.y)

        dataset.on_epoch_start()

        examples = list(dataset)

        self.assertTrue(xp.array_equal(examples[0]["inputs"], self.X[0]))
        self.assertEqual(int(examples[0]["targets"]), int(self.y[0]))

    def test_map_dataset_yields_pre_shaped_dicts(self):
        dataset = MapDataset(
            [
                {
                    "input_ids": xp.array([1, 2], dtype=xp.int32),
                    "labels": xp.array([3, 4], dtype=xp.int32),
                }
            ]
        )

        example = dataset[0]

        self.assertEqual(tuple(example.keys()), ("input_ids", "labels"))
        self.assertTrue(
            xp.array_equal(example["input_ids"], xp.array([1, 2], dtype=xp.int32))
        )
        self.assertTrue(
            xp.array_equal(example["labels"], xp.array([3, 4], dtype=xp.int32))
        )

    def test_token_window_dataset_is_map_style(self):
        dataset = TokenWindowMapDataset(
            xp.arange(10, dtype=xp.int32),
            window_len=5,
        )

        self.assertIsInstance(dataset, MapDataset)

    def test_seq2seq_dataset_yields_single_example(self):
        dataset = PairedMapDataset(
            [xp.array([1, 2])],
            [xp.array([3, 4])],
            input_key="input_ids",
            target_key="labels",
            dtype=xp.int32,
        )

        example = next(iter(dataset))

        self.assertTrue(
            xp.array_equal(example["input_ids"], xp.array([1, 2], dtype=xp.int32))
        )
        self.assertTrue(
            xp.array_equal(example["labels"], xp.array([3, 4], dtype=xp.int32))
        )

    def test_token_sequence_dataset_yields_single_example(self):
        dataset = PairedMapDataset(
            [xp.arange(5, dtype=xp.int32)],
            [xp.ones((5,), dtype=xp.int32)],
            input_key="tokens",
            target_key="loss_mask",
            dtype=xp.int32,
        )

        example = next(iter(dataset))

        self.assertEqual(tuple(example.keys()), ("tokens", "loss_mask"))
        self.assertTrue(xp.array_equal(example["tokens"], xp.arange(5, dtype=xp.int32)))
        self.assertTrue(
            xp.array_equal(example["loss_mask"], xp.ones((5,), dtype=xp.int32))
        )

    def test_token_sequence_dataset_holds_tokenized_causal_lm_examples(self):
        dataset = PairedMapDataset(
            [xp.array([65, 66, 67, 2, 68, 69, 2], dtype=xp.int32)],
            [xp.array([0, 0, 0, 0, 1, 1, 1], dtype=xp.int32)],
            input_key="tokens",
            target_key="loss_mask",
            dtype=xp.int32,
        )

        example = next(iter(dataset))

        self.assertTrue(
            xp.array_equal(
                example["tokens"], xp.array([65, 66, 67, 2, 68, 69, 2], dtype=xp.int32)
            )
        )
        self.assertTrue(
            xp.array_equal(
                example["loss_mask"], xp.array([0, 0, 0, 0, 1, 1, 1], dtype=xp.int32)
            )
        )


def test_minimal_token_stream_is_valid():
    data = xp.arange(5, dtype=xp.int32)
    dataset = TokenWindowMapDataset(data, window_len=5)

    examples = list(dataset)

    assert len(examples) == 1
    assert examples[0].offset == 0


def test_last_offset_is_included():
    data = xp.arange(10, dtype=xp.int32)
    window_len = 5
    dataset = TokenWindowMapDataset(data, window_len=window_len)

    offsets = [example.offset for example in dataset]

    assert offsets[-1] == len(data) - window_len


def test_token_window_dataset_rejects_too_short_stream():
    with pytest.raises(ValueError, match="Need at least 6 tokens, got 5"):
        TokenWindowMapDataset(
            xp.arange(5, dtype=xp.int32),
            window_len=6,
        )


def test_token_window_dataset_maps_offset_to_lazy_example():
    data = xp.arange(10, dtype=xp.int32)
    dataset = TokenWindowMapDataset(data, window_len=5)

    example = dataset[3]

    assert example.offset == 3
    assert example.window_len == 5
    assert example.stream is dataset.stream
