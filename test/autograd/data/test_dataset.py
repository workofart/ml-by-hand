import unittest
from types import SimpleNamespace

import pytest

from autograd.backend import xp
from autograd.data.dataset import (
    IterableDataset,
    PairedIterableDataset,
    Seq2SeqDataset,
    TokenSequenceDataset,
    TokenWindowDataset,
    TransformDataset,
)


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.X = xp.arange(20).reshape(10, 2)
        self.y = xp.arange(10)

    def test_paired_iterable_dataset_yields_single_example(self):
        dataset = PairedIterableDataset(self.X, self.y, shuffle=False)

        example = next(iter(dataset))

        self.assertEqual(tuple(example.keys()), ("inputs", "targets"))
        self.assertTrue(xp.array_equal(example["inputs"], self.X[0]))
        self.assertEqual(int(example["targets"]), int(self.y[0]))

    def test_paired_iterable_dataset_shuffle(self):
        dataset = PairedIterableDataset(self.X, self.y, shuffle=True)

        dataset.on_epoch_start()

        self.assertTrue(
            xp.array_equal(xp.sort(dataset.indices), xp.arange(len(self.X)))
        )

    def test_transform_dataset_applies_example_transform(self):
        dataset = TransformDataset(
            PairedIterableDataset(
                xp.array([[1, 2], [3, 4]]), xp.array([10, 20]), shuffle=False
            ),
            transform=lambda example: {
                "inputs": example["inputs"] * 2,
                "targets": example["targets"] * 3,
            },
        )

        examples = list(dataset)

        self.assertTrue(xp.array_equal(examples[0]["inputs"], xp.array([2, 4])))
        self.assertEqual(int(examples[0]["targets"]), 30)
        self.assertTrue(xp.array_equal(examples[1]["inputs"], xp.array([6, 8])))
        self.assertEqual(int(examples[1]["targets"]), 60)

    def test_transform_dataset_can_transform_arbitrary_example_shape(self):
        class DummyDataset(IterableDataset):
            def __iter__(self):
                yield {"foo": xp.array([1, 2], dtype=xp.int32)}

            def __len__(self):
                return 1

        dataset = TransformDataset(
            DummyDataset(),
            transform=lambda example: {"bar": example["foo"] + 1},
        )

        example = next(iter(dataset))

        self.assertEqual(tuple(example.keys()), ("bar",))
        self.assertTrue(
            xp.array_equal(example["bar"], xp.array([2, 3], dtype=xp.int32))
        )

    def test_seq2seq_dataset_yields_single_example(self):
        dataset = Seq2SeqDataset(
            input_sequences=[xp.array([1, 2], dtype=xp.int32)],
            label_sequences=[xp.array([3, 4], dtype=xp.int32)],
            shuffle=False,
        )

        example = next(iter(dataset))

        self.assertTrue(
            xp.array_equal(example["input_ids"], xp.array([1, 2], dtype=xp.int32))
        )
        self.assertTrue(
            xp.array_equal(example["labels"], xp.array([3, 4], dtype=xp.int32))
        )

    def test_token_sequence_dataset_yields_single_example(self):
        dataset = TokenSequenceDataset(
            token_sequences=[xp.arange(5, dtype=xp.int32)],
            shuffle=False,
        )

        example = next(iter(dataset))

        self.assertEqual(tuple(example.keys()), ("tokens", "loss_mask"))
        self.assertTrue(xp.array_equal(example["tokens"], xp.arange(5, dtype=xp.int32)))
        self.assertTrue(
            xp.array_equal(example["loss_mask"], xp.ones((5,), dtype=xp.int32))
        )

    def test_token_sequence_dataset_requires_token_sequences(self):
        with self.assertRaises(ValueError):
            TokenSequenceDataset(shuffle=False)

    def test_token_sequence_dataset_requires_matching_loss_mask_lengths(self):
        with self.assertRaises(ValueError):
            TokenSequenceDataset(
                token_sequences=[xp.arange(5, dtype=xp.int32)],
                loss_masks=[xp.ones((4,), dtype=xp.int32)],
                shuffle=False,
            )

    def test_token_sequence_dataset_holds_tokenized_causal_lm_examples(self):
        dataset = TokenSequenceDataset(
            token_sequences=[xp.array([65, 66, 67, 2, 68, 69, 2], dtype=xp.int32)],
            loss_masks=[xp.array([0, 0, 0, 0, 1, 1, 1], dtype=xp.int32)],
            shuffle=False,
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
    dataset = TokenWindowDataset(data, window_len=5, sampling="sequential")

    examples = list(dataset)

    assert len(examples) == 1
    assert examples[0].offset == 0


def test_last_offset_is_included():
    data = xp.arange(10, dtype=xp.int32)
    window_len = 5
    dataset = TokenWindowDataset(data, window_len=window_len, sampling="sequential")

    offsets = [example.offset for example in dataset]

    assert offsets[-1] == len(data) - window_len


def test_token_window_dataset_rejects_unknown_sampling_mode():
    with pytest.raises(ValueError, match="sampling must be 'random' or 'sequential'"):
        TokenWindowDataset(
            xp.arange(5, dtype=xp.int32),
            window_len=5,
            sampling="shuffle",
        )


def test_token_window_dataset_rejects_too_short_stream():
    with pytest.raises(ValueError, match="Need at least 6 tokens, got 5"):
        TokenWindowDataset(
            xp.arange(5, dtype=xp.int32),
            window_len=6,
            sampling="sequential",
        )


def test_token_window_dataset_rejects_zero_offset_buffer_size():
    with pytest.raises(ValueError, match="offset_buffer_size must be >= 1"):
        TokenWindowDataset(
            xp.arange(10, dtype=xp.int32),
            window_len=5,
            sampling="random",
            offset_buffer_size=0,
        )


@pytest.mark.parametrize("examples_per_epoch", [0, -1])
def test_token_window_dataset_requires_positive_examples_per_epoch(
    examples_per_epoch: int,
):
    with pytest.raises(
        ValueError, match="examples_per_epoch must be >= 1 when provided"
    ):
        TokenWindowDataset(
            xp.arange(10, dtype=xp.int32),
            window_len=5,
            sampling="random",
            examples_per_epoch=examples_per_epoch,
        )


def test_token_window_dataset_rejects_examples_per_epoch_for_sequential_sampling():
    with pytest.raises(
        ValueError,
        match="examples_per_epoch is only valid for sampling='random'",
    ):
        TokenWindowDataset(
            xp.arange(10, dtype=xp.int32),
            window_len=5,
            sampling="sequential",
            examples_per_epoch=3,
        )


def test_token_window_dataset_random_offsets_do_not_use_backend_scalar_rng(
    monkeypatch,
):
    dataset = TokenWindowDataset(
        xp.arange(20, dtype=xp.int32),
        window_len=5,
        sampling="random",
        examples_per_epoch=4,
        offset_buffer_size=2,
    )
    monkeypatch.setattr(
        "autograd.data.dataset.xp",
        SimpleNamespace(
            random=SimpleNamespace(
                randint=lambda *args, **kwargs: (_ for _ in ()).throw(
                    AssertionError(
                        "backend random offsets would require scalar readback"
                    )
                ),
            ),
        ),
    )

    offsets = [example.offset for example in dataset]

    assert len(offsets) == 4
    assert all(0 <= offset < dataset.valid_window_count for offset in offsets)
