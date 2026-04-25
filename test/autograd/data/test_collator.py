import unittest
from unittest.mock import patch

import numpy as np
import pytest

from autograd.backend import xp
from autograd.data.collator import (
    BatchMaxLengthCausalLMCollator,
    CausalLMWindowCollator,
    FixedLengthCausalLMCollator,
    OneHotCollator,
    PairedCollator,
    Seq2SeqCollator,
    create_padding_mask,
    truncate_and_pad_tokens,
)
from autograd.data.types import CausalLMBatch, Seq2SeqBatch, TokenWindowExample
from autograd.functional import IGNORE_INDEX


def mock_padding_mask(X_chunk, pad_idx, dims=None):
    if dims is not None:
        return xp.zeros(dims)
    if len(X_chunk.shape) == 1:
        return xp.zeros((1, 1, X_chunk.shape[0]))
    batch_size, seq_len = X_chunk.shape
    return xp.zeros((batch_size, 1, 1, seq_len))


def make_causal_lm_collator(
    max_tokens: int, pad_idx: int
) -> FixedLengthCausalLMCollator:
    return FixedLengthCausalLMCollator(
        max_tokens=max_tokens,
        pad_idx=pad_idx,
    )


class MockBPE:
    def encode(self, token, allowed_special=set()):
        if token in allowed_special:
            if token == "<PAD>":
                return [0]
            if token == "<SOS>":
                return [1]
            if token == "<|endoftext|>":
                return [2]
        special_token = "<|endoftext|>"
        if token == special_token:
            return [2]

        encoded = []
        start = 0
        while True:
            special_index = token.find(special_token, start)
            if special_index == -1:
                encoded.extend(ord(char) for char in token[start:])
                break
            encoded.extend(ord(char) for char in token[start:special_index])
            encoded.append(2)
            start = special_index + len(special_token)
        return encoded


class TestCollator(unittest.TestCase):
    def setUp(self):
        self.bpe = MockBPE()

    def test_create_padding_mask_default_dims(self):
        token_indices = xp.array(
            [
                [1, 2, 0, 0],
                [3, 4, 5, 0],
            ],
            dtype=xp.int32,
        )

        mask = create_padding_mask(token_indices, pad_idx=0, dims=None)

        self.assertEqual(mask.shape, (2, 1, 1, 4))
        assert xp.array_equal(mask[0, 0, 0], xp.array([0, 0, 1, 1]))
        assert xp.array_equal(mask[1, 0, 0], xp.array([0, 0, 0, 1]))

    def test_create_padding_mask_custom_dims(self):
        token_indices = xp.array([[1, 0, 0], [2, 2, 0]], dtype=xp.int32)

        mask = create_padding_mask(token_indices, pad_idx=0, dims=(2, 1, 3))

        self.assertEqual(mask.shape, (2, 1, 3))
        assert xp.array_equal(mask[0, 0], xp.array([0, 1, 1]))
        assert xp.array_equal(mask[1, 0], xp.array([0, 0, 1]))

    def test_truncate_and_pad_tokens_left_truncates_then_right_pads(self):
        tokens = truncate_and_pad_tokens(
            tokens=xp.array([10, 11, 12, 13, 20, 21, 22], dtype=xp.int32),
            max_tokens=5,
            pad_idx=0,
        )

        self.assertTrue(
            xp.array_equal(tokens, xp.array([12, 13, 20, 21, 22], dtype=xp.int32))
        )

    def test_causal_lm_collator_requires_aligned_tokens_and_loss_mask(self):
        collator = make_causal_lm_collator(max_tokens=5, pad_idx=0)

        with self.assertRaisesRegex(
            ValueError, "tokens and loss_mask must have the same length"
        ):
            collator(
                [
                    {
                        "tokens": xp.array([10, 11, 12], dtype=xp.int32),
                        "loss_mask": xp.array([1, 1], dtype=xp.int32),
                    }
                ]
            )

    def test_paired_collator_batches_examples(self):
        collator = PairedCollator()

        batch_X, batch_y = collator(
            [
                {
                    "inputs": xp.array([1, 2], dtype=xp.int32),
                    "targets": xp.array(3, dtype=xp.int32),
                },
                {
                    "inputs": xp.array([4, 5], dtype=xp.int32),
                    "targets": xp.array(6, dtype=xp.int32),
                },
            ]
        )

        self.assertTrue(
            xp.array_equal(
                batch_X,
                xp.array([[1, 2], [4, 5]], dtype=xp.int32),
            )
        )
        self.assertTrue(xp.array_equal(batch_y, xp.array([3, 6], dtype=xp.int32)))

    def test_paired_collator_accepts_numpy_scalar_targets(self):
        collator = PairedCollator()

        _, batch_y = collator(
            [
                {
                    "inputs": xp.array([1, 2], dtype=xp.float32),
                    "targets": np.int64(3),
                },
                {
                    "inputs": xp.array([4, 5], dtype=xp.float32),
                    "targets": np.int64(6),
                },
            ]
        )

        self.assertTrue(xp.array_equal(batch_y, xp.array([3, 6], dtype=xp.int64)))

    def test_one_hot_collator_materializes_one_hot_inputs(self):
        collator = OneHotCollator(num_classes=4)

        batch_X, batch_y = collator(
            [
                {
                    "inputs": xp.array([0, 2], dtype=xp.int32),
                    "targets": xp.array(1, dtype=xp.int32),
                },
                {
                    "inputs": xp.array([1, 3], dtype=xp.int32),
                    "targets": xp.array(0, dtype=xp.int32),
                },
            ]
        )

        expected_X = xp.array(
            [
                [[1, 0, 0, 0], [0, 0, 1, 0]],
                [[0, 1, 0, 0], [0, 0, 0, 1]],
            ],
            dtype=xp.float32,
        )
        self.assertTrue(xp.array_equal(batch_X, expected_X))
        self.assertTrue(xp.array_equal(batch_y, xp.array([1, 0], dtype=xp.int32)))

    @patch(
        "autograd.data.collator.create_padding_mask",
        side_effect=mock_padding_mask,
    )
    def test_encoder_decoder_collator_uses_seq2seq_examples(self, mock_padding):
        collator = Seq2SeqCollator(
            max_tokens=4,
            pad_idx=0,
            sos_idx=1,
        )

        batch = collator(
            [
                {
                    "input_ids": xp.array([10, 11, 12], dtype=xp.int32),
                    "labels": xp.array([20, 21], dtype=xp.int32),
                }
            ]
        )

        self.assertIsInstance(batch, Seq2SeqBatch)
        self.assertTrue(
            xp.array_equal(
                batch.input_ids[0], xp.array([10, 11, 12, 0], dtype=xp.int32)
            )
        )
        self.assertTrue(
            xp.array_equal(
                batch.decoder_input_ids[0],
                xp.array([1, 20, 21, 0], dtype=xp.int32),
            )
        )
        self.assertTrue(
            xp.array_equal(
                batch.labels[0],
                xp.array([20, 21, IGNORE_INDEX, IGNORE_INDEX], dtype=xp.int32),
            )
        )


def test_window_collator_shapes():
    data = xp.arange(20, dtype=xp.int32)
    examples = [
        TokenWindowExample(data, offset=0, window_len=5),
        TokenWindowExample(data, offset=3, window_len=5),
    ]

    batch = CausalLMWindowCollator()(examples)

    assert batch.input_ids.shape == (2, 4)
    assert batch.labels.shape == (2, 4)
    assert not hasattr(batch, "loss_mask")


def test_window_collator_shift():
    data = xp.arange(10, dtype=xp.int32)
    example = TokenWindowExample(data, offset=2, window_len=5)

    batch = CausalLMWindowCollator()([example])

    np.testing.assert_array_equal(
        xp.to_numpy(batch.input_ids[0]),
        np.array([2, 3, 4, 5], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        xp.to_numpy(batch.labels[0]),
        np.array([3, 4, 5, 6], dtype=np.int32),
    )


def test_window_collator_requires_shiftable_window():
    example = TokenWindowExample(xp.arange(4, dtype=xp.int32), offset=0, window_len=1)

    with pytest.raises(ValueError, match="window_len must be >= 2"):
        CausalLMWindowCollator()([example])


def test_causal_lm_collator_builds_prompt_masked_batch():
    batch = make_causal_lm_collator(max_tokens=6, pad_idx=0)(
        [
            {
                "tokens": xp.array([65, 66, 67, 2, 68, 69, 2], dtype=xp.int32),
                "loss_mask": xp.array([0, 0, 0, 0, 1, 1, 1], dtype=xp.int32),
            }
        ]
    )

    assert isinstance(batch, CausalLMBatch)
    np.testing.assert_array_equal(
        xp.to_numpy(batch.input_ids[0]),
        np.array([66, 67, 2, 68, 69], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        xp.to_numpy(batch.labels[0]),
        np.array([IGNORE_INDEX, IGNORE_INDEX, 68, 69, 2], dtype=np.int32),
    )
    assert not hasattr(batch, "loss_mask")


def test_batch_max_length_causal_lm_collator_pads_to_longest_row():
    batch = BatchMaxLengthCausalLMCollator(
        max_tokens=8,
        pad_idx=0,
    )(
        [
            {
                "tokens": xp.array([10, 11, 12], dtype=xp.int32),
                "loss_mask": xp.array([0, 1, 1], dtype=xp.int32),
            },
            {
                "tokens": xp.array([20, 21, 22, 23, 24], dtype=xp.int32),
                "loss_mask": xp.array([0, 0, 1, 1, 1], dtype=xp.int32),
            },
        ]
    )

    assert batch.input_ids.shape == (2, 4)
    assert batch.labels.shape == (2, 4)
    np.testing.assert_array_equal(
        xp.to_numpy(batch.input_ids[0]),
        np.array([10, 11, 12, 0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        xp.to_numpy(batch.labels[0]),
        np.array([11, 12, IGNORE_INDEX, IGNORE_INDEX], dtype=np.int32),
    )


def test_causal_lm_collator_requires_max_tokens_at_least_two():
    with pytest.raises(ValueError, match="max_tokens must be >= 2 for causal LM"):
        make_causal_lm_collator(max_tokens=1, pad_idx=0)


def test_causal_lm_batch_rejects_loss_mask():
    with pytest.raises(TypeError):
        CausalLMBatch(
            input_ids=xp.zeros((1, 2), dtype=xp.int32),
            labels=xp.zeros((1, 2), dtype=xp.int32),
            loss_mask=xp.ones((1, 2), dtype=xp.float32),
        )
