import unittest
from unittest.mock import patch

import numpy as np
import pytest

from autograd.backend import xp
from autograd.data.collator import (
    CausalLMCollator,
    CausalLMWindowCollator,
    OneHotCollator,
    PairedCollator,
    Seq2SeqCollator,
    build_causal_lm_inputs_and_labels,
    greedy_pack_aligned_examples,
    pack_tokens,
    pad_aligned_right,
    truncate_aligned_left,
)
from autograd.data.types import CausalLMBatch, Seq2SeqBatch, TokenWindowExample
from autograd.functional import IGNORE_INDEX


def mock_padding_mask(X_chunk, pad_idx):
    batch_size, seq_len = X_chunk.shape
    return xp.zeros((batch_size, 1, 1, seq_len))


def make_causal_lm_collator(max_tokens: int, pad_idx: int) -> CausalLMCollator:
    return CausalLMCollator(
        max_tokens=max_tokens,
        pad_idx=pad_idx,
        truncator=truncate_aligned_left,
        padder=pad_aligned_right,
        label_builder=build_causal_lm_inputs_and_labels,
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

    def test_pack_tokens_left_truncates_then_pads(self):
        packed = pack_tokens(
            tokens=xp.array([10, 11, 12, 13, 20, 21, 22], dtype=xp.int32),
            max_tokens=5,
            pad_idx=0,
        )

        self.assertTrue(
            xp.array_equal(packed, xp.array([12, 13, 20, 21, 22], dtype=xp.int32))
        )

    def test_truncate_aligned_left_trims_values_together(self):
        tokens, loss_mask = truncate_aligned_left(
            (
                xp.array([10, 11, 12, 13], dtype=xp.int32),
                xp.array([0, 0, 1, 1], dtype=xp.int32),
            ),
            max_tokens=3,
        )

        self.assertTrue(xp.array_equal(tokens, xp.array([11, 12, 13], dtype=xp.int32)))
        self.assertTrue(xp.array_equal(loss_mask, xp.array([0, 1, 1], dtype=xp.int32)))

    def test_pad_aligned_right_uses_field_specific_pad_values(self):
        tokens, loss_mask = pad_aligned_right(
            (
                xp.array([10, 11], dtype=xp.int32),
                xp.array([0, 1], dtype=xp.int32),
            ),
            max_tokens=4,
            pad_values=(257, 0),
        )

        self.assertTrue(
            xp.array_equal(tokens, xp.array([10, 11, 257, 257], dtype=xp.int32))
        )
        self.assertTrue(
            xp.array_equal(loss_mask, xp.array([0, 1, 0, 0], dtype=xp.int32))
        )

    def test_greedy_pack_aligned_examples_preserves_selected_fields(self):
        packed = greedy_pack_aligned_examples(
            [
                {
                    "tokens": xp.array([10, 11, 12], dtype=xp.int32),
                    "loss_mask": xp.array([0, 1, 1], dtype=xp.int32),
                    "ignored": xp.array([100], dtype=xp.int32),
                },
                {
                    "tokens": xp.array([20, 21, 22, 23], dtype=xp.int32),
                    "loss_mask": xp.array([0, 0, 1, 1], dtype=xp.int32),
                    "ignored": xp.array([200], dtype=xp.int32),
                },
                {
                    "tokens": xp.array([30, 31], dtype=xp.int32),
                    "loss_mask": xp.array([0, 1], dtype=xp.int32),
                    "ignored": xp.array([300], dtype=xp.int32),
                },
            ],
            fields=("tokens", "loss_mask"),
            max_tokens=7,
        )

        self.assertEqual(len(packed), 2)
        self.assertTrue(
            xp.array_equal(
                packed[0]["tokens"],
                xp.array([10, 11, 12, 20, 21, 22, 23], dtype=xp.int32),
            )
        )
        self.assertTrue(
            xp.array_equal(
                packed[0]["loss_mask"],
                xp.array([0, 1, 1, 0, 0, 1, 1], dtype=xp.int32),
            )
        )
        self.assertTrue(
            xp.array_equal(packed[1]["tokens"], xp.array([30, 31], dtype=xp.int32))
        )
        self.assertNotIn("ignored", packed[0])

    def test_build_causal_lm_inputs_and_labels_masks_shifted_targets(self):
        input_ids, labels = build_causal_lm_inputs_and_labels(
            xp.array([10, 20, 30, 40], dtype=xp.int32),
            xp.array([0, 0, 1, 1], dtype=xp.int32),
        )

        self.assertTrue(
            xp.array_equal(input_ids, xp.array([10, 20, 30], dtype=xp.int32))
        )
        self.assertTrue(
            xp.array_equal(
                labels,
                xp.array([IGNORE_INDEX, 30, 40], dtype=xp.int32),
            )
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
        "autograd.data.collator.text_utils.create_padding_mask",
        side_effect=mock_padding_mask,
    )
    def test_encoder_decoder_collator_uses_seq2seq_examples(self, mock_padding):
        collator = Seq2SeqCollator(
            max_tokens=4,
            pad_idx=0,
            sos_idx=1,
            packer=pack_tokens,
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
