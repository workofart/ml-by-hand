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
    pack_tokens,
)
from autograd.data.types import CausalLMBatch, Seq2SeqBatch, TokenWindowExample
from autograd.data.utils import (
    openai_chat_to_prompt_completion,
    tokenize_prompt_completion,
)
from autograd.functional import IGNORE_INDEX


def mock_padding_mask(X_chunk, pad_idx):
    batch_size, seq_len = X_chunk.shape
    return xp.zeros((batch_size, 1, 1, seq_len))


class MockBPE:
    def encode(self, token, allowed_special=set()):
        if token in allowed_special:
            if token == "<PAD>":
                return [0]
            if token == "<SOS>":
                return [1]
        return [ord(char) for char in token]


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
        collator = Seq2SeqCollator(max_tokens=4, pad_idx=0, sos_idx=1)

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


def test_causal_lm_collator_builds_masked_sft_batch():
    class DummyBPE:
        def encode(self, text, allowed_special=None):
            return [ord(char) for char in text]

    tokenized_example = tokenize_prompt_completion(
        openai_chat_to_prompt_completion(
            {
                "messages": [
                    {"role": "user", "content": "ABC"},
                    {"role": "assistant", "content": "DE"},
                ]
            }
        ),
        DummyBPE(),
    )
    batch = CausalLMCollator(max_tokens=6, pad_idx=0)(
        [
            {
                "tokens": tokenized_example["tokens"],
                "loss_mask": tokenized_example["loss_mask"],
            }
        ]
    )

    assert isinstance(batch, CausalLMBatch)
    np.testing.assert_array_equal(
        xp.to_numpy(batch.input_ids[0]),
        np.array([65, 66, 67, 68, 69], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        xp.to_numpy(batch.labels[0]),
        np.array([IGNORE_INDEX, IGNORE_INDEX, 68, 69, IGNORE_INDEX], dtype=np.int32),
    )
    assert not hasattr(batch, "loss_mask")


def test_causal_lm_collator_requires_max_tokens_at_least_two():
    with pytest.raises(ValueError, match="max_tokens must be >= 2 for causal LM"):
        CausalLMCollator(max_tokens=1, pad_idx=0)


def test_causal_lm_collator_rejects_empty_batch():
    collator = CausalLMCollator(max_tokens=4, pad_idx=0)

    with pytest.raises(ValueError, match="examples must not be empty"):
        collator([])


def test_causal_lm_batch_rejects_loss_mask():
    with pytest.raises(TypeError):
        CausalLMBatch(
            input_ids=xp.zeros((1, 2), dtype=xp.int32),
            labels=xp.zeros((1, 2), dtype=xp.int32),
            loss_mask=xp.ones((1, 2), dtype=xp.float32),
        )
