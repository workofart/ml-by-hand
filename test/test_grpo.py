import numpy as np
import pytest

from autograd.backend import xp
from autograd.data.collator import Collator
from examples.grpo import GRPOCollator, PromptGroup, Sample, generation


def test_grpo_collator_implements_collator_interface():
    assert isinstance(GRPOCollator(max_tokens=4, pad_idx=0), Collator)


def test_grpo_collator_aligns_actions_with_shifted_completion_labels():
    prompt_tokens = xp.array([10, 11], dtype=xp.int32)
    group = PromptGroup(
        prompt_id="prompt-1",
        prompt_tokens=prompt_tokens,
        samples=[
            Sample(
                completion_tokens=xp.array([20, 21], dtype=xp.int32),
                old_logprobs=xp.array([-0.1, -0.2], dtype=xp.float32),
                reward=1.0,
                advantage=0.5,
                metadata=None,
            ),
            Sample(
                completion_tokens=xp.array([30], dtype=xp.int32),
                old_logprobs=xp.array([-0.3], dtype=xp.float32),
                reward=0.0,
                advantage=-1.0,
                metadata=None,
            ),
        ],
    )

    batch = GRPOCollator(max_tokens=5, pad_idx=0)([group])

    np.testing.assert_array_equal(
        xp.to_numpy(batch.input_ids),
        np.array([[10, 11, 20, 21], [10, 11, 30, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        xp.to_numpy(batch.labels),
        np.array([[11, 20, 21, 0], [11, 30, 0, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        xp.to_numpy(batch.action_mask),
        np.array([[0, 1, 1, 0], [0, 1, 0, 0]], dtype=np.int32),
    )
    np.testing.assert_allclose(
        xp.to_numpy(batch.old_logprobs),
        np.array(
            [[0.0, -0.1, -0.2, 0.0], [0.0, -0.3, 0.0, 0.0]],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        xp.to_numpy(batch.advantages),
        np.array(
            [[0.0, 0.5, 0.5, 0.0], [0.0, -1.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
    )
    assert float(xp.to_scalar(batch.loss_total_weight)) == 3.0


def test_sample_requires_completion_logprob_alignment():
    with pytest.raises(ValueError, match="completion_tokens and old_logprobs"):
        Sample(
            completion_tokens=xp.array([20, 21], dtype=xp.int32),
            old_logprobs=xp.array([-0.1], dtype=xp.float32),
            reward=1.0,
            advantage=0.5,
            metadata=None,
        )


def test_prompt_group_requires_samples():
    with pytest.raises(ValueError, match="at least one sample"):
        PromptGroup(
            prompt_id="prompt-1",
            prompt_tokens=xp.array([10, 11], dtype=xp.int32),
            samples=[],
        )


def test_prompt_group_requires_prompt_tokens():
    with pytest.raises(ValueError, match="prompt_tokens"):
        PromptGroup(
            prompt_id="prompt-1",
            prompt_tokens=xp.array([], dtype=xp.int32),
            samples=[
                Sample(
                    completion_tokens=xp.array([20], dtype=xp.int32),
                    old_logprobs=xp.array([-0.1], dtype=xp.float32),
                    reward=1.0,
                    advantage=0.5,
                    metadata=None,
                ),
            ],
        )


def test_grpo_collator_rejects_rows_longer_than_max_tokens():
    group = PromptGroup(
        prompt_id="prompt-1",
        prompt_tokens=xp.array([10, 11], dtype=xp.int32),
        samples=[
            Sample(
                completion_tokens=xp.array([20, 21], dtype=xp.int32),
                old_logprobs=xp.array([-0.1, -0.2], dtype=xp.float32),
                reward=1.0,
                advantage=0.5,
                metadata=None,
            ),
        ],
    )

    with pytest.raises(ValueError, match="exceeds max_tokens"):
        GRPOCollator(max_tokens=3, pad_idx=0)([group])


def test_generation_rejects_prompt_that_fills_context_window():
    class FakeModel:
        max_seq_len = 2

    class FakeTokenizer:
        def encode(self, token):
            return [9]

    with pytest.raises(ValueError, match="prompt length"):
        generation(
            FakeModel(),
            xp.array([1, 2], dtype=xp.int32),
            FakeTokenizer(),
            max_generation_tokens=1,
            temperature=1.0,
            top_k=None,
        )
