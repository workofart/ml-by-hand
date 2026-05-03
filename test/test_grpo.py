import numpy as np
import pytest
import torch

from autograd.backend import xp
from autograd.data.collator import Collator
from autograd.tensor import Tensor
from autograd.tools.config_schema import GenericTrainingConfig
from examples.grpo import (
    GRPOBatch,
    GRPOCollator,
    GRPOTrainer,
    GRPOTrainingConfig,
    MathEnvironment,
    RolloutGenerator,
    RolloutGroup,
    Sample,
    Task,
    generation,
    grpo_loss,
)


def _grpo_training_config(**kwargs):
    config_kwargs = {
        "max_steps": 1,
        "checkpoint_freq": 1,
        "model_kwargs": {},
        "optimizer_kwargs": {},
        "max_generation_tokens": 32,
        "temperature": 1.0,
        "top_k": None,
        "num_generations": 2,
    }
    config_kwargs.update(kwargs)
    return GRPOTrainingConfig(**config_kwargs)


def test_grpo_training_config_extends_generic_training_config_for_rollouts():
    config = _grpo_training_config(
        max_generation_tokens=7,
        temperature=1.0,
        top_k=None,
        num_generations=3,
    )

    assert isinstance(config, GenericTrainingConfig)
    assert config.max_generation_tokens == 7
    assert config.temperature == 1.0
    assert config.top_k is None
    assert config.num_generations == 3


def test_grpo_training_config_rejects_sampling_that_breaks_logprob_contract():
    with pytest.raises(ValueError, match="temperature=1.0 and top_k=None"):
        _grpo_training_config(temperature=0.8)


def test_grpo_collator_implements_collator_interface():
    assert isinstance(GRPOCollator(max_tokens=4, pad_idx=0), Collator)


def test_grpo_collator_aligns_generated_tokens_with_shifted_completion_labels():
    prompt_tokens = xp.array([10, 11], dtype=xp.int32)
    group = RolloutGroup(
        prompt_id="prompt-1",
        prompt_tokens=prompt_tokens,
        samples=[
            Sample(
                completion_tokens=xp.array([20, 21], dtype=xp.int32),
                completion_text="",
                sampled_token_logprobs=xp.array([-0.1, -0.2], dtype=xp.float32),
                reward=1.0,
                advantage=0.5,
                metadata=None,
            ),
            Sample(
                completion_tokens=xp.array([30], dtype=xp.int32),
                completion_text="",
                sampled_token_logprobs=xp.array([-0.3], dtype=xp.float32),
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
        xp.to_numpy(batch.generated_token_mask),
        np.array([[0, 1, 1, 0], [0, 1, 0, 0]], dtype=np.int32),
    )
    np.testing.assert_allclose(
        xp.to_numpy(batch.sampled_token_logprobs),
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
    assert not hasattr(batch, "loss_total_weight")
    assert float(xp.to_scalar(GRPOTrainer._loss_total_weight(None, batch))) == 3.0


def test_sample_requires_completion_logprob_alignment():
    with pytest.raises(
        ValueError, match="completion_tokens and sampled_token_logprobs"
    ):
        Sample(
            completion_tokens=xp.array([20, 21], dtype=xp.int32),
            completion_text="",
            sampled_token_logprobs=xp.array([-0.1], dtype=xp.float32),
            reward=1.0,
            advantage=0.5,
            metadata=None,
        )


def test_rollout_group_requires_samples():
    with pytest.raises(ValueError, match="at least one sample"):
        RolloutGroup(
            prompt_id="prompt-1",
            prompt_tokens=xp.array([10, 11], dtype=xp.int32),
            samples=[],
        )


def test_rollout_group_requires_prompt_tokens():
    with pytest.raises(ValueError, match="prompt_tokens"):
        RolloutGroup(
            prompt_id="prompt-1",
            prompt_tokens=xp.array([], dtype=xp.int32),
            samples=[
                Sample(
                    completion_tokens=xp.array([20], dtype=xp.int32),
                    completion_text="",
                    sampled_token_logprobs=xp.array([-0.1], dtype=xp.float32),
                    reward=1.0,
                    advantage=0.5,
                    metadata=None,
                ),
            ],
        )


def test_grpo_collator_rejects_rows_longer_than_max_tokens():
    group = RolloutGroup(
        prompt_id="prompt-1",
        prompt_tokens=xp.array([10, 11], dtype=xp.int32),
        samples=[
            Sample(
                completion_tokens=xp.array([20, 21], dtype=xp.int32),
                completion_text="",
                sampled_token_logprobs=xp.array([-0.1, -0.2], dtype=xp.float32),
                reward=1.0,
                advantage=0.5,
                metadata=None,
            ),
        ],
    )

    with pytest.raises(ValueError, match="exceeds max_tokens"):
        GRPOCollator(max_tokens=3, pad_idx=0)([group])


def test_math_environment_parses_answer_tags_for_reward():
    environment = MathEnvironment()
    task = Task(task_id="math-1", raw_input="What is 1 + 1?", answer="2")

    correct = Sample(
        completion_tokens=xp.array([20], dtype=xp.int32),
        completion_text="<think>1 + 1 = 2</think><answer>2</answer>",
        sampled_token_logprobs=xp.array([-0.1], dtype=xp.float32),
    )
    wrong = Sample(
        completion_tokens=xp.array([21], dtype=xp.int32),
        completion_text="<answer>3</answer>",
        sampled_token_logprobs=xp.array([-0.2], dtype=xp.float32),
    )
    unformatted = Sample(
        completion_tokens=xp.array([22], dtype=xp.int32),
        completion_text="2",
        sampled_token_logprobs=xp.array([-0.3], dtype=xp.float32),
    )

    assert environment._compute_reward(task, correct) == 1.1
    assert environment._compute_reward(task, wrong) == 0.1
    assert environment._compute_reward(task, unformatted) == 0.0


def test_score_group_attaches_rewards_without_advantages():
    environment = MathEnvironment()
    task = Task(task_id="math-1", raw_input="What is 1 + 1?", answer="2")
    group = RolloutGroup(
        prompt_id="math-1",
        prompt_tokens=xp.array([10], dtype=xp.int32),
        samples=[
            Sample(
                completion_tokens=xp.array([20], dtype=xp.int32),
                completion_text="not tagged",
                sampled_token_logprobs=xp.array([-0.1], dtype=xp.float32),
            ),
            Sample(
                completion_tokens=xp.array([21], dtype=xp.int32),
                completion_text="also not tagged",
                sampled_token_logprobs=xp.array([-0.2], dtype=xp.float32),
            ),
        ],
    )

    scored_group = environment.score_group(task, group)

    assert [sample.reward for sample in scored_group.samples] == [0.0, 0.0]
    assert [sample.advantage for sample in scored_group.samples] == [None, None]


def test_rollout_generator_uses_zero_advantage_when_rewards_have_no_variance():
    group = RolloutGroup(
        prompt_id="math-1",
        prompt_tokens=xp.array([10], dtype=xp.int32),
        samples=[
            Sample(
                completion_tokens=xp.array([20], dtype=xp.int32),
                completion_text="",
                sampled_token_logprobs=xp.array([-0.1], dtype=xp.float32),
                reward=0.0,
            ),
            Sample(
                completion_tokens=xp.array([21], dtype=xp.int32),
                completion_text="",
                sampled_token_logprobs=xp.array([-0.2], dtype=xp.float32),
                reward=0.0,
            ),
        ],
    )

    RolloutGenerator(_grpo_training_config())._compute_advantages(group)

    assert [sample.advantage for sample in group.samples] == [0.0, 0.0]


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


def test_grpo_loss_matches_pytorch_reference():
    logits_data = np.array(
        [[[1.2, -0.4, 0.3], [0.1, 1.4, -0.7], [0.8, -0.2, 0.5]]],
        dtype=np.float32,
    )
    labels = np.array([[0, 1, 2]], dtype=np.int32)
    sampled_token_logprobs = np.array([[-100.0, 100.0, 0.0]], dtype=np.float32)
    generated_token_mask = np.array([[1, 1, 0]], dtype=np.int32)
    advantages = np.array([[0.7, -0.4, 0.0]], dtype=np.float32)

    logits = Tensor(xp.array(logits_data), requires_grad=True)
    batch = GRPOBatch(
        input_ids=xp.array([[4, 5, 6]], dtype=xp.int32),
        labels=xp.array(labels, dtype=xp.int32),
        sampled_token_logprobs=xp.array(sampled_token_logprobs, dtype=xp.float32),
        generated_token_mask=xp.array(generated_token_mask, dtype=xp.int32),
        advantages=xp.array(advantages, dtype=xp.float32),
    )

    loss = grpo_loss(logits, batch)

    torch_logits = torch.tensor(logits_data, requires_grad=True)
    torch_labels = torch.tensor(labels, dtype=torch.int64)
    torch_generated_token_mask = torch.tensor(generated_token_mask, dtype=torch.float32)
    torch_advantages = torch.tensor(advantages)
    torch_logprobs = torch.log_softmax(torch_logits, dim=-1)
    torch_sampled_token_logprobs = torch_logprobs.gather(
        dim=-1,
        index=torch_labels.unsqueeze(-1),
    ).squeeze(-1)
    torch_loss = -(
        torch_sampled_token_logprobs * torch_advantages * torch_generated_token_mask
    ).sum()

    np.testing.assert_allclose(
        xp.to_numpy(loss.data),
        torch_loss.detach().numpy(),
        atol=1e-6,
    )

    loss.backward()
    torch_loss.backward()
    np.testing.assert_allclose(
        xp.to_numpy(logits.grad.data),
        torch_logits.grad.detach().numpy(),
        atol=1e-6,
    )
