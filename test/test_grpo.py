from typing import cast

import numpy as np
import pytest
import torch

from autograd.backend import xp
from autograd.data.collator import Collator
from autograd.data.sft import (
    SFT_ROLE_MARKERS,
    SFT_SYSTEM_PROMPT,
    SFT_TURN_SEPARATOR,
)
from autograd.nn import Module
from autograd.tensor import Tensor
from autograd.text.tokenizer import BytePairEncoder
from autograd.tools.config_schema import GenericTrainingConfig
from examples.grpo import (
    TOKENIZER_VOCAB_PATH,
    GRPOBatch,
    GRPOCollator,
    GRPOTrainer,
    GRPOTrainingConfig,
    MathEnvironment,
    RolloutGenerator,
    RolloutGroup,
    Sample,
    Task,
    filter_fitting_tasks,
    generation,
    grpo_loss,
    rollout_metrics,
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


def test_grpo_training_config_requires_max_steps_argument():
    with pytest.raises(TypeError, match="max_steps"):
        GRPOTrainingConfig(
            checkpoint_freq=1,
            model_kwargs={},
            optimizer_kwargs={},
            max_generation_tokens=32,
            temperature=1.0,
            top_k=None,
            num_generations=2,
        )


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
        np.array([[10, 11, 20], [10, 11, 30]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        xp.to_numpy(batch.labels),
        np.array([[11, 20, 21], [11, 30, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        xp.to_numpy(batch.generated_token_mask),
        np.array([[0, 1, 1], [0, 1, 0]], dtype=np.int32),
    )
    np.testing.assert_allclose(
        xp.to_numpy(batch.sampled_token_logprobs),
        np.array(
            [[0.0, -0.1, -0.2], [0.0, -0.3, 0.0]],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        xp.to_numpy(batch.advantages),
        np.array(
            [[0.0, 0.5, 0.5], [0.0, -1.0, 0.0]],
            dtype=np.float32,
        ),
    )
    assert not hasattr(batch, "loss_total_weight")
    assert float(xp.to_scalar(GRPOTrainer._loss_total_weight(None, batch))) == 3.0


def test_grpo_tokenizer_vocab_path_exists_and_matches_pretrained_tokenizer_size():
    tokenizer = BytePairEncoder(num_merges=49990, vocab_file_path=TOKENIZER_VOCAB_PATH)

    assert tokenizer.n_vocab == 50257
    assert tokenizer.encode("<|endoftext|>") == [256]


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


def test_math_environment_rewards_only_clean_turn_terminated_answers():
    environment = MathEnvironment()
    task = Task(task_id="math-1", raw_input="What is 1 + 1?", answer="2")

    correct = Sample(
        completion_tokens=xp.array([20], dtype=xp.int32),
        completion_text=(
            f"<think>1 + 1 = 2</think><answer>2</answer>{SFT_TURN_SEPARATOR}"
        ),
        sampled_token_logprobs=xp.array([-0.1], dtype=xp.float32),
    )
    wrong = Sample(
        completion_tokens=xp.array([21], dtype=xp.int32),
        completion_text=f"<think>1 + 1 = 3</think><answer>3</answer>{SFT_TURN_SEPARATOR}",
        sampled_token_logprobs=xp.array([-0.2], dtype=xp.float32),
    )
    tag_junk = Sample(
        completion_tokens=xp.array([22], dtype=xp.int32),
        completion_text=(
            f"<think>1 + 1 = 2</think><answer>2</answer>junk{SFT_TURN_SEPARATOR}"
        ),
        sampled_token_logprobs=xp.array([-0.3], dtype=xp.float32),
    )
    missing_turn_end = Sample(
        completion_tokens=xp.array([23], dtype=xp.int32),
        completion_text="<think>1 + 1 = 2</think><answer>2</answer>",
        sampled_token_logprobs=xp.array([-0.4], dtype=xp.float32),
    )

    assert environment._compute_reward(task, correct) == pytest.approx(1.4)
    assert correct.metadata["format_valid"] is True
    assert correct.metadata["exact_match"] is True
    assert correct.metadata["parsed_answer"] == "2"
    assert environment._compute_reward(task, wrong) == pytest.approx(0.4)
    assert wrong.metadata["format_valid"] is True
    assert wrong.metadata["exact_match"] is False
    assert environment._compute_reward(task, tag_junk) == 0.0
    assert tag_junk.metadata["format_valid"] is False
    assert tag_junk.metadata["exact_match"] is False
    assert environment._compute_reward(task, missing_turn_end) == 0.0
    assert missing_turn_end.metadata["format_valid"] is False
    assert missing_turn_end.metadata["exact_match"] is False


def test_math_environment_normalizes_comma_separated_answers():
    environment = MathEnvironment()
    task = Task(task_id="math-1", raw_input="How much profit?", answer="22500")
    sample = Sample(
        completion_tokens=xp.array([20], dtype=xp.int32),
        completion_text=f"<think>math</think><answer>22,500</answer>{SFT_TURN_SEPARATOR}",
        sampled_token_logprobs=xp.array([-0.1], dtype=xp.float32),
    )

    assert environment._compute_reward(task, sample) == pytest.approx(1.4)


def test_extract_gsm8k_final_answer_from_hash_marker():
    assert MathEnvironment.extract_gsm8k_final_answer("scratch work #### 22,500") == (
        "22500"
    )


def test_gsm8k_row_to_task_uses_question_and_final_answer():
    task = MathEnvironment.gsm8k_row_to_task(
        3,
        {
            "question": "What is 1 + 1?",
            "answer": "1 + 1 = <<1+1=2>>2 #### 2",
        },
    )

    assert task.task_id == "gsm8k-3"
    assert task.raw_input == "What is 1 + 1?"
    assert task.answer == "2"
    assert task.metadata == {"source": "openai/gsm8k"}


def test_load_gsm8k_tasks_uses_shared_row_loader(monkeypatch):
    calls = []

    def fake_load_gsm8k_rows(split, max_rows):
        calls.append((split, max_rows))
        return [
            {
                "question": "What is 1 + 1?",
                "answer": "1 + 1 = <<1+1=2>>2 #### 2",
            }
        ]

    monkeypatch.setattr("examples.grpo.load_gsm8k_rows", fake_load_gsm8k_rows)

    tasks = MathEnvironment.load_gsm8k_tasks(split="train", max_tasks=4)

    assert calls == [("train", 4)]
    assert tasks == [
        Task(
            task_id="gsm8k-0",
            raw_input="What is 1 + 1?",
            answer="2",
            metadata={"source": "openai/gsm8k"},
        )
    ]


def test_math_environment_renders_tasks_with_sft_prompt_contract():
    prompt = MathEnvironment().render_task(
        Task(task_id="math-1", raw_input="What is 1 + 1?", answer="2")
    )

    assert prompt == (
        f"{SFT_ROLE_MARKERS['system']}{SFT_SYSTEM_PROMPT}"
        f"{SFT_TURN_SEPARATOR}{SFT_ROLE_MARKERS['user']}What is 1 + 1?"
        f"{SFT_TURN_SEPARATOR}{SFT_ROLE_MARKERS['assistant']}"
    )


def test_filter_fitting_tasks_applies_prompt_plus_generation_budget():
    class FakeTokenizer:
        def encode(self, text):
            if "short" in text:
                return [1, 2]
            return [1, 2, 3, 4]

    tasks = [
        Task(task_id="short", raw_input="short", answer="1"),
        Task(task_id="long", raw_input="long", answer="2"),
    ]

    fitting_tasks = filter_fitting_tasks(
        tasks,
        environment=MathEnvironment(),
        tokenizer=cast(BytePairEncoder, FakeTokenizer()),
        max_tokens=5,
        max_generation_tokens=2,
    )

    assert fitting_tasks == [tasks[0]]


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


def test_rollout_metrics_reports_reward_quality_and_termination():
    samples = [
        Sample(
            completion_tokens=xp.array([20, 21], dtype=xp.int32),
            completion_text="",
            sampled_token_logprobs=xp.array([-0.1, -0.2], dtype=xp.float32),
            reward=1.4,
            metadata={
                "format_valid": True,
                "exact_match": True,
                "stop_reason": "eos",
            },
        ),
        Sample(
            completion_tokens=xp.array([30], dtype=xp.int32),
            completion_text="",
            sampled_token_logprobs=xp.array([-0.3], dtype=xp.float32),
            reward=0.0,
            metadata={
                "format_valid": False,
                "exact_match": False,
                "stop_reason": "max_new_tokens",
            },
        ),
    ]

    metrics = rollout_metrics(samples)

    assert metrics["reward_mean"] == pytest.approx(0.7)
    assert metrics["reward_max"] == pytest.approx(1.4)
    assert metrics["format_valid_rate"] == pytest.approx(0.5)
    assert metrics["exact_match_rate"] == pytest.approx(0.5)
    assert metrics["turn_end_rate"] == pytest.approx(0.5)
    assert metrics["completion_len_mean"] == pytest.approx(1.5)


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


def test_rollout_generator_batches_generation_forward_passes(monkeypatch):
    class FakeModel:
        max_seq_len = 8

        def __init__(self):
            self._is_training = True
            self.batch_shapes = []
            self.eval_called = False
            self.train_called = False

        def eval(self):
            self.eval_called = True
            self._is_training = False

        def train(self):
            self.train_called = True
            self._is_training = True

        def __call__(self, input_ids):
            self.batch_shapes.append(tuple(input_ids.shape))
            batch_size, seq_len = input_ids.shape
            return Tensor(xp.zeros((batch_size, seq_len, 5), dtype=xp.float32))

    class FakeTokenizer:
        def encode(self, text):
            if text == "<|endoftext|>":
                return [0]
            return [3, 4]

        def decode(self, tokens):
            return " ".join(str(token) for token in tokens)

    monkeypatch.setattr(
        xp,
        "sample_categorical",
        lambda logits: xp.array(1, dtype=xp.int32),
    )
    model = FakeModel()

    rollout_group = RolloutGenerator(
        _grpo_training_config(num_generations=4, max_generation_tokens=2)
    ).rollout(
        model=cast(Module, model),
        task=Task(task_id="math-1", raw_input="What is 1 + 1?", answer="2"),
        tokenizer=cast(BytePairEncoder, FakeTokenizer()),
        environment=MathEnvironment(),
    )

    assert model.batch_shapes == [(4, 2), (4, 3)]
    assert model.eval_called
    assert model.train_called
    assert len(rollout_group.samples) == 4
    for sample in rollout_group.samples:
        np.testing.assert_array_equal(xp.to_numpy(sample.completion_tokens), [1, 1])


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
            num_generations=1,
            max_generation_tokens=1,
            temperature=1.0,
            top_k=None,
        )


def test_generation_defaults_to_sft_turn_separator_stop_token(monkeypatch):
    class FakeModel:
        max_seq_len = 4

        def eval(self):
            pass

        def __call__(self, input_ids):
            batch_size, seq_len = input_ids.shape
            return Tensor(xp.zeros((batch_size, seq_len, 10), dtype=xp.float32))

    class FakeTokenizer:
        def __init__(self):
            self.encoded_tokens = []

        def encode(self, token):
            self.encoded_tokens.append(token)
            if token == SFT_TURN_SEPARATOR:
                return [7]
            return [9]

        def decode(self, tokens):
            return "".join(
                SFT_TURN_SEPARATOR if int(token) == 7 else str(int(token))
                for token in tokens
            )

    monkeypatch.setattr(
        xp,
        "sample_categorical",
        lambda logits: xp.array(7, dtype=xp.int32),
    )
    tokenizer = FakeTokenizer()

    samples = generation(
        FakeModel(),
        xp.array([1], dtype=xp.int32),
        tokenizer,
        num_generations=1,
        max_generation_tokens=3,
        temperature=1.0,
        top_k=None,
    )

    assert tokenizer.encoded_tokens == [SFT_TURN_SEPARATOR]
    assert samples[0].completion_text == SFT_TURN_SEPARATOR
    assert samples[0].metadata["stop_reason"] == "eos"


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


def test_grpo_trainer_train_step_runs_one_optimizer_step():
    class FakeModel:
        def __init__(self):
            self.train_called = False

        def train(self):
            self.train_called = True

    class FakeOptimizer:
        def __init__(self):
            self.zero_grad_called = False

        def zero_grad(self):
            self.zero_grad_called = True

    class FakeTrainer(GRPOTrainer):
        def __init__(self):
            self.model = FakeModel()
            self.optimizer = FakeOptimizer()
            self.global_step = 0
            self.loss = Tensor(xp.array(2.0, dtype=xp.float32), requires_grad=True)
            self.total_weight = xp.array(4.0, dtype=xp.float32)
            self.optimizer_step_called = False

        def _forward_and_loss(self, batch):
            return self.loss

        def _loss_total_weight(self, batch):
            return self.total_weight

        def optimizer_step(self, state, *, record_grad_norm=True):
            self.optimizer_step_called = True
            assert state.accumulated_batches == 1
            np.testing.assert_allclose(
                xp.to_numpy(state.accumulated_loss_total_weight),
                xp.to_numpy(self.total_weight),
            )
            return True

        def _evaluate(self, val_data_loader):
            pass

    trainer = FakeTrainer()
    batch = GRPOBatch(
        input_ids=xp.array([[1]], dtype=xp.int32),
        labels=xp.array([[1]], dtype=xp.int32),
        sampled_token_logprobs=xp.array([[0.0]], dtype=xp.float32),
        generated_token_mask=xp.array([[1]], dtype=xp.int32),
        advantages=xp.array([[1.0]], dtype=xp.float32),
    )

    loss = trainer.train_step(batch)

    assert loss is trainer.loss
    assert trainer.model.train_called
    assert trainer.optimizer.zero_grad_called
    assert trainer.optimizer_step_called
    assert trainer.global_step == 1
