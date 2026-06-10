# ruff: noqa: E402

import re
import sys
from abc import abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autograd import functional
from autograd.backend import Array, ArrayLike, materialize, xp
from autograd.data.collator import Collator, pad_right_1d
from autograd.data.gsm8k import load_gsm8k_rows, split_gsm8k_answer
from autograd.data.sft import (
    SFT_ROLE_MARKERS,
    SFT_SYSTEM_PROMPT,
    SFT_TURN_SEPARATOR,
)
from autograd.nn import Module
from autograd.optim import Adam
from autograd.tensor import Tensor, no_grad
from autograd.text.tokenizer import BytePairEncoder
from autograd.text.utils import generate
from autograd.tools.config_schema import GenericTrainingConfig
from autograd.tools.model import load_checkpoint_metadata, save_checkpoint
from autograd.tools.trainer import AbstractTrainer, TrainingState
from examples.gpt_2 import GPT2, GPT2ForwardFn

EOS_TOKEN = SFT_TURN_SEPARATOR
PRETRAINED_CHECKPOINT_PATH = "checkpoints/openwebtext_gpt2_124m_baseline_GPT2_14000"
TOKENIZER_VOCAB_PATH = "training_data/openwebtext_vocab_49990.pkl"
VALIDATION_TEMPERATURE = 0.05
TRAIN_VALIDATION_COUNT = 64


def _clear_backend_cache() -> None:
    clear_cache = getattr(xp, "clear_cache", None)
    if callable(clear_cache):
        clear_cache()


@dataclass(kw_only=True)
class GRPOTrainingConfig(GenericTrainingConfig):
    max_steps: int = field()  # pyright: ignore[reportGeneralTypeIssues, reportIncompatibleVariableOverride]
    max_generation_tokens: int
    temperature: float
    top_k: Optional[int]
    # GRPO group size G: number of completions sampled for one prompt.
    num_generations: int
    validation_num_generations: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.max_generation_tokens < 1:
            raise ValueError(
                f"max_generation_tokens must be >= 1, got {self.max_generation_tokens}"
            )
        if self.temperature <= 0.0 or self.top_k is not None:
            raise ValueError("GRPO rollout requires temperature > 0 and top_k=None")
        if self.num_generations < 1:
            raise ValueError(
                f"num_generations must be >= 1, got {self.num_generations}"
            )
        if self.validation_num_generations < 1:
            raise ValueError(
                "validation_num_generations must be >= 1, "
                f"got {self.validation_num_generations}"
            )


# 1. Caller code owns one Task for the first loop.
# 2. Each outer GRPO iteration refreshes rollouts for that current task.
# 3. RolloutGenerator samples group_size completions from the current
#    policy, caching sampled-token logprobs, and computes group-relative advantages
# 4. Environment scores each Sample in each RolloutGroup
#    within each RolloutGroup.
# 5. MapDataset stores the already-generated RolloutGroup.
# 6. DataLoader batches RolloutGroup objects and calls GRPOCollator.
# 7. GRPOCollator emits GRPOBatch, where rows = group_size for one task.
# 8. GRPOTrainer(AbstractTrainer)._forward_and_loss computes the GRPO objective.
# 9. AbstractTrainer.fit owns backward(), gradient accumulation, clipping,
#     optimizer.step(), checkpointing, and reporting.
#
# Keep boundaries explicit:
# - Environment: prompt rendering + reward design only
# - RolloutGenerator: model sampling + sampled-token logprobs + RolloutGroup creation + advantage calculation
# - MapDataset: static container for already-generated rollout groups.
# - GRPOCollator: padding, causal shift, generated-token masks, and GRPOBatch assembly.
# - GRPOTrainer: GRPO loss, and inherited optimizer mechanics.


@dataclass
class Sample:
    """
    One sampled completion for one prompt.

    GRPO should not treat this as the standalone dataset item because the
    advantage is relative to sibling samples from the same prompt.
    """

    completion_tokens: Array
    # This completion_text field is derivable from completion_tokens, but it has a
    # separate purpose: tokens feed the trainer/collator, text feeds rewards and
    # debugging without making Environment depend on a tokenizer.
    completion_text: str
    sampled_token_logprobs: Array
    reward: Optional[float] = None
    advantage: Optional[float] = None  # derived field
    metadata: Optional[dict] = None  # env trace, verifier result etc...

    def __post_init__(self) -> None:
        if len(self.completion_tokens) == 0:
            raise ValueError("completion_tokens must contain at least one token")
        if len(self.completion_tokens) != len(self.sampled_token_logprobs):
            raise ValueError(
                "completion_tokens and sampled_token_logprobs must have the same length"
            )


@dataclass
class Task:
    """
    One environment task.

    Attributes:
        task_id: Stable identifier used to connect the task to its RolloutGroup.
        raw_input: User-facing task text rendered into the model prompt.
        answer: Reference target used by concrete environment rewards.
        metadata: Optional source/debug details that are not part of reward.
    """

    task_id: str
    raw_input: str
    answer: str
    metadata: Optional[dict] = None


@dataclass(frozen=True)
class CurriculumDatasets:
    gsm8k_train: list[Task]
    gsm8k_validation: list[Task]


@dataclass
class RolloutGroup:
    """
    Dataset item for GRPO.

    One RolloutGroup is one rendered prompt plus G sampled completions. This
    keeps the group structure intact so reward normalization can compare sibling
    samples from the same prompt.
    """

    prompt_id: str  # the dataset or prompt builder or rollout coordinate will fill this in. This should be 1-1 mapped to the rendered prompt instance, which can change if anything like the system prompt changes
    prompt_tokens: Array
    samples: List[Sample]

    def __post_init__(self) -> None:
        if len(self.prompt_tokens) == 0:
            raise ValueError("prompt_tokens must contain at least one token")
        if len(self.samples) == 0:
            raise ValueError("RolloutGroup must contain at least one sample")


@dataclass(frozen=True)
class GRPOBatch:
    input_ids: ArrayLike
    labels: ArrayLike
    # Cached rollout-time logprobs (log π_old). Unused by the current loss; kept
    # for E39's clipped surrogate to form the importance ratio.
    sampled_token_logprobs: ArrayLike
    generated_token_mask: ArrayLike
    advantages: ArrayLike
    # Sparse output-head speed optimization: flat indices into (batch*seq) where
    # generated_token_mask == 1, plus their count. Lets the trainer project only
    # generated-token rows through the tied vocab head instead of the full sequence.
    generated_token_indices: Optional[ArrayLike] = None
    has_nonzero_advantage: Optional[bool] = None
    generated_token_count: Optional[int] = None


class GRPOCollator(Collator):
    """
    Builds GRPO batches from rollout groups.

    A rollout group is one prompt plus G sampled completions. In the first
    one-task loop, the collated batch has `group_size` training rows.

    `max_tokens` is the configured upper bound; the actual batch is padded to the
    batch-local maximum (`batch_max_tokens`) to avoid padding to `self.max_tokens`
    when the actual rows are shorter. This is the dominant rollout-stage cost
    saver.

    Boundary decision: DataLoader calls this with RolloutGroup objects and this
    returns the trainer-facing GRPOBatch. The trainer should not need to know how
    prompt/completion tokens are padded, shifted, or masked.
    """

    def __init__(self, max_tokens: int, pad_idx: int) -> None:
        if max_tokens < 2:
            raise ValueError(
                "max_tokens must be >= 2 for GRPO, since this is autoregressive"
            )
        self.max_tokens = max_tokens
        self.pad_idx = pad_idx

    def __call__(self, rollout_groups: Sequence[RolloutGroup]) -> GRPOBatch:
        batch_input_ids = []
        batch_labels = []
        batch_sampled_token_logprobs = []
        batch_generated_token_mask = []
        batch_advantages = []
        has_nonzero_advantage = False

        # this avoids batching to the self.max_tokens
        # instead we use the actual maximum tokens in this group
        batch_max_tokens = max(
            len(rollout_group.prompt_tokens) + len(sample.completion_tokens)
            for rollout_group in rollout_groups
            for sample in rollout_group.samples
        )

        for rollout_group in rollout_groups:
            for sample in rollout_group.samples:
                (
                    input_ids,
                    labels,
                    sampled_token_logprobs,
                    generated_token_mask,
                    advantages,
                ) = self._build_row(
                    rollout_group.prompt_tokens, sample, batch_max_tokens
                )

                batch_input_ids.append(input_ids)
                batch_labels.append(labels)
                batch_sampled_token_logprobs.append(sampled_token_logprobs)
                batch_generated_token_mask.append(generated_token_mask)
                batch_advantages.append(advantages)
                has_nonzero_advantage = has_nonzero_advantage or bool(
                    sample.advantage != 0.0
                )

        generated_token_mask_np = np.stack(batch_generated_token_mask, axis=0)
        generated_token_indices = np.flatnonzero(
            generated_token_mask_np.reshape(-1)
        ).astype(np.int32)

        return GRPOBatch(
            input_ids=xp.array(np.stack(batch_input_ids, axis=0)),
            labels=xp.array(np.stack(batch_labels, axis=0)),
            sampled_token_logprobs=xp.array(
                np.stack(batch_sampled_token_logprobs, axis=0)
            ),
            generated_token_mask=xp.array(generated_token_mask_np),
            advantages=xp.array(np.stack(batch_advantages, axis=0)),
            generated_token_indices=xp.array(generated_token_indices),
            has_nonzero_advantage=has_nonzero_advantage,
            generated_token_count=int(generated_token_mask_np.sum()),
        )

    def _build_row(
        self,
        prompt_tokens: Array,
        sample: Sample,
        batch_max_tokens: int,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        completion_tokens = np.asarray(sample.completion_tokens)

        prompt_np = np.asarray(prompt_tokens, dtype=np.int32)
        prompt_len = len(prompt_np)
        completion_len = len(completion_tokens)
        row_len = prompt_len + completion_len

        if row_len > self.max_tokens:
            raise ValueError(
                f"GRPO row length {row_len} exceeds max_tokens {self.max_tokens}"
            )

        tokens = np.concatenate([prompt_np, completion_tokens], axis=0)

        # Before padding and causal shift, align all per-token rows on the
        # prompt+completion sequence. Example:
        # tokens       = [prompt_token_0, prompt_token_1, completion_token_0]
        # generated_token_mask = [             0,              0,                  1]
        # sampled_token_logprobs = [           0.0,            0.0,  completion_logprob_0]
        generated_token_mask = np.concatenate(
            [
                np.zeros(prompt_len, dtype=np.int32),
                np.ones(completion_len, dtype=np.int32),
            ],
            axis=0,
        )

        aligned_sampled_token_logprobs = np.concatenate(
            [
                np.zeros(prompt_len, dtype=np.float32),
                np.asarray(sample.sampled_token_logprobs, dtype=np.float32),
            ],
            axis=0,
        )

        if sample.advantage is None:
            raise ValueError("sample.advantage must be set before collation")

        aligned_advantages = generated_token_mask.astype(np.float32) * float(
            sample.advantage
        )

        tokens = pad_right_1d(tokens, batch_max_tokens, self.pad_idx)
        generated_token_mask = pad_right_1d(generated_token_mask, batch_max_tokens, 0)
        aligned_sampled_token_logprobs = pad_right_1d(
            aligned_sampled_token_logprobs,
            batch_max_tokens,
            0.0,
        )
        aligned_advantages = pad_right_1d(
            aligned_advantages,
            batch_max_tokens,
            0.0,
        )

        return (
            tokens[:-1],
            tokens[1:],
            aligned_sampled_token_logprobs[1:],
            generated_token_mask[1:],
            aligned_advantages[1:],
        )


class Environment:
    """
    Owns task rendering and reward design.

    Concrete environments decide how to render tasks and score sampled
    completions. They should not know about optimizers or loss.
    """

    def render_task(self, task: Task) -> str:
        # Public prompt-rendering entry point. This is Task-aware so concrete
        # environments can use task metadata or domain-specific wording later.
        return self._render_prompt(task.raw_input)

    def _render_prompt(self, user_prompt: str) -> str:
        # Private template helper. This should stay string-in/string-out and not
        # know about Task, rewards, or rollout state.
        return (
            f"{SFT_ROLE_MARKERS['system']}{SFT_SYSTEM_PROMPT}"
            f"{SFT_TURN_SEPARATOR}{SFT_ROLE_MARKERS['user']}{user_prompt}"
            f"{SFT_TURN_SEPARATOR}{SFT_ROLE_MARKERS['assistant']}"
        )

    @abstractmethod
    def _compute_reward(self, task: Task, sample: Sample) -> float:  # pyright: ignore[reportReturnType]
        raise NotImplementedError()

    def score_group(self, task: Task, rollout_group: RolloutGroup) -> RolloutGroup:
        # Keep Task explicit here: RolloutGroup stores the rendered prompt tokens
        # used for training, but reward code may still need task-level reference
        # data or metadata.
        # Public scoring entry point. Expected to attach rewards to each Sample
        # in the group, usually by calling _compute_reward per sample.
        for sample in rollout_group.samples:
            reward = self._compute_reward(task, sample)
            sample.reward = reward

        return rollout_group


class MathEnvironment(Environment):
    """
    Concrete Environment target for the first working GRPO loop.

    Math is chosen because rewards can be rule-based and fast: exact final-answer
    checking is enough to prove the full loop learns.

    Exact-only reward: format-valid + correct numeric answer earns 1.0, anything
    else earns 0.0. Settled by E24/E33 — shaped wrong-answer rewards collapse
    diversity without lifting pass@1.
    """

    # This regex should be consistent with the SYSTEM_PROMPT defined at the
    # top of this module
    RESPONSE_RE = re.compile(
        r"\A\s*<think>((?:(?!</?think>|</?answer>).)*?)</think>\s*"
        r"<answer>\s*((?:(?!</?think>|</?answer>).)*?)\s*</answer>\s*"
        + re.escape(SFT_TURN_SEPARATOR)
        + r"\s*\Z",
        re.DOTALL,
    )

    @staticmethod
    def normalize_answer(answer: str) -> str:
        return answer.strip().replace(",", "")

    @classmethod
    def parse_numeric_answer(cls, answer: str) -> Optional[float]:
        normalized = cls.normalize_answer(answer).strip()
        if normalized.startswith("$"):
            normalized = normalized[1:].strip()
        if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", normalized):
            return None
        return float(normalized)

    @classmethod
    def answers_match(cls, predicted_answer: str, expected_answer: str) -> bool:
        predicted = cls.normalize_answer(predicted_answer)
        expected = cls.normalize_answer(expected_answer)
        if predicted == expected:
            return True
        predicted_number = cls.parse_numeric_answer(predicted)
        expected_number = cls.parse_numeric_answer(expected)
        return (
            predicted_number is not None
            and expected_number is not None
            and predicted_number == expected_number
        )

    @classmethod
    def extract_gsm8k_final_answer(cls, answer: str) -> str:
        _, final_answer = split_gsm8k_answer(answer)
        return cls.normalize_answer(final_answer)

    @classmethod
    def gsm8k_row_to_task(cls, row_idx: int, row: dict[str, Any]) -> Task:
        question = row.get("question")
        answer = row.get("answer")
        if not isinstance(question, str) or not isinstance(answer, str):
            raise ValueError(
                "GSM8K rows must contain string question and answer fields"
            )
        reasoning, final_answer = split_gsm8k_answer(answer)
        return Task(
            task_id=f"gsm8k-{row_idx}",
            raw_input=question,
            answer=cls.normalize_answer(final_answer),
            metadata={"source": "openai/gsm8k", "reasoning": reasoning},
        )

    @classmethod
    def load_gsm8k_tasks(cls, split: str, max_tasks: Optional[int]) -> list[Task]:
        rows = load_gsm8k_rows(split=split, max_rows=max_tasks)
        return [cls.gsm8k_row_to_task(row_idx, row) for row_idx, row in enumerate(rows)]

    def _compute_reward(self, task: Task, sample: Sample) -> float:
        match = self.RESPONSE_RE.match(sample.completion_text)
        format_valid = match is not None
        parsed_answer = None
        parsed_answer_number = None
        expected_answer_number = self.parse_numeric_answer(task.answer)
        relative_error = None
        exact_match = False
        if format_valid:
            parsed_answer = self.normalize_answer(match.group(2))
            parsed_answer_number = self.parse_numeric_answer(parsed_answer)
            exact_match = self.answers_match(parsed_answer, task.answer)
            if parsed_answer_number is not None and expected_answer_number is not None:
                abs_error = abs(parsed_answer_number - expected_answer_number)
                relative_error = abs_error / max(1.0, abs(expected_answer_number))

        if sample.metadata is None:
            sample.metadata = {}
        sample.metadata.update(
            {
                "format_valid": format_valid,
                "exact_match": exact_match,
                "parsed_answer": parsed_answer,
                "parsed_answer_number": parsed_answer_number,
                "expected_answer_number": expected_answer_number,
                "relative_error": relative_error,
            }
        )

        return 1.0 if exact_match else 0.0


class RolloutGenerator:
    """
    Samples on-policy completions from the current model.

    Given one Task, it renders the prompt via Environment, samples G completions,
    caches sampled-token logprobs, asks Environment to score samples, and returns
    one RolloutGroup. It does not perform optimizer work.
    """

    def __init__(self, config: GRPOTrainingConfig) -> None:
        self.config = config

    def rollout(
        self,
        model: Module,
        task: Task,
        tokenizer: BytePairEncoder,
        environment: Environment,
        *,
        num_generations: Optional[int] = None,
    ) -> RolloutGroup:
        task_prompt: str = environment.render_task(task)
        prompt_tokens = np.array(tokenizer.encode(task_prompt), dtype=np.int32)
        if num_generations is None:
            num_generations = self.config.num_generations

        # Sample G completions from the current model. `compute_logprobs=False`
        # skips logprob computation in the sampler — the current GRPO loss
        # recomputes them during training. E39's clipped surrogate will need to
        # flip this back on so the cached log π_old is meaningful.
        prompt_token_list = [int(token) for token in prompt_tokens]
        available_new_tokens = model.max_seq_len - len(prompt_token_list)
        if available_new_tokens <= 0:
            raise ValueError(
                f"prompt length {len(prompt_token_list)} leaves no room for "
                f"generation within model max_seq_len {model.max_seq_len}"
            )
        eos_token_id = tokenizer.encode(EOS_TOKEN)[0]
        was_training = getattr(model, "_is_training", None)
        model.eval()
        try:
            results = generate(
                model=model,
                prediction_func=GPT2ForwardFn(),
                prompt_tokens=prompt_token_list,
                max_new_tokens=min(
                    self.config.max_generation_tokens, available_new_tokens
                ),
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                eos_token_id=eos_token_id,
                show_progress=False,
                num_generations=num_generations,
                compute_logprobs=False,
            )
        finally:
            if was_training:
                model.train()

        # Keep sampled token ids directly. Decoding and re-encoding can change
        # BPE boundaries at the rendered-prompt/completion join.
        samples = [
            Sample(
                completion_tokens=np.array(result.completion_tokens, dtype=np.int32),
                completion_text=tokenizer.decode(result.completion_tokens),
                sampled_token_logprobs=np.array(result.logprobs, dtype=np.float32),
                reward=None,
                advantage=None,
                metadata={
                    "stop_reason": result.stop_reason,
                    "temperature": self.config.temperature,
                    "top_k": self.config.top_k,
                },
            )
            for result in results
        ]

        rollout_group = RolloutGroup(
            prompt_id=task.task_id,
            prompt_tokens=prompt_tokens,
            samples=samples,
        )
        rollout_group = environment.score_group(task, rollout_group)
        self._compute_advantages(rollout_group)
        return rollout_group

    def _compute_advantages(self, rollout_group: RolloutGroup) -> None:
        r"""
        Advantage normalization
        G = group size
        r = reward

        1. Calculate the mean
        2. Calculate the standard deviation
        3. Calculate the advantage

        $A_i = \frac{r_i - \mu}{\sigma}$
        """
        rewards = []
        for sample in rollout_group.samples:
            if sample.reward is None:
                raise ValueError(
                    "sample.reward must be set before advantage calculation"
                )
            rewards.append(sample.reward)

        rewards = np.array(rewards)
        rewards_mean = rewards.mean()
        rewards_std_dev = rewards.std()
        for sample in rollout_group.samples:
            if rewards_std_dev == 0.0:
                sample.advantage = 0.0
            else:
                sample.advantage = (sample.reward - rewards_mean) / rewards_std_dev


class GRPOTrainer(AbstractTrainer):
    """
    Trainer boundary for GRPOBatch optimization.

    Rollout happens upstream; this owns forward, loss, backward, gradient
    scaling/clipping, optimizer step, and global-step bookkeeping.

    A higher-level orchestration loop is responsible for:

        Task -> RolloutGenerator.rollout(...) -> GRPOCollator -> self.train_step(...)

    Eval is rollout-based (pass@1, any@k, majority@k), not loss-based, so it
    lives in `eval_callbacks` rather than `_evaluate`. The orchestration loop
    triggers it via `run_eval_callbacks()` at its chosen cadence.
    """

    def __init__(
        self,
        *args,
        eval_callbacks: Optional[Sequence[Callable[["GRPOTrainer"], None]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.eval_callbacks: List[Callable[["GRPOTrainer"], None]] = list(
            eval_callbacks or ()
        )

    def run_eval_callbacks(self) -> None:
        for callback in self.eval_callbacks:
            callback(self)

    def _forward_and_loss(self, batch: GRPOBatch):
        """
        Compute the GRPO objective from a trainer-facing batch.

        The loss should use current-policy logprobs from
        `self.model(batch.input_ids)`, `batch.advantages` for the group-relative
        learning signal, and `batch.generated_token_mask` so prompt/pad/forced
        tokens do not contribute.

        Uses the sparse-output-head speed optimization when generated-token
        indices are available: the model forward projects only rows that
        contribute to the GRPO loss. Falls back to full-logits loss otherwise.
        """

        generated_token_indices = batch.generated_token_indices
        if generated_token_indices is not None:
            # Prompt/pad rows have zero GRPO weight; ask the normal model
            # forward to return logits only for rows that affect the loss.
            selected_logits = self.model(
                batch.input_ids,
                selected_token_indices=generated_token_indices,
            )
            return grpo_loss_from_selected_logits(
                selected_logits,
                batch,
                generated_token_indices,
            )

        logits = self.model(batch.input_ids)
        return self.loss_fn(logits, batch)

    def _loss_total_weight(self, batch: GRPOBatch):
        r"""
        Currently this is token-level loss.
        $$
        \frac{1}{\sum_i T_i} \sum_{i=1}^G \sum_{t=1}^T \text{Advantage}_i \log \text{prob}_{i, t}
        $$
        Long bad outputs -> large negative influence
        Long verbose correct outputs -> large positive influence
        """
        if batch.generated_token_count is not None:
            return xp.array(float(batch.generated_token_count), dtype=xp.float32)
        return xp.sum(batch.generated_token_mask)

    def _evaluate(self, val_data_loader):
        # Required override; GRPO evaluation happens via validate_policy outside
        # the AbstractTrainer fit-loop.
        del val_data_loader

    def train_step(self, batch: GRPOBatch) -> Tensor:
        """
        Apply one optimizer update to one already-generated GRPO batch.

        Online GRPO refreshes rollout data outside the trainer. This method keeps
        the optimization boundary here: forward, loss, backward, gradient
        scaling/clipping, optimizer step, and global step bookkeeping.

        We might want to resort back to the normal trainer.fit() way of training later, after we decide to go off-policy with a separate generation policy and trained policy

        Skips the optimizer entirely when reward variance is zero across the
        group (all-correct or all-wrong) — in that case all advantages are zero
        and the backward pass is wasted work.
        """
        self.model.train()
        self.optimizer.zero_grad()

        has_nonzero_advantage = batch.has_nonzero_advantage
        if has_nonzero_advantage is None:
            has_nonzero_advantage = not bool(
                xp.to_scalar(xp.all(xp.asarray(batch.advantages) == 0))
            )
        if not has_nonzero_advantage:
            self.global_step += 1
            return Tensor(xp.array(0.0, dtype=xp.float32), requires_grad=False)

        state = TrainingState()
        loss = self._forward_and_loss(batch)
        total_weight = self._loss_total_weight(batch)
        loss.backward()
        state.accumulated_batches = 1
        state.accumulated_loss_total_weight = total_weight

        if self.optimizer_step(state, record_grad_norm=False):
            self.global_step += 1

        return loss

    def save_checkpoint(self) -> tuple[str, str]:
        checkpoint = {
            "epoch": 0,
            "step_count": self.global_step,
            "steps_per_epoch": None,
            "best_val_loss": self.best_val_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_init_kwargs": self.checkpoint.get(
                "model_init_kwargs",
                self.config.model_kwargs,
            ),
            "optimizer_init_kwargs": self.checkpoint.get(
                "optimizer_init_kwargs",
                self.config.optimizer_kwargs,
            ),
            "config_repr": repr(self.config),
        }
        checkpoint_name = (
            f"{self.config.training_run_name}_"
            f"{self.model.__class__.__name__}_{self.global_step}_final"
        )
        return save_checkpoint(
            checkpoint,
            checkpoint_dir=self.CHECKPOINT_DIR,
            checkpoint_name=checkpoint_name,
        )


def grpo_loss(logits: Tensor, batch: GRPOBatch) -> Tensor:
    """
    Compute the summed simplified GRPO objective from a model-facing batch.

    `batch.advantages` and `batch.generated_token_mask` are rollout-time
    constants. Only `logits` participates in autograd.
    """
    labels = xp.asarray(batch.labels, dtype=xp.int32)
    generated_token_mask = xp.asarray(batch.generated_token_mask, dtype=xp.float32)
    advantages = xp.asarray(batch.advantages, dtype=xp.float32)

    if logits.ndim != 3:
        raise ValueError("logits must have shape (batch, seq_len, vocab_size)")
    if labels.shape != logits.shape[:2]:
        raise ValueError("labels must have shape (batch, seq_len)")
    if generated_token_mask.shape != labels.shape:
        raise ValueError("generated_token_mask must have shape (batch, seq_len)")
    if advantages.shape != labels.shape:
        raise ValueError("advantages must have shape (batch, seq_len)")

    logprobs = functional.log_softmax(logits, dim=-1)
    batch_idx = xp.arange(labels.shape[0])[:, None]
    seq_idx = xp.arange(labels.shape[1])[None, :]

    r"""
    Gather the current-policy logprob for each sampled token:
    $$
    \ell_{i,t}(\theta)
    = \log \pi_\theta(\text{token}_{i,t}\mid \text{prompt},\text{token}_{i,<t})
    $$

    TODO (E39): add KL regularization against a frozen reference policy.
    TODO (E39): add a clipped surrogate that uses cached sampled_token_logprobs
    as log π_old and applies DAPO Clip-Higher (lower 0.8, upper ~1.28).
    TODO (E39): switch the per-row length normalization to Dr.GRPO's constant
    max_completion_len so length bias is removed.

    Advantage-weighted token objective:
    $$
    Loss(\theta) = -\sum_{i,t} mask_{i,t} \text{Advantage}_i \ell_{i,t}(\theta)
    $$
    """
    sampled_token_logprobs = logprobs[batch_idx, seq_idx, labels]
    return -(sampled_token_logprobs * advantages * generated_token_mask).sum()


def grpo_loss_from_selected_logits(
    logits: Tensor,
    batch: GRPOBatch,
    generated_token_indices: ArrayLike,
) -> Tensor:
    """
    Same GRPO objective as grpo_loss when logits already contain only the
    generated-token positions.
    """
    indices = xp.asarray(generated_token_indices, dtype=xp.int32)

    labels = xp.asarray(batch.labels, dtype=xp.int32).reshape(-1)[indices]
    advantages = xp.asarray(batch.advantages, dtype=xp.float32).reshape(-1)[indices]

    logprobs = functional.log_softmax(logits, dim=-1)
    row_idx = xp.arange(labels.shape[0])
    sampled_token_logprobs = logprobs[row_idx, labels]
    return -(sampled_token_logprobs * advantages).sum()


def rollout_metrics(samples: Sequence[Sample]) -> dict[str, float]:
    if not samples:
        raise ValueError("rollout metrics require at least one sample")

    rewards = []
    format_valid = 0
    exact_match = 0
    turn_end = 0
    completion_lengths = []
    parsed_answers = []
    nonzero_advantages = 0
    for sample in samples:
        if sample.reward is None:
            raise ValueError("sample.reward must be set before logging")
        metadata = sample.metadata or {}
        rewards.append(sample.reward)
        format_valid += int(bool(metadata.get("format_valid", False)))
        exact_match += int(bool(metadata.get("exact_match", False)))
        turn_end += int(metadata.get("stop_reason") == "eos")
        completion_lengths.append(len(sample.completion_tokens))
        parsed_answers.append(metadata.get("parsed_answer"))
        nonzero_advantages += int(
            sample.advantage is not None and sample.advantage != 0.0
        )

    rewards_array = np.array(rewards, dtype=np.float32)
    sample_count = float(len(samples))
    answer_counts = {
        parsed_answer: parsed_answers.count(parsed_answer)
        for parsed_answer in set(parsed_answers)
    }
    dominant_answer_count = max(answer_counts.values())
    return {
        "reward_mean": float(rewards_array.mean()),
        "reward_std": float(rewards_array.std()),
        "reward_max": float(rewards_array.max()),
        "format_valid_rate": format_valid / sample_count,
        "exact_match_rate": exact_match / sample_count,
        "turn_end_rate": turn_end / sample_count,
        "completion_len_mean": float(np.mean(completion_lengths)),
        "answer_unique_count": float(len(answer_counts)),
        "dominant_answer_rate": dominant_answer_count / sample_count,
        "nonzero_advantage_rate": nonzero_advantages / sample_count,
    }


def validate_policy(
    *,
    step: int,
    model: Module,
    rollout_generator: RolloutGenerator,
    validation_tasks: Sequence[Task],
    tokenizer: BytePairEncoder,
    environment: Environment,
    progress_bar: tqdm,
    label: str,
    num_generations: Optional[int] = None,
) -> dict[str, float]:
    validation_samples = []
    majority_exact = 0
    any_exact = 0
    best_sample = None
    best_prompt_tokens = None
    best_reward = -float("inf")
    with no_grad():
        for validation_task in validation_tasks:
            validation_group = rollout_generator.rollout(
                model=model,
                task=validation_task,
                tokenizer=tokenizer,
                environment=environment,
                num_generations=num_generations,
            )
            validation_samples.extend(validation_group.samples)
            parsed_answers = [
                sample.metadata.get("parsed_answer")
                for sample in validation_group.samples
                if sample.metadata is not None
                and sample.metadata.get("parsed_answer") is not None
            ]
            answer_counts = {
                parsed_answer: parsed_answers.count(parsed_answer)
                for parsed_answer in set(parsed_answers)
            }
            majority_answer = None
            if answer_counts:
                majority_answer = max(answer_counts.items(), key=lambda item: item[1])[
                    0
                ]
            majority_exact += int(
                majority_answer is not None
                and MathEnvironment.answers_match(
                    majority_answer, validation_task.answer
                )
            )
            any_exact += int(
                any(
                    bool((sample.metadata or {}).get("exact_match", False))
                    for sample in validation_group.samples
                )
            )
            for sample in validation_group.samples:
                if sample.reward is None:
                    raise ValueError(
                        "validation sample reward must be set before logging"
                    )
                if sample.reward > best_reward:
                    best_reward = sample.reward
                    best_sample = sample
                    best_prompt_tokens = validation_group.prompt_tokens

    validation_metrics = rollout_metrics(validation_samples)
    task_count = float(len(validation_tasks))
    validation_metrics["majority_exact_match_rate"] = majority_exact / task_count
    validation_metrics["any_exact_match_rate"] = any_exact / task_count
    progress_bar.write(
        "validation "
        f"{label} "
        f"step={step} "
        f"reward_mean={validation_metrics['reward_mean']:.3f} "
        f"reward_max={validation_metrics['reward_max']:.3f} "
        f"exact={validation_metrics['majority_exact_match_rate']:.3f} "
        f"sample_exact={validation_metrics['exact_match_rate']:.3f} "
        f"any_exact={validation_metrics['any_exact_match_rate']:.3f} "
        f"format={validation_metrics['format_valid_rate']:.3f} "
        f"turn_end={validation_metrics['turn_end_rate']:.3f} "
        f"len={validation_metrics['completion_len_mean']:.1f} "
        f"answers={validation_metrics['answer_unique_count']:.0f} "
        f"dominant={validation_metrics['dominant_answer_rate']:.3f} "
        f"updates={validation_metrics['nonzero_advantage_rate']:.3f}"
    )
    if best_sample is not None and best_prompt_tokens is not None:
        progress_bar.write("Prompt:")
        progress_bar.write(tokenizer.decode(best_prompt_tokens))
        progress_bar.write(best_sample.completion_text)
        progress_bar.write(
            "--------------------------------------------------------------"
        )
    _clear_backend_cache()
    return validation_metrics


def make_validation_callback(
    *,
    label: str,
    validation_tasks: Sequence[Task],
    rollout_generator: RolloutGenerator,
    tokenizer: BytePairEncoder,
    environment: Environment,
    progress_bar: tqdm,
    num_generations: Optional[int],
) -> Callable[[GRPOTrainer], None]:
    # Wraps validate_policy as a GRPOTrainer.eval_callbacks entry. The closure
    # captures everything validate_policy needs except the live model + step,
    # which it reads off the trainer when fired.
    def callback(trainer: GRPOTrainer) -> None:
        validate_policy(
            step=trainer.global_step,
            model=trainer.model,
            rollout_generator=rollout_generator,
            validation_tasks=validation_tasks,
            tokenizer=tokenizer,
            environment=environment,
            progress_bar=progress_bar,
            label=label,
            num_generations=num_generations,
        )

    return callback


def filter_fitting_tasks(
    tasks: Sequence[Task],
    *,
    environment: Environment,
    tokenizer: BytePairEncoder,
    max_tokens: int,
    max_generation_tokens: int,
) -> list[Task]:
    fitting_tasks = []
    for task in tasks:
        prompt_len = len(tokenizer.encode(environment.render_task(task)))
        if prompt_len + max_generation_tokens <= max_tokens:
            fitting_tasks.append(task)
    return fitting_tasks


def build_curriculum_datasets(
    *,
    environment: MathEnvironment,
    tokenizer: BytePairEncoder,
    max_tokens: int,
    max_generation_tokens: int,
    gsm8k_train_count: Optional[int] = None,
    gsm8k_validation_count: int = 64,
    gsm8k_validation_offset: Optional[int] = None,
) -> CurriculumDatasets:
    if gsm8k_validation_offset is None:
        gsm8k_validation_offset = (
            128 if gsm8k_train_count is None else gsm8k_train_count
        )
    max_gsm8k_rows = None
    if gsm8k_train_count is not None:
        max_gsm8k_rows = max(
            gsm8k_train_count,
            gsm8k_validation_offset + gsm8k_validation_count,
        )
    gsm8k_tasks = filter_fitting_tasks(
        environment.load_gsm8k_tasks(
            split="train",
            max_tasks=max_gsm8k_rows,
        ),
        environment=environment,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        max_generation_tokens=max_generation_tokens,
    )
    if len(gsm8k_tasks) < gsm8k_validation_offset + gsm8k_validation_count:
        raise ValueError(
            "Not enough fitting GSM8K tasks for requested train/validation split: "
            f"got {len(gsm8k_tasks)}"
        )
    if gsm8k_train_count is None:
        gsm8k_train = (
            gsm8k_tasks[:gsm8k_validation_offset]
            + gsm8k_tasks[gsm8k_validation_offset + gsm8k_validation_count :]
        )
    else:
        gsm8k_train = gsm8k_tasks[:gsm8k_train_count]
    return CurriculumDatasets(
        gsm8k_train=gsm8k_train,
        gsm8k_validation=gsm8k_tasks[
            gsm8k_validation_offset : gsm8k_validation_offset + gsm8k_validation_count
        ],
    )


def build_training_config(ckpt: dict[str, Any]) -> GRPOTrainingConfig:
    # Default config runs actual GRPO from the configured checkpoint. The clean
    # OpenWebText base produces ~0 reward signal on GSM8K, so groups will short
    # -circuit through `has_nonzero_advantage` until the starting checkpoint has
    # a non-trivial pass@1. Swap PRETRAINED_CHECKPOINT_PATH to an SFT checkpoint
    # before expecting learning to happen.
    return GRPOTrainingConfig(
        training_run_name="grpo_gpt2_llama",
        max_steps=200,
        max_eval_steps=32,
        checkpoint_freq=100,
        report_every_steps=50,
        # GRPO train_step currently performs one optimizer step per rollout
        # group. num_generations is the effective GRPO group-size knob until we
        # add GRPO gradient accumulation.
        global_batch_size=1,
        micro_batch_size=1,
        max_grad_norm=1.0,
        model_kwargs=ckpt["model_init_kwargs"],
        optimizer_kwargs={"lr": 1e-6},
        pretrained_checkpoint_path=PRETRAINED_CHECKPOINT_PATH,
        max_generation_tokens=128,
        temperature=0.7,
        top_k=None,
        num_generations=16,
        validation_num_generations=1,
    )


def main():
    ckpt = load_checkpoint_metadata(
        f"{PRETRAINED_CHECKPOINT_PATH}.json",
        f"{PRETRAINED_CHECKPOINT_PATH}.npz",
    )

    TRAIN_CONFIG = build_training_config(ckpt)

    trainer = GRPOTrainer(
        model_cls=GPT2,
        optimizer_cls=Adam,
        loss_fn=grpo_loss,
        config=TRAIN_CONFIG,
    )

    bpe = BytePairEncoder(
        num_merges=49990,
        vocab_file_path=TOKENIZER_VOCAB_PATH,
    )
    environment = MathEnvironment()
    datasets = build_curriculum_datasets(
        environment=environment,
        tokenizer=bpe,
        max_tokens=trainer.model.max_seq_len,
        max_generation_tokens=TRAIN_CONFIG.max_generation_tokens,
    )
    train_tasks = datasets.gsm8k_train
    if not datasets.gsm8k_train or not datasets.gsm8k_validation:
        raise ValueError("No GSM8K tasks fit within the model context window")

    print(
        "Data length: "
        f"gsm8k_train={len(datasets.gsm8k_train)} "
        f"gsm8k_val={len(datasets.gsm8k_validation)}"
    )
    rollout_generator = RolloutGenerator(TRAIN_CONFIG)
    validation_rollout_generator = RolloutGenerator(
        replace(TRAIN_CONFIG, temperature=VALIDATION_TEMPERATURE)
    )
    gsm8k_validation_sets = [
        ("gsm8k_train_seen", datasets.gsm8k_train[:TRAIN_VALIDATION_COUNT]),
        ("gsm8k", datasets.gsm8k_validation),
    ]
    print(
        "Validation: "
        f"temperature={VALIDATION_TEMPERATURE:.2f} "
        f"gsm8k_train_seen={len(gsm8k_validation_sets[0][1])} "
        f"gsm8k_heldout={len(datasets.gsm8k_validation)}"
    )
    collator = GRPOCollator(
        max_tokens=trainer.model.max_seq_len,
        pad_idx=bpe.encode("<PAD>")[0],
    )
    warmup_kv = getattr(trainer.model, "warmup_kv", None)
    if callable(warmup_kv):
        first_prompt_tokens = bpe.encode(
            environment.render_task(datasets.gsm8k_train[0])
        )
        warmup_kv(
            prompt_len=max(1, len(first_prompt_tokens)),
            decode_steps=8,
            batch_size=TRAIN_CONFIG.num_generations,
        )
        _clear_backend_cache()
    report_every_steps = TRAIN_CONFIG.report_every_steps or TRAIN_CONFIG.checkpoint_freq

    with tqdm(
        total=TRAIN_CONFIG.max_steps,
        initial=trainer.global_step,
        desc="GRPO training",
    ) as progress_bar:
        # Register validation as eval callbacks once. Each callback closes over
        # its own task split + label so the training loop can fire them all via
        # trainer.run_eval_callbacks() without re-passing the per-split state.
        trainer.eval_callbacks = [
            make_validation_callback(
                label=label,
                validation_tasks=validation_tasks,
                rollout_generator=validation_rollout_generator,
                tokenizer=bpe,
                environment=environment,
                progress_bar=progress_bar,
                num_generations=TRAIN_CONFIG.validation_num_generations,
            )
            for label, validation_tasks in gsm8k_validation_sets
        ]
        trainer.run_eval_callbacks()

        while trainer.global_step < TRAIN_CONFIG.max_steps:
            step_before = trainer.global_step
            task = train_tasks[trainer.global_step % len(train_tasks)]
            rollout_group = rollout_generator.rollout(
                model=trainer.model,
                task=task,
                tokenizer=bpe,
                environment=environment,
            )
            train_batch = collator([rollout_group])
            train_loss = trainer.train_step(train_batch)
            materialize(train_loss)
            _clear_backend_cache()

            train_metrics = rollout_metrics(rollout_group.samples)
            progress_bar.update(trainer.global_step - step_before)
            progress_bar.set_postfix(
                reward_mean=f"{train_metrics['reward_mean']:.3f}",
                exact=f"{train_metrics['exact_match_rate']:.3f}",
                format=f"{train_metrics['format_valid_rate']:.3f}",
                answers=f"{train_metrics['answer_unique_count']:.0f}",
                updates=f"{train_metrics['nonzero_advantage_rate']:.3f}",
            )

            if trainer.global_step != step_before:
                should_validate = trainer.global_step % report_every_steps == 0
                if should_validate or trainer.global_step >= TRAIN_CONFIG.max_steps:
                    trainer.run_eval_callbacks()

        cp_path_json, cp_path_npz = trainer.save_checkpoint()
        progress_bar.write(f"Saved final checkpoint: {cp_path_json}, {cp_path_npz}")


if __name__ == "__main__":
    main()
