import re
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from tqdm import tqdm

from autograd import functional
from autograd.backend import Array, ArrayLike, xp
from autograd.data.collator import Collator, pad_right_1d
from autograd.nn import Module
from autograd.optim import Adam
from autograd.tensor import Tensor, no_grad
from autograd.text.tokenizer import BytePairEncoder
from autograd.text.utils import generate
from autograd.tools.config_schema import GenericTrainingConfig
from autograd.tools.model import load_checkpoint
from autograd.tools.trainer import AbstractTrainer, TrainingState
from examples.gpt_2 import GPT2, GPT2ForwardFn

# Template for DeepSeek-R1-Zero. Table 1
# Paper: https://arxiv.org/pdf/2501.12948
SYSTEM_PROMPT = """
A conversation between User and Assistant. The user asks a question, and the Assistant solves
it. The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <think>...</think>
and <answer>...</answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>.
"""

EOS_TOKEN = "<|endoftext|>"


@dataclass(kw_only=True)
class GRPOTrainingConfig(GenericTrainingConfig):
    max_steps: int = field()  # pyright: ignore[reportGeneralTypeIssues, reportIncompatibleVariableOverride]
    max_generation_tokens: int
    temperature: float
    top_k: Optional[int]
    # GRPO group size G: number of completions sampled for one prompt.
    num_generations: int

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.max_generation_tokens < 1:
            raise ValueError(
                f"max_generation_tokens must be >= 1, got {self.max_generation_tokens}"
            )
        if self.temperature != 1.0 or self.top_k is not None:
            raise ValueError("GRPO rollout requires temperature=1.0 and top_k=None")
        if self.num_generations < 1:
            raise ValueError(
                f"num_generations must be >= 1, got {self.num_generations}"
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
# 8. GRPOTrainer(AbstractTrainer)._compute_loss computes the GRPO objective.
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
    sampled_token_logprobs: ArrayLike
    generated_token_mask: ArrayLike
    advantages: ArrayLike


class GRPOCollator(Collator):
    """
    Builds fixed-length GRPO batches from rollout groups.

    A rollout group is one prompt plus G sampled completions. In the first
    one-task loop, the collated batch has `group_size` training rows.

    `max_tokens` is the total row length before causal shifting:
    `len(prompt_tokens) + len(completion_tokens)`. It is not the rollout
    generation limit, which only caps completion length.

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

        for rollout_group in rollout_groups:
            for sample in rollout_group.samples:
                (
                    input_ids,
                    labels,
                    sampled_token_logprobs,
                    generated_token_mask,
                    advantages,
                ) = self._build_row(rollout_group.prompt_tokens, sample)

                batch_input_ids.append(input_ids)
                batch_labels.append(labels)
                batch_sampled_token_logprobs.append(sampled_token_logprobs)
                batch_generated_token_mask.append(generated_token_mask)
                batch_advantages.append(advantages)

        return GRPOBatch(
            input_ids=xp.stack(batch_input_ids, axis=0),
            labels=xp.stack(batch_labels, axis=0),
            sampled_token_logprobs=xp.stack(batch_sampled_token_logprobs, axis=0),
            generated_token_mask=xp.stack(batch_generated_token_mask, axis=0),
            advantages=xp.stack(batch_advantages, axis=0),
        )

    def _build_row(
        self,
        prompt_tokens: Array,
        sample: Sample,
    ) -> tuple[Array, Array, Array, Array, Array]:
        completion_tokens = sample.completion_tokens

        prompt_len = len(prompt_tokens)
        completion_len = len(completion_tokens)
        row_len = prompt_len + completion_len

        if row_len > self.max_tokens:
            raise ValueError(
                f"GRPO row length {row_len} exceeds max_tokens {self.max_tokens}"
            )

        tokens = xp.concatenate(
            [prompt_tokens, completion_tokens],
            axis=0,
        )

        # Before padding and causal shift, align all per-token rows on the
        # prompt+completion sequence. Example:
        # tokens       = [prompt_token_0, prompt_token_1, completion_token_0]
        # generated_token_mask = [             0,              0,                  1]
        # sampled_token_logprobs = [           0.0,            0.0,  completion_logprob_0]
        generated_token_mask = xp.concatenate(
            [
                xp.zeros(prompt_len, dtype=xp.int32),
                xp.ones(completion_len, dtype=xp.int32),
            ],
            axis=0,
        )

        aligned_sampled_token_logprobs = xp.concatenate(
            [
                xp.zeros(prompt_len, dtype=xp.float32),
                sample.sampled_token_logprobs,
            ],
            axis=0,
        )

        if sample.advantage is None:
            raise ValueError("sample.advantage must be set before collation")

        aligned_advantages = generated_token_mask.astype(xp.float32) * float(
            sample.advantage
        )

        # Fixed padding keeps this first GRPO collator simple. We can switch to
        # dynamic padding later if padding waste becomes a measured bottleneck.
        tokens = pad_right_1d(tokens, self.max_tokens, self.pad_idx)
        generated_token_mask = pad_right_1d(generated_token_mask, self.max_tokens, 0)
        aligned_sampled_token_logprobs = pad_right_1d(
            aligned_sampled_token_logprobs,
            self.max_tokens,
            0.0,
        )
        aligned_advantages = pad_right_1d(
            aligned_advantages,
            self.max_tokens,
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
        return SYSTEM_PROMPT + f"User: {user_prompt}{EOS_TOKEN}Assistant: "

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
    """

    # This regex should be consistent with the SYSTEM_PROMPT defined at the
    # top of this module
    ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)

    def _compute_reward(self, task: Task, sample: Sample) -> float:
        reward = 0.0
        if "<think>" in sample.completion_text:
            reward += 0.1
        if "</think>" in sample.completion_text:
            reward += 0.1
        if "<answer>" in sample.completion_text:
            reward += 0.1
        if "</answer>" in sample.completion_text:
            reward += 0.1

        match = self.ANSWER_RE.search(sample.completion_text)
        if match is None:
            return reward

        parsed_answer = match.group(1).strip()
        expected_answer = task.answer.strip()
        if parsed_answer == expected_answer:
            reward += 1.0

        return reward


class RolloutGenerator:
    """
    Samples on-policy completions from the current model.

    Given one Task, it should render the prompt via Environment, sample G
    completions, cache sampled-token logprobs, ask Environment to score
    samples, and return one RolloutGroup. It should not perform optimizer work.
    """

    def __init__(self, config: GRPOTrainingConfig) -> None:
        self.config = config

    def rollout(
        self,
        model: Module,
        task: Task,
        tokenizer: BytePairEncoder,
        environment: Environment,
    ) -> RolloutGroup:
        task_prompt: str = environment.render_task(task)
        prompt_tokens = xp.array(tokenizer.encode(task_prompt), dtype=xp.int32)
        samples = generation(
            model,
            prompt_tokens=prompt_tokens,
            tokenizer=tokenizer,
            num_generations=self.config.num_generations,
            max_generation_tokens=self.config.max_generation_tokens,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
        )

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
        2. Calcuate the standard deviation
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

    This can subclass AbstractTrainer because rollout has already happened by
    the time it receives a GRPOBatch. The trainer can keep owning backward(),
    gradient scaling, clipping, optimizer.step(), and step bookkeeping.

    It should not own raw tasks, Environment, or RolloutGenerator. A higher
    level orchestration loop is responsible for:

        Task -> RolloutGenerator.rollout(...) -> GRPOCollator -> self.train_step(...)

    Each one-task GRPOBatch contains group_size rows.
    """

    def _compute_loss(self, batch: GRPOBatch):
        """
        Compute the GRPO objective from a trainer-facing batch.

        The loss should use current-policy logprobs from
        `self.model(batch.input_ids)`, `batch.advantages` for the group-relative
        learning signal, and `batch.generated_token_mask` so prompt/pad/forced
        tokens do not contribute.
        """

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
        return xp.sum(batch.generated_token_mask)

    def _evaluate(self, val_data_loader):
        pass

    def train_step(self, batch: GRPOBatch) -> Tensor:
        """
        Apply one optimizer update to one already-generated GRPO batch.

        Online GRPO refreshes rollout data outside the trainer. This method keeps
        the optimization boundary here: forward, loss, backward, gradient
        scaling/clipping, optimizer step, and global step bookkeeping.

        We might want to resort back to the normal trainer.fit() way of training later, after we decide to go off-policy with a separate generation policy and trained policy
        """
        self.model.train()
        self.optimizer.zero_grad()

        state = TrainingState()
        loss = self._compute_loss(batch)
        total_weight = self._loss_total_weight(batch)
        loss.backward()
        state.record_loss(loss, total_weight=total_weight)

        if self.optimizer_step(state):
            self.global_step += 1

        return loss


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

    TODO: add KL regularization against a reference policy once this example
    introduces a separate reference model.
    TODO: add a clipped surrogate once sampled-token logprobs are meant to
    represent a distinct behavior policy.
    TODO: add the policy-gradient/policy-ratio surrogate when old/current policy
    ownership is explicit in the training loop.

    Advantage-weighted token objective:
    $$
    Loss(\theta) = -\sum_{i,t} mask_{i,t} \text{Advantage}_i \ell_{i,t}(\theta)
    $$
    """
    sampled_token_logprobs = logprobs[batch_idx, seq_idx, labels]
    return -(sampled_token_logprobs * advantages * generated_token_mask).sum()


def generation(
    model: Module,
    prompt_tokens: Array,
    tokenizer: BytePairEncoder,
    *,
    num_generations: int,
    max_generation_tokens: int,
    temperature: float,
    top_k: Optional[int] = None,
    eos_token: str = EOS_TOKEN,
) -> List[Sample]:
    # Sampled-token logprobs must be raw-policy logprobs. With temperature=1 and no
    # top-k, the current generate() sampling logprobs match raw model logprobs.
    # TODO: split sampling-logprobs from raw-policy logprobs if we add rollout
    # temperature/top-k exploration.
    if temperature != 1.0 or top_k is not None:
        raise ValueError("GRPO rollout requires temperature=1.0 and top_k=None")

    prompt_token_list = [int(token) for token in prompt_tokens]
    eos_token_id = tokenizer.encode(eos_token)[0]

    available_new_tokens = model.max_seq_len - len(prompt_token_list)
    if available_new_tokens <= 0:
        raise ValueError(
            f"prompt length {len(prompt_token_list)} leaves no room for generation "
            f"within model max_seq_len {model.max_seq_len}"
        )

    max_steps = min(max_generation_tokens, available_new_tokens)
    was_training = getattr(model, "_is_training", None)
    model.eval()
    try:
        results = generate(
            model=model,
            prediction_func=GPT2ForwardFn(),
            prompt_tokens=prompt_token_list,
            max_new_tokens=max_steps,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=eos_token_id,
            show_progress=False,
            num_generations=num_generations,
        )
    finally:
        if was_training:
            model.train()

    # Keep sampled token ids directly. Decoding and re-encoding can change BPE
    # boundaries at the rendered-prompt/completion join.
    return [
        Sample(
            completion_tokens=xp.array(result.completion_tokens, dtype=xp.int32),
            completion_text=tokenizer.decode(result.completion_tokens),
            sampled_token_logprobs=xp.array(result.logprobs, dtype=xp.float32),
            reward=None,
            advantage=None,
            metadata={
                "stop_reason": result.stop_reason,
                "temperature": temperature,
                "top_k": top_k,
            },
        )
        for result in results
    ]


def main():
    pretrained_checkpoint_path = "checkpoints/sft_0428_GPT2_300"
    ckpt = load_checkpoint(
        f"{pretrained_checkpoint_path}.json",
        f"{pretrained_checkpoint_path}.npz",
    )

    TRAIN_CONFIG = GRPOTrainingConfig(
        max_steps=10,
        max_eval_steps=5,
        checkpoint_freq=10,
        report_every_steps=5,
        global_batch_size=32,
        micro_batch_size=8,
        max_grad_norm=1.0,
        model_kwargs=ckpt["model_init_kwargs"],
        optimizer_kwargs={"lr": 5e-5},
        pretrained_checkpoint_path=pretrained_checkpoint_path,
        max_generation_tokens=64,
        temperature=1.0,
        top_k=None,
        num_generations=8,
    )

    trainer = GRPOTrainer(
        model_cls=GPT2,
        optimizer_cls=Adam,
        loss_fn=grpo_loss,
        config=TRAIN_CONFIG,
    )

    bpe = BytePairEncoder(
        num_merges=12000,
        vocab_file_path="training_data/wikipedia_simpleenglish_vocab_12000.pkl",
    )
    environment = MathEnvironment()
    task = Task(task_id="1", raw_input="What is 1 + 1?", answer="2")
    rollout_generator = RolloutGenerator(TRAIN_CONFIG)
    collator = GRPOCollator(
        max_tokens=trainer.model.max_seq_len,
        pad_idx=bpe.encode("<|pad|>")[0],
    )
    report_every_steps = TRAIN_CONFIG.report_every_steps or TRAIN_CONFIG.checkpoint_freq

    with tqdm(
        total=TRAIN_CONFIG.max_steps,
        initial=trainer.global_step,
        desc="GRPO training",
    ) as progress_bar:
        while trainer.global_step < TRAIN_CONFIG.max_steps:
            step_before = trainer.global_step
            rollout_group = rollout_generator.rollout(
                model=trainer.model,
                task=task,
                tokenizer=bpe,
                environment=environment,
            )
            loss = trainer.train_step(collator([rollout_group]))

            rewards = []
            for sample in rollout_group.samples:
                if sample.reward is None:
                    raise ValueError("sample.reward must be set before logging")
                rewards.append(sample.reward)
            rewards_array = np.array(rewards, dtype=np.float32)
            progress_bar.update(trainer.global_step - step_before)
            progress_bar.set_postfix(
                loss=f"{float(xp.to_scalar(loss.data)):.4f}",
                reward_mean=f"{float(rewards_array.mean()):.3f}",
                reward_std=f"{float(rewards_array.std()):.3f}",
            )

            if trainer.global_step != step_before:
                should_validate = trainer.global_step % report_every_steps == 0
                if should_validate or trainer.global_step >= TRAIN_CONFIG.max_steps:
                    with no_grad():
                        validation_group = rollout_generator.rollout(
                            model=trainer.model,
                            task=task,
                            tokenizer=bpe,
                            environment=environment,
                        )
                    validation_rewards = []
                    for sample in validation_group.samples:
                        if sample.reward is None:
                            raise ValueError(
                                "validation sample reward must be set before logging"
                            )
                        validation_rewards.append(sample.reward)
                    validation_rewards_array = np.array(
                        validation_rewards,
                        dtype=np.float32,
                    )
                    best_sample_idx = int(np.argmax(validation_rewards_array))
                    best_sample = validation_group.samples[best_sample_idx]
                    progress_bar.write(
                        "validation "
                        f"step={trainer.global_step} "
                        f"reward_mean={float(validation_rewards_array.mean()):.3f} "
                        f"reward_max={float(validation_rewards_array.max()):.3f} "
                        f"completion={best_sample.completion_text!r}"
                    )


if __name__ == "__main__":
    main()
