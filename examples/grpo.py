import re
from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from autograd.backend import Array, ArrayLike, xp
from autograd.data.collator import Collator, pad_right_1d
from autograd.data.dataset import MapDataset
from autograd.nn import Module
from autograd.text.tokenizer import BytePairEncoder
from autograd.text.utils import generate
from autograd.tools.model import load_checkpoint
from autograd.tools.trainer import AbstractTrainer
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
CONFIG = {
    "max_generation_tokens": 32,
    "temperature": 1.0,
    "top_k": None,
    # GRPO group size G: number of completions sampled for one prompt.
    "num_generations": 2,
}

# 1. Caller code owns one Task for the first loop.
# 2. Each outer GRPO iteration refreshes rollouts for that current task.
# 3. RolloutGenerator samples group_size completions from the current
#    policy, caching old_logprobs for the behavior policy, and computes group-relative advantages
# 4. Environment scores each Sample in each RolloutGroup
#    within each RolloutGroup.
# 5. RolloutDataset stores the already-generated RolloutGroup.
# 6. DataLoader batches RolloutGroup objects and calls GRPOCollator.
# 7. GRPOCollator emits GRPOBatch, where rows = group_size for one task.
# 8. GRPOTrainer(AbstractTrainer)._compute_loss computes the GRPO objective.
# 9. AbstractTrainer.fit owns backward(), gradient accumulation, clipping,
#     optimizer.step(), checkpointing, and reporting.
#
# Keep boundaries explicit:
# - Environment: prompt rendering + reward design only
# - RolloutGenerator: model sampling + old_logprobs + RolloutGroup creation + advantage calculation
# - RolloutDataset: static container for already-generated rollout groups.
# - GRPOCollator: padding, causal shift, action masks, and GRPOBatch assembly.
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
    old_logprobs: Array  # cached behavior-policy logprobs for the policy ratio
    reward: Optional[float] = None
    advantage: Optional[float] = None  # derived field
    metadata: Optional[dict] = None  # env trace, verifier result etc...

    def __post_init__(self) -> None:
        if len(self.completion_tokens) == 0:
            raise ValueError("completion_tokens must contain at least one token")
        if len(self.completion_tokens) != len(self.old_logprobs):
            raise ValueError(
                "completion_tokens and old_logprobs must have the same length"
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
    old_logprobs: ArrayLike
    action_mask: ArrayLike
    advantages: ArrayLike
    loss_total_weight: ArrayLike


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
    prompt/completion tokens are padded, shifted, or action-masked.
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
        batch_old_logprobs = []
        batch_action_mask = []
        batch_advantages = []
        loss_total_weight = xp.array(0.0, dtype=xp.float32)

        for rollout_group in rollout_groups:
            for sample in rollout_group.samples:
                (
                    input_ids,
                    labels,
                    old_logprobs,
                    action_mask,
                    advantages,
                ) = self._build_row(rollout_group.prompt_tokens, sample)

                batch_input_ids.append(input_ids)
                batch_labels.append(labels)
                batch_old_logprobs.append(old_logprobs)
                batch_action_mask.append(action_mask)
                batch_advantages.append(advantages)

                loss_total_weight = loss_total_weight + xp.sum(action_mask)

        return GRPOBatch(
            input_ids=xp.stack(batch_input_ids, axis=0),
            labels=xp.stack(batch_labels, axis=0),
            old_logprobs=xp.stack(batch_old_logprobs, axis=0),
            action_mask=xp.stack(batch_action_mask, axis=0),
            advantages=xp.stack(batch_advantages, axis=0),
            loss_total_weight=loss_total_weight,
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
        # action_mask  = [             0,              0,                  1]
        # old_logprobs = [           0.0,            0.0,  completion_logprob_0]
        action_mask = xp.concatenate(
            [
                xp.zeros(prompt_len, dtype=xp.int32),
                xp.ones(completion_len, dtype=xp.int32),
            ],
            axis=0,
        )

        aligned_old_logprobs = xp.concatenate(
            [
                xp.zeros(prompt_len, dtype=xp.float32),
                sample.old_logprobs,
            ],
            axis=0,
        )

        if sample.advantage is None:
            raise ValueError("sample.advantage must be set before collation")

        aligned_advantages = action_mask.astype(xp.float32) * float(sample.advantage)

        # Fixed padding keeps this first GRPO collator simple. We can switch to
        # dynamic padding later if padding waste becomes a measured bottleneck.
        tokens = pad_right_1d(tokens, self.max_tokens, self.pad_idx)
        action_mask = pad_right_1d(action_mask, self.max_tokens, 0)
        aligned_old_logprobs = pad_right_1d(
            aligned_old_logprobs,
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
            aligned_old_logprobs[1:],
            action_mask[1:],
            aligned_advantages[1:],
        )


class RolloutDataset(MapDataset):
    """
    Static container for already-generated on-policy rollout groups.

    Rollout refresh is intentionally outside Dataset.on_epoch_start() for now:
    caller code should generate fresh RolloutGroups from the current policy,
    wrap them in RolloutDataset, then build a DataLoader with GRPOCollator.
    That keeps model/tokenizer/environment dependencies out of this dataset.
    """

    pass


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
        match = self.ANSWER_RE.search(sample.completion_text)
        if match is None:
            return 0.0

        reward = 0.1
        parsed_answer = match.group(1).strip()
        expected_answer = task.answer.strip()
        if parsed_answer == expected_answer:
            reward += 1.0

        return reward


class RolloutGenerator:
    """
    Samples on-policy completions from the current model.

    Given one Task, it should render the prompt via Environment, sample G
    completions, cache behavior-policy old_logprobs, ask Environment to score
    samples, and return one RolloutGroup. It should not perform optimizer work.
    """

    def rollout(
        self,
        model: Module,
        task: Task,
        tokenizer: BytePairEncoder,
        environment: Environment,
    ) -> RolloutGroup:
        task_prompt: str = environment.render_task(task)
        prompt_tokens = xp.array(tokenizer.encode(task_prompt), dtype=xp.int32)
        samples: List[Sample] = []

        for _ in range(CONFIG["num_generations"]):
            sample = generation(
                model,
                prompt_tokens=prompt_tokens,
                tokenizer=tokenizer,
                max_generation_tokens=CONFIG["max_generation_tokens"],
                temperature=CONFIG["temperature"],
                top_k=CONFIG["top_k"],
            )
            samples.append(sample)

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
    the time DataLoader yields a GRPOBatch. The base trainer can keep owning
    backward(), gradient accumulation, clipping, optimizer.step(), checkpointing,
    and reporting.

    It should not own raw tasks, Environment, or RolloutGenerator. A higher
    level orchestration loop is responsible for:

        Task -> RolloutGenerator.rollout(...) -> RolloutDataset
        -> DataLoader(..., collator=GRPOCollator(...)) -> self.fit(...)

    One optimizer step consumes one or more GRPOBatch objects depending on
    gradient accumulation. Each one-task GRPOBatch contains group_size rows.
    """

    def _compute_loss(self, batch: GRPOBatch):
        """
        Compute the GRPO objective from a trainer-facing batch.

        The loss should use current-policy logprobs from
        `self.model(batch.input_ids)`, cached `batch.old_logprobs` for the
        policy ratio, `batch.advantages` for the group-relative learning signal,
        and `batch.action_mask` so prompt/pad tokens do not contribute.
        """

        pass

    def _loss_total_weight(self, batch: GRPOBatch):
        pass

    def _evaluate(self, val_data_loader):
        pass


def generation(
    model: Module,
    prompt_tokens: Array,
    tokenizer: BytePairEncoder,
    *,
    max_generation_tokens: int,
    temperature: float,
    top_k: Optional[int] = None,
    eos_token: str = EOS_TOKEN,
) -> Sample:
    # GRPO old_logprobs must be raw-policy logprobs. With temperature=1 and no
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
        result = generate(
            model=model,
            prediction_func=GPT2ForwardFn(),
            prompt_tokens=prompt_token_list,
            max_new_tokens=max_steps,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=eos_token_id,
        )
    finally:
        if was_training:
            model.train()

    completion_array = xp.array(result.completion_tokens, dtype=xp.int32)
    # Keep sampled token ids directly. Decoding and re-encoding can change BPE
    # boundaries at the rendered-prompt/completion join.
    return Sample(
        completion_tokens=completion_array,
        completion_text=tokenizer.decode(result.completion_tokens),
        old_logprobs=xp.array(result.logprobs, dtype=xp.float32),
        reward=None,
        advantage=None,
        metadata={
            "stop_reason": result.stop_reason,
            "temperature": temperature,
            "top_k": top_k,
        },
    )


def main():
    CHECKPOINT_PATH = "checkpoints/sft_0428_GPT2_300"
    ckpt = load_checkpoint(f"{CHECKPOINT_PATH}.json", f"{CHECKPOINT_PATH}.npz")
    model = GPT2(**ckpt["model_init_kwargs"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    bpe = BytePairEncoder(
        num_merges=12000,
        vocab_file_path="training_data/wikipedia_simpleenglish_vocab_12000.pkl",
    )
    environment = MathEnvironment()
    task = Task(task_id="1", raw_input="What is 1 + 1?", answer="2")
    rollout_generator = RolloutGenerator()

    rollout_group = rollout_generator.rollout(
        model=model,
        task=task,
        tokenizer=bpe,
        environment=environment,
    )
    print(rollout_group)


if __name__ == "__main__":
    main()
