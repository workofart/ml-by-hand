from dataclasses import dataclass
from typing import List, Optional, Sequence

from autograd.backend import Array, ArrayLike, xp
from autograd.data.collator import Collator, pad_right_1d
from autograd.nn import Module
from autograd.text.tokenizer import BytePairEncoder
from autograd.text.utils import generate
from autograd.tools.model import load_checkpoint
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

TASK_PROMPT = "What is 1 + 1?"
EOS_TOKEN = "<|endoftext|>"
CONFIG = {
    "max_generation_tokens": 128,
    "temperature": 1.0,
    "top_k": None,
    "num_generations": 2,
}


@dataclass
class Sample:
    completion_tokens: Array
    old_logprobs: Array  # cached behavior-policy logprobs for the policy ratio
    reward: Optional[float]
    advantage: Optional[float]  # derived field
    metadata: Optional[dict]  # env trace, verifier result etc...

    def __post_init__(self) -> None:
        if len(self.completion_tokens) == 0:
            raise ValueError("completion_tokens must contain at least one token")
        if len(self.completion_tokens) != len(self.old_logprobs):
            raise ValueError(
                "completion_tokens and old_logprobs must have the same length"
            )


@dataclass
class Task:
    task_id: str
    raw_input: str
    metadata: Optional[dict]


@dataclass
class PromptGroup:
    prompt_id: str  # the dataset or prompt builder or rollout coordinate will fill this in. This should be 1-1 mapped to the rendered prompt instance, which can change if anything like the system prompt changes
    prompt_tokens: Array
    samples: List[Sample]

    def __post_init__(self) -> None:
        if len(self.prompt_tokens) == 0:
            raise ValueError("prompt_tokens must contain at least one token")
        if len(self.samples) == 0:
            raise ValueError("PromptGroup must contain at least one sample")


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
    Builds fixed-length GRPO batches from prompt groups.

    `max_tokens` is the total row length before causal shifting:
    `len(prompt_tokens) + len(completion_tokens)`. It is not the rollout
    generation limit, which only caps completion length.
    """

    def __init__(self, max_tokens: int, pad_idx: int) -> None:
        if max_tokens < 2:
            raise ValueError(
                "max_tokens must be >= 2 for GRPO, since this is autoregressive"
            )
        self.max_tokens = max_tokens
        self.pad_idx = pad_idx

    def __call__(self, prompt_groups: Sequence[PromptGroup]) -> GRPOBatch:
        batch_input_ids = []
        batch_labels = []
        batch_old_logprobs = []
        batch_action_mask = []
        batch_advantages = []
        loss_total_weight = xp.array(0.0, dtype=xp.float32)

        for prompt_group in prompt_groups:
            for sample in prompt_group.samples:
                (
                    input_ids,
                    labels,
                    old_logprobs,
                    action_mask,
                    advantages,
                ) = self._build_row(prompt_group.prompt_tokens, sample)

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


class Verifier:
    def compute_reward(
        self, prompt_tokens: ArrayLike, completion_tokens: ArrayLike
    ) -> float:
        # print(f"{prompt_tokens=}, {completion_tokens=}")
        return 1.0


def render_prompt(user_prompt: str) -> str:
    return SYSTEM_PROMPT + f"User: {user_prompt}{EOS_TOKEN}Assistant: "


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
        old_logprobs=xp.array(result.logprobs, dtype=xp.float32),
        reward=None,
        advantage=None,
        metadata={
            "completion_text": tokenizer.decode(result.completion_tokens),
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
    verifier = Verifier()

    # PromptGroup stores the exact rendered prompt tokens used for generation,
    # not just the raw task text.
    prompt_tokens = xp.array(bpe.encode(render_prompt(TASK_PROMPT)), dtype=xp.int32)

    samples = []

    for _ in range(CONFIG["num_generations"]):
        sample = generation(
            model,
            prompt_tokens,
            bpe,
            max_generation_tokens=CONFIG["max_generation_tokens"],
            temperature=CONFIG["temperature"],
            top_k=CONFIG["top_k"],
        )
        sample.reward = verifier.compute_reward(prompt_tokens, sample.completion_tokens)
        # if sample.metadata is not None:
        #     print(sample.metadata["completion_text"])
        samples.append(sample)

    prompt_group = PromptGroup(
        prompt_id="1", prompt_tokens=prompt_tokens, samples=samples
    )
    print(prompt_group)


if __name__ == "__main__":
    main()
