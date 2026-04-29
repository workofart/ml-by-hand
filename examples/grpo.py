from dataclasses import dataclass
from typing import List, Optional

from autograd.backend import Array, ArrayLike, xp
from autograd.nn import Module
from autograd.text.tokenizer import BytePairEncoder
from autograd.text.utils import generate
from autograd.tools.model import load_checkpoint
from examples.gpt_2 import GPT2, GPT2ForwardFn

SYSTEM_PROMPT = """
A conversation between User and Assistant. The user asks a question, and the Assistant solves
it. The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <think>...</think>
and <answer>...</answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>.
"""

TASK_PROMPT = "What is 1 + 1?"

CONFIG = {
    "max_tokens": 128,
    "temperature": 1.0,
    "top_k": None,
    "num_generations": 2,
}


@dataclass
class Sample:
    completion_tokens: ArrayLike
    old_logprobs: ArrayLike  # cached behavior-policy logprobs for the policy ratio
    reward: Optional[float]
    advantage: Optional[float]  # derived field
    metadata: Optional[dict]  # env trace, verifier result etc...


@dataclass
class Task:
    task_id: str
    raw_input: str
    metadata: Optional[dict]


@dataclass
class PromptGroup:
    prompt_id: str  # the dataset or prompt builder or rollout coordinate will fill this in. This should be 1-1 mapped to the rendered prompt instance, which can change if anything like the system prompt changes
    prompt_tokens: ArrayLike
    samples: List[Sample]


class Verifier:
    def __init__(self) -> None:
        pass

    def compute_reward(self, prompt_tokens: ArrayLike, completion_tokens: ArrayLike):
        # print(f"{prompt_tokens=}, {completion_tokens=}")
        return 1


def render_prompt(user_prompt: str) -> str:
    return SYSTEM_PROMPT + f"User: {user_prompt}<|endoftext|>Assistant: "


def generation(
    model: Module,
    prompt_tokens: Array,
    tokenizer: BytePairEncoder,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int] = None,
    eos_token: str = "<|endoftext|>",
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

    max_steps = min(max_new_tokens, available_new_tokens)
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
    prompt_tokens = bpe.encode(render_prompt(TASK_PROMPT))
    prompt_group = PromptGroup(prompt_id="1", prompt_tokens=prompt_tokens, samples=[])

    samples = []

    for _ in range(CONFIG["num_generations"]):
        sample = generation(
            model,
            prompt_tokens,
            bpe,
            max_new_tokens=CONFIG["max_tokens"],
            temperature=CONFIG["temperature"],
            top_k=CONFIG["top_k"],
        )
        sample.reward = verifier.compute_reward(prompt_tokens, sample.completion_tokens)
        # if sample.metadata is not None:
        #     print(sample.metadata["completion_text"])
        samples.append(sample)

    prompt_group.samples = samples


if __name__ == "__main__":
    main()
