from typing import Optional

from autograd import nn
from autograd.backend import Array
from autograd.text import utils as text_utils
from autograd.text.tokenizer import BytePairEncoder


# TODO: Convert these qualitative inference helpers into real trainer lifecycle
# callbacks once the callback boundary is redesigned. For now they are invoked
# manually from examples and do not participate in trainer evaluation.
def run_teacher_forcing_inference(
    model: nn.Module,
    forward_fn: nn.AbstractLLMForwardFn,
    bpe: BytePairEncoder,
    groundtruth_data: Array,
    max_length: int,
) -> str:
    """Runs teacher-forcing qualitative inference with explicit context."""
    return text_utils.inference(
        model=model,
        prediction_func=forward_fn,
        bpe=bpe,
        groundtruth_data=groundtruth_data,
        max_length=max_length,
    )


def run_sampling_inference(
    model: nn.Module,
    forward_fn: nn.AbstractLLMForwardFn,
    bpe: BytePairEncoder,
    start_tokens: Optional[str],
    max_length: int,
    top_k: Optional[int] = None,
) -> str:
    """Runs free-sampling qualitative inference with explicit context."""
    was_training = getattr(model, "_is_training", None)
    model.eval()
    try:
        return text_utils.inference(
            model=model,
            prediction_func=forward_fn,
            bpe=bpe,
            start_tokens=start_tokens,
            max_length=max_length,
            temperature=1.0,
            top_k=top_k,
        )
    finally:
        if was_training:
            model.train()
