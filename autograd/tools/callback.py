from typing import Any

from autograd import nn
from autograd.text import utils as text_utils
from autograd.tools.config_schema import TransformerTrainingConfig


def teacher_forcing_callback(
    model: nn.Module,
    forward_fn: nn.AbstractLLMForwardFn,
    val_data_loader: Any,
    config: TransformerTrainingConfig,
) -> None:
    """Runs teacher-forcing qualitative evaluation when enabled in the config."""
    if config.teacher_enforcing and hasattr(val_data_loader, "data"):
        text_utils.inference(
            model=model,
            prediction_func=forward_fn,
            bpe=val_data_loader.bpe,
            groundtruth_data=val_data_loader.data[: val_data_loader.seq_len // 3],
            max_length=val_data_loader.seq_len // 3,
        )


def sampling_callback(
    model: nn.Module,
    forward_fn: nn.AbstractLLMForwardFn,
    val_data_loader: Any,
    config: TransformerTrainingConfig,
) -> None:
    """Runs free-sampling qualitative evaluation."""
    text_utils.inference(
        model=model,
        prediction_func=forward_fn,
        bpe=val_data_loader.bpe,
        start_tokens=config.eval_start_string,
        max_length=min(128, int(val_data_loader.seq_len * 0.4)),
        temperature=1.0,
        top_k=config.eval_top_k,
    )
