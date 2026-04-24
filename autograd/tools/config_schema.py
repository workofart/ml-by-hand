"""
This module contains the schema for defining various configs for the training pipeline.
It's optional to use, but may provide some quality of life improvements
"""

from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class GenericTrainingConfig:
    """
    Generic Configuration for training a model.
    This follows the same interface as the AbstractTrainer class in `autograd.tools.trainer.py`
    """

    checkpoint_freq: int
    # When we load from checkpoint, the below kwargs are restored from the checkpoint
    # and cannot be modified because they are inherently tied to the model and optimizer
    # state (e.g. hidden_size cannot be changed because the model artifact was created
    # using the previous checkpoint's hidden_size)
    # The above configs can be changed, and don't need to be loaded from the checkpoint
    model_kwargs: dict
    optimizer_kwargs: dict
    max_epochs: Optional[int] = None
    max_steps: Optional[int] = None
    max_eval_steps: Optional[int] = None
    # Controls how often step-mode training reports/logs/evaluates.
    # This is intentionally separate from checkpoint_freq: reporting cadence
    # should not imply checkpoint cadence.
    #
    # Important: this is not another batch-size knob. micro_batch_size controls
    # per-forward/backward memory, and global_batch_size controls gradient
    # accumulation / optimizer-update cadence. report_every_steps only controls
    # how long trainer-side reporting state can stay unsynchronized before we
    # scalarize/log it.
    #
    # On MLX, larger values can retain deferred metric/loss accumulation work
    # for longer before a host sync point, which may show up as bursty sync or
    # allocator pressure even though the effective training batch size is
    # unchanged.
    report_every_steps: Optional[int] = None
    # Whether to load from a checkpoint
    resume_epoch: Optional[int] = None
    # Optional checkpoint basename (without .json/.npz) used to initialize
    # model weights before training. The configured architecture must match
    # the checkpoint; mismatches should fail during load_state_dict.
    pretrained_checkpoint_path: Optional[str] = None
    training_run_name: str = "default"
    dataset_name: Optional[str] = ""
    # Effective batch size after gradient accumulation.
    global_batch_size: int = 1
    # Per-forward/backward batch size that must fit in memory.
    micro_batch_size: int = 1
    # Optional trainer-level gradient clipping threshold.
    max_grad_norm: Optional[float] = None

    def __post_init__(self) -> None:
        if self.max_epochs is None and self.max_steps is None:
            raise ValueError("At least one of max_epochs or max_steps must be set")
        if self.max_epochs is not None and self.max_steps is not None:
            raise ValueError("max_epochs and max_steps are mutually exclusive")
        if self.max_epochs is not None and self.max_epochs < 1:
            raise ValueError(f"max_epochs must be >= 1, got {self.max_epochs}")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")
        if self.max_eval_steps is not None and self.max_eval_steps < 1:
            raise ValueError(f"max_eval_steps must be >= 1, got {self.max_eval_steps}")
        if self.report_every_steps is not None:
            if not isinstance(self.report_every_steps, int):
                raise ValueError(
                    "report_every_steps must be an int, "
                    f"got {self.report_every_steps!r}"
                )
            if self.report_every_steps < 1:
                raise ValueError(
                    f"report_every_steps must be >= 1, got {self.report_every_steps}"
                )
        if not isinstance(self.checkpoint_freq, int):
            raise ValueError(
                f"checkpoint_freq must be an int, got {self.checkpoint_freq!r}"
            )
        if self.checkpoint_freq <= 0:
            raise ValueError(f"checkpoint_freq must be > 0, got {self.checkpoint_freq}")
        if self.global_batch_size < 1:
            raise ValueError(
                f"global_batch_size must be >= 1, got {self.global_batch_size}"
            )
        if self.micro_batch_size < 1:
            raise ValueError(
                f"micro_batch_size must be >= 1, got {self.micro_batch_size}"
            )
        if self.global_batch_size % self.micro_batch_size != 0:
            raise ValueError(
                "global_batch_size must be divisible by micro_batch_size, "
                f"got {self.global_batch_size} and {self.micro_batch_size}"
            )
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be > 0, got {self.max_grad_norm}")
        self._validate_checkpoint_accumulation_alignment()

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.global_batch_size // self.micro_batch_size

    def _validate_checkpoint_accumulation_alignment(self) -> None:
        if self.max_steps is None:
            return

        accumulation_steps = self.gradient_accumulation_steps
        if accumulation_steps == 1 or self.checkpoint_freq > self.max_steps:
            return

        if self.checkpoint_freq % accumulation_steps != 0:
            raise ValueError(
                "checkpoint_freq must be divisible by gradient_accumulation_steps "
                "when step-mode checkpointing uses gradient accumulation, "
                f"got checkpoint_freq={self.checkpoint_freq}, "
                f"gradient_accumulation_steps={accumulation_steps}. "
                "Otherwise a checkpoint can be written before optimizer.step() "
                "has materialized optimizer state for the accumulated gradients. "
                "Use a checkpoint_freq that is a multiple of "
                "gradient_accumulation_steps; if you need a final checkpoint, "
                "max_steps should land on the same boundary."
            )


@dataclass
class CustomBpeConfig:
    """
    The configs for our custom BytePairEncoder class in `autograd.text.tokenizer.py`
    """

    num_merges: int
    encoded_data_path: str
    vocab_path: str
    overwrite_encoded_data: bool
    overwrite_vocabulary_file: bool
    split_token: str


@dataclass(kw_only=True)
class TransformerTrainingConfig(GenericTrainingConfig):
    # Whether to check the model performance by feeding the groundtruth tokens to compare whether the model can predict the next token correctly.
    teacher_forcing: bool
    custom_bpe: Union[CustomBpeConfig, None] = None
    label_smoothing: float  # TODO: refactor this into loss function config
    eval_start_string: Optional[str] = (
        "\n"  # starting token for the evaluation during training
    )
    eval_top_k: Optional[int] = None
