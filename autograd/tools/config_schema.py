"""
This module contains the schema for defining various configs for the training pipeline.
It's optional to use, but may provide some quality of life improvements
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GenericTrainingConfig:
    """
    Generic Configuration for training a model.
    This follows the same interface as the AbstractTrainer class in `autograd.tools.trainer.py`
    """

    total_epochs: int
    checkpoint_freq: float
    # When we load from checkpoint, the below kwargs are restored from the checkpoint
    # and cannot be modified because they are inherently tied to the model and optimizer
    # state (e.g. hidden_size cannot be changed because the model artifact was created
    # using the previous checkpoint's hidden_size)
    # The above configs can be changed, and don't need to be loaded from the checkpoint
    model_kwargs: dict
    optimizer_kwargs: dict
    steps_per_epoch: Optional[int] = 16
    eval_iters: Optional[int] = 16
    batch_size: Optional[int] = 32
    # Whether to load from a checkpoint
    resume_epoch: Optional[int] = None
    training_run_name: Optional[str] = "default"
    dataset_name: Optional[str] = ""


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
    teacher_enforcing: bool
    custom_bpe: Optional[CustomBpeConfig] = None
    # If True, we create a separate 'dec_inp'
    # array (common in seq2seq). If you just want normal GPT next-token,
    # you can set this false.
    include_decoder_input: bool
    # If True, we create a padding for cases where
    # we have sequences of different lengths across the training samples.
    # If you're doing standard GPT, you'd typically want a causal mask, which is created
    # by default, and isn't controlled by this flag.
    create_padding_masks: bool
    label_smoothing: float  # TODO: refactor this into loss function config
    eval_start_string: Optional[str] = (
        "\n"  # starting token for the evaluation during training
    )
