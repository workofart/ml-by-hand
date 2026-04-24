# ruff: noqa: E402

import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autograd import functional, nn, optim
from autograd.backend import xp
from autograd.data.collator import CausalLMWindowCollator
from autograd.data.data_loader import DataLoader
from autograd.data.dataset import TokenWindowDataset
from autograd.data.types import CausalLMBatch
from autograd.tensor import Tensor
from autograd.text import utils as text_utils
from autograd.text.tokenizer import BytePairEncoder
from autograd.tools.callback import (
    run_sampling_inference,
    run_teacher_forcing_inference,
)
from autograd.tools.config_schema import CustomBpeConfig, TransformerTrainingConfig
from autograd.tools.trainer import LLMTrainer

# TODO: Extract the common modules out to nn.py module
from examples.transformers import (
    FeedForward,
    ResidualAddAndNorm,
)


class GPT1(nn.Module):
    """
    GPT-1
    Paper: Improving Language Understanding by Generative Pre-Training
    https://www.semanticscholar.org/paper/Improving-Language-Understanding-by-Generative-Radford-Narasimhan/cd18800a0fe0b668a1cc19f2ec95b5003d0a5035
    """

    def __init__(
        self,
        vocab_size,
        hidden_size,
        num_attention_heads,
        max_seq_len,
        dropout_prob,
        num_decoder_layers,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.sublayers = nn.ModuleList(
            [
                DecoderSublayer(
                    hidden_size=hidden_size,
                    ff_hidden_size=hidden_size * 4,
                    num_attention_heads=num_attention_heads,
                    dropout_prob=dropout_prob,
                )
                for _ in range(num_decoder_layers)
            ]
        )

    def forward(self, tokens):
        """
        Following the same notation in the original paper
        Section 3.1 Unsupervised pre-training
        """
        batch_size, seq_len = tokens.shape
        positions = xp.arange(seq_len, dtype=xp.int32)  # shape (seq_len,)
        positions = xp.tile(positions, (batch_size, 1))  # shape (batch, seq_len)

        token_embedding = self.token_embedding(
            tokens
        )  # shape: (batch, seq_len, hidden_dim)
        position_embedding = self.position_embedding(
            positions
        )  # shape: (batch, seq_len, hidden_dim)
        h_0 = self.dropout(token_embedding + position_embedding)

        for sublayer in self.sublayers:
            h_0 = sublayer(h_0)

        output = self.layer_norm(h_0)
        output = output @ self.token_embedding.parameters["weight"].T
        return output


class DecoderSublayer(nn.Module):
    def __init__(
        self,
        hidden_size=512,
        ff_hidden_size=2048,
        dropout_prob=0.1,
        num_attention_heads=2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.multi_head_attention = nn.MultiHeadAttention(
            hidden_size=hidden_size, num_heads=num_attention_heads
        )
        self.add_and_norm1 = ResidualAddAndNorm(hidden_size)
        self.add_and_norm2 = ResidualAddAndNorm(hidden_size)
        self.feedforward = FeedForward(
            fc_input_size=hidden_size,
            hidden_size=ff_hidden_size,
            dropout_prob=dropout_prob,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.add_and_norm1(
            x, lambda x_: self.multi_head_attention(x_, x_, x_, is_causal=True)
        )
        x = self.add_and_norm2(x, self.feedforward)
        return x


class GPT1ForwardFn(nn.AbstractLLMForwardFn):
    def train(self, model: GPT1, batch_data: CausalLMBatch):
        return model(batch_data.input_ids)

    def sample(self, model: GPT1, batch_data: Any):
        return model(batch_data)


if __name__ == "__main__":
    # TODO: parse the hyperparams from CLI
    # Based on the paper
    # Section 4.1 Setup - Model Specifications
    # Note: The current hyperparameters are not optimal, they are just used
    # for overfitting the model quickly to test the model architecture and training
    # loop are free of bugs.
    train_global_batch_size = 64
    train_micro_batch_size = train_global_batch_size
    CONFIG = TransformerTrainingConfig(
        training_run_name="shakespeare_mini",
        dataset_name="shakespeare_mini",
        max_steps=500,
        max_eval_steps=16,
        checkpoint_freq=2,
        global_batch_size=train_global_batch_size,
        micro_batch_size=train_micro_batch_size,
        max_grad_norm=1.0,
        model_kwargs={
            "num_attention_heads": 6,
            "hidden_size": 144,
            "dropout_prob": 0.1,
            "max_seq_len": 128,
            "num_decoder_layers": 6,
        },
        optimizer_kwargs={
            "lr": 1e-3,
            "beta2": 0.99,
            "weight_decay": 0.1,
            "lr_scheduler_kwargs": {
                "lr_scheduler_cls": optim.CosineScheduler,
                "warmup_steps": 100,
                "lr_decay_iters": 500,
            },
        },
        resume_epoch=None,
        teacher_forcing=True,
        label_smoothing=0.1,
        eval_start_string="First",
        custom_bpe=CustomBpeConfig(
            num_merges=0,
            encoded_data_path="training_data/bpe_0_shakespeare_encoded_data.npz",
            vocab_path="training_data/shakespeare_vocab_0.pkl",
            overwrite_encoded_data=False,
            overwrite_vocabulary_file=False,
            split_token="<|endoftext|>",
        ),
    )

    logger = logging.getLogger(__name__)
    data = text_utils.load_shakespeare_mini()

    # Create a Byte Pair Encoder and prepare data
    if CONFIG.custom_bpe:
        bpe = BytePairEncoder(
            num_merges=CONFIG.custom_bpe.num_merges,
            vocab_file_path=CONFIG.custom_bpe.vocab_path,
            encoded_data_path=CONFIG.custom_bpe.encoded_data_path,
        )

        encoded_data = bpe.prepare_data(
            raw_text=data,
            overwrite_encoded_data=CONFIG.custom_bpe.overwrite_encoded_data,
            overwrite_vocabulary_file=CONFIG.custom_bpe.overwrite_vocabulary_file,
            split_token=CONFIG.custom_bpe.split_token,
        )
    else:
        raise ValueError(
            "Currently this GPT-1 model can only be trained with the custom BytePairEncoder, please specify the custom_bpe config"
        )

    # encoded_data is a list of integers without the concept of samples
    n = int(len(encoded_data) * 0.9)
    train_data, test_data = encoded_data[:n], encoded_data[n:]
    print(f"Data length: {len(train_data)=} {len(test_data)=}")

    CONFIG.model_kwargs["vocab_size"] = bpe.n_vocab

    trainer = LLMTrainer(
        model_cls=GPT1,
        optimizer_cls=optim.Adam,
        loss_fn=functional.cross_entropy,
        config=CONFIG,
        forward_fn=GPT1ForwardFn(),
    )

    train_data_loader = DataLoader(
        dataset=TokenWindowDataset(
            data=xp.array(train_data, dtype=xp.int32),
            # CausalLMWindowCollator shifts one token to build input_ids/labels,
            # so a length-T model context needs a raw window of length T + 1.
            window_len=trainer.model.max_seq_len + 1,
            sampling="random",
        ),
        batch_size=train_micro_batch_size,
        collate_fn=CausalLMWindowCollator(),
    )
    test_data_loader = DataLoader(
        dataset=TokenWindowDataset(
            data=xp.array(test_data, dtype=xp.int32),
            # CausalLMWindowCollator shifts one token to build input_ids/labels,
            # so a length-T model context needs a raw window of length T + 1.
            window_len=trainer.model.max_seq_len + 1,
            sampling="sequential",
        ),
        batch_size=max(1, train_micro_batch_size // 2),
        collate_fn=CausalLMWindowCollator(),
    )

    trainer.fit(train_data_loader, test_data_loader)

    if CONFIG.teacher_forcing:
        run_teacher_forcing_inference(
            model=trainer.model,
            forward_fn=GPT1ForwardFn(),
            bpe=bpe,
            groundtruth_data=xp.array(
                test_data[: trainer.model.max_seq_len // 3], dtype=xp.int32
            ),
            max_length=trainer.model.max_seq_len // 3,
        )

    run_sampling_inference(
        model=trainer.model,
        forward_fn=GPT1ForwardFn(),
        bpe=bpe,
        start_tokens=CONFIG.eval_start_string,
        max_length=int(trainer.model.max_seq_len * 0.9),
        top_k=CONFIG.eval_top_k,
    )
