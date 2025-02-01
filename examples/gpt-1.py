import logging
from typing import Any, Optional, Tuple

try:
    # drop-in replacement for numpy for GPU acceleration
    import cupy as np  # type: ignore

    _ = np.cuda.runtime.getDeviceCount()  # Check if a CUDA device is available
except Exception:
    import numpy as np

from autograd import functional, nn, optim
from autograd.tensor import Tensor
from autograd.text import utils as text_utils
from autograd.text.tokenizer import BytePairEncoder
from autograd.tools.data import LLMDataLoader
from autograd.tools.trainer import LLMTrainer, load_model_and_optimizer

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

    def forward(self, tokens, mask: Optional[Tensor]):
        """
        Following the same notation in the original paper
        Section 3.1 Unsupervised pre-training
        """
        batch_size, seq_len = tokens.shape
        positions = np.arange(seq_len)  # shape (seq_len,)
        positions = np.tile(positions, (batch_size, 1))  # shape (batch, seq_len)

        token_embedding = self.token_embedding(
            tokens
        )  # shape: (batch, seq_len, hidden_dim)
        position_embedding = self.position_embedding(
            positions
        )  # shape: (batch, seq_len, hidden_dim)
        h_0 = self.dropout(token_embedding + position_embedding)

        for sublayer in self.sublayers:
            h_0 = sublayer(h_0, mask)

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

    def forward(self, x: Tensor, mask: Optional[Tensor]) -> Tensor:
        x = self.add_and_norm1(
            x, lambda x_: self.multi_head_attention(x_, x_, x_, mask=mask)
        )
        x = self.add_and_norm2(x, self.feedforward)
        return x


class GPT1ForwardFn(nn.AbstractLLMForwardFn):
    def train(self, model: GPT1, batch_data: Any, **kwargs) -> Tuple[Any, Any]:
        X, dec_inp, y, src_mask, tgt_mask, causal_mask = batch_data
        logits = model(X, causal_mask)
        return logits, y

    def sample(self, model: GPT1, batch_data: Any, **kwargs) -> Tuple[Any, Any]:
        logits = model(batch_data, None)
        return logits, None


if __name__ == "__main__":
    # TODO: parse the hyperparams from CLI
    # Based on the paper
    # Section 4.1 Setup - Model Specifications
    # Note: The current hyperparameters are not optimal, they are just used
    # for overfitting the model quickly to test the model architecture and training
    # loop are free of bugs.
    CONFIG = {
        "model_kwargs": {
            "num_attention_heads": 6,  # 12
            "hidden_size": 144,  # 768, must be divisible by num_attention_heads
            "dropout_prob": 0.1,
            "max_seq_len": 128,  # 512
            "num_decoder_layers": 6,
        },
        "optimizer_kwargs": {
            "lr": 1e-3,
            "beta2": 0.99,
            "max_grad_norm": 1.0,
            "weight_decay": 0.1,
        },
        "num_epochs": 25,
        "warmup_steps": 100,
        "eval_iters": 16,
        "steps_per_epoch": 20,
        "checkpoint_freq": 2,
        "batch_size": 64,  # 64
        "label_smoothing": 0.1,
        # Whether to check the model performance by feeding the groundtruth tokens to compare whether the model can predict the next token correctly.
        "teacher_enforcing": True,
        # Whether to load from a checkpoint
        "resume_epoch": None,
        "custom_bpe": {
            "num_merges": 0,
            "npz_file_path": "training_data/bpe_0_shakespeare_encoded_data",
            "vocab_file_path": "training_data/shakespeare_vocab_0.pkl",
        },
    }

    logger = logging.getLogger(__name__)

    data = text_utils.load_shakespeare_mini()
    logger.info(f"{len(data)} characters in the entire dataset")

    # encoded_data is a list of integers without the concept of samples
    n = int(len(data) * 0.9)
    train_data, test_data = data[:n], data[n:]

    # Create a Byte Pair Encoder and prepare data
    bpe = BytePairEncoder(
        num_merges=CONFIG["custom_bpe"]["num_merges"],
        vocab_file_path=CONFIG["custom_bpe"]["vocab_file_path"],
        encoded_data_path=f"{CONFIG["custom_bpe"]["npz_file_path"]}.npz",
    )
    # Train the vocabulary on the entire dataset
    bpe.train_vocabulary(
        data,
        overwrite_saved_file=False,
    )
    data_len = len(data.split("\n\n"))
    print(f"Total length of data after split: {data_len}")
    # Override the path with training path
    bpe.encoded_data_path = f"{CONFIG["custom_bpe"]["npz_file_path"]}_train.npz"
    train_data = bpe.prepare_data(
        raw_text_list=train_data.split("\n\n"),
        overwrite_encoded_data=False,
        overwrite_vocabulary_file=False,
        split_token="<|endoftext|>",
    )
    # Override the path with test path
    bpe.encoded_data_path = f"{CONFIG["custom_bpe"]["npz_file_path"]}_test.npz"
    test_data = bpe.prepare_data(
        raw_text_list=test_data.split("\n\n"),
        overwrite_encoded_data=False,
        overwrite_vocabulary_file=False,
        split_token="<|endoftext|>",
    )

    print(f"Data length: {len(train_data)=} {len(test_data)=}")

    CONFIG["model_kwargs"]["vocab_size"] = bpe.n_vocab

    model, optimizer, checkpoint = load_model_and_optimizer(
        GPT1,
        optim.Adam,
        model_kwargs=CONFIG["model_kwargs"],
        optimizer_kwargs=CONFIG["optimizer_kwargs"],
        resume_epoch=CONFIG["resume_epoch"],
    )

    hparams = checkpoint.get("hyperparams", CONFIG)

    train_data_loader = LLMDataLoader(
        data=np.array(train_data),
        bpe=bpe,
        batch_size=hparams["batch_size"],
        seq_len=model.max_seq_len,
        steps_per_epoch=hparams["steps_per_epoch"],
        shuffle=True,
        include_decoder_input=False,
        create_decoder_inp=False,
        create_masks=False,
    )
    test_data_loader = LLMDataLoader(
        data=np.array(test_data),
        bpe=bpe,
        batch_size=hparams["batch_size"] // 2,
        seq_len=model.max_seq_len,
        steps_per_epoch=hparams["eval_iters"],
        shuffle=False,
        include_decoder_input=False,
        create_decoder_inp=False,
        create_masks=False,
    )

    trainer = LLMTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=functional.cross_entropy,
        epochs=hparams["num_epochs"],
        warmup_steps=hparams["warmup_steps"],
        label_smoothing=hparams["label_smoothing"],
        checkpoint_freq=hparams["checkpoint_freq"],
        forward_fn=GPT1ForwardFn(),
        teacher_enforcing=hparams["teacher_enforcing"],
        hyperparams=hparams,
        checkpoint=checkpoint,
        start_tokens="First",
    )

    trainer.fit(train_data_loader, test_data_loader)

    text_utils.inference(
        model=model,
        prediction_func=GPT1ForwardFn(),
        bpe=bpe,
        start_tokens="All",  # Dummy token to start the generation
        max_length=int(model.max_seq_len * 0.9),  # this should be shorter than context
        temperature=1.0,
        top_k=10,
    )
