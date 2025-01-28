import logging
from typing import Optional

try:
    # drop-in replacement for numpy for GPU acceleration
    import cupy as np  # type: ignore
except ImportError:
    import numpy as np

from autograd import functional, nn, optim
from autograd.tensor import Tensor
from autograd.text import utils as text_utils
from autograd.text.tokenizer import BytePairEncoder
from autograd.tools.data import LLMDataLoader, load_data
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


if __name__ == "__main__":

    def gpt_1_forward(model, batch_or_tokens, mode="train"):
        if mode == "train":
            X, dec_inp, y, src_mask, tgt_mask, causal_mask = batch_or_tokens
            logits = model(X, causal_mask)
            return logits, y
        elif mode == "sample":
            tokens = batch_or_tokens
            logits = model(tokens, None)
            return logits
        else:
            raise ValueError(f"Unknown mode {mode}, must be 'train' or 'sample'")

    logger = logging.getLogger(__name__)

    data = load_data(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "examples/tinyshakespeare.txt",
    )
    logger.info(f"{len(data)} characters in the entire dataset")

    # Create the vocabulary first
    bpe = BytePairEncoder(num_merges=3000, vocab_file_path="vocab.pkl")
    encoded_data = bpe.prepare_data(
        raw_text_list=data.split("\n\n"),
        npz_file_path="bpe_mini_shakespeare.npz",
        overwrite_saved_file=False,
        split_token="<|endoftext|>",
    )[:50000]

    # encoded_data is a list of integers without the concept of samples
    n = int(len(encoded_data) * 0.9)
    train_data, test_data = encoded_data[:n], encoded_data[n:]

    # TODO: parse the hyperparams from CLI
    # Based on the paper
    # Section 4.1 Setup - Model Specifications
    # Note: The current hyperparameters are not optimal, they are just used
    # for overfitting the model quickly to test the model architecture and training
    # loop are free of bugs.
    CONFIG = {
        "model_kwargs": {
            "vocab_size": len(bpe._unicode_to_int_vocab),
            "num_attention_heads": 6,  # 12
            "hidden_size": 144,  # 768, must be divisible by num_attention_heads
            "dropout_prob": 0.1,
            "max_seq_len": 128,  # 512
            "num_decoder_layers": 6,
        },
        "optimizer_kwargs": {
            "lr": 0.0  # We may schedule it later with warmup
        },
        "num_epochs": 90,
        "warmup_steps": 100,
        "eval_iters": 16,
        "batch_size": 64,  # 64
        # Whether to check the model performance by feeding the groundtruth tokens to compare whether the model can predict the next token correctly.
        "teacher_enforcing": True,
        # Whether to load from a checkpoint
        "resume_epoch": None,
    }

    model, optimizer, checkpoint = load_model_and_optimizer(
        GPT1,
        optim.Adam,
        model_kwargs=CONFIG["model_kwargs"],
        optimizer_kwargs=CONFIG["optimizer_kwargs"],
        resume_epoch=CONFIG["resume_epoch"],
    )

    hparams = checkpoint.get("hyperparams", CONFIG)

    train_data_loader = LLMDataLoader(
        data=train_data,
        vocab=bpe._unicode_to_int_vocab,
        batch_size=hparams["batch_size"],
        seq_len=model.max_seq_len,
        shuffle=True,
        include_decoder_input=False,
    )
    test_data_loader = LLMDataLoader(
        data=test_data,
        vocab=bpe._unicode_to_int_vocab,
        batch_size=hparams["batch_size"] // 4,
        seq_len=model.max_seq_len,
        shuffle=False,
        include_decoder_input=False,
    )

    trainer = LLMTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=functional.cross_entropy,
        epochs=hparams["num_epochs"],
        warmup_steps=hparams["warmup_steps"],
        label_smoothing=0.1,
        checkpoint_freq=1,
        forward_fn=gpt_1_forward,
        tokenizer=bpe,
        teacher_enforcing=hparams["teacher_enforcing"],
        hyperparams=hparams,
        checkpoint=checkpoint,
    )

    trainer.fit(train_data_loader, test_data_loader)

    text_utils.inference(
        prediction_func=lambda seq_so_far: gpt_1_forward(
            model, seq_so_far, mode="sample"
        ),
        bpe=bpe,
        start_tokens=["All"],  # Dummy token to start the generation
        max_length=int(model.max_seq_len * 0.9),  # this should be shorter than context
        temperature=1.0,
        top_k=10,
    )
