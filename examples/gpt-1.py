from typing import Optional
from autograd.tensor import Tensor
from autograd import nn, functional, optim

# TODO: Extract the common modules out to nn.py module
from examples.transformers import (
    ResidualAddAndNorm,
    FeedForward,
    Transformer,
)
from autograd.text.tokenizer import BytePairEncoder
from autograd.tools.data import load_data, LLMDataLoader
from autograd.tools.trainer import LLMTrainer
import logging
import os
import numpy as np


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


def gpt_1_forward(model: Transformer, batch_data):
    X, dec_inp, y, src_mask, tgt_mask, causal_mask = batch_data
    pred_probs = model(X, causal_mask)
    # Suppose the labels are inside X or we have a separate label array?
    # Return (pred_probs, y). If needed, we might store y in batch_data[-1].
    return pred_probs, y


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    # Based on the paper
    # Section 4.1 Setup - Model Specifications
    # Note: The current hyperparameters are not optimal, they are just used
    # for overfitting the model quickly to test the model architecture and training
    # loop are free of bugs.
    # TODO: parse the hyperparams from CLI
    HYPERPARAMS = {
        "num_epochs": 90,
        "warmup_steps": 100,
        "num_attention_heads": 6,  # 12
        "d_model": 144,  # 768, must be divisible by num_attention_heads
        "batch_size": 64,  # 64
        "dropout_prob": 0.1,
        "seq_len": 128,  # 512
        "num_decoder_layers": 6,
        "eval_iters": 16,
    }
    # Whether to check the model performance by feeding the groundtruth tokens to compare whether the model can predict the next token correctly.
    teacher_enforcing = True

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filename = "examples/tinyshakespeare.txt"

    data = load_data(url, filename)
    logger.info(f"{len(data)} characters in the entire dataset")

    # Create the vocabulary first
    bpe = BytePairEncoder(num_merges=3000, vocab_file_path="vocab.pkl")
    vocab, idx2word = bpe.train_vocabulary(data, overwrite_saved_file=False)

    # Now encode the subset of data
    logger.info("Encoding the new data...")
    data = data.split("\n\n")

    if os.path.exists("bpe_mini_shakespeare.npz"):
        logger.info("Found existing encoded data, loading it...")
        with np.load("bpe_mini_shakespeare.npz", allow_pickle=True) as npz_data:
            encoded_data = npz_data.get("arr_0")[:50000]
    else:
        logger.info("Encoding the new data...")
        encoded_data = np.array(bpe.encode("<|endoftext|>".join(data)))
        np.savez_compressed("bpe_mini_shakespeare.npz", encoded_data)
        logger.info("Saved encoded data to bpe_mini_shakespeare.npz")

    logger.info(
        f"Vocabulary size: {len(vocab)}, encoded_data length: {len(encoded_data)}"
    )
    logger.info(f"Data: {data[:3]}, Encoded_Data: {encoded_data[:50]}")

    # encoded_data is a list of integers without the concept of samples
    n = int(len(encoded_data) * 0.9)
    train_data, test_data = encoded_data[:n], encoded_data[n:]

    model = GPT1(
        vocab_size=len(vocab),
        hidden_size=HYPERPARAMS["d_model"],
        num_attention_heads=HYPERPARAMS["num_attention_heads"],
        dropout_prob=HYPERPARAMS["dropout_prob"],
        max_seq_len=int(HYPERPARAMS["seq_len"] * 1.1),
        num_decoder_layers=HYPERPARAMS["num_decoder_layers"],
    )
    model.train()
    logger.info(f"Model parameters: {model.num_parameters()}")

    optimizer = optim.Adam(model.parameters, lr=0)
    train_data_loader = LLMDataLoader(
        train_data,
        vocab,
        batch_size=HYPERPARAMS["batch_size"],
        seq_len=HYPERPARAMS["seq_len"],
        shuffle=True,
        include_decoder_input=False,
    )
    test_data_loader = LLMDataLoader(
        test_data,
        vocab,
        batch_size=HYPERPARAMS["batch_size"] // 4,
        seq_len=HYPERPARAMS["seq_len"],
        shuffle=False,
        include_decoder_input=False,
    )

    trainer = LLMTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=functional.cross_entropy,
        epochs=HYPERPARAMS["num_epochs"],
        warmup_steps=HYPERPARAMS["warmup_steps"],
        label_smoothing=0.1,
        checkpoint_freq=1,
        forward_fn=gpt_1_forward,
    )

    trainer.fit(train_data_loader, test_data_loader)

    # TODO: Remove the following after integrating the following with LLMTrainer
    #         if teacher_enforcing:
    #             text_utils.teacher_forcing_inference(
    #                 lambda x: model(
    #                     x,
    #                     text_utils.create_causal_mask(seq_len=x.shape[1], batch_size=1),
    #                 ),  # shape: (1, seq_len, vocab_size)
    #                 bpe,
    #                 train_data[: HYPERPARAMS["seq_len"]],
    #                 vocab_idx2word=idx2word,
    #             )
    #         else:
    #             text_utils.inference(
    #                 lambda x: model(
    #                     x,
    #                     text_utils.create_causal_mask(seq_len=x.shape[1], batch_size=1),
    #                 ),  # shape: (1, seq_len, vocab_size)
    #                 bpe,
    #                 start_tokens=["All"],  # Dummy token to start the generation
    #                 max_length=int(HYPERPARAMS["seq_len"] * 1.1),
    #                 temperature=1.0,
    #                 top_k=10,
    #             )

    #         model.train()
