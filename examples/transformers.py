import numpy as np

from autograd.tools.data import load_data, LLMDataLoader
from autograd.tools.model import load_model
from autograd.tools.trainer import LLMTrainer
from autograd.text import tokenizer
from autograd.tensor import Tensor
from autograd import nn, functional, optim
import logging
import os
from typing import Optional, Any


class Transformer(nn.Module):
    """
    Implements the Paper "Attention is All You Need"
    Paper: https://arxiv.org/abs/1706.03762

    More specifically, it uses a encoder-decoder architecture with attention mechanisms embedded inside.

    Overall Architecture:
    - Encoder-decoder framework (Section 3.1)
    - Encoder: 6 identical layers (Section 3.1)
    - Decoder: 6 identical layers (Section 3.1)
    - Each layer has self-attention or masked self-attention in the decoder (Section 3.2)
      and feed-forward sub-layers
    """

    def __init__(
        self, vocab_size: int, hidden_size: int, num_attention_heads: int, **kwargs: Any
    ) -> None:
        """
        Args:
            vocab_size (int): Vocabulary size for the embeddings
            hidden_size (int): Dimension of model (d_model in the paper)
            num_attention_heads (int): Number of attention heads (Section 3.2.2)
        """
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # Encoder and dedcoder each have 6 layers
        self.encoder = Encoder(
            embedding_size=hidden_size, num_attention_heads=num_attention_heads
        )
        self.decoder = Decoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
        )

    def forward(
        self,
        source: Tensor,
        target: Tensor,
        source_mask: Optional[Tensor] = None,
        target_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass of the transformer

        Args:
            source (Tensor): Source sequence indices (batch_size, seq_len)
            target (Tensor): Target sequence indices (batch_size, seq_len)
            source_mask (Tensor, optional): Source mask to cover padding. Defaults to None.
            target_mask (Tensor, optional): Target mask to cover padding + future tokens (causal mask). Defaults to None.

        Returns:
            Tensor: Probability distribution over entire vocabulary
        """
        encoder_output = self.encoder(
            source, embedding_layer=self.embedding, mask=source_mask
        )
        output = self.decoder(
            target,
            embedding_layer=self.embedding,
            encoder_output=encoder_output,
            source_mask=source_mask,
            target_mask=target_mask,
        )
        return output


class Encoder(nn.Module):
    """
    - Embeddings (Section 3.1) + Positional Encoding (Section 3.5)
    - 6 identical EncoderSublayer (Section 3.1)
    """

    def __init__(self, embedding_size: int, num_attention_heads: int) -> None:
        super().__init__()
        self.embedding_size = embedding_size

        self.positional_encoder = PositionalEncoding(hidden_size=embedding_size)

        self.sublayers = nn.ModuleList(
            [
                EncoderSublayer(
                    hidden_size=embedding_size,
                    ff_hidden_size=embedding_size * 4,
                    num_attention_heads=num_attention_heads,
                )
                for _ in range(6)
            ]
        )
        self.layer_norm = nn.LayerNorm(embedding_size)

    def forward(
        self, x: Tensor, embedding_layer: nn.Module, mask: Optional[Tensor] = None
    ) -> Tensor:
        # Section 3.4 (Embedding Scaling) embedding layer is shared between Encoder and Decoder
        x = embedding_layer(x) * Tensor(self.embedding_size).sqrt()

        # Section 3.5
        x = self.positional_encoder(x)

        # Section 3.1
        for sublayer in self.sublayers:
            x = sublayer(x, mask)
        return self.layer_norm(x)


class Decoder(nn.Module):
    """
    Decoder (Section 3.1)
    - Embedding + PostionalEncoding (Section 3.5)
    - 6 identical DecoderSublayer (Section 3.1)
    - Linear + Softmax to produce probability distribution over entire vocabulary
    """

    def __init__(
        self, vocab_size: int, hidden_size: int, num_attention_heads: int = 2
    ) -> None:
        super().__init__()
        self.positional_encoder = PositionalEncoding(hidden_size=hidden_size)
        self.hidden_size = hidden_size

        self.sublayers = nn.ModuleList(
            [
                DecoderSublayer(
                    hidden_size=hidden_size,
                    ff_hidden_size=hidden_size * 4,
                    num_attention_heads=num_attention_heads,
                )
                for _ in range(6)
            ]
        )
        self.linear = nn.Linear(hidden_size, output_size=vocab_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x: Tensor,
        embedding_layer: nn.Module,
        encoder_output: Tensor,
        source_mask: Optional[Tensor],
        target_mask: Optional[Tensor],
    ) -> Tensor:
        """
        Args:
            x (Tensor): Target sequence (decoder input)
            embedding_layer (nn.Module): The shared embedding layer between Encoder and Decoder (Section 3.4)
            encoder_output (Tensor): Output of encoder
            source_mask (Tensor): Source (padding) mask
            target_mask (Tensor): Target (causal + padding) mask
        """
        x = embedding_layer(x) * Tensor(self.hidden_size).sqrt()

        # Section 3.5
        x = self.positional_encoder(x)

        # Section 3.1
        for sublayer in self.sublayers:
            x = sublayer(x, encoder_output, source_mask, target_mask)

        x = self.layer_norm(x)
        return self.linear(x)


class ResidualAddAndNorm(nn.Module):
    """
    Implements the residual connection + Layer Normalization
    from Section 3.1 in the paper.
    """

    def __init__(self, input_size: int, dropout_prob: float = 0.1) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_size)

        # 5.4 in Paper. Apply Dropout to the output of each layer before
        # adding to sublayer input
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x: Tensor, previous_layer: nn.Module) -> Tensor:
        # Residual connection from input x
        # Post Layer normalization (same as the paper)
        return x + self.layer_norm(self.dropout(previous_layer(x)))


class EncoderSublayer(nn.Module):
    """
    3.1 Encoder Stack in the paper.
    - Multi-head self-attention (Section 3.2.2)
    - Residual connection and layer normalization
    - Position-wise feed-forward (Section 3.3)
    - Residual connection and layer normalization
    """

    def __init__(
        self,
        hidden_size: int = 512,
        ff_hidden_size: int = 2048,
        dropout_prob: float = 0.1,
        num_attention_heads: int = 2,
    ) -> None:
        super().__init__()

        # Multi-head self attention
        self.add_and_norm1 = ResidualAddAndNorm(hidden_size)
        self.multi_head_attention = nn.MultiHeadAttention(
            num_heads=num_attention_heads, hidden_size=hidden_size
        )

        # Position-wise feedforward
        self.add_and_norm2 = ResidualAddAndNorm(hidden_size)
        self.feedforward = FeedForward(
            fc_input_size=hidden_size,
            hidden_size=ff_hidden_size,
            dropout_prob=dropout_prob,
        )

    def forward(self, x: Tensor, mask: Optional[Tensor]) -> Tensor:
        # (Section 3.2.2) Multi-head self attention
        x = self.add_and_norm1(
            x, lambda x_: self.multi_head_attention(x_, x_, x_, mask=mask)
        )

        # (Section 3.3) Position-wise feedforward
        x = self.add_and_norm2(x, self.feedforward)
        return x


class DecoderSublayer(nn.Module):
    """
    3.1 Decoder Stack in the paper.
    - Masked Multi-head self-attention (Section 3.2.3)
    - Residual connection with layer normalization
    - Encoder-Decoder multi-head attention (Section 3.2.3)
    - Position-wise Feed-forward (Section 3.3)
    """

    def __init__(
        self,
        hidden_size: int = 512,
        ff_hidden_size: int = 2048,
        dropout_prob: float = 0.1,
        num_attention_heads: int = 2,
    ) -> None:
        super().__init__()

        # Section 3.2.3 Masked Multi-head self-attention
        self.add_and_norm1 = ResidualAddAndNorm(hidden_size)
        self.masked_multi_head_attention = nn.MultiHeadAttention(
            num_heads=num_attention_heads, hidden_size=hidden_size
        )

        # Section 3.2.3 Encoder-Decoder Attention in the paper
        self.add_and_norm2 = ResidualAddAndNorm(hidden_size)
        self.multi_head_attention = nn.MultiHeadAttention(
            num_heads=num_attention_heads, hidden_size=hidden_size
        )

        # Section 3.3 Position-wise Feed-forward
        self.add_and_norm3 = ResidualAddAndNorm(hidden_size)
        self.feedforward = FeedForward(
            fc_input_size=hidden_size,
            hidden_size=ff_hidden_size,
            dropout_prob=dropout_prob,
        )

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        source_mask: Optional[Tensor],
        target_mask: Optional[Tensor],
    ) -> Tensor:
        # Masked Multi-head Attention
        # Figure 1 in Paper.
        x = self.add_and_norm1(
            x, lambda x_: self.masked_multi_head_attention(x_, x_, x_, mask=target_mask)
        )

        # 3.2.3 in Paper. Encoder-Decoder Attention
        # Queries come from previous decoder layer
        # keys, and values from output of encoder
        x = self.add_and_norm2(
            x,
            lambda x_: self.multi_head_attention(
                query=x_, key=encoder_output, value=encoder_output, mask=source_mask
            ),
        )

        # Section 3.3 Final Position-wise Feedforward
        x = self.add_and_norm3(x, self.feedforward)
        return x


class PositionalEncoding(nn.Module):
    """
    Implements the Positional Encoding from Section 3.5 in the paper.
    This allows the model to learn to attend to relative positions even
    without the sequence order information.

    $$ PE(pos, 2i) = sin(\frac{pos}{10000^{(\frac{2i}{\text{hidden\_size}})}}) $$
    $$ PE(pos, 2i+1) = cos(\frac{pos}{10000^{(\frac{2i}{\text{hidden\_size}})}}) $$
    Where:
        pos is the position
        i is the dimension
    """

    def __init__(
        self, hidden_size: int, max_seq_len: int = 5000, dropout_prob: float = 0.1
    ) -> None:
        super().__init__()
        pe = np.zeros((max_seq_len, hidden_size), dtype=np.float32)
        position = np.arange(0, max_seq_len)[:, np.newaxis]
        inverse_freq = 1.0 / 10000 ** (np.arange(0, hidden_size, 2) / hidden_size)
        pe[:, 0::2] = np.sin(position * inverse_freq)
        pe[:, 1::2] = np.cos(position * inverse_freq)

        # Shape (max_seq_len, hidden_size)
        self._parameters["pe"] = Tensor(pe, requires_grad=False)

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x: Tensor) -> Tensor:
        """
        Takes (batch_size, seq_len, vocab_size) and returns the same shape

        Args:
            x (Tensor): Embedding representation for the input text
        """
        batch_size, seq_len, input_size = x.shape
        positional_embedding = self._parameters["pe"][:seq_len, :].expand(
            (batch_size, seq_len, input_size)
        )
        x = x + positional_embedding
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):
    """
    Position-wise Feed-forward Network (Section 3.3)

    $$ FFN(x) = max(0, xW1 + b1)W2 + b2 $$
    """

    def __init__(
        self, fc_input_size: int, hidden_size: int, dropout_prob: float
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(fc_input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, fc_input_size)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x: Tensor) -> Tensor:
        x = functional.relu(self.fc1(x))
        x = self.fc2(self.dropout(x))
        return x


# ------------------ Training Helper Functions --------------------------#


def evaluate(
    model: nn.Module,
    bpe: tokenizer.BytePairEncoder,
    seq_len: int,
    teacher_enforcing: bool = False,
):
    # def transformer_predict(model: nn.Module, encoded_input: np.ndarray):
    #     source_mask = Tensor(
    #         text_utils.create_padding_mask(encoded_input, pad_idx=pad_idx),
    #         requires_grad=False,
    #     )
    #     # Create causal + pad mask for this partial sequence
    #     # Then run model(...) with encoder_output if necessary
    #     pad_mask = text_utils.create_padding_mask(encoded_input, pad_idx=pad_idx)
    #     causal_mask = text_utils.create_causal_mask(
    #         seq_len=encoded_input.shape[1], batch_size=1
    #     )
    #     target_mask = Tensor(pad_mask + causal_mask, requires_grad=False)
    #     return model(encoded_input, encoded_input, source_mask, target_mask)

    # TODO: Integrate this into the trainer class
    # if teacher_enforcing:
    #     text_utils.teacher_forcing_inference(
    #         lambda x: transformer_predict(model, x),
    #         bpe,
    #         train_data[:seq_len],
    #         vocab_idx2word=idx2word,
    #     )
    # else:
    #     text_utils.inference(
    #         lambda x: transformer_predict(model, x),
    #         bpe,
    #         start_tokens=["<SOS>"],
    #         max_length=seq_len,
    #         temperature=1.0,
    #     )

    model.train()


def initialize(hyperparams: dict, vocab: dict):
    model = Transformer(
        vocab_size=len(vocab),
        hidden_size=hyperparams["d_model"],
        num_attention_heads=hyperparams["num_attention_heads"],
    )
    optimizer = optim.Adam(model.parameters, lr=0)
    train_data_loader = LLMDataLoader(
        data=train_data,
        vocab=vocab,
        batch_size=hyperparams["batch_size"],
        seq_len=hyperparams["seq_len"],
        shuffle=True,
        include_decoder_input=True,  # By default, create (dec_inp) as well
        sos_token=b"<SOS>",
        pad_token=b"<PAD>",
    )

    test_data_loader = LLMDataLoader(
        data=test_data,
        vocab=vocab,
        batch_size=hyperparams["batch_size"] // 4,
        seq_len=hyperparams["seq_len"],
        shuffle=False,
        include_decoder_input=True,
        sos_token=b"<SOS>",
        pad_token=b"<PAD>",
    )

    return model, optimizer, train_data_loader, test_data_loader


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filename = "examples/tinyshakespeare.txt"
    resume_epoch = None  # determine whether to read from checkpoint. TODO: change this to CLI args later

    data = load_data(url, filename)
    logger.info(f"Length of entire dataset: {len(data)}")

    # Create the vocabulary
    bpe = tokenizer.BytePairEncoder(num_merges=3000, vocab_file_path="vocab.pkl")
    vocab, idx2word = bpe.train_vocabulary(data, overwrite_saved_file=False)

    # Encode a subset of the data
    data = data.split("\n\n")

    if os.path.exists("bpe_mini_shakespeare.npz"):
        logger.info("Found existing encoded data, loading it...")
        with np.load("bpe_mini_shakespeare.npz", allow_pickle=True) as npz_data:
            encoded_data = npz_data.get("arr_0")[:10000]
    else:
        logger.info("Encoding the new data...")
        encoded_data = np.array(bpe.encode("<|endoftext|>".join(data)))
        np.savez_compressed("bpe_mini_shakespeare.npz", encoded_data)
        logger.info("Saved encoded data to bpe_mini_shakespeare.npz")

    logger.info(
        f"Vocabulary size: {len(vocab)}, encoded_data length: {len(encoded_data)}"
    )
    logger.info(f"Data: {data[:3]}, Encoded_Data: {encoded_data[:50]}")

    # Split data into train and test sets
    n = int(len(encoded_data) * 0.9)
    train_data, test_data = encoded_data[:n], encoded_data[n:]

    if resume_epoch is not None:
        ckpt_json = f"checkpoints/transformer_{resume_epoch}.json"
        ckpt_npz = f"checkpoints/transformer_{resume_epoch}.npz"
        loaded_ckpt = load_model(ckpt_json, ckpt_npz)
        HYPERPARAMS = loaded_ckpt["hyperparams"]
        model, optimizer, train_data_loader, test_data_loader = initialize(
            HYPERPARAMS, vocab
        )

        model.load_state_dict(loaded_ckpt["model_state_dict"])
        optimizer.load_state_dict(loaded_ckpt["optimizer_state_dict"])
        step_count = loaded_ckpt["step_count"]
        start_epoch = loaded_ckpt["epoch"] + 1
        logger.info(
            f"Loaded model from checkpoint, resuming at epoch {start_epoch}, step {step_count}"
        )
    else:
        # Note: The current hyperparameters are not optimal, they are just used
        # for overfitting the model quickly to test the model architecture and training
        # loop are free of bugs.
        # TODO: parse the hyperparams from CLI
        HYPERPARAMS = {
            "NUM_EPOCHS": 90,
            "seq_len": 80,
            "batch_size": 16,
            "warmup_steps": 100,
            "d_model": 128,  # must be divisible by num_attention_heads
            "num_attention_heads": 4,
            "eval_iters": 20,
        }
        model, optimizer, train_data_loader, test_data_loader = initialize(
            HYPERPARAMS, vocab
        )
        start_epoch = 0
        step_count = 0

    logger.info(f"Model parameters: {model.num_parameters()}")

    trainer = LLMTrainer(
        model=model,
        optimizer=optimizer,
        warmup_steps=HYPERPARAMS["warmup_steps"],
        loss_fn=functional.cross_entropy,
        epochs=HYPERPARAMS["NUM_EPOCHS"],
        label_smoothing=0.1,
        checkpoint_freq=1,
    )

    trainer.fit(
        train_data_loader=train_data_loader,
        val_data_loader=test_data_loader,
        pad_idx=train_data_loader.pad_idx,
    )
