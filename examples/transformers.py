import logging
from typing import Any, Optional

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
from autograd.tools.config_schema import CustomBpeConfig, TransformerTrainingConfig
from autograd.tools.data import LLMDataLoader
from autograd.tools.trainer import LLMTrainer


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
        Initialize the Transformer model.

        Args:
            vocab_size (int): Size of the vocabulary.
            hidden_size (int): Dimensionality of the model.
            num_attention_heads (int): Number of attention heads.
            **kwargs: Additional keyword arguments (may include "max_seq_len").
        """
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_seq_len = kwargs.get("max_seq_len")
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
        Compute the forward pass of the Transformer model.

        Args:
            source (Tensor): Source sequence indices of shape (batch_size, seq_len).
            target (Tensor): Target sequence indices of shape (batch_size, seq_len).
            source_mask (Optional[Tensor]): Mask for the source sequence (e.g., to ignore padding).
            target_mask (Optional[Tensor]): Mask for the target sequence (e.g., causal mask).

        Returns:
            Tensor: Logits over the vocabulary for each position in the target sequence.
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
    Encoder component of the Transformer model.
    - Embeddings (Section 3.1) + Positional Encoding (Section 3.5)
    - 6 identical EncoderSublayer (Section 3.1)
    """

    def __init__(self, embedding_size: int, num_attention_heads: int) -> None:
        """
        Initialize the Encoder.

        Args:
            embedding_size (int): Dimensionality of the token embeddings.
            num_attention_heads (int): Number of attention heads for the self-attention mechanism.
        """
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
        """
        Compute the forward pass of the Encoder.

        The method first applies token embedding (scaled by the square root of the embedding size),
        then adds positional encodings, and processes the result through a stack of encoder sublayers.
        Finally, a layer normalization is applied.

        Args:
            x (Tensor): Input token indices.
            embedding_layer (nn.Module): Shared embedding layer.
            mask (Optional[Tensor]): Mask to ignore certain positions (e.g., padding).

        Returns:
            Tensor: Encoder output of shape (batch_size, seq_len, embedding_size).
        """
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
        """
        Initialize the Decoder.

        Args:
            vocab_size (int): Size of the vocabulary.
            hidden_size (int): Dimensionality of the model.
            num_attention_heads (int, optional): Number of attention heads. Defaults to 2.
        """
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
        Compute the forward pass of the Decoder.

        The method applies token embedding with scaling, positional encoding, and processes
        the result through a stack of decoder sublayers that attend both to the decoder input
        and the encoder output. Finally, a layer normalization is applied followed by a linear projection.

        Args:
            x (Tensor): Target token indices.
            embedding_layer (nn.Module): Shared embedding layer.
            encoder_output (Tensor): Output from the encoder.
            source_mask (Optional[Tensor]): Mask for the encoder input.
            target_mask (Optional[Tensor]): Mask for the target input (e.g., causal mask).

        Returns:
            Tensor: Logits over the vocabulary for each time step.
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

    This module applies dropout to the output of a sublayer, adds it to the original input,
    and then applies layer normalization.
    """

    def __init__(self, input_size: int, dropout_prob: float = 0.1) -> None:
        """
        Initialize the ResidualAddAndNorm module.

        Args:
            input_size (int): Dimensionality of the input tensor.
            dropout_prob (float): Dropout probability.
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_size)
        # 5.4 in Paper. Apply Dropout to the output of each layer before
        # adding to sublayer input
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x: Tensor, previous_layer: nn.Module) -> Tensor:
        """
        Apply a residual connection, dropout, and layer normalization.
        Post Layer normalization (same as the paper)

        Args:
            x (Tensor): The input tensor.
            previous_layer (nn.Module): A function representing the sublayer transformation.

        Returns:
            Tensor: The output tensor after residual addition and normalization.
        """
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
        """
        Initialize an EncoderSublayer.

        Args:
            hidden_size (int): Dimensionality of the model.
            ff_hidden_size (int): Dimensionality of the hidden layer in the feed-forward network.
            dropout_prob (float): Dropout probability.
            num_attention_heads (int): Number of attention heads.
        """
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
        """
        Compute the forward pass of the encoder sublayer.
        (Section 3.2.2) Multi-head self attention

        The sublayer applies multi-head self-attention followed by a position-wise feed-forward network,
        each with residual connections and layer normalization.

        Args:
            x (Tensor): Input tensor.
            mask (Optional[Tensor]): Attention mask to ignore certain positions.

        Returns:
            Tensor: The output tensor of the encoder sublayer.
        """
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
        """
        Initialize a DecoderSublayer.

        Args:
            hidden_size (int): Dimensionality of the model.
            ff_hidden_size (int): Dimensionality of the hidden layer in the feed-forward network.
            dropout_prob (float): Dropout probability.
            num_attention_heads (int): Number of attention heads.
        """
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
        """
        Compute the forward pass of the decoder sublayer.

        Masked Multi-head Attention
        Figure 1 in Paper.

        The sublayer applies masked multi-head self-attention on the decoder input, then
        performs encoder-decoder attention, followed by a feed-forward network. Residual
        connections and layer normalization are applied after each step.

        Args:
            x (Tensor): Decoder input tensor.
            encoder_output (Tensor): Encoder output tensor.
            source_mask (Optional[Tensor]): Mask for the encoder input.
            target_mask (Optional[Tensor]): Mask for the decoder input (e.g., to enforce causality).

        Returns:
            Tensor: The output tensor of the decoder sublayer.
        """
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
    Implements positional encoding as described in Section 3.5 of the paper.

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
        """
        Initialize the PositionalEncoding module.

        Args:
            hidden_size (int): Dimensionality of the model.
            max_seq_len (int): Maximum sequence length for which to compute positional encodings.
            dropout_prob (float): Dropout probability.
        """
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
        Add positional encoding to the input embeddings and apply dropout.

        Args:
            x (Tensor): Input embeddings of shape (batch_size, seq_len, input_size).

        Returns:
            Tensor: The embeddings with positional encodings added.
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
    Position-wise feed-forward network as described in Section 3.3 of the paper.

    This module consists of two linear layers with a ReLU activation in between and dropout applied
    to the intermediate representation.
    """

    def __init__(
        self, fc_input_size: int, hidden_size: int, dropout_prob: float
    ) -> None:
        """
        Initialize the FeedForward network.

        Args:
            fc_input_size (int): Dimensionality of the input.
            hidden_size (int): Dimensionality of the hidden layer.
            dropout_prob (float): Dropout probability.
        """
        super().__init__()
        self.fc1 = nn.Linear(fc_input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, fc_input_size)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the forward pass of the FeedForward network.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying ReLU, dropout, and a second linear transformation.
        """
        x = functional.relu(self.fc1(x))
        x = self.fc2(self.dropout(x))
        return x


class TransformerForwardFn(nn.AbstractLLMForwardFn):
    """
    Forward function implementation for the Transformer model.

    This class implements the AbstractLLMForwardFn interface for the Transformer.
    It provides separate methods for training (which returns logits and ground truth)
    and sampling (which returns logits and None for the ground truth).
    """

    def train(self, model: Transformer, batch_data: Any, **kwargs):
        """
        Compute the forward pass during training.

        Args:
            model (Transformer): The Transformer model.
            batch_data (Any): A tuple containing (X, decoder_input, y, src_mask, tgt_mask, causal_mask).

        Returns:
            Tuple[Tensor, Any]: The output logits and the target labels.
        """
        X, dec_inp, y, src_mask, tgt_mask, causal_mask = batch_data
        logits = model(X, dec_inp, src_mask, tgt_mask)
        return logits, y

    def sample(self, model: Transformer, batch_data: Any, **kwargs):
        """
        Compute the forward pass during sampling.

        Args:
            model (Transformer): The Transformer model.
            batch_data (Any): Input batch data used for sampling.

        Returns:
            Tuple[Tensor, None]: The output logits and None (no ground truth during sampling).
        """
        logits = model(batch_data, batch_data, None, None)
        return logits, None


if __name__ == "__main__":
    """
    Main script for training a Transformer model on a text summarization task.

    The pipeline is as follows:
      1) Define a training configuration using TransformerTrainingConfig.
      2) Load a text dataset (here, the "shakespeare_mini" dataset) via a helper function.
      3) Create a BytePairEncoder (BPE) using custom BPE configuration from the training config.
      4) Prepare the data by encoding the raw text into a sequence of integer tokens.
      5) Split the encoded data into training and testing portions.
      6) Update the model configuration with the vocabulary size from the BPE.
      7) Instantiate an LLMTrainer with the Transformer model, optimizer, loss function, and forward function.
      8) Create LLMDataLoader objects for training and evaluation.
      9) Train the Transformer model.
      10) Run inference on the trained model to generate output text.

    No value is returned; training progress and generated outputs are logged.
    """
    CONFIG = TransformerTrainingConfig(
        training_run_name="shakespeare_mini",
        dataset_name="shakespeare_mini",
        batch_size=16,
        total_epochs=25,
        eval_iters=16,
        steps_per_epoch=20,
        checkpoint_freq=2,
        model_kwargs={
            "num_attention_heads": 4,
            "hidden_size": 128,
            "dropout_prob": 0.1,
            "max_seq_len": 96,
            "num_decoder_layers": 4,
        },
        optimizer_kwargs={
            "lr": 1e-3,
            "beta2": 0.99,
            "max_grad_norm": 1.0,
            "weight_decay": 0.1,
            "lr_scheduler_kwargs": {
                "lr_scheduler_cls": optim.CosineScheduler,
                "warmup_steps": 100,
                "lr_decay_iters": 500,
            },
        },
        resume_epoch=None,
        teacher_enforcing=True,
        include_decoder_input=True,
        create_padding_masks=True,
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
            "Currently this original Transformers model can only be trained with the custom BytePairEncoder, please specify the custom_bpe config"
        )

    # encoded_data is a list of integers without the concept of samples
    n = int(len(encoded_data) * 0.9)
    train_data, test_data = encoded_data[:n], encoded_data[n:]
    print(f"Data length: {len(train_data)=} {len(test_data)=}")

    CONFIG.model_kwargs["vocab_size"] = bpe.n_vocab

    trainer = LLMTrainer(
        model_cls=Transformer,
        optimizer_cls=optim.Adam,
        loss_fn=functional.cross_entropy,
        config=CONFIG,
        forward_fn=TransformerForwardFn(),
    )

    train_data_loader = LLMDataLoader(
        data=np.array(train_data),
        bpe=bpe,
        batch_size=CONFIG.batch_size,
        seq_len=trainer.model.max_seq_len,
        steps_per_epoch=CONFIG.steps_per_epoch,
        shuffle=True,
        include_decoder_input=CONFIG.include_decoder_input,
        create_padding_masks=CONFIG.create_padding_masks,
    )
    test_data_loader = LLMDataLoader(
        data=np.array(test_data),
        bpe=bpe,
        batch_size=CONFIG.batch_size // 2,
        seq_len=trainer.model.max_seq_len,
        steps_per_epoch=CONFIG.eval_iters,
        shuffle=False,
        include_decoder_input=CONFIG.include_decoder_input,
        create_padding_masks=CONFIG.create_padding_masks,
    )

    trainer.fit(train_data_loader, test_data_loader)

    text_utils.inference(
        model=trainer.model,
        prediction_func=TransformerForwardFn(),
        bpe=bpe,
        start_tokens="All",  # Dummy token to start the generation
        max_length=int(
            trainer.model.max_seq_len * 0.9
        ),  # this should be shorter than context
        temperature=1.0,
    )
