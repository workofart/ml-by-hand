import logging
from typing import Optional

import numpy as np

from autograd import functional, nn, optim
from autograd.tensor import Tensor
from autograd.text import utils as text_utils
from autograd.text.tokenizer import BytePairEncoder
from autograd.tools.data import LLMDataLoader, load_data
from autograd.tools.trainer import LLMTrainer, load_model_and_optimizer

# The feedforward layer is the same as the original transformers
from examples.transformers import (
    FeedForward,
)


class GPT2(nn.Module):
    """
    GPT-2
    Paper: Language Models are Unsupervised Multitask Learners
    https://www.semanticscholar.org/paper/Language-Models-are-Unsupervised-Multitask-Learners-Radford-Wu/9405cc0d6169988371b2755e573cc28650d14dfe

    Key Differences from GPT-1:
    1) We apply Layer Normalization before attention/feedforward in each decoder sublayer. (i.e. self.sublayers[...].layer_norm1 and self.sublayers[...].layer_norm2)
    2) Apply a final layer normalization at the end of the transformer stack (i.e. self.layer_norm)
    3) Larger hidden size (varying from 768 to 1600), more layers (varying from 12 to 48 layers), more heads, and a longer context (1024 tokens compared to GPT-1 512 tokens).
    4) Scale the weights of residual layers by 1 / sqrt(number of residual layers)
    5) Expanded vocabulary to 50257, but still using BytePairEncoder
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,  # GPT-2 small uses 768
        num_attention_heads: int = 12,  # GPT-2 small uses 12 heads
        max_seq_len: int = 1024,  # GPT-2 small uses 1024 context window
        dropout_prob: float = 0.1,
        num_decoder_layers: int = 12,  # GPT-2 small has 12 layers
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        # Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)

        # Dropout applied after embeddings (same as GPT-1)
        self.dropout = nn.Dropout(dropout_prob)

        self.sublayers = nn.ModuleList(
            [
                DecoderSublayer(
                    hidden_size=hidden_size,
                    ff_hidden_size=4 * hidden_size,  # GPT-2 typically 4 * hidden
                    num_attention_heads=num_attention_heads,
                    dropout_prob=dropout_prob,
                )
                for _ in range(num_decoder_layers)
            ]
        )

        # Final layernorm after all Transformer blocks
        # Section 3.2 "Model" in the paper
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.apply(
            lambda m: self._scale_weights(m, num_decoder_layers * 2)
        )  # there are 2 residual layers in each decoder sublayer

    def forward(self, tokens: Tensor, mask: Optional[Tensor]) -> Tensor:
        """
        Forward pass for GPT-2.
        tokens: shape (batch_size, seq_len)
        mask: optional shape (batch_size, 1, seq_len, seq_len) for causal masking
        """
        batch_size, seq_len = tokens.shape

        # Create positions [0,1,2,...,seq_len-1], repeated for each batch
        positions = np.arange(seq_len)  # shape (seq_len, )
        positions = np.tile(positions, (batch_size, 1))  # shape (batch_size, seq_len)

        token_emb = self.token_embedding(tokens)  # shape: (batch, seq_len, hidden_dim)
        pos_emb = self.position_embedding(
            positions
        )  # shape: (batch, seq_len, hidden_dim)

        # Dropout on the sum of token + position embeddings
        h_0 = self.dropout(token_emb + pos_emb)

        # Pass through each Decoder sublayer
        for sublayer in self.sublayers:
            h_0 = sublayer(h_0, mask)

        # Final normalization
        output = self.layer_norm(h_0)

        # Output logits: multiply by the transpose of the embedding matrix
        # This ties the weights with the input embedding,
        output = (
            output @ self.token_embedding.parameters["weight"].T
        )  # shape (batch_size, seq_len, vocab_size)
        return output

    def _scale_weights(self, module: nn.Module, number_of_layers: int):
        """
        Scale the weights of the model by the square root of the number of layers.
        This is done to prevent the variance of the activations from exploding.
        """
        if module.__class__.__name__ == "Linear":
            module.parameters["weight"] /= np.sqrt(number_of_layers)
            logger.info("Down scaled weights of linear layer")


class DecoderSublayer(nn.Module):
    """
    A single GPT-2 Decoder block, using pre-layernorm.
    Notice that each sub-layer does a layernorm before the actual
    attention (or MLP). GPT-1 often used post-norm instead.

    Section 2.3 "Model" of the Paper.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        ff_hidden_size: int = 3072,
        num_attention_heads: int = 12,
        dropout_prob: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # First LayerNorm (for the attention sub-layer)
        self.layer_norm1 = nn.LayerNorm(hidden_size)

        self.multi_head_attention = nn.MultiHeadAttention(
            hidden_size=hidden_size, num_heads=num_attention_heads
        )

        # Second LayerNorm (for the feed-forward sub-layer)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

        self.feedforward = FeedForward(
            fc_input_size=hidden_size,
            hidden_size=ff_hidden_size,
            dropout_prob=dropout_prob,
        )

    def forward(self, x: Tensor, mask: Optional[Tensor]) -> Tensor:
        # Pre-norm before attention
        a = self.layer_norm1(x)

        x = x + self.multi_head_attention(
            a, a, a, mask=mask
        )  # (batch, seq_len, hidden_size)

        # Pre-norm before feed-forward
        b = self.layer_norm2(x)
        x = x + self.feedforward(b)
        return x


if __name__ == "__main__":

    def gpt_2_forward(model, batch_or_tokens, mode="train"):
        if mode == "train":
            # We assume the data loader returns:
            # X, dec_inp, y, src_mask, tgt_mask, causal_mask
            X, _, y, _, _, causal_mask = batch_or_tokens
            logits = model(X, causal_mask)
            return logits, y
        elif mode == "sample":
            tokens = batch_or_tokens
            logits = model(tokens, None)
            return logits
        else:
            raise ValueError(f"Unknown mode {mode}, must be 'train' or 'sample'")

    logger = logging.getLogger(__name__)

    # Load some data
    data = load_data(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "examples/tinyshakespeare.txt",
    )
    logger.info(f"{len(data)} characters in the entire dataset")

    # Create a Byte Pair Encoder and prepare data
    bpe = BytePairEncoder(num_merges=3000, vocab_file_path="vocab.pkl")
    encoded_data = bpe.prepare_data(
        raw_text_list=data.split("\n\n"),
        npz_file_path="bpe_mini_shakespeare.npz",
        overwrite_saved_file=False,
        split_token="<|endoftext|>",
    )[:60000]

    n = int(len(encoded_data) * 0.9)
    train_data, test_data = encoded_data[:n], encoded_data[n:]

    CONFIG = {
        "model_kwargs": {
            "vocab_size": len(bpe._unicode_to_int_vocab),
            "num_attention_heads": 8,  # GPT-2 small uses 12
            "hidden_size": 512,  # GPT-2 small uses 768, must be divisible by num_attention_heads
            "dropout_prob": 0.1,
            "max_seq_len": 256,  # GPT-2 uses 1024
            "num_decoder_layers": 6,  # GPT-2 uses 12
        },
        "optimizer_kwargs": {
            "lr": 0.0  # We can later schedule it with warmup
        },
        "num_epochs": 20,
        "warmup_steps": 100,
        "eval_iters": 16,
        "batch_size": 32,  # GPT-2 uses 512
        # Whether to check the model performance by feeding the groundtruth tokens to compare whether the model can predict the next token correctly.
        "teacher_enforcing": True,
        "resume_epoch": None,  # Whether to load from a checkpoint
    }

    # Build GPT-2 model, reusing the same training logic
    model, optimizer, checkpoint = load_model_and_optimizer(
        GPT2,
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
        forward_fn=gpt_2_forward,
        tokenizer=bpe,
        teacher_enforcing=hparams["teacher_enforcing"],
        hyperparams=hparams,
        checkpoint=checkpoint,
    )

    trainer.fit(train_data_loader, test_data_loader)

    # Inference test
    text_utils.inference(
        prediction_func=lambda seq_so_far: gpt_2_forward(
            model, seq_so_far, mode="sample"
        ),
        bpe=bpe,
        start_tokens=["All"],  # Example start token
        max_length=int(model.max_seq_len * 0.9),
        temperature=1.0,
        top_k=10,
    )
