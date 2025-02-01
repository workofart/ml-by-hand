import logging
import os
from typing import Any, Optional

import numpy as np
import tiktoken

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
        Each residual block (decoder sublayer) in a deep stack might add up large signals,
        especially as the stack gets deeper. This is especially true at the start of training,
        so we want to prevent outputs from blowing up in magnitude early in training.
        """
        if module.__class__.__name__ == "Linear":
            module.parameters["weight"] /= np.sqrt(number_of_layers)


class DecoderSublayer(nn.Module):
    """
    A single GPT-2 Decoder block, using pre-layernorm.
    Notice that each sub-layer does a layernorm before the actual
    attention (or feedforward). GPT-1 often used post-layernorm instead.

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
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            dropout_prob=dropout_prob,
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


def load_wiki_simple():
    if not os.path.exists("training_data/wiki_simple_english.txt"):
        print("Downloading data...")
        os.system(
            "curl -L -o examples/plain-text-wikipedia-simpleenglish.zip https://www.kaggle.com/api/v1/datasets/download/ffatty/plain-text-wikipedia-simpleenglish"
        )
        os.system("unzip examples/plain-text-wikipedia-simpleenglish.zip -d examples")
        os.system("rm -rf examples/1of2")
        os.system("rm -rf examples/2of2")
        os.system("mv examples/AllCombined.txt training_data/wiki_simple_english.txt")

    data = load_data(
        "training_data/wiki_simple_english.txt",
        "training_data/wiki_simple_english.txt",
    )
    return data


def load_shakespeare_mini():
    return load_data(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "training_data/tinyshakespeare.txt",
    )


class GPT2ForwardFn(nn.AbstractLLMForwardFn):
    """
    A forward function for the Transformer model.
    """

    def train(self, model: GPT2, batch_data: Any, mode="train", **kwargs):
        X, dec_inp, y, src_mask, tgt_mask, causal_mask = batch_data
        logits = model(X, causal_mask)
        return logits, y

    def sample(self, model: GPT2, batch_data: Any, mode="train", **kwargs):
        X, dec_inp, y, src_mask, tgt_mask, causal_mask = batch_data
        logits = model(X, None)
        return logits, y


if __name__ == "__main__":
    SHAPESPEARE_CONFIG = {
        "training_run_name": "shakespeare_mini",
        "model_kwargs": {
            "num_attention_heads": 6,  # GPT-2 small uses 12
            "hidden_size": 384,  # GPT-2 small uses 768, must be divisible by num_attention_heads
            "dropout_prob": 0.2,
            "max_seq_len": 256,  # GPT-2 uses 1024
            "num_decoder_layers": 6,  # GPT-2 uses 12
        },
        "optimizer_kwargs": {
            "lr": 1e-3,
            "beta2": 0.99,
            "max_grad_norm": 1.0,
            "weight_decay": 0.1,
        },
        "num_epochs": 25,
        "warmup_steps": 100,
        "eval_iters": 100,
        "steps_per_epoch": 250,
        "checkpoint_freq": 2,
        "batch_size": 64,  # GPT-2 uses 512
        "label_smoothing": 0.1,
        # Whether to check the model performance by feeding the groundtruth tokens to compare whether the model can predict the next token correctly.
        "teacher_enforcing": False,
        "resume_epoch": 8,  # Whether to load from a checkpoint
        # if True, we use our own BPE, otherwise we use TikToken library
        "custom_bpe": {
            "num_merges": 0,
            "npz_file_path": "training_data/bpe_0_shakespeare_encoded_data",
            "vocab_file_path": "training_data/shakespeare_vocab_0.pkl",
        },
    }

    WIKI_CONFIG = {
        "training_run_name": "wiki",
        "model_kwargs": {
            "num_attention_heads": 12,  # GPT-2 small uses 12
            "hidden_size": 768,  # GPT-2 small uses 768, must be divisible by num_attention_heads
            "dropout_prob": 0.2,
            "max_seq_len": 192,  # GPT-2 uses 1024
            "num_decoder_layers": 12,  # GPT-2 uses 12
        },
        "optimizer_kwargs": {
            "lr": 1e-3,
            "beta2": 0.99,
            "max_grad_norm": 1.0,
        },
        "num_epochs": 60,
        "warmup_steps": 100,
        "eval_iters": 100,
        "steps_per_epoch": 200,
        "checkpoint_freq": 2,
        "batch_size": 64,  # GPT-2 uses 512
        "label_smoothing": 0.1,
        # Whether to check the model performance by feeding the groundtruth tokens to compare whether the model can predict the next token correctly.
        "teacher_enforcing": False,
        "resume_epoch": 29,  # Whether to load from a checkpoint
        "custom_bpe": {
            "num_merges": 12000,
            "npz_file_path": "training_data/bpe_12000_wiki_simple_encoded_data",
            "vocab_file_path": "training_data/wikipedia_simpleenglish_vocab_12000.pkl",
        },  # if non-empty, we use our own BPE, otherwise we use TikToken library
    }

    CONFIG = WIKI_CONFIG

    logger = logging.getLogger(__name__)

    # Load some data
    # data = load_shakespeare_mini()
    data = load_wiki_simple()

    logger.info(f"{len(data)} characters in the entire dataset")
    # data = data[:50_000]
    print(data[:100])

    n = int(len(data) * 0.9)
    train_data, test_data = data[:n], data[n:]

    if CONFIG.get("custom_bpe"):
        # Create a Byte Pair Encoder and prepare data
        bpe = BytePairEncoder(
            num_merges=CONFIG["custom_bpe"]["num_merges"],
            vocab_file_path=CONFIG["custom_bpe"]["vocab_file_path"],
        )
        # Train the vocabulary on the entire dataset
        bpe.train_vocabulary(
            data,
            overwrite_saved_file=False,
        )
        data_len = len(data.split("\n\n"))
        print(f"Total length of data after split: {data_len}")
        train_data = bpe.prepare_data_parallel(
            raw_text_list=train_data.split("\n\n"),
            npz_file_path=f"{CONFIG["custom_bpe"]["npz_file_path"]}_train.npz",
            overwrite_saved_file=False,
            split_token="<|endoftext|>",
        )
        test_data = bpe.prepare_data_parallel(
            raw_text_list=test_data.split("\n\n"),
            npz_file_path=f"{CONFIG["custom_bpe"]["npz_file_path"]}_test.npz",
            overwrite_saved_file=False,
            split_token="<|endoftext|>",
        )
    else:
        # Using Tiktoken for tokenizer
        bpe = tiktoken.get_encoding("gpt2")
        train_data = bpe.encode(train_data)
        test_data = bpe.encode(test_data)

    print(f"Data length: {len(train_data)=} {len(test_data)=}")

    CONFIG["model_kwargs"]["vocab_size"] = bpe.n_vocab

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
        epochs=60 or hparams["num_epochs"],
        warmup_steps=hparams["warmup_steps"],
        label_smoothing=hparams["label_smoothing"],
        checkpoint_freq=hparams["checkpoint_freq"],
        forward_fn=GPT2ForwardFn(),
        tokenizer=bpe,
        teacher_enforcing=hparams["teacher_enforcing"],
        hyperparams=hparams,
        checkpoint=checkpoint,
        start_tokens="First",
    )

    trainer.fit(train_data_loader, test_data_loader, pad_idx=train_data_loader.pad_idx)

    # Inference test
    for k in range(10):
        text_utils.inference(
            model=model,
            prediction_func=GPT2ForwardFn(),
            bpe=bpe,
            start_tokens="April is a charming day",  # Example start token
            max_length=int(model.max_seq_len),
            temperature=0.7,
            # top_k=200,
        )
        print("\n------------------------\n")
