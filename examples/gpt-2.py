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


class GPT2ForwardFn(nn.AbstractLLMForwardFn):
    """
    A forward function for the Transformer model.
    """

    def train(self, model: GPT2, batch_data: Any, **kwargs):
        X, dec_inp, y, src_mask, tgt_mask, causal_mask = batch_data
        logits = model(X, causal_mask)
        return logits, y

    def sample(self, model: GPT2, batch_data: Any, **kwargs):
        logits = model(batch_data, None)
        return logits, None


if __name__ == "__main__":
    SHAPESPEARE_CONFIG = TransformerTrainingConfig(
        training_run_name="shakespeare_mini",
        dataset_name="shakespeare_mini",
        batch_size=64,  # GPT-2 uses 512
        total_epochs=15,
        eval_iters=50,
        steps_per_epoch=100,
        checkpoint_freq=4,
        model_kwargs={
            "num_attention_heads": 6,  # GPT-2 small uses 12
            "hidden_size": 768,  # GPT-2 small uses 768, must be divisible by num_attention_heads
            "dropout_prob": 0.0,
            "max_seq_len": 256,  # GPT-2 uses 1024
            "num_decoder_layers": 6,  # GPT-2 uses 12
        },
        optimizer_kwargs={
            "lr": 1e-3,
            "beta2": 0.99,
            "max_grad_norm": 1.0,
            "weight_decay": 0.1,
            "lr_scheduler_kwargs": {
                "lr_scheduler_cls": optim.CosineScheduler,  # TODO: check if we can serialize this into checkpoint
                "warmup_steps": 100,
                "lr_decay_iters": 1000,  # steps_per_epoch * total_epochs
            },
        },
        resume_epoch=4,
        teacher_enforcing=True,
        include_decoder_input=False,
        create_padding_masks=False,
        label_smoothing=0.1,
        eval_start_string="First",
        eval_top_k=50,  # Shakespeare only has ~60 unique characters, we so will just sample top 50
        custom_bpe=CustomBpeConfig(
            num_merges=0,
            encoded_data_path="training_data/bpe_0_shakespeare_encoded_data.npz",
            vocab_path="training_data/shakespeare_vocab_0.pkl",
            overwrite_encoded_data=True,
            overwrite_vocabulary_file=True,
            split_token="<|endoftext|>",
        ),
    )

    WIKI_CONFIG = TransformerTrainingConfig(
        training_run_name="wiki",
        dataset_name="wiki_simple_english",
        batch_size=16,  # GPT-2 uses 512
        total_epochs=30,
        eval_iters=100,
        steps_per_epoch=1600,
        update_weights_every_n_steps=8, # simulate larger batch sizes
        checkpoint_freq=4,
        model_kwargs={
            "num_attention_heads": 12,  # GPT-2 small uses 12
            "hidden_size": 768,  # GPT-2 small uses 768, must be divisible by num_attention_heads
            "dropout_prob": 0.2,
            "max_seq_len": 568,  # GPT-2 uses 1024
            "num_decoder_layers": 12,  # GPT-2 uses 12
        },
        optimizer_kwargs={
            "lr": 1e-3,
            "beta2": 0.99,
            "max_grad_norm": 1.0,
            "weight_decay": 0.1,
            "lr_scheduler_kwargs": {
                "lr_scheduler_cls": optim.CosineScheduler,
                "warmup_steps": 100,
                "lr_decay_iters": 40000,
            },
        },
        resume_epoch=None,
        teacher_enforcing=False,
        include_decoder_input=False,
        create_padding_masks=False,
        label_smoothing=0.1,
        eval_start_string="April is",
        custom_bpe=CustomBpeConfig(
            num_merges=12000,
            encoded_data_path="training_data/bpe_12000_wiki_simple_encoded_data.npz",
            vocab_path="training_data/wikipedia_simpleenglish_vocab_12000.pkl",
            overwrite_encoded_data=False,
            overwrite_vocabulary_file=False,
            split_token="<|endoftext|>",
        ),
    )
    
    OPENWEBTEXT_CONFIG = TransformerTrainingConfig(
        training_run_name="open_web_text",
        dataset_name="open_web_text",
        batch_size=16,  # GPT-2 uses 512
        total_epochs=500,
        eval_iters=100,
        steps_per_epoch=200,
        checkpoint_freq=4,
        model_kwargs={
            "num_attention_heads": 12,  # GPT-2 small uses 12
            "hidden_size": 768,  # GPT-2 small uses 768, must be divisible by num_attention_heads
            "dropout_prob": 0.2,
            "max_seq_len": 392,  # GPT-2 uses 1024
            "num_decoder_layers": 12,  # GPT-2 uses 12
        },
        optimizer_kwargs={
            "lr": 1e-3,
            "beta2": 0.99,
            "max_grad_norm": 1.0,
            "weight_decay": 0.1,
            "lr_scheduler_kwargs": {
                "lr_scheduler_cls": optim.CosineScheduler,
                "warmup_steps": 100,
                "lr_decay_iters": 5000,
            },
        },
        resume_epoch=380,
        teacher_enforcing=False,
        include_decoder_input=False,
        create_padding_masks=False,
        label_smoothing=0.1,
        eval_start_string="April is",
        custom_bpe=None,
    )

    CONFIG = WIKI_CONFIG

    logger = logging.getLogger(__name__)

    # Load some data
    # data = text_utils.load_shakespeare_mini()
    data = text_utils.load_wiki_simple()

    if CONFIG.custom_bpe:
        # Create a Byte Pair Encoder and prepare data
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
        # TODO: Experimental to debug whether the model learning is poorly due to
        # the tokenizer. Need to install the python package manually via `uv pip install tiktoken`
        import tiktoken

        bpe = tiktoken.get_encoding("gpt2")
        # encoded_data = bpe.encode(data)
        # import numpy
        # with np.load("training_data/openwebtext_train.npz", allow_pickle=True) as npz_data:
        #     train_data = npz_data["arr_0"]
        # with np.load("training_data/openwebtext_val.npz", allow_pickle=True) as npz_data:
        #     test_data = npz_data["arr_0"]
            
        # train_data = np.asarray(numpy.memmap("training_data/openwebtext_train.bin", dtype=numpy.uint16, mode="r"))
        # test_data = np.asarray(numpy.memmap("training_data/openwebtext_val.bin", dtype=numpy.uint16, mode="r"))
        # encoded_data = np.asarray(numpy.memmap("training_data/openwebtext_val.bin", dtype=numpy.uint16, mode="r"))
        # np.savez_compressed("training_data/openwebtext_train.npz", numpy.memmap("training_data/openwebtext_train.bin", dtype=numpy.uint16, mode="r"))

    n = int(len(encoded_data) * 0.9)
    train_data, test_data = encoded_data[:n], encoded_data[n:]
    print(f"Data length: {len(train_data)=} {len(test_data)=}")

    CONFIG.model_kwargs["vocab_size"] = bpe.n_vocab

    trainer = LLMTrainer(
        model_cls=GPT2,
        optimizer_cls=optim.Adam,
        loss_fn=functional.cross_entropy,
        config=CONFIG,
        forward_fn=GPT2ForwardFn(),
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

    # Inference test
    for k in range(5):
        text_utils.inference(
            model=trainer.model,
            prediction_func=GPT2ForwardFn(),
            bpe=bpe,
            # start_tokens="\n",  # Example start token
            start_tokens="The capital of China is",  # Example start token
            max_length=int(trainer.model.max_seq_len),
            temperature=1.0,
            top_k=200,  # for shakespeare, there are only 63 vocabulary that are used, so let's limit to the top 50 to avoid printing weird characters
        )
        print("\n------------------------\n")
