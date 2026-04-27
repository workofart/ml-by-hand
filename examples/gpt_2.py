# ruff: noqa: E402

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autograd import functional, nn, optim
from autograd.backend import LOW_PRECISION_FLOAT_DTYPES, Array, xp
from autograd.data.collator import CausalLMWindowCollator
from autograd.data.data_loader import DataLoader
from autograd.data.dataset import TokenWindowMapDataset
from autograd.data.sampler import RandomSampler, SequentialSampler
from autograd.data.types import CausalLMBatch
from autograd.tensor import Tensor, checkpoint
from autograd.text import utils as text_utils
from autograd.text.tokenizer import BytePairEncoder
from autograd.tools.callback import (
    run_sampling_inference,
    run_teacher_forcing_inference,
)
from autograd.tools.config_schema import CustomBpeConfig, TransformerTrainingConfig
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
        activation_checkpointing: bool = False,
        parameter_dtype=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if isinstance(parameter_dtype, str):
            parameter_dtype = getattr(xp, parameter_dtype)
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.activation_checkpointing = activation_checkpointing

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
        if parameter_dtype is not None:
            for parameter in self.parameters.values():
                parameter.data = parameter.data.astype(parameter_dtype)

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Forward pass for GPT-2.
        tokens: shape (batch_size, seq_len)
        """
        batch_size, seq_len = tokens.shape

        # Create positions [0,1,2,...,seq_len-1], repeated for each batch
        positions = xp.arange(seq_len, dtype=xp.int32)  # shape (seq_len, )
        positions = xp.tile(positions, (batch_size, 1))  # shape (batch_size, seq_len)

        token_emb = self.token_embedding(tokens)  # shape: (batch, seq_len, hidden_dim)
        pos_emb = self.position_embedding(
            positions
        )  # shape: (batch, seq_len, hidden_dim)

        # Dropout on the sum of token + position embeddings
        h_0 = self.dropout(token_emb + pos_emb)

        # Pass through each Decoder sublayer
        for sublayer in self.sublayers:
            if self._is_training and self.activation_checkpointing:
                h_0 = checkpoint(sublayer, h_0)
            else:
                h_0 = sublayer(h_0)

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
            module._parameters["weight"].data /= float(number_of_layers) ** 0.5


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

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.data.dtype
        low_precision_input = input_dtype in LOW_PRECISION_FLOAT_DTYPES

        # Pre-norm before attention
        a = self.layer_norm1(x)

        x = x + self.multi_head_attention(a, a, a, is_causal=True)
        if low_precision_input:
            # Dense attention can promote through its fp32 mask path; cast the residual
            # stream back so later matmuls keep the intended low-precision activations.
            x = x.astype(input_dtype)

        # Pre-norm before feed-forward
        b = self.layer_norm2(x)
        x = x + self.feedforward(b)
        if low_precision_input:
            # Linear bias/addition can also promote; preserve the block's input dtype.
            x = x.astype(input_dtype)
        return x


class GPT2ForwardFn(nn.AbstractLLMForwardFn):
    """
    A forward function for the GPT-2 model.
    """

    def train(self, model: GPT2, batch: CausalLMBatch) -> Tensor:
        return model(batch.input_ids)

    def sample(self, model: GPT2, input_ids: Array) -> Tensor:
        return model(input_ids)


if __name__ == "__main__":
    train_global_batch_size = 16
    SHAKESPEARE_CONFIG = TransformerTrainingConfig(
        training_run_name="shakespeare_mini",
        dataset_name="shakespeare_mini",
        max_steps=1000,
        max_eval_steps=50,
        checkpoint_freq=4,
        global_batch_size=train_global_batch_size,
        micro_batch_size=train_global_batch_size,
        max_grad_norm=1.0,
        model_kwargs={
            "num_attention_heads": 6,  # GPT-2 small uses 12
            "hidden_size": 768,  # GPT-2 small uses 768, must be divisible by num_attention_heads
            "dropout_prob": 0.3,
            "max_seq_len": 96,  # GPT-2 uses 1024
            "num_decoder_layers": 6,  # GPT-2 uses 12
        },
        optimizer_kwargs={
            "lr": 1e-3,
            "beta2": 0.99,
            "weight_decay": 0.1,
            "lr_scheduler_kwargs": {
                "lr_scheduler_cls": "CosineScheduler",
                "warmup_steps": 100,
                "lr_decay_iters": 1000,  # matches max_steps
            },
        },
        resume_epoch=None,  # Set this to None if you don't want to load from checkpoint
        teacher_forcing=True,
        label_smoothing=0.1,
        eval_start_string="First",
        eval_top_k=50,  # Shakespeare only has ~60 unique characters, and our if we do 3000 merges in BPE, our vocabulary size is 260, we so will just sample top 50.
        # The following shows what we use to tokenize and encode our input data
        # We are using our own BytePairEncoder class in autograd/text/tokenizer.py
        # Feel free to play around with the "num_merges". This controls the tradeoff between vocabulary size
        # and the total sequence length of the encoded text.
        # Double-check whether we want to overwrite the encoded_data and vocabulary
        custom_bpe=CustomBpeConfig(
            num_merges=3000,
            encoded_data_path="training_data/bpe_3000_shakespeare_encoded_data.npz",
            vocab_path="training_data/shakespeare_vocab_3000.pkl",
            overwrite_encoded_data=False,
            overwrite_vocabulary_file=False,
            split_token="<|endoftext|>",
        ),
    )

    WIKI_CONFIG = TransformerTrainingConfig(
        training_run_name="wiki",
        dataset_name="wiki_simple_english",
        max_steps=25000,
        max_eval_steps=20,
        checkpoint_freq=1000,
        report_every_steps=50,
        global_batch_size=32,
        micro_batch_size=8,
        max_grad_norm=1.0,
        model_kwargs={
            "num_attention_heads": 9,  # GPT-2 small uses 12
            "hidden_size": 576,  # GPT-2 small uses 768, must be divisible by num_attention_heads
            "dropout_prob": 0.1,
            "max_seq_len": 1024,  # GPT-2 uses 1024
            "num_decoder_layers": 8,  # GPT-2 uses 12
            "activation_checkpointing": False,
            "parameter_dtype": "bfloat16",
        },
        optimizer_kwargs={
            "lr": 1e-3,
            "beta2": 0.99,
            "weight_decay": 0.1,
            "lr_scheduler_kwargs": {
                "lr_scheduler_cls": optim.CosineScheduler,
                "warmup_steps": 3750,  # 15% of max_steps
                "lr_decay_iters": 20000,  # 80% of max_steps
            },
        },
        resume_epoch=None,  # Set this to None if you don't want to load from checkpoint
        teacher_forcing=False,
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

    CONFIG = WIKI_CONFIG

    logger = logging.getLogger(__name__)

    # Load some data
    # Note: Please supply the correct data for your model
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
        )
    else:
        raise ValueError(
            "Please supply a custom_bpe config. Check out CustomBpeConfig for more details."
        )

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

    train_dataset = TokenWindowMapDataset(
        data=xp.array(train_data, dtype=xp.int32),
        # CausalLMWindowCollator shifts one token to build input_ids/labels,
        # so a length-T model context needs a raw window of length T + 1.
        window_len=trainer.model.max_seq_len + 1,
    )
    test_dataset = TokenWindowMapDataset(
        data=xp.array(test_data, dtype=xp.int32),
        # CausalLMWindowCollator shifts one token to build input_ids/labels,
        # so a length-T model context needs a raw window of length T + 1.
        window_len=trainer.model.max_seq_len + 1,
    )
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=CONFIG.micro_batch_size,
        collator=CausalLMWindowCollator(),
        sampler=RandomSampler(
            train_dataset,
            replacement=True,
            num_samples=len(train_dataset),
        ),
    )
    test_data_loader = DataLoader(
        dataset=test_dataset,
        batch_size=max(1, CONFIG.micro_batch_size // 2),
        collator=CausalLMWindowCollator(),
        sampler=SequentialSampler(test_dataset),
    )

    trainer.fit(train_data_loader, test_data_loader)

    if CONFIG.teacher_forcing:
        run_teacher_forcing_inference(
            model=trainer.model,
            forward_fn=GPT2ForwardFn(),
            bpe=bpe,
            groundtruth_data=xp.array(
                test_data[: trainer.model.max_seq_len // 3], dtype=xp.int32
            ),
            max_length=trainer.model.max_seq_len // 3,
        )

    # Inference test
    for k in range(5):
        run_sampling_inference(
            model=trainer.model,
            forward_fn=GPT2ForwardFn(),
            bpe=bpe,
            start_tokens=CONFIG.eval_start_string,
            max_length=int(trainer.model.max_seq_len),
            top_k=CONFIG.eval_top_k,
        )
        print("\n------------------------\n")
