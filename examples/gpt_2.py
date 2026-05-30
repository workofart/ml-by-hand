# ruff: noqa: E402

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autograd import functional, nn, optim
from autograd.backend import (
    LOW_PRECISION_FLOAT_DTYPES,
    NAME,
    Array,
    ArrayLike,
    resolve_dtype,
    xp,
)
from autograd.data.collator import CausalLMWindowCollator
from autograd.data.data_loader import DataLoader
from autograd.data.dataset import TokenWindowMapDataset
from autograd.data.sampler import (
    DistributedSamplerAdapter,
    RandomSampler,
    SequentialSampler,
)
from autograd.data.types import CausalLMBatch
from autograd.data.utils import train_test_split
from autograd.distributed import is_distributed, rank, world_size
from autograd.tensor import Tensor, _matmul_autocast, checkpoint
from autograd.text import utils as text_utils
from autograd.text.tokenizer import BytePairEncoder
from autograd.text.utils import (
    generate_text,
    teacher_force,
)
from autograd.tools.config_schema import CustomBpeConfig, TransformerTrainingConfig
from autograd.tools.trainer import LLMTrainer

# The feedforward layer is the same as the original transformers
from examples.transformers import (
    FeedForward,
)


def _array_layer_norm(x, gain, bias, epsilon: float):
    input_dtype = x.dtype
    low_precision_input = input_dtype in LOW_PRECISION_FLOAT_DTYPES
    stats_x = x.astype(xp.float32) if low_precision_input else x
    mean = xp.mean(stats_x, axis=-1, keepdims=True)
    var = xp.mean((stats_x - mean) ** 2, axis=-1, keepdims=True)
    out = (stats_x - mean) / xp.sqrt(var + epsilon) * gain + bias
    if low_precision_input:
        out = out.astype(input_dtype)
    return out


def _array_linear(x, weight, bias=None):
    if x.ndim > 2 and weight.ndim == 2:
        x_2d = x.reshape(-1, x.shape[-1])
        out_2d = _matmul_autocast(x_2d, weight, bias)
        return out_2d.reshape(*x.shape[:-1], weight.shape[-1])
    return _matmul_autocast(x, weight, bias)


def _array_causal_attention(q, k, v, *, offset: int, input_dtype):
    q_scores = q.astype(xp.float32) if q.dtype in LOW_PRECISION_FLOAT_DTYPES else q
    k_scores = k.astype(xp.float32) if k.dtype in LOW_PRECISION_FLOAT_DTYPES else k
    v_values = v.astype(xp.float32) if v.dtype in LOW_PRECISION_FLOAT_DTYPES else v
    scores = (q_scores * (q.shape[-1] ** -0.5)) @ k_scores.transpose(0, 1, 3, 2)

    T_new = q.shape[2]
    T_full = k.shape[2]
    key_positions = xp.arange(T_full).reshape(1, 1, 1, T_full)
    query_positions = xp.arange(offset, offset + T_new).reshape(1, 1, T_new, 1)
    scores = xp.where(key_positions > query_positions, -1e9, scores)

    scores = scores - xp.max(scores, axis=-1, keepdims=True)
    probs = xp.exp(scores)
    probs = probs / xp.sum(probs, axis=-1, keepdims=True)
    out = probs @ v_values
    if input_dtype in LOW_PRECISION_FLOAT_DTYPES and out.dtype != input_dtype:
        out = out.astype(input_dtype)
    return out


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
        use_packed_qkv: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if isinstance(parameter_dtype, str):
            parameter_dtype = resolve_dtype(parameter_dtype)
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.activation_checkpointing = activation_checkpointing

        # Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)

        # Dropout applied after embeddings (same as GPT-1)
        self.dropout = nn.Dropout(dropout_prob)

        # Packed QKV is a zero-dropout self-attention path; dropout configs use
        # the split projections so GPT-2's default dropout remains constructible.
        effective_use_packed_qkv = use_packed_qkv and dropout_prob == 0.0
        self.sublayers = nn.ModuleList(
            [
                DecoderSublayer(
                    hidden_size=hidden_size,
                    ff_hidden_size=4 * hidden_size,  # GPT-2 typically 4 * hidden
                    num_attention_heads=num_attention_heads,
                    dropout_prob=dropout_prob,
                    use_packed_qkv=effective_use_packed_qkv,
                )
                for _ in range(num_decoder_layers)
            ]
        )

        # Final layernorm after all Transformer blocks
        # Section 3.2 "Model" in the paper
        self.layer_norm = nn.LayerNorm(hidden_size)
        # Scale only the 2 residual output projections per block by 1/sqrt(2N) (GPT-2 paper §2.3):
        # attention output (fc) and feedforward output (fc2). Q/K/V and fc1 are not scaled.
        # Each residual block adds signals that accumulate with depth; scaling the output
        # projections prevents activations from blowing up in magnitude early in training.
        scale = float(num_decoder_layers * 2) ** 0.5
        for sublayer in self.sublayers:
            sublayer.multi_head_attention.fc.parameters["weight"].data /= scale
            sublayer.feedforward.fc2.parameters["weight"].data /= scale
        if parameter_dtype is not None:
            for parameter in self.parameters.values():
                parameter.data = parameter.data.astype(parameter_dtype)

    def forward(
        self, tokens: Tensor, selected_token_indices: ArrayLike | None = None
    ) -> Tensor:
        """
        Forward pass for GPT-2.
        tokens: shape (batch_size, seq_len)
        selected_token_indices: optional flattened token positions to project.
            When provided, only those positions return logits.
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

        # GRPO only scores generated tokens, so callers can skip final
        # layernorm + vocab projection for prompt/pad positions.
        if selected_token_indices is not None:
            flat_h = h_0.reshape(-1, h_0.shape[-1])
            h_0 = flat_h[xp.asarray(selected_token_indices, dtype=xp.int32)]

        # Final normalization
        output = self.layer_norm(h_0)

        # Output logits: multiply by the transpose of the embedding matrix
        # This ties the weights with the input embedding,
        output = (
            output @ self.token_embedding.parameters["weight"].T
        )  # shape (batch_size, seq_len, vocab_size)
        return output

    def fused_model_and_loss(
        self,
        input_ids: Array,
        labels: Array,
        *,
        label_smoothing: float = 0.0,
        reduction: str = "sum",
    ) -> Tensor:
        logits = self(input_ids)
        return functional.cross_entropy_private_logits(
            logits,
            labels,
            label_smoothing=label_smoothing,
            reduction=reduction,
        )

    # ------------------------------------------------------------------
    # KV-cached inference path
    # ------------------------------------------------------------------
    # Same shape as `GPT2Llama.forward_kv`: bypass the autograd Tensor wrapper
    # at inference and route to backend arrays. MLX uses fused kernels
    # (`mx.fast.layer_norm`, `mx.fast.scaled_dot_product_attention`); CUDA/CuPy
    # uses raw arrays to avoid building autograd graphs. `generate()` in
    # `autograd/text/utils.py` auto-dispatches to this path via
    # `hasattr(model, "forward_kv")`, so no per-architecture wrapper is needed.
    #
    # Differences from the Llama path:
    # - `LayerNorm` (with bias, mean-centered) instead of `RMSNorm`.
    # - Learned absolute position embeddings added to the token embedding
    #   (no RoPE inside attention).
    # - Plain MHA (`num_kv_heads == num_heads`), so no GQA tiling.

    def forward_kv(
        self,
        input_ids,
        kv_cache=None,
        offset: int = 0,
    ):
        """Inference forward with optional KV cache.

        Args:
            input_ids: raw `mx.array` of shape (B, T) with the *new* tokens
                for this call (T = prompt length on prefill, T = 1 each decode
                step).
            kv_cache: per-layer list of `(K, V)` tuples returned by a previous
                call, or `None` for prefill.
            offset: absolute position the new tokens start at, used to index
                into the learned position embedding so decode tokens see the
                correct positional vectors.

        Returns:
            `(logits_last, new_kv_cache)`. `logits_last` is (B, 1, vocab); only
            the last position is projected since the sampler doesn't need the
            rest.
        """
        if NAME == "cupy":
            return self._forward_kv_array(input_ids, kv_cache=kv_cache, offset=offset)
        if NAME != "mlx":
            raise NotImplementedError(
                "GPT2.forward_kv requires the MLX or CUDA/CuPy backend"
            )
        import mlx.core as mx  # pyright: ignore[reportMissingImports]

        T_new = input_ids.shape[1]
        token_emb = self.token_embedding.parameters["weight"].data[input_ids]
        positions = mx.arange(offset, offset + T_new, dtype=mx.int32)
        pos_emb = self.position_embedding.parameters["weight"].data[positions]
        h = token_emb + pos_emb  # (B, T_new, hidden)

        new_caches = []
        for layer_idx, sublayer in enumerate(self.sublayers):
            layer_cache = None if kv_cache is None else kv_cache[layer_idx]
            h, kv = self._decode_sublayer_inference(h, sublayer, layer_cache)
            new_caches.append(kv)

        h = mx.fast.layer_norm(
            h,
            self.layer_norm.parameters["gain"].data,
            self.layer_norm.parameters["bias"].data,
            1e-5,
        )
        h_last = h[:, -1:, :]
        # GPT-2 always ties output to input embedding.
        logits = mx.matmul(h_last, self.token_embedding.parameters["weight"].data.T)
        return logits, new_caches

    def _forward_kv_array(
        self,
        input_ids,
        kv_cache=None,
        offset: int = 0,
    ):
        T_new = input_ids.shape[1]
        token_emb = self.token_embedding.parameters["weight"].data[input_ids]
        positions = xp.arange(offset, offset + T_new, dtype=xp.int32)
        pos_emb = self.position_embedding.parameters["weight"].data[positions]
        h = token_emb + pos_emb

        new_caches = []
        for layer_idx, sublayer in enumerate(self.sublayers):
            layer_cache = None if kv_cache is None else kv_cache[layer_idx]
            h, kv = self._decode_sublayer_inference_array(
                h,
                sublayer,
                layer_cache,
                offset,
            )
            new_caches.append(kv)

        h = _array_layer_norm(
            h,
            self.layer_norm.parameters["gain"].data,
            self.layer_norm.parameters["bias"].data,
            1e-5,
        )
        h_last = h[:, -1:, :]
        logits = _array_linear(
            h_last,
            self.token_embedding.parameters["weight"].data.T,
            None,
        )
        return logits, new_caches

    def _decode_sublayer_inference(self, h, sublayer, kv_cache_layer):
        """Run one `DecoderSublayer` forward on raw MLX arrays.

        Mirrors `DecoderSublayer.forward` exactly, but operates on `mx.array`
        and uses `mx.fast.*` fused kernels. The KV cache is grown by
        concatenating the new tokens' K/V onto the per-layer prefix.
        """
        import mlx.core as mx  # pyright: ignore[reportMissingImports]

        attn = sublayer.multi_head_attention
        ffn = sublayer.feedforward
        num_heads = attn.num_heads
        head_dim = attn.attention_size
        hidden_size = num_heads * head_dim

        # pre-norm + packed QKV
        a = mx.fast.layer_norm(
            h,
            sublayer.layer_norm1.parameters["gain"].data,
            sublayer.layer_norm1.parameters["bias"].data,
            1e-5,
        )
        qkv_w = attn.qkv_linear.parameters["weight"].data
        qkv_b = attn.qkv_linear.parameters["bias"].data
        qkv = mx.matmul(a, qkv_w) + qkv_b  # (B, T_new, 3 * hidden)
        q = qkv[..., :hidden_size]
        k_new = qkv[..., hidden_size : 2 * hidden_size]
        v_new = qkv[..., 2 * hidden_size :]

        B, T_new = h.shape[0], h.shape[1]
        q = q.reshape(B, T_new, num_heads, head_dim).transpose(0, 2, 1, 3)
        k_new = k_new.reshape(B, T_new, num_heads, head_dim).transpose(0, 2, 1, 3)
        v_new = v_new.reshape(B, T_new, num_heads, head_dim).transpose(0, 2, 1, 3)

        if kv_cache_layer is None:
            k_full, v_full = k_new, v_new
        else:
            kc, vc = kv_cache_layer
            k_full = mx.concatenate([kc, k_new], axis=2)
            v_full = mx.concatenate([vc, v_new], axis=2)

        # Plain MHA: N_kv == N_q so no tiling. Causal mask only matters when
        # T_new > 1 (prompt prefill); a single decode-step query is trivially
        # "causal" over the cached prefix.
        mask = "causal" if T_new > 1 else None
        attn_out = mx.fast.scaled_dot_product_attention(
            q, k_full, v_full, scale=head_dim**-0.5, mask=mask
        )
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T_new, hidden_size)
        attn_out = (
            mx.matmul(attn_out, attn.fc.parameters["weight"].data)
            + attn.fc.parameters["bias"].data
        )
        h = h + attn_out

        # FFN (ReLU MLP, same as the training-path `FeedForward`).
        b = mx.fast.layer_norm(
            h,
            sublayer.layer_norm2.parameters["gain"].data,
            sublayer.layer_norm2.parameters["bias"].data,
            1e-5,
        )
        inter = (
            mx.matmul(b, ffn.fc1.parameters["weight"].data)
            + ffn.fc1.parameters["bias"].data
        )
        inter = mx.maximum(inter, mx.array(0.0, dtype=inter.dtype))
        out = (
            mx.matmul(inter, ffn.fc2.parameters["weight"].data)
            + ffn.fc2.parameters["bias"].data
        )
        h = h + out
        return h, (k_full, v_full)

    def _decode_sublayer_inference_array(self, h, sublayer, kv_cache_layer, offset):
        """Run one `DecoderSublayer` forward on raw CuPy-compatible arrays."""
        attn = sublayer.multi_head_attention
        ffn = sublayer.feedforward
        num_heads = attn.num_heads
        head_dim = attn.attention_size
        hidden_size = num_heads * head_dim

        a = _array_layer_norm(
            h,
            sublayer.layer_norm1.parameters["gain"].data,
            sublayer.layer_norm1.parameters["bias"].data,
            1e-5,
        )
        if attn.use_packed_qkv:
            qkv = _array_linear(
                a,
                attn.qkv_linear.parameters["weight"].data,
                attn.qkv_linear.parameters["bias"].data,
            )
            q = qkv[..., :hidden_size]
            k_new = qkv[..., hidden_size : 2 * hidden_size]
            v_new = qkv[..., 2 * hidden_size :]
        else:
            q = _array_linear(
                a,
                attn.q_linear.parameters["weight"].data,
                attn.q_linear.parameters["bias"].data,
            )
            k_new = _array_linear(
                a,
                attn.k_linear.parameters["weight"].data,
                attn.k_linear.parameters["bias"].data,
            )
            v_new = _array_linear(
                a,
                attn.v_linear.parameters["weight"].data,
                attn.v_linear.parameters["bias"].data,
            )

        B, T_new = h.shape[0], h.shape[1]
        q = q.reshape(B, T_new, num_heads, head_dim).transpose(0, 2, 1, 3)
        k_new = k_new.reshape(B, T_new, num_heads, head_dim).transpose(0, 2, 1, 3)
        v_new = v_new.reshape(B, T_new, num_heads, head_dim).transpose(0, 2, 1, 3)

        if kv_cache_layer is None:
            k_full, v_full = k_new, v_new
        else:
            kc, vc = kv_cache_layer
            k_full = xp.concatenate([kc, k_new], axis=2)
            v_full = xp.concatenate([vc, v_new], axis=2)

        attn_out = _array_causal_attention(
            q,
            k_full,
            v_full,
            offset=offset,
            input_dtype=h.dtype,
        )
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T_new, hidden_size)
        if h.dtype in LOW_PRECISION_FLOAT_DTYPES and attn_out.dtype != h.dtype:
            attn_out = attn_out.astype(h.dtype)
        attn_out = _array_linear(
            attn_out,
            attn.fc.parameters["weight"].data,
            attn.fc.parameters["bias"].data,
        )
        h = h + attn_out

        b = _array_layer_norm(
            h,
            sublayer.layer_norm2.parameters["gain"].data,
            sublayer.layer_norm2.parameters["bias"].data,
            1e-5,
        )
        inter = _array_linear(
            b,
            ffn.fc1.parameters["weight"].data,
            ffn.fc1.parameters["bias"].data,
        )
        inter = xp.maximum(inter, xp.array(0.0, dtype=inter.dtype))
        out = _array_linear(
            inter,
            ffn.fc2.parameters["weight"].data,
            ffn.fc2.parameters["bias"].data,
        )
        h = h + out
        return h, (k_full, v_full)

    def warmup_kv(
        self,
        prompt_len: int = 1,
        decode_steps: int = 4,
        batch_size: int = 1,
    ) -> None:
        """JIT-compile the `forward_kv` kernels.

        Same role as `GPT2Llama.warmup_kv`: a throwaway prefill + a few decode
        steps with the real shapes so the actual generation loop hits
        steady-state immediately instead of paying ~240ms of JIT cost on the
        first prefill.
        """
        if NAME != "mlx":
            return
        import mlx.core as mx  # pyright: ignore[reportMissingImports]

        batch_size = max(1, int(batch_size))
        prompt_len = max(1, int(prompt_len))
        dummy_prompt = mx.array([[0] * prompt_len], dtype=mx.int32)
        logits, caches = self.forward_kv(dummy_prompt)
        mx.eval(logits, *[t for kv in caches for t in kv])
        if batch_size > 1:
            # GRPO rolls out many completions for one prompt; repeat the
            # prefilled cache so warmup covers the real batched decode shape.
            logits = mx.repeat(logits, batch_size, axis=0)
            caches = [
                (mx.repeat(k, batch_size, axis=0), mx.repeat(v, batch_size, axis=0))
                for k, v in caches
            ]
            mx.eval(logits, *[t for kv in caches for t in kv])
        for step in range(decode_steps):
            logits, caches = self.forward_kv(
                mx.zeros((batch_size, 1), dtype=mx.int32),
                kv_cache=caches,
                offset=prompt_len + step,
            )
            mx.eval(logits, *[t for kv in caches for t in kv])


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
        use_packed_qkv: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # First LayerNorm (for the attention sub-layer)
        self.layer_norm1 = nn.LayerNorm(hidden_size)

        self.multi_head_attention = nn.MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            dropout_prob=dropout_prob,
            use_packed_qkv=use_packed_qkv,
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
        if low_precision_input and x.data.dtype != input_dtype:
            # Dense attention can promote through its fp32 mask path; cast the residual
            # stream back so later matmuls keep the intended low-precision activations.
            x = x.astype(input_dtype)

        # Pre-norm before feed-forward
        b = self.layer_norm2(x)
        x = x + self.feedforward(b)
        if low_precision_input and x.data.dtype != input_dtype:
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
        micro_batch_size=train_global_batch_size // 4,
        eval_batch_size=2,
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
            encoded_data_path="training_data/bpe_3000_shakespeare_bos_eos_encoded_data.npz",
            vocab_path="training_data/shakespeare_vocab_3000.pkl",
            overwrite_encoded_data=False,
            overwrite_vocabulary_file=False,
            start_token="<SOS>",
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
        global_batch_size=76,
        micro_batch_size=19,
        eval_batch_size=9,
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
            encoded_data_path="training_data/bpe_12000_wiki_simple_bos_eos_encoded_data.npz",
            vocab_path="training_data/wikipedia_simpleenglish_vocab_12000.pkl",
            overwrite_encoded_data=False,
            overwrite_vocabulary_file=False,
            start_token="<SOS>",
            split_token="<|endoftext|>",
        ),
    )

    GPT2_TOKENIZER_VOCAB_SIZE = 50_257
    GPT2_PADDED_VOCAB_SIZE = 50_304
    GPT2_CUSTOM_BPE_NUM_MERGES = (
        GPT2_TOKENIZER_VOCAB_SIZE - 256 - len(BytePairEncoder.SPECIAL_TOKENS)
    )
    OPENWEBTEXT_CONFIG = TransformerTrainingConfig(
        training_run_name="openwebtext_gpt2_124m_baseline",
        dataset_name="openwebtext",
        max_steps=600_000,
        max_eval_steps=100,
        checkpoint_freq=500,
        report_every_steps=250,
        global_batch_size=480,
        micro_batch_size=48,
        eval_batch_size=24,
        max_grad_norm=1.0,
        model_kwargs={
            "num_attention_heads": 12,
            "hidden_size": 768,
            "vocab_size": GPT2_PADDED_VOCAB_SIZE,
            # Match nanoGPT's OpenWebText GPT-2 reproduction: pretraining uses
            # no dropout, and the loss config below uses plain next-token CE.
            "dropout_prob": 0.0,
            "max_seq_len": 1024,
            "num_decoder_layers": 12,
            "activation_checkpointing": False,
            "parameter_dtype": "bfloat16",
            "use_packed_qkv": True,
        },
        optimizer_kwargs={
            "lr": 6e-4,
            "beta2": 0.95,
            "weight_decay": 0.1,
            "lr_scheduler_kwargs": {
                "lr_scheduler_cls": optim.CosineScheduler,
                "warmup_steps": 1000,
                "lr_decay_iters": 600_000,
            },
        },
        resume_epoch=None,
        teacher_forcing=False,
        label_smoothing=0.0,
        eval_start_string="The",
        custom_bpe=CustomBpeConfig(
            # Matches GPT-2's 50,257-token cardinality with this repo's custom
            # BPE. Exact GPT-2 tokenizer parity requires GPT-2's merge ranks.
            num_merges=GPT2_CUSTOM_BPE_NUM_MERGES,
            encoded_data_path=(
                f"training_data/bpe_{GPT2_CUSTOM_BPE_NUM_MERGES}_"
                "openwebtext_encoded_data.npz"
            ),
            vocab_path=(
                f"training_data/openwebtext_vocab_{GPT2_CUSTOM_BPE_NUM_MERGES}.pkl"
            ),
            overwrite_encoded_data=False,
            overwrite_vocabulary_file=False,
            start_token="",
            split_token="<|endoftext|>",
            parquet_shards_per_batch=32,
        ),
    )

    CONFIG = OPENWEBTEXT_CONFIG

    logger = logging.getLogger(__name__)

    if CONFIG.custom_bpe:
        # Create a Byte Pair Encoder and prepare data
        bpe = BytePairEncoder(
            num_merges=CONFIG.custom_bpe.num_merges,
            vocab_file_path=CONFIG.custom_bpe.vocab_path,
            encoded_data_path=CONFIG.custom_bpe.encoded_data_path,
            n_workers=CONFIG.custom_bpe.n_workers,
            min_word_freq=5,  # this is about 99.7% coverage
        )
        encoded_path = Path(bpe.mmap_path)
        vocab_path = Path(CONFIG.custom_bpe.vocab_path)
        use_cached_encoded_data = (
            encoded_path.exists()
            and vocab_path.exists()
            and not CONFIG.custom_bpe.overwrite_encoded_data
            and not CONFIG.custom_bpe.overwrite_vocabulary_file
        )
        if use_cached_encoded_data:
            logger.info(
                "Found existing encoded data at '%s', loading it without fetching raw data.",
                encoded_path,
            )
            encoded_data = BytePairEncoder.load_encoded(str(encoded_path))
            logger.info(f"Vocabulary size: {bpe.n_vocab}")
            logger.info(f"Encoded data length: {len(encoded_data)}")
        else:
            if CONFIG.dataset_name == "openwebtext":
                text_source = text_utils.load_openwebtext(
                    parquet_shards_per_batch=(
                        CONFIG.custom_bpe.parquet_shards_per_batch
                    ),
                    start_token=CONFIG.custom_bpe.start_token,
                    split_token=CONFIG.custom_bpe.split_token,
                )
            elif CONFIG.dataset_name == "wiki_simple_english":
                text_source = [
                    text_utils.format_document_for_causal_lm(
                        text_utils.load_wiki_simple(),
                        start_token=CONFIG.custom_bpe.start_token,
                        split_token=CONFIG.custom_bpe.split_token,
                    )
                ]
            else:
                text_source = [
                    text_utils.format_document_for_causal_lm(
                        text_utils.load_shakespeare_mini(),
                        start_token=CONFIG.custom_bpe.start_token,
                        split_token=CONFIG.custom_bpe.split_token,
                    )
                ]
            encoded_data = bpe.prepare_data(
                text_source,
                overwrite_vocabulary_file=CONFIG.custom_bpe.overwrite_vocabulary_file,
                overwrite_encoded_data=CONFIG.custom_bpe.overwrite_encoded_data,
            )
    else:
        raise ValueError(
            "Please supply a custom_bpe config. Check out CustomBpeConfig for more details."
        )

    train_data, test_data = train_test_split(encoded_data, test_size=0.1, shuffle=False)
    del encoded_data

    def generate_eval_samples(
        model,
        _forward_fn,
        _val_data_loader,
        config: TransformerTrainingConfig,
    ) -> None:
        generate_text(
            model=model,
            prediction_func=GPT2ForwardFn(),
            bpe=bpe,
            start_tokens=config.eval_start_string,
            max_length=min(64, int(model.max_seq_len)),
            temperature=0.8,
            top_k=config.eval_top_k,
        )

    trainer = LLMTrainer(
        model_cls=GPT2,
        optimizer_cls=optim.Adam,
        loss_fn=functional.cross_entropy,
        config=CONFIG,
        forward_fn=GPT2ForwardFn(),
        eval_callbacks=[generate_eval_samples],
    )

    train_dataset = TokenWindowMapDataset(
        data=train_data,
        # CausalLMWindowCollator shifts one token to build input_ids/labels,
        # so a length-T model context needs a raw window of length T + 1.
        window_len=trainer.model.max_seq_len + 1,
    )
    test_dataset = TokenWindowMapDataset(
        data=test_data,
        # CausalLMWindowCollator shifts one token to build input_ids/labels,
        # so a length-T model context needs a raw window of length T + 1.
        window_len=trainer.model.max_seq_len + 1,
    )
    train_sampler = RandomSampler(
        train_dataset,
        replacement=True,
        num_samples=len(train_dataset),
    )
    test_sampler = SequentialSampler(test_dataset)
    if is_distributed():
        train_sampler = DistributedSamplerAdapter(
            train_sampler, rank=rank(), world_size=world_size()
        )
        test_sampler = DistributedSamplerAdapter(
            test_sampler, rank=rank(), world_size=world_size()
        )

    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=CONFIG.micro_batch_size,
        collator=CausalLMWindowCollator(),
        sampler=train_sampler,
    )
    test_data_loader = DataLoader(
        dataset=test_dataset,
        batch_size=CONFIG.eval_batch_size,
        collator=CausalLMWindowCollator(),
        sampler=test_sampler,
    )

    trainer.fit(train_data_loader, test_data_loader)

    if CONFIG.teacher_forcing:
        teacher_force(
            model=trainer.model,
            prediction_func=GPT2ForwardFn(),
            bpe=bpe,
            groundtruth_data=xp.array(
                test_data[: trainer.model.max_seq_len // 3], dtype=xp.int32
            ),
            max_length=trainer.model.max_seq_len // 3,
        )
