# ruff: noqa: E402

"""Llama-style GPT-2 variant with RMSNorm, RoPE, and Grouped-Query Attention.

Quality-neutral architectural refresh on top of the dense GPT-2 baseline:
- RMSNorm replaces LayerNorm (no mean centering, no bias)
- RoPE replaces learned positional embeddings (relative-position rotation)
- GQA reduces num_kv_heads vs num_heads (Llama-2/3 style)

Uses the standard eager autograd ops (no fused Metal kernels) so this runs on
any backend supported by the repo.

Run as a script:
    AUTOGRAD_BACKEND=mlx uv run python examples/gpt_2_llama.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autograd import functional, nn, optim
from autograd.backend import (
    IS_CUPY,
    IS_MLX,
    LOW_PRECISION_FLOAT_DTYPES,
    NAME,
    Array,
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
from autograd.text.utils import generate_text
from autograd.tools.config_schema import CustomBpeConfig, TransformerTrainingConfig
from autograd.tools.trainer import LLMTrainer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Paper: https://arxiv.org/abs/1910.07467 (Zhang & Sennrich 2019).

    Compared to LayerNorm:
    - No mean subtraction; only scales by 1/sqrt(mean(x^2) + eps).
    - No bias parameter; only the gain.

    Used in Llama, Mistral, etc. Quality is comparable to LayerNorm in
    practice while saving one reduction and one parameter tensor per layer.
    """

    def __init__(
        self,
        input_size: int,
        epsilon: float = 1e-5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon
        # Llama uses gain only (no bias). Initialize to ones.
        self._parameters["gain"] = Tensor(xp.ones((input_size,), dtype=xp.float32))

    def forward(self, x: Tensor) -> Tensor:
        gain = self._parameters["gain"]
        # Fused CuPy fast path: contiguous fp32 or bf16 with matching gain dtype.
        if (
            NAME == "cupy"
            and x.data.flags.c_contiguous
            and x.data.dtype == gain.data.dtype
            and x.data.dtype in (xp.float32, *LOW_PRECISION_FLOAT_DTYPES)
            and x.data.shape[-1] == gain.data.shape[0]
        ):
            return functional.rms_norm_affine(x, gain, epsilon=self.epsilon)

        # Eager fallback (numpy / MLX / non-contiguous / dtype mismatch).
        input_dtype = x.data.dtype
        low_precision_input = input_dtype in LOW_PRECISION_FLOAT_DTYPES
        stats_x = x.astype(xp.float32) if low_precision_input else x
        mean_sq = (stats_x * stats_x).mean(axis=-1, keepdims=True)
        x_norm = stats_x / (mean_sq + self.epsilon).sqrt()
        out = x_norm * gain.expand(x_norm.shape)
        if low_precision_input:
            out = out.astype(input_dtype)
        return out


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------


def _rope_cache(seq_len: int, head_dim: int, base: float = 10000.0):
    """Precompute the cos/sin tables used by RoPE.

    Returns a pair `(cos, sin)` each of shape `(seq_len, head_dim)`.
    The table is laid out with paired-dimension repetition so that applying
    `(x * cos) + (rotate_half(x) * sin)` correctly rotates each `(2i, 2i+1)`
    pair.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires even head_dim, got {head_dim}")
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (xp.arange(0, half, dtype=xp.float32) / half))
    # positions: (seq_len,)
    positions = xp.arange(seq_len, dtype=xp.float32)
    # angles: (seq_len, half)
    angles = positions.reshape(-1, 1) * inv_freq.reshape(1, -1)
    cos = xp.cos(angles)
    sin = xp.sin(angles)
    # Repeat each angle so the table covers paired dims (i, i+half) — this
    # matches the rotate_half layout used in `apply_rope`.
    cos = xp.concatenate([cos, cos], axis=-1)  # (seq_len, head_dim)
    sin = xp.concatenate([sin, sin], axis=-1)
    return cos, sin


def _rotate_half(x: Tensor) -> Tensor:
    """Rotate the second half of the head-dim axis by 90 degrees.

    For x of shape `(..., head_dim)`, returns the array `[-x_b, x_a]` where
    `x_a` is the first half and `x_b` is the second half.
    """
    head_dim = x.shape[-1]
    half = head_dim // 2
    # Slice via xp directly (Tensor.__getitem__ would build a GetItem node,
    # which produces grads through a scatter). For RoPE the tensor is q or k
    # before the attention matmul; the grads still propagate through the
    # surrounding `cos`/`sin` operations via standard mul/add, so a raw view
    # here is enough as long as we wrap back into a Tensor with a creator.
    # Easiest: do it through Tensor ops to keep the autograd path clean.
    x_a = x[..., :half]
    x_b = x[..., half:]
    return Tensor.cat([-x_b, x_a], axis=-1)


def apply_rope(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> Tensor:
    """Apply rotary position embedding to a `(..., T, head_dim)` tensor.

    `cos` and `sin` are precomputed tables of shape `(T, head_dim)` that
    broadcast over the leading batch/head dims.
    """
    return x * cos + _rotate_half(x) * sin


# ---------------------------------------------------------------------------
# Grouped-Query Attention (GQA)
# ---------------------------------------------------------------------------


class GroupedQueryAttention(nn.Module):
    """Multi-Head Attention with optional grouping of K/V heads.

    When `num_kv_heads == num_heads`, this is plain MHA. When
    `num_kv_heads < num_heads`, K and V are projected to `num_kv_heads`
    heads and broadcast across `num_heads / num_kv_heads` query groups
    (Ainslie et al. 2023; Llama-2/3, Mistral).

    The attention math uses the standard `nn.ScaledDotProductAttention`,
    which expects K, V to have the same head count as Q. We materialize
    K and V to `num_heads` via `_repeat_heads` before calling it. The
    compute saving is in the smaller K/V projections (and their backward).
    """

    def __init__(
        self,
        num_heads: int,
        hidden_size: int,
        num_kv_heads: Optional[int] = None,
        dropout_prob: float = 0.1,
        use_packed_qkv: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
            )
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads {num_heads} must be divisible by num_kv_heads {num_kv_heads}"
            )
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.kv_dim = num_kv_heads * self.head_dim
        self.group_size = num_heads // num_kv_heads
        self.dropout_prob = dropout_prob
        self.use_packed_qkv = use_packed_qkv

        if use_packed_qkv:
            self.qkv_linear = nn.Linear(hidden_size, hidden_size + 2 * self.kv_dim)
        else:
            self.q_linear = nn.Linear(hidden_size, hidden_size)
            # Smaller K/V projections when num_kv_heads < num_heads.
            self.k_linear = nn.Linear(hidden_size, self.kv_dim)
            self.v_linear = nn.Linear(hidden_size, self.kv_dim)
        self.attention = nn.ScaledDotProductAttention(dropout_prob=dropout_prob)
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
    ) -> Tensor:
        B = x.shape[0]
        T = x.shape[1]
        input_dtype = x.data.dtype
        low_precision_input = input_dtype in LOW_PRECISION_FLOAT_DTYPES
        if (
            self.use_packed_qkv
            and self.dropout_prob == 0.0
            and NAME == "cupy"
            and low_precision_input
            and self.head_dim % 8 == 0
            and self.head_dim <= 128
        ):
            try:
                return functional.packed_rope_gqa_attention(
                    x,
                    self.qkv_linear.parameters["weight"],
                    self.qkv_linear.parameters["bias"],
                    self.fc.parameters["weight"],
                    self.fc.parameters["bias"],
                    cos,
                    sin,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    is_causal=True,
                )
            except ModuleNotFoundError:
                logger.warning(
                    "cuDNN frontend (nvidia-cudnn-frontend) not installed; "
                    "packed RoPE/GQA attention falling back to dense attention."
                )

        # Q: (B, T, num_heads * head_dim)
        if self.use_packed_qkv:
            qkv = self.qkv_linear(x)
            hidden_size = self.num_heads * self.head_dim
            q = qkv[:, :, :hidden_size]
            k = qkv[:, :, hidden_size : hidden_size + self.kv_dim]
            v = qkv[:, :, hidden_size + self.kv_dim :]
        else:
            q = self.q_linear(x)
            k = self.k_linear(x)
            v = self.v_linear(x)

        # Reshape into per-head form. Q has num_heads, K/V have num_kv_heads.
        q = q.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        # Apply RoPE to Q and K (V is not rotated). cos/sin have shape (T, head_dim)
        # and broadcast across (B, num_heads_*, T, head_dim).
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Broadcast K and V to num_heads via repeat so the standard
        # ScaledDotProductAttention sees matching head counts.
        if self.num_kv_heads != self.num_heads:
            k = _repeat_heads(k, self.group_size)
            v = _repeat_heads(v, self.group_size)

        attn = self.attention(q, k, v, is_causal=True)

        # (B, num_heads, T, head_dim) -> (B, T, num_heads * head_dim)
        attn = attn.permute(0, 2, 1, 3).view(B, T, self.num_heads * self.head_dim)
        if low_precision_input and attn.data.dtype != input_dtype:
            attn = attn.astype(input_dtype)
        return self.fc(attn)


def _repeat_heads(x: Tensor, repeats: int) -> Tensor:
    """Repeat a `(B, num_kv_heads, T, head_dim)` tensor `repeats` times along
    the head axis to produce `(B, num_kv_heads * repeats, T, head_dim)`.

    Uses `Tensor.cat([x] * repeats, axis=1)` to force a contiguous copy along
    the head axis. An earlier `(view->expand->reshape)` produced a
    non-contiguous broadcast view that crashed the attention matmul with a
    GPU page fault.
    """
    if repeats == 1:
        return x
    return Tensor.cat([x] * repeats, axis=1)


def _array_rms_norm(x, gain, epsilon: float):
    input_dtype = x.dtype
    low_precision_input = input_dtype in LOW_PRECISION_FLOAT_DTYPES
    stats_x = x.astype(xp.float32) if low_precision_input else x
    mean_sq = xp.mean(stats_x * stats_x, axis=-1, keepdims=True)
    out = stats_x / xp.sqrt(mean_sq + epsilon) * gain
    if low_precision_input:
        out = out.astype(input_dtype)
    return out


def _array_apply_rope(x, cos, sin):
    head_dim = x.shape[-1]
    half = head_dim // 2
    rotated = xp.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cos + rotated * sin


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


# ---------------------------------------------------------------------------
# SwiGLU / SiLU MLP — kept simple, optional.
# We DO NOT use SwiGLU by default because it adds a third matmul (gate) and
# slightly slows tok/s. The base FeedForward (ReLU) from examples.transformers
# is the production path and is kept here.
# ---------------------------------------------------------------------------


class FeedForwardSiLU(nn.Module):
    """Two-matmul FFN with SiLU activation (no gating).

    Same shape as the production ReLU FFN; only the activation differs.
    SiLU is smoother and slightly improves quality at no extra parameter
    cost. Slightly more compute per element (sigmoid + mul vs max(0,x)).
    """

    def __init__(self, hidden_size: int, ff_hidden_size: int, dropout_prob: float):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ff_hidden_size)
        self.fc2 = nn.Linear(ff_hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x: Tensor) -> Tensor:
        h = self.fc1(x)
        # SiLU(z) = z * sigmoid(z)
        h = h * functional.sigmoid(h)
        return self.fc2(self.dropout(h))


# ---------------------------------------------------------------------------
# Llama-style decoder block
# ---------------------------------------------------------------------------


class LlamaDecoderSublayer(nn.Module):
    """Pre-norm decoder block with RMSNorm, GQA, and ReLU FFN.

    Mirrors the structure of `examples.gpt_2.DecoderSublayer` with the
    LayerNorm replaced by RMSNorm and MultiHeadAttention replaced by
    GroupedQueryAttention.
    """

    def __init__(
        self,
        hidden_size: int,
        ff_hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        dropout_prob: float,
        use_packed_qkv: bool,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attention = GroupedQueryAttention(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            hidden_size=hidden_size,
            dropout_prob=dropout_prob,
            use_packed_qkv=use_packed_qkv,
        )
        self.norm2 = RMSNorm(hidden_size)
        from examples.transformers import FeedForward  # noqa: WPS433  (lazy)

        self.feedforward = FeedForward(
            fc_input_size=hidden_size,
            hidden_size=ff_hidden_size,
            dropout_prob=dropout_prob,
        )

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        input_dtype = x.data.dtype
        low_precision_input = input_dtype in LOW_PRECISION_FLOAT_DTYPES
        a = self.norm1(x)
        x = x + self.attention(a, cos, sin)
        if low_precision_input and x.data.dtype != input_dtype:
            x = x.astype(input_dtype)
        b = self.norm2(x)
        x = x + self.feedforward(b)
        if low_precision_input and x.data.dtype != input_dtype:
            x = x.astype(input_dtype)
        return x


# ---------------------------------------------------------------------------
# GPT2Llama model
# ---------------------------------------------------------------------------


class GPT2Llama(nn.Module):
    """GPT-2 with Llama-style updates: RMSNorm, RoPE, optional GQA.

    Architectural changes vs the baseline `examples.gpt_2.GPT2`:
    - LayerNorm -> RMSNorm (no mean centering, no bias).
    - Learned positional embedding -> RoPE applied in attention.
    - Configurable `num_kv_heads`; defaults to MHA when unset.
    - Final LayerNorm -> Final RMSNorm.

    The output head ties weights with the input embedding by default, with an
    optional untied output matrix for parameter-budget tradeoffs.
    A larger input embedding can also be projected down to `hidden_size` when
    the output head is untied.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_kv_heads: Optional[int] = None,
        max_seq_len: int = 1024,
        dropout_prob: float = 0.1,
        num_decoder_layers: int = 12,
        ff_hidden_size: Optional[int] = None,
        input_embedding_size: Optional[int] = None,
        untie_output_embedding: bool = False,
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
        self.input_embedding_size = input_embedding_size or hidden_size
        self.untie_output_embedding = untie_output_embedding
        self.activation_checkpointing = activation_checkpointing
        ff_hidden_size = ff_hidden_size or 4 * hidden_size
        if self.input_embedding_size != hidden_size and not untie_output_embedding:
            raise ValueError(
                "input_embedding_size can differ from hidden_size only when "
                "untie_output_embedding=True"
            )

        self.token_embedding = nn.Embedding(vocab_size, self.input_embedding_size)
        if self.input_embedding_size != hidden_size:
            self.embedding_projection = nn.Linear(
                self.input_embedding_size, hidden_size
            )
        if untie_output_embedding:
            self._parameters["output_weight"] = Tensor(
                xp.asarray(
                    xp.random.normal(shape=(vocab_size, hidden_size), scale=0.01),
                    dtype=xp.float32,
                ),
                requires_grad=True,
            )
        # No learned position embedding; RoPE is applied inside attention.
        self.dropout = nn.Dropout(dropout_prob)
        head_dim = hidden_size // num_attention_heads
        effective_use_packed_qkv = use_packed_qkv and dropout_prob == 0.0

        # Precompute the RoPE cos/sin table once; both are constants. We hold
        # them as Tensors with requires_grad=False so they participate in the
        # autograd graph as leaves.
        cos_arr, sin_arr = _rope_cache(max_seq_len, head_dim)
        self._rope_cos_arr = cos_arr
        self._rope_sin_arr = sin_arr

        self.sublayers = nn.ModuleList(
            [
                LlamaDecoderSublayer(
                    hidden_size=hidden_size,
                    ff_hidden_size=ff_hidden_size,
                    num_heads=num_attention_heads,
                    num_kv_heads=num_kv_heads or num_attention_heads,
                    dropout_prob=dropout_prob,
                    use_packed_qkv=effective_use_packed_qkv,
                )
                for _ in range(num_decoder_layers)
            ]
        )
        self.norm_final = RMSNorm(hidden_size)

        # Per GPT-2 §2.3: scale residual output projections by 1/sqrt(2N).
        scale = float(num_decoder_layers * 2) ** 0.5
        for sublayer in self.sublayers:
            sublayer.attention.fc.parameters["weight"].data /= scale
            sublayer.feedforward.fc2.parameters["weight"].data /= scale

        if parameter_dtype is not None:
            for parameter in self.parameters.values():
                parameter.data = parameter.data.astype(parameter_dtype)
            # Keep the RoPE tables in fp32 so the rotation precision is high;
            # broadcasting against bf16 q/k upcasts as needed.

    def _rope_tensors(self, T: int) -> Tuple[Tensor, Tensor]:
        cos = self._rope_cos_arr[:T]
        sin = self._rope_sin_arr[:T]
        # Add singleton broadcast dims for (B, num_heads, T, head_dim).
        cos = cos.reshape(1, 1, T, -1)
        sin = sin.reshape(1, 1, T, -1)
        return Tensor(cos, requires_grad=False), Tensor(sin, requires_grad=False)

    def _hidden_states(self, tokens: Tensor) -> Tensor:
        _, T = tokens.shape
        h = self.token_embedding(tokens)
        if self.input_embedding_size != self.hidden_size:
            h = self.embedding_projection(h)
        # Embedding dropout (no positional embedding to add).
        h = self.dropout(h)
        cos, sin = self._rope_tensors(T)
        for sublayer in self.sublayers:
            if self._is_training and self.activation_checkpointing:
                h = checkpoint(sublayer, h, cos, sin)
            else:
                h = sublayer(h, cos, sin)
        return self.norm_final(h)

    def _output_weight(self) -> Tensor:
        if self.untie_output_embedding:
            return self._parameters["output_weight"]
        return self.token_embedding.parameters["weight"]

    def forward(self, tokens: Tensor) -> Tensor:
        h = self._hidden_states(tokens)
        return h @ self._output_weight().T

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
    # `forward_kv` is the inference counterpart to `forward`: it bypasses the
    # autograd Tensor wrapper and calls MLX's fused kernels (`mx.fast.rms_norm`,
    # `mx.fast.rope`, `mx.fast.scaled_dot_product_attention`) directly. Profiling
    # showed ~75% of single-step decode latency was the wrapper's per-op Python
    # cost (building backward graphs we never use at inference). Combined with a
    # per-layer KV cache so each decode step processes only one new token, this
    # drops per-step time from ~16ms to ~3.5ms on M-series.
    #
    # The math here is intentionally the same as `forward` / `_hidden_states` /
    # `LlamaDecoderSublayer.forward`; correctness is validated against the
    # wrapper path in tests. MLX uses fused kernels; CUDA/CuPy uses raw backend
    # arrays to avoid building autograd graphs during generation.

    def forward_kv(
        self,
        input_ids,
        kv_cache=None,
        offset: int = 0,
    ):
        """Inference forward with optional KV cache.

        Args:
            input_ids: raw `mx.array` of shape (B, T) with the *new* tokens for
                this call. T = prompt length on the prefill call, T = 1 on each
                decode step.
            kv_cache: per-layer list of `(K, V)` tuples returned by a previous
                call, or `None` for prefill.
            offset: absolute position at which `input_ids` starts (used by RoPE
                so the new tokens see the correct positional rotation).

        Returns:
            `(logits_last, new_kv_cache)`. `logits_last` is (B, 1, vocab) — only
            the last position is projected to vocab since callers always sample
            from it.
        """
        if IS_MLX:
            return self._forward_kv_mlx(input_ids, kv_cache=kv_cache, offset=offset)
        if IS_CUPY:
            return self._forward_kv_array(input_ids, kv_cache=kv_cache, offset=offset)
        raise NotImplementedError(
            "GPT2Llama.forward_kv requires the MLX or CUDA/CuPy backend"
        )

    def _forward_kv_mlx(
        self,
        input_ids,
        kv_cache=None,
        offset: int = 0,
    ):
        import mlx.core as mx

        params = self._parameters
        h = self.token_embedding.parameters["weight"].data[input_ids]
        if self.input_embedding_size != self.hidden_size:
            ep = self.embedding_projection
            h = mx.matmul(h, ep.parameters["weight"].data) + ep.parameters["bias"].data

        new_caches = []
        for layer_idx, sublayer in enumerate(self.sublayers):
            layer_cache = None if kv_cache is None else kv_cache[layer_idx]
            h, kv = self._decode_sublayer_inference_mlx(
                h, sublayer, layer_cache, offset
            )
            new_caches.append(kv)

        h = mx.fast.rms_norm(h, self.norm_final.parameters["gain"].data, 1e-5)
        h_last = h[:, -1:, :]
        output_weight = (
            params["output_weight"].data
            if self.untie_output_embedding
            else self.token_embedding.parameters["weight"].data
        )
        logits = mx.matmul(h_last, output_weight.T)
        return logits, new_caches

    def _decode_sublayer_inference_mlx(self, h, sublayer, kv_cache_layer, offset):
        """Run one `LlamaDecoderSublayer` forward on raw MLX arrays.

        Mirrors `LlamaDecoderSublayer.forward` exactly, but operates on
        `mx.array` and uses `mx.fast.*` fused kernels. The KV cache is grown by
        concatenating the new tokens' K/V onto the per-layer prefix.
        """
        import mlx.core as mx

        attn = sublayer.attention
        ffn = sublayer.feedforward
        num_heads = attn.num_heads
        num_kv_heads = attn.num_kv_heads
        head_dim = attn.head_dim
        hidden_size = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim

        # pre-norm + packed QKV
        a = mx.fast.rms_norm(h, sublayer.norm1.parameters["gain"].data, 1e-5)
        qkv_w = attn.qkv_linear.parameters["weight"].data
        qkv_b = attn.qkv_linear.parameters["bias"].data
        qkv = mx.matmul(a, qkv_w) + qkv_b
        q = qkv[..., :hidden_size]
        k_new = qkv[..., hidden_size : hidden_size + kv_dim]
        v_new = qkv[..., hidden_size + kv_dim :]

        B, T_new = h.shape[0], h.shape[1]
        q = q.reshape(B, T_new, num_heads, head_dim).transpose(0, 2, 1, 3)
        k_new = k_new.reshape(B, T_new, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
        v_new = v_new.reshape(B, T_new, num_kv_heads, head_dim).transpose(0, 2, 1, 3)

        # RoPE the new positions only; `offset` puts them at the right absolute
        # position even when prefix tokens already sit in the cache.
        q = mx.fast.rope(
            q, dims=head_dim, traditional=False, base=10000.0, scale=1.0, offset=offset
        )
        k_new = mx.fast.rope(
            k_new,
            dims=head_dim,
            traditional=False,
            base=10000.0,
            scale=1.0,
            offset=offset,
        )

        if kv_cache_layer is None:
            k_full, v_full = k_new, v_new
        else:
            kc, vc = kv_cache_layer
            k_full = mx.concatenate([kc, k_new], axis=2)
            v_full = mx.concatenate([vc, v_new], axis=2)

        # GQA head-mapping contract: the training-path tiles via
        # `Tensor.cat([kv] * group_size, axis=1)` which produces the interleaved
        # layout `[kv0, kv1, .., kv0, kv1, ..]` mapping Q head i -> KV head
        # (i % num_kv_heads). MLX's native GQA in `mx.fast.SDPA` instead expects
        # contiguous groups, so we replicate the interleaved tiling here and
        # leave the un-tiled K/V in the cache (smaller memory footprint).
        if num_kv_heads != num_heads:
            group_size = num_heads // num_kv_heads
            k_for_attn = mx.concatenate([k_full] * group_size, axis=1)
            v_for_attn = mx.concatenate([v_full] * group_size, axis=1)
        else:
            k_for_attn, v_for_attn = k_full, v_full

        # Causal mask only matters when T_new > 1 (prompt prefill). A single
        # decode-step query is trivially "causal" over the cached prefix.
        mask = "causal" if T_new > 1 else None
        attn_out = mx.fast.scaled_dot_product_attention(
            q, k_for_attn, v_for_attn, scale=head_dim**-0.5, mask=mask
        )
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T_new, hidden_size)
        attn_out = (
            mx.matmul(attn_out, attn.fc.parameters["weight"].data)
            + attn.fc.parameters["bias"].data
        )
        h = h + attn_out

        # FFN (ReLU MLP, same as the training-path `FeedForward`).
        b = mx.fast.rms_norm(h, sublayer.norm2.parameters["gain"].data, 1e-5)
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

    def _forward_kv_array(
        self,
        input_ids,
        kv_cache=None,
        offset: int = 0,
    ):
        params = self._parameters
        h = self.token_embedding.parameters["weight"].data[input_ids]
        if self.input_embedding_size != self.hidden_size:
            ep = self.embedding_projection
            h = _array_linear(
                h,
                ep.parameters["weight"].data,
                ep.parameters["bias"].data,
            )

        new_caches = []
        for layer_idx, sublayer in enumerate(self.sublayers):
            layer_cache = None if kv_cache is None else kv_cache[layer_idx]
            h, kv = self._decode_sublayer_inference_array(
                h, sublayer, layer_cache, offset
            )
            new_caches.append(kv)

        h = _array_rms_norm(h, self.norm_final.parameters["gain"].data, 1e-5)
        h_last = h[:, -1:, :]
        output_weight = (
            params["output_weight"].data
            if self.untie_output_embedding
            else self.token_embedding.parameters["weight"].data
        )
        logits = _array_linear(h_last, output_weight.T, None)
        return logits, new_caches

    def _decode_sublayer_inference_array(self, h, sublayer, kv_cache_layer, offset):
        """Run one `LlamaDecoderSublayer` forward on raw CuPy-compatible arrays."""
        attn = sublayer.attention
        ffn = sublayer.feedforward
        num_heads = attn.num_heads
        num_kv_heads = attn.num_kv_heads
        head_dim = attn.head_dim
        hidden_size = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim

        a = _array_rms_norm(h, sublayer.norm1.parameters["gain"].data, 1e-5)
        if attn.use_packed_qkv:
            qkv = _array_linear(
                a,
                attn.qkv_linear.parameters["weight"].data,
                attn.qkv_linear.parameters["bias"].data,
            )
            q = qkv[..., :hidden_size]
            k_new = qkv[..., hidden_size : hidden_size + kv_dim]
            v_new = qkv[..., hidden_size + kv_dim :]
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
        k_new = k_new.reshape(B, T_new, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
        v_new = v_new.reshape(B, T_new, num_kv_heads, head_dim).transpose(0, 2, 1, 3)

        cos = self._rope_cos_arr[offset : offset + T_new].reshape(1, 1, T_new, -1)
        sin = self._rope_sin_arr[offset : offset + T_new].reshape(1, 1, T_new, -1)
        q = _array_apply_rope(q, cos, sin)
        k_new = _array_apply_rope(k_new, cos, sin)

        if kv_cache_layer is None:
            k_full, v_full = k_new, v_new
        else:
            kc, vc = kv_cache_layer
            k_full = xp.concatenate([kc, k_new], axis=2)
            v_full = xp.concatenate([vc, v_new], axis=2)

        if num_kv_heads != num_heads:
            group_size = num_heads // num_kv_heads
            k_for_attn = xp.concatenate([k_full] * group_size, axis=1)
            v_for_attn = xp.concatenate([v_full] * group_size, axis=1)
        else:
            k_for_attn, v_for_attn = k_full, v_full

        attn_out = _array_causal_attention(
            q,
            k_for_attn,
            v_for_attn,
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

        b = _array_rms_norm(h, sublayer.norm2.parameters["gain"].data, 1e-5)
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

    def warmup_kv(self, prompt_len: int = 1, decode_steps: int = 4) -> None:
        """JIT-compile the `forward_kv` kernels.

        The first `mx.fast.*` call for a new (shape, dtype) signature pays a
        compile cost — ~240ms for prefill and ~6ms each for the first few
        decode steps. Calling `warmup_kv()` before the timed loop runs a
        throwaway prefill + a few decode steps with the real shapes so the
        actual generation hits steady-state immediately.
        """
        if not IS_MLX:
            return
        import mlx.core as mx

        dummy_prompt = mx.array([[0] * max(1, prompt_len)], dtype=mx.int32)
        logits, caches = self.forward_kv(dummy_prompt)
        mx.eval(logits, *[t for kv in caches for t in kv])
        for step in range(decode_steps):
            logits, caches = self.forward_kv(
                mx.array([[0]], dtype=mx.int32),
                kv_cache=caches,
                offset=prompt_len + step,
            )
            mx.eval(logits, *[t for kv in caches for t in kv])


class GPT2LlamaForwardFn(nn.AbstractLLMForwardFn):
    def train(self, model: GPT2Llama, batch: CausalLMBatch) -> Tensor:
        return model(batch.input_ids)

    def sample(self, model: GPT2Llama, input_ids: Array) -> Tensor:
        return model(input_ids)


# ---------------------------------------------------------------------------
# Script: same OPENWEBTEXT_CONFIG as examples.gpt_2 with Llama variant
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    GPT2_TOKENIZER_VOCAB_SIZE = 50_257
    GPT2_PADDED_VOCAB_SIZE = 50_304
    GPT2_CUSTOM_BPE_NUM_MERGES = (
        GPT2_TOKENIZER_VOCAB_SIZE - 256 - len(BytePairEncoder.SPECIAL_TOKENS)
    )

    OPENWEBTEXT_CONFIG = TransformerTrainingConfig(
        training_run_name="openwebtext_llama",
        dataset_name="openwebtext",
        max_steps=600_000,
        max_eval_steps=100,
        checkpoint_freq=500,
        report_every_steps=250,
        global_batch_size=480,
        micro_batch_size=24,
        eval_batch_size=12,
        max_grad_norm=1.0,
        log_global_loss=True,
        model_kwargs={
            "num_attention_heads": 6,
            "num_kv_heads": 3,
            "hidden_size": 768,
            "input_embedding_size": 768,
            "ff_hidden_size": 3072,
            "untie_output_embedding": True,
            "dropout_prob": 0.0,
            "max_seq_len": 1024,
            "num_decoder_layers": 12,
            "parameter_dtype": "bfloat16",
        },
        optimizer_kwargs={
            "lr": 1e-3,
            "beta2": 0.99,
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
            num_merges=GPT2_CUSTOM_BPE_NUM_MERGES,
            encoded_data_path=f"training_data/bpe_{GPT2_CUSTOM_BPE_NUM_MERGES}_openwebtext_encoded_data.npz",
            vocab_path=f"training_data/openwebtext_vocab_{GPT2_CUSTOM_BPE_NUM_MERGES}.pkl",
            overwrite_encoded_data=False,
            overwrite_vocabulary_file=False,
            start_token="<SOS>",
            split_token="<|endoftext|>",
            parquet_shards_per_batch=32,
        ),
    )

    CONFIG = OPENWEBTEXT_CONFIG

    custom_bpe = CONFIG.custom_bpe
    if custom_bpe is None:
        raise RuntimeError("OpenWebText GPT2Llama config requires custom_bpe.")

    bpe = BytePairEncoder(
        num_merges=custom_bpe.num_merges,
        vocab_file_path=custom_bpe.vocab_path,
        encoded_data_path=custom_bpe.encoded_data_path,
        n_workers=custom_bpe.n_workers,
        min_word_freq=5,
    )
    encoded_path = Path(bpe.mmap_path)
    vocab_path = Path(custom_bpe.vocab_path)
    if (
        encoded_path.exists()
        and vocab_path.exists()
        and not custom_bpe.overwrite_encoded_data
        and not custom_bpe.overwrite_vocabulary_file
    ):
        encoded_data = BytePairEncoder.load_encoded(str(encoded_path))
    else:
        text_source = text_utils.load_openwebtext(
            parquet_shards_per_batch=custom_bpe.parquet_shards_per_batch,
            start_token=custom_bpe.start_token,
            split_token=custom_bpe.split_token,
        )
        encoded_data = bpe.prepare_data(
            text_source,
            overwrite_vocabulary_file=custom_bpe.overwrite_vocabulary_file,
            overwrite_encoded_data=custom_bpe.overwrite_encoded_data,
        )

    train_data, test_data = train_test_split(encoded_data, test_size=0.1, shuffle=False)
    del encoded_data

    CONFIG.model_kwargs["vocab_size"] = bpe.n_vocab

    def generate_eval_samples(
        model: Any,
        _forward_fn: nn.AbstractLLMForwardFn,
        _val_data_loader: Any,
        config: TransformerTrainingConfig,
    ) -> None:
        generate_text(
            model=model,
            prediction_func=GPT2LlamaForwardFn(),
            bpe=bpe,
            start_tokens=config.eval_start_string,
            max_length=min(64, int(model.max_seq_len)),
            temperature=0.8,
            top_k=config.eval_top_k,
        )

    trainer = LLMTrainer(
        model_cls=GPT2Llama,
        optimizer_cls=optim.Adam,
        loss_fn=functional.cross_entropy,
        config=CONFIG,
        forward_fn=GPT2LlamaForwardFn(),
        eval_callbacks=[generate_eval_samples],
    )

    train_dataset = TokenWindowMapDataset(
        data=train_data, window_len=trainer.model.max_seq_len + 1
    )
    test_dataset = TokenWindowMapDataset(
        data=test_data, window_len=trainer.model.max_seq_len + 1
    )
    # Build samplers, then (under DDP) wrap them so each rank sees a
    # disjoint slice of the index stream. The wrap is opt-in here — the
    # DataLoader stays DDP-agnostic — and falls through to identity on a
    # single-rank run.
    train_sampler = RandomSampler(
        train_dataset, replacement=True, num_samples=len(train_dataset)
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
