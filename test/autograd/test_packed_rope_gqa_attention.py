"""Correctness tests for the packed RoPE/GQA attention fast path."""

from __future__ import annotations

import numpy as _np
import pytest

from autograd import functional, nn
from autograd.backend import IS_CUPY, xp
from autograd.tensor import Tensor
from examples.gpt_2_llama import GroupedQueryAttention, _rope_cache, apply_rope


def _skip_if_not_cupy():
    if not IS_CUPY:
        pytest.skip("packed_rope_gqa_attention is a cuDNN-only fast path")


def _to_np(a):
    return xp.to_numpy(a) if hasattr(xp, "to_numpy") else _np.asarray(a)


def _rel_max(a_data, b_data) -> float:
    a32 = _to_np(a_data.astype(xp.float32))
    b32 = _to_np(b_data.astype(xp.float32))
    denom = max(float(_np.abs(b32).max()), 1e-6)
    return float(_np.abs(a32 - b32).max() / denom)


def _repeat_heads(x: Tensor, repeats: int) -> Tensor:
    if repeats == 1:
        return x
    return Tensor.cat([x] * repeats, axis=1)


def _packed_reference(
    x: Tensor,
    weight_qkv: Tensor,
    bias_qkv: Tensor,
    weight_o: Tensor,
    bias_o: Tensor,
    cos: Tensor,
    sin: Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
) -> Tensor:
    batch_size, seq_len, hidden_size = x.shape
    head_dim = hidden_size // num_heads
    kv_dim = num_kv_heads * head_dim
    qkv = functional.linear(x, weight_qkv, bias_qkv)
    q = qkv[:, :, :hidden_size]
    k = qkv[:, :, hidden_size : hidden_size + kv_dim]
    v = qkv[:, :, hidden_size + kv_dim :]
    q = q.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    k = k.view(batch_size, seq_len, num_kv_heads, head_dim).permute(0, 2, 1, 3)
    v = v.view(batch_size, seq_len, num_kv_heads, head_dim).permute(0, 2, 1, 3)
    q = apply_rope(q, cos, sin).astype(x.data.dtype)
    k = apply_rope(k, cos, sin).astype(x.data.dtype)
    k = _repeat_heads(k, num_heads // num_kv_heads)
    v = _repeat_heads(v, num_heads // num_kv_heads)
    attn = nn.ScaledDotProductAttention(dropout_prob=0.0)(q, k, v, is_causal=True)
    attn = attn.permute(0, 2, 1, 3).view(batch_size, seq_len, hidden_size)
    return functional.linear(attn, weight_o, bias_o)


def _build_weights(*, hidden_size: int, num_heads: int, num_kv_heads: int, dtype):
    xp.random.seed(3)
    gqa = GroupedQueryAttention(
        num_heads=num_heads,
        hidden_size=hidden_size,
        num_kv_heads=num_kv_heads,
        dropout_prob=0.0,
    )
    for parameter in gqa.parameters.values():
        parameter.data = parameter.data.astype(dtype)
    weight_qkv = xp.concatenate(
        [
            gqa.q_linear.parameters["weight"].data,
            gqa.k_linear.parameters["weight"].data,
            gqa.v_linear.parameters["weight"].data,
        ],
        axis=1,
    )
    bias_qkv = xp.concatenate(
        [
            gqa.q_linear.parameters["bias"].data,
            gqa.k_linear.parameters["bias"].data,
            gqa.v_linear.parameters["bias"].data,
        ],
        axis=0,
    )
    return (
        weight_qkv,
        bias_qkv,
        gqa.fc.parameters["weight"].data,
        gqa.fc.parameters["bias"].data,
    )


def test_packed_rope_gqa_fallback_matches_split_attention_on_cpu_backends():
    batch_size, seq_len, hidden_size, num_heads, num_kv_heads = 2, 4, 16, 4, 2
    dtype = xp.float32
    xp.random.seed(5)
    x_data = (xp.random.normal(shape=(batch_size, seq_len, hidden_size)) * 0.05).astype(
        dtype
    )
    split = GroupedQueryAttention(
        num_heads=num_heads,
        hidden_size=hidden_size,
        num_kv_heads=num_kv_heads,
        dropout_prob=0.0,
        use_packed_qkv=False,
    )
    packed = GroupedQueryAttention(
        num_heads=num_heads,
        hidden_size=hidden_size,
        num_kv_heads=num_kv_heads,
        dropout_prob=0.0,
        use_packed_qkv=True,
    )
    packed.qkv_linear.parameters["weight"].data = xp.concatenate(
        [
            split.q_linear.parameters["weight"].data,
            split.k_linear.parameters["weight"].data,
            split.v_linear.parameters["weight"].data,
        ],
        axis=1,
    )
    packed.qkv_linear.parameters["bias"].data = xp.concatenate(
        [
            split.q_linear.parameters["bias"].data,
            split.k_linear.parameters["bias"].data,
            split.v_linear.parameters["bias"].data,
        ],
        axis=0,
    )
    packed.fc.parameters["weight"].data = split.fc.parameters["weight"].data
    packed.fc.parameters["bias"].data = split.fc.parameters["bias"].data
    cos_arr, sin_arr = _rope_cache(seq_len, hidden_size // num_heads)
    cos = Tensor(cos_arr.reshape(1, 1, seq_len, -1), requires_grad=False)
    sin = Tensor(sin_arr.reshape(1, 1, seq_len, -1), requires_grad=False)

    out_split = split(Tensor(x_data, requires_grad=False), cos, sin)
    out_packed = packed(Tensor(x_data, requires_grad=False), cos, sin)

    assert _rel_max(out_packed.data, out_split.data) < 1e-6


@pytest.mark.parametrize("num_heads,num_kv_heads", [(4, 2), (4, 4)])
def test_packed_rope_gqa_forward_and_backward_match_reference(
    num_heads: int,
    num_kv_heads: int,
):
    _skip_if_not_cupy()
    batch_size, seq_len, hidden_size = 2, 16, 256
    dtype = xp.bfloat16
    xp.random.seed(7)
    x_data = (xp.random.normal(shape=(batch_size, seq_len, hidden_size)) * 0.05).astype(
        dtype
    )
    weight_qkv, bias_qkv, weight_o, bias_o = _build_weights(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        dtype=dtype,
    )
    cos_arr, sin_arr = _rope_cache(seq_len, hidden_size // num_heads)
    cos = Tensor(cos_arr.reshape(1, 1, seq_len, -1), requires_grad=False)
    sin = Tensor(sin_arr.reshape(1, 1, seq_len, -1), requires_grad=False)

    x_ref = Tensor(x_data, requires_grad=True)
    W_ref = Tensor(weight_qkv, requires_grad=True)
    b_ref = Tensor(bias_qkv, requires_grad=True)
    Wo_ref = Tensor(weight_o, requires_grad=True)
    bo_ref = Tensor(bias_o, requires_grad=True)
    out_ref = _packed_reference(
        x_ref,
        W_ref,
        b_ref,
        Wo_ref,
        bo_ref,
        cos,
        sin,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )
    out_ref.sum().backward()

    x_fast = Tensor(x_data, requires_grad=True)
    W_fast = Tensor(weight_qkv, requires_grad=True)
    b_fast = Tensor(bias_qkv, requires_grad=True)
    Wo_fast = Tensor(weight_o, requires_grad=True)
    bo_fast = Tensor(bias_o, requires_grad=True)
    out_fast = functional.packed_rope_gqa_attention(
        x_fast,
        W_fast,
        b_fast,
        Wo_fast,
        bo_fast,
        cos,
        sin,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        is_causal=True,
    )
    out_fast.sum().backward()

    assert _rel_max(out_fast.data, out_ref.data) < 1e-3
    assert _rel_max(W_fast.grad.data, W_ref.grad.data) < 2e-3
    assert _rel_max(b_fast.grad.data, b_ref.grad.data) < 2e-3
    assert _rel_max(Wo_fast.grad.data, Wo_ref.grad.data) < 2e-3
    assert _rel_max(bo_fast.grad.data, bo_ref.grad.data) < 2e-3
    assert _rel_max(x_fast.grad.data, x_ref.grad.data) < 2e-2
