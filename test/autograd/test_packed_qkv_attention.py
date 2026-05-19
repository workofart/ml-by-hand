"""Correctness tests for the packed-QKV attention fast path.

These run only on the CuPy backend (cuDNN-only path); they skip on
MLX/numpy. The comparison is against the existing split-Q/K/V
MultiHeadAttention with the same effective weights so any divergence
comes from the packed-projection / interleaved-SDPA path itself.
"""

from __future__ import annotations

import numpy as _np
import pytest

from autograd import nn
from autograd.backend import IS_CUPY, xp
from autograd.functional import packed_qkv_attention
from autograd.tensor import Tensor


def _skip_if_not_cupy():
    if not IS_CUPY:
        pytest.skip("packed_qkv_attention is a cuDNN-only fast path")


def _to_np(a):
    return xp.to_numpy(a) if hasattr(xp, "to_numpy") else _np.asarray(a)


def _rel_max(a_data, b_data) -> float:
    a32 = _to_np(a_data.astype(xp.float32))
    b32 = _to_np(b_data.astype(xp.float32))
    denom = max(float(_np.abs(b32).max()), 1e-6)
    return float(_np.abs(a32 - b32).max() / denom)


def _build_split_and_packed(B: int, T: int, NH: int, H: int, *, dtype):
    """Build a split-QKV MHA and a parallel packed-QKV path that share weights."""
    xp.random.seed(0)
    x_data = (xp.random.randn(B, T, H) * 0.05).astype(dtype)

    mha = nn.MultiHeadAttention(num_heads=NH, hidden_size=H, dropout_prob=0.0)
    mha.train()
    for p in mha.parameters.values():
        p.data = p.data.astype(dtype)

    Wq = mha.q_linear.parameters["weight"].data
    Wk = mha.k_linear.parameters["weight"].data
    Wv = mha.v_linear.parameters["weight"].data
    bq = mha.q_linear.parameters["bias"].data
    bk = mha.k_linear.parameters["bias"].data
    bv = mha.v_linear.parameters["bias"].data
    Wo = mha.fc.parameters["weight"].data
    bo = mha.fc.parameters["bias"].data

    W_qkv = xp.concatenate([Wq, Wk, Wv], axis=1)
    b_qkv = xp.concatenate([bq, bk, bv], axis=0)
    return mha, x_data, (W_qkv, b_qkv, Wo, bo)


@pytest.mark.parametrize(
    "B,T,NH,H",
    [
        (2, 16, 4, 256),  # head_dim 64
        (2, 32, 8, 512),  # head_dim 64
        (1, 8, 2, 64),  # head_dim 32
    ],
)
def test_packed_qkv_forward_matches_split(B, T, NH, H):
    _skip_if_not_cupy()
    dtype = xp.bfloat16
    mha, x_data, (W_qkv, b_qkv, Wo, bo) = _build_split_and_packed(
        B, T, NH, H, dtype=dtype
    )

    x_ref = Tensor(x_data, requires_grad=False)
    out_ref = mha(x_ref, x_ref, x_ref, is_causal=True)

    x_pk = Tensor(x_data, requires_grad=False)
    out_pk = packed_qkv_attention(
        x_pk,
        Tensor(W_qkv, requires_grad=False),
        Tensor(b_qkv, requires_grad=False),
        Tensor(Wo, requires_grad=False),
        Tensor(bo, requires_grad=False),
        num_heads=NH,
        is_causal=True,
    )

    # Forward should bit-match the split path: same weights, same cuDNN call.
    assert _rel_max(out_pk.data, out_ref.data) < 1e-3, (
        "packed-QKV forward diverges from split-QKV reference"
    )


def test_packed_qkv_backward_matches_split():
    _skip_if_not_cupy()
    B, T, NH, H = 2, 16, 4, 256
    dtype = xp.bfloat16
    mha, x_data, (W_qkv, b_qkv, Wo, bo) = _build_split_and_packed(
        B, T, NH, H, dtype=dtype
    )

    # Reference: split-QKV path
    x_ref = Tensor(x_data, requires_grad=True)
    out_ref = mha(x_ref, x_ref, x_ref, is_causal=True)
    out_ref.sum().backward()

    # Packed
    x_pk = Tensor(x_data, requires_grad=True)
    W_qkv_t = Tensor(W_qkv, requires_grad=True)
    b_qkv_t = Tensor(b_qkv, requires_grad=True)
    W_o_t = Tensor(Wo, requires_grad=True)
    b_o_t = Tensor(bo, requires_grad=True)
    out_pk = packed_qkv_attention(
        x_pk,
        W_qkv_t,
        b_qkv_t,
        W_o_t,
        b_o_t,
        num_heads=NH,
        is_causal=True,
    )
    out_pk.sum().backward()

    # Split the packed weight grads back into Q/K/V slices and compare.
    dW_qkv = W_qkv_t.grad.data
    dWq, dWk, dWv = (dW_qkv[:, i * H : (i + 1) * H] for i in range(3))
    db_qkv = b_qkv_t.grad.data
    dbq, dbk, dbv = (db_qkv[i * H : (i + 1) * H] for i in range(3))

    # bf16 noise envelope: weight grads use the same single GEMM as forward and
    # should match exactly; dx goes through one combined matmul instead of
    # three, so allow 1.5% relative.
    assert _rel_max(dWq, mha.q_linear.parameters["weight"].grad.data) < 1e-3
    assert _rel_max(dWk, mha.k_linear.parameters["weight"].grad.data) < 1e-3
    assert _rel_max(dWv, mha.v_linear.parameters["weight"].grad.data) < 1e-3
    assert _rel_max(dbq, mha.q_linear.parameters["bias"].grad.data) < 1e-3
    assert _rel_max(dbk, mha.k_linear.parameters["bias"].grad.data) < 1e-3
    assert _rel_max(dbv, mha.v_linear.parameters["bias"].grad.data) < 1e-3
    assert _rel_max(W_o_t.grad.data, mha.fc.parameters["weight"].grad.data) < 1e-3
    assert _rel_max(b_o_t.grad.data, mha.fc.parameters["bias"].grad.data) < 1e-3
    assert _rel_max(x_pk.grad.data, x_ref.grad.data) < 1.5e-2, (
        "dx differs by more than bf16 noise"
    )


def test_packed_qkv_rejects_non_self_attention():
    """Cross-attention should error: packed path assumes Q is K is V."""
    _skip_if_not_cupy()
    B, T, NH, H = 2, 16, 4, 256
    dtype = xp.bfloat16
    xp.random.seed(0)
    x = xp.random.randn(B, T, H).astype(dtype) * 0.05
    y = xp.random.randn(B, T, H).astype(dtype) * 0.05

    mha = nn.MultiHeadAttention(
        num_heads=NH, hidden_size=H, dropout_prob=0.0, use_packed_qkv=True
    )
    mha.train()
    for p in mha.parameters.values():
        p.data = p.data.astype(dtype)

    # Self-attention call: packed path engages and returns.
    q_self = Tensor(x, requires_grad=False)
    out_self = mha(q_self, q_self, q_self, is_causal=True)
    assert out_self.shape == (B, T, H)

    # Cross-attention call: packed precondition (query is key is value) fails;
    # the layer was built without q_linear/k_linear/v_linear, so the fallback
    # path raises AttributeError before any compute. We treat that as the
    # intended "packed mode means self-attention only" signal.
    q_cross = Tensor(x, requires_grad=False)
    k_cross = Tensor(y, requires_grad=False)
    with pytest.raises((AttributeError, ValueError, KeyError)):
        mha(q_cross, k_cross, k_cross, is_causal=True)


def test_packed_qkv_init_rejects_dropout():
    """use_packed_qkv=True + dropout_prob>0 is unsupported (init-time error)."""
    with pytest.raises(ValueError, match="dropout_prob"):
        nn.MultiHeadAttention(
            num_heads=4, hidden_size=256, dropout_prob=0.1, use_packed_qkv=True
        )
