"""LN remat parity tests.

Verifies the rematerialized LayerNormAffine (stores only mean/rstd; recomputes
x_hat in backward) matches a numpy reference for both fp32 and bf16 paths.

The bf16 path is CuPy-only; the fp32 path runs on either backend but is also
gated on CuPy because LayerNormAffine.forward rejects non-CuPy inputs (the
fast path is the only path that engages on CuPy bf16/fp32).
"""

from __future__ import annotations

import numpy as np
import pytest

from autograd.backend import IS_CUPY, xp
from autograd.functional import layer_norm_affine
from autograd.tensor import Tensor


def _skip_if_not_cupy():
    if not IS_CUPY:
        pytest.skip("LayerNormAffine fast path requires CuPy")


def _to_np(arr):
    return xp.to_numpy(arr) if hasattr(xp, "to_numpy") else np.asarray(arr)


def _reference_fwd_bwd(
    x_np: np.ndarray,
    gain_np: np.ndarray,
    bias_np: np.ndarray,
    grad_out_np: np.ndarray,
    epsilon: float,
):
    """Numpy reference: forward y, backward dx/dgain/dbias in fp32."""
    x32 = x_np.astype(np.float32)
    g32 = gain_np.astype(np.float32)
    b32 = bias_np.astype(np.float32)
    grad32 = grad_out_np.astype(np.float32)
    cols = x32.shape[-1]
    rows = int(x32.size // cols)
    x2d = x32.reshape(rows, cols)
    grad2d = grad32.reshape(rows, cols)

    mean = x2d.mean(axis=1, keepdims=True)
    var = ((x2d - mean) ** 2).mean(axis=1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + epsilon)
    x_hat = (x2d - mean) * rstd
    y = x_hat * g32 + b32

    # Param grads
    d_gain = (grad2d * x_hat).sum(axis=0)
    d_bias = grad2d.sum(axis=0)

    # dX via standard LayerNorm backward identity
    dx_hat = grad2d * g32
    sum1 = dx_hat.sum(axis=1, keepdims=True)
    sum2 = (dx_hat * x_hat).sum(axis=1, keepdims=True)
    dx = (rstd / cols) * (cols * dx_hat - sum1 - x_hat * sum2)

    return y.reshape(x_np.shape), dx.reshape(x_np.shape), d_gain, d_bias


def _rel_max(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = max(float(np.abs(b).max()), 1e-6)
    return float(np.abs(a - b).max() / denom)


@pytest.mark.parametrize("B,T,H", [(2, 16, 64), (4, 32, 128), (1, 8, 256)])
def test_layernorm_remat_fp32(B, T, H):
    _skip_if_not_cupy()
    np.random.seed(0)
    x_np = np.random.randn(B, T, H).astype(np.float32) * 0.5
    gain_np = np.random.randn(H).astype(np.float32) * 0.1 + 1.0
    bias_np = np.random.randn(H).astype(np.float32) * 0.05
    grad_np = np.random.randn(B, T, H).astype(np.float32)
    eps = 1e-5

    x = Tensor(xp.asarray(x_np, dtype=xp.float32))
    gain = Tensor(xp.asarray(gain_np, dtype=xp.float32))
    bias = Tensor(xp.asarray(bias_np, dtype=xp.float32))
    y = layer_norm_affine(x, gain, bias, epsilon=eps)
    y.backward(Tensor(xp.asarray(grad_np, dtype=xp.float32), requires_grad=False))

    y_ref, dx_ref, dg_ref, db_ref = _reference_fwd_bwd(
        x_np, gain_np, bias_np, grad_np, eps
    )

    assert x.grad is not None and gain.grad is not None and bias.grad is not None
    # fp32 envelope: kernel uses naive shared-mem sum, numpy uses pairwise — ~2e-5 typical.
    assert _rel_max(_to_np(y.data), y_ref) < 1e-4, (
        "fp32 forward diverges from reference"
    )
    assert _rel_max(_to_np(x.grad.data), dx_ref) < 1e-3, "fp32 dx diverges"
    assert _rel_max(_to_np(gain.grad.data), dg_ref) < 1e-3, "fp32 d_gain diverges"
    assert _rel_max(_to_np(bias.grad.data), db_ref) < 1e-3, "fp32 d_bias diverges"


@pytest.mark.parametrize("B,T,H", [(2, 16, 64), (4, 32, 128), (1, 8, 256)])
def test_layernorm_remat_bf16(B, T, H):
    _skip_if_not_cupy()
    if not hasattr(xp, "bfloat16"):
        pytest.skip("bf16 not available")
    np.random.seed(0)
    x_np = np.random.randn(B, T, H).astype(np.float32) * 0.5
    gain_np = np.random.randn(H).astype(np.float32) * 0.1 + 1.0
    bias_np = np.random.randn(H).astype(np.float32) * 0.05
    grad_np = np.random.randn(B, T, H).astype(np.float32)
    eps = 1e-5

    x = Tensor(xp.asarray(x_np).astype(xp.bfloat16))
    gain = Tensor(xp.asarray(gain_np).astype(xp.bfloat16))
    bias = Tensor(xp.asarray(bias_np).astype(xp.bfloat16))
    y = layer_norm_affine(x, gain, bias, epsilon=eps)
    grad_t = Tensor(xp.asarray(grad_np).astype(xp.bfloat16), requires_grad=False)
    y.backward(grad_t)

    # Run the reference using the bf16-truncated inputs so the comparison is fair.
    x_for_ref = _to_np(x.data).astype(np.float32)
    gain_for_ref = _to_np(gain.data).astype(np.float32)
    bias_for_ref = _to_np(bias.data).astype(np.float32)
    grad_for_ref = _to_np(grad_t.data).astype(np.float32)
    y_ref, dx_ref, dg_ref, db_ref = _reference_fwd_bwd(
        x_for_ref, gain_for_ref, bias_for_ref, grad_for_ref, eps
    )

    # bf16 has ~0.4% per cast; allow a generous envelope.
    assert x.grad is not None and gain.grad is not None and bias.grad is not None
    assert _rel_max(_to_np(y.data), y_ref) < 1e-2, "bf16 forward outside noise"
    assert _rel_max(_to_np(x.grad.data), dx_ref) < 5e-2, "bf16 dx outside noise"
    assert _rel_max(_to_np(gain.grad.data), dg_ref) < 1e-2, "bf16 d_gain outside noise"
    assert _rel_max(_to_np(bias.grad.data), db_ref) < 1e-2, "bf16 d_bias outside noise"


def test_layernorm_remat_no_xhat_attribute():
    """Sanity check: the LN node should NOT carry self.x_hat anymore."""
    _skip_if_not_cupy()
    x = Tensor(xp.random.randn(2, 4, 16).astype(xp.float32))
    gain = Tensor(xp.ones((16,), dtype=xp.float32))
    bias = Tensor(xp.zeros((16,), dtype=xp.float32))
    y = layer_norm_affine(x, gain, bias, epsilon=1e-5)
    creator = y.creator
    assert creator is not None
    assert not hasattr(creator, "x_hat"), "LN remat should not allocate self.x_hat"
    assert hasattr(creator, "mean"), "LN remat should store self.mean"
    assert hasattr(creator, "rstd"), "LN remat should store self.rstd"
