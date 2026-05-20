"""Correctness tests for fused RMSNormAffine vs the eager implementation."""

from __future__ import annotations

import numpy as np
import pytest

from autograd.backend import IS_CUPY, xp
from autograd.tensor import Tensor

CUPY_REQUIRED = pytest.mark.skipif(
    not IS_CUPY, reason="fused RMSNorm requires the CuPy backend"
)


def _eager_rms_norm(x: Tensor, gain: Tensor, eps: float) -> Tensor:
    """Reference: original eager implementation (kept matching gpt_2_llama)."""
    from autograd.backend import LOW_PRECISION_FLOAT_DTYPES

    input_dtype = x.data.dtype
    low_precision_input = input_dtype in LOW_PRECISION_FLOAT_DTYPES
    stats_x = x.astype(xp.float32) if low_precision_input else x
    mean_sq = (stats_x * stats_x).mean(axis=-1, keepdims=True)
    x_norm = stats_x / (mean_sq + eps).sqrt()
    out = x_norm * gain.expand(x_norm.shape)
    if low_precision_input:
        out = out.astype(input_dtype)
    return out


@CUPY_REQUIRED
@pytest.mark.parametrize("shape", [(4, 128), (24, 1024, 768), (1, 768)])
def test_fused_rms_norm_fp32_matches_eager(shape):
    from autograd import functional

    x_np = np.random.randn(*shape).astype(np.float32)
    g_np = np.random.randn(shape[-1]).astype(np.float32) * 0.5 + 1.0

    x_eager = Tensor(xp.asarray(x_np), requires_grad=True)
    g_eager = Tensor(xp.asarray(g_np), requires_grad=True)
    y_eager = _eager_rms_norm(x_eager, g_eager, eps=1e-5)
    y_eager.sum().backward()

    x_fused = Tensor(xp.asarray(x_np), requires_grad=True)
    g_fused = Tensor(xp.asarray(g_np), requires_grad=True)
    y_fused = functional.rms_norm_affine(x_fused, g_fused, epsilon=1e-5)
    y_fused.sum().backward()

    np.testing.assert_allclose(
        xp.asnumpy(y_fused.data), xp.asnumpy(y_eager.data), atol=1e-5, rtol=1e-5
    )
    np.testing.assert_allclose(
        xp.asnumpy(x_fused.grad.data),
        xp.asnumpy(x_eager.grad.data),
        atol=2e-5,
        rtol=1e-4,
    )
    # Parameter-grad is a per-column reduction over many rows; fp32 cancellation
    # error scales with row count, so allow a small relative-to-magnitude floor.
    dg_eager = xp.asnumpy(g_eager.grad.data)
    dg_fused = xp.asnumpy(g_fused.grad.data)
    rms = float(np.sqrt(np.mean(dg_eager**2)))
    np.testing.assert_allclose(dg_fused, dg_eager, atol=rms * 1e-4 + 1e-5, rtol=1e-3)


@CUPY_REQUIRED
@pytest.mark.parametrize("shape", [(4, 128), (24, 1024, 768)])
def test_fused_rms_norm_bf16_matches_eager(shape):
    if not hasattr(xp, "bfloat16"):
        pytest.skip("bf16 dtype not available")
    from autograd import functional

    x_np = (np.random.randn(*shape) * 0.5).astype(np.float32)
    g_np = (np.random.randn(shape[-1]) * 0.1 + 1.0).astype(np.float32)

    x_eager = Tensor(xp.asarray(x_np).astype(xp.bfloat16), requires_grad=True)
    g_eager = Tensor(xp.asarray(g_np).astype(xp.bfloat16), requires_grad=True)
    y_eager = _eager_rms_norm(x_eager, g_eager, eps=1e-5)
    y_eager.sum().backward()

    x_fused = Tensor(xp.asarray(x_np).astype(xp.bfloat16), requires_grad=True)
    g_fused = Tensor(xp.asarray(g_np).astype(xp.bfloat16), requires_grad=True)
    y_fused = functional.rms_norm_affine(x_fused, g_fused, epsilon=1e-5)
    y_fused.sum().backward()

    # bf16 envelope: ~0.5-1% relative is normal; fused path has fewer cast
    # round-trips than the eager path, so a per-element diff of one bf16 ulp
    # is expected on a tiny fraction of elements.
    y_eager_np = xp.asnumpy(y_eager.data.astype(xp.float32))
    y_fused_np = xp.asnumpy(y_fused.data.astype(xp.float32))
    np.testing.assert_allclose(y_fused_np, y_eager_np, atol=4e-2, rtol=1e-2)

    dx_eager = xp.asnumpy(x_eager.grad.data.astype(xp.float32))
    dx_fused = xp.asnumpy(x_fused.grad.data.astype(xp.float32))
    np.testing.assert_allclose(dx_fused, dx_eager, atol=1e-2, rtol=2e-2)

    dg_eager = xp.asnumpy(g_eager.grad.data.astype(xp.float32))
    dg_fused = xp.asnumpy(g_fused.grad.data.astype(xp.float32))
    # Reductions over many rows: bf16 accumulator divergence is larger.
    # Allow ~1% relative + a small absolute floor proportional to magnitude.
    rms = float(np.sqrt(np.mean(dg_eager**2)))
    np.testing.assert_allclose(dg_fused, dg_eager, atol=rms * 0.05 + 1e-3, rtol=2e-2)
