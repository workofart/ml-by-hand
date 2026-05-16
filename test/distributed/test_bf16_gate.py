"""bf16 hardware gate (autograd.backend.check_bf16_capability).

The gate runs unconditionally on CuPy startup. Tests:

- Non-CuPy backends pass-through (the gate is a no-op on numpy/MLX).
- When `parameter_dtype` is None / fp32 / float16 / other, the gate is a no-op.
- On CuPy, an Ampere+ device (cc >= 8.0) passes; an older device hard-fails
  with an actionable message naming compute capability and the device name.

Dtype-availability is intentionally NOT gated here; `resolve_dtype` in
`autograd.backend` already raises a clear error when neither
`cupy.bfloat16` nor `ml_dtypes.bfloat16` is reachable. The compute-
capability check stays because it catches a silent slow-training footgun
that no other check covers.

The CuPy paths are exercised against a monkey-patched cupy device so the
tests don't need real hardware.
"""

from __future__ import annotations

import types
from contextlib import contextmanager

import pytest

import autograd.backend as backend_mod


def test_noop_on_non_bf16_dtype():
    """fp32 / float16 / other dtypes (including None) don't trigger the gate."""
    backend_mod.check_bf16_capability(None)
    backend_mod.check_bf16_capability("float32")
    backend_mod.check_bf16_capability("float16")
    backend_mod.check_bf16_capability("int8")


def test_noop_on_non_cupy_backends(monkeypatch):
    """MLX/numpy backends skip the gate (each has its own bf16 path)."""
    monkeypatch.setattr(backend_mod, "IS_CUPY", False)
    # Should not raise.
    backend_mod.check_bf16_capability("bfloat16")


@contextmanager
def _fake_cupy_device(*, compute_capability: str, name: str = "FakeGPU"):
    """Install a stand-in `xp` + `IS_CUPY=True` so check_bf16_capability
    exercises its CuPy path against an in-memory fake device."""

    class _FakeDevice:
        def __init__(self) -> None:
            self.compute_capability = compute_capability
            self.attributes = {"Name": name}

    fake_xp = types.SimpleNamespace(cuda=types.SimpleNamespace(Device=_FakeDevice))

    saved_is_cupy = backend_mod.IS_CUPY
    saved_xp = backend_mod.xp
    backend_mod.IS_CUPY = True
    backend_mod.xp = fake_xp
    try:
        yield
    finally:
        backend_mod.IS_CUPY = saved_is_cupy
        backend_mod.xp = saved_xp


def test_passes_on_ampere():
    """Compute capability 8.0 is the supported floor — should not raise."""
    with _fake_cupy_device(compute_capability="80"):
        backend_mod.check_bf16_capability("bfloat16")


def test_passes_on_hopper():
    """Compute capability 9.0 should pass."""
    with _fake_cupy_device(compute_capability="90"):
        backend_mod.check_bf16_capability("bfloat16")


def test_hard_fails_on_pre_ampere():
    """Compute capability < 8.0 (e.g. Turing T4 at 7.5) should hard-fail
    with a message naming the dtype, the cc threshold, and the device."""
    with _fake_cupy_device(compute_capability="75", name="Turing-T4"):
        with pytest.raises(RuntimeError) as exc_info:
            backend_mod.check_bf16_capability("bfloat16")

    msg = str(exc_info.value)
    assert "bfloat16" in msg
    assert "8.0" in msg
    assert "7.5" in msg
    assert "Turing-T4" in msg
