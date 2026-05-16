"""bf16 hardware gate (autograd.distributed.comm.check_bf16_capability).

The gate runs unconditionally on CuPy startup. Tests:

- Non-CuPy backends pass-through (the gate is a no-op on numpy/MLX).
- When `parameter_dtype` is fp32 / float32, the gate is a no-op.
- On CuPy, an Ampere+ device (cc >= 8.0) passes; an older device hard-fails
  with an actionable message naming compute capability and the device name.

The CuPy paths are exercised against a monkey-patched cupy device so the
tests don't need real hardware.
"""

from __future__ import annotations

import types
from contextlib import contextmanager

import pytest

import autograd.distributed as comm_mod


def test_noop_on_non_bf16_dtype():
    """fp32 / float16 / other dtypes don't trigger the gate."""
    comm_mod.check_bf16_capability("float32")
    comm_mod.check_bf16_capability("float16")
    comm_mod.check_bf16_capability("int8")


def test_noop_on_non_cupy_backends(monkeypatch):
    """MLX/numpy backends skip the gate (each has its own bf16 path)."""
    import autograd.backend as backend_mod

    monkeypatch.setattr(backend_mod, "IS_CUPY", False)
    # Should not raise.
    comm_mod.check_bf16_capability("bfloat16")


@contextmanager
def _fake_cupy_device(
    *, compute_capability: str, has_bfloat16: bool, name: str = "FakeGPU"
):
    """Install a stand-in `xp` + `IS_CUPY=True` so check_bf16_capability
    exercises its CuPy path against an in-memory fake device."""
    import autograd.backend as backend_mod

    fake_xp = types.SimpleNamespace()
    if has_bfloat16:
        fake_xp.bfloat16 = "bfloat16-marker"

    class _FakeDevice:
        def __init__(self) -> None:
            self.compute_capability = compute_capability
            self.attributes = {"Name": name}

    fake_xp.cuda = types.SimpleNamespace(Device=_FakeDevice)

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
    with _fake_cupy_device(compute_capability="80", has_bfloat16=True):
        comm_mod.check_bf16_capability("bfloat16")


def test_passes_on_hopper():
    """Compute capability 9.0 should pass."""
    with _fake_cupy_device(compute_capability="90", has_bfloat16=True):
        comm_mod.check_bf16_capability("bfloat16")


def test_hard_fails_on_pre_ampere():
    """Compute capability < 8.0 (e.g. Turing T4 at 7.5) should hard-fail
    with a message naming the dtype, the cc threshold, and the device."""
    with _fake_cupy_device(
        compute_capability="75", has_bfloat16=True, name="Turing-T4"
    ):
        with pytest.raises(RuntimeError) as exc_info:
            comm_mod.check_bf16_capability("bfloat16")

    msg = str(exc_info.value)
    assert "bfloat16" in msg
    assert "8.0" in msg
    assert "7.5" in msg
    assert "Turing-T4" in msg


def test_hard_fails_when_cupy_lacks_bfloat16():
    """If CuPy was built without bfloat16, raise with an upgrade message."""
    with _fake_cupy_device(compute_capability="80", has_bfloat16=False):
        with pytest.raises(RuntimeError) as exc_info:
            comm_mod.check_bf16_capability("bfloat16")

    msg = str(exc_info.value)
    assert "cupy.bfloat16" in msg
