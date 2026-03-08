from __future__ import annotations

import importlib
import os
from collections.abc import Sequence
from typing import Any, Union

import numpy as _host_np

Array = Any
ScalarLike = Union[int, float, bool]
ArrayLike = Union[Array, Sequence[Any], ScalarLike]


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def _discover_backend_name() -> str:
    # This function is for *optional discovery*, not for selecting a required
    # backend. Missing optional packages should therefore fall through to the
    # next candidate instead of crashing import of `autograd.backend`.
    #
    # By contrast, once NAME is chosen below, the real backend import should
    # fail loudly if that backend is unavailable or broken.
    try:
        mx = importlib.import_module("mlx.core")
    except ModuleNotFoundError:
        mx = None
    if mx is not None:
        # MLX being importable is enough for our purposes here.
        _ = mx.default_device()
        return "mlx"

    try:
        cp = importlib.import_module("cupy")
    except ModuleNotFoundError:
        cp = None
    # For CuPy, importability alone is not enough: we only want to auto-select
    # it when there is an actual CUDA device to run on.
    if cp is not None and cp.cuda.runtime.getDeviceCount() > 0:
        return "cupy"

    return "numpy"


_env_backend = os.getenv("AUTOGRAD_BACKEND")
# Important: only call `_discover_backend_name()` when the env var is absent.
# That lets `AUTOGRAD_BACKEND=numpy` skip MLX/CuPy probing entirely.
NAME = (_env_backend if _env_backend is not None else _discover_backend_name()).lower()
IS_MLX = NAME == "mlx"
IS_NUMPY = NAME == "numpy"
IS_CUPY = NAME == "cupy"


xp: Any
if IS_NUMPY:
    xp = _host_np
    ARRAY_TYPE = xp.ndarray

elif IS_CUPY:
    xp = importlib.import_module("cupy")
    ARRAY_TYPE = xp.ndarray

elif IS_MLX:
    xp = importlib.import_module("mlx.core")
    ARRAY_TYPE = type(xp.array(0, dtype=xp.float32))

else:
    raise ValueError(f"Unknown backend: {NAME}")


# ---------------------------------------------------------------------------
# Compatibility helpers used across the repo
# ---------------------------------------------------------------------------


def _array(obj: Any, dtype: Any | None = None):
    # We want the API to be `xp.array(...)` everywhere. If a
    # backend already exposes that, use it directly; otherwise provide the
    # narrow compatibility shim here instead of spreading backend conditionals
    # across the codebase.
    return xp.asarray(obj, dtype=dtype)


def _scatter_add(dst: Any, idx: Any, updates: Any):
    # Contract: return a new array and leave ``dst`` untouched.
    # MLX already behaves that way; NumPy/CuPy need an explicit copy first.
    if IS_MLX:
        return dst.at[idx].add(updates)

    out = xp.array(dst)
    xp.add.at(out, idx, updates)
    return out


def eval(*xs: Any) -> None:
    if IS_MLX and xs:
        xp.eval(*xs)


def _to_scalar(x: Any) -> Any:
    # Keep this lower-level than `Tensor`: callers should unwrap higher-level
    # objects before they reach the backend helper.
    """Convert a backend scalar/0-d array to a plain Python scalar."""
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().item()

    eval(x)
    if hasattr(x, "item"):
        return x.item()
    return x


# Focused backend mismatch helpers.
#
# These branches live in this module so there is a single place to inspect the
# backend surface. The repo is educational, so we keep the differences explicit
# instead of hiding them behind another abstraction layer.


def _as_strided_view(x, *, shape, strides):
    # Stride units differ across backends, so keep the branch explicit here.
    if IS_MLX:
        return xp.as_strided(x, shape=shape, strides=strides)

    return xp.lib.stride_tricks.as_strided(
        x,
        shape=shape,
        strides=tuple(s * x.itemsize for s in strides),
    )


# ---------------------------------------------------------------------------
# Random helpers and compatibility
# ---------------------------------------------------------------------------


def sample_categorical(logits):
    # This helper intentionally supports only the 1-D logits case used in the
    # repo. Batched/axis-aware categorical sampling should use the backend API
    # directly once we need that wider surface.
    if logits.ndim != 1:
        raise ValueError("sample_categorical only supports 1-D logits")

    if IS_MLX:
        return xp.random.categorical(logits)

    shifted = logits - xp.max(logits)
    probs = xp.exp(shifted)
    probs = probs / xp.sum(probs)
    return xp.random.choice(probs.shape[-1], p=probs)


# The random namespace exists on every backend, but the signatures do not fully
# line up. In practice:
# - `normal` needs a backend branch because MLX requires `shape=...`
# - `uniform` and `randint` are shared directly at call sites via the positional
#   shape argument, so they do not need wrapper functions here
# - `bernoulli` is a real semantic mismatch (`bernoulli` vs `binomial`)
#
# Capture the original callables before patching the compatibility wrappers in.
# That keeps the wrapper implementations simple and avoids recursive self-calls.
_native_random_normal: Any = xp.random.normal
_native_random_bernoulli: Any = getattr(xp.random, "bernoulli", None)
_native_random_binomial: Any = getattr(xp.random, "binomial", None)


def _random_normal(
    loc: float = 0.0,
    scale: float = 1.0,
    size: Any | None = None,
    *,
    shape: Any | None = None,
):
    # NumPy/CuPy use `size=...`, while MLX uses `shape=...`. Accept both names
    # here and forward only the one the backend actually understands.
    target_shape = shape if shape is not None else size
    if IS_MLX:
        return _native_random_normal(shape=target_shape, loc=loc, scale=scale)
    return _native_random_normal(loc=loc, scale=scale, size=target_shape)


def _random_bernoulli(
    p: float,
    size: Any | None = None,
    *,
    shape: Any | None = None,
):
    # NumPy/CuPy express Bernoulli sampling via `binomial(1, p, ...)`, while
    # MLX exposes a dedicated `bernoulli(...)`. This wrapper makes the semantic
    # operation explicit at the call site and hides only that API mismatch.
    target_shape = shape if shape is not None else size
    if IS_MLX:
        return _native_random_bernoulli(p, shape=target_shape)
    return _native_random_binomial(1, p, size=target_shape)


# ---------------------------------------------------------------------------
# Host conversion
# ---------------------------------------------------------------------------


def _to_numpy(x: Any):
    # This helper exists only for host-side testing/debugging conversions.
    # MLX arrays need an explicit evaluation boundary before NumPy conversion,
    # and CuPy arrays need a device-to-host transfer. NumPy is the trivial path.
    #
    # This is intentionally a *host conversion* helper, not a generic "convert
    # to backend array" helper. The name stays specific so readers know crossing
    # to NumPy/CPU is exactly what happens here.
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().numpy()

    if IS_MLX:
        eval(x)
        return _host_np.asarray(x)

    if IS_CUPY:
        if hasattr(xp, "asnumpy"):
            return xp.asnumpy(x)
        if hasattr(x, "get"):
            return x.get()

    return _host_np.asarray(x)


# Attach the small compatibility surface we rely on across the repo.
if not hasattr(xp, "array"):
    xp.array = _array

if not hasattr(xp, "scatter_add"):
    xp.scatter_add = _scatter_add

xp.to_scalar = _to_scalar
xp.as_strided_view = _as_strided_view
xp.random.normal = _random_normal
xp.random.bernoulli = _random_bernoulli
xp.to_numpy = _to_numpy
