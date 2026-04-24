from __future__ import annotations

import importlib
import os
from collections.abc import Sequence
from typing import Any

import numpy as _host_np

Array = Any
ScalarLike = int | float | bool
ArrayLike = Array | Sequence[Any] | ScalarLike

_BACKEND_MOD = {"cupy": "cupy", "mlx": "mlx.core"}


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


NAME = (os.getenv("AUTOGRAD_BACKEND") or _discover_backend_name()).lower()

if NAME == "numpy":
    xp: Any = _host_np
elif NAME in _BACKEND_MOD:
    xp = importlib.import_module(_BACKEND_MOD[NAME])
else:
    raise ValueError(f"Unknown backend: {NAME}")

IS_MLX = NAME == "mlx"
IS_CUPY = NAME == "cupy"
IS_NUMPY = NAME == "numpy"

if IS_CUPY and xp.cuda.runtime.getDeviceCount() <= 0:
    raise RuntimeError(
        "AUTOGRAD_BACKEND=cupy requested, but no CUDA device was detected"
    )

ARRAY_TYPE = type(xp.array(0, dtype=xp.float32)) if IS_MLX else xp.ndarray


# ---------------------------------------------------------------------------
# Compatibility helpers used across the repo
# ---------------------------------------------------------------------------


def eval(*xs: Any) -> None:
    if IS_MLX and xs:
        xp.eval(*xs)


def eval_backend(*values: Any) -> None:
    """
    Force MLX's lazy parameter/state updates at optimizer-step boundaries.

    Without this explicit boundary, update graphs can be evaluated later at
    less predictable synchronization points such as scalar reads or checkpoints.
    """
    if not IS_MLX:
        return

    from mlx.utils import tree_map

    def unwrap_tensor(value: Any) -> Any:
        data = getattr(value, "data", None)
        return data if hasattr(data, "shape") else value

    eval(tree_map(unwrap_tensor, values))


def _scatter_add(dst: Any, idx: Any, updates: Any):
    if IS_MLX:
        return dst.at[idx].add(updates)
    out = xp.array(dst)
    xp.add.at(out, idx, updates)
    return out


def _to_scalar(x: Any) -> Any:
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().item()
    eval(x)
    return x.item() if hasattr(x, "item") else x


# Focused backend mismatch helpers.
#
# These branches live in this module so there is a single place to inspect the
# backend surface. The repo is educational, so we keep the differences explicit
# instead of hiding them behind another abstraction layer.


def _as_strided_view(x: Any, *, shape: Any, strides: Any):
    if IS_MLX:
        return xp.as_strided(x, shape=shape, strides=strides)
    return xp.lib.stride_tricks.as_strided(
        x, shape=shape, strides=tuple(s * x.itemsize for s in strides)
    )


def _to_numpy(x: Any):
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


# -----------------------------------------------------------------------------
# RNG compatibility
# -----------------------------------------------------------------------------

_backend_random = xp.random
_host_random_seed = _host_np.random.seed
_native_random_fns = {
    n: getattr(_backend_random, n, None)
    for n in (
        "seed",
        "uniform",
        "normal",
        "randint",
        "bernoulli",
        "binomial",
        "permutation",
        "categorical",
        "choice",
    )
}
_backend_random_state: Any | None = _backend_random.key(0) if IS_MLX else None


def _shape_or_size_kwargs(shape: Any = None, size: Any = None) -> dict[str, Any]:
    s = shape if shape is not None else size
    return {"shape": s} if IS_MLX else {"size": s}


def _cast_numpy_random_output(out: Any, dtype: Any):
    if dtype is None or not IS_NUMPY:
        return out
    if hasattr(out, "astype"):
        return out.astype(dtype)
    return _host_np.dtype(dtype).type(out)


def _seed_backend_random(seed: int) -> None:
    global _backend_random_state
    fn = _native_random_fns["seed"]
    if callable(fn):
        fn(seed)
    _host_random_seed(seed)
    if IS_MLX:
        _backend_random_state = _backend_random.key(seed)


def _next_mlx_random_key() -> Any:
    global _backend_random_state
    if _backend_random_state is None:
        _backend_random_state = _backend_random.key(0)
    _backend_random_state, key = _backend_random.split(_backend_random_state, 2)
    return key


def _call_backend_random(name: str, /, *args: Any, **kwargs: Any):
    fn = _native_random_fns.get(name)
    if fn is None:
        raise RuntimeError(f"{name} is not available on backend {NAME}")
    if IS_MLX and "key" not in kwargs:
        kwargs["key"] = _next_mlx_random_key()
    return fn(*args, **kwargs)


def _sample_two_parameter_float_distribution(
    name: str,
    a: Any,
    b: Any,
    k0: str,
    k1: str,
    shape: Any = None,
    *,
    size: Any = None,
    dtype: Any = None,
):
    kw = {k0: a, k1: b, **_shape_or_size_kwargs(shape, size)}
    if dtype is not None and not IS_NUMPY:
        kw["dtype"] = dtype
    return _cast_numpy_random_output(_call_backend_random(name, **kw), dtype)


def _sample_uniform(
    low: Any = 0.0,
    high: Any = 1.0,
    shape: Any = None,
    *,
    size: Any = None,
    dtype: Any = None,
):
    return _sample_two_parameter_float_distribution(
        "uniform", low, high, "low", "high", shape, size=size, dtype=dtype
    )


def _sample_normal(
    loc: Any = 0.0,
    scale: Any = 1.0,
    shape: Any = None,
    *,
    size: Any = None,
    dtype: Any = None,
):
    return _sample_two_parameter_float_distribution(
        "normal", loc, scale, "loc", "scale", shape, size=size, dtype=dtype
    )


def _sample_randint(
    low: Any,
    high: Any = None,
    shape: Any = None,
    *,
    size: Any = None,
    dtype: Any = None,
):
    if high is None:
        low, high = 0, low
    kw = _shape_or_size_kwargs(shape, size)
    if dtype is not None:
        kw["dtype"] = dtype
    return _call_backend_random("randint", low, high, **kw)


def _sample_bernoulli(p: Any = 0.5, shape: Any = None, *, size: Any = None):
    s = shape if shape is not None else size
    if IS_MLX and _native_random_fns["bernoulli"] is not None:
        return _call_backend_random("bernoulli", p, shape=s)

    binomial = _native_random_fns.get("binomial")
    if binomial is None:
        raise RuntimeError(f"bernoulli is not available on backend {NAME}")
    return binomial(1, p, size=s)


def _sample_permutation(x: Any, *, axis: int = 0):
    return _call_backend_random("permutation", x, **({"axis": axis} if IS_MLX else {}))


def _sample_categorical(logits: Any):
    if logits.ndim != 1:
        raise ValueError("sample_categorical only supports 1-D logits")

    if _native_random_fns["categorical"] is not None:
        return _call_backend_random("categorical", logits)

    probs = xp.exp(logits - xp.max(logits))
    probs = probs / xp.sum(probs)

    choice = _native_random_fns.get("choice")
    if choice is None:
        raise RuntimeError(f"categorical sampling is not available on backend {NAME}")
    return choice(probs.shape[-1], p=probs)


def _sample_categorical_with_options(
    logits: Any,
    *,
    axis: int = -1,
    shape: Any = None,
    num_samples: Any = None,
):
    if _native_random_fns["categorical"] is None:
        if axis != -1 or shape is not None or num_samples is not None:
            raise RuntimeError(
                f"categorical options are not available on backend {NAME}"
            )
        return _sample_categorical(logits)

    return _call_backend_random(
        "categorical", logits, axis=axis, shape=shape, num_samples=num_samples
    )


def _clone_random_state(state: Any):
    if isinstance(state, tuple):
        return tuple(_clone_random_state(x) for x in state)
    if isinstance(state, list):
        return [_clone_random_state(x) for x in state]
    if isinstance(state, ARRAY_TYPE):
        return xp.array(state)
    copy = getattr(state, "copy", None)
    return copy() if callable(copy) else state


def get_random_state() -> Any:
    if IS_MLX:
        if _backend_random_state is None:
            raise RuntimeError("backend random state is not initialized")
        return _clone_random_state(_backend_random_state)

    get_state = getattr(_backend_random, "get_state", None) or getattr(
        _backend_random, "get_random_state", None
    )
    if callable(get_state):
        return _clone_random_state(get_state())

    raise RuntimeError(f"Random state capture is not supported on backend {NAME}")


def set_random_state(state: Any) -> None:
    global _backend_random_state
    if IS_MLX:
        _backend_random_state = _clone_random_state(state)
        return

    set_state = getattr(_backend_random, "set_state", None) or getattr(
        _backend_random, "set_random_state", None
    )
    if callable(set_state):
        set_state(_clone_random_state(state))
        return

    raise RuntimeError(f"Random state restore is not supported on backend {NAME}")


# -----------------------------------------------------------------------------
# Install repo-level API
# -----------------------------------------------------------------------------

if not hasattr(xp, "scatter_add"):
    xp.scatter_add = _scatter_add

xp.to_scalar = _to_scalar
xp.as_strided_view = _as_strided_view
xp.sample_categorical = _sample_categorical
xp.to_numpy = _to_numpy

_backend_random.seed = _seed_backend_random
_backend_random.normal = _sample_normal
_backend_random.uniform = _sample_uniform
_backend_random.randint = _sample_randint
_backend_random.bernoulli = _sample_bernoulli
_backend_random.permutation = _sample_permutation
_backend_random.categorical = _sample_categorical_with_options
