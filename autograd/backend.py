import importlib
import os
from contextlib import contextmanager
from types import ModuleType
from typing import Iterator

BACKEND_ENV_VAR = "AUTOGRAD_BACKEND"
SUPPORTED_BACKENDS = ("numpy", "mlx")

_backend_name: str
_backend_module: ModuleType


def _normalize_backend_name(backend_name: str) -> str:
    normalized = backend_name.strip().lower()
    if normalized not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported backend '{backend_name}'. Expected one of {SUPPORTED_BACKENDS}."
        )
    return normalized


def _load_backend_module(backend_name: str) -> ModuleType:
    normalized = _normalize_backend_name(backend_name)

    if normalized == "numpy":
        import numpy as backend_module

        return backend_module

    try:
        backend_module = importlib.import_module("mlx.core")
    except ImportError as exc:
        raise ImportError(
            "AUTOGRAD_BACKEND=mlx requires the 'mlx' package to be installed."
        ) from exc

    return backend_module


def set_backend(backend_name: str) -> ModuleType:
    global _backend_name, _backend_module

    normalized = _normalize_backend_name(backend_name)
    _backend_module = _load_backend_module(normalized)
    _backend_name = normalized
    return _backend_module


def get_array_module() -> ModuleType:
    return _backend_module


def current_backend_name() -> str:
    return _backend_name


def is_backend_available(backend_name: str) -> bool:
    try:
        _load_backend_module(backend_name)
    except ImportError:
        return False
    return True


@contextmanager
def using_backend(backend_name: str) -> Iterator[ModuleType]:
    previous_backend = current_backend_name()
    set_backend(backend_name)
    try:
        yield get_array_module()
    finally:
        set_backend(previous_backend)


class BackendProxy:
    def __getattr__(self, attr: str):
        return getattr(get_array_module(), attr)

    def __dir__(self):
        return sorted(set(dir(type(self)) + dir(get_array_module())))


np = BackendProxy()
set_backend(os.environ.get(BACKEND_ENV_VAR, "numpy"))
