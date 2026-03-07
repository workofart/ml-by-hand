import importlib
import sys

import numpy as real_numpy
import pytest


def load_backend_module(
    monkeypatch: pytest.MonkeyPatch, backend_name: str | None = None
):
    if backend_name is None:
        monkeypatch.delenv("AUTOGRAD_BACKEND", raising=False)
    else:
        monkeypatch.setenv("AUTOGRAD_BACKEND", backend_name)

    sys.modules.pop("autograd.backend", None)
    import autograd.backend as backend_module

    return importlib.reload(backend_module)


def test_backend_defaults_to_numpy(monkeypatch: pytest.MonkeyPatch):
    backend_module = load_backend_module(monkeypatch)

    assert backend_module.current_backend_name() == "numpy"
    array_module = backend_module.get_array_module()
    arr = array_module.asarray([1.0, 2.0], dtype=backend_module.np.float32)

    assert isinstance(arr, real_numpy.ndarray)
    assert arr.dtype == real_numpy.float32


def test_unknown_backend_name_raises(monkeypatch: pytest.MonkeyPatch):
    with pytest.raises(ValueError, match="Unsupported backend"):
        load_backend_module(monkeypatch, "not-a-backend")


def test_missing_mlx_backend_raises_clear_error(monkeypatch: pytest.MonkeyPatch):
    with pytest.raises(
        ImportError, match="AUTOGRAD_BACKEND=mlx requires the 'mlx' package"
    ):
        load_backend_module(monkeypatch, "mlx")
