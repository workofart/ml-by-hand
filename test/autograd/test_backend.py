import importlib
import sys

import numpy as real_numpy
import pytest

# TODO(mlx-migration): delete this file after `autograd.backend` is removed.
# Replace any coverage that still matters with direct MLX smoke tests.


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


def load_tensor_module(monkeypatch: pytest.MonkeyPatch, backend_name: str):
    monkeypatch.setenv("AUTOGRAD_BACKEND", backend_name)
    sys.modules.pop("autograd.tensor", None)
    sys.modules.pop("autograd.backend", None)

    import autograd.tensor as tensor_module

    return importlib.reload(tensor_module)


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


def test_mlx_backend_selection(monkeypatch: pytest.MonkeyPatch):
    backend_module = load_backend_module(monkeypatch)

    if not backend_module.is_backend_available("mlx"):
        with pytest.raises(
            ImportError, match="AUTOGRAD_BACKEND=mlx requires the 'mlx' package"
        ):
            load_backend_module(monkeypatch, "mlx")
        return

    backend_module = load_backend_module(monkeypatch, "mlx")
    assert backend_module.current_backend_name() == "mlx"
    assert backend_module.get_array_module().__name__ == "mlx.core"


def test_tensor_basic_add_backward_runs_on_mlx(monkeypatch: pytest.MonkeyPatch):
    if not load_backend_module(monkeypatch).is_backend_available("mlx"):
        pytest.skip("mlx is not installed")

    tensor_module = load_tensor_module(monkeypatch, "mlx")
    Tensor = tensor_module.Tensor

    x = Tensor([1.0, 2.0], requires_grad=True)
    y = Tensor([3.0, 4.0], requires_grad=True)

    z = x + y
    z.backward()

    assert real_numpy.allclose(real_numpy.asarray(z.data), [4.0, 6.0])
    assert real_numpy.allclose(real_numpy.asarray(x.grad.data), [1.0, 1.0])
    assert real_numpy.allclose(real_numpy.asarray(y.grad.data), [1.0, 1.0])


def test_mlx_random_randn_compat(monkeypatch: pytest.MonkeyPatch):
    backend_module = load_backend_module(monkeypatch)
    if not backend_module.is_backend_available("mlx"):
        pytest.skip("mlx is not installed")

    backend_module = load_backend_module(monkeypatch, "mlx")
    backend_module.np.random.seed(42)
    samples = backend_module.np.random.randn(2, 3)

    assert samples.shape == (2, 3)
