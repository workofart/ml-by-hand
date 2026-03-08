import importlib
import importlib.util
from pathlib import Path

import numpy as np

from autograd.backend import (
    xp,
)
from autograd.nn import extract_windows
from autograd.tensor import Tensor


def test_scatter_add_accumulates_repeated_indices():
    dst = xp.zeros(3, dtype=xp.float32)
    idx = [1, 1]
    updates = xp.asarray([2.0, 3.0], dtype=xp.float32)

    out = xp.scatter_add(dst, idx, updates)

    assert np.array_equal(xp.to_numpy(dst), np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert np.array_equal(xp.to_numpy(out), np.array([0.0, 5.0, 0.0], dtype=np.float32))


def test_as_strided_view_matches_explicit_windows():
    x = xp.arange(1, 17, dtype=xp.float32).reshape(1, 1, 4, 4)
    windows = xp.as_strided_view(
        x,
        shape=(3, 3, 1, 1, 2, 2),
        strides=(4, 1, 16, 16, 4, 1),
    )

    expected = np.zeros((3, 3, 1, 1, 2, 2), dtype=np.float32)
    x_np = np.arange(1, 17, dtype=np.float32).reshape(1, 1, 4, 4)
    for i in range(3):
        for j in range(3):
            expected[i, j, 0, 0] = x_np[0, 0, i : i + 2, j : j + 2]

    assert np.array_equal(xp.to_numpy(windows), expected)


def test_extract_windows_handles_non_contiguous_input():
    base = xp.arange(1, 33, dtype=xp.float32).reshape(1, 4, 4, 2)
    # This mirrors the NHWC -> NCHW permutation used in Conv2d forward and
    # should still produce correct sliding windows.
    x = Tensor(base).permute(0, 3, 1, 2)

    windows, _ = extract_windows(x, kernel_size=2, stride=2, padding_mode="valid")

    out = xp.to_numpy(windows.data)
    x_np = np.arange(1, 33, dtype=np.float32).reshape(1, 4, 4, 2).transpose(0, 3, 1, 2)
    expected = np.zeros((2, 2, 1, 2, 2, 2), dtype=np.float32)
    for i in range(2):
        for j in range(2):
            expected[i, j, 0] = x_np[0, :, i * 2 : i * 2 + 2, j * 2 : j * 2 + 2]

    assert np.array_equal(out, expected)


def test_random_uniform_uses_shape_keyword():
    out = xp.random.uniform(-1.0, 1.0, (2, 3))

    out_np = xp.to_numpy(out)
    assert out_np.shape == (2, 3)
    assert np.all(out_np >= -1.0)
    assert np.all(out_np <= 1.0)


def test_random_normal_uses_shape_keyword():
    out = xp.random.normal(shape=(3, 2))

    assert xp.to_numpy(out).shape == (3, 2)


def test_random_bernoulli_returns_binary_mask():
    out = xp.random.bernoulli(0.5, shape=(128,))
    values = set(np.unique(xp.to_numpy(out)).tolist())

    assert values.issubset({0, 1, False, True})


def test_tensor_item_uses_backend_scalar_conversion():
    x = Tensor(xp.asarray(3.5, dtype=xp.float32), requires_grad=True)

    assert x.item() == 3.5


def test_env_override_skips_backend_detection(monkeypatch):
    backend_path = Path(__file__).resolve().parents[2] / "autograd" / "backend.py"
    real_import_module = importlib.import_module
    probed: list[str] = []

    def tracking_import(module_name: str, package: str | None = None):
        if module_name in {"mlx.core", "cupy"}:
            probed.append(module_name)
        return real_import_module(module_name, package)

    monkeypatch.setenv("AUTOGRAD_BACKEND", "numpy")
    monkeypatch.setattr(importlib, "import_module", tracking_import)

    spec = importlib.util.spec_from_file_location(
        "_backend_env_override_test", backend_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.NAME == "numpy"
    assert probed == []
