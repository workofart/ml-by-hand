import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autograd.backend import (
    get_random_state,
    materialize,
    set_random_state,
    xp,
)
from autograd.nn import extract_windows
from autograd.tensor import Tensor


def _load_backend_module(
    monkeypatch, *, env_backend=None, fake_cupy=None, fake_ml_dtypes=None
):
    backend_path = Path(__file__).resolve().parents[2] / "autograd" / "backend.py"
    real_import_module = importlib.import_module

    if env_backend is None:
        monkeypatch.delenv("AUTOGRAD_BACKEND", raising=False)
    else:
        monkeypatch.setenv("AUTOGRAD_BACKEND", env_backend)

    def tracking_import(module_name: str, package: str | None = None):
        if module_name == "mlx.core":
            raise ModuleNotFoundError(module_name)
        if module_name == "cupy" and fake_cupy is not None:
            return fake_cupy
        if module_name == "ml_dtypes" and fake_ml_dtypes is not None:
            return fake_ml_dtypes
        return real_import_module(module_name, package)

    monkeypatch.setattr(importlib, "import_module", tracking_import)

    spec = importlib.util.spec_from_file_location("_backend_test_module", backend_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_cupy_module(device_count: int):
    random = SimpleNamespace(
        seed=np.random.seed,
        uniform=np.random.uniform,
        normal=np.random.normal,
        randint=np.random.randint,
        binomial=np.random.binomial,
        permutation=np.random.permutation,
        choice=np.random.choice,
    )
    return SimpleNamespace(
        ndarray=np.ndarray,
        array=np.array,
        asarray=np.asarray,
        float16=np.float16,
        random=random,
        cuda=SimpleNamespace(
            runtime=SimpleNamespace(getDeviceCount=lambda: device_count),
            # autograd/backend.py calls `xp.cuda.Device(LOCAL_RANK).use()`
            # at import for DDP device pinning; the fake needs a no-op
            # stand-in or the import-time pin raises.
            Device=lambda _idx: SimpleNamespace(use=lambda: None),
        ),
        asnumpy=np.asarray,
    )


def test_seed_env_initializes_backend_and_host_rng(monkeypatch):
    monkeypatch.setenv("SEED", "123")
    module = _load_backend_module(monkeypatch, env_backend="numpy")

    got_backend = module.xp.random.randint(0, 1000, size=5)
    got_host = np.random.randint(0, 1000, size=5)

    np.random.seed(123)
    expected_backend = np.random.randint(0, 1000, size=5)
    expected_host = np.random.randint(0, 1000, size=5)
    assert np.array_equal(got_backend, expected_backend)
    assert np.array_equal(got_host, expected_host)


def test_scatter_add_accumulates_repeated_indices():
    dst = xp.zeros(3, dtype=xp.float32)
    idx = [1, 1]
    updates = xp.asarray([2.0, 3.0], dtype=xp.float32)

    out = xp.scatter_add(dst, idx, updates)

    assert np.array_equal(xp.to_numpy(dst), np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert np.array_equal(xp.to_numpy(out), np.array([0.0, 5.0, 0.0], dtype=np.float32))


def test_scatter_add_accumulates_cupy_bfloat16_via_float32(monkeypatch):
    import autograd.backend as backend

    class FakeArray:
        def __init__(self, values, dtype):
            self.values = np.asarray(values, dtype=np.float32)
            self.dtype = dtype

        def astype(self, dtype):
            return FakeArray(self.values, dtype)

    class FakeAdd:
        @staticmethod
        def at(out, idx, updates):
            if out.dtype == "bfloat16":
                raise TypeError("cupy.add.at does not support bfloat16")
            np.add.at(out.values, idx, updates.values)

    class FakeXP:
        float32 = "float32"
        add = FakeAdd()

        @staticmethod
        def array(value):
            return FakeArray(value.values.copy(), value.dtype)

    monkeypatch.setattr(backend, "IS_MLX", False)
    monkeypatch.setattr(backend, "IS_CUPY", True)
    monkeypatch.setattr(backend, "LOW_PRECISION_FLOAT_DTYPES", ("bfloat16",))
    monkeypatch.setattr(backend, "xp", FakeXP())

    dst = FakeArray([0.0, 0.0, 0.0], "bfloat16")
    updates = FakeArray([2.0, 3.0], "bfloat16")

    out = backend._scatter_add(dst, [1, 1], updates)

    assert out.dtype == "bfloat16"
    assert np.array_equal(out.values, np.array([0.0, 5.0, 0.0], dtype=np.float32))


def test_materialize_collects_nested_arrays_and_tensor_data(monkeypatch):
    import autograd.backend as backend

    arrays = [
        np.asarray([1.0], dtype=np.float32),
        np.asarray([2.0], dtype=np.float32),
        np.asarray([3.0], dtype=np.float32),
    ]
    observed = []

    monkeypatch.setattr(backend, "IS_MLX", True)
    monkeypatch.setattr(backend, "ARRAY_TYPE", np.ndarray)
    monkeypatch.setattr(
        backend,
        "xp",
        SimpleNamespace(eval=lambda *xs: observed.extend(xs)),
    )

    nested = {
        "tensor": SimpleNamespace(data=arrays[0]),
        "array": arrays[1],
        "ignored": [None, "abc", 7],
    }

    materialize(nested, (arrays[2],))

    assert observed == arrays


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


def test_sample_categorical_passes_scalar_size_to_choice(monkeypatch):
    import autograd.backend as backend

    observed = {}

    def fake_choice(a, size=None, replace=True, p=None):
        observed["a"] = a
        observed["size"] = size
        observed["replace"] = replace
        observed["p"] = p
        return np.asarray(1)

    fake_xp = SimpleNamespace(
        exp=np.exp,
        max=np.max,
        sum=np.sum,
    )
    monkeypatch.setattr(backend, "xp", fake_xp)
    monkeypatch.setitem(backend._native_random_fns, "categorical", None)
    monkeypatch.setitem(backend._native_random_fns, "choice", fake_choice)

    out = backend._sample_categorical(np.asarray([0.0, 1.0, 2.0]))

    assert int(out) == 1
    assert observed["a"] == 3
    assert observed["size"] == ()
    assert observed["replace"] is True
    assert np.isclose(observed["p"].sum(), 1.0)


def test_active_backend_random_state_round_trip_replays_sample():
    xp.random.seed(123)
    saved_state = get_random_state()
    first = xp.random.bernoulli(0.5, shape=(32,))
    set_random_state(saved_state)
    second = xp.random.bernoulli(0.5, shape=(32,))

    assert np.array_equal(xp.to_numpy(first), xp.to_numpy(second))


def test_backend_seed_replays_host_numpy_random_sample():
    xp.random.seed(123)
    first = np.random.randint(0, 100, size=8, dtype=np.int32)
    xp.random.seed(123)
    second = np.random.randint(0, 100, size=8, dtype=np.int32)

    assert np.array_equal(first, second)


def test_numpy_random_state_round_trip_replays_sample(monkeypatch):
    module = _load_backend_module(monkeypatch, env_backend="numpy")

    module.xp.random.seed(123)
    saved_state = module.get_random_state()
    first = module.xp.random.bernoulli(0.5, shape=(32,))
    module.set_random_state(saved_state)
    second = module.xp.random.bernoulli(0.5, shape=(32,))

    assert np.array_equal(module.xp.to_numpy(first), module.xp.to_numpy(second))


def test_tensor_item_uses_backend_scalar_conversion():
    x = Tensor(xp.asarray(3.5, dtype=xp.float32), requires_grad=True)

    assert x.item() == 3.5


def test_env_override_skips_backend_detection(monkeypatch):
    probed: list[str] = []
    real_import_module = importlib.import_module

    def tracking_import(module_name: str, package: str | None = None):
        if module_name in {"mlx.core", "cupy"}:
            probed.append(module_name)
        return real_import_module(module_name, package)

    monkeypatch.setenv("AUTOGRAD_BACKEND", "numpy")
    monkeypatch.setattr(importlib, "import_module", tracking_import)

    backend_path = Path(__file__).resolve().parents[2] / "autograd" / "backend.py"
    spec = importlib.util.spec_from_file_location(
        "_backend_env_override_test", backend_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.NAME == "numpy"
    assert probed == []


def test_auto_detect_selects_cupy_when_cuda_is_available(monkeypatch):
    module = _load_backend_module(
        monkeypatch,
        fake_cupy=_fake_cupy_module(device_count=1),
    )

    assert module.NAME == "cupy"
    assert module.IS_CUPY is True


def test_auto_detect_skips_cupy_without_cuda_device(monkeypatch):
    module = _load_backend_module(
        monkeypatch,
        fake_cupy=_fake_cupy_module(device_count=0),
    )

    assert module.NAME == "numpy"
    assert module.IS_NUMPY is True


def test_cupy_override_requires_cuda_device(monkeypatch):
    with pytest.raises(
        RuntimeError, match="AUTOGRAD_BACKEND=cupy requested, but no CUDA device"
    ):
        _load_backend_module(
            monkeypatch,
            env_backend="cupy",
            fake_cupy=_fake_cupy_module(device_count=0),
        )


def test_cupy_low_precision_includes_ml_dtypes_bfloat16(monkeypatch):
    fake_ml_dtypes = SimpleNamespace(bfloat16=np.dtype("V2"))

    module = _load_backend_module(
        monkeypatch,
        env_backend="cupy",
        fake_cupy=_fake_cupy_module(device_count=1),
        fake_ml_dtypes=fake_ml_dtypes,
    )

    assert fake_ml_dtypes.bfloat16 in module.LOW_PRECISION_FLOAT_DTYPES
