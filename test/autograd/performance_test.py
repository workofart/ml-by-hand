import gc
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Optional
from unittest import TestCase

import psutil

from autograd import functional, nn, optim
from autograd.backend import IS_CUPY, NAME, xp
from autograd.backend import eval as backend_eval
from autograd.tensor import Tensor
from autograd.tools.config_schema import TransformerTrainingConfig
from autograd.tools.data import (
    DataLoader,
    LanguageModelingCollator,
    PairedCollator,
    PairedIterableDataset,
    TokenSequenceDataset,
)
from autograd.tools.trainer import LLMTrainer
from examples.gpt_2 import GPT2, GPT2ForwardFn


@dataclass(frozen=True)
class MemorySnapshot:
    rss_bytes: int
    device_active_bytes: Optional[int] = None
    device_cache_bytes: Optional[int] = None
    device_peak_bytes: Optional[int] = None


@dataclass(frozen=True)
class CallProfile:
    elapsed_s: float
    rss_delta_bytes: int
    device_active_delta_bytes: Optional[int]
    device_cache_delta_bytes: Optional[int]
    device_peak_delta_bytes: Optional[int]


@dataclass(frozen=True)
class WorkEstimate:
    node_count: int
    saved_activation_bytes: int
    parameter_count: int
    parameter_bytes: int
    gradient_bytes: int
    optimizer_state_bytes: int
    static_resident_bytes: int
    logical_flops_per_step: int
    logical_bytes_per_step: int


BENCHMARK_RANDOM_SEED = 42

BENCHMARK_WARMUP_STEPS_ENV = "AUTOGRAD_PERF_WARMUP_STEPS"
BENCHMARK_MIN_STEPS_ENV = "AUTOGRAD_PERF_MIN_STEPS"
BENCHMARK_MIN_SECONDS_ENV = "AUTOGRAD_PERF_MIN_SECONDS"

DEFAULT_BENCHMARK_WARMUP_STEPS = 5
DEFAULT_BENCHMARK_MIN_STEPS = 10
DEFAULT_BENCHMARK_MIN_SECONDS = 1.0

MLP_CASE_NAME = "Complex MLP Model"
MLP_MODEL_KWARGS = {
    "input_size": 1024,
    "hidden_size": 512,
    "num_layers": 4,
    "output_size": 10,
}
MLP_BATCH_KWARGS = {
    "input_shape": (1024, 1024),
    "num_classes": 10,
}

CNN_CASE_NAME = "Deep CNN Model"
CNN_MODEL_KWARGS = {
    "input_channels": 3,
    "image_size": 24,
    "num_classes": 20,
}
CNN_BATCH_KWARGS = {
    "input_shape": (256, 3, 24, 24),
    "num_classes": 20,
}

GPT_CASE_NAME = "Mini GPT-2 Model"
GPT_LOADER_KWARGS = {
    "batch_size": 16,
    "seq_len": 768,
    "vocab_size": 1024 * 10,
}
GPT_MODEL_KWARGS = {
    "vocab_size": 1024 * 10,
    "hidden_size": 128 * 6,
    "num_attention_heads": 6,
    "max_seq_len": 768,
    "dropout_prob": 0.1,
    "num_decoder_layers": 6,
}
GPT_OPTIMIZER_KWARGS = {
    "lr": 1e-3,
    "beta2": 0.99,
    "max_grad_norm": 1.0,
    "weight_decay": 0.1,
}
GPT_TRAINER_CONFIG_KWARGS = {
    "max_epochs": 1,
    "checkpoint_freq": 1,
    "global_batch_size": 16,
    "micro_batch_size": 16,
    "model_kwargs": GPT_MODEL_KWARGS,
    "optimizer_kwargs": GPT_OPTIMIZER_KWARGS,
    "resume_epoch": None,
    "teacher_enforcing": False,
    "include_decoder_input": False,
    "create_padding_masks": False,
    "label_smoothing": 0.1,
}


def _flatten_sync_targets(value):
    if value is None:
        return []
    if isinstance(value, Tensor):
        return [value.data]
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return [value]
    if isinstance(value, Mapping):
        leaves = []
        for item in value.values():
            leaves.extend(_flatten_sync_targets(item))
        return leaves
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        leaves = []
        for item in value:
            leaves.extend(_flatten_sync_targets(item))
        return leaves
    return []


def synchronize_trees(*values):
    arrays = []
    for value in values:
        arrays.extend(_flatten_sync_targets(value))
    if arrays:
        backend_eval(*arrays)
    if IS_CUPY:
        cuda = getattr(xp, "cuda", None)
        device_cls = getattr(cuda, "Device", None)
        if callable(device_cls):
            device_cls().synchronize()
            return
        null_stream = getattr(getattr(cuda, "Stream", None), "null", None)
        stream_sync = getattr(null_stream, "synchronize", None)
        if callable(stream_sync):
            stream_sync()
            return
    if hasattr(xp, "synchronize"):
        xp.synchronize()


def _device_memory_stat(name):
    if IS_CUPY:
        pool = getattr(xp, "get_default_memory_pool", lambda: None)()
        if pool is None:
            return None
        if name == "get_active_memory":
            used_bytes = getattr(pool, "used_bytes", None)
            return int(used_bytes()) if callable(used_bytes) else None
        if name == "get_cache_memory":
            used_bytes = getattr(pool, "used_bytes", None)
            total_bytes = getattr(pool, "total_bytes", None)
            if callable(used_bytes) and callable(total_bytes):
                return int(total_bytes()) - int(used_bytes())
            return None
        return None

    fn = getattr(xp, name, None)
    if callable(fn):
        value = fn()
        return int(value) if value is not None else None
    metal = getattr(xp, "metal", None)
    fn = getattr(metal, name, None)
    if callable(fn):
        value = fn()
        return int(value) if value is not None else None
    return None


def reset_device_peak_memory():
    if IS_CUPY:
        return
    fn = getattr(xp, "reset_peak_memory", None)
    if callable(fn):
        fn()
        return
    metal = getattr(xp, "metal", None)
    fn = getattr(metal, "reset_peak_memory", None)
    if callable(fn):
        fn()


def clear_device_cache():
    fn = getattr(xp, "clear_cache", None)
    if callable(fn):
        fn()
        return
    metal = getattr(xp, "metal", None)
    fn = getattr(metal, "clear_cache", None)
    if callable(fn):
        fn()


def capture_memory_snapshot(process=None):
    if process is None:
        process = psutil.Process(os.getpid())
    rss_bytes = int(process.memory_info().rss)
    return MemorySnapshot(
        rss_bytes=rss_bytes,
        device_active_bytes=_device_memory_stat("get_active_memory"),
        device_cache_bytes=_device_memory_stat("get_cache_memory"),
        device_peak_bytes=_device_memory_stat("get_peak_memory"),
    )


def _delta(end, start):
    if end is None or start is None:
        return None
    return end - start


def _peak_over_start_bytes(start: MemorySnapshot, end: MemorySnapshot):
    return _delta(end.device_peak_bytes, start.device_peak_bytes)


def measure_call(
    fn: Callable[[], object],
    *,
    sync_trees: Optional[Callable[[object], object]] = None,
    process=None,
    reset_peak_memory: bool = False,
):
    synchronize_trees()
    start_snapshot = capture_memory_snapshot(process=process)
    if reset_peak_memory:
        reset_device_peak_memory()

    start_time = time.perf_counter()
    result = fn()
    if sync_trees is None:
        synchronize_trees(result)
    else:
        synchronize_trees(sync_trees(result))
    elapsed_s = time.perf_counter() - start_time
    end_snapshot = capture_memory_snapshot(process=process)

    device_peak_delta_bytes = (
        end_snapshot.device_peak_bytes
        if reset_peak_memory
        else _delta(end_snapshot.device_peak_bytes, start_snapshot.device_peak_bytes)
    )
    profile = CallProfile(
        elapsed_s=elapsed_s,
        rss_delta_bytes=end_snapshot.rss_bytes - start_snapshot.rss_bytes,
        device_active_delta_bytes=_delta(
            end_snapshot.device_active_bytes,
            start_snapshot.device_active_bytes,
        ),
        device_cache_delta_bytes=_delta(
            end_snapshot.device_cache_bytes,
            start_snapshot.device_cache_bytes,
        ),
        device_peak_delta_bytes=device_peak_delta_bytes,
    )
    return result, profile


def _format_memory_value(num_bytes: int) -> str:
    abs_bytes = abs(num_bytes)
    if abs_bytes < 1024:
        return f"{num_bytes:.2f} B"
    if abs_bytes < 1024**2:
        return f"{num_bytes / float(1024):.2f} KiB"
    if abs_bytes < 1024**3:
        return f"{num_bytes / float(1024**2):.2f} MiB"
    return f"{num_bytes / float(1024**3):.2f} GiB"


def _format_duration_value(duration_s: float) -> str:
    abs_seconds = abs(duration_s)
    if abs_seconds < 1.0:
        return f"{duration_s * 1000.0:.3f} ms"
    return f"{duration_s:.3f} s"


def _format_count_value(count: int) -> str:
    abs_count = abs(count)
    if abs_count < 1_000:
        return str(count)
    if abs_count < 1_000_000:
        return f"{count / 1_000.0:.2f} K"
    if abs_count < 1_000_000_000:
        return f"{count / 1_000_000.0:.2f} M"
    if abs_count < 1_000_000_000_000:
        return f"{count / 1_000_000_000.0:.2f} G"
    return f"{count / 1_000_000_000_000.0:.2f} T"


def _format_ratio_value(value: float) -> str:
    if value < 10.0:
        return f"{value:.2f}"
    if value < 100.0:
        return f"{value:.1f}"
    return f"{value:.0f}"


def _format_compact_number(value: float) -> str:
    if math.isnan(value):
        return "nan"

    abs_value = abs(value)
    if abs_value >= 1_000:
        precision = 0
    elif abs_value >= 100:
        precision = 1
    elif abs_value >= 1:
        precision = 2
    elif abs_value == 0:
        precision = 0
    else:
        precision = 3

    formatted = f"{value:,.{precision}f}"
    if precision == 0:
        return formatted
    return formatted.rstrip("0").rstrip(".")


def _format_compact_measurement_value(text: str) -> str:
    value_text, separator, unit = text.partition(" ")
    if not separator:
        return text

    try:
        value = float(value_text.replace(",", ""))
    except ValueError:
        return text
    return f"{_format_compact_number(value)} {unit}"


def compute_stats(values):
    vals = [float(value) for value in values if value is not None]
    if not vals:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
        }

    vals.sort()

    def _pct(p):
        if len(vals) == 1:
            return vals[0]
        index = (len(vals) - 1) * p
        low = math.floor(index)
        high = math.ceil(index)
        if low == high:
            return vals[low]
        weight = index - low
        return vals[low] + (vals[high] - vals[low]) * weight

    mean = sum(vals) / len(vals)
    variance = sum((value - mean) ** 2 for value in vals) / len(vals)
    return {
        "mean": mean,
        "std": math.sqrt(variance),
        "min": vals[0],
        "max": vals[-1],
        "median": _pct(0.5),
        "p90": _pct(0.9),
    }


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _array_nbytes(array) -> int:
    nbytes = getattr(array, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)

    dtype = getattr(array, "dtype", None)
    itemsize = getattr(dtype, "itemsize", None)
    if itemsize is None:
        itemsize = getattr(array, "itemsize", None)
    if itemsize is None:
        return 0
    return _numel(tuple(getattr(array, "shape", ()))) * int(itemsize)


_GRAPH_ALIAS_OPS = {"Expand", "Permute", "Reshape", "Transpose", "View"}


def _estimate_matmul_flops(
    x_shape: tuple[int, ...],
    y_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
) -> int:
    if not x_shape or not y_shape:
        return 0
    reduction_dim = int(x_shape[-1])
    return 2 * _numel(out_shape) * reduction_dim


def _estimate_forward_node_flops(tensor: Tensor) -> int:
    creator = tensor.creator
    if creator is None:
        return 0

    op_name = type(creator).__name__
    output_numel = _numel(tensor.shape)
    if op_name in {"Add", "IAdd", "Mul", "Pow", "Relu", "Sigmoid", "Sqrt", "Tanh"}:
        return output_numel
    if op_name == "Gelu":
        return 8 * output_numel
    if op_name == "Matmul":
        return _estimate_matmul_flops(
            creator.tensors[0].shape,
            creator.tensors[1].shape,
            tensor.shape,
        )
    if op_name in {"Mean", "Sum"}:
        return _numel(creator.tensors[0].shape)
    if op_name == "Max":
        return max(0, _numel(creator.tensors[0].shape) - output_numel)
    if op_name == "Softmax":
        return 5 * output_numel
    if op_name == "CrossEntropy":
        return 5 * _numel(creator.tensors[0].shape)
    return 0


def _estimate_forward_node_logical_bytes(tensor: Tensor) -> int:
    creator = tensor.creator
    if creator is None:
        return 0

    if type(creator).__name__ in _GRAPH_ALIAS_OPS:
        return 0

    input_bytes = 0
    for parent in creator.tensors:
        if parent is not None:
            input_bytes += _array_nbytes(parent.data)
    return input_bytes + _array_nbytes(tensor.data)


def estimate_step_work(
    *,
    loss: Tensor,
    model,
    optimizer,
) -> WorkEstimate:
    parameter_tensors = tuple(model.parameters.values())
    parameter_ids = {id(param) for param in parameter_tensors}

    node_count = 0
    saved_activation_bytes = 0
    forward_flops = 0
    forward_logical_bytes = 0
    visited: set[Tensor] = set()
    stack = [loss]

    while stack:
        tensor = stack.pop()
        if tensor in visited:
            continue
        visited.add(tensor)

        if tensor.creator is not None:
            node_count += 1
            forward_flops += _estimate_forward_node_flops(tensor)
            forward_logical_bytes += _estimate_forward_node_logical_bytes(tensor)
            if (
                id(tensor) not in parameter_ids
                and type(tensor.creator).__name__ not in _GRAPH_ALIAS_OPS
            ):
                saved_activation_bytes += _array_nbytes(tensor.data)

            for parent in tensor.creator.tensors:
                if parent is not None and parent.requires_grad:
                    stack.append(parent)

    parameter_count = sum(_numel(param.shape) for param in parameter_tensors)
    parameter_bytes = sum(_array_nbytes(param.data) for param in parameter_tensors)
    gradient_bytes = parameter_bytes
    if isinstance(optimizer, optim.SGD):
        optimizer_state_bytes = 0
        optimizer_step_bytes = 3 * parameter_bytes
        optimizer_step_flops = parameter_count
    elif isinstance(optimizer, optim.Adam):
        optimizer_state_bytes = 2 * parameter_bytes
        optimizer_step_bytes = 7 * parameter_bytes
        optimizer_step_flops = 10 * parameter_count
    else:
        optimizer_state_bytes = 0
        optimizer_step_bytes = 0
        optimizer_step_flops = 0

    static_resident_bytes = parameter_bytes + gradient_bytes + optimizer_state_bytes
    backward_logical_bytes = (
        forward_logical_bytes + saved_activation_bytes + gradient_bytes
    )
    logical_bytes_per_step = (
        forward_logical_bytes + backward_logical_bytes + optimizer_step_bytes
    )
    logical_flops_per_step = 3 * forward_flops + optimizer_step_flops

    return WorkEstimate(
        node_count=node_count,
        saved_activation_bytes=saved_activation_bytes,
        parameter_count=parameter_count,
        parameter_bytes=parameter_bytes,
        gradient_bytes=gradient_bytes,
        optimizer_state_bytes=optimizer_state_bytes,
        static_resident_bytes=static_resident_bytes,
        logical_flops_per_step=logical_flops_per_step,
        logical_bytes_per_step=logical_bytes_per_step,
    )


class ComplexMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.layers = []
        in_size = input_size
        for _ in range(num_layers):
            linear = nn.Linear(in_size, hidden_size)
            bn = nn.BatchNorm(hidden_size)
            self.layers.append((linear, bn))
            in_size = hidden_size
        self.final = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        for linear, bn in self.layers:
            x = linear(x)
            x = bn(x)
            x = functional.relu(x)
        return self.final(x)


class DeepCNN(nn.Module):
    def __init__(self, input_channels, image_size, num_classes):
        super().__init__()
        # Convolutional layers and associated batch norms + pooling
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding_mode="same")
        self.bn1 = nn.BatchNorm(32 * image_size * image_size)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding_mode="same")
        self.bn2 = nn.BatchNorm(64 * (image_size // 2) * (image_size // 2))
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding_mode="same")
        self.bn3 = nn.BatchNorm(128 * (image_size // 4) * (image_size // 4))
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        # Fully connected layers
        fc_input = 128 * (image_size // 8) * (image_size // 8)
        self.fc1 = nn.Linear(fc_input, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self.image_size = image_size

    def forward(self, x):
        # Conv layer 1
        x = self.conv1(x)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = self.bn1(x)
        x = x.reshape(batch_size, 32, self.image_size, self.image_size)
        x = functional.relu(x)
        x = self.pool1(x)

        # Conv layer 2
        x = self.conv2(x)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = self.bn2(x)
        x = x.reshape(batch_size, 64, self.image_size // 2, self.image_size // 2)
        x = functional.relu(x)
        x = self.pool2(x)

        # Conv layer 3
        x = self.conv3(x)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = self.bn3(x)
        x = x.reshape(batch_size, 128, self.image_size // 4, self.image_size // 4)
        x = functional.relu(x)
        x = self.pool3(x)

        # Fully connected layers
        x = x.reshape(batch_size, -1)
        x = functional.relu(self.fc1(x))
        return self.fc2(x)


class CIPipelinePerformanceTest(TestCase):
    def _estimate_case_work(self, *, model, optimizer, loss_builder):
        loss = loss_builder()
        return estimate_step_work(
            loss=loss,
            model=model,
            optimizer=optimizer,
        )

    def _benchmark_config(self):
        return {
            "warmup_steps": int(
                os.getenv(
                    BENCHMARK_WARMUP_STEPS_ENV,
                    str(DEFAULT_BENCHMARK_WARMUP_STEPS),
                )
            ),
            "min_measure_steps": int(
                os.getenv(BENCHMARK_MIN_STEPS_ENV, str(DEFAULT_BENCHMARK_MIN_STEPS))
            ),
            "min_measure_seconds": float(
                os.getenv(
                    BENCHMARK_MIN_SECONDS_ENV,
                    str(DEFAULT_BENCHMARK_MIN_SECONDS),
                )
            ),
        }

    def _cleanup_benchmark_case(self):
        synchronize_trees()
        gc.collect()
        clear_device_cache()
        reset_device_peak_memory()

    def _record_benchmark_case(
        self,
        results,
        summary_rows,
        *,
        case_name,
        model,
        batch,
        throughput_unit,
        optimizer_lr=0.01,
        loss_fn=functional.cross_entropy,
    ):
        try:
            model_inputs, targets, throughput_count = batch
            results["cases"][case_name] = self._run_benchmark_case(
                model=model,
                model_inputs=model_inputs,
                targets=targets,
                throughput_count=throughput_count,
                throughput_unit=throughput_unit,
                optimizer_lr=optimizer_lr,
                loss_fn=loss_fn,
            )
            summary_rows.append(
                self._summary_row(case_name, results["cases"][case_name])
            )
        finally:
            self._cleanup_benchmark_case()

    def _build_classification_batch(
        self,
        *,
        input_shape: tuple[int, ...],
        num_classes: int,
    ):
        """Materialize one paired classification batch via the repo data pipeline."""
        inputs = xp.random.normal(shape=input_shape).astype(xp.float32)
        targets = xp.random.randint(
            0,
            num_classes,
            (input_shape[0],),
            dtype=xp.int64,
        )
        loader = DataLoader(
            dataset=PairedIterableDataset(inputs, targets, shuffle=False),
            batch_size=input_shape[0],
            collate_fn=PairedCollator(),
        )
        batch_inputs, batch_targets = next(iter(loader))
        return (
            (Tensor(batch_inputs),),
            batch_targets,
            int(batch_inputs.shape[0]),
        )

    def _build_causal_lm_loader(
        self,
        *,
        batch_size: int,
        seq_len: int,
        vocab_size: int,
        num_batches: int = 1,
    ):
        """Materialize a causal LM data loader for trainer-path benchmarks."""
        token_sequences = [
            xp.random.randint(2, vocab_size, (seq_len + 1,), dtype=xp.int32)
            for _ in range(batch_size * num_batches)
        ]
        loader = DataLoader(
            dataset=TokenSequenceDataset(
                token_sequences=token_sequences,
                shuffle=False,
            ),
            batch_size=batch_size,
            collate_fn=LanguageModelingCollator(
                max_tokens=seq_len + 1,
                pad_idx=0,
                sos_idx=1,
                include_decoder_input=False,
                create_padding_masks=False,
            ),
        )
        first_batch = next(iter(loader))
        inputs = first_batch[0]
        return loader, first_batch, int(inputs.shape[0] * inputs.shape[1])

    def _run_benchmark_case(
        self,
        *,
        model,
        model_inputs,
        targets,
        throughput_count,
        throughput_unit,
        optimizer_lr=0.01,
        loss_fn=functional.cross_entropy,
    ):
        model.train()
        optimizer = optim.SGD(model.parameters, lr=optimizer_lr)
        work = self._estimate_case_work(
            model=model,
            optimizer=optimizer,
            loss_builder=lambda: loss_fn(model(*model_inputs), targets),
        )
        optimizer.zero_grad()

        def _run_training_step():
            y_pred = model(*model_inputs)
            loss = loss_fn(y_pred, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            return {"loss": loss, "params": tuple(model.parameters.values())}

        return self._measure_training_step_case(
            throughput_count=throughput_count,
            throughput_unit=throughput_unit,
            benchmark_fn=_run_training_step,
            sync_targets=lambda result: (result["loss"], result["params"]),
            work=work,
        )

    def _run_trainer_benchmark_case(
        self,
        *,
        trainer,
        batch_data,
        data_loader,
        throughput_count,
        throughput_unit,
        work,
    ):
        trainer.model.train()
        trainer.optimizer.zero_grad()

        def _run_training_step():
            trainer.optimizer.zero_grad()
            loss = trainer.train_step(batch_data, data_loader)
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()
            return {
                "loss": loss,
                "params": tuple(trainer.optimizer.model_parameters.values()),
            }

        return self._measure_training_step_case(
            throughput_count=throughput_count,
            throughput_unit=throughput_unit,
            benchmark_fn=_run_training_step,
            sync_targets=lambda result: (result["loss"], result["params"]),
            work=work,
        )

    def _measure_training_step_case(
        self,
        *,
        throughput_count,
        throughput_unit,
        benchmark_fn,
        sync_targets,
        work,
    ):
        config = self._benchmark_config()
        metrics = {
            "step_times": [],
            "throughput_per_second": [],
            "achieved_gflops": [],
            "achieved_gbps": [],
            "rss_deltas": [],
            "device_active_deltas": [],
            "device_cache_deltas": [],
            "device_peaks": [],
            "device_peak_over_start_deltas": [],
        }

        for _ in range(config["warmup_steps"]):
            _, _ = measure_call(benchmark_fn, sync_trees=sync_targets)

        measured_seconds = 0.0
        while (
            len(metrics["step_times"]) < config["min_measure_steps"]
            or measured_seconds < config["min_measure_seconds"]
        ):
            synchronize_trees()
            reset_device_peak_memory()
            step_start = capture_memory_snapshot()
            _, profile = measure_call(benchmark_fn, sync_trees=sync_targets)
            synchronize_trees()
            step_end = capture_memory_snapshot()

            metrics["step_times"].append(profile.elapsed_s)
            measured_seconds += profile.elapsed_s
            metrics["throughput_per_second"].append(
                throughput_count / profile.elapsed_s
            )
            if work.logical_flops_per_step > 0:
                metrics["achieved_gflops"].append(
                    work.logical_flops_per_step / profile.elapsed_s / 1e9
                )
            if work.logical_bytes_per_step > 0:
                metrics["achieved_gbps"].append(
                    work.logical_bytes_per_step / profile.elapsed_s / 1e9
                )
            metrics["rss_deltas"].append(step_end.rss_bytes - step_start.rss_bytes)
            metrics["device_active_deltas"].append(
                _delta(step_end.device_active_bytes, step_start.device_active_bytes)
            )
            metrics["device_cache_deltas"].append(
                _delta(step_end.device_cache_bytes, step_start.device_cache_bytes)
            )
            metrics["device_peaks"].append(step_end.device_peak_bytes)
            metrics["device_peak_over_start_deltas"].append(
                _peak_over_start_bytes(step_start, step_end)
            )

        return self._summarize_metrics(
            metrics=metrics,
            throughput_unit=throughput_unit,
            work=work,
            measured_steps=len(metrics["step_times"]),
            measured_seconds=measured_seconds,
        )

    def _scalar_stats(self, values):
        return compute_stats(values)

    def _duration_stats(self, values):
        stats = self._scalar_stats(values)
        return {key: _format_duration_value(value) for key, value in stats.items()}

    def _byte_stats(self, values):
        stats = self._scalar_stats(values)
        return {
            key: _format_memory_value(int(round(value))) for key, value in stats.items()
        }

    def _rate_stats(self, values, unit):
        stats = self._scalar_stats(values)
        return {
            "unit": unit,
            **{
                key: (round(value, 2) if not math.isnan(value) else value)
                for key, value in stats.items()
            },
        }

    def _format_work_summary(self, work: WorkEstimate):
        return {
            "parameters": _format_count_value(work.parameter_count),
            "parameter_bytes": _format_memory_value(work.parameter_bytes),
            "gradient_bytes": _format_memory_value(work.gradient_bytes),
            "optimizer_state_bytes": _format_memory_value(work.optimizer_state_bytes),
            "static_resident_bytes": _format_memory_value(work.static_resident_bytes),
            "forward_graph_nodes": str(work.node_count),
            "saved_activation_bytes": _format_memory_value(work.saved_activation_bytes),
            "logical_flops_per_step": _format_count_value(work.logical_flops_per_step),
            "logical_bytes_per_step": _format_memory_value(work.logical_bytes_per_step),
            "arithmetic_intensity": _format_ratio_value(
                work.logical_flops_per_step / max(work.logical_bytes_per_step, 1)
            ),
        }

    def _summarize_metrics(
        self,
        *,
        metrics,
        throughput_unit,
        work,
        measured_steps,
        measured_seconds,
    ):
        memory_summary = {
            "host_rss_delta": self._byte_stats(metrics["rss_deltas"]),
        }
        for metric_suffix, values in (
            ("device_active_delta", metrics["device_active_deltas"]),
            ("device_cache_delta", metrics["device_cache_deltas"]),
            ("device_peak", metrics["device_peaks"]),
            ("device_peak_over_start", metrics["device_peak_over_start_deltas"]),
        ):
            values = [value for value in values if value is not None]
            if values:
                memory_summary[f"{NAME}_{metric_suffix}"] = self._byte_stats(values)

        return {
            "measurement": {
                "steps": measured_steps,
                "seconds": round(measured_seconds, 3),
            },
            "timing": {
                "training_step": self._duration_stats(metrics["step_times"]),
            },
            "throughput": self._rate_stats(
                metrics["throughput_per_second"], f"{throughput_unit}/s"
            ),
            "efficiency": {
                "achieved_gflops": self._rate_stats(
                    metrics["achieved_gflops"], "GFLOP/s"
                ),
                "achieved_gbps": self._rate_stats(metrics["achieved_gbps"], "GB/s"),
            },
            "memory": memory_summary,
            "work": self._format_work_summary(work),
        }

    def _summary_row(self, case_name, case_metrics):
        row = {
            "case": case_name,
            "step_p50": _format_compact_measurement_value(
                case_metrics["timing"]["training_step"]["median"]
            ),
            "step_p90": _format_compact_measurement_value(
                case_metrics["timing"]["training_step"]["p90"]
            ),
            "achieved_gflops": (
                f"{_format_compact_number(case_metrics['efficiency']['achieved_gflops']['median'])} "
                f"{case_metrics['efficiency']['achieved_gflops']['unit']}"
            ),
            "node_count": case_metrics["work"]["forward_graph_nodes"],
            "static_resident": case_metrics["work"]["static_resident_bytes"],
        }
        peak_over_start_key = f"{NAME}_device_peak_over_start"
        if peak_over_start_key in case_metrics["memory"]:
            row["peak_over_start"] = case_metrics["memory"][peak_over_start_key]["mean"]
        return row

    def _format_summary_table(self, rows):
        columns = [
            ("case", "Case"),
            ("step_p50", "Step P50"),
            ("step_p90", "Step P90"),
            ("achieved_gflops", "GFLOP/s"),
            ("node_count", "Node Counts"),
            ("static_resident", "Static Memory"),
        ]
        if any("peak_over_start" in row for row in rows):
            columns.append(("peak_over_start", "Memory Peak Over Start"))

        widths = {}
        for key, header in columns:
            widths[key] = len(header)
            for row in rows:
                widths[key] = max(widths[key], len(str(row.get(key, ""))))

        def _format_row(row):
            cells = []
            for key, _ in columns:
                cells.append(f"{str(row.get(key, '')):<{widths[key]}}")
            return "| " + " | ".join(cells) + " |"

        header = _format_row({key: label for key, label in columns})
        separator = "|-" + "-|-".join("-" * widths[key] for key, _ in columns) + "-|"
        body = [_format_row(row) for row in rows]
        return "\n".join([header, separator, *body])

    def test_resource_usage_metrics(self):
        """
        Measure step throughput and memory for representative MLP, CNN, and GPT-2
        training steps on the resolved backend.
        """
        xp.random.seed(BENCHMARK_RANDOM_SEED)
        results = {
            "backend": NAME,
            "benchmark_config": self._benchmark_config(),
            "cases": {},
        }
        summary_rows = []
        self._record_benchmark_case(
            results,
            summary_rows,
            case_name=MLP_CASE_NAME,
            model=ComplexMLP(**MLP_MODEL_KWARGS),
            batch=self._build_classification_batch(**MLP_BATCH_KWARGS),
            throughput_unit="examples",
        )
        self._record_benchmark_case(
            results,
            summary_rows,
            case_name=CNN_CASE_NAME,
            model=DeepCNN(**CNN_MODEL_KWARGS),
            batch=self._build_classification_batch(**CNN_BATCH_KWARGS),
            throughput_unit="examples",
        )
        trainer_loader, trainer_batch, trainer_throughput_count = (
            self._build_causal_lm_loader(**GPT_LOADER_KWARGS)
        )
        llm_trainer = LLMTrainer(
            model_cls=GPT2,
            optimizer_cls=optim.Adam,
            loss_fn=functional.cross_entropy,
            forward_fn=GPT2ForwardFn(),
            config=TransformerTrainingConfig(**GPT_TRAINER_CONFIG_KWARGS),
        )
        llm_trainer.model.train()
        trainer_work = self._estimate_case_work(
            model=llm_trainer.model,
            optimizer=llm_trainer.optimizer,
            loss_builder=lambda: llm_trainer.loss_fn(
                *llm_trainer.forward_fn(
                    llm_trainer.model,
                    trainer_batch,
                    mode="train",
                ),
                pad_idx=trainer_loader.pad_idx,
                label_smoothing=llm_trainer.config.label_smoothing,
            ),
        )
        try:
            results["cases"][GPT_CASE_NAME] = self._run_trainer_benchmark_case(
                trainer=llm_trainer,
                batch_data=trainer_batch,
                data_loader=trainer_loader,
                throughput_count=trainer_throughput_count,
                throughput_unit="tokens",
                work=trainer_work,
            )
            summary_rows.append(
                self._summary_row(
                    GPT_CASE_NAME,
                    results["cases"][GPT_CASE_NAME],
                )
            )
        finally:
            self._cleanup_benchmark_case()
        print("\n")
        print(self._format_summary_table(summary_rows))
