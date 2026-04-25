import gc
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
from autograd.data.collator import FixedLengthCausalLMCollator, PairedCollator
from autograd.data.data_loader import DataLoader
from autograd.data.dataset import PairedMapDataset
from autograd.tensor import Tensor
from examples.gpt_2 import GPT2


@dataclass(frozen=True)
class MemorySnapshot:
    rss_bytes: int
    swap_used_bytes: int
    device_active_bytes: Optional[int] = None
    device_cache_bytes: Optional[int] = None
    device_peak_bytes: Optional[int] = None


@dataclass(frozen=True)
class CallProfile:
    elapsed_s: float
    rss_delta_bytes: int
    swap_used_delta_bytes: int
    device_active_delta_bytes: Optional[int]
    device_cache_delta_bytes: Optional[int]
    device_peak_delta_bytes: Optional[int]


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
    swap_used_bytes = int(psutil.swap_memory().used)
    return MemorySnapshot(
        rss_bytes=rss_bytes,
        swap_used_bytes=swap_used_bytes,
        device_active_bytes=_device_memory_stat("get_active_memory"),
        device_cache_bytes=_device_memory_stat("get_cache_memory"),
        device_peak_bytes=_device_memory_stat("get_peak_memory"),
    )


def _delta(end, start):
    if end is None or start is None:
        return None
    return end - start


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
        swap_used_delta_bytes=end_snapshot.swap_used_bytes
        - start_snapshot.swap_used_bytes,
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
    if abs_bytes < 1000 * 1024**2:
        return f"{num_bytes / float(1024**2):.2f} MiB"
    return f"{num_bytes / float(1024**3):.2f} GiB"


def _format_duration_value(duration_s: float) -> str:
    abs_seconds = abs(duration_s)
    if abs_seconds < 1.0:
        return f"{duration_s * 1000.0:.3f} ms"
    return f"{duration_s:.3f} s"


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
        total_epochs,
        optimizer_lr=0.01,
        loss_fn=functional.cross_entropy,
    ):
        try:
            model_inputs, targets, throughput_count = batch
            results["cases"][case_name] = self._run_benchmark_case(
                model=model,
                model_inputs=model_inputs,
                targets=targets,
                total_epochs=total_epochs,
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
            dataset=PairedMapDataset(inputs, targets),
            batch_size=input_shape[0],
            collator=PairedCollator(),
        )
        batch_inputs, batch_targets = next(iter(loader))
        return (
            (Tensor(batch_inputs),),
            batch_targets,
            int(batch_inputs.shape[0]),
        )

    def _build_causal_lm_batch(
        self,
        *,
        batch_size: int,
        seq_len: int,
        vocab_size: int,
    ):
        """Materialize one causal LM batch via the repo token dataset/collator path."""
        token_sequences = [
            xp.random.randint(2, vocab_size, (seq_len + 1,), dtype=xp.int32)
            for _ in range(batch_size)
        ]
        loader = DataLoader(
            dataset=PairedMapDataset(
                token_sequences,
                [xp.ones((seq_len + 1,), dtype=xp.int32) for _ in range(batch_size)],
                input_key="tokens",
                target_key="loss_mask",
                dtype=xp.int32,
            ),
            batch_size=batch_size,
            collator=FixedLengthCausalLMCollator(
                max_tokens=seq_len + 1,
                pad_idx=0,
            ),
        )
        batch = next(iter(loader))
        inputs = batch.input_ids
        targets = batch.labels
        return (
            (Tensor(inputs, requires_grad=False),),
            targets,
            int(inputs.shape[0] * inputs.shape[1]),
        )

    def _run_benchmark_case(
        self,
        *,
        model,
        model_inputs,
        targets,
        total_epochs,
        throughput_count,
        throughput_unit,
        optimizer_lr=0.01,
        loss_fn=functional.cross_entropy,
    ):
        model.train()
        optimizer = optim.SGD(model.parameters, lr=optimizer_lr)
        metrics = {
            "forward_times": [],
            "backward_times": [],
            "update_times": [],
            "step_times": [],
            "throughput_per_second": [],
            "rss_deltas": [],
            "swap_used_deltas": [],
            "device_active_deltas": [],
            "device_cache_deltas": [],
            "device_peak_bytes": [],
        }

        def _loss_and_backward(y_pred):
            loss = loss_fn(y_pred, targets)
            loss.backward()
            grads = [p.grad for p in model.parameters.values() if p.grad is not None]
            return {"loss": loss, "grads": grads}

        def _optimizer_step():
            optimizer.step()
            optimizer.zero_grad()
            return tuple(model.parameters.values())

        # Warm-up iterations (not timed)
        for _ in range(2):
            y_pred, _ = measure_call(lambda: model(*model_inputs))
            _, _ = measure_call(
                lambda: _loss_and_backward(y_pred),
                sync_trees=lambda result: (result["loss"], result["grads"]),
            )
            _, _ = measure_call(_optimizer_step)

        # Actual performance measurements
        for _ in range(total_epochs):
            synchronize_trees()
            reset_device_peak_memory()
            step_start = capture_memory_snapshot()
            y_pred, forward_profile = measure_call(lambda: model(*model_inputs))
            _, backward_profile = measure_call(
                lambda: _loss_and_backward(y_pred),
                sync_trees=lambda result: (result["loss"], result["grads"]),
            )
            _, update_profile = measure_call(_optimizer_step)
            synchronize_trees()
            step_end = capture_memory_snapshot()

            step_elapsed = (
                forward_profile.elapsed_s
                + backward_profile.elapsed_s
                + update_profile.elapsed_s
            )
            metrics["forward_times"].append(forward_profile.elapsed_s)
            metrics["backward_times"].append(backward_profile.elapsed_s)
            metrics["update_times"].append(update_profile.elapsed_s)
            metrics["step_times"].append(step_elapsed)
            metrics["throughput_per_second"].append(throughput_count / step_elapsed)
            metrics["rss_deltas"].append(step_end.rss_bytes - step_start.rss_bytes)
            metrics["swap_used_deltas"].append(
                step_end.swap_used_bytes - step_start.swap_used_bytes
            )
            metrics["device_active_deltas"].append(
                _delta(step_end.device_active_bytes, step_start.device_active_bytes)
            )
            metrics["device_cache_deltas"].append(
                _delta(step_end.device_cache_bytes, step_start.device_cache_bytes)
            )
            metrics["device_peak_bytes"].append(step_end.device_peak_bytes)

        return self._summarize_metrics(metrics, throughput_unit)

    def _compute_stats(self, metrics_list):
        """Compute mean, std, min, max statistics for a given list of values."""
        metrics_array = xp.asarray(metrics_list, dtype=xp.float32)
        return {
            "mean": float(xp.mean(metrics_array)),
            "std": float(xp.std(metrics_array)),
            "min": float(xp.min(metrics_array)),
            "max": float(xp.max(metrics_array)),
        }

    def _duration_stats(self, values):
        stats = self._compute_stats(values)
        return {key: _format_duration_value(value) for key, value in stats.items()}

    def _byte_stats(self, values):
        stats = self._compute_stats(values)
        return {
            key: _format_memory_value(int(round(value))) for key, value in stats.items()
        }

    def _summarize_metrics(self, metrics, throughput_unit):
        timing_summary = {
            "forward": self._duration_stats(metrics["forward_times"]),
            "backward": self._duration_stats(metrics["backward_times"]),
            "optimizer_update": self._duration_stats(metrics["update_times"]),
            "training_step": self._duration_stats(metrics["step_times"]),
        }
        throughput_summary = self._compute_stats(metrics["throughput_per_second"])
        throughput_summary = {
            "unit": f"{throughput_unit}/s",
            "mean": round(throughput_summary["mean"], 2),
            "std": round(throughput_summary["std"], 2),
            "min": round(throughput_summary["min"], 2),
            "max": round(throughput_summary["max"], 2),
        }
        memory_summary = {
            "process_rss_delta": self._byte_stats(metrics["rss_deltas"]),
            "system_swap_used_delta": self._byte_stats(metrics["swap_used_deltas"]),
        }
        for metric_suffix, values in (
            ("device_active_delta", metrics["device_active_deltas"]),
            ("device_cache_delta", metrics["device_cache_deltas"]),
            ("device_peak_allocator", metrics["device_peak_bytes"]),
        ):
            values = [value for value in values if value is not None]
            if values:
                memory_summary[f"{NAME}_{metric_suffix}"] = self._byte_stats(values)
        return {
            "timing": timing_summary,
            "throughput": throughput_summary,
            "memory": memory_summary,
        }

    def _summary_row(self, case_name, case_metrics):
        row = {
            "case": case_name,
            "step_mean": case_metrics["timing"]["training_step"]["mean"],
            "step_std": case_metrics["timing"]["training_step"]["std"],
            "throughput": (
                f"{case_metrics['throughput']['mean']} "
                f"{case_metrics['throughput']['unit']}"
            ),
            "process_rss_delta_mean": case_metrics["memory"]["process_rss_delta"][
                "mean"
            ],
            "system_swap_used_delta_mean": case_metrics["memory"][
                "system_swap_used_delta"
            ]["mean"],
        }
        peak_key = f"{NAME}_device_peak_allocator"
        if peak_key in case_metrics["memory"]:
            row[f"{NAME}_device_peak_allocator_mean"] = case_metrics["memory"][
                peak_key
            ]["mean"]
        return row

    def _format_summary_table(self, rows):
        backend_label = NAME.upper()
        columns = [
            ("case", "Case"),
            ("step_mean", "Step Mean"),
            ("step_std", "Step Std"),
            ("throughput", "Throughput"),
            ("process_rss_delta_mean", "Process RSS Delta"),
            ("system_swap_used_delta_mean", "System Swap Used Delta"),
        ]
        peak_key = f"{NAME}_device_peak_allocator_mean"
        if any(peak_key in row for row in rows):
            columns.append((peak_key, f"{backend_label} Peak Allocator"))

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
        xp.random.seed(42)
        total_epochs = 10
        results = {
            "backend": NAME,
            "measurement_epochs": total_epochs,
            "cases": {},
        }
        summary_rows = []
        self._record_benchmark_case(
            results,
            summary_rows,
            case_name="Complex MLP Model",
            model=ComplexMLP(
                input_size=1024,
                hidden_size=512,
                num_layers=4,
                output_size=10,
            ),
            batch=self._build_classification_batch(
                input_shape=(1024, 1024),
                num_classes=10,
            ),
            total_epochs=total_epochs,
            throughput_unit="examples",
        )
        self._record_benchmark_case(
            results,
            summary_rows,
            case_name="Deep CNN Model",
            model=DeepCNN(
                input_channels=3,
                image_size=24,
                num_classes=20,
            ),
            batch=self._build_classification_batch(
                input_shape=(256, 3, 24, 24),
                num_classes=20,
            ),
            total_epochs=total_epochs,
            throughput_unit="examples",
        )
        self._record_benchmark_case(
            results,
            summary_rows,
            case_name="Mini GPT-2 Model",
            model=GPT2(
                vocab_size=1024 * 10,
                hidden_size=128 * 6,
                num_attention_heads=6,
                max_seq_len=768,
                dropout_prob=0.1,
                num_decoder_layers=6,
            ),
            batch=self._build_causal_lm_batch(
                batch_size=8,
                seq_len=768,
                vocab_size=1024 * 10,
            ),
            total_epochs=total_epochs,
            throughput_unit="tokens",
            optimizer_lr=1e-3,
            loss_fn=functional.cross_entropy,
        )
        print("\n")
        print(self._format_summary_table(summary_rows))
