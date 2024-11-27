import numpy as np
import time
import psutil
import os
import logging
from autograd import nn, optim, functional
from autograd.tensor import Tensor
from unittest import TestCase
from sklearn.datasets import load_breast_cancer
import tracemalloc

logger = logging.getLogger(__name__)


class CIPipelinePerformanceTest(TestCase):
    def setUp(self):
        # Start memory tracking
        tracemalloc.start()
        self.process = psutil.Process(os.getpid())

    def tearDown(self):
        # Stop memory tracking
        tracemalloc.stop()

    def test_resource_usage_metrics(self):
        """
        Measure computational efficiency with memory and CPU usage metrics

        Key goals:
        1. Track memory usage per forward/backward pass
        2. Measure CPU utilization
        3. Provide consistent, reproducible metrics
        """
        # Consistent random seed for reproducibility
        np.random.seed(42)

        # Lightweight model configuration
        batch_size = 128
        input_size = 30
        hidden_size = 64
        total_epochs = 1000

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.h1 = nn.Linear(input_size, hidden_size)
                self.bn1 = nn.BatchNorm(hidden_size)  # BatchNorm after first layer
                self.h2 = nn.Linear(hidden_size, hidden_size)
                self.bn2 = nn.BatchNorm(hidden_size)  # BatchNorm after second layer
                self.h3 = nn.Linear(
                    hidden_size, 1
                )  # Change to 1 output for binary classification

            def forward(self, x):
                x = functional.relu(self.bn1(self.h1(x)))  # Apply BatchNorm
                x = functional.relu(self.bn2(self.h2(x)))  # Apply BatchNorm
                return functional.sigmoid(
                    self.h3(x)
                )  # Use sigmoid for binary classification

        model = SimpleModel()

        optimizer = optim.SGD(model.parameters, lr=0.01)

        # Performance tracking
        performance_metrics = {
            "forward_memory": [],
            "backward_memory": [],
            "forward_cpu": [],
            "backward_cpu": [],
            "forward_times": [],
            "backward_times": [],
        }

        # Load the breast cancer dataset
        X, y = load_breast_cancer(return_X_y=True)
        x = Tensor(X[:batch_size])  # Use the first 'batch_size' samples
        y_true = Tensor(
            y[:batch_size].astype(np.float32)
        )  # Convert labels to float for binary classification

        # Warm-up iteration
        _ = model(x)

        # Performance measurement
        for _ in range(total_epochs):
            # Reset tracemalloc for precise memory tracking
            tracemalloc.reset_peak()

            # Measure forward pass
            start_cpu = self.process.cpu_percent()
            start_time = time.perf_counter()

            # Capture memory before forward pass
            mem_before_forward = tracemalloc.get_traced_memory()[0]

            y_pred = model(x)

            # Capture memory after forward pass
            mem_after_forward = tracemalloc.get_traced_memory()[0]
            forward_time = time.perf_counter() - start_time
            forward_cpu = self.process.cpu_percent() - start_cpu

            # Store forward pass metrics
            performance_metrics["forward_memory"].append(
                mem_after_forward - mem_before_forward
            )
            performance_metrics["forward_times"].append(forward_time)
            performance_metrics["forward_cpu"].append(forward_cpu)

            # Reset tracemalloc for backward pass
            tracemalloc.reset_peak()

            # Measure backward pass
            start_cpu = self.process.cpu_percent()
            start_time = time.perf_counter()

            # Capture memory before backward pass
            mem_before_backward = tracemalloc.get_traced_memory()[0]

            loss = functional.binary_cross_entropy(y_pred, y_true)
            loss.backward()

            # Capture memory after backward pass
            mem_after_backward = tracemalloc.get_traced_memory()[0]
            backward_time = time.perf_counter() - start_time
            backward_cpu = self.process.cpu_percent() - start_cpu

            # Store backward pass metrics
            performance_metrics["backward_memory"].append(
                mem_after_backward - mem_before_backward
            )
            performance_metrics["backward_times"].append(backward_time)
            performance_metrics["backward_cpu"].append(backward_cpu)

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()

        # Compute performance statistics
        def compute_stats(metrics):
            return {
                "mean": np.mean(metrics),
                "std": np.std(metrics),
                "min": np.min(metrics),
                "max": np.max(metrics),
            }

        # Logging and detailed metrics
        logger.info("\nPerformance Metrics:")

        # Memory Usage
        logger.info("\nMemory Usage (megabytes):")
        forward_mem = compute_stats(performance_metrics["forward_memory"])
        backward_mem = compute_stats(performance_metrics["backward_memory"])

        logger.info("Forward Pass Memory:")
        logger.info(
            f"  Mean: {forward_mem['mean'] / 1_000_000:.2f} ± {forward_mem['std'] / 1_000_000:.2f} MB"
        )
        logger.info("Backward Pass Memory:")
        logger.info(
            f"  Mean: {backward_mem['mean'] / 1_000_000:.2f} ± {backward_mem['std'] / 1_000_000:.2f} MB"
        )

        # CPU Usage
        logger.info("\nCPU Usage (%):")
        forward_cpu = compute_stats(performance_metrics["forward_cpu"])
        backward_cpu = compute_stats(performance_metrics["backward_cpu"])

        logger.info("Forward Pass CPU:")
        logger.info(f"  Mean: {forward_cpu['mean']:.2f}% ± {forward_cpu['std']:.2f}%")
        logger.info("Backward Pass CPU:")
        logger.info(f"  Mean: {backward_cpu['mean']:.2f}% ± {backward_cpu['std']:.2f}%")

        # Timing
        logger.info("\nTiming (seconds):")
        forward_time = compute_stats(performance_metrics["forward_times"])
        backward_time = compute_stats(performance_metrics["backward_times"])

        logger.info("Forward Pass:")
        logger.info(
            f"  Mean: {forward_time['mean']:.6f} ± {forward_time['std']:.6f} seconds"
        )
        logger.info("Backward Pass:")
        logger.info(
            f"  Mean: {backward_time['mean']:.6f} ± {backward_time['std']:.6f} seconds"
        )
