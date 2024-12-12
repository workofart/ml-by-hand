import numpy as np
import time
import psutil
import os
import logging
from autograd import nn, optim, functional
from autograd.tensor import Tensor
from unittest import TestCase
from sklearn.datasets import load_breast_cancer, load_digits
import tracemalloc

logger = logging.getLogger(__name__)


class CIPipelinePerformanceTest(TestCase):
    def setUp(self):
        tracemalloc.start()
        self.process = psutil.Process(os.getpid())

    def tearDown(self):
        tracemalloc.stop()

    def _measure_model_performance(
        self, model, x, y_true, optimizer, total_epochs, loss_fn
    ):
        """Helper method to measure performance metrics for any model"""
        performance_metrics = {
            "forward_memory": [],
            "backward_memory": [],
            "forward_cpu": [],
            "backward_cpu": [],
            "forward_times": [],
            "backward_times": [],
        }

        # Warm-up iteration
        _ = model(x)

        # Performance measurement
        for _ in range(total_epochs):
            # Reset memory tracking at the start of each iteration
            tracemalloc.reset_peak()
            start_memory = tracemalloc.get_traced_memory()[0]

            # Forward pass measurements
            start_cpu = self.process.cpu_percent()
            start_time = time.perf_counter()

            y_pred = model(x)

            # Capture forward pass memory
            end_memory = tracemalloc.get_traced_memory()[0]
            forward_memory = max(0, end_memory - start_memory)  # Ensure non-negative
            forward_time = time.perf_counter() - start_time
            forward_cpu = max(0, self.process.cpu_percent() - start_cpu)

            performance_metrics["forward_memory"].append(forward_memory)
            performance_metrics["forward_times"].append(forward_time)
            performance_metrics["forward_cpu"].append(forward_cpu)

            # Backward pass measurements
            tracemalloc.reset_peak()
            start_memory = tracemalloc.get_traced_memory()[0]
            start_cpu = self.process.cpu_percent()
            start_time = time.perf_counter()

            loss = loss_fn(y_pred, y_true)
            loss.backward()

            # Capture backward pass memory
            end_memory = tracemalloc.get_traced_memory()[0]
            backward_memory = max(0, end_memory - start_memory)  # Ensure non-negative
            backward_time = time.perf_counter() - start_time
            backward_cpu = max(0, self.process.cpu_percent() - start_cpu)

            performance_metrics["backward_memory"].append(backward_memory)
            performance_metrics["backward_times"].append(backward_time)
            performance_metrics["backward_cpu"].append(backward_cpu)

            optimizer.step()
            optimizer.zero_grad()

        return performance_metrics

    def _log_performance_metrics(self, metrics, model_name):
        """Helper method to log performance metrics"""

        def compute_stats(metrics_list):
            return {
                "mean": np.mean(metrics_list),
                "std": np.std(metrics_list),
                "min": np.min(metrics_list),
                "max": np.max(metrics_list),
            }

        logger.info(f"\nPerformance Metrics for {model_name}:")

        # Memory Usage
        logger.info("\nMemory Usage (megabytes):")
        forward_mem = compute_stats(metrics["forward_memory"])
        backward_mem = compute_stats(metrics["backward_memory"])

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
        forward_cpu = compute_stats(metrics["forward_cpu"])
        backward_cpu = compute_stats(metrics["backward_cpu"])

        logger.info("Forward Pass CPU:")
        logger.info(f"  Mean: {forward_cpu['mean']:.2f}% ± {forward_cpu['std']:.2f}%")
        logger.info("Backward Pass CPU:")
        logger.info(f"  Mean: {backward_cpu['mean']:.2f}% ± {backward_cpu['std']:.2f}%")

        # Timing
        logger.info("\nTiming (seconds):")
        forward_time = compute_stats(metrics["forward_times"])
        backward_time = compute_stats(metrics["backward_times"])

        logger.info("Forward Pass:")
        logger.info(
            f"  Mean: {forward_time['mean']:.6f} ± {forward_time['std']:.6f} seconds"
        )
        logger.info("Backward Pass:")
        logger.info(
            f"  Mean: {backward_time['mean']:.6f} ± {backward_time['std']:.6f} seconds"
        )

    def test_resource_usage_metrics(self):
        """
        Measure computational efficiency with memory and CPU usage metrics for both
        a simple neural network and a CNN
        """
        np.random.seed(42)

        # Test 1: Simple Neural Network on Binary Classification
        batch_size_mlp = 128
        input_size = 30
        hidden_size = 64
        total_epochs = 100

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.h1 = nn.Linear(input_size, hidden_size)
                self.bn1 = nn.BatchNorm(hidden_size)
                self.h2 = nn.Linear(hidden_size, hidden_size)
                self.bn2 = nn.BatchNorm(hidden_size)
                self.h3 = nn.Linear(hidden_size, 1)

            def forward(self, x):
                x = functional.relu(self.bn1(self.h1(x)))
                x = functional.relu(self.bn2(self.h2(x)))
                return functional.sigmoid(self.h3(x))

        # Load binary classification dataset
        X, y = load_breast_cancer(return_X_y=True)
        x_mlp = Tensor(X[:batch_size_mlp])
        y_mlp = Tensor(y[:batch_size_mlp].astype(np.float32))

        mlp_model = SimpleModel()
        mlp_optimizer = optim.SGD(mlp_model.parameters, lr=0.01)

        # Measure MLP performance
        mlp_metrics = self._measure_model_performance(
            mlp_model,
            x_mlp,
            y_mlp,
            mlp_optimizer,
            total_epochs,
            lambda y_pred, y_true: functional.binary_cross_entropy(y_pred, y_true),
        )
        self._log_performance_metrics(mlp_metrics, "Simple Neural Network")

        # Test 2: CNN on Digit Classification
        batch_size_cnn = 32
        input_channels = 1
        num_classes = 10

        class CNNModel(nn.Module):
            def __init__(self):
                super().__init__()
                # First conv block
                self.conv1 = nn.Conv2d(
                    input_channels, 16, kernel_size=3, padding_mode="same"
                )
                self.bn1 = nn.BatchNorm(
                    16 * 8 * 8
                )  # Adjust BatchNorm for flattened conv output
                self.pool1 = nn.MaxPool2d(kernel_size=2)

                # Second conv block
                self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding_mode="same")
                self.bn2 = nn.BatchNorm(
                    32 * 4 * 4
                )  # Adjust BatchNorm for flattened conv output
                self.pool2 = nn.MaxPool2d(kernel_size=2)

                self.fc1 = nn.Linear(32 * 2 * 2, 64)
                self.fc2 = nn.Linear(64, num_classes)

            def forward(self, x):
                # First conv block
                x = self.conv1(x)
                batch_size = x.shape[0]
                x = x.reshape(batch_size, -1)  # Flatten for BatchNorm
                x = self.bn1(x)
                x = x.reshape(batch_size, 16, 8, 8)  # Reshape back to 4D
                x = functional.relu(x)
                x = self.pool1(x)

                # Second conv block
                x = self.conv2(x)
                batch_size = x.shape[0]
                x = x.reshape(batch_size, -1)  # Flatten for BatchNorm
                x = self.bn2(x)
                x = x.reshape(batch_size, 32, 4, 4)  # Reshape back to 4D
                x = functional.relu(x)
                x = self.pool2(x)

                # Fully connected layers
                x = x.reshape(batch_size, -1)
                x = functional.relu(self.fc1(x))
                x = self.fc2(x)
                return functional.softmax(x)

        # Load and preprocess digits dataset
        digits = load_digits()
        X_digits = (
            digits.images.reshape(-1, 1, 8, 8) / 16.0
        )  # Normalize to [0,1] and add channel dimension
        y_digits = digits.target  # Use integer labels

        x_cnn = Tensor(X_digits[:batch_size_cnn])
        y_cnn = np.array(
            y_digits[:batch_size_cnn], dtype=np.int64
        )  # Ensure numpy integer array

        cnn_model = CNNModel()
        cnn_optimizer = optim.SGD(cnn_model.parameters, lr=0.01)

        # Measure CNN performance
        cnn_metrics = self._measure_model_performance(
            cnn_model,
            x_cnn,
            y_cnn,
            cnn_optimizer,
            total_epochs,
            lambda y_pred, y_true: functional.sparse_cross_entropy(y_pred, y_true),
        )
        self._log_performance_metrics(cnn_metrics, "CNN Model")
