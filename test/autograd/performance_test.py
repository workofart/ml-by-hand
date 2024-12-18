import numpy as np
import time
import psutil
import os
import logging
from autograd import nn, optim, functional
from autograd.tensor import Tensor
from unittest import TestCase
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
        Test with significantly more complex models and larger inputs to better highlight
        performance differences from code optimizations.
        """

        np.random.seed(42)

        total_epochs = 10  # Still large, but not too large to be impractical
        logger.info(f"Running {total_epochs} epochs for performance measurement...")

        ####### Test 1: Complex MLP #######
        # Simulate a higher-dimensional input (e.g., 1024 features)
        # Use larger hidden layers and multiple layers to increase complexity
        input_size = 1024
        hidden_size = 512
        num_layers = 4  # More layers
        output_size = 10
        batch_size_mlp = 1024  # Large batch size

        # Synthetic dataset
        X_mlp = np.random.randn(batch_size_mlp, input_size).astype(np.float32)
        y_mlp = np.random.randint(0, output_size, size=(batch_size_mlp,)).astype(
            np.int64
        )

        class ComplexMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = []
                in_size = input_size
                for _ in range(num_layers):
                    linear = nn.Linear(in_size, hidden_size)
                    bn = nn.BatchNorm(hidden_size)
                    self.layers.append((linear, bn))
                    in_size = hidden_size
                # Final layer
                self.final = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                for linear, bn in self.layers:
                    x = linear(x)
                    x = bn(x)
                    x = functional.relu(x)
                x = self.final(x)
                return functional.softmax(x)

        mlp_model = ComplexMLP()
        mlp_optimizer = optim.SGD(mlp_model.parameters, lr=0.01)

        x_mlp_t = Tensor(X_mlp)
        y_mlp_t = Tensor(y_mlp)

        mlp_metrics = self._measure_model_performance(
            mlp_model,
            x_mlp_t,
            y_mlp_t,
            mlp_optimizer,
            total_epochs,
            functional.binary_cross_entropy,
        )
        self._log_performance_metrics(mlp_metrics, "Complex MLP Model")

        ####### Test 2: Larger CNN #######
        # Increase image size and depth. For example, 64x64 images, 3 channels (like RGB)
        # More layers and channels in CNN to increase complexity.

        input_channels = 3
        image_size = 24
        num_classes = 20
        batch_size_cnn = 256  # Larger batch
        # Synthetic dataset: (batch, channels, height, width)
        X_cnn = np.random.randn(
            batch_size_cnn, input_channels, image_size, image_size
        ).astype(np.float32)
        y_cnn = np.random.randint(0, num_classes, size=(batch_size_cnn,)).astype(
            np.int64
        )

        class DeepCNN(nn.Module):
            def __init__(self):
                super().__init__()
                # First conv block
                self.conv1 = nn.Conv2d(
                    input_channels, 32, kernel_size=3, padding_mode="same"
                )
                self.bn1 = nn.BatchNorm(32 * image_size * image_size)
                self.pool1 = nn.MaxPool2d(kernel_size=2)

                # Second conv block
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding_mode="same")
                self.bn2 = nn.BatchNorm(64 * (image_size // 2) * (image_size // 2))
                self.pool2 = nn.MaxPool2d(kernel_size=2)

                # Third conv block
                self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding_mode="same")
                self.bn3 = nn.BatchNorm(128 * (image_size // 4) * (image_size // 4))
                self.pool3 = nn.MaxPool2d(kernel_size=2)

                # Fully connected layers
                fc_input = 128 * (image_size // 8) * (image_size // 8)
                self.fc1 = nn.Linear(fc_input, 512)
                self.fc2 = nn.Linear(512, num_classes)

            def forward(self, x):
                # Conv block 1
                x = self.conv1(x)
                batch_size = x.shape[0]
                x = x.reshape(batch_size, -1)  # Flatten for BN
                x = self.bn1(x)
                x = x.reshape(batch_size, 32, image_size, image_size)
                x = functional.relu(x)
                x = self.pool1(x)

                # Conv block 2
                x = self.conv2(x)
                batch_size = x.shape[0]
                x = x.reshape(batch_size, -1)
                x = self.bn2(x)
                x = x.reshape(batch_size, 64, image_size // 2, image_size // 2)
                x = functional.relu(x)
                x = self.pool2(x)

                # Conv block 3
                x = self.conv3(x)
                batch_size = x.shape[0]
                x = x.reshape(batch_size, -1)
                x = self.bn3(x)
                x = x.reshape(batch_size, 128, image_size // 4, image_size // 4)
                x = functional.relu(x)
                x = self.pool3(x)

                # FC layers
                x = x.reshape(batch_size, -1)
                x = functional.relu(self.fc1(x))
                x = self.fc2(x)
                return functional.softmax(x)

        cnn_model = DeepCNN()
        cnn_optimizer = optim.SGD(cnn_model.parameters, lr=0.01)

        x_cnn_t = Tensor(X_cnn)
        y_cnn_t = y_cnn  # already int64

        cnn_metrics = self._measure_model_performance(
            cnn_model,
            x_cnn_t,
            y_cnn_t,
            cnn_optimizer,
            total_epochs,
            functional.binary_cross_entropy,
        )
        self._log_performance_metrics(cnn_metrics, "Deep CNN Model")
