import logging
import os
import time
from unittest import TestCase

try:
    # drop-in replacement for numpy for GPU acceleration
    import cupy as np  # type: ignore
except ImportError:
    import numpy as np
import psutil

from autograd import functional, nn, optim
from autograd.tensor import Tensor

logger = logging.getLogger(__name__)


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
        x = self.final(x)
        return functional.softmax(x)


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
        x = self.fc2(x)
        return functional.softmax(x)


class CIPipelinePerformanceTest(TestCase):
    def setUp(self):
        self.process = psutil.Process(os.getpid())

    def tearDown(self):
        pass

    def _measure_model_performance(
        self, model, x, y_true, optimizer, total_epochs, loss_fn
    ):
        """
        Measure the forward and backward pass times for the given model and optimizer
        over a specified number of epochs. Returns a dictionary with recorded timings.
        """
        performance_metrics = {
            "forward_times": [],
            "backward_times": [],
        }

        # Warm-up iteration (not timed)
        _ = model(x)

        # Actual performance measurements
        for _ in range(total_epochs):
            # Forward pass timing
            start_time = time.perf_counter()
            y_pred = model(x)
            forward_elapsed = time.perf_counter() - start_time
            performance_metrics["forward_times"].append(forward_elapsed)

            # Backward pass timing
            start_time = time.perf_counter()
            loss = loss_fn(y_pred, y_true)
            loss.backward()
            backward_elapsed = time.perf_counter() - start_time
            performance_metrics["backward_times"].append(backward_elapsed)

            # Update parameters
            optimizer.step()
            optimizer.zero_grad()

        return performance_metrics

    def _compute_stats(self, metrics_list):
        """Compute mean, std, min, max statistics for a given list of values."""
        return {
            "mean": np.mean(metrics_list),
            "std": np.std(metrics_list),
            "min": np.min(metrics_list),
            "max": np.max(metrics_list),
        }

    def _log_performance_metrics(self, metrics, model_name):
        """
        Log forward and backward pass performance metrics (mean, std, min, max).
        """
        logger.info(f"\nPerformance Metrics for {model_name}:")

        forward_stats = self._compute_stats(np.array(metrics["forward_times"]))
        backward_stats = self._compute_stats(np.array(metrics["backward_times"]))

        logger.info("Timing (seconds):")
        logger.info("Forward Pass:")
        logger.info(
            f"  Mean: {forward_stats['mean']:.6f} ± {forward_stats['std']:.6f} "
            f"(Min: {forward_stats['min']:.6f}, Max: {forward_stats['max']:.6f})"
        )
        logger.info("Backward Pass:")
        logger.info(
            f"  Mean: {backward_stats['mean']:.6f} ± {backward_stats['std']:.6f} "
            f"(Min: {backward_stats['min']:.6f}, Max: {backward_stats['max']:.6f})"
        )

    def _test_complex_mlp(self, total_epochs):
        """Test performance metrics on a complex MLP model."""
        input_size = 1024
        hidden_size = 512
        num_layers = 4
        output_size = 10
        batch_size = 1024

        # Generate random input and labels
        X = np.random.randn(batch_size, input_size).astype(np.float32)
        y = np.random.randint(0, output_size, size=(batch_size,)).astype(np.int64)

        # Initialize model and optimizer
        mlp_model = ComplexMLP(input_size, hidden_size, num_layers, output_size)
        mlp_optimizer = optim.SGD(mlp_model.parameters, lr=0.01)

        # Convert data to tensors
        x_tensor = Tensor(X)
        y_tensor = Tensor(y)

        # Measure and log performance
        mlp_metrics = self._measure_model_performance(
            mlp_model,
            x_tensor,
            y_tensor,
            mlp_optimizer,
            total_epochs,
            functional.binary_cross_entropy,
        )
        self._log_performance_metrics(mlp_metrics, "Complex MLP Model")

    def _test_deep_cnn(self, total_epochs):
        """Test performance metrics on a deeper CNN model."""
        input_channels = 3
        image_size = 24
        num_classes = 20
        batch_size = 256

        # Generate random input and labels
        X = np.random.randn(batch_size, input_channels, image_size, image_size).astype(
            np.float32
        )
        y = np.random.randint(0, num_classes, size=(batch_size,)).astype(np.int64)

        # Initialize model and optimizer
        cnn_model = DeepCNN(input_channels, image_size, num_classes)
        cnn_optimizer = optim.SGD(cnn_model.parameters, lr=0.01)

        x_tensor = Tensor(X)
        y_tensor = y  # Not wrapped in Tensor to highlight loss usage consistency

        # Measure and log performance
        cnn_metrics = self._measure_model_performance(
            cnn_model,
            x_tensor,
            y_tensor,
            cnn_optimizer,
            total_epochs,
            functional.binary_cross_entropy,
        )
        self._log_performance_metrics(cnn_metrics, "Deep CNN Model")

    def test_resource_usage_metrics(self):
        """
        Test that focuses on raw computational time for forward and backward passes
        for both a complex MLP and a deep CNN model.
        """
        np.random.seed(42)
        total_epochs = 10
        logger.info(f"Running {total_epochs} epochs for performance measurement...")

        # Run performance tests on different models
        self._test_complex_mlp(total_epochs)
        self._test_deep_cnn(total_epochs)
