import logging
import os
from unittest import TestCase

try:
    # drop-in replacement for numpy for GPU acceleration
    import cupy as np  # type: ignore

    _ = np.cuda.runtime.getDeviceCount()  # Check if a CUDA device is available
except Exception:
    import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes

from autograd import functional, nn, optim
from autograd.tools.data import SimpleDataLoader
from autograd.tools.metrics import accuracy, mean_squared_error
from autograd.tools.trainer import SimpleTrainer

logger = logging.getLogger(__name__)

np.random.seed(42)


class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = functional.relu(self.linear1(x))
        x = functional.relu(self.linear2(x))
        x = functional.sigmoid(self.linear3(x))
        return x


class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RegressionModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = functional.relu(self.linear1(x))
        x = functional.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class TestTrain(TestCase):
    def tearDown(self) -> None:
        reg_metrics_path = (
            f"{SimpleTrainer.METRICS_DIR}/{RegressionModel.__name__}_default.npz"
        )
        classifier_metrics_path = (
            f"{SimpleTrainer.METRICS_DIR}/{Classifier.__name__}_default.npz"
        )
        if os.path.exists(reg_metrics_path):
            os.remove(reg_metrics_path)
        if os.path.exists(classifier_metrics_path):
            os.remove(classifier_metrics_path)

    def test_binary_classification(self):
        X, y = load_breast_cancer(return_X_y=True)
        logger.info(f"Dataset: {X.shape=}, {y.shape=}")

        X = (X - X.mean()) / X.std()

        model = Classifier(input_size=X.shape[-1], hidden_size=64, output_size=1)
        logger.info(f"Number of parameters: {model.num_parameters()}")

        loss_fn = functional.binary_cross_entropy
        optimizer = optim.SGD(model.parameters, lr=1e-3)
        train_data_loader = SimpleDataLoader(X, y, batch_size=32, shuffle=True)
        trainer = SimpleTrainer(
            model,
            loss_fn,
            optimizer,
            epochs=1000,
            output_type="sigmoid",
        )
        trainer.fit(train_data_loader)

        model.eval()
        y_pred = model(X).data

        # compare y_pred and y on the classification accuracy
        acc = accuracy((y_pred > 0.5).astype(int).squeeze(), y)
        logger.info(f"Accuracy: {acc}")
        assert acc > 0.9

    def test_regression(self):
        X, y = load_diabetes(return_X_y=True)
        logger.info(f"Dataset: {X.shape=}, {y.shape=}")
        logger.info(f"y unique values: {np.unique(y)}")

        eps = 1e-8
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + eps)
        y = (y - y.mean()) / (y.std() + eps)

        train_data_loader = SimpleDataLoader(X, y, batch_size=32, shuffle=True)

        model = RegressionModel(
            input_size=X.shape[-1], hidden_size=32, output_size=1
        )  # Smaller network
        trainer = SimpleTrainer(
            model,
            functional.mean_squared_loss,
            optim.Adam(model.parameters, lr=1e-4),  # Smaller learning rate
            epochs=100,
            output_type=None,
        )
        trainer.fit(train_data_loader)

        model.eval()
        y_pred = model(X).data

        logger.info(f"Mean Squared Error: {mean_squared_error(y_pred, y):.2f}")
        assert mean_squared_error(y_pred, y) < 1.3
