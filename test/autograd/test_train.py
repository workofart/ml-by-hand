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
from autograd.tools.config_schema import GenericTrainingConfig
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

        CONFIG = GenericTrainingConfig(
            training_run_name="default",
            dataset_name="breast_cancer",
            eval_iters=10,
            steps_per_epoch=10,
            checkpoint_freq=10,
            resume_epoch=None,
            total_epochs=1000,
            batch_size=32,
            model_kwargs={
                "input_size": X.shape[-1],
                "hidden_size": 64,
                "output_size": 1,
            },
            optimizer_kwargs={
                "lr": 1e-3,
            },
        )
        train_data_loader = SimpleDataLoader(X, y, batch_size=32, shuffle=True)
        trainer = SimpleTrainer(
            model_cls=Classifier,
            optimizer_cls=optim.SGD,
            loss_fn=functional.binary_cross_entropy,
            config=CONFIG,
            output_type="sigmoid",
        )
        trainer.fit(train_data_loader)

        trainer.model.eval()
        y_pred = trainer.model(X).data

        # compare y_pred and y on the classification accuracy
        acc = accuracy((y_pred > 0.5).astype(int).squeeze(), y)
        logger.info(f"Accuracy: {acc}")
        assert acc > 0.9

    def test_regression(self):
        X, y = load_diabetes(return_X_y=True)
        logger.info(f"Dataset: {X.shape=}, {y.shape=}")
        logger.info(f"y unique values: {np.unique(y)}")

        CONFIG = GenericTrainingConfig(
            training_run_name="default",
            dataset_name="diabetes",
            eval_iters=10,
            steps_per_epoch=10,
            checkpoint_freq=10,
            resume_epoch=None,
            total_epochs=200,
            batch_size=32,
            model_kwargs={
                "input_size": X.shape[-1],
                "hidden_size": 32,
                "output_size": 1,
            },
            optimizer_kwargs={
                "lr": 1e-4,
            },
        )

        eps = 1e-8
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + eps)
        y = (y - y.mean()) / (y.std() + eps)

        train_data_loader = SimpleDataLoader(X, y, batch_size=32, shuffle=True)

        trainer = SimpleTrainer(
            model_cls=RegressionModel,
            optimizer_cls=optim.Adam,
            loss_fn=functional.mean_squared_loss,
            config=CONFIG,
            output_type=None,
        )
        trainer.fit(train_data_loader)

        trainer.model.eval()
        y_pred = trainer.model(X).data

        logger.info(f"Mean Squared Error: {mean_squared_error(y_pred, y):.2f}")
        assert mean_squared_error(y_pred, y) < 1.3
