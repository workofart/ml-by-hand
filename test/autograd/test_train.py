import numpy as np
from autograd import nn, optim, functional
from sklearn.datasets import load_breast_cancer
from openml.datasets import get_dataset
from unittest import TestCase
import logging

from autograd.tools.trainer import Trainer
from autograd.tools.metrics import accuracy, mean_squared_error

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
    def test_binary_classification(self):
        X, y = load_breast_cancer(return_X_y=True)
        logger.info(f"Dataset: {X.shape=}, {y.shape=}")

        X = (X - X.mean()) / X.std()

        model = Classifier(input_size=X.shape[-1], hidden_size=64, output_size=1)
        total_params = sum(
            [
                np.prod(v.data.shape)
                for k, module in model.parameters.items()
                for _, v in module.items()
            ]
        )
        logger.info(f"Number of parameters: {total_params}")

        loss_fn = functional.binary_cross_entropy
        optimizer = optim.SGD(model.parameters, lr=1e-3)
        trainer = Trainer(
            model,
            loss_fn,
            optimizer,
            epochs=1000,
            output_type="sigmoid",
        )
        trainer.fit(X, y)

        model.eval()
        y_pred = model(X).data

        # compare y_pred and y on the classification accuracy
        acc = accuracy((y_pred > 0.5).astype(int).squeeze(), y)
        logger.info(f"Accuracy: {acc}")
        assert acc > 0.9

    def test_regression(self):
        X, y, _, __ = get_dataset(dataset_id=44971, download_data=True).get_data(
            target="quality", dataset_format="array"
        )
        logger.info(f"Dataset: {X.shape=}, {y.shape=}")
        logger.info(f"y unique values: {np.unique(y)}")

        eps = 1e-8
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + eps)
        y = (y - y.mean()) / (y.std() + eps)

        model = RegressionModel(
            input_size=X.shape[-1], hidden_size=32, output_size=1
        )  # Smaller network
        trainer = Trainer(
            model,
            functional.mean_squared_loss,
            optim.Adam(model.parameters, lr=1e-5),  # Smaller learning rate
            epochs=100,
            output_type=None,
        )
        trainer.fit(X, y)

        model.eval()
        y_pred = model(X).data

        logger.info(f"Mean Squared Error: {mean_squared_error(y_pred, y):.2f}")
        assert mean_squared_error(y_pred, y) < 1.3
