import numpy as np
from autograd import nn, optim, functional, utils
from sklearn.datasets import load_breast_cancer
from unittest import TestCase
import logging

logger = logging.getLogger(__name__)


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
        utils.train(model, X, y, loss_fn, optimizer, epochs=1000)

        y_pred = model(X).data

        # compare y_pred and y on the classification accuracy
        logger.info(f"Accuracy: {(sum((y_pred > 0.5).astype(int) == y) / X.shape[0])}")
