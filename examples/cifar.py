from autograd import nn, functional, optim
from autograd import utils
from openml.datasets import get_dataset
import numpy as np
import logging

logger = logging.getLogger(__name__)
np.random.seed(1337)


class CifarMulticlassClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.h1 = nn.Linear(
            3072, 64
        )  # cifar-10 image has shape 32 x 32 x 3 (color channels)
        self.h2 = nn.Linear(64, 64)
        self.h3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = functional.relu(self.h1(x))
        x = functional.relu(self.h2(x))
        x = self.h3(x)
        return functional.softmax(x)


def train_cifar_multiclass_model(model: nn.Module, X_train, y_train, X_test, y_test):
    logger.info("=" * 66)
    logger.info("Starting to train Multi-class CIFAR-10 model")
    logger.info("=" * 66)

    utils.train(
        model=model,
        X=X_train,
        y=y_train.astype(int),
        loss_fn=functional.sparse_cross_entropy,
        optimizer=optim.SGD(model.parameters, lr=1e-3),
        epochs=500,
        batch_size=1024,
    )

    y_pred = model(X_test).data

    logger.info(
        f"Test Accuracy: {utils.accuracy(y_pred.argmax(axis=1), y_test.astype(int))}"
    )
    logger.info(
        f"Test Precision: {utils.precision(y_pred.argmax(axis=1), y_test.astype(int))}"
    )


if __name__ == "__main__":
    logger.info("Fetching data for CIFAR-10")
    X, y, _, __ = get_dataset(dataset_id=40927, download_data=True).get_data(
        target="class", dataset_format="array"
    )
    X = X / 255.0  # normalize to [0, 1] to speed up convergence

    X_train, X_test, y_train, y_test = utils.train_test_split(X, y, test_size=0.1)
    cifar10_model = CifarMulticlassClassifier(num_classes=10)
    train_cifar_multiclass_model(cifar10_model, X_train, y_train, X_test, y_test)

    logger.info("Fetching data for CIFAR-100")
    X, y, _, __ = get_dataset(dataset_id=41983, download_data=True).get_data(
        target="class", dataset_format="array"
    )
    X = X / 255.0  # normalize to [0, 1] to speed up convergence

    X_train, X_test, y_train, y_test = utils.train_test_split(X, y, test_size=0.1)
    cifar100_model = CifarMulticlassClassifier(num_classes=100)
    train_cifar_multiclass_model(cifar100_model, X_train, y_train, X_test, y_test)
