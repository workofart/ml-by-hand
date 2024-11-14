from autograd import nn, optim, functional, utils
from openml.datasets import get_dataset
import logging
import numpy as np

logger = logging.getLogger(__name__)
np.random.seed(1337)


class MnistMultiClassClassifier(nn.Module):
    def __init__(self, batch_norm=False):
        super().__init__()
        self.batch_norm = batch_norm
        self.h1 = nn.Linear(784, 64)  # mnist image has shape 28*28=784
        self.h2 = nn.Linear(64, 64)
        self.h3 = nn.Linear(64, 10)
        if self.batch_norm:
            self.bn1 = nn.BatchNorm(64, momentum=0.1, epsilon=1e-8)

    def forward(self, x):
        x = self.h1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = functional.relu(x)

        x = self.h2(x)
        x = functional.relu(x)

        x = self.h3(x)
        return functional.softmax(x)


class MnistOneVsRestBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.h1 = nn.Linear(784, 64)  # mnist image has shape 28*28=784
        self.h2 = nn.Linear(64, 64)
        self.h3 = nn.Linear(64, 1)

    def forward(self, x):
        x = functional.relu(self.h1(x))
        x = functional.relu(self.h2(x))
        x = self.h3(x)
        return functional.sigmoid(
            x
        )  # <<--- this is the only difference between this and the multiclass classifier


def train_mnist_one_vs_rest_model(X_train, y_train, X_test, y_test):
    logger.info("=" * 50)
    logger.info("Starting to train One vs Rest MNIST model")
    logger.info("=" * 50)
    one_vs_rest_models = []
    for digit in range(10):
        logger.info(f"Training {digit=}")
        y_binary = (y_train == digit).astype(int)
        model = MnistOneVsRestBinaryClassifier()

        utils.train(
            model=model,
            X=X_train,
            y=y_binary,
            loss_fn=functional.binary_cross_entropy,
            optimizer=optim.Adam(model.parameters, lr=1e-3),
            epochs=10,  # each digit will run for small number of epochs
        )
        one_vs_rest_models.append(model)

    logger.info("Training complete")
    # 10 models, each model predicts probability of digit 0-9
    # predictions_by_digit[i][j] is the probability that the ith test example is the jth digit
    predictions_by_digit = np.array(
        [model(X_test).data for model in one_vs_rest_models]
    ).T

    logger.info(
        f"Test Accuracy: {utils.accuracy(predictions_by_digit.argmax(axis=1), y_test.astype(int))}"
    )
    logger.info(
        f"Test Precision: {utils.precision(predictions_by_digit.argmax(axis=1), y_test.astype(int))}"
    )


def train_mnist_multiclass_model(
    X_train, y_train, X_test, y_test, optimizer, model, msg=""
):
    logger.info("=" * 66)
    logger.info(f"Starting to train Multi-class MNIST model {msg}")
    logger.info("=" * 66)
    utils.train(
        model=model,
        X=X_train,
        y=y_train.astype(int),
        loss_fn=functional.sparse_cross_entropy,
        optimizer=optimizer,
        epochs=30,
    )

    model.eval()
    y_pred = model(X_test).data

    logger.info(
        f"Test Accuracy: {utils.accuracy(y_pred.argmax(axis=1), y_test.astype(int))}"
    )
    logger.info(
        f"Test Precision: {utils.precision(y_pred.argmax(axis=1), y_test.astype(int))}"
    )


if __name__ == "__main__":
    logger.info("Fetching data for MNIST_784")
    X, y, _, __ = get_dataset(dataset_id=554, download_data=True).get_data(
        target="class", dataset_format="array"
    )
    X /= 255.0  # normalize to [0, 1] to speed up convergence

    X_train, X_test, y_train, y_test = utils.train_test_split(X, y, test_size=0.1)

    model = MnistMultiClassClassifier(batch_norm=False)
    train_mnist_multiclass_model(
        X_train,
        y_train,
        X_test,
        y_test,
        optimizer=optim.SGD(model.parameters, lr=1e-3),
        model=model,
        msg="(without batch norm, SGD optimizer)",
    )

    model = MnistMultiClassClassifier(batch_norm=True)
    train_mnist_multiclass_model(
        X_train,
        y_train,
        X_test,
        y_test,
        optimizer=optim.SGD(model.parameters, lr=1e-3),
        model=model,
        msg="(with batch norm, SGD optimizer)",
    )

    model = MnistMultiClassClassifier(batch_norm=True)
    train_mnist_multiclass_model(
        X_train,
        y_train,
        X_test,
        y_test,
        optimizer=optim.Adam(model.parameters, lr=1e-3),
        model=model,
        msg="(with batch norm, Adam optimizer)",
    )

    train_mnist_one_vs_rest_model(X_train, y_train, X_test, y_test)
