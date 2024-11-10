from autograd import nn, optim, functional, utils
from sklearn.datasets import fetch_openml
import logging
import numpy as np

logger = logging.getLogger(__name__)
np.random.seed(1337)


class MnistMultiClassClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.h1 = nn.Linear(784, 64)  # mnist image has shape 28*28=784
        self.h2 = nn.Linear(64, 64)
        self.h3 = nn.Linear(64, 10)

    def forward(self, x):
        x = functional.relu(self.h1(x))
        x = functional.relu(self.h2(x))
        x = self.h3(x)
        return functional.softmax(
            x
        )  # <<--- this is the only difference between this and the One vs Rest Binary classifier


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


if __name__ == "__main__":
    logger.info("Fetching data for MNIST_784")
    X, y = fetch_openml(
        "mnist_784", return_X_y=True, as_frame=False, parser="liac-arff"
    )
    X = X / 255.0  # normalize to [0, 1] to speed up convergence

    X_train, X_test, y_train, y_test = utils.train_test_split(X, y, test_size=0.1)

    logger.info("=" * 50)
    logger.info("║   Starting to train One vs Rest MNIST model    ║")
    logger.info("=" * 50)
    one_vs_rest_models = []
    for digit in range(10):
        logger.info(f"Training {digit=}")
        y_binary = (y_train == str(digit)).astype(int)
        model = MnistOneVsRestBinaryClassifier()

        utils.train(
            model=model,
            X=X_train,
            y=y_binary,
            loss_fn=functional.binary_cross_entropy,
            optimizer=optim.SGD(model.parameters, lr=1e-3),
            epochs=20,  # each digit will run for small number of epochs
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

    logger.info("=" * 66)
    logger.info("║            Starting to train Multi-class MNIST model           ║")
    logger.info("=" * 66)

    model = MnistMultiClassClassifier()
    utils.train(
        model=model,
        X=X_train,
        y=y_train.astype(int),
        loss_fn=functional.sparse_cross_entropy,
        optimizer=optim.SGD(model.parameters, lr=1e-3),
        epochs=500,
    )

    y_pred = model(X_test).data

    logger.info(
        f"Test Accuracy: {utils.accuracy(y_pred.argmax(axis=1), y_test.astype(int))}"
    )
    logger.info(
        f"Test Precision: {utils.precision(y_pred.argmax(axis=1), y_test.astype(int))}"
    )
