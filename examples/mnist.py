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


class MnistConvolutionalClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # First conv layer:
        # Input: 28x28x1
        # Conv2d(kernel=3, padding='same'): maintains 28x28 spatial dimensions
        # Channels: 1 -> 16
        # Output: 28x28x16
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=3, padding_mode="same"
        )
        # MaxPool2d(kernel=2, stride=2):
        # Reduces spatial dimensions from 28x28 to 14x14
        # (28 - 2) / 2 + 1 = 14
        # Output: 14x14x16
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second conv layer:
        # Input: 14x14x16
        # Conv2d(kernel=3, padding='same'): maintains 14x14 spatial dimensions
        # Channels: 16 -> 32
        # Output: 14x14x32
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding_mode="same"
        )

        # Flattened input to fully connected layer:
        # 14x14x32 = 32 * 14 * 14 = 12544 features
        self.fc1 = nn.Linear(32 * 14 * 14, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        batch_size = x.shape[0]

        # Input reshape: (batch_size, 784) -> (batch_size, 1, 28, 28)
        x = x.reshape(batch_size, 1, 28, 28)

        # First conv block
        x = functional.relu(self.conv1(x))
        x = self.pool1(x)  # Reduces to ~13x13

        # Second conv block
        x = functional.relu(self.conv2(x))

        # Flatten
        x = x.view(batch_size, -1)

        # Dense layers
        x = functional.relu(self.fc1(x))
        return functional.softmax(self.fc2(x))


class MnistOneVsRestBinaryClassifier(nn.Module):
    def __init__(self, with_logits=True):
        super().__init__()
        self.h1 = nn.Linear(784, 64)  # mnist image has shape 28*28=784
        self.h2 = nn.Linear(64, 64)
        self.h3 = nn.Linear(64, 1)
        self.with_logits = with_logits

    def forward(self, x):
        x = functional.relu(self.h1(x))
        x = functional.relu(self.h2(x))
        x = self.h3(x)
        if self.with_logits:
            return functional.sigmoid(x)
        return x


def train_mnist_with_hinge_loss(X_train, y_train, X_test, y_test):
    logger.info("=" * 50)
    logger.info("Starting to train One vs Rest MNIST model")
    logger.info("=" * 50)
    one_vs_rest_models = []
    for digit in range(10):
        logger.info(f"Training {digit=}")
        y_binary = (y_train == digit).astype(int)
        y_binary = 2 * y_binary - 1  # Convert from {0,1} to {-1,1}
        model = MnistOneVsRestBinaryClassifier(with_logits=False)

        utils.train(
            model=model,
            X=X_train,
            y=y_binary,
            loss_fn=lambda pred, true: functional.hinge_loss(
                pred, true, reduction="mean"
            ),
            optimizer=optim.Adam(model.parameters, lr=1e-3),
            epochs=10,
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


def train_mnist_one_vs_rest_model(X_train, y_train, X_test, y_test):
    logger.info("=" * 50)
    logger.info("Starting to train One vs Rest MNIST model")
    logger.info("=" * 50)
    one_vs_rest_models = []
    for digit in range(10):
        logger.info(f"Training {digit=}")
        y_binary = (y_train == digit).astype(int)
        model = MnistOneVsRestBinaryClassifier(with_logits=False)

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
    X_train, y_train, X_test, y_test, optimizer, model, loss_fn, msg=""
):
    logger.info("=" * 66)
    logger.info(f"Starting to train Multi-class MNIST model {msg}")
    logger.info("=" * 66)
    utils.train(
        model=model,
        X=X_train,
        y=y_train.astype(int),
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=15,
        batch_size=256,
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

    logger.info(f"X shape: {X.shape}")
    logger.info(f"y shape: {y.shape}")

    X_train, X_test, y_train, y_test = utils.train_test_split(X, y, test_size=0.1)

    model = MnistConvolutionalClassifier()
    train_mnist_multiclass_model(
        X_train,
        y_train,
        X_test,
        y_test,
        optimizer=optim.Adam(model.parameters, lr=1e-3),
        model=model,
        loss_fn=functional.sparse_cross_entropy,
        msg="(with batch norm, Adam optimizer)",
    )

    # model = MnistMultiClassClassifier(batch_norm=False)
    # train_mnist_multiclass_model(
    #     X_train,
    #     y_train,
    #     X_test,
    #     y_test,
    #     optimizer=optim.SGD(model.parameters, lr=1e-3),
    #     model=model,
    #     loss_fn=functional.sparse_cross_entropy,
    #     msg="(without batch norm, SGD optimizer)",
    # )

    # model = MnistMultiClassClassifier(batch_norm=True)
    # train_mnist_multiclass_model(
    #     X_train,
    #     y_train,
    #     X_test,
    #     y_test,
    #     optimizer=optim.SGD(model.parameters, lr=1e-3),
    #     model=model,
    #     loss_fn=functional.sparse_cross_entropy,
    #     msg="(with batch norm, SGD optimizer)",
    # )

    # model = MnistMultiClassClassifier(batch_norm=True)
    # train_mnist_multiclass_model(
    #     X_train,
    #     y_train,
    #     X_test,
    #     y_test,
    #     optimizer=optim.Adam(model.parameters, lr=1e-3),
    #     model=model,
    #     loss_fn=functional.sparse_cross_entropy,
    #     msg="(with batch norm, Adam optimizer)",
    # )

    # train_mnist_one_vs_rest_model(X_train, y_train, X_test, y_test)

    # train_mnist_with_hinge_loss(X_train, y_train, X_test, y_test)
