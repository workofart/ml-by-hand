from autograd import nn, optim, functional
from openml.datasets import get_dataset
import logging
import numpy as np

from autograd.tools.data import train_test_split
from autograd.tools.trainer import Trainer
from autograd.tools.metrics import accuracy, precision

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
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=8, kernel_size=3, padding_mode="same"
        )  # (N, 8, 28, 28) maintain the same spatial dimensions because of "same" padding
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=8, kernel_size=3, padding_mode="same"
        )  # (N, 8, 28, 28) maintain the same spatial dimensions because of "same" padding
        self.pool1 = nn.MaxPool2d(
            kernel_size=3, stride=2
        )  # (N, 32, 12, 12), where (28 - 3) / 2 + 1 = 13

        self.conv3 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, padding_mode="same"
        )  # (N, 16, 13, 13) maintain the same spatial dimensions because of "same" padding

        self.conv4 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, padding_mode="same"
        )  # (N, 16, 13, 13) maintain the same spatial dimensions because of "same" padding
        self.pool2 = nn.MaxPool2d(
            kernel_size=3, stride=2
        )  # (N, 16, 6, 6), where (13 - 3) / 2 + 1 = 6

        self.fc1 = nn.Linear(16 * 6 * 6, 10)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, 1, 28, 28)  # (N, in_channels, H, W)

        # First conv block
        x = functional.relu(self.conv1(x))
        x = functional.relu(self.conv2(x))
        x = self.pool1(x)

        # Second conv block
        x = functional.relu(self.conv3(x))
        x = functional.relu(self.conv4(x))
        x = self.pool2(x)

        # Flatten and dense layers
        x = x.reshape(batch_size, -1)
        return functional.softmax(self.fc1(x))


class MnistOneVsRestBinaryClassifier(nn.Module):
    def __init__(self, output_logits=True):
        super().__init__()
        self.h1 = nn.Linear(784, 64)  # mnist image has shape 28*28=784
        self.h2 = nn.Linear(64, 64)
        self.h3 = nn.Linear(64, 1)
        self.output_logits = output_logits

    def forward(self, x):
        x = functional.relu(self.h1(x))
        x = functional.relu(self.h2(x))
        x = self.h3(x)
        if not self.output_logits:
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
        model = MnistOneVsRestBinaryClassifier(output_logits=True)

        trainer = Trainer(
            model=model,
            loss_fn=lambda pred, true: functional.hinge_loss(
                pred, true, reduction="mean"
            ),
            optimizer=optim.Adam(model.parameters, lr=1e-3),
            epochs=10,
            output_type="logits",
        )
        trainer.fit(X_train, y_binary.astype(int))
        one_vs_rest_models.append(model)

    logger.info("Training complete")
    # 10 models, each model predicts probability of digit 0-9
    # predictions_by_digit[i][j] is the probability that the ith test example is the jth digit
    predictions_by_digit = np.array(
        [model(X_test).data for model in one_vs_rest_models]
    ).T

    logger.info(
        f"Test Accuracy: {accuracy(predictions_by_digit.argmax(axis=2).squeeze(), y_test.astype(int))}"
    )
    logger.info(
        f"Test Precision: {precision(predictions_by_digit.argmax(axis=2).squeeze(), y_test.astype(int))}"
    )


def train_mnist_one_vs_rest_model(X_train, y_train, X_test, y_test):
    logger.info("=" * 50)
    logger.info("Starting to train One vs Rest MNIST model")
    logger.info("=" * 50)
    one_vs_rest_models = []
    for digit in range(10):
        logger.info(f"Training {digit=}")
        y_binary = (y_train == digit).astype(int)
        model = MnistOneVsRestBinaryClassifier(output_logits=False)
        trainer = Trainer(
            model=model,
            X=X_train,
            y=y_binary,
            loss_fn=functional.binary_cross_entropy,
            optimizer=optim.Adam(model.parameters, lr=1e-3),
            epochs=10,
            output_type="sigmoid",
        )
        trainer.fit(X_train, y_binary.astype(int))
        one_vs_rest_models.append(model)

    logger.info("Training complete")
    # 10 models, each model predicts probability of digit 0-9
    # predictions_by_digit[i][j] is the probability that the ith test example is the jth digit
    predictions_by_digit = np.array(
        [model(X_test).data for model in one_vs_rest_models]
    ).T

    logger.info(
        f"Test Accuracy: {accuracy(predictions_by_digit.argmax(axis=2).squeeze(), y_test.astype(int))}"
    )
    logger.info(
        f"Test Precision: {precision(predictions_by_digit.argmax(axis=2).squeeze(), y_test.astype(int))}"
    )


def train_mnist_multiclass_model(
    X_train, y_train, X_test, y_test, optimizer, model, loss_fn, epochs=50, msg=""
):
    logger.info("=" * 66)
    logger.info(f"Starting to train Multi-class MNIST model {msg}")
    logger.info("=" * 66)
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=epochs,
        batch_size=256,
        output_type="softmax",
    )
    trainer.fit(X_train, y_train.astype(int))

    # Evaluation mode
    model.eval()
    y_pred = model(X_test).data

    logger.info(f"Test Accuracy: {accuracy(y_pred.argmax(axis=1), y_test.astype(int))}")
    logger.info(
        f"Test Precision: {precision(y_pred.argmax(axis=1), y_test.astype(int))}"
    )


if __name__ == "__main__":
    logger.info("Fetching data for MNIST_784")
    X, y, _, __ = get_dataset(dataset_id=554, download_data=True).get_data(
        target="class", dataset_format="array"
    )
    X /= 255.0  # normalize to [0, 1] to speed up convergence

    logger.info(f"X shape: {X.shape}")
    logger.info(f"y shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    model = MnistConvolutionalClassifier()
    logger.info(f"Number of parameters: {model.num_parameters()}")
    train_mnist_multiclass_model(
        X_train,
        y_train,
        X_test,
        y_test,
        optimizer=optim.Adam(model.parameters, lr=1e-3),
        model=model,
        loss_fn=functional.sparse_cross_entropy,
        epochs=3,
        msg="Convolutional Neural Network (with batch norm, Adam optimizer)",
    )

    model = MnistMultiClassClassifier(batch_norm=False)
    train_mnist_multiclass_model(
        X_train,
        y_train,
        X_test,
        y_test,
        optimizer=optim.SGD(model.parameters, lr=1e-3),
        model=model,
        loss_fn=functional.sparse_cross_entropy,
        epochs=1000,
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
        loss_fn=functional.sparse_cross_entropy,
        epochs=1000,
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
        loss_fn=functional.sparse_cross_entropy,
        epochs=1000,
        msg="(with batch norm, Adam optimizer)",
    )

    train_mnist_one_vs_rest_model(X_train, y_train, X_test, y_test)

    train_mnist_with_hinge_loss(X_train, y_train, X_test, y_test)
