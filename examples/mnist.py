import logging
import numpy as np
from autograd import nn, optim, functional
from openml.datasets import get_dataset
from autograd.tools.data import train_test_split, SimpleDataLoader
from autograd.tools.trainer import SimpleTrainer
from autograd.tools.metrics import accuracy, precision

logger = logging.getLogger(__name__)
np.random.seed(1337)

# --- Model Definitions ---


class MnistResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.res_block1 = nn.ResidualBlock(1, 16)
        self.res_block2 = nn.ResidualBlock(16, 16)
        self.fc1 = nn.Linear(
            16 * 28 * 28, 10
        )  # 28*28 is the output size of the last maxpool layer

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, 1, 28, 28)  # (N, in_channels, H, W)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = x.reshape(batch_size, -1)
        return self.fc1(x)


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

        x = functional.relu(self.h2(x))
        return self.h3(x)


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
        return self.fc1(x)


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
        return x if self.output_logits else functional.sigmoid(x)


# --- Training Functions ---


def train_mnist_with_hinge_loss(
    X_train, y_train, X_test, y_test, batch_size=32, epochs=10
):
    """
    Trains 10 one-vs-rest binary classifiers using hinge loss:
      For each digit d in [0..9], label the data as +1 if y=d, else -1.
      Then train a separate MnistOneVsRestBinaryClassifier for each digit.

    Args:
      X_train, y_train: Full training data/labels (numpy arrays).
      X_test, y_test: Full test data/labels (numpy arrays).
      batch_size (int): Batch size for the SimpleDataLoader.
      epochs (int): Number of epochs for each binary classifier.
    """
    logger.info("=" * 50)
    logger.info("Starting to train One vs Rest MNIST model (Hinge Loss)")
    logger.info("=" * 50)

    one_vs_rest_models = []

    def preprocess_for_digit(x, y, digit):
        # Convert y into {+1, -1}
        y_bin = 2 * (y == digit).astype(int) - 1
        return x, y_bin

    for digit in range(10):
        logger.info(f"Training digit={digit}")
        train_data_loader = SimpleDataLoader(
            X_train.copy(), y_train.copy(), batch_size, shuffle=True
        )
        test_data_loader = SimpleDataLoader(
            X_test.copy(), y_test.copy(), batch_size, shuffle=False
        )

        train_data_loader.preprocess(lambda x, y: preprocess_for_digit(x, y, digit))
        test_data_loader.preprocess(lambda x, y: preprocess_for_digit(x, y, digit))

        model = MnistOneVsRestBinaryClassifier(output_logits=True)
        trainer = SimpleTrainer(
            model=model,
            loss_fn=lambda p, t: functional.hinge_loss(p, t, reduction="mean"),
            optimizer=optim.Adam(model.parameters, lr=1e-3),
            epochs=epochs,
            output_type="logits",
        )
        trainer.fit(train_data_loader, test_data_loader)
        one_vs_rest_models.append(model)

    logger.info("Training complete! Now evaluating on the original test set...")
    predictions_by_digit = np.array([m(X_test).data for m in one_vs_rest_models])
    predictions_by_digit = np.transpose(predictions_by_digit, (1, 0, 2)).squeeze(-1)
    pred_digits = predictions_by_digit.argmax(axis=1)
    acc_val = accuracy(pred_digits, y_test.astype(int))
    prec_val = precision(pred_digits, y_test.astype(int))

    logger.info(f"Final Test Accuracy: {acc_val:.4f}")
    logger.info(f"Final Test Precision: {prec_val:.4f}")


def train_mnist_one_vs_rest_model(
    X_train, y_train, X_test, y_test, batch_size=32, epochs=10
):
    """
    Trains 10 one-vs-rest binary classifiers using binary cross-entropy:
      For each digit d in [0..9], label the data as 1 if y=d, else 0.
    """
    logger.info("=" * 50)
    logger.info("Starting to train One vs Rest MNIST model (Binary Cross Entropy)")
    logger.info("=" * 50)

    one_vs_rest_models = []

    def preprocess_for_digit(x, y, digit):
        """
        Convert y from {0,1,2,...,9} into {0,1},
        where 1 means 'this sample is digit == d', else 0.
        """
        y_bin = (y == digit).astype(int)
        return x, y_bin

    for digit in range(10):
        logger.info(f"Training digit={digit}")
        train_data_loader = SimpleDataLoader(
            X_train.copy(), y_train.copy(), batch_size, shuffle=True
        )
        test_data_loader = SimpleDataLoader(
            X_test.copy(), y_test.copy(), batch_size, shuffle=False
        )

        train_data_loader.preprocess(lambda x, y: preprocess_for_digit(x, y, digit))
        test_data_loader.preprocess(lambda x, y: preprocess_for_digit(x, y, digit))

        model = MnistOneVsRestBinaryClassifier(output_logits=False)
        trainer = SimpleTrainer(
            model=model,
            loss_fn=functional.binary_cross_entropy,
            optimizer=optim.Adam(model.parameters, lr=1e-3),
            epochs=epochs,
            output_type="sigmoid",
        )
        trainer.fit(train_data_loader, test_data_loader)
        one_vs_rest_models.append(model)

    logger.info("Training complete! Now evaluating on the original test set...")
    predictions_by_digit = np.array([m(X_test).data for m in one_vs_rest_models])
    predictions_by_digit = np.transpose(predictions_by_digit, (1, 0, 2)).squeeze(-1)
    pred_digits = predictions_by_digit.argmax(axis=1)

    acc_val = accuracy(pred_digits, y_test.astype(int))
    prec_val = precision(pred_digits, y_test.astype(int))
    logger.info(f"Final Test Accuracy: {acc_val:.4f}")
    logger.info(f"Final Test Precision: {prec_val:.4f}")


def train_mnist_multiclass_model(
    train_data_loader, test_data_loader, optimizer, model, loss_fn, epochs=50, msg=""
):
    logger.info("=" * 66)
    logger.info(f"Starting Multi-class MNIST model {msg}")
    logger.info("=" * 66)
    trainer = SimpleTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=epochs,
        output_type="logits",
    )
    trainer.fit(train_data_loader, test_data_loader)

    # ---- Make predictions on a small sample from test_data_loader ----
    x_sample, y_sample = next(iter(test_data_loader))  # get first batch
    model.eval()
    y_pred = model(x_sample)
    pred_labels = np.argmax(y_pred.data, axis=1)

    accuracy_val = accuracy(pred_labels, y_sample)
    logger.info(f"Accuracy on Test Batch: {accuracy_val:.4f}")

    logger.info("Sample Predictions on Test Batch:")
    for i in range(min(5, len(pred_labels))):
        logger.info(f"  Predicted: {pred_labels[i]}, Actual: {y_sample[i]}")


if __name__ == "__main__":
    logger.info("Fetching data for MNIST_784")
    X, y, _, __ = get_dataset(dataset_id=554, download_data=True).get_data(
        target="class", dataset_format="array"
    )
    X /= 255.0  # normalize to [0, 1] to speed up convergence

    X = X[:3000]
    y = y[:3000]
    logger.info(f"X shape: {X.shape}, y shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    train_data_loader = SimpleDataLoader(X_train, y_train, batch_size=256, shuffle=True)
    val_data_loader = SimpleDataLoader(X_test, y_test, batch_size=256, shuffle=False)

    model = MnistResNet()
    logger.info(f"Number of parameters: {model.num_parameters()}")
    train_mnist_multiclass_model(
        train_data_loader,
        val_data_loader,
        optim.Adam(model.parameters, lr=1e-3),
        model,
        functional.cross_entropy,
        epochs=10,
        msg="ResNet-based",
    )

    model = MnistMultiClassClassifier(batch_norm=False)
    train_mnist_multiclass_model(
        train_data_loader,
        val_data_loader,
        optim.SGD(model.parameters, lr=1e-3),
        model,
        functional.cross_entropy,
        epochs=100,
        msg="(MLP, no batch norm, SGD)",
    )

    model = MnistMultiClassClassifier(batch_norm=True)
    train_mnist_multiclass_model(
        train_data_loader,
        val_data_loader,
        optim.SGD(model.parameters, lr=1e-3),
        model,
        functional.cross_entropy,
        epochs=100,
        msg="(MLP, batch norm, SGD)",
    )

    model = MnistMultiClassClassifier(batch_norm=True)
    train_mnist_multiclass_model(
        train_data_loader,
        val_data_loader,
        optim.Adam(model.parameters, lr=1e-3),
        model,
        functional.cross_entropy,
        epochs=100,
        msg="(MLP, batch norm, Adam)",
    )

    train_mnist_one_vs_rest_model(X_train, y_train, X_test, y_test)

    train_mnist_with_hinge_loss(X_train, y_train, X_test, y_test)
