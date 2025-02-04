import logging

from autograd.tools.config_schema import GenericTrainingConfig

try:
    # drop-in replacement for numpy for GPU acceleration
    import cupy as np  # type: ignore

    _ = np.cuda.runtime.getDeviceCount()  # Check if a CUDA device is available
except Exception:
    import numpy as np
from openml.datasets import get_dataset

from autograd import functional, nn, optim
from autograd.tools.data import SimpleDataLoader, train_test_split
from autograd.tools.metrics import accuracy, precision
from autograd.tools.trainer import SimpleTrainer

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
        train_loader = SimpleDataLoader(
            X_train.copy(), y_train.copy(), batch_size, shuffle=True
        )
        test_loader = SimpleDataLoader(
            X_test.copy(), y_test.copy(), batch_size, shuffle=False
        )

        # Use a lambda with a default parameter to capture the current digit.
        train_loader.preprocess(lambda x, y, d=digit: preprocess_for_digit(x, y, d))
        test_loader.preprocess(lambda x, y, d=digit: preprocess_for_digit(x, y, d))

        # Build a training configuration for the trainer.
        config = GenericTrainingConfig(
            total_epochs=epochs,
            steps_per_epoch=10,
            checkpoint_freq=10,
            model_kwargs={"output_logits": True},
            optimizer_kwargs={"lr": 1e-3},
        )

        # Call SimpleTrainer using the same pattern as the multiclass trainer.
        trainer = SimpleTrainer(
            model_cls=MnistOneVsRestBinaryClassifier,
            optimizer_cls=optim.Adam,
            loss_fn=lambda p, t: functional.hinge_loss(p, t, reduction="mean"),
            output_type="logits",
            config=config,
        )
        trainer.fit(train_loader)
        one_vs_rest_models.append(trainer.model)

    logger.info("Training complete! Now evaluating on the original test set...")
    predictions_by_digit = np.array(
        [model(X_test).data for model in one_vs_rest_models]
    )
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
        # Convert y into {0, 1}
        y_bin = (y == digit).astype(int)
        return x, y_bin

    for digit in range(10):
        logger.info(f"Training digit={digit}")
        # Create fresh data loaders for each digit
        train_loader = SimpleDataLoader(
            X_train.copy(), y_train.copy(), batch_size, shuffle=True
        )
        test_loader = SimpleDataLoader(
            X_test.copy(), y_test.copy(), batch_size, shuffle=False
        )

        # Freeze the current digit in the lambda via a default parameter.
        train_loader.preprocess(lambda x, y, d=digit: preprocess_for_digit(x, y, d))
        test_loader.preprocess(lambda x, y, d=digit: preprocess_for_digit(x, y, d))

        # Build the training configuration.
        config = GenericTrainingConfig(
            total_epochs=epochs,
            checkpoint_freq=10,
            model_kwargs={"output_logits": False},
            optimizer_kwargs={"lr": 1e-3},
        )

        # Create a trainer following the same pattern as for the multiclass model.
        trainer = SimpleTrainer(
            model_cls=MnistOneVsRestBinaryClassifier,
            optimizer_cls=optim.Adam,
            loss_fn=functional.binary_cross_entropy,
            output_type="sigmoid",
            config=config,
        )
        trainer.fit(train_loader)
        one_vs_rest_models.append(trainer.model)

    logger.info("Training complete! Now evaluating on the original test set...")
    predictions_by_digit = np.array(
        [model(X_test).data for model in one_vs_rest_models]
    )
    predictions_by_digit = np.transpose(predictions_by_digit, (1, 0, 2)).squeeze(-1)
    pred_digits = predictions_by_digit.argmax(axis=1)
    acc_val = accuracy(pred_digits, y_test.astype(int))
    prec_val = precision(pred_digits, y_test.astype(int))
    logger.info(f"Final Test Accuracy: {acc_val:.4f}")
    logger.info(f"Final Test Precision: {prec_val:.4f}")


def train_mnist_multiclass_model(
    train_data_loader,
    test_data_loader,
    optimizer_cls,
    model_cls,
    loss_fn,
    config,
    msg="",
):
    logger.info("=" * 66)
    logger.info(f"Starting Multi-class MNIST model {msg}")
    logger.info("=" * 66)
    trainer = SimpleTrainer(
        model_cls=model_cls,
        optimizer_cls=optimizer_cls,
        loss_fn=loss_fn,
        output_type="logits",
        config=config,
        sample_predictions=True,
    )
    trainer.fit(train_data_loader, test_data_loader)


if __name__ == "__main__":
    logger.info("Fetching data for MNIST_784")
    X, y, _, __ = get_dataset(dataset_id=554, download_data=True).get_data(
        target="class", dataset_format="array"
    )
    X /= 255.0  # Normalize to [0, 1] to speed up convergence

    # Use a subset of the data for faster training
    X = X[:3000]
    y = y[:3000]
    logger.info(f"X shape: {X.shape}, y shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # Create data loaders with a consistent batch size.
    train_data_loader = SimpleDataLoader(X_train, y_train, batch_size=512, shuffle=True)
    val_data_loader = SimpleDataLoader(X_test, y_test, batch_size=512, shuffle=False)

    # Train several multi-class models.
    train_mnist_multiclass_model(
        train_data_loader,
        val_data_loader,
        optimizer_cls=optim.Adam,
        model_cls=MnistResNet,
        loss_fn=functional.cross_entropy,
        config=GenericTrainingConfig(
            total_epochs=10,
            checkpoint_freq=10,
            model_kwargs={},
            optimizer_kwargs={
                "lr": 1e-3,
                "max_grad_norm": 1.0,
            },
        ),
        msg="ResNet-based",
    )

    train_mnist_multiclass_model(
        train_data_loader,
        val_data_loader,
        optimizer_cls=optim.SGD,
        model_cls=MnistMultiClassClassifier,
        loss_fn=functional.cross_entropy,
        config=GenericTrainingConfig(
            total_epochs=10,
            checkpoint_freq=10,
            model_kwargs={
                "batch_norm": False,
            },
            optimizer_kwargs={
                "lr": 1e-3,
            },
        ),
        msg="(MLP, no batch norm, SGD)",
    )

    train_mnist_multiclass_model(
        train_data_loader,
        val_data_loader,
        optimizer_cls=optim.SGD,
        model_cls=MnistMultiClassClassifier,
        loss_fn=functional.cross_entropy,
        config=GenericTrainingConfig(
            total_epochs=10,
            steps_per_epoch=10,
            checkpoint_freq=10,
            model_kwargs={
                "batch_norm": True,
            },
            optimizer_kwargs={
                "lr": 1e-3,
            },
        ),
        msg="(MLP, batch norm, SGD)",
    )

    train_mnist_multiclass_model(
        train_data_loader,
        val_data_loader,
        optimizer_cls=optim.Adam,
        model_cls=MnistMultiClassClassifier,
        loss_fn=functional.cross_entropy,
        config=GenericTrainingConfig(
            total_epochs=10,
            checkpoint_freq=10,
            model_kwargs={
                "batch_norm": True,
            },
            optimizer_kwargs={
                "lr": 1e-3,
            },
        ),
        msg="(MLP, batch norm, Adam)",
    )

    # Now train the one-vs-rest models.
    train_mnist_one_vs_rest_model(
        X_train, y_train, X_test, y_test, batch_size=256, epochs=10
    )
    train_mnist_with_hinge_loss(
        X_train, y_train, X_test, y_test, batch_size=256, epochs=10
    )
