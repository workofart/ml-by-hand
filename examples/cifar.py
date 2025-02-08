import logging

from openml.datasets import get_dataset

from autograd import functional, nn, optim
from autograd.tools.config_schema import GenericTrainingConfig
from autograd.tools.data import SimpleDataLoader, train_test_split
from autograd.tools.trainer import SimpleTrainer

logger = logging.getLogger(__name__)
# np.random.seed(1337) # need to comment out for dropout to work


# Best so far: 59% accuracy on CIFAR-10
class CifarMulticlassClassifier(nn.Module):
    """
    A dense neural network for classifying CIFAR-10 images using fully connected layers.

    This model processes flattened CIFAR-10 images (of size 32x32x3=3072) through a series
    of linear layers with intermediate batch normalization and dropout. The final linear layer
    outputs logits for each of the 10 classes.
    """

    def __init__(self, num_classes: int):
        """
        Initialize the CifarMulticlassClassifier.

        Args:
            num_classes (int): The number of target classes (e.g., 10 for CIFAR-10).
        """
        super().__init__()
        self.h1 = nn.Linear(
            3072, 1024
        )  # cifar-10 image has shape 32 x 32 x 3 (color channels)
        self.h2 = nn.Linear(1024, 512)
        self.h3 = nn.Linear(512, 256)
        self.h4 = nn.Linear(256, num_classes)
        self.bn1 = nn.BatchNorm(1024)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        """
        Perform the forward pass for the dense CIFAR-10 classifier.

        The input is expected as a flattened image vector. The forward method applies a linear
        transformation followed by batch normalization, ReLU activation, and dropout at each layer,
        and finally outputs raw logits.

        Args:
            x (np.ndarray): A batch of flattened CIFAR-10 images of shape (batch_size, 3072).

        Returns:
            np.ndarray: Logits of shape (batch_size, num_classes).
        """
        x = functional.relu(self.bn1(self.h1(x)))
        x = self.dropout(x)
        x = functional.relu(self.h2(x))
        x = self.dropout(x)
        x = functional.relu(self.h3(x))
        x = self.dropout(x)
        return self.h4(x)


class CifarResNet(nn.Module):
    """
    A residual network for classifying CIFAR-10 images.

    This model uses residual blocks to enable improved gradient flow during training.
    It expects CIFAR-10 images to be reshaped to (N, 3, 32, 32) and then applies two residual
    blocks followed by a linear layer to produce the final logits.
    """

    def __init__(self, num_classes: int):
        """
        Initialize the CifarResNet model.

        Args:
            num_classes (int): The number of target classes (e.g., 10 for CIFAR-10).
        """
        super().__init__()
        self.res_block1 = nn.ResidualBlock(3, 16)
        self.res_block2 = nn.ResidualBlock(16, 16)
        self.fc1 = nn.Linear(
            16 * 32 * 32, num_classes
        )  # 32*32 is the output size of the last maxpool layer

    def forward(self, x):
        """
        Perform the forward pass of the ResNet-based CIFAR-10 model.

        The method reshapes the input to a 4D tensor and passes it through two residual blocks.
        Finally, the output is flattened and passed through a linear layer.

        Args:
            x (np.ndarray): A batch of flattened CIFAR-10 images of shape (batch_size, 3072).

        Returns:
            np.ndarray: Logits of shape (batch_size, num_classes).
        """
        batch_size = x.shape[0]
        x = x.reshape(batch_size, 3, 32, 32)  # (N, in_channels, H, W)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = x.reshape(batch_size, -1)
        return self.fc1(x)


class CifarConvolutionalClassifier(nn.Module):
    """
    A convolutional neural network for classifying CIFAR-10 images.

    This model applies a series of convolutional layers with "same" padding followed by max pooling,
    then flattens the output and applies a linear layer to produce the class logits.
    """

    def __init__(self, num_classes: int):
        """
        Initialize the CifarConvolutionalClassifier.

        Args:
            num_classes (int): The number of target classes (e.g., 10 for CIFAR-10).
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=8, kernel_size=3, padding_mode="same"
        )  # (N, 8, 32, 32) maintain the same spatial dimensions because of "same" padding
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=8, kernel_size=3, padding_mode="same"
        )  # (N, 8, 32, 32) maintain the same spatial dimensions because of "same" padding
        self.pool1 = nn.MaxPool2d(
            kernel_size=3, stride=2
        )  # (N, 8, 16, 16), where (8 - 3) / 2 + 1 = 3

        self.conv3 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, padding_mode="same"
        )  # (N, 16, 16, 16) maintain the same spatial dimensions because of "same" padding

        self.conv4 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, padding_mode="same"
        )  # (N, 16, 16, 16) maintain the same spatial dimensions because of "same" padding
        self.pool2 = nn.MaxPool2d(
            kernel_size=3, stride=2
        )  # (N, 16, 7, 7), where (16 - 3) / 2 + 1 = 7

        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        """
        Perform the forward pass of the convolutional CIFAR-10 classifier.

        The input is reshaped to a 4D tensor, passed through two convolutional blocks
        (each consisting of convolution, ReLU activation, and pooling), flattened, and finally
        mapped to class logits via a linear layer.

        Args:
            x (np.ndarray): A batch of flattened CIFAR-10 images of shape (batch_size, 3072).

        Returns:
            np.ndarray: Logits of shape (batch_size, num_classes).
        """
        batch_size = x.shape[0]
        # cifar-10 image has shape 32 x 32 x 3 (color channels)
        x = x.reshape(batch_size, 3, 32, 32)  # (N, in_channels, H, W)

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


def train_cifar_multiclass_model(
    train_data_loader,
    test_data_loader,
    optimizer_cls,
    model_cls,
    loss_fn,
    config,
    msg="",
):
    """
    Train a CIFAR multiclass classifier model using a given training configuration.

    This function creates a SimpleTrainer with the provided model class, optimizer class,
    loss function, and configuration. It then trains the model using the provided training
    and testing data loaders.

    Args:
        train_data_loader (SimpleDataLoader): Data loader for training data.
        test_data_loader (SimpleDataLoader): Data loader for testing data.
        optimizer_cls: Optimizer class to use (e.g., optim.Adam, optim.SGD).
        model_cls: Neural network model class to instantiate.
        loss_fn: Loss function to optimize.
        config (GenericTrainingConfig): Configuration for training (epochs, learning rate, etc.).
        msg (str): Additional message for logging purposes.
    """
    logger.info("=" * 66)
    logger.info(f"Starting CIFAR multiclass model {msg}")
    logger.info("=" * 66)
    trainer = SimpleTrainer(
        model_cls=model_cls,
        optimizer_cls=optimizer_cls,
        loss_fn=loss_fn,
        output_type="logits",
        config=config,
    )
    trainer.fit(train_data_loader, test_data_loader)


if __name__ == "__main__":
    """
    Main script for training CIFAR-10 and CIFAR-100 classifiers.

    The pipeline is as follows:
      1) Fetch the CIFAR-10 dataset using OpenML, normalize the pixel values to [0, 1],
         and subsample 5000 examples for faster training.
      2) Split the data into training and testing sets using train_test_split.
      3) Create SimpleDataLoader objects for both training and testing sets.
      4) Train three types of models on CIFAR-10:
           - A ResNet-based model using residual blocks.
           - A convolutional classifier.
           - A dense (fully connected) classifier.
      5) Fetch the CIFAR-100 dataset similarly, subsample it, and create corresponding data loaders.
      6) Train three types of models on CIFAR-100:
           - A ResNet-based model.
           - A convolutional classifier.
           - A dense classifier.
      7) Log training progress, checkpoints, and evaluation metrics (accuracy, precision).

    Note:
      - The training configurations (e.g., number of epochs, checkpoint frequency, learning rate,
        and gradient clipping) are specified via GenericTrainingConfig.
      - This script is designed to be run as the main module.
    """
    # ------------------------------
    # CIFAR-10
    # ------------------------------
    logger.info("Fetching data for CIFAR-10")
    X, y, _, __ = get_dataset(dataset_id=40927, download_data=True).get_data(
        target="class", dataset_format="array"
    )
    X = X / 255.0  # Normalize to [0, 1] to speed up convergence

    # To speed up the training on local machine
    X = X[:5000]
    y = y[:5000]
    logger.info(f"{X.shape=}, {y.shape=}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    train_data_loader = SimpleDataLoader(X_train, y_train, batch_size=256, shuffle=True)
    test_data_loader = SimpleDataLoader(X_test, y_test, batch_size=256, shuffle=False)

    # Train ResNet CIFAR-10 model
    logger.info("Training ResNet CIFAR-10 model")
    config = GenericTrainingConfig(
        total_epochs=30,
        checkpoint_freq=100,
        model_kwargs={"num_classes": 10},
        optimizer_kwargs={"lr": 3e-3, "max_grad_norm": 1.0},
    )
    train_cifar_multiclass_model(
        train_data_loader,
        test_data_loader,
        optimizer_cls=optim.Adam,
        model_cls=CifarResNet,
        loss_fn=functional.cross_entropy,
        config=config,
        msg="ResNet CIFAR-10",
    )

    # Train Convolutional CIFAR-10 model
    logger.info("Training Convolutional CIFAR-10 model")
    config = GenericTrainingConfig(
        total_epochs=100,
        checkpoint_freq=100,
        model_kwargs={"num_classes": 10},
        optimizer_kwargs={"lr": 1e-3, "max_grad_norm": 1.0},
    )
    train_cifar_multiclass_model(
        train_data_loader,
        test_data_loader,
        optimizer_cls=optim.Adam,
        model_cls=CifarConvolutionalClassifier,
        loss_fn=functional.cross_entropy,
        config=config,
        msg="Convolutional CIFAR-10",
    )

    # Train Dense CIFAR-10 model
    logger.info("Training Dense CIFAR-10 model")
    config = GenericTrainingConfig(
        total_epochs=100,
        checkpoint_freq=100,
        model_kwargs={"num_classes": 10},
        optimizer_kwargs={"lr": 1e-3, "max_grad_norm": 1.0},
    )
    train_cifar_multiclass_model(
        train_data_loader,
        test_data_loader,
        optimizer_cls=optim.Adam,
        model_cls=CifarMulticlassClassifier,
        loss_fn=functional.cross_entropy,
        config=config,
        msg="Dense CIFAR-10",
    )

    # ------------------------------
    # CIFAR-100
    # ------------------------------
    logger.info("Fetching data for CIFAR-100")
    X, y, _, __ = get_dataset(dataset_id=41983, download_data=True).get_data(
        target="class", dataset_format="array"
    )
    X = X / 255.0  # Normalize to [0, 1]
    X = X[:5000]
    y = y[:5000]
    logger.info(f"{X.shape=}, {y.shape=}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    train_data_loader = SimpleDataLoader(X_train, y_train, batch_size=256, shuffle=True)
    test_data_loader = SimpleDataLoader(X_test, y_test, batch_size=256, shuffle=False)

    # Train ResNet CIFAR-100 model
    logger.info("Training ResNet CIFAR-100 model")
    config = GenericTrainingConfig(
        total_epochs=100,
        checkpoint_freq=100,
        model_kwargs={"num_classes": 100},
        optimizer_kwargs={"lr": 1e-3, "max_grad_norm": 1.0},
    )
    train_cifar_multiclass_model(
        train_data_loader,
        test_data_loader,
        optimizer_cls=optim.Adam,
        model_cls=CifarResNet,
        loss_fn=functional.cross_entropy,
        config=config,
        msg="ResNet CIFAR-100",
    )

    # Train Convolutional CIFAR-100 model
    logger.info("Training Convolutional CIFAR-100 model")
    config = GenericTrainingConfig(
        total_epochs=100,
        checkpoint_freq=100,
        model_kwargs={"num_classes": 100},
        optimizer_kwargs={"lr": 1e-3, "max_grad_norm": 1.0},
    )
    train_cifar_multiclass_model(
        train_data_loader,
        test_data_loader,
        optimizer_cls=optim.Adam,
        model_cls=CifarConvolutionalClassifier,
        loss_fn=functional.cross_entropy,
        config=config,
        msg="Convolutional CIFAR-100",
    )

    # Train Dense CIFAR-100 model
    logger.info("Training Dense CIFAR-100 model")
    config = GenericTrainingConfig(
        total_epochs=100,
        checkpoint_freq=100,
        model_kwargs={"num_classes": 100},
        optimizer_kwargs={"lr": 1e-3, "max_grad_norm": 1.0},
    )
    train_cifar_multiclass_model(
        train_data_loader,
        test_data_loader,
        optimizer_cls=optim.Adam,
        model_cls=CifarMulticlassClassifier,
        loss_fn=functional.cross_entropy,
        config=config,
        msg="Dense CIFAR-100",
    )
