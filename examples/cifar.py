from autograd import nn, functional, optim
from autograd import utils
from openml.datasets import get_dataset
import logging

logger = logging.getLogger(__name__)
# np.random.seed(1337) # need to comment out for dropout to work


# Best so far: 59% accuracy on CIFAR-10
class CifarMulticlassClassifier(nn.Module):
    def __init__(self, num_classes: int):
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
        x = functional.relu(self.bn1(self.h1(x)))
        x = self.dropout(x)
        x = functional.relu(self.h2(x))
        x = self.dropout(x)
        x = functional.relu(self.h3(x))
        x = self.dropout(x)
        return functional.softmax(self.h4(x))


def train_cifar_multiclass_model(model: nn.Module, X_train, y_train, X_test, y_test):
    logger.info("=" * 66)
    logger.info("Starting to train Multi-class CIFAR model")
    logger.info("=" * 66)

    utils.train(
        model=model,
        X=X_train,
        y=y_train.astype(int),
        loss_fn=functional.sparse_cross_entropy,
        optimizer=optim.Adam(model.parameters, lr=1e-3),
        epochs=100,
        batch_size=512,
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
    logger.info("Fetching data for CIFAR-10")
    X, y, _, __ = get_dataset(dataset_id=40927, download_data=True).get_data(
        target="class", dataset_format="array"
    )
    X = X / 255.0  # normalize to [0, 1] to speed up convergence
    X = (X - X.mean(axis=0)) / X.std(axis=0)  # center the data

    X_train, X_test, y_train, y_test = utils.train_test_split(X, y, test_size=0.1)
    cifar10_model = CifarMulticlassClassifier(num_classes=10)
    train_cifar_multiclass_model(cifar10_model, X_train, y_train, X_test, y_test)

    logger.info("Fetching data for CIFAR-100")
    X, y, _, __ = get_dataset(dataset_id=41983, download_data=True).get_data(
        target="class", dataset_format="array"
    )
    X = X / 255.0  # normalize to [0, 1] to speed up convergence
    X = (X - X.mean(axis=0)) / X.std(axis=0)  # center the data

    X_train, X_test, y_train, y_test = utils.train_test_split(X, y, test_size=0.1)
    cifar100_model = CifarMulticlassClassifier(num_classes=100)
    train_cifar_multiclass_model(cifar100_model, X_train, y_train, X_test, y_test)
