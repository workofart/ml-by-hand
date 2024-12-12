import logging
import time
from autograd import nn, utils
import numpy as np

logger = logging.getLogger(__name__)


def train_test_split(X, y, test_size=0.2, random_state=None):
    num_samples = len(X)
    num_test = int(num_samples * test_size)

    # Create a random permutation of indices
    indices = np.random.permutation(num_samples)

    # Use array indexing to split the data
    X_train, X_test = X[indices[num_test:]], X[indices[:num_test]]
    y_train, y_test = y[indices[num_test:]], y[indices[:num_test]]

    return X_train, X_test, y_train, y_test


def accuracy(y_true, y_pred):
    """
    Accuracy = (True Positives + True Negatives) / Total Predictions
    """
    assert len(y_true) == len(y_pred)
    return np.sum((y_true == y_pred).astype(int)) / len(y_true)


def precision(y_true, y_pred):
    """
    Precision = True Positives / (True Positives + False Positives)
    """
    assert len(y_true) == len(y_pred)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positives = np.sum(y_pred == 1)
    return true_positives / predicted_positives if predicted_positives != 0 else 0.0


def train(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    loss_fn,
    optimizer,
    epochs=100,
    batch_size=256,
    shuffle_each_epoch=False,
):
    model.train()
    n_samples = len(X)
    n_batches = (n_samples + batch_size - 1) // batch_size  # ceil division

    for epoch in range(epochs):
        total_loss = 0.0
        grad_norms = []
        start_time = time.time()

        if shuffle_each_epoch:
            # Create random permutation for shuffling
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
        else:
            X_shuffled = X
            y_shuffled = y

        # Process data in batches
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)

            # Get batch data
            batch_X = X_shuffled[start_idx:end_idx]
            batch_y = y_shuffled[start_idx:end_idx]

            # Zero gradients for each batch
            optimizer.zero_grad()

            # Forward pass
            y_pred = model(batch_X)
            loss = loss_fn(y_pred, batch_y)
            total_loss += loss.data * (end_idx - start_idx)  # weight by batch size

            # Backward pass and optimize
            loss.backward()

            # Monitor gradient norms - handle nested parameters
            def get_grad_norm(params):
                grad_norm_sq = 0.0
                for param in params:
                    if isinstance(param, dict):
                        grad_norm_sq += get_grad_norm(param.values())
                    elif hasattr(param, "grad") and param.grad is not None:
                        grad_norm_sq += np.sum(param.grad.data**2)
                return grad_norm_sq

            grad_norm = np.sqrt(get_grad_norm(model.parameters.values()))
            grad_norms.append(grad_norm)

            optimizer.step()

        # Calculate average loss for the epoch
        avg_loss = total_loss / n_samples
        avg_grad_norm = np.mean(grad_norms)
        epoch_time = time.time() - start_time
        epochs_per_second = n_batches / epoch_time
        logger.info(
            f"\nEpoch: {epoch}"
            f"\n\tLoss: {avg_loss:.4f}"
            f"\n\tGrad Norm: {avg_grad_norm:.4f}"
            f"\n\tEpochs/sec: {epochs_per_second:.2f}"
            f"\n\tAccuracy: {utils.accuracy(y_pred.data.argmax(axis=1), batch_y.astype(int)):.2f}"
        )
