import logging
import time
import numpy as np
from autograd import functional
from autograd.tools.metrics import accuracy  # lazy import to avoid circular deps

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        epochs=100,
        batch_size=256,
        shuffle_each_epoch=False,
        output_type=None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle_each_epoch = shuffle_each_epoch
        self.output_type = output_type

    def _post_process_predictions(self, y_pred):
        """
        Post-processing predictions based on output type.
        """
        if self.output_type == "logits":
            # Convert logits to binary predictions
            y_prob = functional.sigmoid(y_pred).data
            return (y_prob >= 0.5).astype(int).squeeze()
        elif self.output_type == "sigmoid":
            # Already probabilities, just threshold
            return (y_pred >= 0.5).astype(int).squeeze()
        elif self.output_type == "softmax":
            # Multi-class
            return y_pred.data.argmax(axis=1).squeeze()
        else:
            # No transformation
            return y_pred

    def fit(self, X, y):
        n_samples = len(X)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        self.model.train()

        for epoch in range(self.epochs):
            total_loss = 0.0
            start_time = time.time()

            # Shuffle dataset if required
            if self.shuffle_each_epoch:
                indices = np.random.permutation(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
            else:
                X_shuffled = X
                y_shuffled = y

            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, n_samples)
                batch_X = X_shuffled[start_idx:end_idx]
                batch_y = y_shuffled[start_idx:end_idx]

                # Reset gradients
                self.optimizer.zero_grad()

                # Forward pass
                y_pred = self.model(batch_X)
                loss = self.loss_fn(y_pred, batch_y)
                total_loss += loss.detach().data * (end_idx - start_idx)

                # Backward pass
                loss.backward()
                self.optimizer.step()

            # Periodically log information
            if epoch % max(1, (self.epochs // 10)) == 0:
                self.log_epoch(
                    X_shuffled, y_shuffled, total_loss, n_samples, start_time, epoch
                )

    def log_epoch(
        self, X_shuffled, y_shuffled, total_loss, n_samples, start_time, epoch
    ):
        self.model.eval()
        with np.errstate(divide="ignore", invalid="ignore"):
            y_pred = self.model(X_shuffled)

            # If the data is in {-1,1} but we want {0,1} for accuracy
            if y_shuffled.min() == -1 and y_shuffled.max() == 1:
                y_shuffled = (y_shuffled + 1) // 2

            avg_loss = total_loss / n_samples
            epoch_time = time.time() - start_time
            epochs_per_second = (n_samples / self.batch_size) / epoch_time

            # Post-process predictions
            y_pred_processed = self._post_process_predictions(y_pred)

            logger.info(
                f"\nEpoch: {epoch}"
                f"\n\tLoss: {avg_loss:.4f}"
                f"\n\tEpochs/sec: {epochs_per_second:.2f}"
                f"\n\tAccuracy: {accuracy(y_pred_processed, y_shuffled.astype(int)):.2f}"
            )
        self.model.train()
