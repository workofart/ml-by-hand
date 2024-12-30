import logging
import time
import numpy as np
from autograd.tools.metrics import accuracy, mean_squared_error
from autograd.tensor import Tensor

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

        # Decide problem type based on output_type
        if output_type in ["logits", "sigmoid", "softmax"]:
            self.problem_type = "classification"
        else:
            self.problem_type = "regression"

    def fit(self, X, y):
        """
        X, y can each be:
          - (Batch Size, Sequence length) for a sequence classification (with to_one_hot inside the model)
          - (Batch Size, Sequence length, # of Classes) for one-hot sequences
          - (Batch Size, Dimension) for a single classification or regression
          etc.
        """
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

    def evaluate(self, X, y, sample_predictions=True, num_samples_to_show=4):
        """
        Evaluate the model on dataset (X, y) without training.
        Calls log_epoch to display the result.
        By default, prints a few sample predictions.
        """
        self.model.eval()
        n_samples = len(X)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size

        total_loss = 0.0
        for batch_idx in range(n_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, n_samples)
            batch_X = X[start_idx:end_idx]
            batch_y = y[start_idx:end_idx]

            # Forward pass only
            y_pred = self.model(batch_X)
            loss = self.loss_fn(y_pred, batch_y)
            total_loss += loss.detach().data * (end_idx - start_idx)

        # Just reuse log_epoch to display metrics & sample predictions
        # We pass epoch="Evaluation" so it's clear in logs that it’s an eval run
        start_time = time.time()  # so we can pass something to log_epoch
        self.log_epoch(
            X,
            y,
            total_loss,
            n_samples,
            start_time,
            epoch="Evaluation",
            sample_predictions=sample_predictions,
            num_samples_to_show=num_samples_to_show,
        )

    def log_epoch(
        self,
        X_shuffled,
        y_shuffled,
        total_loss,
        n_samples,
        start_time,
        epoch,
        sample_predictions=False,
        num_samples_to_show=4,
    ):
        """
        Logs the loss and metrics. Optionally prints out a few sample predictions.
        """
        self.model.eval()
        with np.errstate(divide="ignore", invalid="ignore"):
            # Forward pass on entire dataset X_shuffled
            y_pred = self.model(X_shuffled)

            avg_loss = total_loss / n_samples
            epoch_time = time.time() - start_time
            epochs_per_second = (n_samples / self.batch_size) / epoch_time
            additional_metrics = []

            # Post-process predictions
            if self.problem_type == "classification":
                y_pred_processed, y_true_processed = self.post_process_classification(
                    y_pred, y_shuffled
                )

                # Flatten if shape is (B,T) or anything beyond 1D
                y_pred_flat = y_pred_processed.reshape(-1)
                y_true_flat = y_true_processed.reshape(-1).astype(int)

                # Compute accuracy
                acc_val = accuracy(y_pred_flat, y_true_flat)
                additional_metrics.append(f"\n\tAccuracy: {acc_val:.2f}")
            else:
                # e.g. for a regression or something else
                additional_metrics.append(
                    f"\n\tMean Squared Error: {mean_squared_error(y_pred.data, y_shuffled):.2f}"
                )

            base_message = (
                f"\nEpoch: {epoch}"
                f"\n\tLoss: {avg_loss:.4f}"
                f"\n\tEpochs/sec: {epochs_per_second:.2f}"
            )

            # Print all metrics
            if additional_metrics:
                metrics_str = "\n\tAdditional Metrics:" + "\n".join(additional_metrics)
                logger.info(base_message + metrics_str)
            else:
                logger.info(base_message)

            # If requested, show some sample predictions vs. labels
            if sample_predictions:
                # Pick a few random samples
                num_samples_to_show = min(num_samples_to_show, len(X_shuffled))
                indices = np.random.choice(
                    len(X_shuffled), size=num_samples_to_show, replace=False
                )
                print("\nSample Validation Predictions:")
                for i, idx in enumerate(indices):
                    # Model output
                    pred_sample = y_pred_processed[idx]
                    # Actual label
                    true_sample = y_true_processed[idx]

                    # pred_sample, true_sample = self.post_process_classification(pred_sample, true_sample)

                    print(f"  Example #{i+1}:")
                    print(f"    Predicted: {pred_sample}")
                    print(f"    Target:    {true_sample}")
                print("")

        self.model.train()

    def post_process_classification(self, y_pred, y_true):
        """
        Convert model outputs to integer class predictions.
        y_pred/y_true may be:
          - 'logits': raw logits for multi-class => we do argmax across the last dimension
          - 'sigmoid': multi-label => threshold at 0.5
          - 'softmax': multi-class => already a probability distribution => argmax
        etc.
        """
        y_pred = y_pred.data if isinstance(y_pred, Tensor) else y_pred
        y_true = y_true.data if isinstance(y_true, Tensor) else y_true
        # Suppose y_pred.shape can be (Batch Size, Sequence length, # of Classes)
        # or (Batch Size, # of Classes). We'll always argmax along axis=-1 for multi-class
        if self.output_type == "logits":
            # Multi-class logits => argmax(softmax(logits))
            # But we can just do an argmax of raw logits.
            # (The max of the logits is the same as the max after softmax.)
            return np.argmax(y_pred, axis=-1), np.argmax(y_true, axis=-1)

        elif self.output_type == "softmax":
            # Already probabilities => argmax
            return np.argmax(y_pred, axis=-1), y_true

        elif self.output_type == "sigmoid":
            # Multi-label => threshold at 0.5
            # If multi-dimensional, we might do > 0.5 per dimension
            return (y_pred >= 0.5).astype(int), (y_true >= 0.5).astype(int)

        else:
            # For unknown output types, just return data
            return y_pred, y_true
