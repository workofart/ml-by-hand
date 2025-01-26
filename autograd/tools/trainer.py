import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm

from autograd.tensor import Tensor
from autograd.text import utils as text_utils
from autograd.text.tokenizer import BytePairEncoder
from autograd.tools.data import LLMDataLoader
from autograd.tools.metrics import accuracy, mean_squared_error
from autograd.tools.model import load_checkpoint, save_checkpoint

logger = logging.getLogger(__name__)


class AbstractTrainer(ABC):
    """
    A base class that defines the high-level training loop but
    delegates domain-specific steps to the subclasses.
    """

    def __init__(
        self,
        model,
        optimizer,
        total_epochs=10,
        checkpoint: Optional[dict] = None,
        **kwargs,
    ):
        """
        Args:
            model: The model to train (generic object with a .train() and .eval() method).
            optimizer: Optimizer that knows how to update model parameters.
            total_epochs (int): Number of epochs to train.
        """
        self.model = model
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.checkpoint = checkpoint if checkpoint is not None else {}
        self.step_count = self.checkpoint.get("step_count", 0)
        self.kwargs = kwargs

    def fit(self, train_data_loader, val_data_loader=None, **kwargs):
        """
        The main training loop: calls on_epoch_start, train_epoch, optional eval, etc.
        """
        logger.info(f"Model parameters: {self.model.num_parameters()}")

        start_epoch = self.checkpoint.get("epoch", 0)
        for epoch in tqdm(
            range(start_epoch, self.total_epochs),
            desc="Training",
            leave=False,
            initial=start_epoch,
        ):
            self.on_epoch_start(epoch)

            if hasattr(train_data_loader, "on_epoch_start"):
                train_data_loader.on_epoch_start()

            self.model.train()
            train_loss = self._train_one_epoch(train_data_loader, **kwargs)

            val_loss = None
            if val_data_loader is not None:
                if hasattr(val_data_loader, "on_epoch_start"):
                    val_data_loader.on_epoch_start()
                val_loss = self.evaluate_and_log(train_data_loader, val_data_loader)

            self.on_epoch_end(epoch, train_loss, val_loss)

    @abstractmethod
    def on_epoch_start(self, epoch):
        pass

    @abstractmethod
    def train_step(self, batch_data):
        """
        Must do:
          - zero_grad
          - forward pass
          - compute loss
          - backward + optimizer step
          - return float loss
        """
        pass

    def on_epoch_end(self, epoch, train_loss, val_loss):
        # TODO: add support for loading model checkpoint
        if epoch % self.kwargs.get("checkpoint_freq", 1) == 0:
            # Save checkpoint
            checkpoint = {
                "epoch": epoch + 1,  # the next epoch to start from
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "hyperparams": self.kwargs.get("hyperparams"),
                "step_count": self.step_count,  # for learning rate scheduler
            }
            save_checkpoint(
                checkpoint,
                json_path=f"checkpoints/{self.model.__class__.__name__}_{epoch}.json",
                npz_path=f"checkpoints/{self.model.__class__.__name__}_{epoch}.npz",
            )
            logger.info(
                f"Saving checkpoint to checkpoints/{self.model.__class__.__name__}_{epoch}.json and .npz"
            )

    def _train_one_epoch(self, train_data_loader, **kwargs):
        self.model.train()
        total_loss = 0.0

        for batch_data in tqdm(train_data_loader, desc="Training Batches", leave=False):
            self.step_count += 1
            loss_val = self.train_step(batch_data, **kwargs)
            total_loss += loss_val

        return total_loss / max(len(train_data_loader), 1)

    @abstractmethod
    def evaluate_and_log(
        self,
        train_data_loader,
        val_data_loader,
        sample_predictions=False,
        num_samples_to_show=4,
    ):
        """
        Must return a scalar loss or metric (e.g. average val loss).
        """
        pass


class SimpleTrainer(AbstractTrainer):
    """
    A trainer for standard supervised tasks where each batch is (X, y).
    """

    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        epochs,
        output_type=None,
        sample_predictions=False,
    ):
        super().__init__(model, optimizer, epochs)
        self.loss_fn = loss_fn
        self.output_type = output_type
        self.sample_predictions = sample_predictions

        # Decide problem type based on output_type
        if output_type in ["logits", "sigmoid", "softmax"]:
            self.problem_type = "classification"
        else:
            self.problem_type = "regression"

    def on_epoch_start(self, epoch):
        pass

    def train_step(self, batch_data, **kwargs):
        """
        batch_data = (batch_X, batch_y)
        """
        batch_X, batch_y = batch_data
        self.optimizer.zero_grad()
        y_pred = self.model(batch_X)
        loss = self.loss_fn(
            y_pred,
            batch_y,
            **(
                {"weight": kwargs.get("weight")}
                if kwargs.get("weight") is not None
                else {}
            ),
        )
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().data)

    def on_epoch_end(self, epoch, train_loss, val_loss):
        msg = f"[Epoch {epoch}] Train Loss = {train_loss:.4f}"
        if val_loss is not None:
            msg += f", Val Loss = {val_loss:.4f}"
        print(msg)

    def evaluate_and_log(
        self,
        train_data_loader,
        val_data_loader,
    ) -> float:
        """
        1) Computes avg validation loss.
        2) Computes additional metrics (accuracy or MSE).
        3) Optionally logs sample predictions.
        4) Returns the average validation loss.
        """
        self.model.eval()
        all_preds = []
        all_targets = []
        total_val_loss = 0.0
        batch_count = 0

        for batch_data in val_data_loader:
            batch_X, batch_y = batch_data
            y_pred = self.model(batch_X)
            loss = self.loss_fn(y_pred, batch_y)
            total_val_loss += float(loss.detach().data)
            batch_count += 1

            all_preds.append(y_pred)
            all_targets.append(batch_y)

        avg_val_loss = total_val_loss / max(batch_count, 1)

        # Convert all_preds / all_targets to numpy
        y_pred_full = np.concatenate([p.data for p in all_preds], axis=0)
        y_true_full = np.concatenate([t.data for t in all_targets], axis=0)

        metrics_str = ""
        if self.problem_type == "classification":
            y_pred_processed, y_true_processed = self.post_process_classification(
                y_pred_full, y_true_full
            )
            y_pred_flat = y_pred_processed.reshape(-1)
            y_true_flat = y_true_processed.reshape(-1).astype(int)
            acc_val = accuracy(y_pred_flat, y_true_flat)
            metrics_str += f"\n\tAccuracy: {acc_val:.2f}"
        else:
            mse_val = mean_squared_error(y_pred_full, y_true_full)
            metrics_str += f"\n\tMean Squared Error: {mse_val:.2f}"

        logger.info(f"Val Loss: {avg_val_loss:.4f}{metrics_str}")

        if self.sample_predictions and len(y_pred_full) > 0:
            # Periodically log information
            y_pred_processed, y_true_processed = self.post_process_classification(
                y_pred, batch_y
            )
            print("------- Ground Truth -------")
            print(y_true_processed[0])
            print("-------- Prediction --------")
            print(y_pred_processed[0])
            print("")

        self.model.train()
        return avg_val_loss

    def post_process_classification(self, y_pred, y_true):
        """
        Convert model outputs to integer class predictions
        based on self.output_type.
        """
        if isinstance(y_pred, Tensor):
            y_pred = y_pred.data
        if isinstance(y_true, Tensor):
            y_true = y_true.data

        if self.output_type == "logits":
            # Argmax over last dimension
            return np.argmax(y_pred, axis=-1), y_true
        elif self.output_type == "softmax":
            # Argmax over last dimension (already probs)
            return np.argmax(y_pred, axis=-1), y_true
        elif self.output_type == "sigmoid":
            # Multi-label => threshold at 0.5
            return (y_pred >= 0.5).astype(int), (y_true >= 0.5).astype(int)
        else:
            # For unknown output types or regression, just return raw
            return y_pred, y_true


class LLMTrainer(AbstractTrainer):
    """
    A trainer specialized for language modeling or next-token tasks.
    Expects that the DataLoader yields batches of the form:
      (X, dec_inp, y, source_mask, target_mask, causal_mask)
    """

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        warmup_steps: int,
        tokenizer: BytePairEncoder,
        epochs=10,
        label_smoothing=0.0,
        forward_fn: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(model, optimizer, epochs, **kwargs)
        self.loss_fn = loss_fn
        self.warmup_steps = warmup_steps
        self.label_smoothing = label_smoothing
        self.d_model = model.hidden_size  # needed for LR scheduelr TODO: ensure this interface is consistent for all LLM models
        self.forward_fn = forward_fn or self.default_forward_fn
        self.tokenizer = tokenizer

    def on_epoch_start(self, epoch):
        # Could adjust learning rate, log epoch number, etc.
        pass

    def train_step(self, batch_data, **kwargs):
        """
        batch_data = (X, dec_inp, y, src_mask, tgt_mask, causal_mask)
        """
        self.step_count += 1
        self.optimizer.lr = get_lr(
            self.step_count,
            self.d_model,
            warmup_steps=self.warmup_steps,
        )

        self.optimizer.zero_grad()

        pred_probs, y = self.forward_fn(self.model, batch_data, mode="train", **kwargs)

        loss = self.loss_fn(
            pred_probs,
            y,
            pad_idx=kwargs.get("pad_idx", None),
            label_smoothing=self.label_smoothing,
        )
        loss.backward()
        self.optimizer.step()

        return float(loss.detach().data)

    def on_epoch_end(self, epoch, train_loss, val_loss):
        msg = f"[Epoch {epoch}] Train Loss = {train_loss:.4f}"
        if val_loss is not None:
            msg += f", Val Loss = {val_loss:.4f}"
        print(msg)
        super().on_epoch_end(epoch, train_loss, val_loss)

    def evaluate_and_log(
        self,
        train_data_loader: LLMDataLoader,
        val_data_loader: LLMDataLoader,
    ) -> float:
        self.model.eval()
        total_val_loss = 0.0

        for batch_data in tqdm(val_data_loader, desc="Evaluation", leave=False):
            pred_probs, y = self.forward_fn(self.model, batch_data, mode="train")
            # X, dec_inp, y, src_mask, tgt_mask, causal_mask = batch_data
            # pred_probs = self.model(X, dec_inp, src_mask, tgt_mask)
            loss = self.loss_fn(pred_probs, y, pad_idx=val_data_loader.pad_idx)
            total_val_loss += float(loss.detach().data)

        logger.warning(
            f"\nGradient L2 Norm: {grad_l2_norm(self.model.parameters):.2f}\n"
            f"| Test Loss: {total_val_loss / len(val_data_loader):.2f}\n"
            f"| Test Perplexity: {np.exp(total_val_loss / len(val_data_loader)):.2f} vs {len(val_data_loader.vocab)} (vocab size)\n"
            f"| Learning Rate: {self.optimizer.lr:.4f}"
        )

        if self.kwargs.get("teacher_enforcing", False):
            text_utils.teacher_forcing_inference(
                prediction_func=lambda seq_so_far: self.forward_fn(
                    self.model, seq_so_far, mode="inference"
                ),
                bpe=self.tokenizer,
                groundtruth_data=train_data_loader.data[: train_data_loader.seq_len],
                max_length=train_data_loader.seq_len // 4,
            )

        self.model.train()

        return total_val_loss / len(val_data_loader)

    def default_forward_fn(self, model, batch_data, **kwargs):
        # e.g. the 6-tuple approach
        X, dec_inp, y, src_mask, tgt_mask, causal_mask = batch_data
        logits = model(X, dec_inp, src_mask, tgt_mask)
        return logits, y


def get_lr(step: int, model_dim: int, warmup_steps: int) -> float:
    """
    Learning rate scheduler with warmup for transformers training. It will start with larger learning rate, then after the transition point sqrt(step) == step * warmup_steps^(-1.5), the learning rate will slowly decrease

    Args:
        step (int): The current timestep (not epoch), each batch will be 1 timestep
        model_dim (int): The model dimension
        warmup_steps (int): The number of timesteps to warm up (increase learning rate) before decreasing learning rate

    Returns:
        float: learning rate
    """
    return model_dim**-0.5 * min(step**-0.5, step * warmup_steps**-1.5)


def grad_l2_norm(parameters: Dict[str, Tensor]) -> float:
    """Compute the L2 norm of the gradients of the given parameters."""
    grad_norm = 0.0
    for p in parameters.values():
        if p.grad is not None:
            grad_norm += (p.grad.data**2).sum()
    return grad_norm**0.5


def load_model_and_optimizer(
    model_class,
    optimizer_class,
    model_kwargs: Dict[str, Any],
    optimizer_kwargs: Dict[str, Any],
    resume_epoch: Optional[int] = None,
    checkpoint_dir: str = "checkpoints",
) -> Tuple[Any, Any, dict]:
    """
    Loads model & optimizer states from checkpoint if resume_epoch is given.
    Otherwise, initializes them from scratch.

    Args:
        model_class: The class object for your model (e.g. Transformer).
        optimizer_class: The class object for your optimizer (e.g. optim.Adam).
        model_kwargs: Dictionary of kwargs to pass when constructing the model_class.
        optimizer_kwargs: Dictionary of kwargs to pass when constructing the optimizer_class.
        resume_epoch (int, optional): If provided, attempts to load from checkpoint
        checkpoint_dir (str): Directory where checkpoints are saved.

    Returns:
        (model, optimizer, checkpoint_dict)
    """
    if resume_epoch is not None:
        # Look for checkpoint files
        ckpt_json = os.path.join(
            checkpoint_dir, f"{model_class.__name__}_{resume_epoch}.json"
        )
        ckpt_npz = os.path.join(
            checkpoint_dir, f"{model_class.__name__}_{resume_epoch}.npz"
        )

        logger.info(f"Attempting to load from checkpoint: {ckpt_json}, {ckpt_npz}")
        loaded_ckpt = load_checkpoint(ckpt_json, ckpt_npz)

        # If your checkpoint has hyperparams or constructor kwargs,
        # you can either override model_kwargs or do a partial merge:
        #  e.g. model_kwargs.update(loaded_ckpt["hyperparams"].get("model_kwargs", {}))
        hyperparams = loaded_ckpt.get("hyperparams", {})
        model_init_kwargs = hyperparams.get("model_kwargs", model_kwargs)
        optimizer_init_kwargs = hyperparams.get("optimizer_kwargs", optimizer_kwargs)

        # Instantiate model & optimizer
        model = model_class(**model_init_kwargs)
        optimizer = optimizer_class(model.parameters, **optimizer_init_kwargs)

        # Load model/optimizer state
        model.load_state_dict(loaded_ckpt["model_state_dict"])
        optimizer.load_state_dict(loaded_ckpt["optimizer_state_dict"])

        logger.info(
            f"Loaded {model_class.__name__} from epoch {loaded_ckpt.get('epoch', '?')} "
            f"with step_count={loaded_ckpt.get('step_count', 0)}"
        )
        return model, optimizer, loaded_ckpt

    else:
        logger.info(
            "No resume_epoch given; initializing model & optimizer from scratch."
        )
        model = model_class(**model_kwargs)
        optimizer = optimizer_class(model.parameters, **optimizer_kwargs)
        return model, optimizer, {}
