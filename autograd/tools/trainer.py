import logging
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm

from autograd import nn, optim
from autograd.tensor import Tensor
from autograd.text import utils as text_utils
from autograd.text.tokenizer import BytePairEncoder
from autograd.tools.metrics import accuracy, mean_squared_error
from autograd.tools.model import load_checkpoint, save_checkpoint

logger = logging.getLogger(__name__)


class AbstractTrainer(ABC):
    """
    A base trainer class that defines the high-level training loop.
    Domain-specific steps (forward pass, loss computation, evaluation)
    are delegated to subclasses.
    """

    CHECKPOINT_DIR = "checkpoints"
    METRICS_DIR = "training_runs"

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        total_epochs: int = 10,
        checkpoint: Optional[dict] = None,
        config: Optional[Dict] = None,
    ):
        """
        Args:
            model: The model to train (must have .train() and .eval() methods).
            optimizer: Optimizer that updates the model parameters.
            total_epochs (int): Number of epochs.
            checkpoint (dict): (Optional) Checkpoint to resume training.
            config (dict): (Optional) A config dictionary (e.g., training_run_name).
        """
        self.model = model
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.config = config or {}
        self.checkpoint = checkpoint or {}
        self.start_epoch = self.checkpoint.get("epoch", 0)
        # Store metrics as lists for easy conversion later.
        self.metrics = defaultdict(list)

    def fit(self, train_data_loader, val_data_loader=None):
        """
        The main training loop. Automatically calls DataLoader.on_epoch_start() if present.
        """
        logger.info(
            f"Training {self.model.__class__.__name__} with {(self.model.num_parameters()/1e6):.2f}M parameters."
        )
        for epoch in tqdm(
            range(self.start_epoch, self.total_epochs),
            desc="Training",
            leave=False,
            initial=self.start_epoch,
        ):
            if hasattr(train_data_loader, "on_epoch_start"):
                train_data_loader.on_epoch_start()

            self.model.train()
            train_loss = self._train_one_epoch(train_data_loader)

            val_loss = None
            if val_data_loader is not None:
                if hasattr(val_data_loader, "on_epoch_start"):
                    val_data_loader.on_epoch_start()
                val_loss = self.evaluate_and_log(
                    train_data_loader, val_data_loader, epoch
                )

            self._on_epoch_end(epoch, train_loss, val_loss)

        self._save_metrics()

    def _train_one_epoch(self, data_loader) -> float:
        total_loss = 0.0
        batch_count = 0
        for batch in tqdm(data_loader, desc="Training Batches", leave=False):
            loss = self.train_step(batch)
            total_loss += loss
            batch_count += 1
        return total_loss / max(batch_count, 1)

    def _save_checkpoint(self, epoch: int, val_loss: Optional[float]):
        """
        Save a checkpoint if the validation loss improves.
        """
        if val_loss is not None:
            if not self.metrics["val_loss"] or val_loss < min(self.metrics["val_loss"]):
                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "config": self.config,
                }
                os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
                cp_path_json = os.path.join(
                    self.CHECKPOINT_DIR, f"{self.model.__class__.__name__}_{epoch}.json"
                )
                cp_path_npz = os.path.join(
                    self.CHECKPOINT_DIR, f"{self.model.__class__.__name__}_{epoch}.npz"
                )
                save_checkpoint(
                    checkpoint, json_path=cp_path_json, npz_path=cp_path_npz
                )
                logger.info(f"Saved checkpoint to {cp_path_json} and {cp_path_npz}")

    def _save_metrics(self):
        os.makedirs(self.METRICS_DIR, exist_ok=True)
        run_name = self.config.get("training_run_name", "default")
        filename = f"{self.model.__class__.__name__}_{run_name}.npz"
        path = os.path.join(self.METRICS_DIR, filename)
        # Convert metrics lists to NumPy arrays.
        metrics_np = {k: np.array(v) for k, v in self.metrics.items()}
        np.savez_compressed(path, **metrics_np)
        logger.info(f"Saved training metrics to {path}")

    @abstractmethod
    def train_step(self, batch_data) -> float:
        """
        Perform a single training step and return the loss as a float.
        """
        pass

    def _on_epoch_end(self, epoch: int, train_loss: float, val_loss: Optional[float]):
        logger.info(
            f"[Epoch {epoch}] Train Loss = {train_loss:.4f}"
            + (f", Val Loss = {val_loss:.4f}" if val_loss is not None else "")
        )
        self.metrics["epoch"].append(epoch)
        self.metrics["train_loss"].append(train_loss)
        self.metrics["val_loss"].append(val_loss)
        self._save_checkpoint(epoch, val_loss)

    @abstractmethod
    def evaluate_and_log(self, train_data_loader, val_data_loader, epoch) -> float:
        """
        Evaluate on the validation set, log metrics, and return average loss.
        """
        pass


class SimpleTrainer(AbstractTrainer):
    """
    Trainer for standard supervised tasks where each batch is (X, y).
    """

    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        epochs: int,
        output_type: Optional[str] = None,
        sample_predictions: bool = False,
        config: Optional[Dict] = None,
    ):
        super().__init__(model, optimizer, epochs, config=config)
        self.loss_fn = loss_fn
        self.sample_predictions = sample_predictions
        self.output_type = output_type
        # Infer problem type.
        self.problem_type = (
            "classification"
            if output_type in ["logits", "sigmoid", "softmax"]
            else "regression"
        )

    def train_step(self, batch_data) -> float:
        batch_X, batch_y = batch_data
        self.optimizer.zero_grad()
        y_pred = self.model(batch_X)
        loss = self.loss_fn(y_pred, batch_y)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().data)

    def evaluate_and_log(self, train_data_loader, val_data_loader, epoch) -> float:
        self.model.eval()
        total_loss = 0.0
        batch_count = 0
        all_preds, all_targets = [], []
        for batch_data in val_data_loader:
            batch_X, batch_y = batch_data
            y_pred = self.model(batch_X)
            loss = self.loss_fn(y_pred, batch_y)
            total_loss += float(loss.detach().data)
            batch_count += 1
            all_preds.append(y_pred)
            all_targets.append(batch_y)
        avg_val_loss = total_loss / max(batch_count, 1)

        # Optionally compute additional metrics.
        y_pred_full = np.concatenate([p.data for p in all_preds], axis=0)
        y_true_full = np.concatenate([t.data for t in all_targets], axis=0)
        if self.problem_type == "classification":
            y_pred_proc, y_true_proc = self._post_process_classification(
                y_pred_full, y_true_full
            )
            acc_val = accuracy(
                y_pred_proc.reshape(-1), y_true_proc.reshape(-1).astype(int)
            )
            logger.info(
                f"Epoch {epoch}: Val Loss = {avg_val_loss:.4f}, Accuracy = {acc_val:.2f}"
            )
        else:
            mse_val = mean_squared_error(y_pred_full, y_true_full)
            logger.info(
                f"Epoch {epoch}: Val Loss = {avg_val_loss:.4f}, MSE = {mse_val:.2f}"
            )
        self.model.train()
        return avg_val_loss

    def _post_process_classification(self, y_pred, y_true):
        if isinstance(y_pred, Tensor):
            y_pred = y_pred.data
        if isinstance(y_true, Tensor):
            y_true = y_true.data
        if self.output_type in ["logits", "softmax"]:
            return np.argmax(y_pred, axis=-1), y_true
        elif self.output_type == "sigmoid":
            return (y_pred >= 0.5).astype(int), (y_true >= 0.5).astype(int)
        else:
            return y_pred, y_true


class LLMTrainer(AbstractTrainer):
    """
    Trainer specialized for language modeling or next-token tasks.
    Expects DataLoader batches as: (X, dec_inp, y, src_mask, tgt_mask, causal_mask)
    """

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        warmup_steps: int,
        tokenizer: BytePairEncoder,
        epochs: int = 10,
        label_smoothing: float = 0.0,
        forward_fn: Optional[Callable] = None,
        start_tokens: Optional[str] = "\n",
        config: Optional[Dict] = None,
    ):
        super().__init__(model, optimizer, epochs, config=config)
        self.loss_fn = loss_fn
        self.warmup_steps = warmup_steps
        self.label_smoothing = label_smoothing
        self.tokenizer = tokenizer
        self.forward_fn = forward_fn or self.default_forward_fn
        self.start_tokens = start_tokens

    def train_step(self, batch_data) -> float:
        """
        batch_data: (X, dec_inp, y, src_mask, tgt_mask, causal_mask)
        """
        self.optimizer.zero_grad()
        pred_probs, y = self.forward_fn(self.model, batch_data, mode="train")
        loss = self.loss_fn(
            pred_probs,
            y,
            pad_idx=getattr(batch_data, "pad_idx", None),
            label_smoothing=self.label_smoothing,
        )
        loss.backward()
        self.optimizer.step()  # increments .timestep, applies LR scheduler if present
        return float(loss.detach().data)

    def evaluate_and_log(self, train_data_loader, val_data_loader, epoch) -> float:
        self.model.eval()
        total_loss = 0.0
        batch_count = 0
        for batch_data in tqdm(val_data_loader, desc="Evaluation", leave=False):
            pred_probs, y = self.forward_fn(self.model, batch_data, mode="train")
            loss = self.loss_fn(pred_probs, y, pad_idx=val_data_loader.pad_idx)
            total_loss += float(loss.detach().data)
            batch_count += 1
        avg_val_loss = total_loss / max(batch_count, 1)
        logger.info(
            f"Epoch {epoch}: Val Loss = {avg_val_loss:.4f}, LR = {self.optimizer.lr:.4f}"
        )

        # Optional inference every few epochs:
        if epoch % 2 == 0:
            self._perform_inference(train_data_loader)
        self.model.train()
        return avg_val_loss

    def default_forward_fn(self, model, batch_data, **kwargs):
        X, dec_inp, y, src_mask, tgt_mask, causal_mask = batch_data
        logits = model(X, dec_inp, src_mask, tgt_mask)
        return logits, y

    def _perform_inference(self, train_data_loader):
        """
        Optionally perform teacher-forcing or sampling.
        Make sure LLMDataLoader has `data` if you rely on `train_data_loader.data`.
        """
        if self.config.get("teacher_enforcing", False) and hasattr(
            train_data_loader, "data"
        ):
            text_utils.inference(
                prediction_func=lambda seq: self.forward_fn(
                    self.model, seq, mode="sample"
                ),
                bpe=self.tokenizer,
                groundtruth_data=train_data_loader.data[: train_data_loader.seq_len],
                max_length=train_data_loader.seq_len // 4,
            )

        # Normal sampling
        text_utils.inference(
            prediction_func=lambda seq: self.forward_fn(self.model, seq, mode="sample"),
            bpe=self.tokenizer,
            start_tokens=self.start_tokens,
            max_length=int(self.model.max_seq_len * 0.4),
            temperature=1.0,
            top_k=200,
        )


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
