import logging
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Tuple

try:
    # drop-in replacement for numpy for GPU acceleration
    import cupy as np  # type: ignore

    _ = np.cuda.runtime.getDeviceCount()  # Check if a CUDA device is available
except Exception:
    import numpy as np
from tqdm import tqdm

from autograd import nn, optim
from autograd.tensor import Tensor
from autograd.text import utils as text_utils
from autograd.tools.config_schema import (
    GenericTrainingConfig,
    TransformerTrainingConfig,
)
from autograd.tools.metrics import accuracy, mean_squared_error
from autograd.tools.model import load_checkpoint, save_checkpoint

logger = logging.getLogger(__name__)


class AbstractTrainer(ABC):
    """Base trainer that defines a high-level training loop.

    Subclasses should implement domain-specific steps such as:
    - forward pass and loss computation in `train_step`
    - evaluation logic in `evaluate`

    Attributes:
        CHECKPOINT_DIR (str): Default directory for checkpoints.
        METRICS_DIR (str): Default directory for training metrics.
    """

    CHECKPOINT_DIR = "checkpoints"
    METRICS_DIR = "training_runs"

    def __init__(
        self,
        model_cls: type[nn.Module],
        optimizer_cls: type[optim.Optimizer],
        loss_fn: Callable,
        config: GenericTrainingConfig,
        **kwargs,
    ):
        """Initializes the trainer with model, optimizer, and config.

        Args:
            model_cls (type[nn.Module]): Class of the neural network model to instantiate.
            optimizer_cls (type[optim.Optimizer]): Class of the optimizer to instantiate.
            loss_fn (Callable): A function or callable that computes the loss.
            config (GenericTrainingConfig): Training configuration object.
            **kwargs: Additional arguments for specialized trainers (e.g., checkpoint paths).
        """
        self.config = config or {}
        self.loss_fn = loss_fn
        self.kwargs = kwargs
        self.model, self.optimizer, self.checkpoint = self._load_model_and_optimizer(
            model_class=model_cls,
            optimizer_class=optimizer_cls,
            model_kwargs=config.model_kwargs,
            optimizer_kwargs=config.optimizer_kwargs,
            resume_epoch=config.resume_epoch,
            checkpoint_path=kwargs.get("checkpoint_path"),
        )
        self.start_epoch = self.config.resume_epoch or 0
        self.metrics = defaultdict(list)

    def fit(self, train_data_loader, val_data_loader=None):
        """Performs the main training loop over the given data loaders.

        Args:
            train_data_loader: An iterable or generator that yields training batches.
            val_data_loader: (Optional) An iterable or generator that yields validation batches.
        """
        logger.info(
            f"Training {self.model.__class__.__name__} with "
            f"{(self.model.num_parameters()/1e6):.2f}M parameters."
        )
        for epoch in tqdm(
            range(self.start_epoch, self.config.total_epochs),
            desc="Training",
            leave=False,
            initial=self.start_epoch,
        ):
            if hasattr(train_data_loader, "on_epoch_start"):
                train_data_loader.on_epoch_start()

            self.model.train()
            train_loss = self._train_one_epoch(train_data_loader)

            if val_data_loader is not None:
                if hasattr(val_data_loader, "on_epoch_start"):
                    val_data_loader.on_epoch_start()
                val_loss = self.evaluate(train_data_loader, val_data_loader, epoch)
            else:
                val_loss = None

            self._on_epoch_end(epoch, train_loss, val_loss)

            if (
                epoch % (self.config.total_epochs // min(20, self.config.total_epochs))
                == 0
            ):
                self._save_metrics()

    def _train_one_epoch(self, data_loader) -> float:
        """Trains the model for one epoch on the provided data loader.

        This method handles gradient accumulation and parameter updates.

        Args:
            data_loader: An iterable or generator that yields training batches.

        Returns:
            float: The average training loss over the epoch.
        """
        total_loss = 0.0
        batch_count = 0
        self.optimizer.zero_grad()
        for batch in tqdm(data_loader, desc="Training Batches", leave=False):
            loss = self.train_step(batch, data_loader)
            total_loss += loss
            # Simulate larger batches by updating weights every N steps
            if (batch_count + 1) % self.config.update_weights_every_n_steps == 0:
                self.optimizer.step()  # increments .timestep, applies LR scheduler if present
                self.metrics["grad_l2_norm"].append(grad_l2_norm(self.model.parameters))
                self.optimizer.zero_grad()
            batch_count += 1
        return total_loss / max(batch_count, 1)

    def _save_checkpoint(self, epoch: int, val_loss: Optional[float]):
        """Saves a model checkpoint if the validation loss improves.

        Args:
            epoch (int): Current epoch number.
            val_loss (Optional[float]): The validation loss at this epoch.
        """
        if val_loss is not None and epoch % self.config.checkpoint_freq == 0:
            if self.metrics["val_loss"] and val_loss < min(self.metrics["val_loss"]):
                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "config": self.config,
                }
                os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
                cp_path_json = os.path.join(
                    self.CHECKPOINT_DIR,
                    f"{self.config.training_run_name}_{self.model.__class__.__name__}_{epoch}.json",
                )
                cp_path_npz = os.path.join(
                    self.CHECKPOINT_DIR,
                    f"{self.config.training_run_name}_{self.model.__class__.__name__}_{epoch}.npz",
                )
                save_checkpoint(
                    checkpoint, json_path=cp_path_json, npz_path=cp_path_npz
                )
                logger.info(f"Saved checkpoint to {cp_path_json} and {cp_path_npz}")

    def _save_metrics(self):
        """Saves accumulated training metrics to a compressed NPZ file."""
        os.makedirs(self.METRICS_DIR, exist_ok=True)
        run_name = self.config.training_run_name or "default"
        filename = f"{self.model.__class__.__name__}_{run_name}.npz"
        path = os.path.join(self.METRICS_DIR, filename)
        metrics_np = {k: np.array(v) for k, v in self.metrics.items()}
        np.savez_compressed(path, **metrics_np)
        logger.info(f"Saved training metrics to {path}")

    @abstractmethod
    def train_step(self, batch_data) -> float:
        """Performs a single training step and returns the loss as a float.

        Subclasses must implement the forward pass, loss computation, and backward pass here.

        Args:
            batch_data: A single batch of training data.

        Returns:
            float: The computed loss value for the batch.
        """
        pass

    def _on_epoch_end(self, epoch: int, train_loss: float, val_loss: Optional[float]):
        """Hook called at the end of each epoch to handle checkpointing and logging.

        Note: `save_checkpoint` needs to come first because it compares
        against historical metrics to determine whether to save
        the checkpoint

        Args:
            epoch (int): Current epoch number.
            train_loss (float): The average training loss for this epoch.
            val_loss (Optional[float]): The validation loss for this epoch, if available.
        """
        self._save_checkpoint(epoch, val_loss)
        self.metrics["epoch"].append(epoch)
        self.metrics["train_loss"].append(train_loss)
        self.metrics["val_loss"].append(val_loss)

        log_msg = f"[Epoch {epoch}]"
        for key in self.metrics:
            if key != "epoch" and self.metrics[key][-1] is not None:
                log_msg += f"\n{key} = {self.metrics[key][-1]:.4f}"
        log_msg += f"\nLR = {self.optimizer.lr:.4f}"
        logger.info(log_msg)

    @abstractmethod
    def evaluate(self, train_data_loader, val_data_loader, epoch) -> float:
        """Evaluates the model on the validation set.

        Args:
            train_data_loader: Training data loader (if needed for evaluation context).
            val_data_loader: Validation data loader providing validation batches.
            epoch (int): Current epoch number, provided for logging or conditional evaluation.

        Returns:
            float: The average validation loss over the validation set.
        """
        pass

    def _load_model_and_optimizer(
        self,
        model_class: type[nn.Module],
        optimizer_class: type[optim.Optimizer],
        model_kwargs: Dict[str, Any],
        optimizer_kwargs: Dict[str, Any],
        resume_epoch: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
    ) -> Tuple[Any, Any, dict]:
        """Loads model & optimizer states from checkpoint if resume_epoch is given.

        If no checkpoint is found or resume_epoch is None, initializes them from scratch.

        Note on configs in checkpoint:
            When we load from checkpoint, the model and optimizer configs are restored from the
            checkpoint and cannot be modified because they are inherently tied to the model and
            optimizer state (e.g. hidden_size cannot be changed because the model artifact was
            created using the previous checkpoint's hidden_size)
            The other configs (e.g. eval_iters) can be changed, and can just be read from the
            self.config attribute instead of checkpoint["hyperparams"].

        Args:
            model_class (type[nn.Module]): Class object for the model.
            optimizer_class (type[optim.Optimizer]): Class object for the optimizer.
            model_kwargs (Dict[str, Any]): Keyword arguments for model initialization.
            optimizer_kwargs (Dict[str, Any]): Keyword arguments for optimizer initialization.
            resume_epoch (Optional[int]): If provided, attempts to load from checkpoint at this epoch.
            checkpoint_path (Optional[str]): If provided, path of the checkpoint file.

        Returns:
            Tuple[Any, Any, dict]: A tuple containing:
                - model (nn.Module): The loaded or newly instantiated model.
                - optimizer (optim.Optimizer): The loaded or newly instantiated optimizer.
                - checkpoint (dict): Dictionary of checkpoint data if loaded, otherwise empty.
        """
        if resume_epoch is not None:
            # Construct paths if not provided
            if checkpoint_path is not None:
                ckpt_json = f"{checkpoint_path}.json"
                ckpt_npz = f"{checkpoint_path}.npz"
            else:
                ckpt_json = os.path.join(
                    self.CHECKPOINT_DIR,
                    f"{self.config.training_run_name}_{model_class.__name__}_{resume_epoch}.json",
                )
                ckpt_npz = os.path.join(
                    self.CHECKPOINT_DIR,
                    f"{self.config.training_run_name}_{model_class.__name__}_{resume_epoch}.npz",
                )

            logger.info(f"Attempting to load from checkpoint: {ckpt_json}, {ckpt_npz}")
            loaded_ckpt = load_checkpoint(ckpt_json, ckpt_npz)

            # If your checkpoint has hyperparams or constructor kwargs,
            # you can either override model_kwargs or do a partial merge:
            #  e.g. model_kwargs.update(loaded_ckpt["hyperparams"].get("model_kwargs", {}))
            hyperparams = loaded_ckpt.get("hyperparams", {})
            model_init_kwargs = hyperparams.get("model_kwargs", model_kwargs)
            optimizer_init_kwargs = hyperparams.get(
                "optimizer_kwargs", optimizer_kwargs
            )

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


class SimpleTrainer(AbstractTrainer):
    """Trainer for standard supervised tasks where each batch is (X, y).

    This trainer supports classification or regression depending on the
    `output_type` parameter.
    """

    def __init__(
        self,
        model_cls: type[nn.Module],
        optimizer_cls: type[optim.Optimizer],
        loss_fn: Callable,
        config: GenericTrainingConfig,
        output_type: Optional[str] = None,
        sample_predictions: bool = False,
        **kwargs,
    ):
        """Initializes the SimpleTrainer.

        Args:
            model_cls (type[nn.Module]): Class object for the model to train.
            optimizer_cls (type[optim.Optimizer]): Class object for the optimizer.
            loss_fn (Callable): A function or callable that computes the loss.
            config (GenericTrainingConfig): Training configuration object.
            output_type (Optional[str]): The type of output from the model
                (e.g. 'logits', 'sigmoid', 'softmax', or None for regression).
            sample_predictions (bool): If True, can enable additional logging
                or sampling of predictions. Defaults to False.
            **kwargs: Additional arguments passed to the parent trainer.
        """
        super().__init__(model_cls, optimizer_cls, loss_fn, config=config, **kwargs)
        self.sample_predictions = sample_predictions
        self.output_type = output_type
        self.problem_type = (
            "classification"
            if output_type in ["logits", "sigmoid", "softmax"]
            else "regression"
        )

    def train_step(self, batch_data, data_loader=None) -> float:
        """Performs a forward pass, computes the loss, and backpropagates.

        Note that .zero_grad() and .step() calls are in the _train_one_epoch() function

        Args:
            batch_data: A tuple (batch_X, batch_y) for supervised learning.
            data_loader: (Optional) The data loader, if additional context is needed.

        Returns:
            float: The computed loss for this batch.
        """
        batch_X, batch_y = batch_data
        y_pred = self.model(batch_X)
        loss = self.loss_fn(y_pred, batch_y)
        loss.backward()
        return float(loss.detach().data)

    def evaluate(self, train_data_loader, val_data_loader, epoch) -> float:
        """Evaluates the model on the validation data loader.

        Args:
            train_data_loader: Unused in this implementation, but kept for interface consistency.
            val_data_loader: An iterable or generator providing validation batches.
            epoch (int): The current epoch number.

        Returns:
            float: The average validation loss over the entire validation dataset.
        """
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

        # Compute additional metrics for classification or regression
        y_pred_full = np.concatenate([p.data for p in all_preds], axis=0)
        y_true_full = np.concatenate([t.data for t in all_targets], axis=0)

        if self.problem_type == "classification":
            y_pred_proc, y_true_proc = self._post_process_classification(
                y_pred_full, y_true_full
            )
            acc_val = accuracy(
                y_pred_proc.reshape(-1), y_true_proc.reshape(-1).astype(int)
            )
            self.metrics["val_accuracy"].append(acc_val)
        else:
            mse_val = mean_squared_error(y_pred_full, y_true_full)
            self.metrics["val_mse"].append(mse_val)

        self.model.train()
        return avg_val_loss

    def _post_process_classification(self, y_pred, y_true):
        """Post-processes outputs for classification tasks.

        Args:
            y_pred (np.ndarray): Predicted logits or probabilities.
            y_true (np.ndarray): Ground truth labels.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Processed predictions and targets
                (e.g., for use in accuracy calculation).
        """
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
    """Trainer specialized for language modeling or next-token prediction tasks.

    Expects DataLoader batches of the form (X, dec_inp, y, src_mask, tgt_mask, causal_mask).
    """

    def __init__(
        self,
        model_cls: type[nn.Module],
        optimizer_cls: type[optim.Optimizer],
        loss_fn: Callable,
        forward_fn: nn.AbstractLLMForwardFn,
        config: TransformerTrainingConfig,
        **kwargs,
    ):
        """Initializes the LLMTrainer.

        Args:
            model_cls (type[nn.Module]): Class object for the language model.
            optimizer_cls (type[optim.Optimizer]): Class object for the optimizer.
            loss_fn (Callable): A callable that computes the language modeling loss.
            forward_fn (nn.AbstractLLMForwardFn): A class implementing `sample` and `train` methods
                to handle how the model is invoked for language modeling tasks.
            config (TransformerTrainingConfig): A specialized config containing transformer
                and training hyperparameters.
            **kwargs: Additional arguments passed to the parent trainer.
        """
        super().__init__(model_cls, optimizer_cls, loss_fn, config=config, **kwargs)
        self.forward_fn = forward_fn

    def train_step(self, batch_data, data_loader) -> float:
        """Performs a forward pass for language modeling, computes the loss, and backpropagates.

        This method expects batch_data in the form:
            (X, dec_inp, y, src_mask, tgt_mask, causal_mask).

        Note that .zero_grad() and .step() calls are in the _train_one_epoch() function

        Args:
            batch_data: The batch data required for the model's forward pass.
            data_loader: The data loader, used here to get pad_idx (padding index).

        Returns:
            float: The computed loss for this batch.
        """
        pred_probs, y = self.forward_fn(self.model, batch_data, mode="train")
        loss = self.loss_fn(
            pred_probs,
            y,
            pad_idx=data_loader.pad_idx,
            label_smoothing=self.config.label_smoothing,  # type: ignore
        )
        loss.backward()
        return float(loss.detach().data)

    def evaluate(self, train_data_loader, val_data_loader, epoch) -> float:
        """Evaluates the model on the validation data loader for language modeling.

        Args:
            train_data_loader: Training data loader (if needed for context).
            val_data_loader: Validation data loader providing batches with
                (X, dec_inp, y, src_mask, tgt_mask, causal_mask).
            epoch (int): Current epoch number.

        Returns:
            float: The average validation loss over the language modeling validation set.
        """
        self.model.eval()
        total_loss = 0.0
        batch_count = 0
        for batch_data in tqdm(val_data_loader, desc="Evaluation", leave=False):
            pred_probs, y = self.forward_fn(self.model, batch_data, mode="train")
            loss = self.loss_fn(pred_probs, y, pad_idx=val_data_loader.pad_idx)
            total_loss += float(loss.detach().data)
            batch_count += 1
        avg_val_loss = total_loss / max(batch_count, 1)

        if epoch % 2 == 0:  # Optionally perform inference every 2 epochs
            self._perform_inference(train_data_loader)

        self.model.train()
        return avg_val_loss

    def _perform_inference(self, train_data_loader):
        """Optionally performs teacher-forcing or sampling-based inference.
        Make sure LLMDataLoader has `data` if you rely on `train_data_loader.data`.

        Args:
            train_data_loader: Data loader that can provide the BPE tokenizer and some
                sample data to generate from.
        """
        if self.config.teacher_enforcing and hasattr(train_data_loader, "data"):
            text_utils.inference(
                model=self.model,
                prediction_func=self.forward_fn,
                bpe=train_data_loader.bpe,
                groundtruth_data=train_data_loader.data[
                    : train_data_loader.seq_len // 3
                ],
                max_length=train_data_loader.seq_len // 3,
            )

        # Normal sampling inference
        text_utils.inference(
            model=self.model,
            prediction_func=self.forward_fn,
            bpe=train_data_loader.bpe,
            start_tokens=self.config.eval_start_string,
            max_length=max(128, int(train_data_loader.seq_len * 0.4)),
            temperature=1.0,
            top_k=self.config.eval_top_k,
        )


def grad_l2_norm(parameters: Dict[str, Tensor]) -> float:
    """Computes the L2 norm of the gradients for the given model parameters.

    Args:
        parameters (Dict[str, Tensor]): A dictionary of named parameters (Tensor objects)
            for which to compute the gradient norm.

    Returns:
        float: The L2 norm of all gradients (i.e., sqrt of sum of squared gradient values).

    Example:
        >>> from autograd.tensor import Tensor
        >>> params = {"weight": Tensor([1, 2], requires_grad=True)}
        >>> # Suppose weight.grad = Tensor([0.1, -0.2])
        >>> params["weight"].grad = Tensor([0.1, -0.2])
        >>> grad_norm = grad_l2_norm(params)
        >>> grad_norm
        0.2236068
    """
    grad_norm = 0.0
    for p in parameters.values():
        if p.grad is not None:
            grad_norm += (p.grad.data**2).sum()
    return float(grad_norm**0.5)
