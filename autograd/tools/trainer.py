import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pprint import pformat
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, cast

from tqdm import tqdm

from autograd import nn, optim
from autograd.backend import (
    Array,
    eval,
    xp,
)
from autograd.data.data_loader import DataLoader
from autograd.tensor import Tensor, no_grad
from autograd.tools.config_schema import (
    GenericTrainingConfig,
    TransformerTrainingConfig,
)
from autograd.tools.model import load_checkpoint, save_checkpoint

logger = logging.getLogger(__name__)


def _pformat_log_dict(payload: Mapping[str, Any]) -> str:
    compact_payload = {
        key: value for key, value in payload.items() if value is not None
    }
    return pformat(compact_payload, sort_dicts=False)


def _format_log_values(
    payload: Mapping[str, Any],
    *,
    default_precision: int = 4,
    precision_overrides: Optional[Mapping[str, int]] = None,
) -> dict[str, Any]:
    formatted: dict[str, Any] = {}
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, float):
            precision = (
                default_precision
                if precision_overrides is None
                else precision_overrides.get(key, default_precision)
            )
            formatted[key] = f"{value:.{precision}f}"
            continue
        formatted[key] = value
    return formatted


@dataclass
class TrainingState:
    report_loss_sum: Optional[float | Array] = None
    report_batches: int = 0
    accumulated_batches: int = 0
    eval_loss_sum: float = 0.0
    eval_loss_batches: int = 0
    eval_metric_totals: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def record_loss(self, loss: float | Tensor) -> None:
        loss_data = loss.data if isinstance(loss, Tensor) else float(loss)
        if self.report_loss_sum is None:
            self.report_loss_sum = loss_data
        else:
            self.report_loss_sum = self.report_loss_sum + loss_data
        self.report_batches += 1
        self.accumulated_batches += 1

    def record_eval_loss(self, loss: float | Tensor) -> None:
        loss_value = loss.item() if isinstance(loss, Tensor) else float(loss)
        self.eval_loss_sum += float(loss_value)
        self.eval_loss_batches += 1

    def record_eval_metric(
        self,
        name: str,
        *,
        numerator: float,
        denominator: float,
    ) -> None:
        totals = self.eval_metric_totals.setdefault(
            name,
            {"numerator": 0.0, "denominator": 0.0},
        )
        totals["numerator"] += float(numerator)
        totals["denominator"] += float(denominator)

    @property
    def train_loss(self) -> float:
        if self.report_loss_sum is None:
            return 0.0
        return float(xp.to_scalar(self.report_loss_sum / max(self.report_batches, 1)))

    @property
    def val_loss(self) -> float:
        return self.eval_loss_sum / max(self.eval_loss_batches, 1)

    @property
    def eval_metrics(self) -> Mapping[str, float]:
        return {
            name: totals["numerator"] / totals["denominator"]
            for name, totals in self.eval_metric_totals.items()
            if totals["denominator"] > 0
        }

    def to_metrics_row(
        self,
        *,
        eval_state: Optional["TrainingState"] = None,
    ) -> dict[str, Optional[float]]:
        row: dict[str, Optional[float]] = {
            "train_loss": self.train_loss,
            "val_loss": None if eval_state is None else eval_state.val_loss,
        }
        if eval_state is not None:
            for key, value in eval_state.eval_metrics.items():
                row[f"val_{key}"] = value
        return row

    def reset_report(self) -> None:
        self.report_loss_sum = None
        self.report_batches = 0

    def has_enough_batches(self, accumulation_steps: int):
        return self.accumulated_batches >= accumulation_steps


@dataclass(frozen=True)
class TrainingPlan:
    by_epoch: bool
    target_step: int
    report_every_steps: int
    checkpoint_every: int
    steps_per_epoch: Optional[int] = None
    metrics_interval: int = 1

    @classmethod
    def for_steps(cls, max_steps: int, checkpoint_every: int) -> "TrainingPlan":
        return cls(
            by_epoch=False,
            target_step=max_steps,
            report_every_steps=checkpoint_every,
            checkpoint_every=checkpoint_every,
        )

    @classmethod
    def for_epochs(
        cls,
        max_epochs: int,
        steps_per_epoch: int,
        checkpoint_every: int,
    ) -> "TrainingPlan":
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be > 0 in epoch mode.")

        return cls(
            by_epoch=True,
            target_step=max_epochs * steps_per_epoch,
            report_every_steps=steps_per_epoch,
            checkpoint_every=checkpoint_every,
            steps_per_epoch=steps_per_epoch,
            metrics_interval=max(1, max_epochs // min(20, max_epochs)),
        )

    def completed_epochs(self, step: int) -> int:
        if self.steps_per_epoch is None:
            raise ValueError("steps_per_epoch is required in epoch mode.")
        return step // self.steps_per_epoch

    def progress_value(self, step: int) -> int:
        return self.completed_epochs(step) if self.by_epoch else step

    def progress_label(self) -> str:
        return "Epoch" if self.by_epoch else "Step"

    def is_done(self, step: int) -> bool:
        return step >= self.target_step

    def should_report(self, step: int) -> bool:
        return self.is_done(step) or step % self.report_every_steps == 0

    def should_checkpoint(self, step: int) -> bool:
        return self.progress_value(step) % self.checkpoint_every == 0

    def should_save_metrics(self, step: int) -> bool:
        if not self.by_epoch:
            return True
        return self.progress_value(step) % self.metrics_interval == 0


class AbstractTrainer(ABC):
    """Base trainer that defines a high-level training loop.

    Subclasses should implement domain-specific steps such as:
    - forward pass and loss computation in `_compute_loss`
    - evaluation logic in `_evaluate`

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
        self.config = config
        self.loss_fn = loss_fn
        # `resume_epoch` is only a checkpoint lookup hint here. Runtime training
        # progress comes from the loaded checkpoint metadata, especially step_count.
        self.model, self.optimizer, self.checkpoint = self._load_model_and_optimizer(
            model_class=model_cls,
            optimizer_class=optimizer_cls,
            model_kwargs=config.model_kwargs,
            optimizer_kwargs=config.optimizer_kwargs,
            resume_epoch=config.resume_epoch,
            checkpoint_path=kwargs.get("checkpoint_path"),
        )
        # `global_step` counts consumed training batches / microbatches,
        # not optimizer updates. Under gradient accumulation, multiple
        # microbatches can contribute to one optimizer step.
        self.global_step = int(self.checkpoint.get("step_count", 0))
        self.metric_rows: list[dict[str, Optional[float]]] = []
        self.best_val_loss: Optional[float] = self.checkpoint.get("best_val_loss")
        self.last_grad_l2_norm: Optional[float] = None

    def fit(
        self,
        train_data_loader: DataLoader,
        val_data_loader: Optional[DataLoader] = None,
    ):
        """Performs the main training loop over the given data loaders.

        Args:
            train_data_loader: An iterable or generator that yields training batches.
            val_data_loader: (Optional) An iterable or generator that yields labeled
                validation batches.

        Notes:
            `global_step` counts consumed training batches / microbatches.
            It does not count optimizer updates when gradient accumulation is enabled.
        """

        self._validate_fit_inputs(train_data_loader)

        logger.info(
            f"Training {self.model.__class__.__name__} with "
            f"{(self.model.num_parameters() / 1e6):.2f}M parameters."
        )
        accumulation_steps = int(self.config.gradient_accumulation_steps)
        if self.config.max_steps is not None:
            plan = TrainingPlan.for_steps(
                max_steps=self.config.max_steps,
                checkpoint_every=self.config.checkpoint_freq,
            )
        else:
            plan = TrainingPlan.for_epochs(
                max_epochs=cast(int, self.config.max_epochs),
                steps_per_epoch=len(train_data_loader),
                checkpoint_every=self.config.checkpoint_freq,
            )

        self._log_fit_start(plan)

        state = TrainingState()
        if self.global_step >= plan.target_step:
            return

        self.model.train()
        self.optimizer.zero_grad()

        with tqdm(
            total=plan.target_step,
            initial=self.global_step,
            desc="Training",
            leave=False,
        ) as progress_bar:
            while not plan.is_done(self.global_step):
                train_data_loader.on_epoch_start()
                batches_this_pass = 0

                for batch in train_data_loader:
                    batches_this_pass += 1

                    loss = self._compute_loss(batch)
                    loss.backward()
                    state.record_loss(loss)

                    self.global_step += 1
                    progress_bar.update(1)

                    # whether we should flush
                    if state.has_enough_batches(
                        accumulation_steps
                    ) or plan.should_report(self.global_step):
                        self.optimizer_step(
                            state,
                            record_grad_norm=plan.should_report(self.global_step),
                        )

                    if plan.should_report(self.global_step):
                        eval_state = (
                            self.evaluate(val_data_loader)
                            if val_data_loader is not None
                            else None
                        )
                        self.report(state, plan, eval_state)

                    if plan.is_done(self.global_step):
                        break

                if (
                    plan.by_epoch
                    and plan.steps_per_epoch is not None
                    and batches_this_pass != plan.steps_per_epoch
                ):
                    raise ValueError(
                        "In epoch mode, len(train_data_loader) must match the actual "
                        "number of batches yielded per loader pass."
                    )

                # This is after all the batches in the data loader
                self.optimizer_step(
                    state,
                    record_grad_norm=False,
                )

    def optimizer_step(
        self, state: TrainingState, *, record_grad_norm: bool = True
    ) -> None:
        if state.accumulated_batches == 0:
            return

        self.optimizer.scale_gradients(1.0 / state.accumulated_batches)

        if record_grad_norm:
            self.last_grad_l2_norm = self.optimizer.grad_l2_norm()
        self.optimizer.step()
        self.optimizer.zero_grad()
        state.accumulated_batches = 0

    def evaluate(self, val_data_loader: DataLoader) -> TrainingState:
        val_data_loader.on_epoch_start()
        was_training = getattr(self.model, "_is_training", None)
        self.model.eval()
        try:
            with no_grad():
                return self._evaluate(val_data_loader)
        finally:
            if was_training is False:
                self.model.eval()
            else:
                self.model.train()

    def report(
        self,
        state: TrainingState,
        plan: TrainingPlan,
        eval_state: Optional[TrainingState],
    ) -> None:
        completed_epochs = (
            plan.completed_epochs(self.global_step) if plan.by_epoch else None
        )
        progress_value = plan.progress_value(self.global_step)
        row: dict[str, Optional[float]] = {
            "epoch": completed_epochs,
            "step": float(self.global_step),
            **state.to_metrics_row(eval_state=eval_state),
            "grad_l2_norm": self.last_grad_l2_norm,
        }

        if plan.should_checkpoint(self.global_step):
            self._maybe_save_checkpoint(plan=plan, eval_state=eval_state)

        self.metric_rows.append(row)
        log_payload = _format_log_values(
            {**row, "step": self.global_step, "lr": self.optimizer.lr},
            precision_overrides={"lr": 6},
        )
        logger.info(
            "[%s %s]\n%s",
            plan.progress_label(),
            progress_value,
            _pformat_log_dict(log_payload),
        )

        if plan.should_save_metrics(self.global_step):
            self._save_metrics()

        state.reset_report()

    @abstractmethod
    def _compute_loss(self, batch_data) -> Tensor:
        """Returns the scalar loss Tensor for a single training batch.

        Subclasses must implement the forward pass and loss computation here.
        `fit()` owns `.backward()`.

        Args:
            batch_data: A single batch of training data.

        Returns:
            Tensor: Scalar loss Tensor for the batch.
        """
        pass

    @abstractmethod
    def _evaluate(self, val_data_loader) -> TrainingState:
        """Evaluates the model on the validation set.

        Args:
            val_data_loader: Validation data loader providing validation batches.

        Returns:
            TrainingState: Raw validation aggregates for the current report.
        """
        pass

    def _maybe_save_checkpoint(
        self,
        *,
        plan: TrainingPlan,
        eval_state: Optional[TrainingState],
    ) -> None:
        if eval_state is not None:
            val_loss = eval_state.val_loss
            if self.best_val_loss is not None and val_loss >= self.best_val_loss:
                return
            self.best_val_loss = val_loss

        completed_epochs = (
            plan.completed_epochs(self.global_step) if plan.by_epoch else 0
        )
        checkpoint_index = plan.progress_value(self.global_step)

        checkpoint = {
            "epoch": completed_epochs,
            "step_count": self.global_step,
            "steps_per_epoch": plan.steps_per_epoch,
            "best_val_loss": self.best_val_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_init_kwargs": self.checkpoint.get(
                "model_init_kwargs", self.config.model_kwargs
            ),
            "optimizer_init_kwargs": self.checkpoint.get(
                "optimizer_init_kwargs", self.config.optimizer_kwargs
            ),
            "config_repr": repr(self.config),
        }
        checkpoint_name = (
            f"{self.config.training_run_name}_"
            f"{self.model.__class__.__name__}_{checkpoint_index}"
        )
        cp_path_json, cp_path_npz = save_checkpoint(
            checkpoint,
            checkpoint_dir=self.CHECKPOINT_DIR,
            checkpoint_name=checkpoint_name,
        )
        logger.info(f"Saved checkpoint to {cp_path_json} and {cp_path_npz}")

    def _save_metrics(self):
        """Saves accumulated training metrics to a compressed NPZ file."""
        # TODO: consider moving this to a centralized utility module instead of keeping it here
        os.makedirs(self.METRICS_DIR, exist_ok=True)
        run_name = self.config.training_run_name or "default"
        filename = f"{self.model.__class__.__name__}_{run_name}.npz"
        path = os.path.join(self.METRICS_DIR, filename)
        keys = sorted({key for row in self.metric_rows for key in row})
        metrics_mx = {}
        for key in keys:
            values = [row.get(key) for row in self.metric_rows]
            if any(value is None for value in values):
                values = [float("nan") if value is None else value for value in values]
                metrics_mx[key] = xp.array(values, dtype=xp.float32)
            else:
                metrics_mx[key] = xp.array(values)
        eval(*metrics_mx.values())
        xp.savez_compressed(path, **metrics_mx)

    def _validate_fit_inputs(self, train_data_loader: DataLoader) -> None:
        if self.config.max_steps is not None:
            return

        try:
            steps_per_epoch = len(train_data_loader)
        except TypeError as exc:
            raise ValueError(
                "Infinite training loaders require max_steps. "
                "Set max_steps or give the dataset examples_per_epoch."
            ) from exc

        if steps_per_epoch <= 0:
            raise ValueError("train_data_loader must yield at least one batch.")

        if self.checkpoint:
            loaded_steps_per_epoch = self.checkpoint.get("steps_per_epoch")
            if loaded_steps_per_epoch != steps_per_epoch:
                raise ValueError(
                    "Cannot resume epoch-mode checkpoint with changed steps_per_epoch."
                )

    def _log_fit_start(self, plan: TrainingPlan) -> None:
        target_label = "steps" if not plan.by_epoch else "epochs"
        payload = {
            "mode": target_label,
            "target_step": plan.target_step,
            "report_every_steps": plan.report_every_steps,
            "global_batch_size": self.config.global_batch_size,
            "micro_batch_size": self.config.micro_batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "target_epochs": (
                cast(int, self.config.max_epochs) if plan.by_epoch else None
            ),
            "steps_per_epoch": plan.steps_per_epoch,
        }
        logger.info("Fit plan:\n%s", _pformat_log_dict(payload))

    def _load_model_and_optimizer(
        self,
        model_class: type[nn.Module],
        optimizer_class: type[optim.Optimizer],
        model_kwargs: Dict[str, Any],
        optimizer_kwargs: Dict[str, Any],
        resume_epoch: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
    ) -> Tuple[Any, Any, dict]:
        """Loads model & optimizer states from checkpoint if a checkpoint is requested.

        `resume_epoch` is only a filename lookup hint. Once a checkpoint is loaded,
        runtime progress comes from checkpoint metadata such as `step_count`.
        If no checkpoint is requested, initializes from scratch.

        Note on configs in checkpoint:
            When we load from checkpoint, the model and optimizer configs are restored from the
            checkpoint and cannot be modified because they are inherently tied to the model and
            optimizer state (e.g. hidden_size cannot be changed because the model artifact was
            created using the previous checkpoint's hidden_size)
            The other configs (e.g. max_eval_steps) can be changed, and can just be read from the
            self.config attribute instead of checkpoint["hyperparams"].

        Args:
            model_class (type[nn.Module]): Class object for the model.
            optimizer_class (type[optim.Optimizer]): Class object for the optimizer.
            model_kwargs (Dict[str, Any]): Keyword arguments for model initialization.
            optimizer_kwargs (Dict[str, Any]): Keyword arguments for optimizer initialization.
            resume_epoch (Optional[int]): Optional checkpoint filename hint used when
                `checkpoint_path` is not provided.
            checkpoint_path (Optional[str]): If provided, path of the checkpoint file.

        Returns:
            Tuple[Any, Any, dict]: A tuple containing:
                - model (nn.Module): The loaded or newly instantiated model.
                - optimizer (optim.Optimizer): The loaded or newly instantiated optimizer.
                - checkpoint (dict): Dictionary of checkpoint data if loaded, otherwise empty.
        """

        if resume_epoch is not None or checkpoint_path is not None:
            # Look for checkpoint files
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
            loaded_step_count = int(loaded_ckpt.get("step_count", 0))
            if (
                self.config.max_steps is not None
                and loaded_step_count > self.config.max_steps
            ):
                raise ValueError("Checkpoint step exceeds max_steps.")

            if self.config.max_steps is None:
                loaded_steps_per_epoch = loaded_ckpt.get("steps_per_epoch")
                if loaded_steps_per_epoch is None:
                    raise ValueError(
                        "Epoch-mode resume requires steps_per_epoch metadata in the checkpoint."
                    )
                if loaded_step_count % int(loaded_steps_per_epoch) != 0:
                    raise ValueError(
                        "Epoch-mode resume requires an epoch-boundary checkpoint. "
                        "Store dataloader/sampler state if mid-epoch resume is needed."
                    )

            # If your checkpoint has hyperparams or constructor kwargs,
            # you can either override model_kwargs or do a partial merge:
            #  e.g. model_kwargs.update(loaded_ckpt["hyperparams"].get("model_kwargs", {}))
            model_init_kwargs = loaded_ckpt["model_init_kwargs"]
            optimizer_init_kwargs = loaded_ckpt["optimizer_init_kwargs"]

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

    Examples:
        The following example demonstrates how to instantiate and run a simple
        supervised training loop using the SimpleTrainer.

        >>> from autograd import nn, optim
        >>> from autograd.tools.config_schema import GenericTrainingConfig
        >>> import numpy as np
        >>>
        >>> # Define a dummy model.
        >>> class DummyModel(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.linear = nn.Linear(10, 2)
        ...     def forward(self, x):
        ...         return self.linear(x)
        >>>
        >>> # Instantiate model, optimizer, and loss function.
        >>> model_cls = DummyModel
        >>> optimizer_cls = optim.SGD
        >>> loss_fn = nn.MSELoss()  # or any other loss function
        >>>
        >>> # Create a dummy training configuration.
        >>> config = GenericTrainingConfig(
        ...     max_epochs=10,
        ...     global_batch_size=1,
        ...     micro_batch_size=1,
        ...     training_run_name='dummy_run',
        ...     model_kwargs={},
        ...     optimizer_kwargs={},
        ...     checkpoint_freq=1,
        ...     resume_epoch=None
        ... )
        >>>
        >>> # Instantiate the trainer.
        >>> trainer = SimpleTrainer(model_cls, optimizer_cls, loss_fn, config, output_type='softmax')
        >>>
        >>> # Create DataLoader instances for train/validation data.
        >>> train_loader = ...
        >>> val_loader = ...
        >>>
        >>> # Run the training loop.
        >>> trainer.fit(train_loader, val_loader)
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

    def _compute_loss(self, batch_data) -> Tensor:
        batch_X, batch_y = batch_data
        y_pred = self.model(batch_X)
        return self.loss_fn(y_pred, batch_y)

    def _evaluate(self, val_data_loader) -> TrainingState:
        state = TrainingState()
        sample_pairs: list[tuple[Any, Any]] = []
        for batch_data in val_data_loader:
            if (
                self.config.max_eval_steps is not None
                and state.eval_loss_batches >= self.config.max_eval_steps
            ):
                break
            batch_X, batch_y = batch_data
            y_pred = self.model(batch_X)
            loss = self.loss_fn(y_pred, batch_y)
            state.record_eval_loss(loss)

            if self.problem_type == "classification":
                y_pred_proc, y_true_proc = self._post_process_classification(
                    y_pred.data, batch_y
                )
                y_pred_flat = xp.array(y_pred_proc).reshape(-1)
                y_true_flat = xp.array(y_true_proc, dtype=xp.int32).reshape(-1)
                correct = float(
                    xp.to_scalar(xp.sum(xp.array(y_pred_flat == y_true_flat)))
                )
                state.record_eval_metric(
                    "accuracy",
                    numerator=correct,
                    denominator=float(y_true_flat.size),
                )

                if self.sample_predictions and len(sample_pairs) < 5:
                    sample_count = min(5 - len(sample_pairs), len(y_pred_proc))
                    for i in range(sample_count):
                        sample_pairs.append((y_pred_proc[i], y_true_proc[i]))
            else:
                diff = xp.array(y_pred.data) - xp.array(batch_y)
                squared_error_sum = float(xp.to_scalar(xp.sum(diff**2)))
                state.record_eval_metric(
                    "mse",
                    numerator=squared_error_sum,
                    denominator=float(diff.size),
                )

        if state.eval_loss_batches == 0:
            raise ValueError("val_data_loader yielded no batches.")

        if self.problem_type == "classification":
            if sample_pairs:
                sample_lines = [
                    f"Predicted: {pred}\nActual: {target}"
                    for pred, target in sample_pairs
                ]
                logger.info(
                    "Sample Predictions on Validation Batch:\n"
                    + "\n".join(sample_lines)
                )
        return state

    def _post_process_classification(
        self, y_pred: Array, y_true: Array
    ) -> Tuple[Array, Array]:
        """Post-processes outputs for classification tasks.

        Args:
            y_pred: Predicted logits or probabilities as a backend array.
            y_true: Ground truth labels as a backend array.

        Returns:
            Tuple[Array, Array]: Processed predictions and targets
                (e.g., for use in accuracy calculation).
        """
        if self.output_type in ["logits", "softmax"]:
            return xp.argmax(y_pred, axis=-1), y_true
        elif self.output_type == "sigmoid":
            return (y_pred >= 0.5).astype(xp.int32), (y_true >= 0.5).astype(xp.int32)
        else:
            return y_pred, y_true


class LLMTrainer(AbstractTrainer):
    """Trainer specialized for language modeling or next-token prediction tasks.

    The trainer is task-specific to language modeling but architecture-agnostic
    about batch shape. It delegates batch interpretation to `forward_fn`, which
    lets decoder-only and encoder-decoder models consume different batch objects
    while sharing the same training loop.

    Examples:
        The following example demonstrates how to instantiate and run a training loop
        for a language modeling task using the LLMTrainer.

        >>> from autograd import nn, optim
        >>> from autograd.tools.config_schema import TransformerTrainingConfig
        >>> import numpy as np
        >>>
        >>> # Define a dummy language model.
        >>> class DummyLLM(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.embedding = nn.Embedding(1000, 64)
        ...     def forward(self, x):
        ...         return self.embedding(x)
        >>>
        >>> # Instantiate model, optimizer, and loss function.
        >>> model_cls = DummyLLM
        >>> optimizer_cls = optim.Adam
        >>> loss_fn = nn.CrossEntropyLoss()  # or any suitable loss function for language modeling
        >>>
        >>> from autograd.data.types import CausalLMBatch
        >>> # Define a dummy forward function that complies with the expected interface.
        >>> class DummyForwardFn(nn.AbstractLLMForwardFn):
        ...     def train(self, model, batch_data):
        ...         return model(batch_data.input_ids)
        ...     def sample(self, model, batch_data):
        ...         return model(batch_data)
        >>>
        >>> forward_fn = DummyForwardFn()
        >>>
        >>> # Create a dummy training configuration.
        >>> config = TransformerTrainingConfig(
        ...     max_epochs=10,
        ...     global_batch_size=1,
        ...     micro_batch_size=1,
        ...     training_run_name='llm_dummy_run',
        ...     model_kwargs={},
        ...     optimizer_kwargs={},
        ...     checkpoint_freq=1,
        ...     resume_epoch=None,
        ...     label_smoothing=0.1,
        ...     teacher_forcing=False,
        ...     eval_start_string='Hello',
        ...     eval_top_k=5
        ... )
        >>>
        >>> # Instantiate the trainer.
        >>> trainer = LLMTrainer(model_cls, optimizer_cls, loss_fn, forward_fn, config)
        >>>
        >>> # Create DataLoader instances for language modeling.
        >>> dummy_batch = CausalLMBatch(
        ...     input_ids=np.random.randint(0, 1000, (32, 10)),
        ...     labels=np.random.randint(0, 1000, (32, 10)),
        ... )
        >>> train_loader = ...
        >>> val_loader = ...
        >>>
        >>> # Run the training loop.
        >>> trainer.fit(train_loader, val_loader)
    """

    def __init__(
        self,
        model_cls: type[nn.Module],
        optimizer_cls: type[optim.Optimizer],
        loss_fn: Callable,
        forward_fn: nn.AbstractLLMForwardFn,
        config: TransformerTrainingConfig,
        eval_callbacks: Optional[
            Sequence[
                Callable[
                    [
                        nn.Module,
                        nn.AbstractLLMForwardFn,
                        Any,
                        TransformerTrainingConfig,
                    ],
                    None,
                ]
            ]
        ] = None,
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
            eval_callbacks: Optional callbacks run after evaluation using
                `(model, forward_fn, val_data_loader, config)`.
            **kwargs: Additional arguments passed to the parent trainer.
        """
        super().__init__(model_cls, optimizer_cls, loss_fn, config=config, **kwargs)
        self.config: TransformerTrainingConfig = cast(
            TransformerTrainingConfig, self.config
        )
        self.forward_fn = forward_fn
        self.eval_callbacks = list(eval_callbacks or [])

    def _compute_loss(self, batch_data) -> Tensor:
        logits = self.forward_fn.train(self.model, batch_data)
        # TODO: Decide whether validation should use the same label_smoothing
        # setting as training, or intentionally report unsmoothed cross-entropy.
        return self.loss_fn(
            logits,
            batch_data.labels,
            label_smoothing=self.config.label_smoothing,
        )

    def _evaluate(self, val_data_loader) -> TrainingState:
        """Evaluates the model on the validation data loader for language modeling.

        Args:
            val_data_loader: Validation data loader providing LM batches.

        Returns:
            EvalResult: The average validation loss and derived metrics.
        """
        state = TrainingState()
        for batch_data in tqdm(val_data_loader, desc="Evaluation", leave=False):
            if (
                self.config.max_eval_steps is not None
                and state.eval_loss_batches >= self.config.max_eval_steps
            ):
                break
            loss = self._compute_loss(batch_data)
            state.record_eval_loss(loss)
        if state.eval_loss_batches == 0:
            raise ValueError("val_data_loader yielded no batches.")

        for callback in self.eval_callbacks:
            callback(self.model, self.forward_fn, val_data_loader, self.config)
        return state
