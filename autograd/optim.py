import logging
from collections import defaultdict
from typing import Any, Callable, Dict, Optional

try:
    # drop-in replacement for numpy for GPU acceleration
    import cupy as np  # type: ignore
except ImportError:
    import numpy as np

from autograd.tensor import Tensor

logger = logging.getLogger(__name__)


class LRScheduler:
    """
    Interface for a learning rate scheduler.
    Subclasses should implement the __call__ method.
    """

    def __call__(self, step: int, initial_lr: float, current_lr: float) -> float:
        raise NotImplementedError


class CosineScheduler(LRScheduler):
    """
    Cosine learning rate scheduler with warmup.
    Implements Section 3 in "SGDR: Stochastic Gradient Descent with Warm Restarts"
    Paper: https://arxiv.org/abs/1608.03983
    """

    def __init__(
        self, warmup_steps: int = 100, lr_decay_iters: int = 5000, min_lr: float = 1e-4
    ):
        self.warmup_steps = warmup_steps
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr

    def __call__(self, step: int, initial_lr: float, current_lr: float) -> float:
        # The initial_lr is the min learning rate in the original paper
        if step < self.warmup_steps:
            return initial_lr * (step + 1) / (self.warmup_steps + 1)
        if step > self.lr_decay_iters:
            return self.min_lr
        decay_ratio = (step - self.warmup_steps) / (
            self.lr_decay_iters - self.warmup_steps
        )
        coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
        return self.min_lr + coeff * (initial_lr - self.min_lr)


class Optimizer:
    """
    Base Optimizer Class

    Usage:
      scheduler = CosineScheduler(warmup_steps=100, lr_decay_iters=5000)
      optimizer = Optimizer(model.parameters(), lr=0.01, lr_scheduler=scheduler)
      for input, target in dataset:
          optimizer.zero_grad()
          output = model(input)
          loss = loss_fn(output, target)
          loss.backward()
          optimizer.step()
    """

    def __init__(
        self,
        model_parameters: Dict[str, Tensor],
        lr: float,
        lr_scheduler: Optional[LRScheduler] = None,
        **kwargs: Any,
    ) -> None:
        self._states: Dict[str, Any] = defaultdict(dict)
        self._hyperparams: Dict[str, Any] = {}
        # We assume this is a flattened named parameter dict
        # passed by calling Optimizer(model.parameters())
        self.model_parameters = model_parameters
        self._hyperparams["lr"] = lr
        self.initial_lr = lr
        self.lr_scheduler = lr_scheduler  # optional

        # Store the global step within _states.
        self._states["timestep"] = 0

        # Store additional hyperparams from kwargs:
        for k, v in kwargs.items():
            self._hyperparams[k] = v

    @property
    def lr(self):
        return self._hyperparams["lr"]

    @lr.setter
    def lr(self, value):
        self._hyperparams["lr"] = value

    @property
    def timestep(self) -> int:
        return self._states.get("timestep", 0)

    @timestep.setter
    def timestep(self, value: int):
        self._states["timestep"] = value

    def update_lr(self) -> None:
        """
        Update the learning rate using the provided scheduler and the global step.
        """
        if self.lr_scheduler is not None:
            self.lr = self.lr_scheduler(self.timestep, self.initial_lr, self.lr)

    def _recursive_param_op(
        self, params: Any, update_fn: Callable[[Any], None]
    ) -> None:
        # 1) If params is a dict
        if isinstance(params, dict):
            for _, v in params.items():
                self._recursive_param_op(v, update_fn)

        # 2) If params is a list or tuple
        elif isinstance(params, (list, tuple)):
            for p in params:
                self._recursive_param_op(p, update_fn)

        # 3) If params is a single parameter with a .grad
        elif hasattr(params, "grad"):
            update_fn(params)

    def _clip_grad_norm(self, max_norm: float, norm_type: float = 2.0) -> None:
        r"""
        Scales the gradients of all parameters (in-place) so that the norm of the
        gradients is at most `max_norm`.
        Implements Section 10.11.1 "Clipping Gradients" in Deep Learning Book by Goodfellow et al.

        $$\frac{\text{max\_norm} \cdot g}{\|g\|_n}$$
        where n is the nth norm

        Args:
           max_norm (float): The maximum norm of the gradients.
           norm_type (float): The type of norm to use. Default is 2 (Euclidean norm).
        """
        # Compute the global norm of all gradients
        total_norm = 0.0

        for param in self.model_parameters.values():
            if param.grad is not None:
                grad_data = param.grad.data
                total_norm += (np.abs(grad_data) ** norm_type).sum()

        # Take the appropriate root of the total_norm
        total_norm = total_norm ** (1.0 / norm_type)

        # If total_norm is greater than max_norm, scale all gradients
        if total_norm > max_norm:
            scale_factor = max_norm / (
                total_norm + 1e-10
            )  # add small value for numerical stability
            for param in self.model_parameters.values():
                if param.grad is not None:
                    param.grad.data *= scale_factor

    def zero_grad(self) -> None:
        """Set the gradients of all optimized tensors to zero."""

        def update_fn(x: Any) -> None:
            x.grad = None

        self._recursive_param_op(self.model_parameters, update_fn)

    def state_dict(self) -> Dict[str, Any]:
        """
        Return a dict of the Optimizer's state, for checkpointing. For example:
          {
            'hyperparams': {
                'lr': 0.01,
                ... any other hyperparams ...
            },
            'states': {
                "module1.weight": {'m': ..., 'v': ..., ...},
                "module1.bias": {...},
            }
          }
        """
        return {
            "hyperparams": dict(self._hyperparams),
            "states": {
                key: dict(value)
                if isinstance(value, dict)
                else value  # shallow copy of each sub-dict
                for key, value in self._states.items()
            },
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the optimizer state from a checkpoint. This performs an in-place
        update of the optimizer's internal state and hyperparameter.

        Args:
           state_dict(Dict[str, Any]): The state dict read from a checkpoint
        """
        # Restore hyperparams:
        for k, v in state_dict["hyperparams"].items():
            self._hyperparams[k] = v

        # Now rebuild the internal _states dict
        self._states.clear()
        for state_key, name_dict in state_dict["states"].items():
            # For timestep, we directly set the value.
            if state_key == "timestep":
                self._states["timestep"] = name_dict
            else:
                for param_name, val in name_dict.items():
                    if param_name in self.model_parameters:
                        self._states[state_key][param_name] = val
                    else:
                        logger.warning(
                            f"Skipping state for param {param_name} not found in current model"
                        )

    def step(self) -> None:
        """
        Performs a single optimization step.
        """
        self.timestep += 1
        self.update_lr()


class SGD(Optimizer):
    """
    Stochastic Gradient Descent Optimizer
    """

    def __init__(self, model_parameters: Any, lr: float, **kwargs: Any) -> None:
        super(SGD, self).__init__(model_parameters, lr=lr, **kwargs)

    def step(self) -> None:
        super().step()

        def update_fn(param: Any) -> None:
            param.data -= self.lr * param.grad.data

        if "max_grad_norm" in self._hyperparams:
            self._clip_grad_norm(self._hyperparams["max_grad_norm"], norm_type=2.0)

        self._recursive_param_op(self.model_parameters, update_fn)


class Adam(Optimizer):
    """
    Adam Optimizer
    Stochastic gradient descent with first and second order momentum
    Paper: https://arxiv.org/abs/1412.6980

    The `weight_decay` parameter is part of the AdamW implementation from the paper
    Decoupled Weight Decay Regularization
    Paper: https://arxiv.org/abs/1711.05101

    We have decoupled the Adam-step and the weight-decay step.
    When `weight_decay` is set to 0, AdamW is equivalent to Adam
    """

    def __init__(
        self,
        model_parameters: Any,
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-7,
        weight_decay: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super(Adam, self).__init__(model_parameters, lr=lr, **kwargs)
        # These notations are based on the same notations in the paper linked above
        self._hyperparams["beta1"] = beta1
        self._hyperparams["beta2"] = beta2
        self._hyperparams["epsilon"] = epsilon
        self._hyperparams["weight_decay"] = weight_decay

        # Internal state
        self._states["m"] = defaultdict(float)  # first momentum estimate
        self._states["v"] = defaultdict(float)  # second momentum estimate

    def step(self):
        super().step()

        if "max_grad_norm" in self._hyperparams:
            self._clip_grad_norm(self._hyperparams["max_grad_norm"], norm_type=2.0)

        beta1 = self._hyperparams["beta1"]
        beta2 = self._hyperparams["beta2"]
        epsilon = self._hyperparams["epsilon"]
        weight_decay = self._hyperparams["weight_decay"]

        # Iterate over all parameters in the model by *name*
        for name, param in self.model_parameters.items():
            if param.grad is None:
                continue

            m_old = self._states["m"][name]
            v_old = self._states["v"][name]

            grad = param.grad.data  # or param.grad if it's np array
            new_m = beta1 * m_old + (1 - beta1) * grad  # update first order momentum
            new_v = beta2 * v_old + (1 - beta2) * (
                grad**2
            )  # update second order momentum

            # Store them back
            self._states["m"][name] = new_m
            self._states["v"][name] = new_v

            # Use the global step for bias correction.
            m_hat = new_m / (1 - beta1**self.timestep)
            v_hat = new_v / (1 - beta2**self.timestep)

            # Weight decay step (decoupled)
            if weight_decay > 0.0:
                param.data = param.data - self.lr * weight_decay * param.data
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + epsilon)
