import numpy as np
import logging
from autograd.tensor import Tensor
from collections import defaultdict
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)


class Optimizer:
    """
    Base Optimizer Class

    Below is a sample API usage for this class:
    optimizer = optim.Optimizer(model.parameters(), lr=0.01)
    for input, target in dataset:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
    """

    def __init__(
        self, model_parameters: Dict[str, Tensor], lr: float, **kwargs: Any
    ) -> None:
        self._states: Dict[Any, Any] = defaultdict(dict)
        self._hyperparams: Dict[str, Any] = {}

        # We assume this is a flattened named parameter dict
        # passed by calling Optimizer(model.parameters())
        self.model_parameters = model_parameters
        self._hyperparams["lr"] = lr

        # Store additional hyperparams from kwargs:
        for k, v in kwargs.items():
            self._hyperparams[k] = v

    @property
    def lr(self):
        return self._hyperparams["lr"]

    @lr.setter
    def lr(self, value):
        self._hyperparams["lr"] = value

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
                123: {'m': ..., 'v': ..., ...},  # keyed by id(param)
                456: {...},
            }
          }
        """
        return {
            "hyperparams": dict(self._hyperparams),
            "states": {
                key: dict(value)  # shallow copy of each sub-dict
                for key, value in self._states.items()
            },
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the optimizer state from a checkpoint.
        """
        # Restore hyperparams:
        for k, v in state_dict["hyperparams"].items():
            self._hyperparams[k] = v

        # Now rebuild the internal _states dict
        self._states.clear()
        for state_key, name_dict in state_dict["states"].items():
            for param_name, val in name_dict.items():
                # Only load if param_name is still in our model
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
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent Optimizer
    """

    def __init__(self, model_parameters: Any, lr: float, **kwargs: Any) -> None:
        super(SGD, self).__init__(model_parameters, lr=lr, **kwargs)

    def step(self) -> None:
        def update_fn(param: Any) -> None:
            param.data -= self.lr * param.grad.data

        self._recursive_param_op(self.model_parameters, update_fn)


class Adam(Optimizer):
    """
    Adam Optimizer
    Stochastic gradient descent with first and second order momentum
    Paper: https://arxiv.org/abs/1412.6980
    """

    def __init__(
        self,
        model_parameters: Any,
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-7,
        **kwargs: Any,
    ) -> None:
        super(Adam, self).__init__(model_parameters, lr=lr, **kwargs)
        self._hyperparams["beta1"] = beta1
        self._hyperparams["beta2"] = beta2
        self._hyperparams["epsilon"] = epsilon

        self._states["m"] = defaultdict(float)
        self._states["v"] = defaultdict(float)
        self._states["timestep"] = defaultdict(int)

    def step(self):
        beta1 = self._hyperparams["beta1"]
        beta2 = self._hyperparams["beta2"]
        epsilon = self._hyperparams["epsilon"]

        # Iterate over all parameters in the model by *name*
        for name, param in self.model_parameters.items():
            if param.grad is None:
                continue

            # Access momentum from the sub-dicts
            self._states["timestep"][name] += 1
            t = self._states["timestep"][name]

            m_old = self._states["m"][name]
            v_old = self._states["v"][name]

            grad = param.grad.data  # or param.grad if it's np array
            new_m = beta1 * m_old + (1 - beta1) * grad
            new_v = beta2 * v_old + (1 - beta2) * (grad**2)

            # Store them back
            self._states["m"][name] = new_m
            self._states["v"][name] = new_v

            # bias correction
            m_hat = new_m / (1 - beta1**t)
            v_hat = new_v / (1 - beta2**t)

            # Update param
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + epsilon)
