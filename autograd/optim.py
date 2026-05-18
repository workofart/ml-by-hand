import logging
import math
from abc import abstractmethod
from collections import defaultdict
from functools import lru_cache
from typing import Any, Dict, Optional

from autograd.backend import IS_CUPY, LOW_PRECISION_FLOAT_DTYPES, Array, materialize, xp
from autograd.distributed import allreduce_grads, broadcast_parameters
from autograd.tensor import Tensor

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _cupy_adamw_bf16_master_kernel() -> Any:
    """AdamW kernel for bf16 params with an fp32 master copy.

    All math happens in fp32: read the master, apply weight decay + Adam
    update, write the master back, and refresh the bf16 working copy. This
    prevents the bf16-ULP-swallowing bug where small updates to a parameter
    near magnitude 1.0 round to zero on store (`2^-7 ~= 0.0078` distance
    between adjacent bf16 values near 1.0, vs. typical Adam updates of
    `lr ~= 6e-4`).
    """
    module = xp.RawModule(
        code=r"""
        #include <cuda_bf16.h>

        extern "C" __global__ void adamw_bf16_master(
            __nv_bfloat16* param,
            float* master,
            const float* grad,
            float* m,
            float* v,
            const long long size,
            const float lr,
            const float beta1,
            const float beta2,
            const float one_minus_beta1,
            const float one_minus_beta2,
            const float bias_correction1,
            const float bias_correction2,
            const float epsilon,
            const float weight_decay
        ) {
            long long idx = blockIdx.x * blockDim.x + threadIdx.x;
            long long stride = blockDim.x * gridDim.x;
            for (; idx < size; idx += stride) {
                float g = grad[idx];
                float new_m = beta1 * m[idx] + one_minus_beta1 * g;
                float new_v = beta2 * v[idx] + one_minus_beta2 * g * g;
                m[idx] = new_m;
                v[idx] = new_v;
                float p = master[idx];
                p = p - lr * weight_decay * p;
                p = p - lr * (new_m / bias_correction1) /
                    (sqrtf(new_v / bias_correction2) + epsilon);
                master[idx] = p;
                param[idx] = __float2bfloat16(p);
            }
        }
        """,
        options=("--std=c++11",),
    )
    return module.get_function("adamw_bf16_master")


class LRScheduler:
    """
    Interface for a learning rate scheduler.

    Subclasses should implement the __call__ method.
    """

    @abstractmethod
    def __call__(self, step: int, initial_lr: float, current_lr: float) -> float:
        """
        Compute the updated learning rate.

        Args:
            step (int): The current global step.
            initial_lr (float): The initial learning rate.
            current_lr (float): The current learning rate.

        Returns:
            float: The updated learning rate.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError


class CosineScheduler(LRScheduler):
    """
    Cosine learning rate scheduler with warmup.

    Implements Section 3 in "SGDR: Stochastic Gradient Descent with Warm Restarts".
    Paper: https://arxiv.org/abs/1608.03983
    """

    def __init__(
        self,
        warmup_steps: int = 100,
        lr_decay_iters: int = 200,
        min_lr: float = 1e-4,
        **kwargs,
    ):
        """
        Initialize the CosineScheduler.

        Args:
            warmup_steps (int, optional): Number of warmup steps. Defaults to 100.
            lr_decay_iters (int, optional): Step at which learning rate decay ends. Defaults to 200.
            min_lr (float, optional): Minimum learning rate. Defaults to 1e-4.
        """
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if lr_decay_iters <= warmup_steps:
            raise ValueError(
                "CosineScheduler requires lr_decay_iters > warmup_steps, "
                f"got warmup_steps={warmup_steps}, lr_decay_iters={lr_decay_iters}"
            )

        self.warmup_steps = warmup_steps
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr

    def __call__(self, step: int, initial_lr: float, current_lr: float) -> float:
        """
        Compute the learning rate for a given step using cosine decay with warmup.

        Note:
            The initial_lr is treated as the maximum learning rate during warmup,
            and then decayed cosine-wise to min_lr.

        Args:
            step (int): The current global step.
            initial_lr (float): The initial (maximum) learning rate.
            current_lr (float): The current learning rate (unused in this scheduler).

        Returns:
            float: The updated learning rate.
        """
        # The initial_lr is the min learning rate in the original paper
        if step < self.warmup_steps:
            return initial_lr * (step + 1) / (self.warmup_steps + 1)
        if step > self.lr_decay_iters:
            return self.min_lr
        decay_ratio = (step - self.warmup_steps) / (
            self.lr_decay_iters - self.warmup_steps
        )
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (initial_lr - self.min_lr)


class Optimizer:
    """
    Base Optimizer Class.

    Usage Example:

    .. code-block:: python

        optimizer = Optimizer(model.parameters(), lr=0.01, lr_scheduler_kwargs={
            "lr_scheduler_cls": CosineScheduler,
            "warmup_steps": 100,
            "lr_decay_iters": 5000,
            "min_lr": 1e-4
        })
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
        lr_scheduler_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Base Optimizer. Usually you wouldn't directly initialize this, but rather initialize a
        specific Optimizer subclass (e.g. Adam)

        Args:
            model_parameters (Dict[str, Tensor]): A flattened dictionary of model parameters.
            lr (float): The initial learning rate.
            lr_scheduler_kwargs (Optional[dict], optional): A dictionary containing the learning rate
                scheduler class under the key 'lr_scheduler_cls' and its initialization parameters.
                Defaults to None.
            **kwargs: Additional hyperparameters.
        """
        self._states: Dict[str, Any] = defaultdict(dict)
        self._hyperparams: Dict[str, Any] = {}
        # We assume this is a flattened named parameter dict
        # passed by calling Optimizer(model.parameters())
        self.model_parameters = model_parameters
        # Mark every owned tensor as a parameter so backward accumulates its
        # gradient in fp32 — the cross-microbatch sum on bf16 would otherwise
        # introduce ~1% rounding noise per accumulation step. The flag is
        # idempotent and only affects subsequent _accumulate_grad calls.
        for param in self.model_parameters.values():
            param._use_fp32_grad_accumulator = True
        self._hyperparams["lr"] = lr
        self.initial_lr = lr
        if lr_scheduler_kwargs:
            scheduler_kwargs = dict(lr_scheduler_kwargs)
            scheduler_spec = scheduler_kwargs.pop("lr_scheduler_cls")

            if isinstance(scheduler_spec, str):
                if scheduler_spec not in globals():
                    raise ValueError(f"Unknown lr scheduler class: {scheduler_spec}")
                lr_scheduler_cls = globals()[scheduler_spec]
            else:
                lr_scheduler_cls = scheduler_spec

            self.lr_scheduler = lr_scheduler_cls(**scheduler_kwargs)
        else:
            self.lr_scheduler = None

        # Store the global step within _states.
        self._states["timestep"] = 0

        # Store additional hyperparams from kwargs:
        for k, v in kwargs.items():
            self._hyperparams[k] = v

        # Belt-and-suspenders parameter sync at construction time so every
        # DDP rank starts from identical weights. broadcast_parameters
        # self-no-ops when world_size == 1, so no guard is needed here.
        broadcast_parameters(self.model_parameters, from_rank=0)

    @property
    def lr(self) -> float:
        """
        Get the current learning rate.

        Returns:
            float: The current learning rate.
        """
        return self._hyperparams["lr"]

    @lr.setter
    def lr(self, value: float) -> None:
        """
        Set a new learning rate.

        Args:
            value (float): The new learning rate.
        """
        self._hyperparams["lr"] = value

    @property
    def timestep(self) -> int:
        """
        Get the current global timestep.

        Returns:
            int: The current timestep.
        """
        return self._states.get("timestep", 0)

    @timestep.setter
    def timestep(self, value: int) -> None:
        """
        Set the global timestep.

        Args:
            value (int): The new timestep value.
        """
        self._states["timestep"] = value

    def update_lr(self) -> None:
        """
        Update the learning rate using the provided scheduler and the global step.
        """
        if self.lr_scheduler is not None:
            self.lr = self.lr_scheduler(self.timestep, self.initial_lr, self.lr)

    def clip_grad_norm(
        self,
        max_norm: float,
        norm_type: float = 2.0,
    ) -> Any:
        r"""
        Scale the gradients of all parameters in-place so that their norm is at most max_norm.

        Implements Section 10.11.1 "Clipping Gradients" in the Deep Learning Book by Goodfellow et al.

        The scaling is done according to:
        $$
        \frac{\text{max\_norm} \cdot g}{\|g\|_n}
        $$
        where $n$ is the norm type.

        Args:
            max_norm (float): The maximum allowed norm of the gradients.
            norm_type (float, optional): The type of norm to use (default is 2, Euclidean norm a.k.a. L2 norm).
        """
        # Compute the global norm of all gradients
        total_norm = xp.asarray(0.0, dtype=xp.float32)
        for param in self.model_parameters.values():
            if param.grad is not None:
                grad_data = param.grad.data
                total_norm += (xp.abs(grad_data) ** norm_type).sum()

        # Take the appropriate root of the total_norm
        total_norm = total_norm ** (1.0 / norm_type)

        # Clamp the scale factor instead of branching to avoid a device/host
        # sync on the common no-clip path. This slightly relaxes strict
        # "only clip if total_norm > max_norm" semantics because the epsilon
        # can shrink norms that are just below max_norm.
        scale_factor = xp.minimum(1.0, max_norm / (total_norm + 1e-10))
        for param in self.model_parameters.values():
            if param.grad is not None:
                param.grad.data *= scale_factor

        # Return the backend scalar lazily. Callers that need to log this value
        # own the device-to-host scalarization.
        return total_norm

    def zero_grad(self) -> None:
        """
        Set the gradients of all optimized tensors to zero.
        """

        for param in self.model_parameters.values():
            param.grad = None

    def scale_gradients(self, scale: Array) -> None:
        """Scale all parameter gradients in-place by ``scale``."""
        for param in self.model_parameters.values():
            if param.grad is not None:
                grad_dtype = param.grad.data.dtype
                param.grad.data *= scale
                if param.grad.data.dtype != grad_dtype:
                    param.grad.data = param.grad.data.astype(grad_dtype)

    def grad_l2_norm(self) -> float:
        """Return the L2 norm of all current parameter gradients."""
        grad_norm = xp.asarray(0.0, dtype=xp.float32)
        for param in self.model_parameters.values():
            if param.grad is not None:
                grad_norm += (param.grad.data**2).sum()
        return float(xp.to_scalar(grad_norm**0.5))

    def gradient_arrays(self) -> Dict[str, Array]:
        """Current accumulated gradient arrays, keyed by parameter name."""
        return {
            name: param.grad.data
            for name, param in self.model_parameters.items()
            if param.grad is not None
        }

    def state_dict(self) -> Dict[str, Any]:
        """
        Return a dictionary representing the optimizer's state for checkpointing.

        The returned dictionary has the following structure:

        .. code-block:: json

            {
                "hyperparams": {
                    "lr": 0.01,
                },
                "states": {
                    "module1.weight": {"m": "...", "v": "..." },
                    "module1.bias": "..."
                }
            }

        Returns:
            Dict[str, Any]: The state dictionary of the optimizer.
        """
        return {
            "hyperparams": dict(self._hyperparams),
            "states": {
                key: dict(value) if isinstance(value, dict) else value
                for key, value in self._states.items()
            },
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the optimizer state from a checkpoint.

        This performs an in-place update of the optimizer's internal state and hyperparameters.

        Args:
            state_dict (Dict[str, Any]): The state dictionary read from a checkpoint.
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
                self._states[state_key] = {}
                for param_name, val in name_dict.items():
                    if param_name in self.model_parameters:
                        self._states[state_key][param_name] = val
                    else:
                        logger.warning(
                            f"Skipping state for param {param_name} not found in current model"
                        )

    def step(self) -> None:
        """
        Perform a single optimization step.
        """
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) Optimizer.
    """

    def __init__(self, model_parameters: Any, lr: float, **kwargs: Any) -> None:
        """
        Initialize the SGD optimizer.

        Args:
            model_parameters (Any): Model parameters to optimize.
            lr (float): The learning rate.
            **kwargs: Additional hyperparameters.
        """
        super(SGD, self).__init__(model_parameters, lr=lr, **kwargs)

    def step(self) -> None:
        """
        Perform a single optimization step using SGD.

        This method updates each parameter by subtracting the product of the learning rate
        and the parameter's gradient.
        """
        allreduce_grads(self.model_parameters)
        self.timestep += 1
        self.update_lr()

        def update_fn(param: Any) -> None:
            if param.grad is None:
                return
            param.data -= self.lr * param.grad.data

        for param in self.model_parameters.values():
            update_fn(param)
        materialize(self.model_parameters, self._states)
        return None


class Adam(Optimizer):
    """
    Adam Optimizer.

    Implements stochastic gradient descent with first and second order momentum.
    Paper: https://arxiv.org/abs/1412.6980

    The `weight_decay` parameter implements decoupled weight decay as described in:
    "Decoupled Weight Decay Regularization" (https://arxiv.org/abs/1711.05101).

    When `weight_decay` is set to 0, AdamW is equivalent to Adam.
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
        """
        Initialize the Adam optimizer.

        Args:
            model_parameters (Any): Model parameters to optimize.
            lr (float): The learning rate.
            beta1 (float, optional): Exponential decay rate for the first moment estimates. Defaults to 0.9.
            beta2 (float, optional): Exponential decay rate for the second moment estimates. Defaults to 0.999.
            epsilon (float, optional): Small constant for numerical stability. Defaults to 1e-7.
            weight_decay (float, optional): Weight decay (L2 penalty) factor. Defaults to 0.0.
            **kwargs: Additional hyperparameters.
        """
        super(Adam, self).__init__(model_parameters, lr=lr, **kwargs)
        # These notations are based on the same notations in the paper linked above
        self._hyperparams["beta1"] = beta1
        self._hyperparams["beta2"] = beta2
        self._hyperparams["epsilon"] = epsilon
        self._hyperparams["weight_decay"] = weight_decay

        # Internal state
        self._states["m"] = defaultdict(float)  # first momentum estimate
        self._states["v"] = defaultdict(float)  # second momentum estimate
        # fp32 master copy for low-precision params. Adam updates are computed
        # in fp32 against the master and the working copy (param.data) is
        # re-derived from it each step.
        self._states["master"] = {}
        for name, param in self.model_parameters.items():
            if param.data.dtype in LOW_PRECISION_FLOAT_DTYPES:
                self._states["master"][name] = param.data.astype(xp.float32)

    def _cupy_bf16_step_param(
        self,
        name: str,
        param: Tensor,
        grad: Array,
        *,
        beta1: float,
        beta2: float,
        epsilon: float,
        weight_decay: float,
    ) -> bool:
        if (
            not IS_CUPY
            or not hasattr(xp, "bfloat16")
            or param.data.dtype != xp.bfloat16
            or not param.data.flags.c_contiguous
        ):
            return False

        # The fused kernel expects an fp32 grad. Tensor._accumulate_grad
        # upgrades low-precision grads to fp32 on intake, so we should rarely
        # need this cast — keep it as a defensive shim for direct .grad
        # assignments (e.g., unit tests).
        if grad.dtype != xp.float32:
            grad = grad.astype(xp.float32)
        if not grad.flags.c_contiguous:
            grad = xp.ascontiguousarray(grad)

        master = self._ensure_master(name, param)
        if master is None or not master.flags.c_contiguous:
            return False

        m_old = self._states["m"][name]
        v_old = self._states["v"][name]
        if not isinstance(m_old, xp.ndarray):
            m_old = xp.zeros(param.data.shape, dtype=xp.float32)
        if not isinstance(v_old, xp.ndarray):
            v_old = xp.zeros(param.data.shape, dtype=xp.float32)
        if (
            m_old.dtype != xp.float32
            or v_old.dtype != xp.float32
            or m_old.shape != param.data.shape
            or v_old.shape != param.data.shape
        ):
            return False

        self._states["m"][name] = m_old
        self._states["v"][name] = v_old
        kernel = _cupy_adamw_bf16_master_kernel()
        threads = 256
        blocks = min((int(param.data.size) + threads - 1) // threads, 1024)
        kernel(
            (blocks,),
            (threads,),
            (
                param.data,
                master,
                grad,
                m_old,
                v_old,
                xp.int64(param.data.size),
                xp.float32(self.lr),
                xp.float32(beta1),
                xp.float32(beta2),
                xp.float32(1.0 - beta1),
                xp.float32(1.0 - beta2),
                xp.float32(1.0 - beta1**self.timestep),
                xp.float32(1.0 - beta2**self.timestep),
                xp.float32(epsilon),
                xp.float32(weight_decay),
            ),
        )
        return True

    def step(self) -> None:
        """
        Perform a single optimization step using the Adam algorithm.

        This method updates the biased first and second order momentum estimates,
        applies bias correction, performs a decoupled weight decay step if specified,
        and updates the parameters accordingly.
        """
        # Average gradients across DDP ranks before Adam consumes them, so
        # the optimizer state (m, v) is computed against the true global
        # mean gradient — equivalent to a single forward+backward over the
        # concatenated global batch. No-op when world_size == 1.
        allreduce_grads(self.model_parameters)
        self.timestep += 1
        self.update_lr()

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

            grad = param.grad.data

            # For mixed precision, keep Adam statistics in fp32 for stability.
            param_dtype = param.data.dtype
            if self._cupy_bf16_step_param(
                name,
                param,
                grad,
                beta1=beta1,
                beta2=beta2,
                epsilon=epsilon,
                weight_decay=weight_decay,
            ):
                continue

            if grad.dtype in LOW_PRECISION_FLOAT_DTYPES:
                grad = grad.astype(xp.float32)
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

            # For low-precision params, do the update against the fp32 master
            # and then re-derive the working copy.
            # For fp32 params (no master), update `param.data` directly.
            master = self._states["master"].get(name)
            if master is not None:
                if weight_decay > 0.0:
                    master = master - self.lr * weight_decay * master
                master -= self.lr * m_hat / (xp.sqrt(v_hat) + epsilon)
                self._states["master"][name] = master
                param.data = master.astype(param_dtype)
            else:
                if weight_decay > 0.0:
                    param.data = param.data - self.lr * weight_decay * param.data
                param.data -= self.lr * m_hat / (xp.sqrt(v_hat) + epsilon)
        materialize(self.model_parameters, self._states)
        return None
