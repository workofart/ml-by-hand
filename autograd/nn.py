import numpy as np
from .tensor import Tensor
import logging

logger = logging.getLogger(__name__)


class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self._is_training = None

    def zero_grad(self):
        for p in self._parameters:
            p.grad = 0

    def forward(self, x):
        raise NotImplementedError

    def __call__(self, x):
        """
        Sometimes people like to call model = Module() then call model(x)
        as a forward pass. So this is an alias.
        """
        return self.forward(x)

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, Tensor):
            self._parameters[name] = value
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name):
        return self._modules[name]

    @property
    def parameters(self):
        params = self._parameters.copy()

        for k, module in self._modules.items():
            params.update({k: module.parameters})

        return params

    def train(self):
        for module in self._modules.values():
            module.train()
        self._is_training = True

    def eval(self):
        for module in self._modules.values():
            module.eval()
        self._is_training = False


class Linear(Module):
    def __init__(self, input_size, output_size, **kwargs):
        super().__init__(**kwargs)

        # weight is a matrix of shape (input_size, output_size)
        # Xavier Normal Initialization
        # https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        self._parameters["weight"] = Tensor(
            np.random.uniform(
                low=-np.sqrt(6.0 / (input_size + output_size)),
                high=np.sqrt(6.0 / (input_size + output_size)),
                size=(input_size, output_size),
            )
        )

        # bias is always 1-dimensional
        self._parameters["bias"] = Tensor(np.random.rand(output_size))

    def forward(self, x) -> Tensor:
        if not isinstance(x, Tensor):
            x = Tensor(x)

        logger.debug(f"{x.data.shape=}")
        logger.debug(f"Linear forward {self._parameters['weight'].data.shape=}")

        # this is just a linear transformation (dot matrix multiplication)
        return x @ self._parameters["weight"] + self._parameters["bias"]


class BatchNorm(Module):
    """
    Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

    This layer normalizes the input tensor by subtracting the batch mean and dividing by the batch standard deviation.

    Paper: http://arxiv.org/abs/1502.03167
    """

    def __init__(self, input_size, momentum=0.1, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)

        self.momentum = momentum  # used in running mean and variance calculation
        self.epsilon = epsilon  # small constant for numeric stability

        # Running stats (used for inference)
        self.running_mean = np.zeros(input_size)
        self.running_var = np.ones(input_size)

        # gamma and beta are learnable parameters
        # gamma is responsible for scaling the normalized input
        # beta is responsible for shifting the normalized input
        self._parameters["weight"] = Tensor(np.ones((1, input_size)))
        self._parameters["bias"] = Tensor(np.zeros((1, input_size)))

    def forward(self, x: Tensor) -> Tensor:
        # Determine reduction axes based on input dimensions
        # reduce_axes = tuple(i for i in range(x.data.ndim) if i != 1)

        if self._is_training:
            batch_mean = x.data.mean(axis=0)
            var = np.sum((x.data - batch_mean) ** 2, axis=0)
            biased_batch_var = var / (x.data.shape[0])

            # Use unbiased variance (N-1) for running statistics
            unbiased_batch_var = var / (x.data.shape[0] - 1)

            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * batch_mean
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * unbiased_batch_var
            x = (x - batch_mean) / np.sqrt(biased_batch_var + self.epsilon)

        else:
            x = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        return x * self._parameters["weight"] + self._parameters["bias"]
