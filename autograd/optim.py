import numpy as np
import logging
from collections import defaultdict

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

    def __init__(self, model_parameters, lr, **kwargs) -> None:
        self.model_parameters = model_parameters
        self.lr = lr

    def zero_grad(self):
        """
        Set the gradients of all optimized tensors to zero.
        """
        for k, module in self.model_parameters.items():
            for param_name, param in module.items():
                param.grad = np.zeros_like(param.data)

    def step(self):
        """
        Performs a single optimization step.
        """
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent Optimizer
    """

    def __init__(self, model_parameters, lr, **kwargs) -> None:
        super(SGD, self).__init__(model_parameters, lr, **kwargs)

    def step(self):
        logger.debug(
            f"Gradient norm {sum([np.linalg.norm(v.grad) for k, module in self.model_parameters.items() for _, v in module.items() ])}"
        )
        for k, module in self.model_parameters.items():
            for param_name, param in module.items():
                param.data -= self.lr * param.grad


class Adam(Optimizer):
    """
    Adam Optimizer
    Stochastic gradient descent with first and second order momentum
    Paper: https://arxiv.org/abs/1412.6980
    """

    def __init__(
        self, model_parameters, lr, beta1=0.9, beta2=0.999, epsilon=1e-7, **kwargs
    ) -> None:
        super(Adam, self).__init__(model_parameters, lr, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon  # for numeric stability

        # Note that we are creating new state attributes to store these
        # These notations are based on the same notations in the paper linked above
        self.m = defaultdict(float)  # first momentum estimate
        self.v = defaultdict(float)  # second momentum estimate
        self.timestep = (
            0  # to keep track of the timestep, this will adapt our learning rate
        )

    def step(self):
        self.timestep += 1
        for k, module in self.model_parameters.items():
            for param_name, param in module.items():
                if param.grad is None:
                    continue

                param_id = id(
                    param
                )  # to avoid cases where the same parameter name is shared across different modules
                grad = param.grad

                # update first order momentum
                self.m[param_id] = (
                    self.beta1 * self.m[param_id] + (1 - self.beta1) * grad
                )
                # update second order momentum
                self.v[param_id] = self.beta2 * self.v[param_id] + (1 - self.beta2) * (
                    grad**2
                )

                # bias correction
                m_hat = self.m[param_id] / (1 - self.beta1**self.timestep)
                v_hat = self.v[param_id] / (1 - self.beta2**self.timestep)

                # update parameters
                param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
