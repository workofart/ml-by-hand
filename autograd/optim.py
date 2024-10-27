import numpy as np

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
        # print("Gradient norm", sum([np.linalg.norm(v.grad) for k, module in self.model_parameters.items() for _, v in module.items() ]))
        for k, module in self.model_parameters.items():
            for param_name, param in module.items():
                param.data -= self.lr * param.grad

    