import numpy as np
from .tensor import Tensor

class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}
    
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
        # print(f"Setting attributes {name} = {value}")
        if isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, Tensor):
            self._parameters[name] = value
        else:
            super().__setattr__(name, value)
    
    @property
    def parameters(self):
        params = self._parameters.copy()
        
        for module in self._modules.values():
            params.update(module.parameters)
            
        return params
    
class Linear(Module):
    def __init__(self, input_size, output_size, **kwargs):
        super().__init__(**kwargs)
        
        # weight is a matrix of shape (input_size, output_size)
        self._parameters['weight'] = Tensor(np.random.randn(input_size, output_size))
        
        # bias is always 1-dimensional
        self._parameters['bias'] = Tensor(np.zeros(output_size))
        
    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        
        print("x.data shape:", x.data.shape)
        print("weights shape:", self._parameters['weight'].data.shape)
    
        # this is just a linear transformation (dot matrix multiplication)
        return x @ self._parameters['weight'] + self._parameters['bias']
    
class ReLU:
    """
    Retified Linear Unit (ReLU) activation function.
    ReLU(x) = max(0, x)
    """
    
    def __init__(self):
        self._mask = None
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x: Tensor):
        self._mask = (x.data > 0)
        return Tensor(np.maximum(0.0, x.data))
    
    def backward(self, grad_output):
        return grad_output * self._mask
    
    
class Sigmoid:
    """
    Sigmoid activation function
    """
    def __init__(self) -> None:
        self._mask = None
        
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x: Tensor):
        return Tensor(1 / (1 + np.exp(-x.data)))
        
    # def backward(self, grad_output):
    #     return grad_output * self._mask * (1 - self._mask)
    
class BinaryCrossEntropyLoss:
    def __init__(self):
        self.predictions = None
        self.targets = None
    
    def __call__(self, predictions, targets):
        if predictions.data.shape != targets.data.shape:
            raise ValueError("predictions and targets must have the same shape")
            
        # Clip probabilities to prevent log(0)
        predictions = np.clip(predictions.data, 1e-7, 1 - 1e-7)
        
        self.predictions = predictions.data
        self.targets = targets.data
        
        # compute the loss
        return -np.mean(self.targets * np.log(predictions) + (1 - self.targets) * np.log(1 - predictions))
    
    # def backward(self):
    #     return (self.predictions - self.targets) / self.predictions.shape[0]
        