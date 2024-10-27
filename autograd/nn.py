import numpy as np
from .tensor import Tensor

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
        # print(f"Setting attributes {name} = {value}")
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
            params.update({ k: module.parameters })
            
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
        self._parameters['weight'] = Tensor(np.random.randn(input_size, output_size))
        
        # bias is always 1-dimensional
        self._parameters['bias'] = Tensor(np.zeros(output_size))
        
    def forward(self, x) -> Tensor:
        if not isinstance(x, Tensor):
            x = Tensor(x)
        
        # print("x.data shape:", x.data.shape)
        # print("weights shape:", self._parameters['weight'].data.shape)
    
        # this is just a linear transformation (dot matrix multiplication)
        return x @ self._parameters['weight'] + self._parameters['bias']
