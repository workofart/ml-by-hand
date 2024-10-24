import numpy as np
from .tensor import Tensor

class Module:
    def __init__(self):
        self._parameters = {}
    
    def forward(self, x):
        raise NotImplementedError
    
    def __call__(self, x):
        """
        Sometimes people like to call model = Module() then call model(x)
        as a forward pass. So this is an alias.
        """
        return self.forward(x)
    
    @property
    def parameters(self):
        return self._parameters
    
class Linear(Module):
    def __init__(self, input_size, output_size, **kwargs):
        super().__init__(**kwargs)
        
        # weight is a matrix of shape (input_size, output_size)
        self._parameters['weight'] = Tensor(np.random.randn(input_size, output_size), requires_grad=True)
        
        # bias is always 1-dimensional
        self._parameters['bias'] = Tensor(np.zeros(output_size), requires_grad=True)
        
    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        
        print("x.data shape:", x.data.shape)
        print("weights shape:", self._parameters['weight'].data.shape)
    
        # this is just a linear transformation (dot matrix multiplication)
        return x @ self._parameters['weight'] + self._parameters['bias']