from autograd.tensor import Tensor
import numpy as np

def relu(x: Tensor):
    """
    Retified Linear Unit (ReLU) activation function.
    ReLU(x) = max(0, x)
    """
    
    out = Tensor(np.maximum(0, x.data), prev=(x,))
    
    def _backward():
        # dL/dx = dL/dy * dy/dx
        if x.grad is None:
            x.grad = out.grad * (x.data > 0)
        else:
            x.grad += out.grad * (x.data > 0)
        
    out._backward = _backward
    return out


def sigmoid(x: Tensor):
    """
    Sigmoid activation function
    """
    
    out = Tensor(1 / (1 + np.exp(-x.data)), prev=(x,))
    
    def _backward():
        # d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
        if x.grad is None:
            x.grad = out.grad * out.data * (1 - out.data)
        else:
            x.grad += out.grad * out.data * (1 - out.data)
    
    out._backward = _backward
    return out