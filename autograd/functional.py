from typing import Union
from autograd.tensor import Tensor
import numpy as np

def relu(x: Tensor) -> Tensor:
    """
    Retified Linear Unit (ReLU) activation function.
    ReLU(x) = max(0, x)
    """
    
    out = Tensor(np.maximum(0, x.data), prev=(x,))
    
    def _backward():
        # dL/dx = dL/dy * dy/dx
        x.grad += out.grad * (x.data > 0)
        
    out._backward = _backward
    return out


def sigmoid(x: Tensor) -> Tensor:
    """
    Sigmoid activation function
    """
    
    out = Tensor(1 / (1 + np.exp(-x.data + 1e-7)), prev=(x,))
    
    def _backward():
        # d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
        x.grad += out.grad * out.data * (1 - out.data)
    
    out._backward = _backward
    return out

###################### Loss Functions #####################
def binary_cross_entropy(y_pred: Tensor, y_true: Union[Tensor, np.ndarray]) -> Tensor:
    """
    Binary Cross Entropy Loss
    -(x * log(y) + (1 - x) * log(1 - y)
    """
    if y_pred.data.shape != y_true.data.shape:
        raise ValueError("y_pred and y_true must have the same shape")
        
    y_true = np.array(y_true.data)
    # Clip probabilities to prevent log(0)
    y_pred_prob = np.clip(y_pred.data, 1e-15, 1 - 1e-15)
    
    # compute the loss
    out = Tensor(
        data=-np.mean(y_true * np.log(y_pred_prob) + (1 - y_true) * np.log(1 - y_pred_prob)),
        prev=(y_pred,), # this is very important to connect the loss tensor with the y_pred tensor
    )
    
    def _backward():
        # dL/dpred = -(y/p - (1-y)/(1-p))
        print(f"y_true shape: {y_true.shape}")
        print(f"y_pred shape: {y_pred.data.shape}")
        y_pred.grad += -(y_true / y_pred_prob - (1 - y_true) / (1 - y_pred_prob)) / len(y_pred_prob)
    
    out._backward = _backward
    return out
