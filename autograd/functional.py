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
    # 709 is the maximum value that can be passed to np.exp without overflowing
    out = Tensor(1 / (1 + np.exp(np.clip(-x.data, -709, 709))), prev=(x,))
    
    def _backward():
        # print(f"Sigmoid backward shapes:")
        # print(f"out.grad shape: {out.grad.shape}")
        # print(f"out.data shape: {out.data.shape}")
        # print(f"x.grad shape: {x.grad.shape}")
        # d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
        x.grad += out.grad * out.data * (1 - out.data)
        # print(f"After backward x.grad shape: {x.grad.shape}")
    
    out._backward = _backward
    return out

###################### Loss Functions #####################
def binary_cross_entropy(y_pred: Tensor, y_true: Union[Tensor, np.ndarray]) -> Tensor:
    """
    Binary Cross Entropy Loss
    -(x * log(y)) + (1 - x) * log(1 - y)
    """

    y_true = np.array(y_true.data)
    
    if y_pred.data.shape[0] != y_true.shape[0]:
        raise ValueError("y_pred and y_true must have the same shape")
        
    # Clip probabilities to prevent log(0)
    y_pred_prob = np.clip(y_pred.data, 1e-15, 1 - 1e-15)
    
    # compute the loss
    out = Tensor(
        data=-np.mean(y_true * np.log(y_pred_prob) + (1 - y_true) * np.log(1 - y_pred_prob)),
        prev=(y_pred,), # this is very important to connect the loss tensor with the y_pred tensor
    )
    
    def _backward():
        # dL/dpred = -(y/p - (1-y)/(1-p))
        # print(f"y_true shape: {y_true.shape}")
        # print(f"y_pred shape: {y_pred.data.shape}")
        y_pred.grad += -(y_true / y_pred_prob - (1 - y_true) / (1 - y_pred_prob)) / len(y_pred_prob)
        # print(f"grad shape: {grad.shape}")
        # print(f"y_pred.grad shape after: {y_pred.grad.shape}")
    
    out._backward = _backward
    return out
