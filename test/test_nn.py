from unittest import TestCase
from autograd.nn import Linear, ReLU, Sigmoid, BinaryCrossEntropyLoss
import random
import numpy as np
import torch # for test comparisons

from autograd.tensor import Tensor

random.seed(1337)
np.random.seed(1337)

class TestLinear(TestCase):
    def test_linear(self):
        linear_layer = Linear(
            input_size=4,
            output_size=2,
        )
        
        parameters = linear_layer.parameters
        assert parameters['weight'].data.shape == (4, 2)
        assert parameters['bias'].data.shape == (2,)
        
        # Trying to pass in (1x4 matrix)
        x = [[2, 2, 2, 2]]
        out = linear_layer(x)
        assert np.allclose(out.data, [-2.75117575, -7.83881729])
        assert np.allclose(out.grad, [0, 0]) # this should still be zero before we call backward
        assert np.allclose(parameters['weight'].grad, np.zeros_like(parameters['weight'].data))
        assert np.allclose(parameters['bias'].grad, np.zeros_like(parameters['bias'].data))
        out.backward()
        
        # weight gradient = x.T @ out.grad = [[2], [2], [2], [2]] * [1, 1]
        assert np.array_equal(
            parameters['weight'].grad,
            [
                [2, 2],
                [2, 2],
                [2, 2],
                [2, 2],
            ]
        )
        assert np.array_equal(parameters['bias'].grad, [1,1])
        assert np.array_equal(out.grad, [1, 1])
        
        # Trying to pass in (4x1 matrix)
        x = [[2],[2],[2],[2]]
        linear_layer = Linear(input_size=1, output_size=2)
        out = linear_layer(x)
        parameters = linear_layer.parameters
        assert parameters['weight'].data.shape == (1, 2)
        assert parameters['bias'].data.shape == (2,)
        
        out.backward()
        assert np.allclose(
            parameters['weight'].grad,
            [
                [8],[8]
            ]
        )
        assert np.allclose(
            parameters['bias'].grad,
            [
                [4],[4]
            ]
        )
        assert np.array_equal(
            out.grad,
            [
                [1, 1],
                [1, 1],
                [1, 1],
                [1, 1],
            ]
        )
        
    def test_relu(self):
        x = Tensor(np.array([[1, -2, 3], [-4, 5, -6]]))
        
        # Test forward pass
        relu = ReLU()
        out = relu(x)
        assert np.array_equal(out.data, [[1, 0, 3], [0, 5, 0]])
    
    def test_sigmoid(self):
        x = Tensor(np.array([[1, -2, 3], [-4, 5, -6]]))
        
        # Test forward pass
        sigmoid = Sigmoid()
        out = sigmoid(x)
        assert np.allclose(out.data, torch.sigmoid(torch.tensor(x.data)).numpy())
        
    
    def test_binary_cross_entropy_loss(self):
        # With logits
        y_pred = Tensor(np.array([0.5, 0, 0.75]))
        y_true = Tensor(np.array([0.0, 1.0, 1.0]))
        
        y_pred_probs = Sigmoid().forward(y_pred)
        bce_loss = BinaryCrossEntropyLoss()(y_pred_probs, y_true)
        torch_loss = torch.nn.BCELoss()
        assert bce_loss == torch_loss(torch.sigmoid(torch.tensor((y_pred.data))), torch.tensor(y_true.data))