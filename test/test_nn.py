from unittest import TestCase
from autograd.nn import Linear
import random
import numpy as np

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