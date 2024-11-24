from unittest import TestCase
from autograd.nn import Linear, BatchNorm, Dropout
from autograd.tensor import Tensor
import random
import numpy as np
import torch  # for comparison

random.seed(1337)
np.random.seed(1337)
torch.manual_seed(1337)


class TestLinear(TestCase):
    def test_linear(self):
        linear_layer = Linear(
            input_size=4,
            output_size=2,
        )

        parameters = linear_layer.parameters
        assert parameters["weight"].data.shape == (4, 2)
        assert parameters["bias"].data.shape == (2,)

        # Trying to pass in (1x4 matrix)
        x = [[2, 2, 2, 2]]
        out = linear_layer(x)
        assert np.allclose(out.data, [-2.7748068, 0.56519009])
        assert out.grad is None
        assert parameters["weight"].grad is None
        assert parameters["bias"].grad is None
        out.backward()

        # weight gradient = x.T @ out.grad = [[2], [2], [2], [2]] * [1, 1]
        assert np.array_equal(
            parameters["weight"].grad.data,
            [
                [2, 2],
                [2, 2],
                [2, 2],
                [2, 2],
            ],
        )
        assert np.array_equal(parameters["bias"].grad.data, [1, 1])
        assert np.array_equal(out.grad.data, [[1, 1]])

        # Trying to pass in (4x1 matrix)
        x = [[2], [2], [2], [2]]
        linear_layer = Linear(input_size=1, output_size=2)
        out = linear_layer(x)
        parameters = linear_layer.parameters
        assert parameters["weight"].data.shape == (1, 2)
        assert parameters["bias"].data.shape == (2,)

        out.backward()
        assert np.allclose(parameters["weight"].grad.data, [[8], [8]])
        assert np.allclose(parameters["bias"].grad.data, [[4], [4]])
        assert np.array_equal(
            out.grad.data,
            [
                [1, 1],
                [1, 1],
                [1, 1],
                [1, 1],
            ],
        )


class TestBatchNorm(TestCase):
    def setUp(self) -> None:
        momentum = 0.1
        self.input_size = 2
        self.batch_size = 3
        self.bn = BatchNorm(input_size=self.input_size, momentum=momentum)
        self.torch_bn = torch.nn.BatchNorm1d(
            num_features=self.input_size, momentum=momentum
        )
        # Initialize torch_bn weights and bias to match your implementation
        self.torch_bn.weight.data = torch.ones(self.input_size)
        self.torch_bn.bias.data = torch.zeros(self.input_size)

    def test_forward(self):
        # Feature 1: [1, 4, 7], Feature 2: [2, 5, 8]
        # Feature 1 Mean: 4, Feature 2 Mean: 5
        # Feature 1 Var: 6, Feature 2 Var: 6
        # Feature 1 Normalized: [-1, 0, 1], Feature 2 Normalized: [-1, 0, 1]
        x_data = np.array(
            [
                [1.0, 2.0],
                [4.0, 5.0],
                [7.0, 8.0],
            ]
        )
        x = Tensor(x_data)
        x_torch = torch.Tensor(x_data)

        self.bn.train()
        self.torch_bn.train()
        output = self.bn(x)
        output_torch = self.torch_bn(x_torch)

        # Compare results
        assert np.allclose(output.data, output_torch.detach().numpy(), atol=1e-5)

        # Compare running stats
        assert np.allclose(
            self.bn.running_mean, self.torch_bn.running_mean.numpy(), atol=1e-5
        )
        assert np.allclose(
            self.bn.running_var, self.torch_bn.running_var.numpy(), atol=1e-5
        )

        # Test Inference mode
        self.bn.eval()
        self.torch_bn.eval()

        x = Tensor(x_data)
        x_torch = torch.FloatTensor(x_data)

        output = self.bn(x)
        output_torch = self.torch_bn(x_torch)

        # Compare results in eval mode
        assert np.allclose(output.data, output_torch.detach().numpy(), atol=1e-5)

        # Verify running stats haven't changed in eval mode
        assert np.allclose(
            self.bn.running_mean, self.torch_bn.running_mean.numpy(), atol=1e-5
        )
        assert np.allclose(
            self.bn.running_var, self.torch_bn.running_var.numpy(), atol=1e-5
        )

    def test_backward(self):
        x_data = np.array([[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]], dtype=np.float32)
        x = Tensor(x_data)
        x_torch = torch.tensor(x_data, requires_grad=True, dtype=torch.float32)

        self.bn.train()
        self.torch_bn.train()

        output = self.bn(x)
        output_torch = self.torch_bn(x_torch)

        # Debug: Print normalization details
        print("Normalized output:")
        print("Our implementation:", output.data)
        print("PyTorch implementation:", output_torch.detach().numpy())

        # Debug: Print running stats
        print("\nRunning Stats:")
        print("Our running mean:", self.bn.running_mean)
        print("PyTorch running mean:", self.torch_bn.running_mean.numpy())
        print("Our running var:", self.bn.running_var)
        print("PyTorch running var:", self.torch_bn.running_var.numpy())

        # Create a simple loss = sum of all elements
        loss = output.sum()
        loss_torch = output_torch.sum()

        loss.backward()
        loss_torch.backward()

        # Debug: Print gradients
        print("\nGradients:")
        print("Our input gradient:", x.grad)
        print("PyTorch input gradient:", x_torch.grad)

        assert np.allclose(x.grad.data, x_torch.grad.numpy(), atol=1e-5)

    def test_batchnorm_components(self):
        x_data = np.array([[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]], dtype=np.float32)
        x = Tensor(x_data)

        # 1. Test mean calculation
        mean = x.mean(axis=0)
        print("Mean:", mean.data)

        # 2. Test centering (x - mean)
        centered = x - mean
        print("Centered:", centered.data)
        print("Centered sum:", centered.data.sum(axis=0))  # Should be ~0

        # 3. Test variance calculation
        var = (centered**2).sum(axis=0) / x.data.shape[0]
        print("Variance:", var.data)

        # 4. Test normalization
        std = (var + 1e-5) ** 0.5
        normalized = centered / std
        print("Normalized:", normalized.data)
        print("Normalized mean:", normalized.data.mean(axis=0))  # Should be ~0
        print("Normalized std:", normalized.data.std(axis=0))  # Should be ~1

        # 5. Test gradients
        normalized.sum().backward()
        print("Input gradients:", x.grad.data)
        print("Gradient sum per feature:", x.grad.data.sum(axis=0))  # Should be ~0


class TestDropout(TestCase):
    def test_forward(self):
        dropout = Dropout(p=1)
        dropout.train()
        x = Tensor(np.array([[1, 2], [3, 4], [5, 6]]))
        output = dropout(x)
        assert np.allclose(output.data, np.array([[0, 0], [0, 0], [0, 0]]))

        dropout.eval()
        output = dropout(x)
        assert np.allclose(output.data, np.array([[1, 2], [3, 4], [5, 6]]))

        dropout = Dropout(p=0)
        dropout.train()
        output = dropout(x)
        assert np.allclose(output.data, np.array([[1, 2], [3, 4], [5, 6]]))
