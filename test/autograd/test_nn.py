from unittest import TestCase
from autograd.nn import Linear, BatchNorm, Dropout, Conv2d, MaxPool2d
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

        # Debug: Print running stats

        # Create a simple loss = sum of all elements
        loss = output.sum()
        loss_torch = output_torch.sum()

        loss.backward()
        loss_torch.backward()

        # Debug: Print gradients

        assert np.allclose(x.grad.data, x_torch.grad.numpy(), atol=1e-5)

    def test_batchnorm_components(self):
        x_data = np.array([[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]], dtype=np.float32)
        x = Tensor(x_data)

        # 1. Test mean calculation
        mean = x.mean(axis=0)

        # 2. Test centering (x - mean)
        centered = x - mean

        # 3. Test variance calculation
        var = (centered**2).sum(axis=0) / x.data.shape[0]

        # 4. Test normalization
        std = (var + 1e-5) ** 0.5
        normalized = centered / std

        # 5. Test gradients
        normalized.sum().backward()


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


class TestConv2d(TestCase):
    def setUp(self):
        self.conv2d = Conv2d(
            in_channels=2, out_channels=2, kernel_size=3, stride=1, padding_mode="valid"
        )
        self.x = Tensor(np.random.randn(1, 2, 6, 6))  # shape: (N, in_channels, H, W)
        self.x_torch = torch.from_numpy(self.x.data).float()
        self.torch_conv2d = torch.nn.Conv2d(
            in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=0
        )

        # Single channel input
        self.x_single = Tensor(np.random.randn(2, 1, 4, 4))
        self.x_single_torch = torch.tensor(self.x_single.data, requires_grad=True)

        # Copy our weights to PyTorch conv layer
        with torch.no_grad():
            self.torch_conv2d.weight.data = torch.from_numpy(
                self.conv2d._parameters["weight"].data
            ).float()
            self.torch_conv2d.bias.data = torch.from_numpy(
                self.conv2d._parameters["bias"].data
            ).float()

    def test_forward(self):
        output = self.conv2d(self.x)
        assert output.data.shape == (1, 2, 4, 4)  # shape: (N, out_channels, H', W')

        output_torch = self.torch_conv2d(self.x_torch)
        assert np.allclose(output.data, output_torch.detach().numpy(), atol=1e-5)

    def test_backward(self):
        # Create input tensor
        x = Tensor(np.random.rand(1, 2, 3, 3))  # shape: (N, in_channels, H, W)

        # Create Conv2d layer
        conv = Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding_mode="same")

        # Forward pass
        output = conv(x)
        target = Tensor(np.random.randn(*output.data.shape))
        loss = ((output - target) ** 2).sum()
        loss.backward()

        # Create PyTorch tensors and layer
        x_torch = torch.tensor(x.data, requires_grad=True)
        conv_torch = torch.nn.Conv2d(2, 1, 3, padding="same")
        with torch.no_grad():
            conv_torch.weight.data = torch.from_numpy(conv._parameters["weight"].data)
            conv_torch.bias.data = torch.from_numpy(conv._parameters["bias"].data)

        # Forward pass in PyTorch
        output_torch = conv_torch(x_torch)
        target_torch = torch.tensor(target.data, requires_grad=True)
        loss_torch = ((output_torch - target_torch) ** 2).sum()

        # Backward pass in PyTorch
        loss_torch.backward()

        # Assert gradients match
        assert np.allclose(x.grad.data, x_torch.grad.numpy(), rtol=1e-5, atol=1e-5)

    def test_sum_operation(self):
        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        y = x.sum()
        y.backward()
        assert np.allclose(x.grad.data, np.ones_like(x.data)), "Sum gradient incorrect"

    def test_simple_conv2d(self):
        # Create a simple 1x1x2x2 input
        x = Tensor(np.array([[[[1.0, 2.0], [3.0, 4.0]]]]))

        # Create Conv2d with 1x1 kernel
        conv = Conv2d(in_channels=1, out_channels=1, kernel_size=1)

        # Set weights and bias for easy verification
        conv._parameters["weight"].data = np.ones((1, 1, 1, 1))
        conv._parameters["bias"].data = np.zeros(1)

        # Forward pass
        output = conv(x)

        # Backward pass
        loss = output.sum(keepdims=True)
        loss.backward()


class TestMaxPool2d(TestConv2d):
    def test_forward(self):
        maxpool = MaxPool2d(kernel_size=2, stride=2)
        out = maxpool(self.x)

        torch_maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        out_torch = torch_maxpool(self.x_torch)

        assert np.allclose(out.data, out_torch.detach().numpy(), atol=1e-5)

    def test_different_kernel_stride(self):
        # Test 3x3 kernel with stride 2
        maxpool = MaxPool2d(kernel_size=3, stride=2)
        out = maxpool(self.x)

        torch_maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        out_torch = torch_maxpool(self.x_torch)

        assert np.allclose(out.data, out_torch.detach().numpy(), atol=1e-5)

    def test_same_padding(self):
        # Test with 'same' padding
        maxpool = MaxPool2d(kernel_size=2, stride=2, padding_mode="same")
        out = maxpool(self.x)

        # PyTorch uses explicit padding, so we need to pad first
        pad = torch.nn.ZeroPad2d(1)
        torch_maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        out_torch = torch_maxpool(pad(self.x_torch))

        assert np.allclose(out.data, out_torch.detach().numpy(), atol=1e-5)

    def test_single_channel(self):
        # Test with single channel input
        maxpool = MaxPool2d(kernel_size=2, stride=2)
        out = maxpool(self.x_single)

        torch_maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        out_torch = torch_maxpool(self.x_single_torch)

        assert np.allclose(out.data, out_torch.detach().numpy(), atol=1e-5)

    def test_kernel_equals_input(self):
        # Test when kernel size equals input size
        x = Tensor(np.random.randn(1, 1, 3, 3))
        x_torch = torch.tensor(x.data, requires_grad=True)

        maxpool = MaxPool2d(kernel_size=3, stride=1)
        out = maxpool(x)

        torch_maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=1)
        out_torch = torch_maxpool(x_torch)

        assert np.allclose(out.data, out_torch.detach().numpy(), atol=1e-5)

    def test_conv2d_gradient_sign(self):
        # Create small input with positive and negative values
        x = Tensor(np.array([[[[1.0, -1.0], [-1.0, 1.0]]]]))
        x_torch = torch.tensor(
            x.data, dtype=torch.float32, requires_grad=True
        )  # Specify float32

        conv = Conv2d(in_channels=1, out_channels=1, kernel_size=2)
        conv_torch = torch.nn.Conv2d(1, 1, 2)

        # Set same weights
        weight_data = np.array(
            [[[[1.0, -1.0], [-1.0, 1.0]]]], dtype=np.float32
        )  # Specify float32
        conv._parameters["weight"].data = weight_data
        with torch.no_grad():
            conv_torch.weight.data = torch.from_numpy(weight_data)
            conv_torch.bias.data = torch.zeros(
                1, dtype=torch.float32
            )  # Set bias to float32

        # Forward and backward
        out = conv(x)
        out_torch = conv_torch(x_torch)

        out.sum().backward()
        out_torch.sum().backward()

        # Compare gradients
        assert np.allclose(
            conv._parameters["weight"].grad.data, conv_torch.weight.grad.numpy()
        ), "Weight gradients do not match!"

    def test_conv_pool_chain(self):
        # Test 1: Simple 2x2 input
        x1 = Tensor(np.array([[[[1.0, -1.0], [-1.0, 1.0]]]]))
        x1_torch = torch.tensor(x1.data, dtype=torch.float32)

        # Test 2: Slightly larger 4x4 input
        x2 = Tensor(
            np.array(
                [
                    [
                        [
                            [1.0, -1.0, 1.0, -1.0],
                            [-1.0, 1.0, -1.0, 1.0],
                            [1.0, -1.0, 1.0, -1.0],
                            [-1.0, 1.0, -1.0, 1.0],
                        ]
                    ]
                ]
            )
        )
        x2_torch = torch.tensor(x2.data, dtype=torch.float32)

        # Create layers
        conv = Conv2d(in_channels=1, out_channels=1, kernel_size=2, bias=False)
        conv_torch = torch.nn.Conv2d(1, 1, 2, bias=False)

        pool = MaxPool2d(kernel_size=2, stride=2)
        pool_torch = torch.nn.MaxPool2d(2)

        # Set weights
        weight_data = np.array([[[[1.0, -1.0], [-1.0, 1.0]]]], dtype=np.float32)
        conv._parameters["weight"].data = weight_data
        with torch.no_grad():
            conv_torch.weight.data = torch.from_numpy(weight_data)

        # Test 1: 2x2 input
        out1 = conv(x1)
        out1_torch = conv_torch(x1_torch)

        # Compare outputs
        assert np.allclose(
            out1.data, out1_torch.detach().numpy()
        ), "Conv outputs do not match!"

        # Test 2: 4x4 input
        out2 = conv(x2)
        out2_torch = conv_torch(x2_torch)

        # Compare outputs
        assert np.allclose(
            out2.data, out2_torch.detach().numpy()
        ), "Conv outputs do not match!"

        # Add pooling
        pool2 = pool(out2)
        pool2_torch = pool_torch(out2_torch)

        # Compare pooling outputs
        assert np.allclose(
            pool2.data, pool2_torch.detach().numpy()
        ), "Pooling outputs do not match!"

    def test_conv_pool_chain_with_grads(self):
        # Setup with more complex input tensor (3 channels, 6x6)
        x2 = Tensor(
            np.array(
                [
                    [
                        [
                            [1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
                            [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
                            [1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
                            [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
                            [1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
                            [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
                        ],
                        [
                            [0.5, -0.5, 0.5, -0.5, 0.5, -0.5],
                            [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5],
                            [0.5, -0.5, 0.5, -0.5, 0.5, -0.5],
                            [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5],
                            [0.5, -0.5, 0.5, -0.5, 0.5, -0.5],
                            [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5],
                        ],
                        [
                            [0.3, -0.3, 0.3, -0.3, 0.3, -0.3],
                            [-0.3, 0.3, -0.3, 0.3, -0.3, 0.3],
                            [0.3, -0.3, 0.3, -0.3, 0.3, -0.3],
                            [-0.3, 0.3, -0.3, 0.3, -0.3, 0.3],
                            [0.3, -0.3, 0.3, -0.3, 0.3, -0.3],
                            [-0.3, 0.3, -0.3, 0.3, -0.3, 0.3],
                        ],
                    ]
                ]
            )
        )
        x2_torch = torch.tensor(x2.data, dtype=torch.float32, requires_grad=True)

        conv = Conv2d(in_channels=3, out_channels=2, kernel_size=4, bias=False)
        conv_torch = torch.nn.Conv2d(3, 2, 4, bias=False)

        pool = MaxPool2d(kernel_size=2, stride=2)
        pool_torch = torch.nn.MaxPool2d(2)

        # Set weights - now 4x4 kernel with 3 input channels and 2 output channels
        weight_data = np.array(
            [
                # First output channel
                [
                    # First input channel
                    [
                        [1.0, -1.0, 0.5, -0.5],
                        [-1.0, 1.0, -0.5, 0.5],
                        [0.5, -0.5, 1.0, -1.0],
                        [-0.5, 0.5, -1.0, 1.0],
                    ],
                    # Second input channel
                    [
                        [0.5, -0.5, 1.0, -1.0],
                        [-0.5, 0.5, -1.0, 1.0],
                        [1.0, -1.0, 0.5, -0.5],
                        [-1.0, 1.0, -0.5, 0.5],
                    ],
                    # Third input channel
                    [
                        [0.3, -0.3, 0.6, -0.6],
                        [-0.3, 0.3, -0.6, 0.6],
                        [0.6, -0.6, 0.3, -0.3],
                        [-0.6, 0.6, -0.3, 0.3],
                    ],
                ],
                # Second output channel
                [
                    # First input channel
                    [
                        [-1.0, 1.0, -0.5, 0.5],
                        [1.0, -1.0, 0.5, -0.5],
                        [-0.5, 0.5, -1.0, 1.0],
                        [0.5, -0.5, 1.0, -1.0],
                    ],
                    # Second input channel
                    [
                        [-0.5, 0.5, -1.0, 1.0],
                        [0.5, -0.5, 1.0, -1.0],
                        [-1.0, 1.0, -0.5, 0.5],
                        [1.0, -1.0, 0.5, -0.5],
                    ],
                    # Third input channel
                    [
                        [-0.3, 0.3, -0.6, 0.6],
                        [0.3, -0.3, 0.6, -0.6],
                        [-0.6, 0.6, -0.3, 0.3],
                        [0.6, -0.6, 0.3, -0.3],
                    ],
                ],
            ],
            dtype=np.float32,
        )

        conv._parameters["weight"].data = weight_data
        with torch.no_grad():
            conv_torch.weight.data = torch.from_numpy(weight_data)

        # Forward pass
        conv_out = conv(x2)
        conv_out_torch = conv_torch(x2_torch)
        conv_out_torch.retain_grad()  # Add this line to retain intermediate gradients

        pool_out = pool(conv_out)
        pool_out_torch = pool_torch(conv_out_torch)

        # Backward pass
        pool_out.sum().backward()
        pool_out_torch.sum().backward()

        # Compare gradients
        assert np.allclose(
            conv_out.grad.data
            if conv_out.grad is not None
            else np.zeros_like(conv_out.data),
            conv_out_torch.grad.numpy()
            if conv_out_torch.grad is not None
            else np.zeros_like(conv_out_torch.data),
        ), "Conv output gradients do not match!"

        assert np.allclose(
            x2.grad.data, x2_torch.grad.numpy()
        ), "Input gradients do not match!"

        assert np.allclose(
            conv._parameters["weight"].grad.data, conv_torch.weight.grad.numpy()
        ), "Weight gradients do not match!"
