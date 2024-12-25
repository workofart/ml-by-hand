from unittest import TestCase
from autograd.nn import (
    Linear,
    BatchNorm,
    Dropout,
    Conv2d,
    MaxPool2d,
    RecurrentBlock,
    LongShortTermMemoryBlock,
)
from autograd.tensor import Tensor
import random
import numpy as np
import torch  # for comparison

random.seed(1337)
np.random.seed(1337)
torch.manual_seed(1337)


class TestLinear(TestCase):
    def test_linear(self):
        linear_layer = Linear(input_size=4, output_size=2)

        # Explicitly set weights and biases
        linear_layer._parameters["weight"].data = np.array(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
        )
        linear_layer._parameters["bias"].data = np.array([0.1, 0.2])

        # Test with known input and expected output
        x = [[2, 2, 2, 2]]
        out = linear_layer(x)

        # Calculate expected output manually
        expected = np.array(
            [[3.3, 4.2]]
        )  # (2*0.1 + 2*0.3 + 2*0.5 + 2*0.7 + 0.1, 2*0.2 + 2*0.4 + 2*0.6 + 2*0.8 + 0.2)
        assert np.allclose(out.data, expected)

        # Test gradient computation
        out.backward()

        # Test gradient shapes and properties
        assert linear_layer._parameters["weight"].grad.shape == (4, 2)
        assert linear_layer._parameters["bias"].grad.shape == (2,)
        assert np.allclose(linear_layer._parameters["bias"].grad.data, [1, 1])
        assert np.all(
            linear_layer._parameters["weight"].grad.data == 2
        )  # All gradients should be 2 since input is all 2s


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


class TestRecurrentNetwork(TestCase):
    def setUp(self):
        self.input_size = 3
        self.hidden_size = 4
        self.output_size = 2
        self.batch_size = 5
        self.seq_length = 10

        # Create RNN with output layer
        self.rnn = RecurrentBlock(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
        )

        # Create RNN without output layer
        self.rnn_no_output = RecurrentBlock(
            input_size=self.input_size, hidden_size=self.hidden_size
        )

    def test_initialization(self):
        # Test parameter shapes
        assert self.rnn._parameters["W_xh"].data.shape == (
            self.input_size,
            self.hidden_size,
        )
        assert self.rnn._parameters["W_hh"].data.shape == (
            self.hidden_size,
            self.hidden_size,
        )
        assert self.rnn._parameters["W_hy"].data.shape == (
            self.hidden_size,
            self.output_size,
        )
        assert self.rnn._parameters["bias"].data.shape == (self.hidden_size,)
        assert self.rnn._parameters["bias_y"].data.shape == (self.output_size,)

        # Test no output layer case
        assert self.rnn_no_output._parameters["W_hy"] is None
        assert self.rnn_no_output._parameters["bias_y"] is None

    def test_forward(self):
        np.random.seed(42)
        torch.manual_seed(42)

        # Use smaller dimensions for easier debugging
        x_data = np.random.randn(2, 3, self.input_size)  # batch=2, seq=3
        x = Tensor(x_data)
        x_torch = torch.FloatTensor(x_data)

        torch_rnn = torch.nn.RNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            batch_first=True,
            nonlinearity="tanh",
        )
        torch_linear = torch.nn.Linear(self.hidden_size, self.output_size)

        # Copy weights and print them to verify
        with torch.no_grad():
            torch_rnn.weight_ih_l0.data = torch.FloatTensor(
                self.rnn._parameters["W_xh"].data.T
            )
            torch_rnn.weight_hh_l0.data = torch.FloatTensor(
                self.rnn._parameters["W_hh"].data.T
            )
            torch_rnn.bias_ih_l0.data = torch.FloatTensor(
                self.rnn._parameters["bias"].data
            )
            torch_rnn.bias_hh_l0.data = torch.zeros_like(torch_rnn.bias_hh_l0)

            torch_linear.weight.data = torch.FloatTensor(
                self.rnn._parameters["W_hy"].data.T
            )
            torch_linear.bias.data = torch.FloatTensor(
                self.rnn._parameters["bias_y"].data
            )

        # Get PyTorch's hidden states
        torch_output, _ = torch_rnn(x_torch)

        # Final output comparison
        output = self.rnn(x)
        torch_output = torch_linear(torch_output[:, -1, :])

        assert np.allclose(
            output.data, torch_output.detach().numpy(), rtol=1e-4, atol=1e-4
        ), "RNN output doesn't match PyTorch's output"

    def test_backward(self):
        np.random.seed(42)
        torch.manual_seed(42)

        # Create small input for easier gradient checking
        x_data = np.random.randn(2, 3, self.input_size)
        x = Tensor(x_data)
        x_torch = torch.FloatTensor(x_data).requires_grad_(True)

        # Create PyTorch RNN
        torch_rnn = torch.nn.RNN(
            input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True
        )
        torch_linear = torch.nn.Linear(self.hidden_size, self.output_size)

        # Copy weights to PyTorch RNN
        with torch.no_grad():
            torch_rnn.weight_ih_l0.data = torch.FloatTensor(
                self.rnn._parameters["W_xh"].data.T
            )
            torch_rnn.weight_hh_l0.data = torch.FloatTensor(
                self.rnn._parameters["W_hh"].data.T
            )
            torch_rnn.bias_ih_l0.data = torch.FloatTensor(
                self.rnn._parameters["bias"].data
            )
            torch_rnn.bias_hh_l0.data = torch.zeros_like(torch_rnn.bias_hh_l0)

            torch_linear.weight.data = torch.FloatTensor(
                self.rnn._parameters["W_hy"].data.T
            )
            torch_linear.bias.data = torch.FloatTensor(
                self.rnn._parameters["bias_y"].data
            )

        # Forward pass
        output = self.rnn(x)
        torch_output, _ = torch_rnn(x_torch)
        torch_output = torch_linear(torch_output[:, -1, :])

        # Create simple loss and backward
        loss = output.sum()
        loss_torch = torch_output.sum()

        loss.backward()
        loss_torch.backward()

        # Compare gradients
        # Input gradients
        assert np.allclose(
            x.grad.data, x_torch.grad.numpy(), rtol=1e-4, atol=1e-4
        ), "Input gradients don't match"

        # Weight gradients - need to transpose PyTorch gradients to match our format
        assert np.allclose(
            self.rnn._parameters["W_xh"].grad.data,
            torch_rnn.weight_ih_l0.grad.numpy().T,
            rtol=1e-4,
            atol=1e-4,
        ), "W_xh gradients don't match"

        assert np.allclose(
            self.rnn._parameters["W_hh"].grad.data,
            torch_rnn.weight_hh_l0.grad.numpy().T,
            rtol=1e-4,
            atol=1e-4,
        ), "W_hh gradients don't match"

        assert np.allclose(
            self.rnn._parameters["W_hy"].grad.data,
            torch_linear.weight.grad.numpy().T,
            rtol=1e-4,
            atol=1e-4,
        ), "W_hy gradients don't match"

        # Bias gradients
        assert np.allclose(
            self.rnn._parameters["bias"].grad.data,
            torch_rnn.bias_ih_l0.grad.numpy(),
            rtol=1e-4,
            atol=1e-4,
        ), "RNN bias gradients don't match"

        assert np.allclose(
            self.rnn._parameters["bias_y"].grad.data,
            torch_linear.bias.grad.numpy(),
            rtol=1e-4,
            atol=1e-4,
        ), "Output bias gradients don't match"

    def test_simple_sequence(self):
        # Test with a simple sequence where we can manually verify the results
        self.rnn = RecurrentBlock(input_size=2, hidden_size=2, output_size=1)

        # Set weights manually for predictable output
        self.rnn._parameters["W_xh"].data = np.array([[0.5, 0.0], [0.0, 0.5]])
        self.rnn._parameters["W_hh"].data = np.array([[0.1, 0.0], [0.0, 0.1]])
        self.rnn._parameters["W_hy"].data = np.array([[1.0], [1.0]])
        self.rnn._parameters["bias"].data = np.zeros(2)
        self.rnn._parameters["bias_y"].data = np.zeros(1)

        # Simple input sequence
        x = Tensor(
            np.array([[[1.0, 0.0], [0.0, 1.0]]])
        )  # batch_size=1, seq_length=2, input_size=2

        output = self.rnn(x)
        # Verify output shape
        assert output.shape == (1, 1)

    def test_sequence_length_one(self):
        # Test with sequence length of 1 (edge case)
        x = Tensor(np.random.randn(self.batch_size, 1, self.input_size))

        output = self.rnn(x)
        assert output.shape == (self.batch_size, self.output_size)


class TestLongShortTermMemoryBlock(TestCase):
    def setUp(self):
        self.input_size = 3
        self.hidden_size = 4
        self.output_size = 2
        self.batch_size = 5
        self.seq_length = 10

        # Create LSTM with output layer
        self.lstm = LongShortTermMemoryBlock(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
        )

        # Create LSTM without output layer
        self.lstm_no_output = LongShortTermMemoryBlock(
            input_size=self.input_size, hidden_size=self.hidden_size
        )

        # Create test data
        np.random.seed(42)
        torch.manual_seed(42)
        self.x_data = np.random.randn(2, 3, self.input_size)  # batch=2, seq=3
        self.x = Tensor(self.x_data)
        self.x_torch = torch.FloatTensor(self.x_data)
        self.x_torch.requires_grad = True

        # Create PyTorch LSTM and output layer
        self.torch_lstm = torch.nn.LSTM(
            input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True
        )
        self.torch_linear = torch.nn.Linear(self.hidden_size, self.output_size)

        # Copy weights to PyTorch LSTM
        with torch.no_grad():
            # Copy gate weights
            # PyTorch concatenates all gates into one matrix, so we need to split them
            W_i_input = (
                self.lstm._parameters["W_i"].data[: self.input_size].T
            )  # (3,4) -> (4,3)
            W_f_input = self.lstm._parameters["W_f"].data[: self.input_size].T
            W_c_input = self.lstm._parameters["W_c"].data[: self.input_size].T
            W_o_input = self.lstm._parameters["W_o"].data[: self.input_size].T

            # Stack them vertically => shape (4+4+4+4, 3) = (16,3)
            ih_weights = np.concatenate(
                (W_i_input, W_f_input, W_c_input, W_o_input), axis=0
            )
            self.torch_lstm.weight_ih_l0.data = torch.FloatTensor(ih_weights)

            W_i_hidden = self.lstm._parameters["W_i"].data[self.input_size :].T  # (4,4)
            W_f_hidden = self.lstm._parameters["W_f"].data[self.input_size :].T
            W_c_hidden = self.lstm._parameters["W_c"].data[self.input_size :].T
            W_o_hidden = self.lstm._parameters["W_o"].data[self.input_size :].T

            hh_weights = np.concatenate(
                (W_i_hidden, W_f_hidden, W_c_hidden, W_o_hidden), axis=0
            )  # (16,4)
            self.torch_lstm.weight_hh_l0.data = torch.FloatTensor(hh_weights)

            # Copy biases
            ih_bias = np.concatenate(
                [
                    self.lstm._parameters["bias_i"].data,
                    self.lstm._parameters["bias_f"].data,
                    self.lstm._parameters["bias_c"].data,
                    self.lstm._parameters["bias_o"].data,
                ]
            )
            self.torch_lstm.bias_ih_l0.data = torch.FloatTensor(ih_bias)
            self.torch_lstm.bias_hh_l0.data = torch.zeros_like(
                self.torch_lstm.bias_hh_l0
            )

            # Copy output layer weights
            self.torch_linear.weight.data = torch.FloatTensor(
                self.lstm._parameters["W_hy"].data.T
            )
            self.torch_linear.bias.data = torch.FloatTensor(
                self.lstm._parameters["bias_y"].data
            )

    def test_initialization(self):
        # Test parameter shapes
        input_hidden = self.input_size + self.hidden_size

        # Test gate parameter shapes
        assert self.lstm._parameters["W_f"].data.shape == (
            input_hidden,
            self.hidden_size,
        )
        assert self.lstm._parameters["W_i"].data.shape == (
            input_hidden,
            self.hidden_size,
        )
        assert self.lstm._parameters["W_c"].data.shape == (
            input_hidden,
            self.hidden_size,
        )
        assert self.lstm._parameters["W_o"].data.shape == (
            input_hidden,
            self.hidden_size,
        )

        # Test bias shapes
        assert self.lstm._parameters["bias_f"].data.shape == (self.hidden_size,)
        assert self.lstm._parameters["bias_i"].data.shape == (self.hidden_size,)
        assert self.lstm._parameters["bias_c"].data.shape == (self.hidden_size,)
        assert self.lstm._parameters["bias_o"].data.shape == (self.hidden_size,)

        # Test output layer parameters
        assert self.lstm._parameters["W_hy"].data.shape == (
            self.hidden_size,
            self.output_size,
        )
        assert self.lstm._parameters["bias_y"].data.shape == (self.output_size,)

        # Test no output layer case
        assert self.lstm_no_output._parameters["W_hy"] is None
        assert self.lstm_no_output._parameters["bias_y"] is None

    def test_forward(self):
        # Forward pass
        output = self.lstm(self.x)
        torch_output, _ = self.torch_lstm(self.x_torch)
        torch_output = self.torch_linear(torch_output[:, -1, :])

        assert np.allclose(
            output.data, torch_output.detach().numpy(), rtol=1e-4, atol=1e-4
        ), "LSTM output doesn't match PyTorch's output"

    def test_backward(self):
        # Forward pass
        output = self.lstm(self.x)
        torch_output, _ = self.torch_lstm(self.x_torch)
        torch_output = self.torch_linear(torch_output[:, -1, :])

        # Create simple loss and backward
        loss = output.sum()
        loss_torch = torch_output.sum()

        loss.backward()
        loss_torch.backward()

        # Compare gradients
        assert np.allclose(
            self.x.grad.data, self.x_torch.grad.numpy(), rtol=1e-4, atol=1e-4
        ), "Input gradients don't match"

        # Compare gate gradients
        # We need to split PyTorch's concatenated gradients
        ih_grad = self.torch_lstm.weight_ih_l0.grad.numpy()
        hh_grad = self.torch_lstm.weight_hh_l0.grad.numpy()

        gates = ["i", "f", "c", "o"]
        for idx, gate in enumerate(gates):
            # Input weights
            start_idx = idx * self.hidden_size
            end_idx = (idx + 1) * self.hidden_size
            assert np.allclose(
                self.lstm._parameters[f"W_{gate}"].grad.data[: self.input_size],
                ih_grad[start_idx:end_idx].T,
                rtol=1e-4,
                atol=1e-4,
            ), f"W_{gate} input gradients don't match"

            # Hidden weights
            assert np.allclose(
                self.lstm._parameters[f"W_{gate}"].grad.data[self.input_size :],
                hh_grad[start_idx:end_idx].T,
                rtol=1e-4,
                atol=1e-4,
            ), f"W_{gate} hidden gradients don't match"

        # Compare output layer gradients
        assert np.allclose(
            self.lstm._parameters["W_hy"].grad.data,
            self.torch_linear.weight.grad.numpy().T,
            rtol=1e-4,
            atol=1e-4,
        ), "W_hy gradients don't match"

    def test_simple_sequence(self):
        # Test with a simple sequence where we can manually verify the results
        lstm = LongShortTermMemoryBlock(input_size=2, hidden_size=2, output_size=1)

        # Set weights manually for predictable output
        input_hidden = 4  # input_size + hidden_size
        for gate in ["f", "i", "c", "o"]:
            lstm._parameters[f"W_{gate}"].data = np.eye(input_hidden, 2) * 0.5
            lstm._parameters[f"bias_{gate}"].data = np.zeros(2)

        lstm._parameters["W_hy"].data = np.ones((2, 1))
        lstm._parameters["bias_y"].data = np.zeros(1)

        # Simple input sequence
        x = Tensor(
            np.array([[[1.0, 0.0], [0.0, 1.0]]])
        )  # batch_size=1, seq_length=2, input_size=2

        output = lstm(x)
        # Verify output shape
        assert output.shape == (1, 1)

    def test_sequence_length_one(self):
        # Test with sequence length of 1 (edge case)
        x = Tensor(np.random.randn(self.batch_size, 1, self.input_size))

        output = self.lstm(x)
        assert output.shape == (self.batch_size, self.output_size)
