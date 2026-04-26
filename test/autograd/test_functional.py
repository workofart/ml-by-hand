from unittest import TestCase
from unittest.mock import patch

import torch  # for test comparisons

from autograd import functional
from autograd.backend import xp
from autograd.tensor import Tensor
from test.helpers import allclose


class TestActivationFunctions(TestCase):
    def setUp(self) -> None:
        self.X = Tensor(
            data=xp.array([[1, 1, 1], [2, 2, 2]], dtype=xp.float32),
            requires_grad=True,
        )

    def test_sigmoid_forward(self):
        assert allclose(
            functional.sigmoid(self.X).data,
            torch.nn.functional.sigmoid(torch.Tensor(self.X.data)).detach().numpy(),
        )

    def test_sigmoid_backward(self):
        out_custom = functional.sigmoid(self.X)
        grad_dummy = xp.ones_like(out_custom.data)
        out_custom.backward(grad_dummy)

        X_torch = torch.tensor(self.X.data, requires_grad=True)
        out_ref = torch.sigmoid(X_torch)
        out_ref.backward(torch.ones_like(out_ref))

        assert allclose(self.X.grad.data, X_torch.grad.detach().numpy(), atol=1e-6), (
            "Sigmoid backward pass did not match PyTorch."
        )

    def test_relu_forward(self):
        assert allclose(
            functional.relu(self.X).data,
            torch.nn.functional.relu(torch.Tensor(self.X.data)).detach().numpy(),
        )

    def test_relu_backward(self):
        out_custom = functional.relu(self.X)
        grad_dummy = xp.ones_like(out_custom.data)
        out_custom.backward(grad_dummy)

        X_torch = torch.tensor(self.X.data, requires_grad=True)
        out_ref = torch.relu(X_torch)
        out_ref.backward(torch.ones_like(out_ref))

        assert allclose(self.X.grad.data, X_torch.grad.detach().numpy(), atol=1e-6), (
            "ReLU backward pass did not match PyTorch."
        )

    def test_softmax_forward(self):
        assert allclose(
            functional.softmax(self.X).data,
            torch.nn.functional.softmax(torch.Tensor(self.X.data), dim=1)
            .detach()
            .numpy(),
            atol=1e-6,
        )

    def test_softmax_backward(self):
        out_custom = functional.softmax(self.X)
        grad_dummy = xp.ones_like(out_custom.data)
        out_custom.backward(grad_dummy)

        X_torch = torch.tensor(self.X.data, requires_grad=True)
        out_ref = torch.softmax(X_torch, dim=1)
        out_ref.backward(torch.ones_like(out_ref))

        assert allclose(self.X.grad.data, X_torch.grad.detach().numpy(), atol=1e-6), (
            "Softmax backward pass did not match PyTorch."
        )

    def test_tanh_forward(self):
        assert allclose(
            functional.tanh(self.X).data,
            torch.nn.functional.tanh(torch.Tensor(self.X.data)).detach().numpy(),
        )

    def test_tanh_backward(self):
        out_custom = functional.tanh(self.X)
        grad_dummy = xp.ones_like(out_custom.data)
        out_custom.backward(grad_dummy)

        X_torch = torch.tensor(self.X.data, requires_grad=True)
        out_ref = torch.tanh(X_torch)
        out_ref.backward(torch.ones_like(out_ref))

        assert allclose(self.X.grad.data, X_torch.grad.detach().numpy(), atol=1e-6), (
            "Tanh backward pass did not match PyTorch."
        )

    def test_gelu_forward(self):
        out_custom = functional.gelu(self.X).data

        gelu_torch = torch.nn.GELU(approximate="tanh")
        X_torch = torch.tensor(self.X.data, requires_grad=False)

        assert allclose(out_custom, gelu_torch(X_torch).detach().numpy(), atol=1e-6), (
            "Approximate GELU forward pass did not match PyTorch GELU with approximate='tanh'."
        )

    def test_gelu_backward(self):
        out_custom = functional.gelu(self.X)
        grad_dummy = xp.ones_like(out_custom.data)  # dL/dY = 1
        out_custom.backward(grad_dummy)

        gelu_torch = torch.nn.GELU(approximate="tanh")
        X_torch = torch.tensor(self.X.data, dtype=torch.float32, requires_grad=True)
        out_ref = gelu_torch(X_torch)
        out_ref.backward(torch.ones_like(out_ref))

        assert allclose(self.X.grad.data, X_torch.grad.detach().numpy(), atol=1e-6), (
            "Approximate GELU backward pass did not match PyTorch GELU with approximate='tanh'."
        )


class TestBinaryCrossEntropy(TestCase):
    def setUp(self) -> None:
        self.y_pred_logits = Tensor(data=xp.array([1.0, 2.0, 3.0]), requires_grad=True)
        self.y_pred_logits_torch = torch.tensor(
            self.y_pred_logits.data, dtype=torch.float32, requires_grad=True
        )
        self.y_pred_probs = Tensor(data=xp.array([0.2, 0.3, 0.1]), requires_grad=True)
        self.y_pred_probs_torch = torch.tensor(
            self.y_pred_probs.data, dtype=torch.float32, requires_grad=True
        )
        self.y_true = xp.array([0.0, 0.0, 1.0])

    def test_binary_cross_entropy_with_probs(self):
        bce_loss = functional.binary_cross_entropy(self.y_pred_probs, self.y_true)
        torch_bce_loss = torch.nn.functional.binary_cross_entropy(
            self.y_pred_probs_torch, torch.tensor(self.y_true, dtype=torch.float32)
        )
        assert allclose(
            bce_loss.data,
            torch_bce_loss.detach().numpy(),
        )

        bce_loss.backward()
        torch_bce_loss.backward()
        assert allclose(self.y_pred_probs.grad.data, self.y_pred_probs_torch.grad)

    def test_binary_cross_entropy_with_logits_error(self):
        with self.assertRaises((ValueError, RuntimeError)):
            allclose(
                functional.binary_cross_entropy(self.y_pred_logits, self.y_true).data,
                torch.nn.functional.binary_cross_entropy(
                    torch.tensor(self.y_pred_logits.data), torch.tensor(self.y_true)
                )
                .detach()
                .numpy(),
            )

    def test_binary_cross_entropy_with_logits(self):
        bce_loss = functional.binary_cross_entropy_with_logits(
            self.y_pred_logits, self.y_true
        )
        torch_bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            self.y_pred_logits_torch, torch.tensor(self.y_true, dtype=torch.float32)
        )
        assert allclose(
            bce_loss.data,
            torch_bce_loss.detach().numpy(),
        )

        bce_loss.backward()
        torch_bce_loss.backward()
        assert allclose(self.y_pred_logits.grad.data, self.y_pred_logits_torch.grad)


class TestCrossEntropy(TestCase):
    def test_cross_entropy_matches_pytorch_without_ignored_targets(self):
        logits = Tensor(
            data=xp.array([[2.0, 0.5, -1.0], [0.1, 0.2, 3.0]], dtype=xp.float32),
            requires_grad=True,
        )
        logits_torch = torch.tensor(
            logits.data, dtype=torch.float32, requires_grad=True
        )
        targets = xp.array([0, 2], dtype=xp.int64)

        loss = functional.cross_entropy(logits, targets)
        torch_loss = torch.nn.functional.cross_entropy(
            logits_torch, torch.tensor(targets, dtype=torch.int64)
        )

        assert allclose(loss.data, torch_loss.detach().numpy(), atol=1e-6)

        loss.backward()
        torch_loss.backward()
        assert allclose(logits.grad.data, logits_torch.grad.detach().numpy(), atol=1e-6)

    def test_cross_entropy_label_smoothing_matches_pytorch_for_2d_logits(self):
        logits = Tensor(
            data=xp.array(
                [[1.5409961, -0.2934289], [-2.1787894, 0.56843126]],
                dtype=xp.float32,
            ),
            requires_grad=True,
        )
        logits_torch = torch.tensor(
            logits.data, dtype=torch.float32, requires_grad=True
        )
        targets = xp.array([0, 1], dtype=xp.int64)

        loss = functional.cross_entropy(logits, targets, label_smoothing=0.2)
        torch_loss = torch.nn.functional.cross_entropy(
            logits_torch,
            torch.tensor(targets, dtype=torch.int64),
            label_smoothing=0.2,
        )

        assert allclose(loss.data, torch_loss.detach().numpy(), atol=1e-6)

        loss.backward()
        torch_loss.backward()
        assert allclose(logits.grad.data, logits_torch.grad.detach().numpy(), atol=1e-6)

    def test_cross_entropy_ignore_index_ignores_masked_sequence_positions(self):
        logits = Tensor(
            data=xp.array(
                [[[0.1, 2.0, -1.0], [3.0, -2.0, 0.5]]],
                dtype=xp.float32,
            ),
            requires_grad=True,
        )
        logits_torch = torch.tensor(
            logits.data[:, :1, :].reshape(1, 3),
            dtype=torch.float32,
            requires_grad=True,
        )
        targets = xp.array([[1, functional.IGNORE_INDEX]], dtype=xp.int64)

        loss = functional.cross_entropy(
            logits,
            targets,
        )
        torch_loss = torch.nn.functional.cross_entropy(
            logits_torch,
            torch.tensor([1], dtype=torch.int64),
        )

        assert allclose(loss.data, torch_loss.detach().numpy(), atol=1e-6)

        loss.backward()
        torch_loss.backward()
        expected_grad = xp.zeros_like(logits.data)
        expected_grad[:, :1, :] = logits_torch.grad.detach().numpy().reshape(1, 1, 3)
        assert allclose(logits.grad.data, expected_grad, atol=1e-6)

    def test_cross_entropy_label_smoothing_matches_pytorch_for_3d_logits(self):
        logits_data = xp.array(
            [
                [[2.0, 0.5, -1.0], [0.1, 0.2, 3.0]],
                [[-0.4, 1.2, 0.7], [1.5, -0.3, 0.0]],
            ],
            dtype=xp.float32,
        )
        logits = Tensor(data=logits_data, requires_grad=True)
        logits_torch = torch.tensor(
            logits.data, dtype=torch.float32, requires_grad=True
        )
        targets = xp.array([[0, 2], [1, 0]], dtype=xp.int64)

        loss = functional.cross_entropy(logits, targets, label_smoothing=0.1)
        torch_loss = torch.nn.functional.cross_entropy(
            logits_torch.reshape(-1, logits_torch.shape[-1]),
            torch.tensor(targets, dtype=torch.int64).reshape(-1),
            label_smoothing=0.1,
        )

        assert allclose(loss.data, torch_loss.detach().numpy(), atol=1e-6)

        loss.backward()
        torch_loss.backward()
        assert allclose(logits.grad.data, logits_torch.grad.detach().numpy(), atol=1e-6)

    def test_cross_entropy_sum_reduction_matches_pytorch_with_ignored_targets(self):
        logits = Tensor(
            data=xp.array(
                [[[0.1, 2.0, -1.0], [3.0, -2.0, 0.5]]],
                dtype=xp.float32,
            ),
            requires_grad=True,
        )
        logits_torch = torch.tensor(
            logits.data,
            dtype=torch.float32,
            requires_grad=True,
        )
        targets = xp.array([[1, functional.IGNORE_INDEX]], dtype=xp.int64)

        loss = functional.cross_entropy(logits, targets, reduction="sum")
        torch_loss = torch.nn.functional.cross_entropy(
            logits_torch.reshape(-1, logits_torch.shape[-1]),
            torch.tensor(targets, dtype=torch.int64).reshape(-1),
            ignore_index=functional.IGNORE_INDEX,
            reduction="sum",
        )

        assert allclose(loss.data, torch_loss.detach().numpy(), atol=1e-6)

        loss.backward()
        torch_loss.backward()
        assert allclose(logits.grad.data, logits_torch.grad.detach().numpy(), atol=1e-6)

    def test_cross_entropy_backward_does_not_build_dense_target_distribution(self):
        logits = Tensor(
            data=xp.array([[1.5, -0.2, 0.3], [0.1, 0.4, -1.0]], dtype=xp.float32),
            requires_grad=True,
        )
        targets = xp.array([0, 1], dtype=xp.int64)

        loss = functional.cross_entropy(logits, targets, label_smoothing=0.1)

        with patch.object(
            functional.xp,
            "ones_like",
            side_effect=AssertionError("cross entropy backward built a dense target"),
        ):
            loss.backward(xp.array(1.0, dtype=xp.float32))

        assert logits.grad is not None
        assert logits.grad.shape == logits.shape


class TestHingeLoss(TestCase):
    def setUp(self):
        # Pytorch doesn't have hinge loss
        self.y_pred = Tensor(data=xp.array([1.0, 2.0, 3.0]), requires_grad=True)
        self.y_true = xp.array([1.0, -1.0, 1.0])

    def test_hinge_loss_none(self):
        hinge_loss_none = functional.hinge_loss(
            self.y_pred, self.y_true, reduction="none"
        )
        expected_none = xp.maximum(0, 1 - self.y_true * self.y_pred.data)  # [0, 3, 0]
        assert allclose(hinge_loss_none.data, expected_none)

        # Test gradients for no reduction
        hinge_loss_none.backward()
        expected_grad_none = xp.where(
            1 - self.y_true * self.y_pred.data > 0, -self.y_true, 0
        )  # [0, 1, 0]
        assert allclose(self.y_pred.grad.data, expected_grad_none)

    def test_hinge_loss_mean(self):
        hinge_loss_mean = functional.hinge_loss(
            self.y_pred, self.y_true, reduction="mean"
        )
        expected_mean = xp.mean(
            xp.maximum(0, 1 - self.y_true * self.y_pred.data)
        )  # 1.0
        assert allclose(hinge_loss_mean.data, expected_mean)

        # Test gradients for mean reduction
        self.y_pred.grad = None  # Reset gradients
        hinge_loss_mean.backward()
        expected_grad_mean = xp.where(
            1 - self.y_true * self.y_pred.data > 0, -self.y_true, 0
        ) / len(self.y_true)  # [0, 1/3, 0]
        assert allclose(self.y_pred.grad.data, expected_grad_mean)

    def test_hinge_loss_sum(self):
        hinge_loss_sum = functional.hinge_loss(
            self.y_pred, self.y_true, reduction="sum"
        )
        expected_sum = xp.sum(xp.maximum(0, 1 - self.y_true * self.y_pred.data))  # 3.0
        assert allclose(hinge_loss_sum.data, expected_sum)

        # Test gradients for sum reduction
        self.y_pred.grad = None  # Reset gradients
        hinge_loss_sum.backward()
        expected_grad_sum = xp.where(
            1 - self.y_true * self.y_pred.data > 0, -self.y_true, 0
        )  # [0, 1, 0]
        assert allclose(self.y_pred.grad.data, expected_grad_sum)

    def test_hinge_loss_invalid_reduction(self):
        with self.assertRaises(ValueError):
            functional.hinge_loss(self.y_pred, self.y_true, reduction="invalid")


class TestMeanSquaredLoss(TestCase):
    def setUp(self):
        # Explicitly use float32 dtype
        self.y_pred = Tensor(
            data=xp.array([1.0, 2.0, 3.0], dtype=xp.float32), requires_grad=True
        )
        self.y_true = xp.array([1.0, 2.0, 3.0], dtype=xp.float32)
        self.y_pred_torch = torch.tensor(
            self.y_pred.data, dtype=torch.float32, requires_grad=True
        )
        self.y_true_torch = torch.tensor(self.y_true, dtype=torch.float32)

    def test_mean_squared_loss(self):
        mse_loss = functional.mean_squared_loss(self.y_pred, self.y_true)
        mse_loss_torch = torch.nn.functional.mse_loss(
            self.y_pred_torch, self.y_true_torch
        )
        assert allclose(mse_loss.data, mse_loss_torch.detach().numpy())

        mse_loss.backward()
        mse_loss_torch.backward()
        assert allclose(self.y_pred.grad.data, self.y_pred_torch.grad.detach().numpy())
