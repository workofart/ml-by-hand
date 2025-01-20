import numpy as np
import torch  # for test comparisons
from autograd.tensor import Tensor
from autograd import functional
from unittest import TestCase


class TestActivationFunctions(TestCase):
    # TODO: add backward pass tests
    def setUp(self) -> None:
        self.X = Tensor(data=np.array([[1, 1, 1], [2, 2, 2]]), requires_grad=True)

    def test_sigmoid(self):
        assert np.allclose(
            functional.sigmoid(self.X).data,
            torch.nn.functional.sigmoid(torch.Tensor(self.X.data)).detach().numpy(),
        )

    def test_relu(self):
        assert np.allclose(
            functional.relu(self.X).data,
            torch.nn.functional.relu(torch.Tensor(self.X.data)).detach().numpy(),
        )

    def test_softmax(self):
        assert np.allclose(
            functional.softmax(self.X).data,
            torch.nn.functional.softmax(torch.Tensor(self.X.data), dim=1)
            .detach()
            .numpy(),
            atol=1e-6,
        )

    def test_tanh(self):
        assert np.allclose(
            functional.tanh(self.X).data,
            torch.nn.functional.tanh(torch.Tensor(self.X.data)).detach().numpy(),
        )


class TestBinaryCrossEntropy(TestCase):
    def setUp(self) -> None:
        self.y_pred_logits = Tensor(data=np.array([1.0, 2.0, 3.0]), requires_grad=True)
        self.y_pred_logits_torch = torch.tensor(
            self.y_pred_logits.data, requires_grad=True
        )
        self.y_pred_probs = Tensor(data=np.array([0.2, 0.3, 0.1]), requires_grad=True)
        self.y_pred_probs_torch = torch.tensor(
            self.y_pred_probs.data, requires_grad=True
        )
        self.y_true = np.array([0.0, 0.0, 1.0])

    def test_binary_cross_entropy_with_probs(self):
        bce_loss = functional.binary_cross_entropy(self.y_pred_probs, self.y_true)
        torch_bce_loss = torch.nn.functional.binary_cross_entropy(
            self.y_pred_probs_torch, torch.tensor(self.y_true, dtype=torch.float32)
        )
        assert np.allclose(
            bce_loss.data,
            torch_bce_loss.detach().numpy(),
        )

        bce_loss.backward()
        torch_bce_loss.backward()
        assert np.allclose(self.y_pred_probs.grad.data, self.y_pred_probs_torch.grad)

    def test_binary_cross_entropy_with_logits_error(self):
        with self.assertRaises((ValueError, RuntimeError)):
            np.allclose(
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
            self.y_pred_logits_torch, torch.tensor(self.y_true)
        )
        assert np.allclose(
            bce_loss.data,
            torch_bce_loss.detach().numpy(),
        )

        bce_loss.backward()
        torch_bce_loss.backward()
        assert np.allclose(self.y_pred_logits.grad.data, self.y_pred_logits_torch.grad)


class TestCrossEntropyWithLogits(TestCase):
    def setUp(self):
        # 2D case: (batch_size=3, num_classes=4)
        self.logits_2d = Tensor(
            data=np.array(
                [[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 0.0, 3.0], [0.0, 3.0, 1.0, 2.0]],
                dtype=np.float32,
            ),
            requires_grad=True,
        )
        self.targets_2d = np.array(
            [[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float32,
        )

        # Create PyTorch tensors for comparison
        self.logits_2d_torch = torch.tensor(self.logits_2d.data, requires_grad=True)
        self.targets_2d_torch = torch.tensor(self.targets_2d)

    def test_cross_entropy_with_logits_2d(self):
        # Test our implementation
        loss = functional.cross_entropy_with_logits(self.logits_2d, self.targets_2d)

        # Test PyTorch implementation
        loss_torch = torch.nn.functional.cross_entropy(
            self.logits_2d_torch,
            self.targets_2d_torch.argmax(dim=1),  # PyTorch expects class indices
        )

        # Compare forward pass
        assert np.allclose(loss.data, loss_torch.detach().numpy(), atol=1e-5)

        # Compare backward pass
        loss.backward()
        loss_torch.backward()

        assert np.allclose(
            self.logits_2d.grad.data,
            self.logits_2d_torch.grad.detach().numpy(),
            atol=1e-5,
        )

    def test_cross_entropy_with_logits_3d(self):
        # 3D case: (batch_size=2, sequence_length=3, num_classes=4)
        logits_3d = Tensor(
            data=np.random.randn(2, 3, 4).astype(np.float32), requires_grad=True
        )
        targets_3d = np.zeros((2, 3, 4), dtype=np.float32)
        # Set one class as target for each position
        targets_3d[
            np.arange(2)[:, None], np.arange(3), np.random.randint(0, 4, (2, 3))
        ] = 1.0

        logits_3d_torch = torch.tensor(logits_3d.data, requires_grad=True)
        targets_3d_torch = torch.tensor(targets_3d)

        # Test our implementation
        loss = functional.cross_entropy_with_logits(logits_3d, targets_3d)

        # Test PyTorch implementation
        loss_torch = torch.nn.functional.cross_entropy(
            logits_3d_torch.reshape(-1, 4),
            targets_3d_torch.reshape(-1, 4).argmax(dim=1),
        )

        # Compare forward pass
        assert np.allclose(loss.data, loss_torch.detach().numpy(), atol=1e-5)

        # Compare backward pass
        loss.backward()
        loss_torch.backward()

        assert np.allclose(
            logits_3d.grad.data, logits_3d_torch.grad.detach().numpy(), atol=1e-5
        )


class TestSparseCrossEntropy(TestCase):
    def setUp(self) -> None:
        self.y_pred_logits = Tensor(
            data=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
            requires_grad=True,
        )
        self.y_pred_logits_torch = torch.tensor(
            self.y_pred_logits.data, requires_grad=True
        )
        self.y_pred_probs = Tensor(
            data=np.array(
                [
                    [0.0900, 0.2447, 0.6652],
                    [0.0900, 0.2447, 0.6652],
                    [0.0900, 0.2447, 0.6652],
                ],
                dtype=np.float64,
            ),
            requires_grad=True,
        )
        self.y_true = np.array([2, 0, 1])

    def test_sparse_cross_entropy_with_logits(self):
        sce_loss = functional.sparse_cross_entropy_with_logits(
            self.y_pred_logits,
            self.y_true,
        )
        ce_loss_torch = torch.nn.functional.cross_entropy(
            self.y_pred_logits_torch,
            torch.LongTensor(self.y_true),
        )
        assert np.allclose(
            sce_loss.data,
            ce_loss_torch.detach().numpy(),
        )

        sce_loss.backward()
        ce_loss_torch.backward()
        assert np.allclose(self.y_pred_logits.grad.data, self.y_pred_logits_torch.grad)

    def test_sparse_cross_entropy_with_probs(self):
        sce_loss = functional.sparse_cross_entropy(
            self.y_pred_probs,
            self.y_true,
        )
        ce_loss_torch = torch.nn.functional.cross_entropy(
            self.y_pred_logits_torch,
            torch.LongTensor(self.y_true),
        )
        assert np.allclose(
            sce_loss.data,
            ce_loss_torch.detach().numpy(),
            atol=1e-3,
        )

    def test_sparse_cross_entropy_invalid_input(self):
        with self.assertRaises(ValueError):
            functional.sparse_cross_entropy(
                self.y_pred_logits,
                self.y_true,
            )
            torch.nn.functional.cross_entropy(
                self.y_pred_logits_torch,
                torch.LongTensor(self.y_true),
            )


class TestHingeLoss(TestCase):
    def setUp(self):
        # Pytorch doesn't have hinge loss
        self.y_pred = Tensor(data=np.array([1.0, 2.0, 3.0]), requires_grad=True)
        self.y_true = np.array([1.0, -1.0, 1.0])

    def test_hinge_loss_none(self):
        hinge_loss_none = functional.hinge_loss(
            self.y_pred, self.y_true, reduction="none"
        )
        expected_none = np.maximum(0, 1 - self.y_true * self.y_pred.data)  # [0, 3, 0]
        assert np.allclose(hinge_loss_none.data, expected_none)

        # Test gradients for no reduction
        hinge_loss_none.backward()
        expected_grad_none = np.where(
            1 - self.y_true * self.y_pred.data > 0, -self.y_true, 0
        )  # [0, 1, 0]
        assert np.allclose(self.y_pred.grad.data, expected_grad_none)

    def test_hinge_loss_mean(self):
        hinge_loss_mean = functional.hinge_loss(
            self.y_pred, self.y_true, reduction="mean"
        )
        expected_mean = np.mean(
            np.maximum(0, 1 - self.y_true * self.y_pred.data)
        )  # 1.0
        assert np.allclose(hinge_loss_mean.data, expected_mean)

        # Test gradients for mean reduction
        self.y_pred.grad = None  # Reset gradients
        hinge_loss_mean.backward()
        expected_grad_mean = np.where(
            1 - self.y_true * self.y_pred.data > 0, -self.y_true, 0
        ) / len(self.y_true)  # [0, 1/3, 0]
        assert np.allclose(self.y_pred.grad.data, expected_grad_mean)

    def test_hinge_loss_sum(self):
        hinge_loss_sum = functional.hinge_loss(
            self.y_pred, self.y_true, reduction="sum"
        )
        expected_sum = np.sum(np.maximum(0, 1 - self.y_true * self.y_pred.data))  # 3.0
        assert np.allclose(hinge_loss_sum.data, expected_sum)

        # Test gradients for sum reduction
        self.y_pred.grad = None  # Reset gradients
        hinge_loss_sum.backward()
        expected_grad_sum = np.where(
            1 - self.y_true * self.y_pred.data > 0, -self.y_true, 0
        )  # [0, 1, 0]
        assert np.allclose(self.y_pred.grad.data, expected_grad_sum)

    def test_hinge_loss_invalid_reduction(self):
        with self.assertRaises(ValueError):
            functional.hinge_loss(self.y_pred, self.y_true, reduction="invalid")


class TestMeanSquaredLoss(TestCase):
    def setUp(self):
        # Explicitly use float32 dtype
        self.y_pred = Tensor(
            data=np.array([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True
        )
        self.y_true = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        self.y_pred_torch = torch.tensor(
            self.y_pred.data, dtype=torch.float32, requires_grad=True
        )
        self.y_true_torch = torch.tensor(self.y_true, dtype=torch.float32)

    def test_mean_squared_loss(self):
        mse_loss = functional.mean_squared_loss(self.y_pred, self.y_true)
        mse_loss_torch = torch.nn.functional.mse_loss(
            self.y_pred_torch, self.y_true_torch
        )
        assert np.allclose(mse_loss.data, mse_loss_torch.detach().numpy())

        mse_loss.backward()
        mse_loss_torch.backward()
        assert np.allclose(
            self.y_pred.grad.data, self.y_pred_torch.grad.detach().numpy()
        )
