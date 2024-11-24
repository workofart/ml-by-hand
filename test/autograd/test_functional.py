import numpy as np
import torch  # for test comparisons
from autograd.tensor import Tensor
from autograd import functional
from unittest import TestCase


class TestActivationFunctions(TestCase):
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


class TestLossFunctions(TestCase):
    def test_binary_cross_entropy(self):
        y_pred_logits = Tensor(data=np.array([1.0, 2.0, 3.0]), requires_grad=True)
        y_pred_logits_torch = torch.tensor(y_pred_logits.data, requires_grad=True)
        y_pred_probs = Tensor(data=np.array([0.2, 0.3, 0.1]), requires_grad=True)
        y_pred_probs_torch = torch.tensor(y_pred_probs.data, requires_grad=True)
        y_true = np.array([0.0, 0.0, 1.0])

        # Test case 1: Pass in probabilities
        bce_loss = functional.binary_cross_entropy(y_pred_probs, y_true)
        torch_bce_loss = torch.nn.functional.binary_cross_entropy(
            y_pred_probs_torch, torch.tensor(y_true)
        )
        assert np.allclose(
            bce_loss.data,
            torch_bce_loss.detach().numpy(),
        )

        bce_loss.backward()
        torch_bce_loss.backward()
        assert np.allclose(y_pred_probs.grad.data, y_pred_probs_torch.grad)

        # Test Case 2: Pass in logits, not probabilities
        with self.assertRaises((ValueError, RuntimeError)):
            np.allclose(
                functional.binary_cross_entropy(y_pred_logits, y_true).data,
                torch.nn.functional.binary_cross_entropy(
                    torch.tensor(y_pred_logits.data), torch.tensor(y_true)
                )
                .detach()
                .numpy(),
            )

        # Test Case 3: Call the logits BCE function
        bce_loss = functional.binary_cross_entropy_with_logits(y_pred_logits, y_true)
        torch_bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            y_pred_logits_torch, torch.tensor(y_true)
        )
        assert np.allclose(
            bce_loss.data,
            torch_bce_loss.detach().numpy(),
        )

        bce_loss.backward()
        torch_bce_loss.backward()
        assert np.allclose(y_pred_logits.grad.data, y_pred_logits_torch.grad)

    def test_sparse_cross_entropy(self):
        y_pred_logits = Tensor(
            data=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
            requires_grad=True,
        )
        y_pred_logits_torch = torch.tensor(y_pred_logits.data, requires_grad=True)
        y_pred_probs = Tensor(
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
        y_true = np.array([2, 0, 1])

        # Test Case 1: Pass in logits
        sce_loss = functional.sparse_cross_entropy_with_logits(
            y_pred_logits,
            y_true,
        )
        ce_loss_torch = torch.nn.functional.cross_entropy(
            y_pred_logits_torch,
            torch.LongTensor(y_true),
        )
        assert np.allclose(
            sce_loss.data,
            ce_loss_torch.detach().numpy(),
        )

        sce_loss.backward()
        ce_loss_torch.backward()
        assert np.allclose(y_pred_logits.grad.data, y_pred_logits_torch.grad)

        # Test Case 2: Pass in probabilities
        # Torch accepts logits
        sce_loss = functional.sparse_cross_entropy(
            y_pred_probs,
            y_true,
        )
        ce_loss_torch = torch.nn.functional.cross_entropy(
            y_pred_logits_torch,
            torch.LongTensor(y_true),
        )
        assert np.allclose(
            sce_loss.data,
            ce_loss_torch.detach().numpy(),
            atol=1e-3,
        )

        # Pass in logits, not probabilities
        with self.assertRaises(ValueError):
            np.allclose(
                functional.sparse_cross_entropy(
                    y_pred_logits,
                    y_true,
                ).data,
                torch.nn.functional.cross_entropy(
                    y_pred_logits_torch,
                    torch.LongTensor(y_true),
                )
                .detach()
                .numpy(),
            )

    def test_hinge_loss(self):
        y_pred = Tensor(data=np.array([1.0, 2.0, 3.0]), requires_grad=True)
        y_true = np.array([1.0, -1.0, 1.0])

        # Test no reduction
        hinge_loss_none = functional.hinge_loss(y_pred, y_true, reduction="none")
        expected_none = np.maximum(0, 1 - y_true * y_pred.data)  # [0, 3, 0]
        assert np.allclose(hinge_loss_none.data, expected_none)

        # Test mean reduction
        hinge_loss_mean = functional.hinge_loss(y_pred, y_true, reduction="mean")
        expected_mean = np.mean(expected_none)  # 1.0
        assert np.allclose(hinge_loss_mean.data, expected_mean)

        # Test sum reduction
        hinge_loss_sum = functional.hinge_loss(y_pred, y_true, reduction="sum")
        expected_sum = np.sum(expected_none)  # 3.0
        assert np.allclose(hinge_loss_sum.data, expected_sum)

        # Test gradients for each reduction
        # No reduction
        hinge_loss_none.backward()
        expected_grad_none = np.where(
            1 - y_true * y_pred.data > 0, -y_true, 0
        )  # [0, 1, 0]
        assert np.allclose(y_pred.grad.data, expected_grad_none)
        y_pred.grad = None  # Reset gradients

        # Mean reduction
        y_pred_mean = Tensor(data=np.array([1.0, 2.0, 3.0]), requires_grad=True)
        hinge_loss_mean = functional.hinge_loss(y_pred_mean, y_true, reduction="mean")
        hinge_loss_mean.backward()
        expected_grad_mean = expected_grad_none / len(expected_grad_none)  # [0, 1/3, 0]
        assert np.allclose(y_pred_mean.grad.data, expected_grad_mean)

        # Sum reduction
        y_pred_sum = Tensor(data=np.array([1.0, 2.0, 3.0]), requires_grad=True)
        hinge_loss_sum = functional.hinge_loss(y_pred_sum, y_true, reduction="sum")
        hinge_loss_sum.backward()
        expected_grad_sum = expected_grad_none  # [0, 1, 0]
        assert np.allclose(y_pred_sum.grad.data, expected_grad_sum)

        # Test invalid reduction
        with self.assertRaises(ValueError):
            functional.hinge_loss(y_pred, y_true, reduction="invalid")
