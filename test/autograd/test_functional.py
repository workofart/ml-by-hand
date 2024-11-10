import numpy as np
import torch
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
            torch.nn.functional.softmax(torch.Tensor(self.X.data), dim=1).detach().numpy(),
            atol=1e-6,
        )
        
class TestLossFunctions(TestCase):
    def setUp(self) -> None:
        self.y_pred_logits = Tensor(data=np.array([1, 2, 3]), requires_grad=True)
        self.y_pred_probs = Tensor(data=np.array([0.2, 0.3, 0.1]), requires_grad=True)
        self.y_true = np.array([0.0, 0.0, 1.0])

    def test_binary_cross_entropy(self):
        # Pass in probabilities
        assert np.allclose(
            functional.binary_cross_entropy(self.y_pred_probs, self.y_true).data,
            torch.nn.functional.binary_cross_entropy(
                torch.Tensor(self.y_pred_probs.data), torch.Tensor(self.y_true)
            ).detach().numpy(),
        )
        
        # Pass in logits, not probabilities
        with self.assertRaises((ValueError, RuntimeError)):
            np.allclose(
                functional.binary_cross_entropy(self.y_pred_logits, self.y_true).data,
                torch.nn.functional.binary_cross_entropy(
                    torch.Tensor(self.y_pred_logits.data), torch.Tensor(self.y_true)
                ).detach().numpy(),
            )

        # Call the logits BCE function
        assert np.allclose(
            functional.binary_cross_entropy_with_logits(self.y_pred_logits, self.y_true).data,
            torch.nn.functional.binary_cross_entropy_with_logits(
                torch.Tensor(self.y_pred_logits.data), torch.Tensor(self.y_true)
            ).detach().numpy(),
        )
    
    def test_sparse_cross_entropy(self):
        y_pred_logits = Tensor(data=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), requires_grad=True)
        y_pred_probs = torch.nn.functional.softmax(torch.Tensor(y_pred_logits.data), dim=1).detach().numpy()
        y_true = np.array([2,0,1])
        
        # Pass in probabilities
        # Torch accepts logits
        assert np.allclose(
            functional.sparse_cross_entropy(
                Tensor(y_pred_probs),
                y_true,
            ).data,
            torch.nn.functional.cross_entropy(
                torch.Tensor(y_pred_logits.data),
                torch.LongTensor(y_true),
            ).detach().numpy(),
        )
        
        # Pass in logits, not probabilities
        with self.assertRaises(ValueError):
            np.allclose(
                functional.sparse_cross_entropy(
                    y_pred_logits,
                    y_true,
                ).data,
                torch.nn.functional.cross_entropy(
                    torch.Tensor(y_pred_logits.data),
                    torch.LongTensor(y_true),
                ).detach().numpy(),
            )
        
        # Call the logits SCE function
        assert np.allclose(
            functional.sparse_cross_entropy_with_logits(
                y_pred_logits,
                y_true,
            ).data,
            torch.nn.functional.cross_entropy(
                torch.Tensor(y_pred_logits.data),
                torch.LongTensor(y_true),
            ).detach().numpy(),
        )