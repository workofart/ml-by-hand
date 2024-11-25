from autograd.tensor import Tensor
import numpy as np
import torch  # for comparison
from unittest import TestCase


class TestTensor(TestCase):
    def setUp(self) -> None:
        self.x_scalar = Tensor(2.0, requires_grad=True)
        self.y_scalar = Tensor(3.0, requires_grad=True)
        self.x_vector = Tensor([1.0, 2.0], requires_grad=True)
        self.y_vector = Tensor([3.0, 4.0], requires_grad=True)
        self.x_vector_negative = Tensor([-1.0, -2.0, -3.0], requires_grad=True)
        self.y_vector_negative = Tensor([-2.0, -1.0, -3.0], requires_grad=True)
        self.x_vector_no_grad = Tensor([1.0, 2.0], requires_grad=False)
        self.y_vector_no_grad = Tensor([2.0, 1.0], requires_grad=False)
        self.x_matrix = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        self.y_matrix = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        self.x_matrix_no_grad = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=False)

        self.three_d_matrix = Tensor(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
        )
        self.three_by_three_matrix = Tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            requires_grad=True,
        )
        self.four_d_matrix = Tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)

        # Torch tensors for comparison
        self.x_vector_torch = torch.tensor(self.x_vector.data, requires_grad=True)
        self.y_vector_torch = torch.tensor(self.y_vector.data, requires_grad=True)
        self.x_vector_no_grad_torch = torch.tensor(
            self.x_vector_no_grad.data, requires_grad=False
        )
        self.y_vector_no_grad_torch = torch.tensor(
            self.y_vector_no_grad.data, requires_grad=False
        )
        self.x_vector_negative_torch = torch.tensor(
            self.x_vector_negative.data, requires_grad=True
        )
        self.y_vector_negative_torch = torch.tensor(
            self.y_vector_negative.data, requires_grad=True
        )
        self.x_matrix_torch = torch.tensor(self.x_matrix.data, requires_grad=True)
        self.y_matrix_torch = torch.tensor(self.y_matrix.data, requires_grad=True)
        self.x_matrix_no_grad_torch = torch.tensor(
            self.x_matrix_no_grad.data, requires_grad=False
        )
        self.three_d_matrix_torch = torch.tensor(
            self.three_d_matrix.data, requires_grad=True
        )
        self.three_by_three_matrix_torch = torch.tensor(
            self.three_by_three_matrix.data, requires_grad=True
        )
        self.four_d_matrix_torch = torch.tensor(
            self.four_d_matrix.data, requires_grad=True
        )


class TestTensorOps(TestTensor):
    def test_tensor_negation(self):
        assert (-self.x_scalar).data == -2.0

    def test_tensor_addition(self):
        assert (self.x_scalar + self.y_scalar).data == 5.0
        assert (self.x_scalar + self.y_scalar).prev == {self.x_scalar, self.y_scalar}

    def test_tensor_multiplication(self):
        assert (self.x_scalar * self.y_scalar).data == 6.0
        assert (self.x_scalar * self.y_scalar).prev == {self.x_scalar, self.y_scalar}

    def test_tensor_subtraction(self):
        assert (self.x_scalar - self.y_scalar).data == -1.0
        assert (self.y_scalar - self.x_scalar).data == 1.0

    def test_tensor_division(self):
        assert (self.x_scalar / self.y_scalar).data == 2.0 / 3.0
        assert (self.y_scalar / self.x_scalar).data == 1.5

    def test_tensor_exponentiation(self):
        assert (self.x_scalar**self.y_scalar).data == 8.0
        assert (self.y_scalar**self.x_scalar).data == 9.0

    def test_tensor_gradients(self):
        assert self.x_scalar.grad is None  # lazy init until backward is called
        assert self.y_scalar.grad is None  # lazy init until backward is called
        assert self.x_scalar.requires_grad
        assert len(self.y_scalar.prev) == 0

    def test_tensor_matrix_multiplication(self):
        z = self.x_vector @ self.y_vector
        assert np.array_equal(z.data, 1.0 * 3.0 + 2.0 * 4.0)

    def test_complex_tensor_ops(self):
        x = Tensor(2.0, requires_grad=True)
        y = Tensor(1.5, requires_grad=True)
        z = Tensor(4.0, requires_grad=True)

        assert ((x * y + z) ** 2).data == 49.0

    def test_backward_multiplication(self):
        z = self.x_scalar * self.y_scalar

        assert z.data == 6.0
        assert z.prev == {self.x_scalar, self.y_scalar}
        assert z.grad is None

        # Call backward and check the gradients
        z.backward()
        assert z.grad.data == 1.0
        assert self.y_scalar.grad.data == 2.0  # dz/dy = d(y*x)/dy = x = 2.0
        assert self.x_scalar.grad.data == 3.0  # dz/dx = d(y*x)/dx = y = 3.0

    def test_backward_division(self):
        z = self.x_scalar / self.y_scalar
        z.backward()
        assert z.grad.data == 1.0
        assert np.isclose(
            self.x_scalar.grad.data, 1.0 / 3.0, atol=1e-5
        )  # dz/dx = d(x/y)/dx = 1/y = 1/3
        assert np.isclose(
            self.y_scalar.grad.data, -2.0 / 9.0, atol=1e-5
        )  # dz/dy = d(x/y)/dy = -x/y^2 = -2/9

    def test_backward_scalar_vector_matmul(self):
        x = Tensor(2.0, requires_grad=True)
        y = Tensor([3.0, 4.0], requires_grad=True)
        self.assertRaises(ValueError, lambda: x @ y)

    def test_backward_vector_vector_matmul(self):
        z = self.x_vector @ self.y_vector
        z.backward()
        assert z.grad.data == 1
        assert np.array_equal(self.x_vector.grad.data, np.array([3.0, 4.0]).T)
        assert np.array_equal(self.y_vector.grad.data, np.array([1.0, 2.0]).T)

    def test_backward_matrix_matrix_matmul(self):
        z = self.x_matrix @ self.y_matrix
        z.backward()
        assert np.array_equal(z.data, np.array([[19.0, 22.0], [43.0, 50.0]]))
        assert np.array_equal(
            self.x_matrix.grad.data, np.array([[11.0, 15.0], [11.0, 15.0]])
        )
        assert np.array_equal(
            self.y_matrix.grad.data, np.array([[4.0, 4.0], [6.0, 6.0]])
        )
        assert np.array_equal(z.grad.data, np.array([[1, 1], [1, 1]]))

    def test_backward_batch_matmul(self):
        # Similar shapes to our Conv2d windows case
        batch_size, windows, features = 3, 784, 144
        out_channels = 32

        x = Tensor(np.random.randn(batch_size, windows, features))
        w = Tensor(np.random.randn(out_channels, features))

        # Forward: (3, 784, 144) @ (32, 144).T -> (3, 784, 32)
        z = x @ w.T

        # Backward
        grad = np.ones_like(z.data)
        z.backward(grad)

        # Verify gradient shapes
        assert x.grad.shape == (batch_size, windows, features)
        assert w.grad.shape == (out_channels, features)

        # Compare with PyTorch
        x_torch = torch.tensor(x.data, requires_grad=True)
        w_torch = torch.tensor(w.data, requires_grad=True)

        z_torch = x_torch @ w_torch.T
        z_torch.backward(torch.ones_like(z_torch))

        # Use allclose instead of array_equal for floating point comparison
        np.testing.assert_allclose(x.grad.data, x_torch.grad.numpy(), rtol=1e-10)
        np.testing.assert_allclose(w.grad.data, w_torch.grad.numpy(), rtol=1e-10)

    def test_batched_matmul_gradient_computation(self):
        # Create small batch example with explicit float dtype
        batch_size = 2
        x = Tensor(
            np.array(
                [  # (2, 2, 2)
                    [[1.0, 1.0], [1.0, 1.0]],
                    [[1.0, 1.0], [1.0, 1.0]],
                ],
                dtype=np.float64,
            )
        )
        w = Tensor(
            np.array(
                [  # (2, 2)
                    [1.0, 1.0],
                    [1.0, 1.0],
                ],
                dtype=np.float64,
            )
        )

        # Manual batch-wise computation
        manual_x_grad = np.zeros_like(x.data)  # (2, 2, 2)
        manual_w_grad = np.zeros_like(w.data)  # (2, 2)

        for b in range(batch_size):
            # Compute gradient for each batch separately
            batch_grad = np.ones((2, 2), dtype=np.float64)  # gradient for this batch

            # Gradient for x[b]
            manual_x_grad[b] = np.matmul(batch_grad, w.data)  # (2,2) @ (2,2)

            # Accumulate gradient for w
            manual_w_grad += np.matmul(x.data[b].T, batch_grad)  # (2,2).T @ (2,2)

        # Compare with PyTorch
        x_torch = torch.tensor(x.data, requires_grad=True)
        w_torch = torch.tensor(w.data, requires_grad=True)
        z_torch = x_torch @ w_torch.T
        z_torch.backward(torch.ones_like(z_torch))

        # Verify our manual computation matches PyTorch
        np.testing.assert_array_equal(manual_x_grad, x_torch.grad.numpy())
        np.testing.assert_array_equal(manual_w_grad, w_torch.grad.numpy())

    def test_shift_invariance(self):
        x_data = np.array([[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]], dtype=np.float32)
        x = Tensor(x_data)

        # Original computation
        batch_mean = x.mean(axis=0)
        diff = x - batch_mean
        var = (diff**2).sum(axis=0)
        biased_batch_var = var / x.data.shape[0]
        std_dev = (biased_batch_var + 1e-5) ** 0.5
        normalized = diff / std_dev

        # Add constant shift
        shift = 100
        x_shifted = x + shift

        # Shifted computation
        batch_mean_shifted = x_shifted.mean(axis=0)
        diff_shifted = x_shifted - batch_mean_shifted
        var_shifted = (diff_shifted**2).sum(axis=0)
        biased_batch_var_shifted = var_shifted / x_shifted.data.shape[0]
        std_dev_shifted = (biased_batch_var_shifted + 1e-5) ** 0.5
        normalized_shifted = diff_shifted / std_dev_shifted
        assert np.allclose(normalized.data, normalized_shifted.data)


class TestTensorSum(TestTensor):
    def test_x_scalar_sum(self):
        s = self.x_scalar.sum()
        assert s.data == 2.0
        assert s.requires_grad == self.x_scalar.requires_grad

    def test_1d_tensor_sum_global(self):
        s = self.x_vector.sum()
        assert s.data == 3.0
        assert s.prev == {self.x_vector}

    def test_1d_tensor_sum_axis(self):
        s = self.x_vector.sum(axis=0)
        assert s.data == 3.0

    def test_2d_tensor_sum_global(self):
        s = self.x_matrix.sum()
        assert s.data == 10.0

    def test_2d_tensor_sum_axis_0(self):
        s = self.x_matrix.sum(axis=0)  # (2, 2) -> (2,)
        assert np.array_equal(s.data, [4.0, 6.0])

    def test_2d_tensor_sum_axis_1(self):
        s = self.x_matrix.sum(axis=1)  # (2, 2) -> (2,)
        assert np.array_equal(s.data, [3.0, 7.0])

    def test_2d_tensor_sum_keepdims(self):
        s = self.x_matrix.sum(axis=0, keepdims=True)
        assert s.data.shape == (1, 2)  # (2,2) -> (1,2)
        assert np.array_equal(s.data, [[4.0, 6.0]])

    def test_3d_tensor_sum(self):
        s = self.three_d_matrix.sum(axis=(1, 2))  # (2,2,2) -> (2,)
        assert np.array_equal(s.data, [10.0, 26.0])

    def test_requires_grad_propagation(self):
        s = self.x_vector_no_grad.sum()
        assert not s.requires_grad


class TestTensorMean(TestTensor):
    def setUp(self) -> None:
        super().setUp()
        self.keepdims_tensor = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        self.no_grad_tensor = Tensor([1.0, 2.0, 3.0], requires_grad=False)

    def test_scalar_tensor_mean(self):
        m = self.x_scalar.mean()
        assert m.data == 2.0
        assert m.requires_grad == self.x_scalar.requires_grad

    def test_1d_tensor_mean_global(self):
        m = self.x_vector.mean()
        assert m.data == 1.5
        assert m.prev == {self.x_vector}

    def test_1d_tensor_mean_axis(self):
        m = self.x_vector.mean(axis=0)
        assert m.data == 1.5

    def test_2d_tensor_mean_global(self):
        m = self.x_matrix.mean()
        assert m.data == 2.5

    def test_2d_tensor_mean_axis_0(self):
        m = self.x_matrix.mean(axis=0)  # (2, 2) -> (2,)
        assert np.array_equal(m.data, [2.0, 3.0])

    def test_2d_tensor_mean_axis_1(self):
        m = self.x_matrix.mean(axis=1)  # (2, 2) -> (2,)
        assert np.array_equal(m.data, [1.5, 3.5])

    def test_2d_tensor_mean_keepdims(self):
        m = self.keepdims_tensor.mean(axis=0, keepdims=True)
        assert m.data.shape == (1, 2)  # (2,2) -> (1,2)
        assert np.array_equal(m.data, [[2.0, 3.0]])

    def test_3d_tensor_mean(self):
        m = self.three_d_matrix.mean(axis=(1, 2))  # (2,2,2) -> (2,)
        assert np.array_equal(m.data, [2.5, 6.5])

    def test_multiple_axis_mean(self):
        m = self.three_d_matrix.mean(axis=(0, 1))  # (2,2,2) -> (2,)
        assert np.array_equal(m.data, [4.0, 5.0])

    def test_requires_grad_propagation(self):
        m = self.no_grad_tensor.mean()
        assert not m.requires_grad


class TestTensorMaximum(TestTensor):
    def test_maximum_basic_vector(self):
        z = self.x_vector.maximum(self.y_vector)
        z_torch = torch.maximum(self.x_vector_torch, self.y_vector_torch)
        assert np.array_equal(z.data, z_torch.detach().numpy())
        z.backward()
        z_torch.backward(torch.ones_like(z_torch))
        assert np.array_equal(self.x_vector.grad.data, self.x_vector_torch.grad.numpy())
        assert np.array_equal(self.y_vector.grad.data, self.y_vector_torch.grad.numpy())

    def test_maximum_with_scalar(self):
        z = self.x_vector.maximum(2.0)
        z_torch = torch.maximum(self.x_vector_torch, torch.tensor(2.0))
        assert np.array_equal(z.data, z_torch.detach().numpy())
        z.backward()
        z_torch.backward(torch.ones_like(z_torch))
        assert np.array_equal(self.x_vector.grad.data, self.x_vector_torch.grad.numpy())

    def test_2d_tensor_maximum(self):
        z = self.x_matrix.maximum(self.y_matrix)
        z_torch = torch.maximum(self.x_matrix_torch, self.y_matrix_torch)
        assert np.array_equal(z.data, z_torch.detach().numpy())
        z.backward()
        z_torch.backward(torch.ones_like(z_torch))
        assert np.array_equal(self.x_matrix.grad.data, self.x_matrix_torch.grad.numpy())
        assert np.array_equal(self.y_matrix.grad.data, self.y_matrix_torch.grad.numpy())

    def test_broadcasting(self):
        # Change the order of arguments for consistent broadcasting
        z = self.y_vector.maximum(
            self.x_matrix
        )  # y_vector shape: (2,) will be broadcast to (2,2)
        z_torch = torch.maximum(self.y_vector_torch, self.x_matrix_torch)
        assert np.array_equal(z.data, z_torch.detach().numpy())
        z.backward()
        z_torch.backward(torch.ones_like(z_torch))
        assert np.array_equal(self.x_matrix.grad.data, self.x_matrix_torch.grad.numpy())
        assert np.array_equal(self.y_vector.grad.data, self.y_vector_torch.grad.numpy())

    def test_requires_grad_propagation(self):
        z = self.x_vector_no_grad.maximum(self.y_vector_no_grad)
        z_torch = torch.maximum(
            self.x_vector_no_grad_torch, self.y_vector_no_grad_torch
        )
        assert z.requires_grad == z_torch.requires_grad

        x_grad = Tensor(self.x_vector_no_grad.data, requires_grad=True)
        z = x_grad.maximum(self.y_vector_no_grad)
        z_torch = torch.maximum(self.x_vector_torch, self.y_vector_no_grad_torch)
        assert z.requires_grad == z_torch.requires_grad

    def test_gradient_accumulation(self):
        z1 = self.x_vector.maximum(self.y_vector)
        z2 = self.x_vector.maximum(self.y_vector)
        z1_torch = torch.maximum(self.x_vector_torch, self.y_vector_torch)
        z2_torch = torch.maximum(self.x_vector_torch, self.y_vector_torch)
        (z1 + z2).backward()
        (z1_torch + z2_torch).backward(torch.ones_like(z1_torch + z2_torch))
        assert np.array_equal(self.x_vector.grad.data, self.x_vector_torch.grad.numpy())
        assert np.array_equal(self.y_vector.grad.data, self.y_vector_torch.grad.numpy())

    def test_maximum_with_negative_numbers(self):
        z = self.x_vector_negative.maximum(self.y_vector_negative)
        z_torch = torch.maximum(
            self.x_vector_negative_torch, self.y_vector_negative_torch
        )
        assert np.array_equal(z.data, z_torch.detach().numpy())
        z.backward()
        z_torch.backward(torch.ones_like(z_torch))
        assert np.array_equal(
            self.x_vector_negative.grad.data, self.x_vector_negative_torch.grad.numpy()
        )
        assert np.array_equal(
            self.y_vector_negative.grad.data, self.y_vector_negative_torch.grad.numpy()
        )


class TestTensorMax(TestTensor):
    def test_max_1d(self):
        z = self.x_vector.max()
        assert z.data == 2.0
        assert z.prev == {self.x_vector}

        z.backward()
        assert np.array_equal(self.x_vector.grad.data, [0.0, 1.0])

    def test_max_axis_0(self):
        z = self.x_matrix.max(axis=0)  # Should return maximum along each column
        assert np.array_equal(z.data, [3.0, 4.0])

    def test_max_axis_0_keepdims(self):
        z = self.x_matrix.max(axis=0, keepdims=True)
        assert z.data.shape == (1, 2)
        assert np.array_equal(z.data, [[3.0, 4.0]])

    def test_max_axis_1(self):
        z = self.x_matrix.max(axis=1)
        assert np.array_equal(z.data, [2.0, 4.0])

    def test_max_axis_1_keepdims(self):
        z = self.x_matrix.max(axis=1, keepdims=True)
        assert z.data.shape == (2, 1)  # (2,3) -> (2,1)
        assert np.array_equal(z.data, [[2.0], [4.0]])


class TestTensorTranspose(TestTensor):
    def test_transpose_2d(self):
        y = self.x_matrix.transpose()  # Default behavior should reverse dims
        assert np.array_equal(y.data, [[1.0, 3.0], [2.0, 4.0]])
        y.backward()
        assert np.array_equal(self.x_matrix.grad.data, np.ones_like(self.x_matrix.data))

        # PyTorch comparison
        y_torch = self.x_matrix_torch.transpose(0, 1)
        assert np.array_equal(y.data, y_torch.detach().numpy())
        y_torch.backward(torch.ones_like(y_torch))
        assert np.array_equal(self.x_matrix.grad.data, self.x_matrix_torch.grad.numpy())

    def test_transpose_gradient_accumulation(self):
        y1 = self.x_matrix.transpose()
        y2 = self.x_matrix.transpose()
        (y1 + y2).backward()
        assert np.array_equal(
            self.x_matrix.grad.data, 2 * np.ones_like(self.x_matrix.data)
        )

        # PyTorch comparison
        y1_torch = self.x_matrix_torch.transpose(0, 1)
        y2_torch = self.x_matrix_torch.transpose(0, 1)
        (y1_torch + y2_torch).backward(torch.ones_like(y1_torch + y2_torch))
        assert np.array_equal(self.x_matrix.grad.data, self.x_matrix_torch.grad.numpy())

    def test_transpose_1d(self):
        # For 1D tensors, transpose with explicit dims is not valid in PyTorch
        # Instead, we should test that attempting to transpose raises an error
        with self.assertRaises(ValueError):
            self.x_vector.transpose(0, 1)  # Should raise error for 1D tensor

        # PyTorch comparison
        with self.assertRaises(IndexError):
            self.x_vector_torch.transpose(0, 1)

    def test_transpose_non_contiguous(self):
        # First get the slice
        sliced = self.three_by_three_matrix[:2, 1:]
        # Then transpose
        y = sliced.transpose(0, 1)
        # PyTorch comparison
        sliced_torch = self.three_by_three_matrix_torch[:2, 1:]
        y_torch = sliced_torch.transpose(0, 1)

        assert np.array_equal(y.data, y_torch.detach().numpy())

        # Test backward pass
        y.backward(np.ones_like(y.data))
        y_torch.backward(torch.ones_like(y_torch))
        assert np.array_equal(
            self.three_by_three_matrix.grad.data,
            self.three_by_three_matrix_torch.grad.numpy(),
        )

    def test_transpose_multiple(self):
        y = self.x_matrix.transpose(0, 1).transpose(0, 1)  # Should get back original
        assert np.array_equal(y.data, self.x_matrix.data)
        y.backward()
        assert np.array_equal(self.x_matrix.grad.data, np.ones_like(self.x_matrix.data))

        # PyTorch comparison
        y_torch = self.x_matrix_torch.transpose(0, 1).transpose(
            0, 1
        )  # Should get back original
        assert np.array_equal(y.data, y_torch.detach().numpy())
        y_torch.backward(torch.ones_like(y_torch))
        assert np.array_equal(self.x_matrix.grad.data, self.x_matrix_torch.grad.numpy())

    def test_transpose_requires_grad_false(self):
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=False)
        y = x.transpose()
        assert not y.requires_grad
        assert len(y.prev) == 0

        # PyTorch comparison
        x_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=False)
        y_torch = x_torch.transpose(0, 1)
        assert not y_torch.requires_grad


class TestTensorCat(TestTensor):
    def test_cat_basic(self):
        z = Tensor.cat([self.x_vector, self.y_vector])
        z_torch = torch.cat([self.x_vector_torch, self.y_vector_torch])

        assert np.array_equal(z.data, z_torch.detach().numpy())
        z.backward()
        z_torch.backward(torch.ones_like(z_torch))
        assert np.array_equal(self.x_vector.grad.data, self.x_vector_torch.grad.numpy())
        assert np.array_equal(self.y_vector.grad.data, self.y_vector_torch.grad.numpy())

    def test_cat_2d(self):
        z = Tensor.cat([self.x_matrix, self.y_matrix])
        z_torch = torch.cat([self.x_matrix_torch, self.y_matrix_torch])
        assert np.array_equal(z.data, z_torch.detach().numpy())
        z.backward()
        z_torch.backward(torch.ones_like(z_torch))
        assert np.array_equal(self.x_matrix.grad.data, self.x_matrix_torch.grad.numpy())
        assert np.array_equal(self.y_matrix.grad.data, self.y_matrix_torch.grad.numpy())


class TestTensorGetitem(TestTensor):
    def test_getitem_single_element(self):
        """Test getting a single element using integer indexing"""
        y = self.x_matrix[0, 1]
        # PyTorch equivalent
        y_torch = self.x_matrix_torch[0, 1]

        assert y.data == y_torch.item()
        y.backward()
        y_torch.backward()
        assert np.array_equal(self.x_matrix.grad.data, self.x_matrix_torch.grad.numpy())

    def test_getitem_slice(self):
        """Test getting elements using slice indexing"""
        z = self.x_matrix[:2, 1]  # Get second column
        # PyTorch equivalent
        z_torch = self.x_matrix_torch[:2, 1]

        assert np.array_equal(z.data, z_torch.detach().numpy())
        z.backward()
        z_torch.backward(torch.ones_like(z_torch))
        assert np.array_equal(self.x_matrix.grad.data, self.x_matrix_torch.grad.numpy())

    def test_getitem_row(self):
        """Test getting an entire row"""
        a = self.x_matrix[0]
        # PyTorch equivalent
        a_torch = self.x_matrix_torch[0]

        assert np.array_equal(a.data, a_torch.detach().numpy())
        a.backward()
        a_torch.backward(torch.ones_like(a_torch))
        assert np.array_equal(self.x_matrix.grad.data, self.x_matrix_torch.grad.numpy())

    def test_getitem_negative_index(self):
        """Test getting elements using negative indexing"""
        b = self.x_matrix[-1, -1]
        c = b * 2
        assert b.data == 4.0
        assert c.data == 8.0
        c.backward()
        expected_grad = np.array([[0.0, 0.0], [0.0, 2.0]])
        assert np.array_equal(self.x_matrix.grad.data, expected_grad)


class TestTensorSetitem(TestTensor):
    def test_setitem(self):
        # Scalar assignment
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        x[0] = Tensor([4.0], requires_grad=True)
        assert np.array_equal(x.data, [4.0, 2.0, 3.0])

        # Slice assignment
        x = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        x[1:3] = Tensor([5.0, 6.0], requires_grad=True)
        assert np.array_equal(x.data, [1.0, 5.0, 6.0, 4.0])

        # Non-tensor assignment
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        x[0] = 4.0
        assert np.array_equal(x.data, [4.0, 2.0, 3.0])

        # Multidimensional assignment
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        x[0, 1] = Tensor([5.0], requires_grad=True)
        assert np.array_equal(x.data, [[1.0, 5.0], [3.0, 4.0]])


class TestTensorIAdd(TestTensor):
    def test_iadd_basic(self):
        """Test basic in-place addition between two tensors"""
        self.x_vector += self.y_vector
        assert np.array_equal(self.x_vector.data, [4.0, 6.0])
        self.x_vector.backward()
        assert np.array_equal(self.x_vector.grad.data, [1.0, 1.0])
        assert np.array_equal(self.y_vector.grad.data, [1.0, 1.0])

    def test_iadd_scalar(self):
        """Test in-place addition with a scalar"""
        self.x_vector += 2.0
        assert np.array_equal(self.x_vector.data, [3.0, 4.0])
        self.x_vector.backward()
        assert np.array_equal(self.x_vector.grad.data, [1.0, 1.0])

    def test_iadd_broadcasting(self):
        """Test in-place addition with broadcasting"""
        self.x_matrix += self.x_vector
        assert np.array_equal(self.x_matrix.data, [[2.0, 4.0], [4.0, 6.0]])
        self.x_matrix.backward()
        assert np.array_equal(self.x_matrix.grad.data, [[1.0, 1.0], [1.0, 1.0]])
        assert np.array_equal(
            self.x_vector.grad.data, [2.0, 2.0]
        )  # Sum across broadcasted dimension

    def test_iadd_requires_grad_propagation(self):
        """Test requires_grad propagation in in-place addition"""
        self.x_vector_no_grad += self.y_vector
        assert (
            self.x_vector_no_grad.requires_grad
        )  # Should be True because y requires grad
        assert np.array_equal(self.x_vector_no_grad.data, [4.0, 6.0])
        self.x_vector_no_grad.backward()
        assert np.array_equal(self.y_vector.grad.data, [1.0, 1.0])


class TestTensorStack(TestTensor):
    def test_stack_basic(self):
        z = Tensor.stack([self.x_vector, self.y_vector])
        z_torch = torch.stack([self.x_vector_torch, self.y_vector_torch])

        assert np.array_equal(z.data, z_torch.detach().numpy())
        z.backward()
        z_torch.backward(torch.ones_like(z_torch))
        assert np.array_equal(self.x_vector.grad.data, self.x_vector_torch.grad.numpy())
        assert np.array_equal(self.y_vector.grad.data, self.y_vector_torch.grad.numpy())


class TestTensorPad(TestTensor):
    def test_pad_1d(self):
        """Test 1D tensor padding"""
        y = self.x_vector.pad((1, 1))  # pad both sides by 1
        y_torch = torch.nn.functional.pad(self.x_vector_torch, (1, 1))

        assert np.array_equal(y.data, y_torch.detach().numpy())
        y.backward()
        y_torch.backward(torch.ones_like(y_torch))
        assert np.array_equal(self.x_vector.grad.data, self.x_vector_torch.grad.numpy())

    def test_pad_2d(self):
        """Test 2D tensor padding"""
        y = self.x_matrix.pad((1, 1, 1, 1))  # pad all sides by 1
        y_torch = torch.nn.functional.pad(self.x_matrix_torch, (1, 1, 1, 1))

        assert np.array_equal(y.data, y_torch.detach().numpy())
        y.backward()
        y_torch.backward(torch.ones_like(y_torch))
        assert np.array_equal(self.x_matrix.grad.data, self.x_matrix_torch.grad.numpy())

    def test_pad_3d(self):
        """Test 3D tensor padding (e.g., single-channel image)"""
        y = self.three_d_matrix.pad(
            ((0, 0), (1, 1), (1, 1))
        )  # no padding on first dim, pad others by 1
        y_torch = torch.nn.functional.pad(self.three_d_matrix_torch, (1, 1, 1, 1, 0, 0))

        assert np.array_equal(y.data, y_torch.detach().numpy())
        y.backward()
        y_torch.backward(torch.ones_like(y_torch))
        assert np.array_equal(
            self.three_d_matrix.grad.data, self.three_d_matrix_torch.grad.numpy()
        )

    def test_pad_4d(self):
        """Test 4D tensor padding (batch of images)"""
        y = self.four_d_matrix.pad(
            ((0, 0), (0, 0), (1, 1), (1, 1))
        )  # pad only spatial dimensions
        y_torch = torch.nn.functional.pad(
            self.four_d_matrix_torch, (1, 1, 1, 1, 0, 0, 0, 0)
        )

        assert np.array_equal(y.data, y_torch.detach().numpy())
        y.backward()
        y_torch.backward(torch.ones_like(y_torch))
        assert np.array_equal(
            self.four_d_matrix.grad.data, self.four_d_matrix_torch.grad.numpy()
        )

    def test_pad_asymmetric(self):
        """Test asymmetric padding"""
        y = self.x_matrix.pad((0, 1, 1, 0))  # pad right by 1, top by 1
        y_torch = torch.nn.functional.pad(self.x_matrix_torch, (0, 1, 1, 0))

        assert np.array_equal(y.data, y_torch.detach().numpy())
        y.backward()
        y_torch.backward(torch.ones_like(y_torch))
        assert np.array_equal(self.x_matrix.grad.data, self.x_matrix_torch.grad.numpy())

    def test_pad_constant_value(self):
        """Test padding with constant value"""
        y = self.x_vector.pad((1, 1), constant_values=5.0)
        y_torch = torch.nn.functional.pad(self.x_vector_torch, (1, 1), value=5.0)

        assert np.array_equal(y.data, y_torch.detach().numpy())
        y.backward()
        y_torch.backward(torch.ones_like(y_torch))
        assert np.array_equal(self.x_vector.grad.data, self.x_vector_torch.grad.numpy())

    def test_pad_requires_grad_false(self):
        """Test padding with requires_grad=False"""
        y = self.x_matrix_no_grad.pad((1, 1))
        y_torch = torch.nn.functional.pad(self.x_matrix_no_grad_torch, (1, 1))

        assert np.array_equal(y.data, y_torch.detach().numpy())
        assert not y.requires_grad

    def test_pad_integer(self):
        """Test padding with integer value"""
        y = self.x_matrix.pad(1)  # pad all sides by 1
        y_torch = torch.nn.functional.pad(self.x_matrix_torch, (1, 1, 1, 1))

        assert np.array_equal(y.data, y_torch.detach().numpy())
        y.backward()
        y_torch.backward(torch.ones_like(y_torch))
        assert np.array_equal(self.x_matrix.grad.data, self.x_matrix_torch.grad.numpy())


class TestTensorView(TestTensor):
    def test_view_basic(self):
        """Test basic view operation and backward"""
        y = self.x_matrix.view(4)
        y_torch = self.x_matrix_torch.view(4)

        # Check view operation
        assert np.array_equal(y.data, y_torch.detach().numpy())

        # Check backward pass
        y.backward()
        y_torch.backward(torch.ones_like(y_torch))
        assert np.array_equal(self.x_matrix.grad.data, self.x_matrix_torch.grad.numpy())

    def test_view_modification(self):
        """Test view modification affects base tensor"""
        y = self.x_matrix.view(4)
        y_torch = self.x_matrix_torch.view(4)

        # Modify through view
        y_detached = y.detach()
        y_detached[0] = 5.0
        y_torch_detached = y_torch.detach()
        y_torch_detached[0] = 5.0

        assert np.array_equal(y.data, y_torch.detach().numpy())
        y.backward()
        y_torch.backward(torch.ones_like(y_torch))
        assert np.array_equal(self.x_matrix.grad.data, self.x_matrix_torch.grad.numpy())

    def test_view_matmul(self):
        """Test matrix multiplication with views"""
        # Using a 2x3 matrix for this specific test since we need to reshape to 3x2
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        x_view = x.view(3, 2)  # Reshape to (3, 2)
        y = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

        x_torch = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        x_torch_view = x_torch.view(3, 2)
        y_torch = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

        xy = x_view @ y
        xy_torch = x_torch_view @ y_torch

        assert np.array_equal(xy.data, xy_torch.detach().numpy())
        xy.backward()
        xy_torch.backward(torch.ones_like(xy_torch))

        assert np.array_equal(x.grad.data, x_torch.grad.numpy())
        assert np.array_equal(y.grad.data, y_torch.grad.numpy())

    def test_view_power(self):
        """Test power operation on views"""
        x_view = self.x_matrix.view(2, 2)
        x_torch_view = self.x_matrix_torch.view(2, 2)

        y = x_view**2
        y_torch = x_torch_view**2

        assert np.array_equal(y.data, y_torch.detach().numpy())
        y.backward()
        y_torch.backward(torch.ones_like(y_torch))

        assert np.array_equal(self.x_matrix.grad.data, self.x_matrix_torch.grad.numpy())

    def test_make_view_against_pytorch(self):
        # Create tensors in both frameworks
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        x = Tensor(data, requires_grad=True)
        x_torch = torch.tensor(data, requires_grad=True)

        # Define view functions
        def view_forward(x):
            return x.reshape(3, 2)

        def view_backward(grad):
            return grad.reshape(2, 3)

        # Create views
        view = x._make_view(view_forward, view_backward)
        view_torch = x_torch.view(3, 2)

        # Test forward pass
        np.testing.assert_array_equal(view.data, view_torch.detach().numpy())

        # Test backward pass with same gradient
        grad_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        view.backward(grad_data)
        view_torch.backward(torch.tensor(grad_data))

        # Compare gradients
        np.testing.assert_array_equal(x.grad.data, x_torch.grad.numpy())

    def test_view_chain_operations_against_pytorch(self):
        # Create tensors
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        x = Tensor(data, requires_grad=True)
        x_torch = torch.tensor(data, requires_grad=True)

        # Create views and perform operations
        view = x._make_view(lambda x: x.reshape(3, 2), lambda g: g.reshape(2, 3))
        view_torch = x_torch.view(3, 2)

        # Perform some operations on the views
        result = (view * 2 + 1).sum()
        result_torch = (view_torch * 2 + 1).sum()

        # Test forward pass
        self.assertAlmostEqual(result.data, result_torch.item())

        # Test backward pass
        result.backward()
        result_torch.backward()

        # Compare gradients
        np.testing.assert_array_equal(x.grad.data, x_torch.grad.numpy())

    def test_view_multiple_uses_against_pytorch(self):
        # Create tensors
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        x = Tensor(data, requires_grad=True)
        x_torch = torch.tensor(data, requires_grad=True)

        # Create views
        view1 = x._make_view(lambda x: x.reshape(3, 2), lambda g: g.reshape(2, 3))
        view2 = x._make_view(lambda x: x.reshape(6), lambda g: g.reshape(2, 3))

        view1_torch = x_torch.view(3, 2)
        view2_torch = x_torch.view(6)

        # Use both views in computation
        result = view1.sum() + view2.sum()
        result_torch = view1_torch.sum() + view2_torch.sum()

        # Test forward pass
        self.assertAlmostEqual(result.data, result_torch.item())

        # Test backward pass
        result.backward()
        result_torch.backward()

        # Compare gradients
        np.testing.assert_array_equal(x.grad.data, x_torch.grad.numpy())

    def test_view_with_broadcasting_against_pytorch(self):
        # Create tensors
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        x = Tensor(data, requires_grad=True)
        x_torch = torch.tensor(data, requires_grad=True)

        # Create views that will be broadcast
        view = x._make_view(lambda x: x.reshape(4), lambda g: g.reshape(2, 2))
        view_torch = x_torch.view(4)

        # Create another tensor for broadcasting
        y = Tensor(np.array([1.0, 2.0, 3.0, 4.0]), requires_grad=True)
        y_torch = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)

        # Perform broadcasting operation
        result = view * y
        result_torch = view_torch * y_torch

        # Test forward pass
        np.testing.assert_array_equal(result.data, result_torch.detach().numpy())

        # Test backward pass
        result.backward(np.ones_like(result.data))
        result_torch.backward(torch.ones_like(result_torch))

        # Compare gradients
        np.testing.assert_array_equal(x.grad.data, x_torch.grad.numpy())
        np.testing.assert_array_equal(y.grad.data, y_torch.grad.numpy())


class TestTensorPermute(TestTensor):
    def setUp(self):
        self.x_nchw = Tensor(np.random.randn(2, 3, 4, 5))  # NCHW
        self.x_nhwc = Tensor(np.random.randn(1, 3, 32, 32))  # NCHW

    def test_basic_permute(self):
        # Test basic permutation
        y = self.x_nchw.permute(0, 2, 3, 1)  # NHWC

        assert y.shape == (2, 4, 5, 3)
        assert np.allclose(y.data, np.transpose(self.x_nchw.data, (0, 2, 3, 1)))

        # Test gradient
        loss = y.sum()
        loss.backward()

        # Gradient should be ones permuted back
        expected_grad = np.ones_like(self.x_nchw.data)
        assert np.allclose(self.x_nchw.grad.data, expected_grad)

    def test_permute_chain(self):
        # Create same input in both frameworks
        np_data = np.random.randn(2, 3, 4, 5)
        x_torch = torch.tensor(np_data, requires_grad=True)
        x_ours = Tensor(np_data)

        # Forward pass
        y_torch = x_torch.permute(0, 2, 3, 1)
        y_ours = x_ours.permute(0, 2, 3, 1)

        # Compare shapes
        assert (
            y_torch.shape == y_ours.shape
        ), f"Shape mismatch: {y_torch.shape} vs {y_ours.shape}"

        z_torch = y_torch.permute(0, 3, 1, 2)
        z_ours = y_ours.permute(0, 3, 1, 2)

        # Compare shapes
        assert (
            z_torch.shape == z_ours.shape
        ), f"Shape mismatch: {z_torch.shape} vs {z_ours.shape}"

        # Backward pass
        loss_torch = z_torch.sum()
        loss_ours = z_ours.sum()

        loss_torch.backward()
        loss_ours.backward()

        # Compare gradient shapes
        assert (
            x_torch.grad.shape == x_ours.grad.shape
        ), f"Gradient shape mismatch: {x_torch.grad.shape} vs {x_ours.grad.shape}"

    def test_invalid_permute(self):
        x = Tensor(np.random.randn(2, 3, 4, 5))

        # Test wrong number of dimensions
        try:
            x.permute(0, 1, 2)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

        # Test invalid permutation
        try:
            x.permute(0, 1, 1, 2)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_conv2d_permute(self):
        # Test permute in Conv2d context
        x_nhwc = self.x_nhwc.permute(0, 2, 3, 1)  # NHWC
        assert x_nhwc.shape == (1, 32, 32, 3)

        # Simulate some computation
        y_nhwc = x_nhwc * 2

        # Back to NCHW
        y = y_nhwc.permute(0, 3, 1, 2)
        assert y.shape == (1, 3, 32, 32)

        # Test gradient
        loss = y.sum()
        loss.backward()

        expected_grad = np.ones_like(self.x_nhwc.data) * 2
        assert np.allclose(self.x_nhwc.grad.data, expected_grad)
