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
        assert z.grad == 1.0
        assert self.y_scalar.grad == 2.0  # dz/dy = d(y*x)/dy = x = 2.0
        assert self.x_scalar.grad == 3.0  # dz/dx = d(y*x)/dx = y = 3.0

    def test_backward_division(self):
        z = self.x_scalar / self.y_scalar
        z.backward()
        assert z.grad == 1.0
        assert np.isclose(
            self.x_scalar.grad, 1.0 / 3.0, atol=1e-5
        )  # dz/dx = d(x/y)/dx = 1/y = 1/3
        assert np.isclose(
            self.y_scalar.grad, -2.0 / 9.0, atol=1e-5
        )  # dz/dy = d(x/y)/dy = -x/y^2 = -2/9

    def test_backward_scalar_vector_matmul(self):
        x = Tensor(2.0, requires_grad=True)
        y = Tensor([3.0, 4.0], requires_grad=True)
        self.assertRaises(RuntimeError, lambda: x @ y)

    def test_backward_vector_vector_matmul(self):
        z = self.x_vector @ self.y_vector
        z.backward()
        assert z.grad == 1
        assert np.array_equal(self.x_vector.grad, np.array([3.0, 4.0]).T)
        assert np.array_equal(self.y_vector.grad, np.array([1.0, 2.0]).T)

    def test_backward_matrix_matrix_matmul(self):
        z = self.x_matrix @ self.y_matrix
        z.backward()
        assert np.array_equal(z.data, np.array([[19.0, 22.0], [43.0, 50.0]]))
        assert np.array_equal(
            self.x_matrix.grad, np.array([[11.0, 15.0], [11.0, 15.0]])
        )
        assert np.array_equal(self.y_matrix.grad, np.array([[4.0, 4.0], [6.0, 6.0]]))
        assert np.array_equal(z.grad, np.array([[1, 1], [1, 1]]))


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
        assert np.array_equal(self.x_vector.grad, self.x_vector_torch.grad.numpy())
        assert np.array_equal(self.y_vector.grad, self.y_vector_torch.grad.numpy())

    def test_maximum_with_scalar(self):
        z = self.x_vector.maximum(2.0)
        z_torch = torch.maximum(self.x_vector_torch, torch.tensor(2.0))
        assert np.array_equal(z.data, z_torch.detach().numpy())
        z.backward()
        z_torch.backward(torch.ones_like(z_torch))
        assert np.array_equal(self.x_vector.grad, self.x_vector_torch.grad.numpy())

    def test_2d_tensor_maximum(self):
        z = self.x_matrix.maximum(self.y_matrix)
        z_torch = torch.maximum(self.x_matrix_torch, self.y_matrix_torch)
        assert np.array_equal(z.data, z_torch.detach().numpy())
        z.backward()
        z_torch.backward(torch.ones_like(z_torch))
        assert np.array_equal(self.x_matrix.grad, self.x_matrix_torch.grad.numpy())
        assert np.array_equal(self.y_matrix.grad, self.y_matrix_torch.grad.numpy())

    def test_broadcasting(self):
        # Change the order of arguments for consistent broadcasting
        z = self.y_vector.maximum(
            self.x_matrix
        )  # y_vector shape: (2,) will be broadcast to (2,2)
        z_torch = torch.maximum(self.y_vector_torch, self.x_matrix_torch)
        assert np.array_equal(z.data, z_torch.detach().numpy())
        z.backward()
        z_torch.backward(torch.ones_like(z_torch))
        assert np.array_equal(self.x_matrix.grad, self.x_matrix_torch.grad.numpy())
        assert np.array_equal(self.y_vector.grad, self.y_vector_torch.grad.numpy())

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
        assert np.array_equal(self.x_vector.grad, self.x_vector_torch.grad.numpy())
        assert np.array_equal(self.y_vector.grad, self.y_vector_torch.grad.numpy())

    def test_maximum_with_negative_numbers(self):
        z = self.x_vector_negative.maximum(self.y_vector_negative)
        z_torch = torch.maximum(
            self.x_vector_negative_torch, self.y_vector_negative_torch
        )
        assert np.array_equal(z.data, z_torch.detach().numpy())
        z.backward()
        z_torch.backward(torch.ones_like(z_torch))
        assert np.array_equal(
            self.x_vector_negative.grad, self.x_vector_negative_torch.grad.numpy()
        )
        assert np.array_equal(
            self.y_vector_negative.grad, self.y_vector_negative_torch.grad.numpy()
        )


class TestTensorMax(TestTensor):
    def test_max_1d(self):
        z = self.x_vector.max()
        assert z.data == 2.0
        assert z.prev == {self.x_vector}

        z.backward()
        assert np.array_equal(self.x_vector.grad, [0.0, 1.0])

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
        assert np.array_equal(self.x_matrix.grad, np.ones_like(self.x_matrix.data))

        # PyTorch comparison
        y_torch = self.x_matrix_torch.transpose(0, 1)
        assert np.array_equal(y.data, y_torch.detach().numpy())
        y_torch.backward(torch.ones_like(y_torch))
        assert np.array_equal(self.x_matrix.grad, self.x_matrix_torch.grad.numpy())

    def test_transpose_gradient_accumulation(self):
        y1 = self.x_matrix.transpose()
        y2 = self.x_matrix.transpose()
        (y1 + y2).backward()
        assert np.array_equal(self.x_matrix.grad, 2 * np.ones_like(self.x_matrix.data))

        # PyTorch comparison
        y1_torch = self.x_matrix_torch.transpose(0, 1)
        y2_torch = self.x_matrix_torch.transpose(0, 1)
        (y1_torch + y2_torch).backward(torch.ones_like(y1_torch + y2_torch))
        assert np.array_equal(self.x_matrix.grad, self.x_matrix_torch.grad.numpy())

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
            self.three_by_three_matrix.grad,
            self.three_by_three_matrix_torch.grad.numpy(),
        )

    def test_transpose_multiple(self):
        y = self.x_matrix.transpose(0, 1).transpose(0, 1)  # Should get back original
        assert np.array_equal(y.data, self.x_matrix.data)
        y.backward()
        assert np.array_equal(self.x_matrix.grad, np.ones_like(self.x_matrix.data))

        # PyTorch comparison
        y_torch = self.x_matrix_torch.transpose(0, 1).transpose(
            0, 1
        )  # Should get back original
        assert np.array_equal(y.data, y_torch.detach().numpy())
        y_torch.backward(torch.ones_like(y_torch))
        assert np.array_equal(self.x_matrix.grad, self.x_matrix_torch.grad.numpy())

    def test_transpose_requires_grad_false(self):
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=False)
        y = x.transpose()
        assert not y.requires_grad
        assert len(y.prev) == 0

        # PyTorch comparison
        x_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=False)
        y_torch = x_torch.transpose(0, 1)
        assert not y_torch.requires_grad


class TestTensorGetitem(TestTensor):
    def test_getitem_single_element(self):
        """Test getting a single element using integer indexing"""
        y = self.x_matrix[0, 1]
        # PyTorch equivalent
        y_torch = self.x_matrix_torch[0, 1]

        assert y.data == y_torch.item()
        y.backward()
        y_torch.backward()
        assert np.array_equal(self.x_matrix.grad, self.x_matrix_torch.grad.numpy())

    def test_getitem_slice(self):
        """Test getting elements using slice indexing"""
        z = self.x_matrix[:2, 1]  # Get second column
        # PyTorch equivalent
        z_torch = self.x_matrix_torch[:2, 1]

        assert np.array_equal(z.data, z_torch.detach().numpy())
        z.backward()
        z_torch.backward(torch.ones_like(z_torch))
        assert np.array_equal(self.x_matrix.grad, self.x_matrix_torch.grad.numpy())

    def test_getitem_row(self):
        """Test getting an entire row"""
        a = self.x_matrix[0]
        # PyTorch equivalent
        a_torch = self.x_matrix_torch[0]

        assert np.array_equal(a.data, a_torch.detach().numpy())
        a.backward()
        a_torch.backward(torch.ones_like(a_torch))
        assert np.array_equal(self.x_matrix.grad, self.x_matrix_torch.grad.numpy())

    def test_getitem_negative_index(self):
        """Test getting elements using negative indexing"""
        b = self.x_matrix[-1, -1]
        c = b * 2
        assert b.data == 4.0
        assert c.data == 8.0
        c.backward()
        expected_grad = np.array([[0.0, 0.0], [0.0, 2.0]])
        assert np.array_equal(self.x_matrix.grad, expected_grad)


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
        assert np.array_equal(self.x_vector.grad, [1.0, 1.0])
        assert np.array_equal(self.y_vector.grad, [1.0, 1.0])

    def test_iadd_scalar(self):
        """Test in-place addition with a scalar"""
        self.x_vector += 2.0
        assert np.array_equal(self.x_vector.data, [3.0, 4.0])
        self.x_vector.backward()
        assert np.array_equal(self.x_vector.grad, [1.0, 1.0])

    def test_iadd_broadcasting(self):
        """Test in-place addition with broadcasting"""
        self.x_matrix += self.x_vector
        assert np.array_equal(self.x_matrix.data, [[2.0, 4.0], [4.0, 6.0]])
        self.x_matrix.backward()
        assert np.array_equal(self.x_matrix.grad, [[1.0, 1.0], [1.0, 1.0]])
        assert np.array_equal(
            self.x_vector.grad, [2.0, 2.0]
        )  # Sum across broadcasted dimension

    def test_iadd_requires_grad_propagation(self):
        """Test requires_grad propagation in in-place addition"""
        self.x_vector_no_grad += self.y_vector
        assert (
            self.x_vector_no_grad.requires_grad
        )  # Should be True because y requires grad
        assert np.array_equal(self.x_vector_no_grad.data, [4.0, 6.0])
        self.x_vector_no_grad.backward()
        assert np.array_equal(self.y_vector.grad, [1.0, 1.0])


class TestTensorPad(TestTensor):
    def test_pad_1d(self):
        """Test 1D tensor padding"""
        y = self.x_vector.pad((1, 1))  # pad both sides by 1
        y_torch = torch.nn.functional.pad(self.x_vector_torch, (1, 1))

        assert np.array_equal(y.data, y_torch.detach().numpy())
        y.backward()
        y_torch.backward(torch.ones_like(y_torch))
        assert np.array_equal(self.x_vector.grad, self.x_vector_torch.grad.numpy())

    def test_pad_2d(self):
        """Test 2D tensor padding"""
        y = self.x_matrix.pad((1, 1, 1, 1))  # pad all sides by 1
        y_torch = torch.nn.functional.pad(self.x_matrix_torch, (1, 1, 1, 1))

        assert np.array_equal(y.data, y_torch.detach().numpy())
        y.backward()
        y_torch.backward(torch.ones_like(y_torch))
        assert np.array_equal(self.x_matrix.grad, self.x_matrix_torch.grad.numpy())

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
            self.three_d_matrix.grad, self.three_d_matrix_torch.grad.numpy()
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
            self.four_d_matrix.grad, self.four_d_matrix_torch.grad.numpy()
        )

    def test_pad_asymmetric(self):
        """Test asymmetric padding"""
        y = self.x_matrix.pad((0, 1, 1, 0))  # pad right by 1, top by 1
        y_torch = torch.nn.functional.pad(self.x_matrix_torch, (0, 1, 1, 0))

        assert np.array_equal(y.data, y_torch.detach().numpy())
        y.backward()
        y_torch.backward(torch.ones_like(y_torch))
        assert np.array_equal(self.x_matrix.grad, self.x_matrix_torch.grad.numpy())

    def test_pad_constant_value(self):
        """Test padding with constant value"""
        y = self.x_vector.pad((1, 1), constant_values=5.0)
        y_torch = torch.nn.functional.pad(self.x_vector_torch, (1, 1), value=5.0)

        assert np.array_equal(y.data, y_torch.detach().numpy())
        y.backward()
        y_torch.backward(torch.ones_like(y_torch))
        assert np.array_equal(self.x_vector.grad, self.x_vector_torch.grad.numpy())

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
        assert np.array_equal(self.x_matrix.grad, self.x_matrix_torch.grad.numpy())


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
        assert np.array_equal(self.x_matrix.grad, self.x_matrix_torch.grad.numpy())

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
        assert np.array_equal(self.x_matrix.grad, self.x_matrix_torch.grad.numpy())

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

        assert np.array_equal(x.grad, x_torch.grad.numpy())
        assert np.array_equal(y.grad, y_torch.grad.numpy())

    def test_view_power(self):
        """Test power operation on views"""
        x_view = self.x_matrix.view(2, 2)
        x_torch_view = self.x_matrix_torch.view(2, 2)

        y = x_view**2
        y_torch = x_torch_view**2

        assert np.array_equal(y.data, y_torch.detach().numpy())
        y.backward()
        y_torch.backward(torch.ones_like(y_torch))

        assert np.array_equal(self.x_matrix.grad, self.x_matrix_torch.grad.numpy())
