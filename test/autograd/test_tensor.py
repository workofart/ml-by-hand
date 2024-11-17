from autograd.tensor import Tensor
import numpy as np
import torch  # for comparison
from unittest import TestCase


class TestTensor(TestCase):
    def test_tensor(self):
        x = Tensor(2.0, requires_grad=True)
        y = Tensor(3.0, requires_grad=True)
        assert (-x).data == -2.0
        assert (x + y).data == 5.0
        assert (x + y).prev == {x, y}
        assert (x * y).data == 6.0
        assert (x * y).prev == {x, y}
        assert (x - y).data == -1.0
        assert (y - x).data == 1.0
        assert (x / y).data == 2.0 / 3.0
        assert (y / x).data == 1.5
        assert (x**y).data == 8.0
        assert (y**x).data == 9.0
        assert x.grad is None  # lazy init until backward is called
        assert y.grad is None  # lazy init until backward is called
        assert x.requires_grad
        assert len(y.prev) == 0

        x = Tensor([1.0, 2.0], requires_grad=True)
        y = Tensor([3.0, 4.0], requires_grad=True)
        z = x @ y
        assert np.array_equal(z.data, 1.0 * 3.0 + 2.0 * 4.0)

    def test_complex_tensor_ops(self):
        x = Tensor(2.0, requires_grad=True)
        y = Tensor(1.5, requires_grad=True)
        z = Tensor(4.0, requires_grad=True)

        assert ((x * y + z) ** 2).data == 49.0

    def test_backward(self):
        x = Tensor(2.0, requires_grad=True)
        y = Tensor(3.0, requires_grad=True)
        z = x * y

        assert z.data == 6.0
        assert z.prev == {x, y}
        assert z.grad is None

        # then we will call backward and check the gradients
        z.backward()
        assert z.grad == 1.0
        assert y.grad == 2.0  # dz/dy = d(y*x)/dy = x = 2.0
        assert x.grad == 3.0  # dz/dx = d(y*x)/dx = y = 3.0

        x = Tensor(2.0, requires_grad=True)
        y = Tensor(3.0, requires_grad=True)
        z = x / y
        z.backward()
        assert z.grad == 1.0
        assert np.isclose(x.grad, 1.0 / 3.0, atol=1e-5)  # dz/dx = d(x/y)/dx = 1/y = 1/3
        assert np.isclose(
            y.grad, -2.0 / 9.0, atol=1e-5
        )  # dz/dy = d(x/y)/dy = -x/y^2 = -2/9

        # Scalar-vector mat-mul
        x = Tensor(2.0, requires_grad=True)
        y = Tensor([3.0, 4.0], requires_grad=True)

        self.assertRaises(RuntimeError, lambda: x @ y)

        # Vector-vector mat-mul
        x = Tensor([1.0, 2.0], requires_grad=True)
        y = Tensor([3.0, 4.0], requires_grad=True)
        z = x @ y
        z.backward()
        assert z.grad == 1
        assert np.array_equal(x.grad, np.array([3.0, 4.0]).T)
        assert np.array_equal(y.grad, np.array([1.0, 2.0]).T)

        # matrix-matrix mat-mul
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        z = x @ y
        z.backward()
        assert np.array_equal(z.data, np.array([[19.0, 22.0], [43.0, 50.0]]))
        # result.grad * y.T
        assert np.array_equal(x.grad, np.array([[11.0, 15.0], [11.0, 15.0]]))
        # x.T * result.grad
        assert np.array_equal(y.grad, np.array([[4.0, 4.0], [6.0, 6.0]]))
        assert np.array_equal(z.grad, np.array([[1, 1], [1, 1]]))

    def test_sum(self):
        # Scalar tensor sum
        x = Tensor(5.0, requires_grad=True)
        s = x.sum()
        assert s.data == 5.0
        assert s.requires_grad == x.requires_grad

        # 1D tensor sum (global)
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        s = x.sum()
        assert s.data == 6.0
        assert s.prev == {x}

        # 1D tensor sum (axis)
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        s = x.sum(axis=0)
        assert s.data == 6.0

        # 2D tensor sum (global)
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        s = x.sum()
        assert s.data == 10.0

        # 2D tensor sum (axis=0)
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        s = x.sum(axis=0)  # (2, 2) -> (2,)
        assert np.array_equal(s.data, [4.0, 6.0])

        # 2D tensor sum (axis=1)
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        s = x.sum(axis=1)  # (2, 2) -> (2,)
        assert np.array_equal(s.data, [3.0, 7.0])

        # 2D tensor sum with keepdims
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        s = x.sum(axis=0, keepdims=True)
        assert s.data.shape == (1, 2)  # (2,2) -> (1,2)
        assert np.array_equal(s.data, [[4.0, 6.0]])

        # 3D tensor sum
        x = Tensor(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
        )
        s = x.sum(axis=(1, 2))  # (2,2,2) -> (2,)
        assert np.array_equal(s.data, [10.0, 26.0])

        # Verify requires_grad propagation
        x = Tensor([1.0, 2.0, 3.0], requires_grad=False)
        s = x.sum()
        assert not s.requires_grad

    def test_mean(self):
        # Scalar tensor mean
        x = Tensor(5.0, requires_grad=True)
        m = x.mean()
        assert m.data == 5.0
        assert m.requires_grad == x.requires_grad

        # 1D tensor mean (global)
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        m = x.mean()
        assert m.data == 2.0
        assert m.prev == {x}

        # 1D tensor mean (axis)
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        m = x.mean(axis=0)
        assert m.data == 2.0

        # 2D tensor mean (global)
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        m = x.mean()
        assert m.data == 2.5

        # 2D tensor mean (axis=0)
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        m = x.mean(axis=0)  # (2, 2) -> (2,)
        assert np.array_equal(m.data, [2.0, 3.0])

        # 2D tensor mean (axis=1)
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        m = x.mean(axis=1)  # (2, 2) -> (2,)
        assert np.array_equal(m.data, [1.5, 3.5])

        # 2D tensor mean with keepdims
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        m = x.mean(axis=0, keepdims=True)
        assert m.data.shape == (1, 2)  # (2,2) -> (1,2)
        assert np.array_equal(m.data, [[2.0, 3.0]])

        # 3D tensor mean
        x = Tensor(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
        )
        m = x.mean(axis=(1, 2))  # (2,2,2) -> (2,)
        assert np.array_equal(m.data, [2.5, 6.5])

        # Multiple axis mean
        x = Tensor(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
        )
        m = x.mean(axis=(0, 1))  # (2,2,2) -> (2,)
        # Intermediate after axis 0:
        # [[3.0, 4.0], [5.0, 6.0]]
        # Imediate after axis 1:
        # [4.0, 5.0]
        assert np.array_equal(m.data, [4.0, 5.0])

        # Verify requires_grad propagation
        x = Tensor([1.0, 2.0, 3.0], requires_grad=False)
        m = x.mean()
        assert not m.requires_grad

    def test_maximum(self):
        # Test case 1: Basic vector maximum
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = Tensor([2.0, 1.0, 3.0], requires_grad=True)
        x_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y_torch = torch.tensor([2.0, 1.0, 3.0], requires_grad=True)

        z = x.maximum(y)
        z_torch = torch.maximum(x_torch, y_torch)
        assert np.array_equal(z.data, z_torch.detach().numpy())

        z.backward()
        z_torch.backward(torch.ones_like(z_torch))
        assert np.array_equal(x.grad, x_torch.grad.numpy())
        assert np.array_equal(y.grad, y_torch.grad.numpy())

        # Test case 2: Maximum with scalar
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        x_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

        z = x.maximum(2.0)
        z_torch = torch.maximum(x_torch, torch.tensor(2.0))
        assert np.array_equal(z.data, z_torch.detach().numpy())

        z.backward()
        z_torch.backward(torch.ones_like(z_torch))
        assert np.array_equal(x.grad, x_torch.grad.numpy())

        # Test case 3: 2D tensor maximum
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = Tensor([[2.0, 1.0], [3.0, 5.0]], requires_grad=True)
        x_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y_torch = torch.tensor([[2.0, 1.0], [3.0, 5.0]], requires_grad=True)

        z = x.maximum(y)
        z_torch = torch.maximum(x_torch, y_torch)
        assert np.array_equal(z.data, z_torch.detach().numpy())

        z.backward()
        z_torch.backward(torch.ones_like(z_torch))
        assert np.array_equal(x.grad, x_torch.grad.numpy())
        assert np.array_equal(y.grad, y_torch.grad.numpy())

        # Test case 4: Broadcasting
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = Tensor([2.0, 3.0], requires_grad=True)
        x_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y_torch = torch.tensor([2.0, 3.0], requires_grad=True)

        z = x.maximum(y)
        z_torch = torch.maximum(x_torch, y_torch)
        assert np.array_equal(z.data, z_torch.detach().numpy())

        z.backward()
        z_torch.backward(torch.ones_like(z_torch))
        assert np.array_equal(x.grad, x_torch.grad.numpy())
        assert np.array_equal(y.grad, y_torch.grad.numpy())

        # Test case 5: requires_grad propagation
        x = Tensor([1.0, 2.0, 3.0], requires_grad=False)
        y = Tensor([2.0, 1.0, 4.0], requires_grad=True)
        x_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=False)
        y_torch = torch.tensor([2.0, 1.0, 4.0], requires_grad=True)

        z = x.maximum(y)
        z_torch = torch.maximum(x_torch, y_torch)
        assert z.requires_grad == z_torch.requires_grad

        z.backward()
        z_torch.backward(torch.ones_like(z_torch))
        assert x.grad is None
        assert np.array_equal(y.grad, y_torch.grad.numpy())

        # Test case 6: Gradient accumulation
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = Tensor([2.0, 1.0, 3.0], requires_grad=True)
        x_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y_torch = torch.tensor([2.0, 1.0, 3.0], requires_grad=True)

        z1 = x.maximum(y)
        z2 = x.maximum(y)
        z1_torch = torch.maximum(x_torch, y_torch)
        z2_torch = torch.maximum(x_torch, y_torch)

        (z1 + z2).backward()
        (z1_torch + z2_torch).backward(torch.ones_like(z1_torch + z2_torch))
        assert np.array_equal(x.grad, x_torch.grad.numpy())
        assert np.array_equal(y.grad, y_torch.grad.numpy())

        # Test case 7: Maximum with negative numbers
        x = Tensor([-1.0, -2.0, -3.0], requires_grad=True)
        y = Tensor([-2.0, -1.0, -3.0], requires_grad=True)
        x_torch = torch.tensor([-1.0, -2.0, -3.0], requires_grad=True)
        y_torch = torch.tensor([-2.0, -1.0, -3.0], requires_grad=True)

        z = x.maximum(y)
        z_torch = torch.maximum(x_torch, y_torch)
        assert np.array_equal(z.data, z_torch.detach().numpy())

        z.backward()
        z_torch.backward(torch.ones_like(z_torch))
        assert np.array_equal(x.grad, x_torch.grad.numpy())
        assert np.array_equal(y.grad, y_torch.grad.numpy())

    def test_max(self):
        # For 1D tensor, we should test without axis first
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        z = x.max()
        assert z.data == 3.0
        assert z.prev == {x}

        z.backward()
        assert np.array_equal(x.grad, [0.0, 0.0, 1.0])

        # For testing max with axis=0, we should use a 2D tensor instead
        x = Tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 0.0]], requires_grad=True)
        z = x.max(axis=0)  # Should return maximum along each column
        assert np.array_equal(z.data, [2.0, 2.0, 3.0])

        z = x.max(axis=0, keepdims=True)
        assert z.data.shape == (1, 3)
        assert np.array_equal(z.data, [[2.0, 2.0, 3.0]])

        z = x.max(axis=1)
        assert np.array_equal(z.data, [3.0, 2.0])

        z = x.max(axis=1, keepdims=True)
        assert z.data.shape == (2, 1)  # (2,3) -> (2,1)
        assert np.array_equal(z.data, [[3.0], [2.0]])

    def test_transpose(self):
        # Test case 1: Basic 2D transpose
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = x.transpose()  # Default behavior should reverse dims
        assert np.array_equal(y.data, [[1.0, 3.0], [2.0, 4.0]])
        y.backward()
        assert np.array_equal(x.grad, np.ones_like(x.data))

        # Test case 2: 3D tensor transpose with explicit dims
        x = Tensor(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
        )
        y = x.transpose(1, 0, 2)  # permute first two dims
        expected = np.array([[[1.0, 2.0], [5.0, 6.0]], [[3.0, 4.0], [7.0, 8.0]]])
        assert np.array_equal(y.data, expected)
        y.backward()
        assert np.array_equal(x.grad, np.ones_like(x.data))

        # Test case 3: Transpose with gradient accumulation
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y1 = x.transpose()
        y2 = x.transpose()
        (y1 + y2).backward()
        assert np.array_equal(x.grad, 2 * np.ones_like(x.data))

        # Test case 4: 1D tensor transpose (should be no-op)
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x.transpose()
        assert np.array_equal(y.data, x.data)
        y.backward()
        assert np.array_equal(x.grad, np.ones_like(x.data))

        # Test case 5: Compare with PyTorch
        x_tensor = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        x_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

        y_tensor = x_tensor.transpose()
        y_torch = x_torch.transpose(0, 1)

        assert np.array_equal(y_tensor.data, y_torch.detach().numpy())

        y_tensor.backward()
        y_torch.backward(torch.ones_like(y_torch))

        assert np.array_equal(x_tensor.grad, x_torch.grad.numpy())

        # Test case 6: Transpose of non-contiguous tensor
        x = Tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], requires_grad=True
        )
        y = x[:2, 1:].transpose()  # Take a slice and transpose
        assert np.array_equal(y.data, [[2.0, 5.0], [3.0, 6.0]])
        y.backward()
        expected_grad = np.array([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
        assert np.array_equal(x.grad, expected_grad)

        # Test case 7: Multiple transpositions
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = x.transpose().transpose()  # Should get back original
        assert np.array_equal(y.data, x.data)
        y.backward()
        assert np.array_equal(x.grad, np.ones_like(x.data))

        # Test case 8: requires_grad=False
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=False)
        y = x.transpose()
        assert not y.requires_grad
        assert len(y.prev) == 0

    def test_getitem(self):
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        y = x[0, 1]
        assert y.data == 2.0
        assert y.prev == {x}
        y.backward()
        assert np.array_equal(x.grad, np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]))

        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        z = x[0:2, 1]
        assert np.array_equal(z.data, [2.0, 5.0])
        z.backward()
        assert np.array_equal(x.grad, np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]))

        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        a = x[0]
        assert np.array_equal(a.data, [1.0, 2.0, 3.0])
        a.backward()
        assert np.array_equal(x.grad, np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]))

        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        b = x[-1, -1]
        c = b * 2
        assert b.data == 6.0
        assert c.data == 12.0
        c.backward()
        assert np.array_equal(x.grad, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]))

    def test_setitem(self):
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        x[0] = 4.0
        np.array_equal(x.data, [4.0, 2.0, 3.0])

    def test_iadd(self):
        # Test case 1: Basic in-place addition
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = Tensor([4.0, 5.0, 6.0], requires_grad=True)
        x += y
        assert np.array_equal(x.data, [5.0, 7.0, 9.0])
        x.backward()
        assert np.array_equal(x.grad, [1.0, 1.0, 1.0])
        assert np.array_equal(y.grad, [1.0, 1.0, 1.0])

        # Test case 2: Scalar addition
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        x += 2.0
        assert np.array_equal(x.data, [3.0, 4.0, 5.0])
        x.backward()
        assert np.array_equal(x.grad, [1.0, 1.0, 1.0])

        # Test case 3: Broadcasting
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = Tensor([1.0, 2.0], requires_grad=True)
        x += y
        assert np.array_equal(x.data, [[2.0, 4.0], [4.0, 6.0]])
        x.backward()
        assert np.array_equal(x.grad, [[1.0, 1.0], [1.0, 1.0]])
        assert np.array_equal(y.grad, [2.0, 2.0])  # Sum across broadcasted dimension

        # Test case 4: requires_grad propagation
        x = Tensor([1.0, 2.0], requires_grad=False)
        y = Tensor([3.0, 4.0], requires_grad=True)
        x += y
        assert x.requires_grad  # Should be True because y requires grad
        assert np.array_equal(x.data, [4.0, 6.0])
        x.backward()
        assert np.array_equal(y.grad, [1.0, 1.0])

    def test_pad(self):
        # Test case 1: 1D tensor padding
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        x_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

        y = x.pad((1, 1))  # pad both sides by 1
        y_torch = torch.nn.functional.pad(x_torch, (1, 1))

        assert np.array_equal(y.data, y_torch.detach().numpy())
        y.backward()
        y_torch.backward(torch.ones_like(y_torch))
        assert np.array_equal(x.grad, x_torch.grad.numpy())

        # Test case 2: 2D tensor padding
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        x_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

        # Convert PyTorch-style padding (left, right, top, bottom) to numpy-style
        y = x.pad((1, 1, 1, 1))  # pad all sides by 1
        y_torch = torch.nn.functional.pad(x_torch, (1, 1, 1, 1))

        assert np.array_equal(y.data, y_torch.detach().numpy())
        y.backward()
        y_torch.backward(torch.ones_like(y_torch))
        assert np.array_equal(x.grad, x_torch.grad.numpy())

        # Test case 3: 3D tensor padding (e.g., single-channel image)
        x = Tensor([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)
        x_torch = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)

        y = x.pad(((0, 0), (1, 1), (1, 1)))  # no padding on first dim, pad others by 1
        y_torch = torch.nn.functional.pad(x_torch, (1, 1, 1, 1, 0, 0))

        assert np.array_equal(y.data, y_torch.detach().numpy())
        y.backward()
        y_torch.backward(torch.ones_like(y_torch))
        assert np.array_equal(x.grad, x_torch.grad.numpy())

        # Test case 4: 4D tensor padding (batch of images)
        x = Tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)
        x_torch = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)

        y = x.pad(((0, 0), (0, 0), (1, 1), (1, 1)))  # pad only spatial dimensions
        y_torch = torch.nn.functional.pad(x_torch, (1, 1, 1, 1, 0, 0, 0, 0))

        assert np.array_equal(y.data, y_torch.detach().numpy())
        y.backward()
        y_torch.backward(torch.ones_like(y_torch))
        assert np.array_equal(x.grad, x_torch.grad.numpy())

        # Test case 5: Asymmetric padding
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        x_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

        y = x.pad((0, 1, 1, 0))  # pad right by 1, top by 1
        y_torch = torch.nn.functional.pad(x_torch, (0, 1, 1, 0))

        assert np.array_equal(y.data, y_torch.detach().numpy())
        y.backward()
        y_torch.backward(torch.ones_like(y_torch))
        assert np.array_equal(x.grad, x_torch.grad.numpy())

        # Test case 6: Padding with constant value
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        x_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

        y = x.pad((1, 1), constant_values=5.0)
        y_torch = torch.nn.functional.pad(x_torch, (1, 1), value=5.0)

        assert np.array_equal(y.data, y_torch.detach().numpy())
        y.backward()
        y_torch.backward(torch.ones_like(y_torch))
        assert np.array_equal(x.grad, x_torch.grad.numpy())

        # Test case 7: requires_grad=False
        x = Tensor([1.0, 2.0, 3.0], requires_grad=False)
        x_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=False)

        y = x.pad((1, 1))
        y_torch = torch.nn.functional.pad(x_torch, (1, 1))

        assert np.array_equal(y.data, y_torch.detach().numpy())
        assert not y.requires_grad

        # Test case 8: Integer padding
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        x_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

        y = x.pad(1)  # pad all sides by 1
        y_torch = torch.nn.functional.pad(x_torch, (1, 1, 1, 1))

        assert np.array_equal(y.data, y_torch.detach().numpy())
        y.backward()
        y_torch.backward(torch.ones_like(y_torch))
        assert np.array_equal(x.grad, x_torch.grad.numpy())
