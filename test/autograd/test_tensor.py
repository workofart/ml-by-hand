from autograd.tensor import Tensor
import numpy as np
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
        assert x.grad == 0.0
        assert y.grad == 0.0
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
        z = y * x

        assert z.data == 6.0
        assert z.prev == {x, y}
        assert z.grad == 0.0

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
