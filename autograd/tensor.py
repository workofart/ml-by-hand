import numpy as np
import logging
from typing import Union, Self

logger = logging.getLogger(__name__)


class Tensor:
    def __init__(self, data, prev=(), requires_grad=True):
        if isinstance(data, (list, tuple)):
            data = np.array(data)
        else:
            data = data

        self.data = data
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self._backward = lambda: None
        self.prev = set(prev)  # all the operations before this Tensor
        self.requires_grad = requires_grad

    # For each of these primitive operations, we need to adjust the backward gradient computation accordingly
    def __add__(self, other):
        if not isinstance(other, Tensor):
            # Ensure other is converted to a Tensor with the same shape as self
            other = Tensor(
                data=np.full_like(self.data, other)
                if np.isscalar(other) or other.ndim == 0
                else other,
                requires_grad=False,  # we don't need to compute gradients for these scalars or constants
            )

        result = Tensor(
            data=self.data + other.data,
            prev=(self, other),
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def _backward():
            """
            d(loss) / dx = d(loss) / d(x + y) * d(x + y) / dx
            d(loss) / d(x + y) = result.grad
            d(x + y) / dx = 1
            d(x + y) / dy = 1
            We need to multiply by result.grad because of the chain rule
            """

            def reverse_broadcast(grad_to_add, target_shape):
                # Calculate the number of dimensions to add to target_shape to match grad_to_add
                num_extra_dims = len(grad_to_add.shape) - len(target_shape)

                # Create a new shape for target_shape with ones in the extra dimensions
                expanded_target_shape = (1,) * num_extra_dims + target_shape

                # Identify the axes to sum over by comparing shapes
                axes_to_sum = tuple(
                    i
                    for i, (g_dim, t_dim) in enumerate(
                        zip(grad_to_add.shape, expanded_target_shape)
                    )
                    if g_dim != t_dim
                )

                # Sum over the identified axes
                return np.sum(grad_to_add, axis=axes_to_sum).reshape(target_shape)

            self.grad += result.grad
            other.grad += reverse_broadcast(result.grad, other.data.shape)

        result._backward = _backward
        return result

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        result = Tensor(
            data=self.data * other.data,
            prev=(self, other),
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def _backward():
            """
            d(loss) / dx = d(loss) / d(xy) * d(xy) / dx
            d(loss) / d(xy) = result.grad
            d(xy) / dx = y
            d(xy) / dy = x
            """
            grad = result.grad
            # Handle broadcasting: sum along broadcasted dimensions
            if np.isscalar(self.data) or (
                isinstance(other.data, np.ndarray)
                and self.data.shape != other.data.shape
            ):
                self.grad += np.sum(other.data * grad)
            else:
                self.grad += other.data * grad

            # Handle broadcasting: sum along broadcasted dimensions
            if np.isscalar(other.data) or (
                isinstance(self.data, np.ndarray)
                and self.data.shape != other.data.shape
            ):
                other.grad += np.sum(self.data * grad)
            else:
                other.grad += self.data * grad

        result._backward = _backward
        return result

    def __matmul__(self, other: Union[Self, float, int]):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        # Raise error if either input is scalar (0D) - Same as Pytorch assumption
        if np.isscalar(self.data) or np.isscalar(other.data):
            raise RuntimeError("both arguments to matmul need to be at least 1D")

        # Handle matrix multiplication shapes:
        # - If input is 1D vector, reshape it for matrix multiplication:
        #   - First operand (x): reshape to (1, n) row vector
        #   - Second operand (y): reshape to (n, 1) column vector
        # - If input is 2D matrix, keep original shape
        x = self.data.reshape((1, -1)) if self.data.ndim == 1 else self.data
        y = other.data.reshape((-1, 1)) if other.data.ndim == 1 else other.data

        result = Tensor(
            data=np.matmul(x, y).squeeze(),
            prev=(self, other),
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def _backward():
            """
            d(loss) / dx
            = self.grad
            = d(loss) / d(x·y) * d(x·y) / dx
            = result.grad * d(x·y) / dx
            = result.grad * y.T

            d(loss) / dy
            = other.grad
            = d(loss) / d(x·y) * d(x·y) / dy
            = result.grad * d(x·y) / dy
            = x.T * result.grad
            Note:
                need to move x.T to the left because:
                1) Each element in result is a dot product of a row from x with a column from y
                2) When we backprop, we need x.T on the left to match dimensions:
                   x = (num_samples, num_features)
                   y = (num_features, num_classes)
                   x.T = (num_features, num_samples)
                   result.grad = (num_samples, num_classes)
                   x.T * result.grad = (num_features, num_classes)  # same shape as y
            """
            # Vector @ Vector case (result is scalar)
            if self.data.ndim == 1 and other.data.ndim == 1:
                self.grad += result.grad * other.data
                other.grad += result.grad * self.data

            # Matrix @ Vector case (result is vector)
            elif self.data.ndim == 2 and other.data.ndim == 1:
                self.grad += np.outer(result.grad, other.data)
                other.grad += np.matmul(self.data.T, result.grad)

            # Matrix @ Matrix case (result is matrix)
            else:
                # Ensure result.grad is 2D
                if result.grad.ndim == 1:
                    result_grad = result.grad.reshape(1, -1)
                else:
                    result_grad = result.grad

                # Ensure result_grad has the same shape as the forward pass output
                if result_grad.shape != (self.data.shape[0], other.data.shape[1]):
                    result_grad = result_grad.reshape(
                        self.data.shape[0], other.data.shape[1]
                    )
                self.grad += np.matmul(result_grad, other.data.T)
                other.grad += np.matmul(self.data.T, result_grad)

        result._backward = _backward
        return result

    def __pow__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        result = Tensor(
            data=self.data**other.data,
            prev=(self, other),
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def _backward():
            """
            d(loss) / dx = d(loss) / d(x**y) * d(x**y) / dx
            d(loss) / d(x**y) = result.grad
            d(x**y) / dx = y*x^(y-1)
            d(x**y) / dy = x**y * ln(x)
            where x is self
            y is other
            """
            # Gradient w.r.t base (self)
            self.grad += other.data * (self.data ** (other.data - 1)) * result.grad

            # Gradient w.r.t exponent (other)
            valid_base = self.data > 0
            grad_y = (self.data**other.data) * np.log(np.abs(self.data)) * result.grad

            # Handle scalar and array cases
            if np.isscalar(other.data):
                grad_y = (
                    np.sum(grad_y[valid_base])
                    if isinstance(grad_y, np.ndarray)
                    else grad_y
                )
            else:
                grad_y = np.where(valid_base, grad_y, 0)

            other.grad += grad_y

        result._backward = _backward
        return result

    def sum(self, axis=None, keepdims=False):
        """
        Compute the sum of tensor elements

        params:
            axis (int or tuple of ints, optional): Axis or axes along which a sum is performed. The default, axis=None, sums all of the elements of the input tensor.
            keepdims (bool, optional): If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input tensor.
        For example:
            - original tensor shape(3,4,5), axis(1,2), keepdims(True) -> result shape(3,1,1)
            - original tensor shape(3,4,5), axis(1,2), keepdims(False) -> result shape(3,)
            - original tensor shape(3,4,5), axis(None), keepdims(True) -> result shape(1,)
            - original tensor shape(3,4,5), axis(None), keepdims(False) -> result shape()
        """
        # Handle scalar case
        if not hasattr(self.data, "ndim") or self.data.ndim == 0:
            return Tensor(
                data=self.data, prev=(self,), requires_grad=self.requires_grad
            )

        # Normalize axis
        axis = (axis,) if isinstance(axis, int) else axis

        # Compute sum
        result = Tensor(
            data=np.sum(self.data, axis=axis, keepdims=keepdims),
            prev=(self,),
            requires_grad=self.requires_grad,
        )

        result._backward = lambda: self._reduce_ops_backward(result, axis, keepdims)
        return result

    def mean(self, axis=None, keepdims=False):
        """
        Compute the mean of tensor elements

        params:
            axis (int or tuple of ints, optional): Axis or axes along which a mean is performed. The default, axis=None, averages all of the elements of the input tensor.
            keepdims (bool, optional): If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input tensor.
        For example:
            - original tensor shape(3,4,5), axis(1,2), keepdims(True) -> result shape(3,1,1)
            - original tensor shape(3,4,5), axis(1,2), keepdims(False) -> result shape(3,)
            - original tensor shape(3,4,5), axis(None), keepdims(True) -> result shape(1,)
            - original tensor shape(3,4,5), axis(None), keepdims(False) -> result shape()
        """
        # Normalize axis to a tuple
        axis = (axis,) if isinstance(axis, int) else axis

        # Create result tensor
        result = Tensor(
            data=np.mean(self.data, axis=axis, keepdims=keepdims),
            prev=(self,),
            requires_grad=self.requires_grad,
        )

        result._backward = lambda: self._reduce_ops_backward(
            output=result, axis=axis, keepdims=keepdims
        )
        return result

    def max(self, axis=None, keepdims=False):
        """
        Compute the max of tensor elements

        Args:
            axis (int or tuple of ints, optional): Axis or axes along which a max is performed. The default, axis=None, maxes all of the elements of the input tensor.
            keepdims (bool, optional): If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input tensor.
        """
        axis = (axis,) if isinstance(axis, int) else axis

        result = Tensor(
            data=np.max(self.data, axis=axis, keepdims=keepdims),
            prev=(self,),
            requires_grad=self.requires_grad,
        )

        def _backward():
            """
            d(loss) / dx = d(loss) / d(max(x)) * d(max(x)) / dx
            d(loss) / d(max(x)) = result.grad
            d(max(x)) / dx = 1 if x == max(x), 0 otherwise
            """
            # Initialize gradient array with zeros
            grad = np.zeros_like(self.data)

            if axis is None:
                # For global max, gradient flows only to elements equal to max value
                # Create boolean mask of elements equal to max
                mask = self.data == np.max(self.data)
                # Multiply mask by upstream gradient
                grad = mask * result.grad
            else:
                # For max along specific axes, handle each axis separately
                for ax in axis:
                    # Create mask of elements equal to max along this axis
                    # keepdims=True preserves original dimensions for broadcasting
                    mask = self.data == np.max(self.data, axis=ax, keepdims=True)

                    # For elements that are max values, distribute gradient evenly
                    # Divide by len(axis) since each axis contributes equally to final gradient
                    grad[mask] = result.grad / len(axis)

            # Add computed gradient to accumulated gradient
            self.grad += grad

        result._backward = _backward
        return result

    def maximum(self, other: Union[Self, float, int]):
        """
        Element-wise maximum between self and other
        """
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)

        result = Tensor(
            data=np.maximum(self.data, other.data),
            prev=(self, other),
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def _backward():
            # For max(self,other), the gradient flows to self where self>other, and to other where other>self
            # When self=other, gradient is split between both (we'll give it to self in this case)
            mask = self.data >= other.data
            self.grad += result.grad * mask
            other.grad += result.grad * (~mask)

        result._backward = _backward
        return result

    def _reduce_ops_backward(self, output, axis=None, keepdims=False):
        """
        This function is a general backward pass function for computing gradients for
        "reduce" operations such as sum, mean, max, min, etc.
        d(reduce_func) / dx = 1 / count for each element that was reduced
        Each element contributes equally to the final mean value
        Global mean gradient is scaled by 1 / count
        """
        # Compute number of summed elements
        num_elements = (
            np.prod([self.data.shape[ax] for ax in axis]) if axis else self.data.size
        )
        grad_value = output.grad / num_elements

        # Create gradient array, to ensure each element has the same gradient contribution
        grad = np.full_like(self.data, grad_value)

        # Handle keepdims case
        # We need to preserve the information about:
        # - Distribute gradient proportionally
        # - Preserve tensor shape
        # - Maintain computational graph integrity

        # For example:
        # Original tensor: (2, 2, 3)
        # Sum result (keepdims=True): (2, 1, 1)
        # Indexing mechanism:
        # - Keep first dimension fully (slice(None))
        # - Reduce second dimension to first index (0)
        # - Reduce third dimension to first index (0)
        if axis and keepdims:
            slices = [slice(None)] * self.data.ndim
            for ax in axis:
                slices[ax] = slice(0, 1)
            grad[tuple(slices)] = grad_value

        # Accumulate gradient
        self.grad += grad

    def forward(self, data):
        pass

    def backward(self):
        topological_sorted_tensors = []
        visited = set()

        def dfs(node: Tensor):
            if node not in visited:
                visited.add(node)
                for prev in node.prev:
                    dfs(prev)
                # the order in which we append to the list is in reverse order
                # because we always move backwards looking at the previous nodes
                # that point to the current node
                topological_sorted_tensors.append(node)

        dfs(self)

        # Note that this is important to ensure our gradients shape is not a scalar
        # to ensure we follow the same matmul assumption as Pytorch.
        self.grad = np.ones_like(self.data)
        for tensor in reversed(topological_sorted_tensors):
            tensor._backward()

    def reshape(self, *shape):
        self.data = self.data.reshape(*shape)
        return self

    ##### Wrappers #####
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        return self * other**-1

    def __neg__(self):
        return self * -1

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __lt__(self, other):
        return self.data < other

    def __le__(self, other):
        return self.data <= other

    def __gt__(self, other):
        return self.data > other

    def __ge__(self, other):
        return self.data >= other
