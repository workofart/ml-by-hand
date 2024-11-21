import numpy as np
import logging
from typing import Union, Self, List, Tuple

logger = logging.getLogger(__name__)


class Tensor:
    def __init__(self, data, prev=None, requires_grad=True):
        self._base_data = np.array(data) if isinstance(data, (list, tuple)) else data
        self._base_grad = None  # lazy initialize, we will only initialize if needed in the backward pass

        self._backward = lambda: None
        self._backward_mask = None
        self.prev = (
            set(prev) if prev else set()
        )  # all the operations before this Tensor
        self.requires_grad = requires_grad

        # View tracking
        self._view_forward_fn = lambda x: x
        self._view_backward_fn = lambda x: x

    @property
    def data(self):
        return self._view_forward_fn(self._base_data)

    @data.setter
    def data(self, value):
        if value is None:
            self._base_data = None
        else:
            self._base_data = self._view_backward_fn(value)

    @property
    def grad(self):
        if self._base_grad is None:
            return None
        return self._view_backward_fn(self._base_grad)

    @grad.setter
    def grad(self, value):
        if value is None:
            self._base_grad = None
        else:
            self._base_grad = self._view_backward_fn(value)

    def _make_view(self, view_forward_fn, view_backward_fn) -> Self:
        """Create a view of the tensor"""

        view = Tensor(
            data=self._base_data,
            prev={self} if self.requires_grad else set(),
            requires_grad=self.requires_grad,
        )

        view._view_forward_fn = lambda x: view_forward_fn(self._view_forward_fn(x))
        view._view_backward_fn = lambda x: self._view_backward_fn(
            view_backward_fn(x) if x is not None else None
        )

        if self.requires_grad:

            def _backward_view():
                if view.grad is not None:
                    grad = view.grad

                    # Handle scalar gradients
                    if np.isscalar(grad):
                        grad = np.array(grad)

                    backward_grad = view._view_backward_fn(grad)

                    # Accumulate gradients in the base tensor
                    if self.grad is None:
                        self.grad = backward_grad
                    else:
                        self.grad += backward_grad

            view._backward = _backward_view

        return view

    def view(self, *shape) -> Self:
        """
        Create a view of the tensor with the same data but with the specified shape
        A view function is a callable that transforms the original tensor data into a new shape or representation without copying the underlying data.
        """
        if self.data.size != np.prod(shape):
            raise ValueError("Size of new view must match size of original tensor")
        return self._make_view(
            lambda x: x.reshape(shape), lambda x: x.reshape(self.data.shape)
        )

    @staticmethod
    def stack(tensors: List[Self], axis=0) -> Self:
        """
        Join a sequence of tensors along a new axis.

        Args:
            tensors (list of Tensors): sequence of tensors to stack
            axis (int): axis along which to stack

        Returns:
            Tensor: stacked tensor
        """
        if not tensors:
            raise ValueError("Need at least one tensor to stack")

        # Stack the underlying numpy arrays
        stacked_data = np.stack([t.data for t in tensors], axis=axis)

        # Create new tensor with stacked data
        result = Tensor(
            data=stacked_data,
            prev=set(tensors),  # maintain connections to all input tensors
            requires_grad=any(t.requires_grad for t in tensors),
        )

        def _backward():
            # Split the gradient along the stacking axis
            grads = np.split(result.grad, len(tensors), axis=axis)

            # Distribute gradients to input tensors
            for tensor, grad in zip(tensors, grads):
                # Squeeze the gradient to match the input tensor's shape
                tensor.grad += np.squeeze(grad, axis=axis)

        result._backward = _backward
        return result

    @staticmethod
    def cat(tensors, axis=0) -> Self:
        """Concatenates tensors along specified axis"""
        data = np.concatenate([t.data for t in tensors], axis=axis)
        result = Tensor(data, requires_grad=any(t.requires_grad for t in tensors))

        def _backward():
            start_idx = 0
            for t in tensors:
                shape = list(t.data.shape)
                slice_idx = [slice(None)] * len(shape)
                slice_idx[axis] = slice(start_idx, start_idx + shape[axis])
                t.grad += result.grad[tuple(slice_idx)]
                start_idx += shape[axis]

        result._backward = _backward
        result.prev = set(tensors)

        return result

    def transpose(self, dim0=0, dim1=1):
        ndim = self.data.ndim
        if not (0 <= dim0 < ndim and 0 <= dim1 < ndim):
            raise ValueError(
                f"Dimensions out of range for tensor with {ndim} dimensions"
            )

        def _get_transpose_axes(dim0: int, dim1: int) -> Tuple[int, ...]:
            axes = list(range(self.data.ndim))
            axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
            return tuple(axes)

        def forward_fn(x):
            return np.transpose(x, _get_transpose_axes(dim0, dim1))

        def backward_fn(grad):
            if grad.shape == self.data.shape:
                return np.transpose(grad, _get_transpose_axes(dim0, dim1))
            return grad

        # Create new view
        return self._make_view(forward_fn, backward_fn)

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
            prev={self, other},
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

                # Sum over the identified axes and ensure output shape matches target
                result = np.sum(grad_to_add, axis=axes_to_sum)
                result = result.reshape(target_shape)
                return result

            # Update self gradient
            if self.requires_grad:
                self._base_grad += self._view_backward_fn(result.grad)

            # Update other gradient
            if other.requires_grad:
                other._base_grad += reverse_broadcast(
                    other._view_backward_fn(result.grad), other._base_data.shape
                )

        result._backward = _backward
        return result

    def __mul__(self, other: Union[float, int, "Tensor"]) -> "Tensor":
        """Multiply two tensors element-wise"""
        other = other if isinstance(other, Tensor) else Tensor(other)

        def _backward():
            if self.requires_grad:
                grad = other.data * result.grad
                # Handle scalar case
                if np.isscalar(grad):
                    grad = np.array(grad)  # Convert scalar to array
                # Reduce grad to match self's shape if necessary
                elif grad.shape != self.data.shape:
                    axes = tuple(
                        i
                        for i, (g_dim, s_dim) in enumerate(
                            zip(grad.shape, self.data.shape)
                        )
                        if g_dim != s_dim
                    )
                    grad = np.sum(grad, axis=axes, keepdims=True)
                self.grad += grad

            if other.requires_grad:
                grad = self.data * result.grad
                # Handle scalar case more thoroughly
                if np.isscalar(grad) or np.isscalar(other.data):
                    # If either grad or other.data is scalar, create a scalar array
                    grad = np.array(grad)
                    if np.isscalar(other.data):
                        # If other.data is scalar, grad should be scalar too
                        grad = np.sum(grad)
                    else:
                        # If other.data is not scalar but grad is, broadcast grad
                        grad = np.full_like(other.data, grad)
                else:
                    # Both are arrays, handle shape mismatch
                    if grad.shape != other.data.shape:
                        axes = tuple(
                            i
                            for i, (g_dim, o_dim) in enumerate(
                                zip(grad.shape, other.data.shape)
                            )
                            if g_dim != o_dim
                        )
                        grad = np.sum(grad, axis=axes, keepdims=True)
                        grad = grad.reshape(other.data.shape)

                other.grad += grad

        result = Tensor(
            self.data * other.data,
            prev=(self, other) if (self.requires_grad or other.requires_grad) else (),
            requires_grad=self.requires_grad or other.requires_grad,
        )
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
        result = Tensor(
            data=np.matmul(self.data, other.data),
            requires_grad=self.requires_grad or other.requires_grad,
            prev={self, other} if (self.requires_grad or other.requires_grad) else None,
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
            # Handle vector @ vector case separately (1D @ 1D)
            if self.data.ndim == 1 and other.data.ndim == 1:
                if self.requires_grad:
                    self.grad += self._view_backward_fn(result.grad.item()) * other.data

                if other.requires_grad:
                    other.grad += (
                        other._view_backward_fn(result.grad.item()) * self.data
                    )
                return

            # Handle N-D tensor multiplication (2D and higher)
            # Let's say we intially have a self.data of shape (n, m)
            # and other.data of shape (m, p)
            # Then result.data will be of shape (n, p)
            # We need to compute the gradient of the loss w.r.t. self.data
            # d(loss) / d(self.data) = d(loss) / d(result) * d(result) / d(self.data)
            # d(result) / d(self.data) = other.data.T

            # The usage of swapaxes is to ensure that the matrix multiplication is correct
            # when the dimensions are not in the expected order
            # self.data:    (n, m)
            # other.data:   (m, p)
            # result.grad:  (n, p)
            # other.T:      (p, m) --> T operation is equivalent to swapaxes(-1, -2)
            # matmul(result.grad, other.T) = (n, p) @ (p, m) = (n, m) = self.grad
            if self.requires_grad:
                self.grad += self._view_backward_fn(
                    np.matmul(result.grad, other.data.swapaxes(-1, -2))
                )

            # Compute gradient of loss w.r.t. other.data
            # d(loss) / d(other.data) = d(loss) / d(result) * d(result) / d(other.data)
            # d(result) / d(other.data) = self.data.T
            # self.data:    (n, m)
            # result.grad:  (n, p)
            # self.T:       (m, n) --> T operation is equivalent to swapaxes(-1, -2)
            # matmul(self.T, result.grad) = (m, n) @ (n, p) = (m, p) = other.grad
            if other.requires_grad:
                other.grad += other._view_backward_fn(
                    np.matmul(self.data.swapaxes(-1, -2), result.grad)
                )

        result._backward = _backward
        return result

    def __pow__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)

        result = Tensor(
            data=self.data**other.data,
            prev={self, other},
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
            if result.grad is None:
                return

            # Gradient w.r.t base (self)
            if self.requires_grad:
                self.grad += (
                    other.data
                    * (self.data ** (other.data - 1))
                    * self._view_backward_fn(result.grad)
                )

            # Gradient w.r.t exponent (other)
            if other.requires_grad:
                valid_base = self.data > 0
                grad_y = (
                    (self.data**other.data) * np.log(np.abs(self.data)) * result.grad
                )

                # Handle scalar and array cases
                if np.isscalar(other.data):
                    grad_y = (
                        np.sum(grad_y[valid_base])
                        if isinstance(grad_y, np.ndarray)
                        else grad_y
                    )
                else:
                    grad_y = np.where(valid_base, grad_y, 0)
                other.grad += other._view_backward_fn(grad_y)

        result._backward = _backward
        return result

    def __iadd__(self, other):
        """
        In-place addition operation (+=).
        This should maintain the computational graph while modifying the tensor in-place.
        """
        if not isinstance(other, Tensor):
            other = Tensor(other)

        # Update data in-place
        self.data += other.data
        self.prev.add(other)
        self.requires_grad = self.requires_grad or other.requires_grad

        # Store original backward function
        original_backward = self._backward

        def _backward():
            # Call original backward first
            original_backward()

            if other.requires_grad:
                if np.isscalar(other.data) or other.data.shape == ():
                    other.grad += self._view_backward_fn(float(np.sum(self.grad)))
                else:
                    # Handle broadcasting
                    grad = self.grad
                    if grad.shape != other.data.shape:
                        # Sum across broadcasted dimensions
                        sum_axes = tuple(range(len(grad.shape) - len(other.data.shape)))
                        if sum_axes:
                            grad = np.sum(grad, axis=sum_axes)
                        # Handle broadcasting within common dimensions
                        for i, (g, o) in enumerate(zip(grad.shape, other.data.shape)):
                            if o == 1:
                                grad = np.sum(grad, axis=i, keepdims=True)
                    other.grad += other._view_backward_fn(grad)
                    # other.grad = other.grad + grad if other.grad is not None else grad

        self._backward = _backward
        return self

    def __getitem__(self, idx):
        """Get item from tensor using numpy-style indexing"""

        def forward_fn(x):
            result = x[idx]
            if isinstance(result, np.ndarray) and result.ndim == 0:
                result = float(result)
            return result

        def backward_fn(grad):
            # Create zero array matching original shape
            full_grad = np.zeros_like(self._base_data)

            # Get the shape of the indexed result
            indexed_shape = np.array(self._base_data[idx]).shape

            if isinstance(grad, np.ndarray):
                # If grad shape matches the full tensor shape, extract the relevant slice
                if grad.shape == self.data.shape:
                    grad = grad[idx]
                # If grad shape doesn't match the indexed shape, try to reshape
                elif grad.shape != indexed_shape:
                    if grad.size == np.prod(indexed_shape):
                        grad = np.reshape(grad, indexed_shape)
                    else:
                        # If shapes don't match and can't be reshaped, raise error
                        raise ValueError(
                            f"Gradient shape {grad.shape} doesn't match indexed shape {indexed_shape}"
                        )

            # Scalar detection
            is_scalar_result = isinstance(idx, (int, tuple)) and (
                isinstance(self.data[idx], (float, int))
                or (isinstance(self.data[idx], np.ndarray) and self.data[idx].ndim == 0)
            )

            if is_scalar_result:
                if np.isscalar(grad):
                    full_grad[idx] = grad
                else:
                    grad_value = np.sum(grad) if isinstance(grad, np.ndarray) else grad
                    full_grad[idx] = grad_value
            else:
                if np.isscalar(grad):
                    full_grad[idx] = grad
                else:
                    # Handle row indexing specifically
                    if isinstance(idx, int):
                        if grad.ndim > len(indexed_shape):
                            grad = grad[0]
                        full_grad[idx] = grad
                    else:
                        full_grad[idx] = grad

            return full_grad

        return self._make_view(forward_fn, backward_fn)

    def __setitem__(
        self, idx: Union[int, slice, tuple], value: Union[Self, float, int]
    ):
        """
        In-place assignment operation using views

        Args:
            idx (tuple): indices to assign the value to
            value (Tensor): value to assign
        """
        if not isinstance(value, Tensor):
            value = Tensor(value, requires_grad=False)  # this is important

        # Update the indexed view with the value
        if np.isscalar(value.data):
            self._base_data[idx] = (
                value.data if np.isscalar(value.data) else value.data.item()
            )
        else:
            self._base_data[idx] = value.data

        # update gradient tracking
        if self.requires_grad or value.requires_grad:
            self.requires_grad = True
            self.prev.add(value)

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
            return Tensor(data=self.data, prev={self}, requires_grad=self.requires_grad)

        # Normalize axis
        axis = (axis,) if isinstance(axis, int) else axis

        # Compute sum
        result = Tensor(
            data=np.sum(self.data, axis=axis, keepdims=keepdims),
            prev={self},
            requires_grad=self.requires_grad,
        )

        def _backward():
            # Expand along the summed axes
            if keepdims:
                expanded_shape = self.data.shape
            else:
                # Convert generator to list before adding to shape
                expanded_shape = tuple(
                    s for i, s in enumerate(self.data.shape) if i != axis
                )

            # Add to existing gradient
            self.grad += self._view_backward_fn(
                np.broadcast_to(result.grad, expanded_shape)
            )

        result._backward = _backward
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
            prev={self},
            requires_grad=self.requires_grad,
        )

        def _backward():
            # Compute number of summed elements
            num_elements = (
                np.prod([self.data.shape[ax] for ax in axis])
                if axis
                else self.data.size
            )
            grad_value = result.grad / num_elements

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
            self.grad += self._view_backward_fn(grad)

        result._backward = _backward
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
            prev={self},
            requires_grad=self.requires_grad,
        )

        def _backward():
            """
            d(loss) / dx = d(loss) / d(max(x)) * d(max(x)) / dx
            d(loss) / d(max(x)) = result.grad
            d(max(x)) / dx = 1 if x == max(x), 0 otherwise
            """
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
            self.grad += self._view_backward_fn(grad)

        result._backward = _backward
        return result

    def maximum(self, other: Union[Self, float, int]):
        """
        Element-wise maximum between self and other.
        When both inputs equal the maximum, gradient is split equally between them.
        """
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)

        result = Tensor(
            data=np.maximum(self.data, other.data),
            prev={self, other},
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def _backward():
            # Create masks for where each input equals the maximum
            x_matches = self.data == result.data
            y_matches = other.data == result.data

            if self.requires_grad:
                grad = result.grad * (x_matches * (1.0 - 0.5 * y_matches))
                # Handle broadcasting for self's gradient
                if grad.shape != self.data.shape:
                    # Reshape grad to match self's shape
                    grad = np.sum(
                        grad, axis=tuple(range(len(grad.shape) - len(self.data.shape)))
                    )
                    # Ensure the gradient has the correct shape for broadcasting
                    while len(grad.shape) < len(self.data.shape):
                        grad = np.expand_dims(grad, axis=0)
                self.grad += self._view_backward_fn(grad)

            if other.requires_grad:
                grad = result.grad * (y_matches * (1.0 - 0.5 * x_matches))
                # Handle broadcasting for other's gradient
                if grad.shape != other.data.shape:
                    # Reshape grad to match other's shape
                    grad = np.sum(
                        grad, axis=tuple(range(len(grad.shape) - len(other.data.shape)))
                    )
                    # Ensure the gradient has the correct shape for broadcasting
                    while len(grad.shape) < len(other.data.shape):
                        grad = np.expand_dims(grad, axis=0)
                other.grad += other._view_backward_fn(grad)

        result._backward = _backward
        return result

    def pad(self, pad_width, mode="constant", constant_values=0):
        """
        Pad the tensor with zeros.
        Args:
            pad_width: If tuple of 2 values, interpreted as padding for last dimension (PyTorch style).
                      If tuple of tuples, each inner tuple is (pad_before, pad_after) for each dimension.
                      If int, pad all dimensions with same value.
            mode: Padding mode (default: "constant")
            constant_values: Value to pad with (default: 0)
        """
        # Handle integer padding
        if isinstance(pad_width, int):
            if len(self.data.shape) == 1:
                pad_width = ((pad_width, pad_width),)
            elif len(self.data.shape) == 2:
                pad_width = ((pad_width, pad_width), (pad_width, pad_width))
            elif len(self.data.shape) == 3:
                pad_width = ((0, 0), (pad_width, pad_width), (pad_width, pad_width))
            elif len(self.data.shape) == 4:
                pad_width = (
                    (0, 0),
                    (0, 0),
                    (pad_width, pad_width),
                    (pad_width, pad_width),
                )
            else:
                raise ValueError("Unsupported number of dimensions")

        # Convert PyTorch-style padding to numpy-style padding
        # Handle PyTorch-style padding
        elif isinstance(pad_width, tuple):
            if len(pad_width) == 2 and not isinstance(pad_width[0], tuple):
                # (left, right) format
                if len(self.data.shape) == 2:
                    pad_width = ((0, 0), (pad_width[0], pad_width[1]))
                elif len(self.data.shape) == 1:
                    pad_width = ((pad_width[0], pad_width[1]),)
                else:
                    raise ValueError(
                        "Unsupported number of dimensions for this padding style"
                    )
            elif len(pad_width) == 4 and not isinstance(pad_width[0], tuple):
                # (left, right, top, bottom) format
                if len(self.data.shape) == 2:
                    pad_width = (
                        (pad_width[2], pad_width[3]),
                        (pad_width[0], pad_width[1]),
                    )
                else:
                    raise ValueError("4-tuple padding only supported for 2D tensors")
        # Handle integer padding
        elif isinstance(pad_width, int):
            if len(self.data.shape) == 1:
                pad_width = ((pad_width, pad_width),)
            elif len(self.data.shape) == 2:
                pad_width = ((pad_width, pad_width), (pad_width, pad_width))
            elif len(self.data.shape) == 3:
                pad_width = ((0, 0), (pad_width, pad_width), (pad_width, pad_width))
            elif len(self.data.shape) == 4:
                pad_width = (
                    (0, 0),
                    (0, 0),
                    (pad_width, pad_width),
                    (pad_width, pad_width),
                )
            else:
                raise ValueError("Unsupported number of dimensions")

        result = Tensor(
            data=np.pad(
                self.data,
                pad_width=pad_width,
                mode=mode,
                constant_values=constant_values,
            ),
            prev={self},
            requires_grad=self.requires_grad,
        )

        def _backward():
            if (
                len(self.data.shape) == 4
            ):  # For 4D tensors (batch, channels, height, width)
                self.grad += self._view_backward_fn(
                    result.grad[
                        :,
                        :,
                        pad_width[2][0] : result.grad.shape[2] - pad_width[2][1],
                        pad_width[3][0] : result.grad.shape[3] - pad_width[3][1],
                    ]
                )
            elif len(self.data.shape) == 3:  # For 3D tensors
                self.grad += self._view_backward_fn(
                    result.grad[
                        :,
                        pad_width[1][0] : result.grad.shape[1] - pad_width[1][1],
                        pad_width[2][0] : result.grad.shape[2] - pad_width[2][1],
                    ]
                )
            elif len(self.data.shape) == 2:  # For 2D tensors
                self.grad += self._view_backward_fn(
                    result.grad[
                        pad_width[0][0] : result.grad.shape[0] - pad_width[0][1],
                        pad_width[1][0] : result.grad.shape[1] - pad_width[1][1],
                    ]
                )
            elif len(self.data.shape) == 1:  # For 1D tensors
                self.grad += self._view_backward_fn(
                    result.grad[
                        pad_width[0][0] : result.grad.shape[0] - pad_width[0][1]
                    ]
                )
            else:
                raise ValueError("Unsupported number of dimensions")

        result._backward = _backward
        return result

    def forward(self, data):
        pass

    def backward(self, grad=None):
        if not self.requires_grad:
            return

        # TODO: Refactor to disallow scalar gradients to follow the same matmul assumption as Pytorch.

        # Initialize gradient if none provided
        # Handle scalar case first
        if grad is None:
            if np.isscalar(self.data) or (
                isinstance(self.data, np.ndarray) and self.data.ndim == 0
            ):
                grad = np.array(1.0)
            else:
                grad = np.ones_like(self.data)
        self.grad = grad

        # Build computational graph in reverse order
        topological_sorted_tensors = []
        visited = set()

        def dfs(node: Tensor):
            if node not in visited:
                visited.add(node)
                for prev in node.prev:
                    if prev.requires_grad:
                        # Initialize the intermediate gradients, the initialization above is
                        # only for the leaf (last) node in which we call backward().
                        if prev.grad is None:
                            prev.grad = np.zeros_like(prev._base_data, dtype=np.float64)
                        dfs(prev)
                # the order in which we append to the list is in reverse order
                # because we always move backwards looking at the previous nodes
                # that point to the current node
                topological_sorted_tensors.append(node)

        dfs(self)

        # Backward pass
        for tensor in reversed(topological_sorted_tensors):
            tensor._backward()

    @property
    def shape(self):
        """
        Return the shape of the tensor data.
        For scalars, returns an empty tuple ().
        For vectors, returns a tuple with one element (n,).
        For matrices, returns a tuple with two elements (m,n).
        """
        if np.isscalar(self.data):
            return ()
        return self.data.shape

    def reshape(self, *shape):
        return self.view(*shape)

    def detach(self) -> Self:
        """
        Detach the tensor from the computational graph.

        Returns:
            A new tensor with the same data but without a gradient.
        """
        return Tensor(self.data, requires_grad=False)

    @property
    def T(self) -> Self:
        """
        Convenience method for 2D matrix transpose.
        For higher dimensions, use transpose() with explicit dims.
        """
        if len(self.data.shape) != 2:
            raise ValueError(
                "T property is only defined for 2D tensors. Use transpose() for higher dimensions."
            )
        return self.transpose(1, 0)

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
