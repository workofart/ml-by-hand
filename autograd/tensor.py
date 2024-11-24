import numpy as np
import logging
from typing import Union, Self, List, Tuple

logger = logging.getLogger(__name__)


class Tensor:
    def __init__(self, data, prev=None, requires_grad=True):
        self.data = np.asarray(data)
        self._grad = None  # lazy initialize, we will only initialize if needed in the backward pass

        self._backward = lambda: None
        self.prev = (
            set(prev) if prev else set()
        )  # all the operations before this Tensor
        self.requires_grad = requires_grad

        # View tracking
        self._view_forward_fn = lambda x: x
        self._view_backward_fn = lambda x: x

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, value):
        if value is None:
            self._grad = None
            return

        # Convert to numpy array directly if it's a Tensor
        if isinstance(value, Tensor):
            value_data = value.data
        else:
            value_data = np.asarray(value)

        # Set or accumulate gradient using numpy operations
        if self._grad is None:
            self._grad = Tensor(value_data, requires_grad=False)
        else:
            # IMPORTANT: this is not the same as self.data + value_data
            # We need to do in-place addition here to preserve any views or references
            # to the original gradient. For example, multiple operations
            # (e.g. __mul__, __add__) might update the same gradient tensor,
            # and we need to ensure that all updates are correctly reflected in
            # the same underlying array.
            self._grad.data += value_data

    def _make_view(self, view_forward_fn, view_backward_fn) -> Self:
        """Create a view of the tensor"""

        view = Tensor(
            data=view_forward_fn(self.data),
            prev={self} if self.requires_grad else set(),
            requires_grad=self.requires_grad,
        )

        def composed_backward_fn(x):
            grad_data = x.data if isinstance(x, Tensor) else np.asarray(x)
            return view_backward_fn(self._view_backward_fn(grad_data))

        view._view_forward_fn = lambda x: view_forward_fn(self._view_forward_fn(x))
        view._view_backward_fn = composed_backward_fn

        def _backward_view():
            if view.grad is not None:
                # Simplified gradient handling
                backward_grad = view._view_backward_fn(view.grad)
                self.grad = backward_grad

        view._backward = _backward_view

        return view

    def view(self, *shape) -> Self:
        """
        Create a view of the tensor with the same data but with the specified shape
        A view function is a callable that transforms the original tensor data into a new shape or representation without copying the underlying data.
        """
        if len(shape) == 1:
            if isinstance(shape[0], (tuple, list)):
                shape = tuple(shape[0])
            else:
                shape = shape  # Keep single-dimension shape as is

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
            # Split the gradient Tensor along the stacking axis
            grads = np.split(result.grad.data, len(tensors), axis=axis)

            # Distribute gradients to input tensors
            for tensor, grad in zip(tensors, grads):
                # Squeeze the gradient to match the input tensor's shape

                # Convert to Tensor and accumulate
                tensor.grad = Tensor(np.squeeze(grad, axis=axis), requires_grad=False)

        result._backward = _backward
        return result

    @staticmethod
    def cat(tensors, axis=0) -> Self:
        """Concatenates tensors along specified axis"""
        data = np.concatenate([t.data for t in tensors], axis=axis)
        result = Tensor(
            data, prev=set(tensors), requires_grad=any(t.requires_grad for t in tensors)
        )

        def _backward():
            start_idx = 0
            for t in tensors:
                shape = list(t.data.shape)
                slice_idx = [slice(None)] * len(shape)
                slice_idx[axis] = slice(start_idx, start_idx + shape[axis])
                t.grad = Tensor(result.grad.data[tuple(slice_idx)], requires_grad=False)
                start_idx += shape[axis]

        result._backward = _backward

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
            grad_data = grad.data if isinstance(grad, Tensor) else np.asarray(grad)
            transposed_grad = np.transpose(grad_data, _get_transpose_axes(dim0, dim1))
            return transposed_grad

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

            if self.requires_grad:
                # Should use grad setter instead of direct assignment
                self.grad = result.grad

            if other.requires_grad:
                # Need to handle broadcasting reduction here
                reduced_grad = result.grad
                if result.grad.shape != other.shape:
                    # Sum across broadcasted dimensions
                    reduce_dims = tuple(
                        range(len(result.grad.shape) - len(other.shape))
                    )
                    reduced_grad = result.grad.sum(axis=reduce_dims)

                other.grad = reduced_grad

        result._backward = _backward
        return result

    def __mul__(self, other: Union[float, int, "Tensor"]) -> "Tensor":
        """Multiply two tensors element-wise"""
        other = other if isinstance(other, Tensor) else Tensor(other)

        def _backward():
            if self.requires_grad:
                # Handle broadcasting for self's gradient
                grad = other * result.grad
                if grad.shape != self.shape:
                    # Sum across broadcasted dimensions
                    reduce_dims = tuple(range(len(grad.shape) - len(self.shape)))
                    grad = grad.sum(axis=reduce_dims)
                self.grad = grad

            if other.requires_grad:
                # Handle broadcasting for other's gradient
                grad = self * result.grad
                if grad.shape != other.shape:
                    # Sum across broadcasted dimensions
                    reduce_dims = tuple(range(len(grad.shape) - len(other.shape)))
                    grad = grad.sum(axis=reduce_dims)
                other.grad = grad

        result = Tensor(
            self.data * other.data,
            prev=(self, other) if (self.requires_grad or other.requires_grad) else (),
            requires_grad=self.requires_grad or other.requires_grad,
        )
        result._backward = _backward
        return result

    def __matmul__(self, other: Union[Self, float, int]):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)

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
                    self.grad = result.grad * other.data

                if other.requires_grad:
                    other.grad = result.grad * self.data
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
                # Compute gradient using matrix multiplication rules
                grad_self = np.matmul(result.grad.data, other.data.swapaxes(-1, -2))

                # Ensure gradient matches the original tensor's shape before view
                self.grad = grad_self.reshape(self.data.shape)

            # Compute gradient of loss w.r.t. other.data
            # d(loss) / d(other.data) = d(loss) / d(result) * d(result) / d(other.data)
            # d(result) / d(other.data) = self.data.T
            # self.data:    (n, m)
            # result.grad:  (n, p)
            # self.T:       (m, n) --> T operation is equivalent to swapaxes(-1, -2)
            # matmul(self.T, result.grad) = (m, n) @ (n, p) = (m, p) = other.grad
            if other.requires_grad:
                # Compute gradient using matrix multiplication rules
                grad_other = np.matmul(self.data.swapaxes(-1, -2), result.grad.data)

                # Ensure gradient matches the original tensor's shape before view
                other.grad = grad_other.reshape(other.data.shape)

                # Accumulate gradient
                # other.grad = (
                #     (other.grad + grad_other) if other.grad is not None else grad_other
                # )

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
                grad = other * (self ** (other - 1)) * result.grad
                self.grad = grad

            # Gradient w.r.t exponent (other)
            if other.requires_grad:
                valid_base = self.data > 0
                grad_y = (self**other) * np.log(np.abs(self.data)) * result.grad

                if np.isscalar(other.data):
                    grad_data = grad_y.data[valid_base]
                    grad_y = (
                        np.sum(grad_data)
                        if isinstance(grad_data, np.ndarray)
                        else grad_data
                    )

                else:
                    grad_y = np.where(valid_base, grad_y.data, 0)

                other.grad = grad_y

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
                # Create gradient tensor
                grad = (
                    self.grad
                    if self.grad is not None
                    else Tensor(np.ones_like(self.data), requires_grad=False)
                )

                if grad.shape != other.data.shape:
                    # Sum across broadcasted dimensions
                    sum_axes = tuple(range(len(grad.shape) - len(other.data.shape)))
                    if sum_axes:
                        grad = grad.sum(axis=sum_axes)
                    # Handle broadcasting within common dimensions
                    for i, (g, o) in enumerate(zip(grad.shape, other.data.shape)):
                        if o == 1:
                            grad = grad.sum(axis=len(grad.shape) - 1 - i, keepdims=True)

                    grad = grad.reshape(other.data.shape)
                other.grad = grad

        self._backward = _backward
        return self

    def __getitem__(self, idx):
        """Get item from tensor using numpy-style indexing"""

        def forward_fn(x):
            result = x[idx]
            return float(result) if np.isscalar(result) else result

        def backward_fn(grad):
            grad_data = grad.data if isinstance(grad, Tensor) else np.asarray(grad)
            return grad_data  # don't expand to full size yet

        view = self._make_view(forward_fn, backward_fn)

        def _backward():
            if view.grad is not None:
                backward_grad = view._view_backward_fn(view.grad)
                # Create full-size gradient only at the final step
                full_grad = np.zeros_like(self.data)
                full_grad[idx] = backward_grad
                self.grad = full_grad

        view._backward = _backward
        return view

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
            self.data[idx] = (
                value.data if np.isscalar(value.data) else value.data.item()
            )
        else:
            self.data[idx] = value.data

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
            self.grad = np.broadcast_to(result.grad.data, expanded_shape)

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
            # Create gradient array, to ensure each element has the same gradient contribution
            if result.grad is None:
                return

            upstream_grad = result.grad.data

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

            # If keepdims is False and axis is specified, we need to reshape upstream gradient
            if not keepdims and axis is not None:
                # Add back reduced dimensions
                if isinstance(axis, (tuple, list)):
                    for ax in sorted(axis):
                        upstream_grad = np.expand_dims(upstream_grad, ax)
                else:
                    upstream_grad = np.expand_dims(upstream_grad, axis)

            # Broadcast upstream gradient to match input shape
            broadcasted_grad = np.broadcast_to(upstream_grad, self.data.shape)

            # Scale gradient by number of elements we averaged over
            num_elements = (
                np.prod([self.data.shape[ax] for ax in axis])
                if axis is not None
                else self.data.size
            )
            scaled_grad = broadcasted_grad / num_elements

            # Accumulate gradient
            self.grad = scaled_grad

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
            upstream_grad = result.grad.data

            if axis is None:
                # For global max, gradient flows only to elements equal to max value
                # Create boolean mask of elements equal to max
                mask = self.data == np.max(self.data)
                # Multiply mask by upstream gradient
                grad = mask * upstream_grad
            else:
                # For max along specific axes, handle each axis separately
                for ax in axis:
                    # Max along specific axes
                    grad = np.zeros_like(self.data)

                    # Create mask of max elements along specified axes
                    max_mask = self.data == np.max(self.data, axis=ax, keepdims=True)

                    # Distribute gradient to max elements
                    # If multiple elements are max, distribute gradient equally
                    grad[max_mask] = upstream_grad / np.sum(
                        max_mask, axis=ax, keepdims=True
                    )

            # Add computed gradient to accumulated gradient
            self.grad = grad

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
            # Ensure we have a gradient to propagate
            if result.grad is None:
                return

            upstream_grad = result.grad.data

            # Create masks for where each input equals the maximum
            x_matches = self.data == result.data
            y_matches = other.data == result.data

            # Gradient computation for self
            if self.requires_grad:
                # Gradient logic:
                # - Full gradient where self is max
                # - Halved gradient where both are max
                grad = upstream_grad * (x_matches * (1.0 - 0.5 * y_matches))

                # Handle broadcasting
                if grad.shape != self.data.shape:
                    # Reduce extra dimensions
                    reduce_axes = tuple(range(len(grad.shape) - len(self.data.shape)))
                    grad = np.sum(grad, axis=reduce_axes)

                    # Ensure correct shape for broadcasting
                    while len(grad.shape) < len(self.data.shape):
                        grad = np.expand_dims(grad, axis=0)

                # Accumulate gradient using the grad.setter to accumulate
                self.grad = Tensor(grad, requires_grad=False)

            # Gradient computation for other
            if other.requires_grad:
                # Similar gradient logic for other tensor
                grad = upstream_grad * (y_matches * (1.0 - 0.5 * x_matches))

                # Handle broadcasting
                if grad.shape != other.data.shape:
                    # Reduce extra dimensions
                    reduce_axes = tuple(range(len(grad.shape) - len(other.data.shape)))
                    grad = np.sum(grad, axis=reduce_axes)

                    # Ensure correct shape for broadcasting
                    while len(grad.shape) < len(other.data.shape):
                        grad = np.expand_dims(grad, axis=0)

                # Accumulate gradient using the grad.setter to accumulate
                other.grad = Tensor(grad, requires_grad=False)

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

        # Create result tensor
        result = Tensor(
            data=np.pad(
                self.data,
                pad_width=pad_width,
                mode=mode,
                constant_values=constant_values,
            ),
            requires_grad=self.requires_grad,
        )

        # Update previous operations
        result.prev = {self}

        def _backward():
            # Ensure we have a gradient to propagate
            if result.grad is None:
                return

            # Gradient extraction based on tensor dimensions
            if (
                len(self.data.shape) == 4
            ):  # For 4D tensors (batch, channels, height, width)
                grad_slice = [
                    slice(None),
                    slice(None),
                    slice(pad_width[2][0], result.grad.shape[2] - pad_width[2][1]),
                    slice(pad_width[3][0], result.grad.shape[3] - pad_width[3][1]),
                ]
                extracted_grad = result.grad.data[tuple(grad_slice)]
            elif len(self.data.shape) == 3:  # For 3D tensors
                grad_slice = [
                    slice(None),
                    slice(pad_width[1][0], result.grad.shape[1] - pad_width[1][1]),
                    slice(pad_width[2][0], result.grad.shape[2] - pad_width[2][1]),
                ]
                extracted_grad = result.grad.data[tuple(grad_slice)]
            elif len(self.data.shape) == 2:  # For 2D tensors
                grad_slice = [
                    slice(pad_width[0][0], result.grad.shape[0] - pad_width[0][1]),
                    slice(pad_width[1][0], result.grad.shape[1] - pad_width[1][1]),
                ]
                extracted_grad = result.grad.data[tuple(grad_slice)]
            elif len(self.data.shape) == 1:  # For 1D tensors
                grad_slice = [
                    slice(pad_width[0][0], result.grad.shape[0] - pad_width[0][1])
                ]
                extracted_grad = result.grad.data[tuple(grad_slice)]
            else:
                raise ValueError("Unsupported number of dimensions")

            # Accumulate gradient
            self.grad = extracted_grad

        # Attach backward method
        result._backward = _backward
        return result

    def forward(self, data):
        pass

    def backward(self, grad=None):
        if not self.requires_grad:
            return

        # Initialize gradient if none provided
        if grad is None:
            grad = Tensor(np.ones_like(self.data), requires_grad=False)
        elif not isinstance(grad, Tensor):
            grad = Tensor(grad, requires_grad=False)

        # Initialize gradient
        self.grad = grad

        # Build computational graph in reverse order
        topological_sorted_tensors = []
        visited = set()

        def dfs(node: Tensor):
            if node not in visited:
                visited.add(node)
                for prev in node.prev:
                    if prev.requires_grad:
                        # Initialize the intermediate gradients
                        if prev.grad is None:
                            prev.grad = Tensor(
                                np.zeros_like(prev.data, dtype=np.float64),
                                requires_grad=False,
                            )
                        dfs(prev)
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
    def ndim(self):
        return len(self.data.shape)

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
