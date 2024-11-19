import numpy as np
import logging
from typing import Union, Self, List

logger = logging.getLogger(__name__)


class Tensor:
    def __init__(self, data, prev=None, requires_grad=True, view_info=None):
        if isinstance(data, (list, tuple)):
            data = np.array(data)

        self.data = data
        self.grad = None  # lazy initialize, we will only initialize if needed in the backward pass

        self._backward = lambda: None
        self._backward_mask = None
        self.prev = (
            set(prev) if prev else set()
        )  # all the operations before this Tensor
        self.requires_grad = requires_grad

        # View tracking
        self._is_view = view_info is not None
        self._view_info = view_info  # tuple of (base tensor, view_fn, inverse_view_fn)
        self._base = None if not self._is_view else view_info[0]

    def _make_view(self, view_fn, inverse_view_fn) -> Self:
        """
        Create a view of the tensor using the provided view function and inverse view function

        Args:
            view_fn (callable): function to create the view
            The view function is to enable operations like reshaping, transposing, or slicing without duplicating the data in memory. This is efficient in terms of both speed and memory usage.

            inverse_view_fn (callable): function to invert the view
            The inverse view function ensures that when gradients are computed for the output of a view operation, they can be correctly mapped back to the original tensor's shape. This is essential for maintaining the integrity of the computational graph and ensuring that gradients flow correctly through the network.

        Returns:
            Tensor: view of the tensor
        """
        base_tensor = self._base if self._is_view else self
        prev = {base_tensor} if self.requires_grad else set()

        view = Tensor(
            data=view_fn(self.data),
            prev=prev,
            requires_grad=self.requires_grad,
            view_info=(base_tensor, view_fn, inverse_view_fn),
        )
        # initialize backward mask for gradient flow
        view._backward_mask = np.ones_like(view.data, dtype=bool)

        def _backward():
            if view.grad is not None:
                masked_grad = view.grad * view._backward_mask
                base_tensor.grad += inverse_view_fn(masked_grad)

        view._backward = _backward

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

        # Convert any non-Tensor inputs to Tensors
        tensors = [t if isinstance(t, Tensor) else Tensor(t) for t in tensors]

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
        requires_grad = any(t.requires_grad for t in tensors)

        result = Tensor(data, requires_grad=requires_grad)

        if requires_grad:

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

    def transpose(self, *dims) -> Self:
        """
        Transpose tensor along specified dimensions.
        If no dims specified, reverse all dims (like numpy.transpose())
        """
        # Handle default case (reverse all dims)
        if not dims:
            dims = tuple(range(self.data.ndim))[::-1]
        # Handle single dimension case
        elif len(dims) == 1:
            return self
        # Ensure dims match data dimensionality
        elif len(dims) != self.data.ndim:
            raise ValueError(
                f"Number of dimensions to transpose must match the number of dimensions in the tensor. Expected {self.data.ndim}, got {len(dims)}"
            )

        def inverse_view_fn(grad):
            # Get inverse permutation
            # Create a list to hold the inverse dimensions for the transpose operation.
            # This will allow us to map the gradient back to the original tensor's shape.
            inverse_dims = [0] * len(dims)
            for i, d in enumerate(dims):
                inverse_dims[d] = i  # Map each dimension to its original index.

            # First transpose the gradient back
            # Apply the inverse permutation to the gradient to reshape it back to the original tensor's dimensions.
            transposed_grad = np.transpose(grad, inverse_dims)

            if self._is_view:
                # If the current tensor is a view, we need to use the existing inverse view function
                # to ensure that the gradient is correctly mapped back to the base tensor.
                return self._view_info[2](
                    transposed_grad
                )  # Call the stored inverse view function.
            else:
                # If the tensor is not a view, simply return the transposed gradient.
                return transposed_grad

        return self._make_view(lambda x: np.transpose(x, dims), inverse_view_fn)

    def __add__(self, other):
        self_base = self._base if self._is_view else self

        if not isinstance(other, Tensor):
            # Ensure other is converted to a Tensor with the same shape as self
            other = Tensor(
                data=np.full_like(self_base.data, other)
                if np.isscalar(other) or other.ndim == 0
                else other,
                requires_grad=False,  # we don't need to compute gradients for these scalars or constants
            )

        other_base = other._base if other._is_view else other

        result = Tensor(
            data=self_base.data + other_base.data,
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
                self_base.grad += result.grad

            # Update other gradient
            if other.requires_grad:
                other_base.grad += reverse_broadcast(result.grad, other_base.shape)

        result._backward = _backward
        return result

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        self_base = self._base if self._is_view else self
        other_base = other._base if other._is_view else other

        # For the forward pass, we need to operate on the view's data instead of the base data
        result_data = self.data * other.data
        # Ensure the result is a scalar if both inputs are scalars
        if np.isscalar(self.data) and np.isscalar(other.data):
            result_data = float(result_data)  # Convert to float for scalar result
        elif np.isscalar(self.data) or np.isscalar(other.data):
            result_data = np.array(result_data)  # Ensure it's a single-element tensor

        result = Tensor(
            data=result_data,
            prev={self, other},
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def _backward():
            """
            d(loss) / dx = d(loss) / d(xy) * d(xy) / dx
            d(loss) / d(xy) = result.grad
            d(xy) / dx = y
            d(xy) / dy = x
            """
            logger.debug(f"Mul backward - self: {self}, grad: {self.grad}")
            logger.debug(f"Mul backward - other: {other}, grad: {other.grad}")
            logger.debug(f"Mul backward - result: {result}, grad: {result.grad}")

            # for views, we need to use the view's backward function
            if self._is_view:
                self.grad = other_base.data * result.grad
                self._backward()
            else:
                # For scalar inputs, ensure we get scalar gradients
                if np.isscalar(self_base.data) and np.isscalar(other_base.data):
                    if self.requires_grad:
                        self.grad += (other_base.data * result.grad).item()
                    if other.requires_grad:
                        other.grad += (self_base.data * result.grad).item()
                    return

                # Handle broadcasting: sum along broadcasted dimensions
                if np.isscalar(self_base.data) or (
                    isinstance(other_base.data, np.ndarray)
                    and self_base.data.shape != other_base.data.shape
                ):
                    if self.requires_grad:
                        self.grad += np.sum(other_base.data * result.grad)
                else:
                    if self.requires_grad:
                        self.grad += other_base.data * result.grad

                # Handle broadcasting: sum along broadcasted dimensions
                if np.isscalar(other_base.data) or (
                    isinstance(self_base.data, np.ndarray)
                    and self_base.data.shape != other_base.data.shape
                ):
                    if other.requires_grad:
                        other.grad += np.sum(self_base.data * result.grad)
                else:
                    if other.requires_grad:
                        other.grad += self_base.data * result.grad

        result._backward = _backward
        return result

    def __matmul__(self, other: Union[Self, float, int]):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        # Use view's data for computation
        data = self.data
        other_data = other.data

        # Raise error if either input is scalar (0D) - Same as Pytorch assumption
        if np.isscalar(data) or np.isscalar(other_data):
            raise RuntimeError("both arguments to matmul need to be at least 1D")

        # Handle matrix multiplication shapes:
        # - If input is 1D vector, reshape it for matrix multiplication:
        #   - First operand (x): reshape to (1, n) row vector
        #   - Second operand (y): reshape to (n, 1) column vector
        # - If input is 2D matrix, keep original shape
        result = Tensor(
            data=np.matmul(data, other_data),
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
            logger.debug("\nMatmul backward debug:")
            logger.debug(f"self shape: {self.data.shape}")
            logger.debug(f"other shape: {other.data.shape}")
            logger.debug(f"result.grad shape: {result.grad.shape}")

            # Handle vector @ vector case separately (1D @ 1D)
            if data.ndim == 1 and other_data.ndim == 1:
                if self.requires_grad:
                    grad = result.grad.item() * other_data
                    if self._is_view:
                        self.grad = grad
                        self._backward()
                    else:
                        self.grad += grad

                if other.requires_grad:
                    grad = result.grad.item() * data
                    if other._is_view:
                        other.grad = grad
                        other._backward()
                    else:
                        other.grad += grad
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
                grad = np.matmul(result.grad, other_data.swapaxes(-1, -2))
                if self._is_view:
                    self.grad = grad
                    self._backward()
                else:
                    self.grad += grad

            # Compute gradient of loss w.r.t. other.data
            # d(loss) / d(other.data) = d(loss) / d(result) * d(result) / d(other.data)
            # d(result) / d(other.data) = self.data.T
            # self.data:    (n, m)
            # result.grad:  (n, p)
            # self.T:       (m, n) --> T operation is equivalent to swapaxes(-1, -2)
            # matmul(self.T, result.grad) = (m, n) @ (n, p) = (m, p) = other.grad
            if other.requires_grad:
                grad = np.matmul(data.swapaxes(-1, -2), result.grad)
                if other._is_view:
                    other.grad = grad
                    other._backward()
                else:
                    other.grad += grad

        result._backward = _backward
        return result

    def __pow__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)

        # Use view's data for computation
        data = self.data
        other_data = other.data

        result = Tensor(
            data=data**other_data,
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
                grad = other_data * (data ** (other_data - 1)) * result.grad
                if self._is_view:
                    self.grad = grad
                    self._backward()
                else:
                    self.grad += grad

            # Gradient w.r.t exponent (other)
            if other.requires_grad:
                valid_base = data > 0
                grad_y = (data**other_data) * np.log(np.abs(data)) * result.grad

                # Handle scalar and array cases
                if np.isscalar(other_data):
                    grad_y = (
                        np.sum(grad_y[valid_base])
                        if isinstance(grad_y, np.ndarray)
                        else grad_y
                    )
                else:
                    grad_y = np.where(valid_base, grad_y, 0)

                if other._is_view:
                    other.grad = grad_y
                    other._backward()
                else:
                    other.grad += grad_y

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
        self.data = self.data + other.data
        self.prev.add(other)
        self.requires_grad = self.requires_grad or other.requires_grad

        # Store original backward function
        original_backward = self._backward

        def _backward():
            # Call original backward first
            original_backward()

            if other.requires_grad:
                if np.isscalar(other.data) or other.data.shape == ():
                    other.grad += float(np.sum(self.grad))
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
                    other.grad = other.grad + grad if other.grad is not None else grad

        self._backward = _backward
        return self

    def __getitem__(self, idx):
        """Get item from tensor using numpy-style indexing"""

        def view_fn(x):
            result = x[idx]
            if isinstance(idx, tuple):
                if all(isinstance(i, (int, np.integer)) for i in idx):
                    return float(result)
            elif isinstance(idx, (int, np.integer)):
                if isinstance(result, np.ndarray) and result.ndim == 0:
                    return float(result)
            return result

        def inverse_view_fn(grad):
            full_grad = np.zeros_like(self.data)
            if isinstance(idx, tuple):
                if all(isinstance(i, (int, np.integer)) for i in idx):
                    full_grad[idx] = grad.item()
                else:
                    full_grad[idx] = grad
            else:
                full_grad[idx] = grad
            return full_grad

        # Store the actual indices in the view_info
        return self._make_view(view_fn, inverse_view_fn)

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

        # If this is a view, we need to update the base tensor instead
        base_tensor = self._base if self._is_view else self

        # Update the indexed view with the value
        if self._is_view:
            if np.isscalar(value.data):
                self._view_info[1](base_tensor.data)[idx] = value.data.item()
            else:
                self._view_info[1](base_tensor.data)[idx] = value.data
        else:
            base_tensor.data[idx] = value.data

        # update gradient tracking
        if self.requires_grad or value.requires_grad:
            self.requires_grad = True
            self.prev.add(value)

            # Create mask for gradient flow
            self._backward_mask = np.ones_like(self.data, dtype=bool)
            if isinstance(idx, tuple):
                self._backward_mask[idx] = False
            else:
                self._backward_mask[idx] = False

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
        # Use base tensor's data if this is a view
        data = self._base.data if self._is_view else self.data

        # Handle scalar case
        if not hasattr(data, "ndim") or data.ndim == 0:
            return Tensor(data=data, prev={self}, requires_grad=self.requires_grad)

        # Normalize axis
        axis = (axis,) if isinstance(axis, int) else axis

        # Compute sum
        result = Tensor(
            data=np.sum(data, axis=axis, keepdims=keepdims),
            prev={self},
            requires_grad=self.requires_grad,
        )

        def _backward():
            # Expand along the summed axes
            if keepdims:
                expanded_shape = data.shape
            else:
                # Convert generator to list before adding to shape
                expanded_shape = tuple(s for i, s in enumerate(data.shape) if i != axis)
            grad_expanded = np.broadcast_to(result.grad, expanded_shape)

            # Add to existing gradient
            self.grad += grad_expanded

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
        # Use base tensor's data if this is a view
        data = self._base.data if self._is_view else self.data

        # Normalize axis to a tuple
        axis = (axis,) if isinstance(axis, int) else axis

        # Create result tensor
        result = Tensor(
            data=np.mean(data, axis=axis, keepdims=keepdims),
            prev={self},
            requires_grad=self.requires_grad,
        )

        def _backward():
            # Compute number of summed elements
            num_elements = (
                np.prod([data.shape[ax] for ax in axis]) if axis else data.size
            )
            grad_value = result.grad / num_elements

            # Create gradient array, to ensure each element has the same gradient contribution
            grad = np.full_like(data, grad_value)

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
                slices = [slice(None)] * data.ndim
                for ax in axis:
                    slices[ax] = slice(0, 1)
                grad[tuple(slices)] = grad_value

            # Accumulate gradient
            self.grad += grad

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
            self.grad += grad

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
                # Full gradient where only x matches, half where both match
                self.grad += result.grad * (x_matches * (1.0 - 0.5 * y_matches))

            if other.requires_grad:
                # Handle broadcasting for y's gradient
                grad = result.grad * (y_matches * (1.0 - 0.5 * x_matches))
                if grad.shape != other.data.shape:
                    # Sum across broadcasted dimensions
                    sum_axes = tuple(range(len(grad.shape) - len(other.data.shape)))
                    if sum_axes:
                        grad = np.sum(grad, axis=sum_axes)
                    # Handle broadcasting within common dimensions
                    for i, (g, o) in enumerate(zip(grad.shape, other.data.shape)):
                        if o == 1:
                            grad = np.sum(grad, axis=i, keepdims=True)
                other.grad += grad

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
                # Convert (left, right) to ((left, right),)
                pad_width = ((pad_width[0], pad_width[1]),)
            elif len(pad_width) == 4 and not isinstance(pad_width[0], tuple):
                # Convert (left, right, top, bottom) to ((top, bottom), (left, right))
                pad_width = ((pad_width[2], pad_width[3]), (pad_width[0], pad_width[1]))

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
                self.grad += result.grad[
                    :,
                    :,
                    pad_width[2][0] : result.grad.shape[2] - pad_width[2][1],
                    pad_width[3][0] : result.grad.shape[3] - pad_width[3][1],
                ]
            elif len(self.data.shape) == 3:  # For 3D tensors
                self.grad += result.grad[
                    :,
                    pad_width[1][0] : result.grad.shape[1] - pad_width[1][1],
                    pad_width[2][0] : result.grad.shape[2] - pad_width[2][1],
                ]
            elif len(self.data.shape) == 2:  # For 2D tensors
                self.grad += result.grad[
                    pad_width[0][0] : result.grad.shape[0] - pad_width[0][1],
                    pad_width[1][0] : result.grad.shape[1] - pad_width[1][1],
                ]
            elif len(self.data.shape) == 1:  # For 1D tensors
                self.grad += result.grad[
                    pad_width[0][0] : result.grad.shape[0] - pad_width[0][1]
                ]
            else:
                raise ValueError("Unsupported number of dimensions")

        result._backward = _backward
        return result

    def forward(self, data):
        pass

    def backward(self, grad=None):
        if not self.requires_grad:
            return

        # Note that this is important to ensure our gradients shape is not a scalar
        # to ensure we follow the same matmul assumption as Pytorch.
        # Initialize gradient if none provided
        if grad is not None:
            self.grad = grad
        else:
            self.grad = (
                np.array([1.0]) if np.isscalar(self.data) else np.ones_like(self.data)
            )
            # apply backward mask
            if self._backward_mask is not None:
                self.grad *= self._backward_mask

        # Handle views: propagate to base tensor and stop
        if self._is_view:
            if self._base.grad is None:
                self._base.grad = np.zeros_like(self._base.data)
            self._base.grad += self._view_info[2](self.grad)
            return  # Views don't need further backward propagation

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
                        target = prev._base if prev._is_view else prev
                        if target.grad is None:
                            target.grad = np.zeros_like(target.data, dtype=np.float64)
                        dfs(prev)
                # the order in which we append to the list is in reverse order
                # because we always move backwards looking at the previous nodes
                # that point to the current node
                topological_sorted_tensors.append(node)

        dfs(self)

        # Backward pass
        for tensor in reversed(topological_sorted_tensors):
            if not tensor._is_view:
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
