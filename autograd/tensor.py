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
        # Always return a Tensor
        if isinstance(self._grad, np.ndarray):
            return Tensor(self._grad, requires_grad=False)
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
            value_data = self.expand(value_data.shape)._view_backward_fn(value_data)
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

        # Store the view functions
        view._view_forward_fn = view_forward_fn
        view._view_backward_fn = view_backward_fn

        def _backward_view():
            if view.grad is not None:
                # Transform gradient to original shape
                grad_data = (
                    view.grad.data if isinstance(view.grad, Tensor) else view.grad
                )
                self._accumulate_grad(view_backward_fn(grad_data))

        view._backward = _backward_view
        return view

    def view(self, *shape) -> Self:
        """
        Create a view of the tensor with the same data but with the specified shape
        A view function is a callable that transforms the original tensor data into a new shape or representation without copying the underlying data.
        """
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]

        # Handle -1 in shape
        if -1 in shape:
            # Can only have one -1 in shape
            if shape.count(-1) > 1:
                raise ValueError("Only one -1 dimension is allowed in shape")

            # Calculate the size of the -1 dimension
            neg_idx = shape.index(-1)
            known_size = np.prod(
                [d for i, d in enumerate(shape) if i != neg_idx and d != -1]
            )
            # Compute the missing dimension
            inferred_size = int(self.data.size // known_size)
            # Replace -1 with inferred size
            shape = tuple(inferred_size if d == -1 else d for d in shape)

        if self.data.size != np.prod(shape):
            raise ValueError(
                f"Size of new view must match size of original tensor: {self.data.size} != {np.prod(shape)}"
            )
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

        # Memory optimization: Use numpy.concatenate with expanded dimensions
        # instead of stack to avoid temporary list creation
        expanded_arrays = [np.expand_dims(t.data, axis=axis) for t in tensors]
        stacked_data = np.concatenate(expanded_arrays, axis=axis)

        # Create new tensor with stacked data
        result = Tensor(
            data=stacked_data,
            prev=set(tensors),  # maintain connections to all input tensors
            requires_grad=any(t.requires_grad for t in tensors),
        )

        def _backward():
            if result.grad is None:
                return

            # Memory optimization: Use views instead of splits where possible
            grad_size = result.grad.data.shape[axis]
            chunk_size = grad_size // len(tensors)

            for i, tensor in enumerate(tensors):
                if not tensor.requires_grad:
                    continue

                # Create slice indices
                idx = [slice(None)] * result.grad.data.ndim
                idx[axis] = slice(i * chunk_size, (i + 1) * chunk_size)

                # Use view instead of copy
                grad_slice = result.grad.data[tuple(idx)]
                tensor.grad = Tensor(grad_slice).reshape(tensor.shape)

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
                if not t.requires_grad:
                    continue

                # Create view using _make_view instead of view
                shape = list(t.data.shape)
                slice_idx = [slice(None)] * len(shape)
                slice_idx[axis] = slice(start_idx, start_idx + shape[axis])

                def forward_fn(x):
                    return x[tuple(slice_idx)]

                def backward_fn(grad):
                    # Create full-size gradient
                    full_grad = np.zeros_like(result.grad.data)
                    full_grad[tuple(slice_idx)] = grad
                    return full_grad

                # Use _make_view directly
                t.grad = result.grad._make_view(forward_fn, backward_fn)
                start_idx += shape[axis]

        result._backward = _backward

        return result

    def __add__(self, other):
        """Compute op: addition with explicit movement"""
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)

        # 1. Calculate broadcast shape
        broadcast_shape = np.broadcast_shapes(self.shape, other.shape)

        # 2. Movement ops: expand both tensors to broadcast shape
        x = self.expand(broadcast_shape)
        y = other.expand(broadcast_shape)

        # 3. Simple compute op (no shape logic)
        result = Tensor(
            data=x.data + y.data,
            prev={self, other},
            requires_grad=self.requires_grad or other.requires_grad,
        )

        # 4. Backward pass just focuses on gradient computation
        def _backward():
            if result.grad is None:
                return

            if self.requires_grad:
                self._accumulate_grad(x._view_backward_fn(result.grad.data))

            if other.requires_grad:
                other._accumulate_grad(y._view_backward_fn(result.grad.data))

        result._backward = _backward
        return result

    def __mul__(self, other):
        """Multiply two tensors element-wise"""
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)

        # 1. Calculate broadcast shape
        broadcast_shape = np.broadcast_shapes(self.shape, other.shape)

        # 2. Movement ops: expand both tensors to broadcast shape
        x = self.expand(broadcast_shape)
        y = other.expand(broadcast_shape)

        # 3. Simple compute op
        result = Tensor(
            data=x.data * y.data,
            prev={self, other},
            requires_grad=self.requires_grad or other.requires_grad,
        )

        # 4. Backward pass
        def _backward():
            if result.grad is None:
                return

            if self.requires_grad:
                self._accumulate_grad(other * result.grad)

            if other.requires_grad:
                other._accumulate_grad(self * result.grad)

        result._backward = _backward
        return result

    def __matmul__(self, other: Union[Self, float, int]):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)

        # Handle vector @ vector case separately (1D @ 1D)
        if self.data.ndim == 1 and other.data.ndim == 1:
            result = Tensor(
                data=np.dot(self.data, other.data),  # Returns scalar
                prev={self, other},
                requires_grad=self.requires_grad or other.requires_grad,
            )

            def _backward():
                if self.requires_grad:
                    self._accumulate_grad(result.grad.data * other.data)
                if other.requires_grad:
                    other._accumulate_grad(result.grad.data * self.data)

            result._backward = _backward
            return result

        # Matrix multiplication case
        else:
            # 1. Handle vector cases with reshape
            x, y = self, other
            if self.data.ndim == 1:
                x = self.reshape(1, -1)  # row vector
            if other.data.ndim == 1:
                y = other.reshape(-1, 1)  # column vector

            # Handle matrix multiplication shapes:
            # - If input is 1D vector, reshape it for matrix multiplication:
            #   - First operand (x): reshape to (1, n) row vector
            #   - Second operand (y): reshape to (n, 1) column vector
            # - If input is 2D matrix, keep original shape
            result = Tensor(
                data=np.matmul(x.data, y.data),
                prev={self, other},
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
                # Handle N-D tensor multiplication (2D and higher)
                # Let's say we intially have a self.data of shape (n, m)
                # and other.data of shape (m, p)
                # Then result.data will be of shape (n, p)
                # We need to compute the gradient of the loss w.r.t. self.data
                # d(loss) / d(self.data) = d(loss) / d(result) * d(result) / d(self.data)
                # d(result) / d(self.data) = other.data.T
                if self.requires_grad and result.grad is not None:
                    self.grad = result.grad.data @ other.data.T

                # Compute gradient of loss w.r.t. other.data
                # d(loss) / d(other.data) = d(loss) / d(result) * d(result) / d(other.data)
                # d(result) / d(other.data) = self.data.T
                # self.data:    (n, m)
                # result.grad:  (n, p)
                # self.T:       (m, n) --> T operation is equivalent to swapaxes(-1, -2)
                # matmul(self.T, result.grad) = (m, n) @ (n, p) = (m, p) = other.grad
                if other.requires_grad and result.grad is not None:
                    x_transposed = (
                        self.transpose(1, 2)
                        if self.data.ndim == 3
                        else self.transpose()
                    )
                    other.grad = (
                        np.sum(x_transposed.data @ result.grad.data, axis=0)
                        if self.data.ndim == 3
                        else x_transposed.data @ result.grad.data
                    )

            result._backward = _backward
            return result

    def __pow__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)

        # Use expand for broadcasting
        broadcast_shape = np.broadcast_shapes(self.shape, other.shape)
        x = self.expand(broadcast_shape)
        y = other.expand(broadcast_shape)

        result = Tensor(
            data=x.data**y.data,
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
            if self.requires_grad:
                self._accumulate_grad(y * (x ** (y - 1)) * result.grad)

            if other.requires_grad:
                valid_base = x.data > 0
                grad_y = (x**y) * np.log(np.abs(x.data)) * result.grad
                other.grad = np.where(valid_base, grad_y.data, 0)

        result._backward = _backward
        return result

    def __iadd__(self, other):
        """
        In-place addition operation (+=).
        This should maintain the computational graph while modifying the tensor in-place.
        """
        if not isinstance(other, Tensor):
            other = Tensor(other)

        # Use expand for broadcasting
        broadcast_shape = np.broadcast_shapes(self.shape, other.shape)
        expanded_other = other.expand(broadcast_shape)

        # Update data in-place
        self.data += expanded_other.data
        self.prev.add(other)
        self.requires_grad = self.requires_grad or other.requires_grad

        original_backward = self._backward

        def _backward():
            original_backward()
            if other.requires_grad:
                # Use expand to handle broadcasting in gradient
                other._accumulate_grad(expanded_other._view_backward_fn(self.grad.data))

        self._backward = _backward
        return self

    def __getitem__(self, idx):
        """Get item from tensor using numpy-style indexing"""

        def forward_fn(x):
            return x[idx]

        def backward_fn(grad):
            # Create a zero tensor of the original shape
            out = np.zeros_like(self.data)
            # Place the gradient in the correct location
            out[idx] = grad
            return out

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
        self.data[idx] = value.data

        # update gradient tracking
        if self.requires_grad or value.requires_grad:
            self.requires_grad = True
            self.prev.add(value)

            original_backward = self._backward

            def _backward():
                # Call original backward first if it exists
                original_backward()
                if value.requires_grad:
                    # Get the gradient at the assigned location
                    if np.isscalar(value.data):
                        value._accumulate_grad(self.grad.data[idx].sum())
                    else:
                        value._accumulate_grad(self.grad.data[idx])

            self._backward = _backward

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
            # Use expand to handle gradient broadcasting
            grad_shape = self.data.shape if keepdims else self.data.shape
            self._accumulate_grad(result.grad.expand(grad_shape))

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
            # Use expand for gradient broadcasting
            grad = result.grad.expand(self.shape if keepdims else self.data.shape)

            # Scale gradient by number of elements
            num_elements = (
                np.prod([self.data.shape[ax] for ax in axis])
                if axis is not None
                else self.data.size
            )
            self._accumulate_grad(grad.data / num_elements)

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
            grad = result.grad.expand(self.shape)
            max_vals = np.max(self.data, axis=axis, keepdims=True)
            mask = self.data == max_vals

            # Treat multiple axes or None as global max
            if axis is None or (isinstance(axis, tuple) and len(axis) > 1):
                # Global max: distribute equally
                count = np.sum(mask)
                self._accumulate_grad(grad.data * mask / count)
            else:
                # Single axis: use first occurrence
                ax = axis[0] if isinstance(axis, tuple) else axis
                cumsum = np.cumsum(mask, axis=ax)
                first_occur = cumsum == 1
                self._accumulate_grad(grad.data * (mask * first_occur))

        result._backward = _backward
        return result

    def maximum(self, other: Union[Self, float, int]):
        """
        Element-wise maximum between self and other.
        When both inputs equal the maximum, gradient is split equally between them.
        """
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)

        # Use expand for broadcasting
        broadcast_shape = np.broadcast_shapes(self.shape, other.shape)
        x = self.expand(broadcast_shape)
        y = other.expand(broadcast_shape)

        result = Tensor(
            data=np.maximum(x.data, y.data),
            prev={self, other},
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def _backward():
            upstream_grad = result.grad.data
            x_matches = x.data == result.data
            y_matches = y.data == result.data

            if self.requires_grad:
                grad = upstream_grad * (x_matches * (1.0 - 0.5 * y_matches))
                self._accumulate_grad(x._view_backward_fn(grad))

            if other.requires_grad:
                grad = upstream_grad * (y_matches * (1.0 - 0.5 * x_matches))
                other._accumulate_grad(y._view_backward_fn(grad))

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
        # Normalize pad_width to numpy style
        if isinstance(pad_width, int):
            # For int, create tuple of tuples for all dimensions
            pad_width = tuple((pad_width, pad_width) for _ in range(self.data.ndim))
        elif isinstance(pad_width, (tuple, list)):
            if len(pad_width) == 2 and not isinstance(pad_width[0], (tuple, list)):
                # (left, right) -> pad only last dimension
                pad_width = tuple((0, 0) for _ in range(self.data.ndim - 1)) + (
                    tuple(pad_width),
                )
            elif len(pad_width) == 4 and not isinstance(pad_width[0], (tuple, list)):
                # (left, right, top, bottom) -> pad last two dimensions
                pad_width = tuple((0, 0) for _ in range(self.data.ndim - 2)) + (
                    (pad_width[2], pad_width[3]),  # height/rows padding
                    (pad_width[0], pad_width[1]),  # width/cols padding
                )

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
            if result.grad is None:
                return

            # Extract the unpadded region
            slices = tuple(
                slice(p[0], s - p[1]) for s, p in zip(result.data.shape, pad_width)
            )
            self._accumulate_grad(result.grad.data[slices])

        result._backward = _backward
        return result

    def forward(self, data):
        pass

    def backward(self, grad=None):
        if not self.requires_grad:
            return

        # TODO: Refactor to disallow scalar gradients to follow the same matmul assumption as Pytorch.

        # Initialize gradient if none provided
        self._grad = np.ones_like(self.data) if grad is None else np.asarray(grad)

        # Build computational graph in reverse order
        topological_sorted_tensors = []
        visited = set()

        def dfs(node: Tensor):
            if node not in visited:
                visited.add(node)
                for prev in node.prev:
                    if prev.requires_grad:
                        dfs(prev)
                # the order in which we append to the list is in reverse order
                # because we always move backwards looking at the previous nodes
                # that point to the current node
                topological_sorted_tensors.append(node)

        dfs(self)

        # Backward pass
        for tensor in reversed(topological_sorted_tensors):
            tensor._backward()

        # Clear computational graph (safely)
        visited.clear()  # Clear the set we used for topo sort
        for node in topological_sorted_tensors:
            node.prev.clear()  # Clear references but don't recurse

    @property
    def shape(self):
        """
        Return the shape of the tensor data.
        For scalars, returns an empty tuple ().
        For vectors, returns a tuple with one element (n,).
        For matrices, returns a tuple with two elements (m,n).
        """
        if isinstance(self.data, (int, float)) or not hasattr(self.data, "shape"):
            return ()
        return self.data.shape

    ########### Movement ops ###########
    def reshape(self, *shape) -> "Tensor":
        """Movement op: reshape"""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]

        return self._make_view(
            lambda x: x.reshape(shape), lambda grad: grad.reshape(self.shape)
        )

    def expand(self, *shape) -> "Tensor":
        """Movement op: broadcast without copying"""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]

        def _view_backward_fn(grad):
            # Handle extra leading dimensions
            if len(grad.shape) > len(self.shape):
                reduce_dims = list(range(len(grad.shape) - len(self.shape)))
            else:
                reduce_dims = []

            # Handle broadcasting dimensions
            for i, (self_dim, grad_dim) in enumerate(
                zip(self.shape[::-1], grad.shape[-len(self.shape) :][::-1])
            ):
                if self_dim == 1 and grad_dim != 1:
                    reduce_dims.append(len(grad.shape) - 1 - i)

            # Sum across all reduction dimensions
            if reduce_dims:
                grad = np.sum(grad, axis=tuple(reduce_dims), keepdims=True)

            # Ensure final shape matches original
            if grad.shape != self.shape:
                grad = grad.reshape(self.shape)

            return grad

        return self._make_view(lambda x: np.broadcast_to(x, shape), _view_backward_fn)

    def permute(self, *dims) -> "Tensor":
        """Movement op: reorder dimensions"""
        dims = (
            dims[0] if len(dims) == 1 and isinstance(dims[0], (tuple, list)) else dims
        )

        return self._make_view(
            lambda x: np.transpose(x, dims),
            lambda grad: np.transpose(grad, [dims.index(i) for i in range(len(dims))]),
        )

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

    def strided_windows(self, kernel_size: int, stride: int) -> "Tensor":
        """Movement op: create strided windows view of the tensor"""
        batch_size, channels, height, width = self.shape
        H_out = (height - kernel_size) // stride + 1
        W_out = (width - kernel_size) // stride + 1

        def forward_fn(x):
            # Create windows using stride_tricks
            windows = np.lib.stride_tricks.as_strided(
                x,
                shape=(batch_size, channels, H_out, W_out, kernel_size, kernel_size),
                strides=(
                    x.strides[0],  # batch stride
                    x.strides[1],  # channel stride
                    x.strides[2] * stride,  # vertical stride between windows
                    x.strides[3] * stride,  # horizontal stride between windows
                    x.strides[2],  # vertical stride within window
                    x.strides[3],  # horizontal stride within window
                ),
                writeable=False,
            )
            # Reshape to (out_height * out_width, batch_size, channels, kernel_size, kernel_size)
            # This ensures row-major order matching PyTorch
            windows = windows.transpose(2, 3, 0, 1, 4, 5)
            return windows.reshape(
                H_out * W_out, batch_size, channels, kernel_size, kernel_size
            )

        def backward_fn(grad):
            # Reshape grad back to original window format
            grad = grad.reshape(
                H_out, W_out, batch_size, channels, kernel_size, kernel_size
            )
            grad = grad.transpose(2, 3, 0, 1, 4, 5)
            grad_padded = np.zeros_like(self.data)

            for i in range(kernel_size):
                for j in range(kernel_size):
                    grad_padded[
                        :,
                        :,
                        i : i + H_out * stride : stride,
                        j : j + W_out * stride : stride,
                    ] += grad[:, :, :, :, i, j]

            return grad_padded

        return self._make_view(forward_fn, backward_fn)

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

    def _accumulate_grad(self, grad, idx=None):
        """
        Helper method to lazily initialize and accumulate gradients
        Args:
            grad: gradient to accumulate
            idx: optional index for accumulating at specific locations
        """
        if grad is None:
            return

        # Convert numpy array or scalar to Tensor
        if not isinstance(grad, Tensor):
            grad = Tensor(grad, requires_grad=False)

        # Initialize or accumulate gradient
        if self._grad is None:
            if idx is not None:
                # Initialize with zeros
                self._grad = Tensor(np.zeros_like(self.data), requires_grad=False)
                # Accumulate at index without reshaping
                self._grad.data[idx] = grad.data
            else:
                self._grad = grad
        else:
            if idx is not None:
                # Accumulate at index without reshaping
                self._grad.data[idx] += grad.data
            else:
                self._grad = self._grad + grad

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
