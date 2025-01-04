import numpy as np
import logging
from typing import Union, Self, List, Tuple, Iterable

logger = logging.getLogger(__name__)


class Function:
    def __init__(self, *tensors: Iterable["Tensor"]):
        self.tensors = tensors

    def forward(self, *args, **kwargs) -> "Tensor":
        raise NotImplementedError("Forward pass not implemented for this function")

    def backward(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError("Backward pass not implemented for this function")

    @classmethod
    def apply(cls, *tensors, **kwargs):
        # Create the function object
        func = cls(*tensors)
        # Run forward pass with tensor.data already, so we don't need to get it again
        out_data = func.forward(*(inp.data for inp in tensors), **kwargs)

        # Create output tensor
        requires_grad = any(inp.requires_grad for inp in tensors)
        out = Tensor(out_data, creator=func, requires_grad=requires_grad)
        return out


class Tensor:
    def __init__(self, data, creator=None, requires_grad=True):
        self.data = np.asarray(data, dtype=np.float32)
        self._grad = None  # lazy initialize, we will only initialize if needed in the backward pass
        self.creator = creator

        self._backward = lambda: None
        self.requires_grad = requires_grad

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
            value_data = np.broadcast_to(value_data, value_data.shape)
            value_data = value_data.copy()
            # IMPORTANT: this is not the same as self.data + value_data
            # We need to do in-place addition here to preserve any views or references
            # to the original gradient. For example, multiple operations
            # (e.g. __mul__, __add__) might update the same gradient tensor,
            # and we need to ensure that all updates are correctly reflected in
            # the same underlying array.
            self._grad.data += value_data

    def view(self, *shape) -> "Tensor":
        """
        Create a view of the tensor with the same data but with the specified shape
        A view function is a callable that transforms the original tensor data into a new shape or representation without copying the underlying data.
        """
        return View.apply(self, new_shape=shape)

    @staticmethod
    def stack(tensors: List[Self], axis=0) -> Self:
        return Stack.apply(*tensors, axis=axis)

    @staticmethod
    def cat(tensors, axis=0) -> "Tensor":
        return Cat.apply(*tensors, axis=axis)

    def __add__(self, other):
        """Compute op: addition with explicit movement"""
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        if self.shape == other.shape:
            return Add.apply(self, other)

        # # 1. Calculate broadcast shape
        broadcast_shape = np.broadcast_shapes(self.shape, other.shape)

        # 2. Movement ops: expand both tensors to broadcast shape
        x = self.expand(broadcast_shape)
        y = other.expand(broadcast_shape)

        # 3. Simple compute op (no shape logic)
        return Add.apply(x, y)

    def __mul__(self, other):
        """Multiply two tensors element-wise"""
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)

        if self.shape == other.shape:
            return Mul.apply(self, other)

        # 1. Calculate broadcast shape
        broadcast_shape = np.broadcast_shapes(self.shape, other.shape)

        # 2. Movement ops: expand both tensors to broadcast shape
        x = self.expand(broadcast_shape)
        y = other.expand(broadcast_shape)

        # 3. Simple compute op
        return Mul.apply(x, y)

    def __matmul__(self, other: Union[Self, float, int]):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)

        return Matmul.apply(self, other)

    def __pow__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)

        if self.shape == other.shape:
            return Pow.apply(self, other)

        # Use expand for broadcasting
        broadcast_shape = np.broadcast_shapes(self.shape, other.shape)
        x = self.expand(broadcast_shape)
        y = other.expand(broadcast_shape)

        return Pow.apply(x, y)

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

        return IAdd.apply(self, expanded_other)

    def __getitem__(self, idx):
        return GetItem.apply(self, idx=idx)

    def __setitem__(
        self, idx: Union[int, slice, tuple], value: Union[Self, float, int]
    ):
        if not isinstance(value, Tensor):
            value = Tensor(value, requires_grad=False)  # this is important

        return SetItem.apply(self, idx=idx, value=value)

    def sum(self, axis=None, keepdims=False):
        return Sum.apply(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims=False):
        return Mean.apply(self, axis=axis, keepdims=keepdims)

    def max(self, axis=None, keepdims=False):
        return Max.apply(self, axis=axis, keepdims=keepdims)

    def sqrt(self):
        return Sqrt.apply(self)

    def maximum(self, other: Union[Self, float, int]):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)

        # Use expand for broadcasting
        broadcast_shape = np.broadcast_shapes(self.shape, other.shape)
        x = self.expand(broadcast_shape)
        y = other.expand(broadcast_shape)

        return Maximum.apply(x, y)

    def pad(self, pad_width, mode="constant", constant_values=0):
        return Pad.apply(
            self,
            pad_width=pad_width,
            mode=mode,
            constant_values=constant_values,
        )

    def forward(self, data):
        pass

    def backward(self, grad=None):
        if not self.requires_grad:
            return

        # Initialize gradient if none provided
        if grad is None:
            grad = np.ones_like(self.data)
        elif isinstance(grad, Tensor):
            grad = grad.data
        else:
            grad = np.asarray(grad)

        self._grad = grad  # store as np array directly

        # Build computational graph in reverse order
        topological_sorted_tensors = []
        visited = set()

        stack = [(self, False)]  # node, whether all children are visited

        # Post-order traversal
        while stack:
            node, visited_children = stack.pop()
            if node not in visited:
                if not visited_children:
                    # first time we see this node, push it again with visited_children=True
                    stack.append((node, True))
                    # then push its parents
                    if node.creator is not None:
                        for p in node.creator.tensors:
                            if p.requires_grad:
                                stack.append((p, False))
                else:
                    # second time we see this node, children are done
                    visited.add(node)
                    topological_sorted_tensors.append(node)

        # Backward pass
        for tensor in reversed(topological_sorted_tensors):
            if tensor.creator is not None:
                # Call function's backward to get gradients for inputs
                grads = tensor.creator.backward(tensor.grad)
                if not isinstance(grads, tuple):
                    grads = (grads,)
                # Accumulate grads into the input tensors
                for input_tensor, g in zip(tensor.creator.tensors, grads):
                    if (
                        input_tensor is not None
                        and input_tensor.requires_grad
                        and g is not None
                    ):
                        input_tensor._accumulate_grad(g)
                tensor.creator.tensors = None

        # Clear references if needed
        for node in topological_sorted_tensors:
            node.creator = None

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
        return Reshape.apply(self, shape=shape)

    def expand(self, *shape) -> "Tensor":
        """Movement op: broadcast without copying"""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]

        return Expand.apply(self, shape=shape)

    def permute(self, *dims) -> "Tensor":
        return Permute.apply(self, dims=dims)

    def transpose(self, dim0=0, dim1=1):
        return Transpose.apply(self, dim0=dim0, dim1=dim1)

    def strided_windows(self, kernel_size: int, stride: int) -> "Tensor":
        return StridedWindows.apply(self, kernel_size=kernel_size, stride=stride)

    def roll(self, shifts: int, dims: int):
        return Roll.apply(self, shifts=shifts, dims=dims)

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

        # Initialize or accumulate gradient
        if self._grad is None:
            if idx is not None:
                # Initialize with zeros
                self._grad = Tensor(np.zeros_like(self.data), requires_grad=False)
                # Accumulate at index without reshaping
                self._grad.data[idx] = grad.data
            else:
                if not isinstance(grad, Tensor):
                    grad = Tensor(grad, requires_grad=False)
                self._grad = grad
        else:
            if idx is not None:
                # Accumulate at index without reshaping
                self._grad.data[idx] += grad.data
            else:
                self._grad.data += grad.data

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

    def __eq__(self, other):
        return self.data == other

    def __hash__(self):
        return id(self)


"""
Binary Ops
"""


class Add(Function):
    def forward(self, x, y):
        return x + y

    def backward(self, grad: Tensor):
        grad_x = grad.data if self.tensors[0].requires_grad else None
        grad_y = grad.data if self.tensors[1].requires_grad else None
        return grad_x, grad_y


class Mul(Function):
    def forward(self, x, y):
        return x * y

    def backward(self, grad: Tensor):
        grad_x = (
            grad.data * self.tensors[1].data if self.tensors[0].requires_grad else None
        )
        grad_y = (
            grad.data * self.tensors[0].data if self.tensors[1].requires_grad else None
        )
        return grad_x, grad_y


class Pow(Function):
    def forward(self, x, y):
        return x**y

    def backward(self, grad):
        """
        d(loss) / dx = d(loss) / d(x**y) * d(x**y) / dx
        d(loss) / d(x**y) = result.grad
        d(x**y) / dx = y*x^(y-1)
        d(x**y) / dy = x**y * ln(x)
        where x is self
        y is other
        """
        x = self.tensors[0]
        y = self.tensors[1]
        grad_x = None
        grad_y = None

        if x.requires_grad:
            grad_x = y.data * (x.data ** (y.data - 1)) * grad.data

        if y.requires_grad:
            valid_base = x.data > 0
            grad_y = (x.data**y.data) * np.log(np.abs(x.data)) * grad.data
            grad_y = np.where(valid_base, grad_y, 0)
        return grad_x, grad_y


class Matmul(Function):
    def forward(self, x, y):
        if x.data.ndim == 1 and y.data.ndim == 1:
            return np.dot(x.data, y.data)
        else:
            return np.matmul(x.data, y.data)

    def backward(self, grad):
        x = self.tensors[0]
        y = self.tensors[1]
        grad_x = None
        grad_y = None

        # Handle vector @ vector case separately (1D @ 1D)
        if x.data.ndim == 1 and y.data.ndim == 1:
            if x.requires_grad:
                grad_x = grad.data * y.data
            if y.requires_grad:
                grad_y = grad.data * x.data
            return grad_x, grad_y
        # Matrix multiplication case
        else:
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
            if x.requires_grad and grad is not None:
                grad_x = grad.data @ y.data.T

            # Compute gradient of loss w.r.t. y.data
            # d(loss) / d(y.data) = d(loss) / d(result) * d(result) / d(y.data)
            # d(result) / d(y.data) = x.data.T
            # x.data:    (n, m)
            # grad:  (n, p)
            # x.T:       (m, n) --> T operation is equivalent to swapaxes(-1, -2)
            # matmul(x.T, grad) = (m, n) @ (n, p) = (m, p) = y.grad
            if y.requires_grad and grad is not None:
                x_transposed = (
                    x.transpose(1, 2) if x.data.ndim == 3 else x.data.transpose()
                )
                grad_y = (
                    np.sum(x_transposed.data @ grad.data, axis=0)
                    if x.data.ndim == 3
                    else x_transposed.data @ grad.data
                )

        return grad_x, grad_y


class IAdd(Function):
    def forward(self, x, y):
        # Update data in-place
        x += y
        return x

    def backward(self, grad):
        # Both inputs receive the same gradient (like Add)
        return grad.data, grad.data


class GetItem(Function):
    """Get item from tensor using numpy-style indexing"""

    def forward(self, x, idx):
        self.idx = idx
        return x[idx]

    def backward(self, grad):
        grad = grad.data if isinstance(grad, Tensor) else grad

        # Create a zero tensor of the original shape
        out = np.zeros_like(self.tensors[0].data)
        # Place the gradient in the correct location
        out[self.idx] = grad
        return out


class SetItem(Function):
    """
    In-place assignment operation using views

    Args:
        idx (tuple): indices to assign the value to
        value (Tensor): value to assign
    """

    def forward(self, x, idx, value):
        # Extract numpy array from value
        val_data = value.data if isinstance(value, Tensor) else value
        x[idx] = val_data
        self.idx = idx
        return x

    def backward(self, grad):
        grad = grad.data
        if np.isscalar(grad):
            return grad[self.idx].sum()
        else:
            return grad[self.idx]


class Sqrt(Function):
    def forward(self, x):
        # Store input for backward pass
        self.x = x
        return np.sqrt(x)

    def backward(self, grad):
        # d/dx(sqrt(x)) = 1/(2*sqrt(x))
        # dL/dx = dL/dy * dy/dx = grad * 1/(2*sqrt(x))
        # where dL/dy is the current gradient
        return grad.data * 0.5 / np.sqrt(self.x.data)


"""
Reduction Ops
"""


class Sum(Function):
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

    def forward(self, x, axis=None, keepdims=False):
        # Handle scalar case
        if not hasattr(x, "ndim") or x.ndim == 0:
            return x
        # Normalize axis
        self.axis = (axis,) if isinstance(axis, int) else axis
        self.keepdims = keepdims
        self.x_shape = x.shape
        return np.sum(x, axis=self.axis, keepdims=self.keepdims)

    def backward(self, grad):
        # Use expand to handle gradient broadcasting

        if grad is None:
            return None

        grad = grad.data

        # Turn axis into a tuple
        if isinstance(self.axis, int):
            reduce_axes = (self.axis,)
        else:
            reduce_axes = self.axis

        # If we never specified an axis, then reduce_axes=None means a global sum
        if reduce_axes is None:
            # Summed over all dims, so grad is scalar, shape=()
            # Just broadcast to original shape:
            return np.broadcast_to(grad, self.x_shape).copy()

        if not self.keepdims:
            # Re-insert those axes as size=1 so that broadcasting works
            # Sort them or reverse them so that inserting doesn’t shift the later dims incorrectly
            for ax in sorted(reduce_axes):
                grad = np.expand_dims(grad, ax)

        # Now grad has shape (2,8,1) if we did sum over axis=2
        # broadcast to (2,8,5)
        return np.broadcast_to(grad, self.x_shape).copy()


class Max(Function):
    def forward(self, x, axis=None, keepdims=False):
        """
        Compute the max of tensor elements

        Args:
            axis (int or tuple of ints, optional): Axis or axes along which a max is performed. The default, axis=None, maxes all of the elements of the input tensor.
            keepdims (bool, optional): If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input tensor.
        """
        axis = (axis,) if isinstance(axis, int) else axis
        self.axis = axis
        self.keepdims = keepdims
        return np.max(x, axis=axis, keepdims=keepdims)

    def backward(self, grad):
        """
        d(loss) / dx = d(loss) / d(max(x)) * d(max(x)) / dx
        d(loss) / d(max(x)) = result.grad
        d(max(x)) / dx = 1 if x == max(x), 0 otherwise
        """
        x = self.tensors[0]

        # Compute max values along the specified axes
        max_vals = np.max(x.data, axis=self.axis, keepdims=True)
        mask = x.data == max_vals

        if self.axis is None or (isinstance(self.axis, tuple) and len(self.axis) > 1):
            # Global max: distribute equally among all max occurrences
            count = np.sum(mask)
            return grad.data * mask / count
        else:
            # Single axis: use first occurrence
            ax = self.axis[0] if isinstance(self.axis, tuple) else self.axis
            cumsum = np.cumsum(mask, axis=ax)
            first_occur = cumsum == 1
            return grad.data * (mask * first_occur)


class Maximum(Function):
    """
    Element-wise maximum between self and other.
    When both inputs equal the maximum, gradient is split equally between them.
    """

    def forward(self, x, y):
        out = np.maximum(x, y)
        self.out_data = out
        return out

    def backward(self, grad):
        x = self.tensors[0]
        y = self.tensors[1]

        grad_x = None
        grad_y = None

        x_matches = x.data == self.out_data
        y_matches = y.data == self.out_data

        if x.requires_grad:
            grad_x = grad.data * (x_matches * (1.0 - 0.5 * y_matches))

        if y.requires_grad:
            grad_y = grad.data * (y_matches * (1.0 - 0.5 * x_matches))

        return grad_x, grad_y


class Mean(Function):
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

    def forward(self, x, axis=None, keepdims=False):
        # Normalize axis to a tuple
        axis = (axis,) if isinstance(axis, int) else axis
        self.axis = axis
        self.keepdims = keepdims
        return np.mean(x, axis=axis, keepdims=keepdims)

    def backward(self, grad):
        # Use expand for gradient broadcasting
        grad = grad.expand(
            self.tensors[0].shape if self.keepdims else self.tensors[0].shape
        )

        # Scale gradient by number of elements
        num_elements = (
            np.prod([self.tensors[0].shape[ax] for ax in self.axis])
            if self.axis is not None
            else self.tensors[0].size
        )
        return grad.data / num_elements


"""
Movement Ops
"""


class View(Function):
    def forward(self, x, new_shape=(1,)):
        if len(new_shape) == 1 and isinstance(new_shape[0], (tuple, list)):
            new_shape = new_shape[0]

        # Handle -1 in shape
        if -1 in new_shape:
            # Can only have one -1 in shape
            if new_shape.count(-1) > 1:
                raise ValueError("Only one -1 dimension is allowed in shape")

            # Calculate the size of the -1 dimension
            neg_idx = new_shape.index(-1)
            known_size = np.prod(
                [d for i, d in enumerate(new_shape) if i != neg_idx and d != -1]
            )
            # Compute the missing dimension
            inferred_size = int(x.size // known_size)
            # Replace -1 with inferred size
            new_shape = tuple(inferred_size if d == -1 else d for d in new_shape)

        if x.size != np.prod(new_shape):
            raise ValueError(
                f"Size of new view must match size of original tensor: {x.size} != {np.prod(new_shape)}"
            )
        # Store original shape for backward
        self.original_shape = x.shape

        return np.reshape(x, new_shape)

    def backward(self, grad):
        # reshape grad to original shape
        return grad.reshape(self.original_shape).data if grad is not None else None


class Expand(Function):
    def forward(self, x, shape=(1,)):
        self.original_shape = x.shape
        expanded = np.broadcast_to(x, shape)
        return expanded.copy()

    def backward(self, grad):
        # Ensure grad is a NumPy array
        if isinstance(grad, Tensor):
            grad = grad.data

        # Handle extra leading dimensions
        if len(grad.shape) > len(self.original_shape):
            reduce_dims = list(range(len(grad.shape) - len(self.original_shape)))
        else:
            reduce_dims = []

        # Handle broadcasting dimensions
        for i, (self_dim, grad_dim) in enumerate(
            zip(
                self.original_shape[::-1], grad.shape[-len(self.original_shape) :][::-1]
            )
        ):
            if self_dim == 1 and grad_dim != 1:
                reduce_dims.append(len(grad.shape) - 1 - i)

        # Sum across all reduction dimensions
        if reduce_dims:
            grad = np.sum(grad, axis=tuple(reduce_dims), keepdims=True)

        # Ensure final shape matches original
        if grad.shape != self.original_shape:
            grad = grad.reshape(self.original_shape)

        return grad


class Reshape(Function):
    def forward(self, x, shape=(1,)):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]

        self.original_shape = x.shape
        return np.reshape(x, shape)

    def backward(self, grad):
        return grad.data.reshape(self.original_shape) if grad is not None else None


class Transpose(Function):
    def _get_transpose_axes(self, x, dim0: int, dim1: int) -> Tuple[int, ...]:
        axes = list(range(x.ndim))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        return tuple(axes)

    def forward(self, x, dim0=0, dim1=1):
        ndim = x.ndim
        if not (0 <= dim0 < ndim and 0 <= dim1 < ndim):
            raise ValueError(
                f"Dimensions out of range for tensor with {ndim} dimensions"
            )

        axes = self._get_transpose_axes(x, dim0, dim1)
        self.dim0 = dim0
        self.dim1 = dim1
        return np.transpose(x, axes)

    def backward(self, grad):
        transposed_grad = np.transpose(
            grad.data,
            self._get_transpose_axes(self.tensors[0].data, self.dim0, self.dim1),
        )
        return transposed_grad


class Pad(Function):
    """
    Pad the tensor with zeros.
    Args:
        pad_width: If tuple of 2 values, interpreted as padding for last dimension (PyTorch style).
                    If tuple of tuples, each inner tuple is (pad_before, pad_after) for each dimension.
                    If int, pad all dimensions with same value.
        mode: Padding mode (default: "constant")
        constant_values: Value to pad with (default: 0)
    """

    def forward(self, x, pad_width, mode="constant", constant_values=0):
        # Normalize pad_width to numpy style
        if isinstance(pad_width, int):
            # For int, create tuple of tuples for all dimensions
            pad_width = tuple((pad_width, pad_width) for _ in range(x.data.ndim))
        elif isinstance(pad_width, (tuple, list)):
            if len(pad_width) == 2 and not isinstance(pad_width[0], (tuple, list)):
                # (left, right) -> pad only last dimension
                pad_width = tuple((0, 0) for _ in range(x.data.ndim - 1)) + (
                    tuple(pad_width),
                )
            elif len(pad_width) == 4 and not isinstance(pad_width[0], (tuple, list)):
                # (left, right, top, bottom) -> pad last two dimensions
                pad_width = tuple((0, 0) for _ in range(x.data.ndim - 2)) + (
                    (pad_width[2], pad_width[3]),  # height/rows padding
                    (pad_width[0], pad_width[1]),  # width/cols padding
                )
        self.pad_width = pad_width
        self.out_data = np.pad(x, pad_width, mode=mode, constant_values=constant_values)
        return self.out_data

    def backward(self, grad):
        # Extract the unpadded region
        slices = tuple(
            slice(p[0], s - p[1]) for s, p in zip(self.out_data.shape, self.pad_width)
        )
        return grad.data[slices]


class Cat(Function):
    """Concatenates tensors along specified axis"""

    def forward(self, *tensors, axis=0):
        self.axis = axis
        self.original_shapes = [t.data.shape for t in tensors]
        return np.concatenate([t.data for t in tensors], axis=axis)

    def backward(self, grad):
        grads = []
        start_idx = 0
        for t, shape in zip(self.tensors, self.original_shapes):
            if t.requires_grad:
                slice_idx = [slice(None)] * len(shape)
                slice_idx[self.axis] = slice(start_idx, start_idx + shape[self.axis])
                # Extract the portion of grad corresponding to this input tensor
                grad_slice = grad.data[tuple(slice_idx)]
                grads.append(grad_slice)
            else:
                grads.append(None)
            start_idx += shape[self.axis]

        return tuple(grads)


class Permute(Function):
    """Movement op: reorder dimensions"""

    def forward(self, x, dims):
        # Normalize dims if it's nested (e.g. a single tuple in dims)
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = dims[0]
        self.dims = dims
        return np.transpose(x, dims)

    def backward(self, grad):
        # Compute inverse permutation to get back to original ordering
        inv_dims = [self.dims.index(i) for i in range(len(self.dims))]
        return np.transpose(grad.data, inv_dims)


class Stack(Function):
    """
    Join a sequence of tensors along a new axis.

    Args:
        tensors (list of Tensors): sequence of tensors to stack
        axis (int): axis along which to stack

    Returns:
        Tensor: stacked tensor
    """

    def forward(self, *tensors, axis=0):
        if not tensors:
            raise ValueError("Need at least one tensor to stack")

        # Memory optimization: Use numpy.concatenate with expanded dimensions
        # instead of stack to avoid temporary list creation
        expanded_arrays = [np.expand_dims(t.data, axis=axis) for t in tensors]
        stacked_data = np.concatenate(expanded_arrays, axis=axis)
        self.axis = axis
        return stacked_data

    def backward(self, grad):
        # Memory optimization: Use views instead of splits where possible
        grad_size = grad.shape[self.axis]
        chunk_size = grad_size // len(self.tensors)

        grads = []
        for i, tensor in enumerate(self.tensors):
            if not tensor.requires_grad:
                continue

            # Create slice indices
            idx = [slice(None)] * grad.ndim
            idx[self.axis] = slice(i * chunk_size, (i + 1) * chunk_size)
            grad_slice = grad.data[tuple(idx)]
            grads.append(grad_slice.reshape(tensor.shape))
        return tuple(grads)


class StridedWindows(Function):
    """Movement op: create strided windows view of the tensor"""

    def forward(self, x, kernel_size: int, stride: int):
        batch_size, channels, height, width = x.shape
        H_out = (height - kernel_size) // stride + 1
        W_out = (width - kernel_size) // stride + 1

        self.kernel_size = kernel_size
        self.stride = stride
        self.H_out = H_out
        self.W_out = W_out
        self.batch_size = batch_size
        self.channels = channels
        self.data_shape = x.shape
        # Directly produce (batch_size, channels, H_out, W_out, kernel_size, kernel_size)
        return np.lib.stride_tricks.as_strided(
            x,
            shape=(H_out, W_out, batch_size, channels, kernel_size, kernel_size),
            strides=(
                x.strides[2] * stride,  # steps in the height dimension
                x.strides[3] * stride,  # steps in the width dimension
                x.strides[0],  # batch dimension stride
                x.strides[1],  # channels dimension stride
                x.strides[2],  # inside window vertical steps
                x.strides[3],  # inside window horizontal steps
            ),
            writeable=False,
        )

    def backward(self, grad):
        # Reshape grad back to original window format
        grad = grad.data.reshape(
            self.H_out,
            self.W_out,
            self.batch_size,
            self.channels,
            self.kernel_size,
            self.kernel_size,
        )
        grad = grad.transpose(2, 3, 0, 1, 4, 5)
        grad_padded = np.zeros(self.data_shape, dtype=grad.dtype)

        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                grad_padded[
                    :,
                    :,
                    i : i + self.H_out * self.stride : self.stride,
                    j : j + self.W_out * self.stride : self.stride,
                ] += grad[:, :, :, :, i, j]

        return grad_padded


class Roll(Function):
    def forward(self, x, shifts, dims=None):
        """Roll tensor elements along a given dimension

        Args:
            x: Input tensor
            shifts: Number of places by which elements are shifted
            dims: Dimension along which elements are shifted. None means flatten first
        """
        self.shifts = shifts
        self.dims = dims
        self.input_shape = x.shape
        return np.roll(x, shift=shifts, axis=dims)

    def backward(self, grad):
        """Backward pass for roll operation

        The gradient is rolled in the opposite direction to undo the forward roll.
        For example, if we rolled right by 2 in forward pass, we roll left by 2 here.

        Args:
            grad: Gradient of the loss with respect to output
        """
        grad = grad.data

        # Handle scalar gradients by reshaping to original input shape
        if grad.ndim == 0:
            grad = np.full(self.input_shape, grad)

        # Roll gradient in opposite direction by negating the shift
        return np.roll(grad, shift=-self.shifts, axis=self.dims)
