import logging
from abc import abstractmethod
from typing import (
    Any,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

try:
    # drop-in replacement for numpy for GPU acceleration
    import cupy as np  # type: ignore

    _ = np.cuda.runtime.getDeviceCount()  # Check if a CUDA device is available
except Exception:
    import numpy as np

logger = logging.getLogger(__name__)


class Function:
    """
    Base class for differentiable operations.

    Subclasses of `Function` should implement the `forward` and `backward` methods to define the
    forward and backward passes of a particular operation. Some subclasses can be found in `functional.py` module
    """

    def __init__(self, *tensors: "Tensor"):
        """
        Initialize a `Function` with a set of input tensors.

        Args:
            *tensors (Tensor): The input tensors for this operation.
        """
        self.tensors = tensors

    @abstractmethod
    def forward(self, *args: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Perform the forward pass of this operation.

        This method should be overridden by subclasses to define the specific behavior
        of the operation. It receives NumPy arrays corresponding to the data of the input tensors.

        Args:
            *args (np.ndarray): Data arrays for the input tensors.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            np.ndarray: The result of the forward pass as a NumPy array.

        Raises:
            NotImplementedError: If this method is not implemented in a subclass.
        """
        raise NotImplementedError("Forward pass not implemented for this function")

    @abstractmethod
    def backward(self, grad: "Tensor") -> np.ndarray:
        """
        Perform the backward pass of this operation.

        This method should be overridden by subclasses to define how gradients are
        computed and propagated back to the input tensors.

        In this context:
        - "grad" (the method argument) is the gradient of the loss function with respect to the *output* of this operation (dL/d[out]).
        - The return value should be the gradient of the loss function with respect to the *input* of this operation (dL/d[input]), so it can be passed further back along the computational graph.

        Args:
            grad (Tensor): The gradient with respect to the **output** of this operation.

        Returns:
            np.ndarray: The gradient with respect to the **input(s)**.

        Raises:
            NotImplementedError: If this method is not implemented in a subclass.
        """
        raise NotImplementedError("Backward pass not implemented for this function")

    @classmethod
    def apply(cls, *tensors: "Tensor", **kwargs: Any) -> "Tensor":
        """
        Construct and apply this function to the given tensors.

        This method:
        1) Creates an instance of the function.
        2) Extracts the `.data` from the input tensors to pass into the function's `forward` method.
        3) Wraps the result in a new `Tensor` that references this function (for backprop).

        Args:
            *tensors (Tensor): Input tensors to the operation.
            **kwargs (Any): Additional keyword arguments passed to the forward method.

        Returns:
            Tensor: The resulting tensor after the forward operation.
        """
        func = cls(*tensors)
        # Run forward pass with tensor.data already, so we don't need to get it again
        out_data = func.forward(*(inp.data for inp in tensors), **kwargs)

        # Create output tensor
        requires_grad = any(inp.requires_grad for inp in tensors)
        out = Tensor(out_data, creator=func, requires_grad=requires_grad)
        return out

    @staticmethod
    def unbroadcast(grad_arr: np.ndarray, to_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Sum out broadcasted dimensions so that grad_arr can match to_shape.
        Essentially the inverse of numpy's broadcasting.
        Args:
            grad_arr (np.ndarray): Gradient array to unbroadcast.
            to_shape (Tuple[int, ...]): Shape to unbroadcast to.
        Returns:
            np.ndarray: Unbroadcasted gradient array.
        """
        if grad_arr.shape == to_shape:
            # No broadcasting happened
            return grad_arr

        # e.g. grad_arr.shape might be (4,3,2) but to_shape is (1,3,2).
        # We need to sum over the broadcasted dimension (dim=0) which was 1 in to_shape.

        # 1) If grad_arr.ndim > len(to_shape), we must sum across extra leading dims:
        while len(grad_arr.shape) > len(to_shape):
            grad_arr = grad_arr.sum(axis=0, keepdims=False)

        # 2) Now both have same ndim.  For each dim where to_shape[dim] == 1 but grad_arr.shape[dim] != 1, we sum out that dimension.
        for dim in range(len(to_shape)):
            if to_shape[dim] == 1 and grad_arr.shape[dim] != 1:
                grad_arr = grad_arr.sum(axis=dim, keepdims=True)

        # At this point grad_arr.shape should match to_shape exactly
        return grad_arr


class Tensor:
    """
    A `Tensor` is the core data structure of this autograd engine.

    It holds a NumPy array, an optional reference to a creator function, and gradient information.
    """

    def __init__(
        self,
        data: Union[np.ndarray, float, int, Sequence[float], Sequence[Sequence[float]]],
        creator: Optional[Function] = None,
        requires_grad: bool = True,
    ):
        """
        Initialize a `Tensor`.

        Args:
            data (Union[np.ndarray, float, int, Sequence[float], Sequence[Sequence[float]]]):
                The data for this tensor. Will be converted to a NumPy array of type float32.
            creator (Optional[Function], optional): The function that created this tensor.
                Defaults to None if this tensor is a leaf.
            requires_grad (bool, optional): Whether this tensor requires gradients. Defaults to True.
        """
        self.data = np.asarray(data, dtype=np.float32)
        self._grad: Optional["Tensor"] = None  # Lazily initialized
        self.creator = creator
        self._backward = lambda: None
        self.requires_grad = requires_grad

    @property
    def grad(self) -> Optional["Tensor"]:
        """
        Getter method of the gradient of this tensor.

        The internal `_grad` is stored either as a `Tensor` or `None`. If it is stored
        as a NumPy array, it will be wrapped in a `Tensor` before returning.

        Returns:
            Optional[Tensor]: The gradient if it exists, or None.
        """
        if isinstance(self._grad, np.ndarray):
            return Tensor(self._grad, requires_grad=False)
        return self._grad

    @grad.setter
    def grad(self, value: Union["Tensor", np.ndarray, float, int, None]) -> None:
        """
        Set or accumulate gradient for this tensor.

        If the tensor does not have a gradient yet, we initialize it with the provided value.
        Otherwise, the provided value is added in place to the existing gradient.

        Args:
            value (Union[Tensor, np.ndarray, float, int, None]): The gradient value or None.
        """
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

    def view(self, *shape: int) -> "Tensor":
        """
        Create a view of the tensor with the specified shape without copying the underlying data.

        The new shape must be compatible with the total number of elements in the input tensor.

        Raises:
            ValueError: If more than one -1 is specified in the new shape or if the new shape does
                not match the input tensor's total size.

        Args:
            *shape (int): The desired shape. If -1 is present, it is inferred based on the remaining dimensions.

        Returns:
            Tensor: A new tensor that shares data with the original but is shaped differently.
        """
        return View.apply(self, new_shape=shape)

    @staticmethod
    def stack(tensors: List["Tensor"], axis: int = 0) -> "Tensor":
        """
        Stack a list of tensors along a new dimension.
        This operation joins a sequence of tensors by inserting a new axis at the specified position
        and concatenating along that axis.

        Args:
            tensors (List[Tensor]): The list of tensors to stack.
            axis (int, optional): The dimension along which to stack. Defaults to 0.

        Returns:
            Tensor: A new tensor created by stacking.

        Examples:
            >>> import numpy as np
            >>> from your_module import Tensor, Stack
            >>> t1 = Tensor(np.array([1, 2]))
            >>> t2 = Tensor(np.array([3, 4]))
            >>> op = Tensor.stack([t1, t2], axis=0)
            >>> print(op)
            Tensor([[1, 2], [3, 4]])
        """
        return Stack.apply(*tensors, axis=axis)

    @staticmethod
    def cat(tensors: List["Tensor"], axis: int = 0) -> "Tensor":
        """
        Concatenate a list of tensors along the specified dimension.
        This operation concatenates the input tensors along the given axis.

        Args:
            tensors (List[Tensor]): The list of tensors to concatenate.
            axis (int, optional): The dimension along which to concatenate. Defaults to 0.

        Returns:
            Tensor: The concatenated tensor.

        Examples:
            >>> import numpy as np
            >>> from your_module import Tensor, Cat
            >>> t1 = Tensor(np.array([[1, 2]]))
            >>> t2 = Tensor(np.array([[3, 4]]))
            >>> result = Tensor.cat( [t1, t2], axis=0)
            >>> result
            Tensor([[1, 2],[3, 4]]
        """
        return Cat.apply(*tensors, axis=axis)

    def __add__(self, other: Union["Tensor", float, int]) -> "Tensor":
        """
        Element-wise addition of two tensors (or a tensor and a scalar).

        Args:
            other (Union[Tensor, float, int]): The tensor or scalar to add.

        Returns:
            Tensor: The result of addition.
        """
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)

        return Add.apply(self, other)

    def __mul__(self, other: Union["Tensor", float, int]) -> "Tensor":
        r"""
        Element-wise multiplication of two tensors (or a tensor and a scalar).
        $$
        z = x \cdot y
        $$

        Args:
            other (Union[Tensor, float, int]): The tensor or scalar to multiply with.

        Returns:
            Tensor: The result of multiplication.
        """
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)

        return Mul.apply(self, other)

    def __matmul__(self, other: Union["Tensor", float, int]) -> "Tensor":
        """
        Perform matrix multiplication (dot product) with another tensor.

        For higher-dimensional tensors, np.matmul broadcasting rules are followed.

        Args:
            other (Union[Tensor, float, int]): The tensor or scalar to matmul with.

        Returns:
            Tensor: The result of matrix multiplication.
        """
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        return Matmul.apply(self, other)

    def __pow__(self, other: Union["Tensor", float, int]) -> "Tensor":
        """
        Compute the power operation $z = x^y$ with another tensor or scalar.

        Args:
            other (Union[Tensor, float, int]): The exponent.

        Returns:
            Tensor: The result of the power operation.
        """
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)

        return Pow.apply(self, other)

    def __iadd__(self, other: Union["Tensor", float, int]) -> "Tensor":
        """
        In-place addition (self += other).

        Broadcasting rules apply if shapes differ.
        This should maintain the computational graph while modifying the tensor in-place.

        Args:
            other (Union[Tensor, float, int]): The tensor or scalar to add.

        Returns:
            Tensor: This tensor, after in-place addition.
        """
        if not isinstance(other, Tensor):
            other = Tensor(other)

        # Use expand for broadcasting
        broadcast_shape = np.broadcast_shapes(self.shape, other.shape)
        expanded_other = other.expand(broadcast_shape)
        return IAdd.apply(self, expanded_other)

    def __getitem__(self, idx: Union[int, slice, tuple]) -> "Tensor":
        """
        Get a sliced or indexed view of the tensor.

        Args:
            idx (Union[int, slice, tuple]): The index or slice.

        Returns:
            Tensor: A new tensor that shares data with the original.
        """
        return GetItem.apply(self, idx=idx)

    def __setitem__(
        self, idx: Union[int, slice, tuple], value: Union["Tensor", float, int]
    ) -> "Tensor":
        """
        Set a portion of the tensor to a given value.

        Args:
            idx (Union[int, slice, tuple]): The index or slice.
            value (Union[Tensor, float, int]): The value to set.

        Returns:
            Tensor: The same tensor after the in-place assignment.
        """
        if not isinstance(value, Tensor):
            value = Tensor(value, requires_grad=False)  # this is important

        return SetItem.apply(self, value, idx=idx)

    def sum(
        self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False
    ) -> "Tensor":
        r"""
        Compute the sum of all elements (or along specified axis).

        This function computes the sum of the input tensor elements along a specified axis or axes.
        If no axis is specified, all elements of the tensor are summed. Optionally, the reduced
        dimensions can be kept in the output tensor.

        The summation is mathematically represented as:

            $$
            y = \sum_{i \in A} x_i
            $$

        where A represents the specified axis or axes.

        Args:
            axis (int or tuple of ints, optional): Axis or axes along which the sum is performed.
                If None, the sum of all elements is computed.
            keepdims (bool, optional): If True, the reduced axes are left in the result as dimensions
                with size one so that the result can be broadcast correctly against the input tensor.

        Examples:
            For example:
                - Original tensor shape (3, 4, 5), axis (1, 2), keepdims True  → result shape (3, 1, 1)
                - Original tensor shape (3, 4, 5), axis (1, 2), keepdims False → result shape (3,)
                - Original tensor shape (3, 4, 5), axis None, keepdims True   → result shape (1,)
                - Original tensor shape (3, 4, 5), axis None, keepdims False  → result shape ()

        Args:
            axis (Optional[Union[int, Tuple[int, ...]]], optional): Axis or axes to sum over.
                If None, sums over all elements. Defaults to None.
            keepdims (bool, optional): Keep the reduced dimensions as size 1. Defaults to False.

        Returns:
            Tensor: The tensor with summed values.
        """
        return Sum.apply(self, axis=axis, keepdims=keepdims)

    def mean(
        self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False
    ) -> "Tensor":
        r"""
        Compute the mean of elements (or along specified axis).

        This function computes the mean of the input tensor elements along a specified axis or axes.
        If no axis is specified, the mean of all elements is computed. Optionally, the reduced
        dimensions can be kept in the output tensor.

        The mean is mathematically defined as:
            $$
            y = \frac{1}{N} \sum_{i \in A} x_i
            $$

        where A represents the specified axis or axes and N is the number of elements summed.

        For example:
            - Original tensor shape (3, 4, 5), axis (1, 2), keepdims True  → result shape (3, 1, 1)
            - Original tensor shape (3, 4, 5), axis (1, 2), keepdims False → result shape (3,)
            - Original tensor shape (3, 4, 5), axis None, keepdims True   → result shape (1,)
            - Original tensor shape (3, 4, 5), axis None, keepdims False  → result shape ()

        Args:
            axis (Optional[Union[int, Tuple[int, ...]]], optional): Axis or axes to average over.
                If None, averages over all elements. Defaults to None.
            keepdims (bool, optional): Keep the reduced dimensions as size 1. Defaults to False.

        Returns:
            Tensor: The tensor with mean values.
        """
        return Mean.apply(self, axis=axis, keepdims=keepdims)

    def max(
        self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False
    ) -> "Tensor":
        r"""
        Compute the maximum value of elements (or along specified axis).

        This function computes the maximum value of the input tensor along a specified axis or axes.
        If no axis is specified, the maximum over all elements is computed. Optionally, the reduced
        dimensions can be kept in the output tensor.

        Mathematically, the maximum is computed as:

            $$
            y = \max_{i \in A} \; x_i
            $$

        where A represents the specified axis or axes.

        Args:
            axis (Optional[Union[int, Tuple[int, ...]]], optional): Axis or axes to compute max over.
                If None, computes global max. Defaults to None.
            keepdims (bool, optional): Keep the reduced dimensions as size 1. Defaults to False.

        Returns:
            Tensor: The tensor with maximum values.
        """
        return Max.apply(self, axis=axis, keepdims=keepdims)

    def gather(self, index: int = 0) -> "Tensor":
        """
        Gather rows from a 2D tensor using specified row indices.

        This operation extracts rows from the input tensor corresponding to the given index
        or indices. It is particularly useful for selecting specific rows from a matrix,
        such as picking particular examples from a batch of data. When a single index is provided,
        it returns the corresponding row; when multiple indices are provided (e.g., as a list or tuple),
        it returns a new tensor composed of rows at those positions.

        Args:
            index (int or list/tuple of ints): The row index or indices to gather from the tensor.
                Defaults to 0.

        Returns:
            Tensor: A new tensor containing the gathered rows.

        Example:
            >>> tensor = Tensor([[10, 20], [30, 40], [50, 60]])
            >>> gathered = tensor.gather([0, 2])
            >>> print(gathered)
            Tensor([[10, 20],
                    [50, 60]])
        """
        return Gather.apply(self, index=index)

    def sqrt(self) -> "Tensor":
        """
        Compute the element-wise square root of the tensor.

        Returns:
            Tensor: The result of the sqrt operation.
        """
        return Sqrt.apply(self)

    def maximum(self, other: Union["Tensor", float, int]) -> "Tensor":
        """
        Element-wise maximum between two tensors or a tensor and a scalar.

        This function performs an element-wise comparison between two input tensors and returns a new tensor
        containing the maximum value from each pair of elements. When both inputs are equal, the gradient is
        split equally between them.

        Args:
            other (Union[Tensor, float, int]): The tensor or scalar to compare.

        Returns:
            Tensor: Element-wise maximum.
        """
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)

        return Maximum.apply(self, other)

    def pad(
        self,
        pad_width: Union[
            int, Tuple[int, int], Tuple[int, int, int, int], Tuple[Tuple[int, int], ...]
        ],
        mode: str = "constant",
        constant_values: Union[int, float] = 0,
    ) -> "Tensor":
        """
        Pad the tensor according to specified widths in each dimension.

        This operation pads the input tensor using the given padding widths and mode.
        The interpretation of the ``pad_width`` argument is as follows:

        - If an int is provided, all dimensions are padded with that value.
        - If a tuple of 2 values is provided, it is interpreted as padding for the last dimension (PyTorch style): (pad_left, pad_right).
        - If a tuple of 4 values is provided, it is interpreted as padding for the last two dimensions: (pad_left, pad_right, pad_top, pad_bottom).
        - If a tuple of tuples is provided, each inner tuple specifies (pad_before, pad_after) for each dimension.

        The padded values are determined by the specified mode (default is "constant") and the constant
        value provided.

        Args:
            pad_width (int or tuple): Specifies how much padding to add on each dimension.
            mode (str, optional): Padding mode. Defaults to "constant".
            constant_values (int or float, optional): Fill value for constant padding. Defaults to 0.

        Returns:
            Tensor: The padded tensor.

        Example:
            >>> tensor = Tensor([[1, 2], [3, 4]])
            >>> padded_tensor = tensor.pad(pad_width=1, mode="constant", constant_values=0)
            >>> print(padded_tensor)
            Tensor([[0, 0, 0, 0],
                    [0, 1, 2, 0],
                    [0, 3, 4, 0],
                    [0, 0, 0, 0]])
        """
        return Pad.apply(
            self,
            pad_width=pad_width,
            mode=mode,
            constant_values=constant_values,
        )

    def forward(self, data: Any) -> None:
        """
        Placeholder for forward logic if needed. Currently unused.
        """
        pass

    def backward(
        self, grad: Optional[Union["Tensor", np.ndarray, float, int]] = None
    ) -> None:
        """
        Compute gradients for all upstream nodes in the graph via backpropagation.

        1. If `grad` is None, we treat the gradient as ones (like d(self)/d(self) = 1).
        2. We then do a post-order traversal of the graph: gather all nodes that lead to this tensor
        and store them in a topologically sorted list.
        3. Finally, we go through that list in reverse order to apply each node's .backward(...),
        passing gradients back to the node's inputs.

        As a side effect, each ancestor Tensor accumulates its .grad field.

        Args:
            grad (Optional[Union[Tensor, np.ndarray, float, int]]): The gradient w.r.t. this tensor's output.
        """
        if not self.requires_grad:
            # If this tensor doesn't require grad, there's nothing to do.
            return

        # If caller didn't supply a gradient, we assume d(self)/d(self) = 1
        if grad is None:
            grad = Tensor(np.ones_like(self.data))

        self.grad = grad  # store as np array directly

        # Build computational graph in reverse order
        topological_sorted_tensors = []
        visited = set()
        stack = [(self, False)]  # node, has_visited_children flag

        # Post-order traversal to figure out the order of the backprop
        while stack:
            node, has_visited_children = stack.pop()
            if node not in visited:
                if not has_visited_children:
                    # first time we see this node, push it again with has_visited_children=True
                    stack.append((node, True))
                    # then push its parents
                    if node.creator is not None:
                        for p in node.creator.tensors:
                            if p.requires_grad:
                                stack.append((p, False))
                else:
                    # Now we've visited node's inputs (second time seeing node),
                    # so node is in correct post-order
                    visited.add(node)
                    topological_sorted_tensors.append(node)

        # Backward pass
        # Traverse the sorted list in reverse to propagate gradients
        for tensor in reversed(topological_sorted_tensors):
            if tensor.creator is not None:
                # Call function's backward to get gradients w.r.t. to each input
                grads = tensor.creator.backward(tensor.grad)
                if not isinstance(grads, tuple):
                    grads = (grads,)  # handle single input
                # Accumulate grads into the input tensors
                for input_tensor, g in zip(tensor.creator.tensors, grads):
                    if (
                        input_tensor is not None
                        and input_tensor.requires_grad
                        and g is not None
                    ):
                        input_tensor._accumulate_grad(g)
                # Free references to reduce memory usage
                tensor.creator.tensors = None

        # Clear references to break the graph after everything is done
        # to reduce memory usage
        for node in topological_sorted_tensors:
            node.creator = None

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Return the shape of the underlying NumPy data.

        Returns:
            Tuple[int, ...]: The shape of this tensor.
        """
        if isinstance(self.data, (int, float)) or not hasattr(self.data, "shape"):
            return ()
        return self.data.shape

    ########### Movement ops ###########
    def reshape(self, *shape: int) -> "Tensor":
        """
        Return a new tensor with the same data but a different shape.
        It is functionally similar to numpy's reshape.

        Args:
            *shape (int): The desired new shape.

        Returns:
            Tensor: A reshaped tensor.
        """
        return Reshape.apply(self, shape=shape)

    def expand(self, *shape: Union[int, Sequence[int]]) -> "Tensor":
        """
        Broadcast the tensor to a new shape without copying data.

        This operation broadcasts the input tensor to a new shape. The forward pass creates a new array
        with the specified shape (via broadcasting), and the backward pass reduces the gradient back to the
        shape of the original tensor.

        Args:
            *shape (Union[int, Sequence[int]]): The target shape, which can be specified as multiple
                int arguments or as a single tuple/list.

        Returns:
            Tensor: A new tensor broadcast to the specified shape.

        Example:
            >>> tensor = Tensor([1, 2, 3])
            >>> expanded_tensor = tensor.expand(3, 3)
            >>> print(expanded_tensor)
            Tensor([[1, 2, 3],
                    [1, 2, 3],
                    [1, 2, 3]])
        """
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Expand.apply(self, shape=shape)

    def permute(self, *dims: int) -> "Tensor":
        """
        Reorder (permute) the dimensions of this tensor.
        Examples:
            >>> import numpy as np
            >>> from your_module import Tensor, Permute
            >>> t = Tensor(np.array([[1, 2], [3, 4]]))
            >>> op = Permute()
            >>> result = op.forward(t.data, dims=[1, 0])
            >>> print(result)
            [[1, 3],
            [2, 4]]

        Args:
            *dims (int): A sequence of dimension indices indicating the new order.

        Returns:
            Tensor: A new tensor with permuted dimensions.

        Example:
            >>> tensor = Tensor([[1, 2], [3, 4]])
            >>> permuted_tensor = tensor.permute(1, 0)
            >>> print(permuted_tensor)
            Tensor([[1, 3],
                    [2, 4]])
        """
        return Permute.apply(self, dims=dims)

    def transpose(self, dim0: int = 0, dim1: int = 1) -> "Tensor":
        """
        Swap two dimensions of this tensor.

        This operation swaps the positions of two specified dimensions of the input tensor.
        The backward pass applies the same transposition to the gradient, restoring the original dimension order.

        Args:
            dim0 (int, optional): First dimension to swap. Defaults to 0.
            dim1 (int, optional): Second dimension to swap. Defaults to 1.

        Returns:
            Tensor: A new tensor with the specified dimensions swapped.
        """
        return Transpose.apply(self, dim0=dim0, dim1=dim1)

    def strided_windows(self, kernel_size: int, stride: int) -> "Tensor":
        r"""
        Extract sliding windows of size `kernel_size` with stride `stride`.


        This operation generates overlapping windows from the input tensor using the specified kernel size and stride.
        The output shape is given by:

            $$
            (H_{out}, W_{out}, batch\_size, channels, kernel\_size, kernel\_size)
            $$

        where

            $$
            \begin{align}
            H_{out} = \frac{height - kernel\_size}{stride} + 1 \\
            W_{out} = \frac{width - kernel\_size}{stride} + 1
            \end{align}
            $$

        Examples:
            >>> import numpy as np
            >>> x = Tensor(np.random.rand(2, 3, 10, 10))  # shape: (batch, channels, height, width)
            >>> op = StridedWindows()
            >>> windows = x.strided_windows(x, kernel_size=3, stride=1)
            >>> print(windows.shape)
            (8, 8, 2, 3, 3, 3)

        Args:
            kernel_size (int): The size of each window.
            stride (int): The stride between windows.

        Returns:
            Tensor: A tensor representing the strided windows.
        """
        return StridedWindows.apply(self, kernel_size=kernel_size, stride=stride)

    def roll(self, shifts: int, dims: int) -> "Tensor":
        """
        Roll tensor elements along a given dimension.
        This operation shifts the elements of the input tensor along the given dimension by the
        specified number of positions. Elements that roll beyond the last position reappear at the beginning.

        Args:
            shifts (int): Number of places by which to shift.
            dims (int): Dimension along which to roll.

        Returns:
            Tensor: The rolled tensor.

        Example:
            >>> tensor = Tensor([1, 2, 3, 4, 5])
            >>> rolled_tensor = tensor.roll(shifts=2, dims=0)
            >>> print(rolled_tensor)
            Tensor([4, 5, 1, 2, 3])
        """
        return Roll.apply(self, shifts=shifts, dims=dims)

    def detach(self) -> "Tensor":
        """
        Detach this tensor from the computational graph, returning a new tensor with the same data
        but no gradient.

        Returns:
            Tensor: A new tensor that does not track gradients.
        """
        return Tensor(self.data, requires_grad=False)

    @property
    def ndim(self) -> int:
        """
        Return the number of dimensions of this tensor.

        Returns:
            int: The number of dimensions.
        """
        return len(self.data.shape)

    @property
    def T(self) -> "Tensor":
        """
        Convenience property to transpose a 2D tensor.
        For higher dimensions, use transpose() with explicit dims.

        Returns:
            Tensor: Transposed tensor.

        Raises:
            ValueError: If the tensor is not 2D.
        """
        if len(self.data.shape) != 2:
            raise ValueError(
                "T property is only defined for 2D tensors. Use transpose() for higher dimensions."
            )
        return self.transpose(1, 0)

    def _accumulate_grad(
        self, grad: Union["Tensor", np.ndarray], idx: Optional[Any] = None
    ) -> None:
        """
        Accumulate the gradient in this tensor.
        Lazily initialize and accumulate gradients

        If a gradient doesn't exist, create one. Otherwise, add to the existing gradient.

        Args:
            grad (Union[Tensor, np.ndarray]): The gradient to accumulate.
            idx (Optional[Any]): An index for partial accumulation (e.g., for SetItem).
        """
        if grad is None:
            return
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
                self._grad.data += grad.data

    # Operator wrappers
    def __radd__(self, other: Union["Tensor", float, int]) -> "Tensor":
        return self + other

    def __rmul__(self, other: Union["Tensor", float, int, np.ndarray]) -> "Tensor":
        return self * other

    def __sub__(self, other: Union["Tensor", float, int]) -> "Tensor":
        return self + (-other)

    def __rsub__(self, other: Union["Tensor", float, int]) -> "Tensor":
        return other + (-self)

    def __truediv__(self, other: Union["Tensor", float, int]) -> "Tensor":
        return self * other**-1

    def __neg__(self) -> "Tensor":
        return self * -1

    def __repr__(self) -> str:
        """
        Return a string representation of the tensor, showing its data and gradient.
        """
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __lt__(self, other: Union["Tensor", float, int]) -> np.ndarray:
        return self.data < other

    def __le__(self, other: Union["Tensor", float, int]) -> np.ndarray:
        return self.data <= other

    def __gt__(self, other: Union["Tensor", float, int]) -> np.ndarray:
        return self.data > other

    def __ge__(self, other: Union["Tensor", float, int]) -> np.ndarray:
        return self.data >= other

    def __eq__(self, other: Union["Tensor", float, int]) -> np.ndarray:
        return self.data == other

    def __hash__(self) -> int:
        # Hash is based on the id of the tensor object
        return id(self)


"""
Binary Ops
"""


class Add(Function):
    """Element-wise addition of two tensors.
    See :func:`autograd.tensor.Tensor.__add__` function
    """

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute the element-wise sum of two tensors.

        Args:
            x (np.ndarray): The first input tensor.
            y (np.ndarray): The second input tensor.

        Returns:
            np.ndarray: The element-wise sum of ``x`` and ``y``.
        """
        self.x_shape = x.shape  # for backward unbroadcast
        self.y_shape = y.shape  # for backward unbroadcast
        return x + y

    def backward(
        self, grad: "Tensor"
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Compute the gradient for the addition operation.

        Since addition is linear, the gradient with respect to both inputs is the same as the
        incoming gradient.

        Args:
            grad (Tensor): The gradient of the loss with respect to the output.

        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: The gradients with respect to
            ``x`` and ``y``.
        """
        grad_x = grad.data if self.tensors[0].requires_grad else None
        grad_y = grad.data if self.tensors[1].requires_grad else None

        # 1) If x required grad, we have to sum out the dims that were broadcast.
        if grad_x is not None:
            grad_x = Function.unbroadcast(grad_x, self.x_shape)

        # 2) If y required grad, do the same
        if grad_y is not None:
            grad_y = Function.unbroadcast(grad_y, self.y_shape)

        return grad_x, grad_y


class Mul(Function):
    """Element-wise multiplication of two tensors.
    See :func:`autograd.tensor.Tensor.__mul__` function
    """

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute the element-wise product of two tensors.

        Args:
            x (np.ndarray): The first input tensor.
            y (np.ndarray): The second input tensor.

        Returns:
            np.ndarray: The element-wise product of ``x`` and ``y``.
        """
        self.x_shape = x.shape  # for backward unbroadcast
        self.y_shape = y.shape  # for backward unbroadcast
        return x * y

    def backward(
        self, grad: "Tensor"
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        r"""Compute the gradient for the multiplication operation.

        The gradients are computed as:

            $$
            \begin{align}
            \frac{\partial z}{\partial x} = y \\
            \frac{\partial z}{\partial y} = x
            \end{align}
            $$

        and then multiplied by the incoming gradient.

        Args:
            grad (Tensor): The gradient of the loss with respect to the output.

        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: The gradients with respect to
            ``x`` and ``y``.
        """
        grad_x = (
            grad.data * self.tensors[1].data if self.tensors[0].requires_grad else None
        )
        grad_y = (
            grad.data * self.tensors[0].data if self.tensors[1].requires_grad else None
        )

        # 1) If x required grad, we have to sum out the dims that were broadcast.
        if grad_x is not None:
            grad_x = Function.unbroadcast(grad_x, self.x_shape)

        # 2) If y required grad, do the same
        if grad_y is not None:
            grad_y = Function.unbroadcast(grad_y, self.y_shape)
        return grad_x, grad_y


class Pow(Function):
    """Element-wise power operation.
    See :func:`autograd.tensor.Tensor.__pow__` function
    """

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute the element-wise power operation.

        Args:
            x (np.ndarray): The base tensor.
            y (np.ndarray): The exponent tensor.

        Returns:
            np.ndarray: The result of raising ``x`` to the power ``y``.
        """
        self.x_shape = x.shape  # for backward unbroadcast
        self.y_shape = y.shape  # for backward unbroadcast
        return x**y

    def backward(
        self, grad: "Tensor"
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        r"""Compute the gradient for the power operation.

        The derivatives are given by:

        $$
        \begin{align}
        \frac{\partial (x^y)}{\partial x} = y \cdot x^{y-1} \\
        \frac{\partial (x^y)}{\partial y} = x^y \cdot \ln(x)
        \end{align}
        $$

        These derivatives are multiplied by the incoming gradient.

        Args:
            grad (Tensor): The gradient of the loss with respect to the output.

        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: The gradients with respect to
            ``x`` and ``y``.
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

        # 1) If x required grad, we have to sum out the dims that were broadcast.
        if grad_x is not None:
            grad_x = Function.unbroadcast(grad_x, self.x_shape)

        # 2) If y required grad, do the same
        if grad_y is not None:
            grad_y = Function.unbroadcast(grad_y, self.y_shape)

        return grad_x, grad_y


class Matmul(Function):
    """Matrix multiplication of two tensors.
    See :func:`autograd.tensor.Tensor.__matmul__` function
    """

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute the matrix multiplication of two tensors.

        The operation uses ``np.matmul``, which handles broadcasting and batching.

        Args:
            x (np.ndarray): The first tensor.
            y (np.ndarray): The second tensor.

        Returns:
            np.ndarray: The result of matrix multiplying ``x`` and ``y``.
        """
        # Save references so backward() can know which Tensors to differentiate
        self.x_shape = x.shape
        self.y_shape = y.shape
        out = np.matmul(x, y)
        return out

    def backward(
        self, grad: "Tensor"
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        r"""Compute the gradient for the matrix multiplication operation.

        Args:
            grad (Tensor): The gradient of the loss with respect to the output.

        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: The gradients with respect to
            ``x`` and ``y``.

        For matrix multiplication:

            $z = x \cdot y$

        the gradients are computed as:
            $$ \text{grad}_x = \text{grad} \cdot y^T $$
            $$ \text{grad}_y = x^T \cdot \text{grad} $$

        Special handling is provided for the vector @ vector case and for batched multiplications.

        """
        x = self.tensors[0]
        y = self.tensors[1]
        grad_x = grad_y = None

        # Handle vector @ vector case separately (1D @ 1D)
        if x.data.ndim == 1 and y.data.ndim == 1:
            if x.requires_grad:
                grad_x = grad.data * y.data
            if y.requires_grad:
                grad_y = grad.data * x.data
            return grad_x, grad_y

        """
        Otherwise do "batched" matmul logic
        # If one operand doesn't have batch dims, we sum out the batch dims from the result.

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

        if x.requires_grad:
            # Transpose y on the last two dims (y^T)
            # np.swapaxes(y, -1, -2) is effectively y^T for each batch.
            y_t = np.swapaxes(y.data, -1, -2)
            grad_x = np.matmul(grad.data, y_t)
            # shape of grad_x should match x.data.shape

        if y.requires_grad:
            # Transpose x on the last two dims (x^T)
            x_t = np.swapaxes(x.data, -1, -2)
            raw_grad_y = np.matmul(x_t, grad.data)
            # Now if y was 2D => shape (m, p), but raw_grad_y might be (B,m,p). We sum over batch dims.
            # Let's figure out how many leading dims y has (besides the last 2).
            if y.data.ndim == 2 and raw_grad_y.ndim > 2:
                # sum over all batch/time dims => axis=0..(raw_grad_y.ndim-2)
                # e.g. raw_grad_y is (B, m, p), we want (m, p)
                axes_to_sum = tuple(range(raw_grad_y.ndim - 2))
                grad_y = np.sum(raw_grad_y, axis=axes_to_sum)
            else:
                # If y had a batch dimension as well, raw_grad_y already has the right shape
                grad_y = raw_grad_y

        return grad_x, grad_y


class IAdd(Function):
    """In-place addition of two tensors.
    See :func:`autograd.tensor.Tensor.__iadd__` function
    """

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Perform in-place addition on the input tensor.

        Args:
            x (np.ndarray): The tensor to be updated.
            y (np.ndarray): The tensor to add.

        Returns:
            np.ndarray: The updated tensor ``x`` after addition.
        """
        # Update data in-place
        x += y
        return x

    def backward(self, grad: "Tensor") -> Tuple[np.ndarray, np.ndarray]:
        """Compute the gradient for the in-place addition operation.

        Both inputs receive the same gradient as in the standard addition.

        Args:
            grad (Tensor): The gradient of the loss with respect to the output.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The gradients with respect to ``x`` and ``y``.
        """
        return grad.data, grad.data


class GetItem(Function):
    """Retrieve an item from a tensor using numpy-style indexing.
    See :func:`autograd.tensor.Tensor.__getitem__` function
    """

    def forward(self, x: np.ndarray, idx: Any) -> np.ndarray:
        """Return a subset of the tensor based on the specified index.

        Args:
            x (np.ndarray): The input tensor.
            idx (Any): The index used to retrieve a subset of ``x`` (e.g., slices, integers).

        Returns:
            np.ndarray: The indexed subset of the tensor.
        """
        self.idx = idx
        return x[idx]

    def backward(self, grad: "Tensor") -> np.ndarray:
        """Propagate gradients through the indexing operation.

        A zero tensor of the original shape is created and the gradient is placed
        in the correct location corresponding to the index.

        Args:
            grad (Tensor): The gradient of the loss with respect to the output.

        Returns:
            np.ndarray: The gradient with respect to the input tensor.
        """
        grad = grad.data if isinstance(grad, Tensor) else grad

        # Create a zero tensor of the original shape
        out = np.zeros_like(self.tensors[0].data)
        # Place the gradient in the correct location
        out[self.idx] = grad
        return out


class SetItem(Function):
    """In-place assignment to a tensor using numpy-style indexing.
    See :func:`autograd.tensor.Tensor.__setitem__` function
    """

    def forward(self, x: np.ndarray, value: np.ndarray, idx: Any) -> np.ndarray:
        """Perform in-place assignment on the input tensor.

        Args:
            x (np.ndarray): The input tensor.
            idx (Any): The indices at which to assign the new value.
            value (np.ndarray): The value to assign.

        Returns:
            np.ndarray: The tensor after assignment.
        """
        # Extract numpy array from value
        val_data = value.data if isinstance(value, Tensor) else value
        x[idx] = val_data
        self.idx = idx
        return x

    def backward(self, grad: "Tensor") -> np.ndarray:
        """Compute the gradient for the in-place assignment operation.

        The gradient is extracted only from the region specified by the index.

        Args:
            grad (Tensor): The gradient of the loss with respect to the output.

        Returns:
            np.ndarray: The gradient with respect to the input tensor.
        """
        return grad.data[self.idx]


class Sqrt(Function):
    """Compute the element-wise square root of a tensor.
    See :func:`autograd.tensor.Tensor.sqrt` function
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the square root of each element in the input tensor.

        Args:
            x (np.ndarray): The input tensor.

        Returns:
            np.ndarray: The element-wise square root of ``x``.
        """
        # Store input for backward pass
        self.x = x
        return np.sqrt(x)

    def backward(self, grad: "Tensor") -> np.ndarray:
        r"""Compute the gradient for the square root operation.

        The derivative of the square root is given by:

            $$
            \frac{d}{dx}\sqrt{x} = \frac{1}{2\sqrt{x}}
            $$

        The gradient is computed by multiplying the incoming gradient ``grad`` by this derivative.

        Args:
            grad (Tensor): The gradient of the loss with respect to the output.

        Returns:
            np.ndarray: The gradient of the loss with respect to the input tensor.
        """
        return grad.data * 0.5 / np.sqrt(self.x)


"""
Reduction Ops
"""


class Sum(Function):
    """Compute the sum of tensor elements.

    See :func:`autograd.tensor.Tensor.sum` function
    """

    def forward(
        self,
        x: np.ndarray,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> np.ndarray:
        """Compute the forward pass for the sum operation.

        Args:
            x (np.ndarray): Input tensor.
            axis (int or tuple of ints, optional): Axis or axes along which the sum is performed.
                If None, the sum of all elements is computed.
            keepdims (bool, optional): If True, the reduced axes are kept in the output as dimensions with size one.

        Returns:
            np.ndarray: The sum of the tensor elements.
        """
        # Handle scalar case
        if not hasattr(x, "ndim") or x.ndim == 0:
            return x
        # Normalize axis
        self.axis = (axis,) if isinstance(axis, int) else axis
        self.keepdims = keepdims
        self.x_shape = x.shape
        return np.sum(x, axis=self.axis, keepdims=self.keepdims)

    def backward(self, grad: "Tensor") -> np.ndarray:
        """Compute the backward pass for the sum operation.

        This method computes the gradient of the sum operation by broadcasting the gradient to the shape
        of the input tensor.

        Args:
            grad (Tensor): The gradient of the loss with respect to the output of the sum operation.

        Returns:
            np.ndarray: The gradient of the loss with respect to the input tensor.
        """
        # Use expand to handle gradient broadcasting
        if grad is None:
            return None

        grad_arr = grad.data
        # Turn axis into a tuple
        if isinstance(self.axis, int):
            reduce_axes = (self.axis,)
        else:
            reduce_axes = self.axis

        # If we never specified an axis, then reduce_axes=None means a global sum
        if reduce_axes is None:
            # Summed over all dims, so grad is scalar, shape=()
            # Just broadcast to original shape:
            return np.broadcast_to(grad_arr, self.x_shape).copy()

        if not self.keepdims:
            # Re-insert those axes as size=1 so that broadcasting works
            # Sort them or reverse them so that inserting doesn’t shift the later dims incorrectly
            for ax in sorted(reduce_axes):
                grad_arr = np.expand_dims(grad_arr, ax)

        # Now grad has shape (2,8,1) if we did sum over axis=2
        # broadcast to (2,8,5)
        return np.broadcast_to(grad_arr, self.x_shape).copy()


class Max(Function):
    """
    See :func:`autograd.tensor.Tensor.max` function
    """

    def forward(
        self,
        x: np.ndarray,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> np.ndarray:
        """Compute the maximum of tensor elements.
        Args:
            x (np.ndarray): Input tensor.
            axis (int or tuple of ints, optional): Axis or axes along which the maximum is computed. If None (default), the maximum of all elements is computed.
            keepdims (bool, optional): If True, the reduced axes are kept in the output as dimensions with size one.

        Returns:
            np.ndarray: The maximum values computed along the specified axis.
        """
        axis = (axis,) if isinstance(axis, int) else axis
        self.axis = axis
        self.keepdims = keepdims
        return np.max(x, axis=axis, keepdims=keepdims)

    def backward(self, grad: "Tensor") -> np.ndarray:
        r"""Compute the gradient of the maximum operation.

        The backward pass for the maximum operation is computed using the chain rule:

            $$
            \frac{\partial \text{loss}}{\partial x} = \frac{\partial \text{loss}}{\partial \max(x)} \cdot \frac{\partial \max(x)}{\partial x}
            $$

        where

            $$
            \frac{\partial \max(x)}{\partial x} =
            \begin{cases}
            1, & \text{if } x = \max(x) \\
            0, & \text{otherwise}
            \end{cases}
            $$

        In cases where multiple elements are equal to the maximum, the gradient is distributed equally or
        assigned to the first occurrence along the specified axis.

        Args:
            grad (Tensor): The gradient of the loss with respect to the output of the maximum operation.

        Returns:
            np.ndarray: The gradient of the loss with respect to the input tensor.
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
    """Compute the element-wise maximum between two tensors.
    See :func:`autograd.tensor.Tensor.maximum` function
    """

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute the element-wise maximum of two tensors.

        Args:
            x (np.ndarray): First input tensor.
            y (np.ndarray): Second input tensor.

        Returns:
            np.ndarray: The element-wise maximum of the two input tensors.
        """
        self.x_shape = x.shape  # for backward unbroadcast
        self.y_shape = y.shape  # for backward unbroadcast
        out = np.maximum(x, y)
        self.out_data = out
        return out

    def backward(
        self, grad: "Tensor"
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        r"""Compute the gradient of the element-wise maximum operation.

        During the backward pass, the gradient is distributed to the inputs according to the rule:

            $$
            \frac{\partial \text{loss}}{\partial x} = \text{grad} \times
            \begin{cases}
            1, & \text{if } x > y \\
            0.5, & \text{if } x = y \\
            0, & \text{otherwise}
            \end{cases}
            $$

        and similarly for $y$.

        Args:
            grad (Tensor): The gradient of the loss with respect to the output of the maximum operation.

        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
                Gradients of the loss with respect to the input tensors x and y.
        """
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

        # 1) If x required grad, we have to sum out the dims that were broadcast.
        if grad_x is not None:
            grad_x = Function.unbroadcast(grad_x, self.x_shape)

        # 2) If y required grad, do the same
        if grad_y is not None:
            grad_y = Function.unbroadcast(grad_y, self.y_shape)

        return grad_x, grad_y


class Mean(Function):
    """Compute the mean of tensor elements.
    See :func:`autograd.tensor.Tensor.mean` function
    """

    def forward(
        self,
        x: np.ndarray,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> np.ndarray:
        """Compute the forward pass for the mean operation.

        Args:
            x (np.ndarray): Input tensor.
            axis (int or tuple of ints, optional): Axis or axes along which the mean is computed.
                If None, the mean of all elements is computed.
            keepdims (bool, optional): If True, the reduced axes are retained in the output as dimensions with size one.

        Returns:
            np.ndarray: The mean of the tensor elements.
        """
        # Normalize axis to a tuple
        axis = (axis,) if isinstance(axis, int) else axis
        self.axis = axis
        self.keepdims = keepdims
        return np.mean(x, axis=axis, keepdims=keepdims)

    def backward(self, grad: "Tensor") -> np.ndarray:
        r"""Compute the gradient of the mean operation.

        The gradient is computed by broadcasting the gradient to the shape of the input tensor and scaling it by
        the number of elements that were averaged:

            $$
            \frac{\partial \text{loss}}{\partial x} = \frac{\text{grad}}{N}
            $$

        where N is the number of elements over which the mean was computed.

        Args:
            grad (Tensor): The gradient of the loss with respect to the output of the mean operation.

        Returns:
            np.ndarray: The gradient of the loss with respect to the input tensor.
        """
        # Use expand for gradient broadcasting
        grad_expanded = grad.expand(
            self.tensors[0].shape if self.keepdims else self.tensors[0].shape
        )
        grad_arr = grad_expanded.data
        # Scale gradient by number of elements
        if self.axis is not None:
            num_elements = np.prod([self.tensors[0].shape[ax] for ax in self.axis])
        else:
            num_elements = np.prod(self.tensors[0].shape)
        return grad_arr / num_elements


class Gather(Function):
    """Gather operation for 2D tensors along axis 0 using integer indices.
    See :func:`autograd.tensor.Tensor.gather` function
    """

    def forward(self, x: np.ndarray, index: np.ndarray) -> np.ndarray:
        """Perform the forward pass of the gather operation.

        Args:
            x (np.ndarray): The input 2D tensor.
            index (np.ndarray): An array of integer indices specifying the rows to gather.

        Returns:
            np.ndarray: A tensor containing the gathered rows.
        """
        out = x[index, :]
        self.x = x
        self.index = index
        return out

    def backward(self, grad: "Tensor") -> Tuple[np.ndarray, None]:
        """Perform the backward pass of the gather operation.

        The backward pass accumulates the gradients from the output back into the corresponding
        rows of the input tensor using numpy's in-place addition.

        Args:
            grad (Tensor): The gradient of the loss with respect to the output.

        Returns:
            Tuple[np.ndarray, None]: A tuple where the first element is the gradient with respect to
            the input tensor, and the second element is None (since indices are not differentiable).
        """
        dx = np.zeros_like(self.x)
        flat_indices = self.index.ravel()
        flat_grads = grad.data.reshape(-1, dx.shape[1])
        np.add.at(dx, flat_indices, flat_grads)
        return dx, None


"""
Movement Ops
"""


class View(Function):
    """View the tensor with a new shape without copying data.
    See :func:`autograd.tensor.Tensor.view` function
    """

    def forward(
        self, x: np.ndarray, new_shape: Union[Tuple[int, ...], List[int]] = (1,)
    ) -> np.ndarray:
        """Reshape the input tensor to a new view with the specified shape.

        Args:
            x (np.ndarray): The input tensor.
            new_shape (Union[Tuple[int, ...], List[int]], optional): The desired new shape.
                If a -1 is present, that dimension is inferred from the size of the input tensor.
                Defaults to (1,).

        Returns:
            np.ndarray: A view of the tensor with the specified new shape.

        Raises:
            ValueError: If more than one -1 is specified or if the new shape is incompatible
                with the total number of elements in x.
        """
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
                np.array(
                    [d for i, d in enumerate(new_shape) if i != neg_idx and d != -1]
                )
            )
            # Compute the missing dimension
            inferred_size = int(x.size // known_size)
            # Replace -1 with inferred size
            new_shape = tuple(inferred_size if d == -1 else d for d in new_shape)

        if x.size != np.prod(np.array(new_shape)):
            raise ValueError(
                f"Size of new view must match size of original tensor: {x.size} != {np.prod(new_shape)}"
            )
        # Store original shape for backward
        self.original_shape = x.shape

        return np.reshape(x, new_shape)

    def backward(self, grad: Optional["Tensor"]) -> Optional[np.ndarray]:
        """Reshape the gradient to match the original tensor shape.

        Args:
            grad (Tensor, optional): The gradient of the loss with respect to the output.

        Returns:
            Optional[np.ndarray]: The gradient reshaped to the original tensor shape, or None if grad is None.
        """
        return grad.reshape(self.original_shape).data if grad is not None else None


class Expand(Function):
    """Expand the tensor to a given shape without copying data (broadcasting).
    See :func:`autograd.tensor.Tensor.expand` function
    """

    def forward(
        self, x: np.ndarray, shape: Union[Tuple[int, ...], List[int]] = (1,)
    ) -> np.ndarray:
        """Broadcast the input tensor to the specified shape.

        Args:
            x (np.ndarray): The input tensor.
            shape (Union[Tuple[int, ...], List[int]], optional): The target shape for broadcasting.
                Defaults to (1,).

        Returns:
            np.ndarray: A new tensor broadcast to the specified shape.
        """
        self.original_shape = x.shape
        expanded = np.broadcast_to(x, shape)
        return expanded

    def backward(self, grad: "Tensor") -> np.ndarray:
        """Compute the gradient of the expand operation.

        The gradient is reduced by summing over the broadcast dimensions so that its shape matches
        the original input tensor.

        Args:
            grad (Tensor): The gradient of the loss with respect to the output.

        Returns:
            np.ndarray: The gradient of the loss with respect to the input tensor.
        """
        grad_arr = grad.data
        # Handle extra leading dimensions
        if len(grad_arr.shape) > len(self.original_shape):
            reduce_dims = list(range(len(grad_arr.shape) - len(self.original_shape)))
        else:
            reduce_dims = []

        # Handle broadcasting dimensions
        for i, (self_dim, grad_dim) in enumerate(
            zip(
                self.original_shape[::-1],
                grad_arr.shape[-len(self.original_shape) :][::-1],
            )
        ):
            if self_dim == 1 and grad_dim != 1:
                reduce_dims.append(len(grad_arr.shape) - 1 - i)

        # Sum across all reduction dimensions
        if reduce_dims:
            grad_arr = np.sum(grad_arr, axis=tuple(reduce_dims), keepdims=True)

        # Ensure final shape matches original
        if grad_arr.shape != self.original_shape:
            grad_arr = grad_arr.reshape(self.original_shape)

        return grad_arr


class Reshape(Function):
    """Reshape the tensor to a new shape without changing its data content.
    See :func:`autograd.tensor.Tensor.reshape` function
    """

    def forward(
        self, x: np.ndarray, shape: Union[Tuple[int, ...], List[int]] = (1,)
    ) -> np.ndarray:
        """Reshape the input tensor to the specified new shape.

        Args:
            x (np.ndarray): The input tensor.
            shape (Union[Tuple[int, ...], List[int]], optional): The new shape for the tensor.
                If a nested tuple or list is provided, it will be flattened. Defaults to (1,).

        Returns:
            np.ndarray: The reshaped tensor.
        """
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        self.original_shape = x.shape
        return np.reshape(x, shape)

    def backward(self, grad: Optional["Tensor"]) -> Optional[np.ndarray]:
        """Reshape the gradient to match the original tensor shape.

        Args:
            grad (Tensor, optional): The gradient of the loss with respect to the reshaped output.

        Returns:
            Optional[np.ndarray]: The gradient reshaped to the original tensor shape, or None if grad is None.
        """
        return grad.data.reshape(self.original_shape) if grad is not None else None


class Transpose(Function):
    """Transpose operation for swapping any two dimensions of a tensor.
    See :func:`autograd.tensor.Tensor.transpose` function
    """

    def _get_transpose_axes(
        self, x: np.ndarray, dim0: int, dim1: int
    ) -> Tuple[int, ...]:
        """Compute the axes order for transposing the tensor by swapping two dimensions.

        Args:
            x (np.ndarray): The input tensor.
            dim0 (int): The first dimension to swap.
            dim1 (int): The second dimension to swap.

        Returns:
            Tuple[int, ...]: A tuple representing the new order of axes after swapping.
        """
        axes = list(range(x.ndim))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        return tuple(axes)

    def forward(self, x: np.ndarray, dim0: int = 0, dim1: int = 1) -> np.ndarray:
        """Transpose the input tensor by swapping two specified dimensions.

        Args:
            x (np.ndarray): The input tensor.
            dim0 (int, optional): The first dimension to swap. Defaults to 0.
            dim1 (int, optional): The second dimension to swap. Defaults to 1.

        Returns:
            np.ndarray: The transposed tensor.

        Raises:
            ValueError: If the specified dimensions are out of range for the input tensor.
        """
        ndim = x.ndim
        if not (0 <= dim0 < ndim and 0 <= dim1 < ndim):
            raise ValueError(
                f"Dimensions out of range for tensor with {ndim} dimensions"
            )

        axes = self._get_transpose_axes(x, dim0, dim1)
        self.dim0 = dim0
        self.dim1 = dim1
        return np.transpose(x, axes)

    def backward(self, grad: "Tensor") -> np.ndarray:
        """Transpose the gradient tensor to match the original input tensor's dimension order.

        Args:
            grad (Tensor): The gradient of the loss with respect to the output tensor.

        Returns:
            np.ndarray: The gradient with dimensions swapped back to the original order.
        """
        transposed_grad = np.transpose(
            grad.data,
            self._get_transpose_axes(self.tensors[0].data, self.dim0, self.dim1),
        )
        return transposed_grad


class Pad(Function):
    """Pad the tensor with a specified padding.
    See :func:`autograd.tensor.Tensor.pad` function
    """

    def forward(
        self,
        x: np.ndarray,
        pad_width: Union[
            int,
            Tuple[int, int],
            Tuple[int, int, int, int],
            Tuple[Tuple[int, int], ...],
        ],
        mode: str = "constant",
        constant_values: Union[int, float] = 0,
    ) -> np.ndarray:
        """Pad the input tensor according to the specified pad width and mode.

        Args:
            x (np.ndarray): The input tensor.
            pad_width (int or tuple): Padding specification. See class docstring for details.
            mode (str, optional): Padding mode. Defaults to "constant".
            constant_values (int or float, optional): Value for constant padding. Defaults to 0.

        Returns:
            np.ndarray: The padded tensor.
        """
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

    def backward(self, grad: "Tensor") -> np.ndarray:
        """Extract the unpadded region from the gradient.

        This method removes the padding from the gradient tensor, returning only the region
        corresponding to the original input tensor.

        Args:
            grad (Tensor): The gradient of the loss with respect to the padded output.

        Returns:
            np.ndarray: The gradient corresponding to the unpadded input.
        """
        # Extract the unpadded region
        slices = tuple(
            slice(p[0], s - p[1]) for s, p in zip(self.out_data.shape, self.pad_width)
        )
        return grad.data[slices]


class Cat(Function):
    """Concatenate a sequence of tensors along a specified axis.
    See :func:`autograd.tensor.Tensor.cat` function
    """

    def forward(self, *tensors: "Tensor", axis: int = 0) -> np.ndarray:
        """Concatenate input tensors along the specified axis.

        Args:
            *tensors (Tensor): A sequence of tensors to concatenate.
            axis (int, optional): The axis along which to concatenate. Defaults to 0.

        Returns:
            np.ndarray: The concatenated tensor.
        """
        self.axis = axis
        self.original_shapes = [t.data.shape for t in tensors]
        return np.concatenate([t.data for t in tensors], axis=axis)

    def backward(self, grad: "Tensor") -> Tuple[Optional[np.ndarray], ...]:
        """Split the gradient among the concatenated tensors.

        The gradient is divided along the concatenation axis based on the original shapes of the input tensors.

        Args:
            grad (Tensor): The gradient of the loss with respect to the concatenated output.

        Returns:
            Tuple[Optional[np.ndarray], ...]: A tuple of gradients corresponding to each input tensor.
        """
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
    """Reorder the dimensions of a tensor.
    See :func:`autograd.tensor.Tensor.permute` function
    """

    def forward(self, x: np.ndarray, dims: Sequence[int]) -> np.ndarray:
        """Permute the dimensions of the input tensor.

        Args:
            x (np.ndarray): The input tensor.
            dims (Sequence[int]): The new order of dimensions. If a single element that is a tuple or list is provided,
                it will be unpacked.

        Returns:
            np.ndarray: The tensor with permuted dimensions.
        """
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = dims[0]

        self.dims = dims
        return np.transpose(x, dims)

    def backward(self, grad: "Tensor") -> np.ndarray:
        """Compute the gradient for the permutation operation.

        The gradient is transposed using the inverse permutation of the forward pass.

        Args:
            grad (Tensor): The gradient of the loss with respect to the permuted output.

        Returns:
            np.ndarray: The gradient with dimensions restored to their original order.
        """
        inv_dims = [self.dims.index(i) for i in range(len(self.dims))]
        return np.transpose(grad.data, inv_dims)


class Stack(Function):
    """Stack a sequence of tensors along a new axis.
    See :func:`autograd.tensor.Tensor.stack` function
    """

    def forward(self, *tensors: "Tensor", axis: int = 0) -> np.ndarray:
        """Stack input tensors along a new axis.

        This method expands the dimensions of each input tensor along the specified axis and concatenates them.

        Args:
            *tensors (Tensor): A sequence of tensors to be stacked.
            axis (int, optional): The axis along which to stack the tensors. Defaults to 0.

        Returns:
            np.ndarray: The stacked tensor.

        Raises:
            ValueError: If no tensors are provided.
        """
        if not tensors:
            raise ValueError("Need at least one tensor to stack")

        # Memory optimization: Use numpy.concatenate with expanded dimensions
        # instead of stack to avoid temporary list creation
        expanded_arrays = [np.expand_dims(t.data, axis=axis) for t in tensors]
        stacked_data = np.concatenate(expanded_arrays, axis=axis)
        self.axis = axis
        return stacked_data

    def backward(self, grad: "Tensor") -> Tuple[Optional[np.ndarray], ...]:
        """Split the gradient among the stacked tensors.

        The gradient is divided along the stacking axis and reshaped to match each input tensor's original shape.

        Args:
            grad (Tensor): The gradient of the loss with respect to the stacked tensor.

        Returns:
            Tuple[Optional[np.ndarray], ...]: A tuple of gradients corresponding to each input tensor.
        """
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
    """Create a strided windows view of a tensor.
    See :func:`autograd.tensor.Tensor.strided_windows` function
    """

    def forward(self, x: np.ndarray, kernel_size: int, stride: int) -> np.ndarray:
        r"""Create a strided windows view of the input tensor.

        Args:
            x (np.ndarray): The input tensor of shape (batch_size, channels, height, width).
            kernel_size (int): The size of each window.
            stride (int): The stride between windows.

        Returns:
            np.ndarray: A view of the tensor with shape

        $(H_{out}, W_{out}, batch\_size, channels, kernel\_size, kernel\_size)$,
        where $H_{out} = \frac{height - kernel\_size}{stride} + 1$ and
        $W_{out} = \frac{width - kernel\_size}{stride} + 1$.
        """
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
        # Directly produce (H_out, W_out, batch_size, channels, kernel_size, kernel_size)
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
        )

    def backward(self, grad: "Tensor") -> np.ndarray:
        """Reconstruct the gradient for the input tensor from the strided windows gradient.

        This method reshapes and transposes the gradient of the strided windows view back to the
        original input tensor shape by accumulating overlapping gradients.

        Args:
            grad (Tensor): The gradient of the loss with respect to the strided windows output.

        Returns:
            np.ndarray: The gradient of the loss with respect to the original input tensor.
        """
        grad_arr = grad.data.reshape(
            self.H_out,
            self.W_out,
            self.batch_size,
            self.channels,
            self.kernel_size,
            self.kernel_size,
        )
        grad_arr = grad_arr.transpose(2, 3, 0, 1, 4, 5)
        grad_padded = np.zeros(self.data_shape, dtype=grad_arr.dtype)

        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                grad_padded[
                    :,
                    :,
                    i : i + self.H_out * self.stride : self.stride,
                    j : j + self.W_out * self.stride : self.stride,
                ] += grad_arr[:, :, :, :, i, j]

        return grad_padded


class Roll(Function):
    """Roll tensor elements along a specified dimension.
    See :func:`autograd.tensor.Tensor.roll` function
    """

    def forward(
        self, x: np.ndarray, shifts: int, dims: Optional[int] = None
    ) -> np.ndarray:
        """Roll the elements of the input tensor.

        Args:
            x (np.ndarray): The input tensor.
            shifts (int): The number of positions to shift the elements.
            dims (int, optional): The axis along which to roll the elements. If None, the tensor is flattened before rolling.

        Returns:
            np.ndarray: The tensor with its elements rolled along the specified dimension.
        """
        self.shifts = shifts
        self.dims = dims
        self.input_shape = x.shape
        return np.roll(x, shift=shifts, axis=dims)

    def backward(self, grad: "Tensor") -> np.ndarray:
        """Perform the backward pass for the roll operation.

        The gradient is rolled in the opposite direction (by negating the shift) to reverse the forward roll.

        Args:
            grad (Tensor): The gradient of the loss with respect to the rolled output.

        Returns:
            np.ndarray: The gradient of the loss with respect to the input tensor.
        """
        grad_arr = grad.data
        # Handle scalar gradients by reshaping to original input shape
        if grad_arr.ndim == 0:
            grad_arr = np.full(self.input_shape, grad_arr)

        # Roll gradient in opposite direction by negating the shift
        return np.roll(grad_arr, shift=-self.shifts, axis=self.dims)
