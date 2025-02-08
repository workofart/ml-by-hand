"""
Initialization methods for weights of the neural network
"""

try:
    # drop-in replacement for numpy for GPU acceleration
    import cupy as np  # type: ignore

    _ = np.cuda.runtime.getDeviceCount()  # Check if a CUDA device is available
except Exception:
    import numpy as np

from autograd.tensor import Tensor


def xavier_uniform(tensor: Tensor):
    r"""
    Applies in-place Xavier Uniform Initialization to the given tensor.

    This method initializes the weights of a neural network using the Xavier (Glorot)
    uniform initialization technique, as described in this paper:
    https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

    The weight tensor is assumed to have the shape:
        (input_size, output_size, additional_dimensions...)

    The limits for the uniform distribution are computed using the number of input and output
    tensor counts of the given tensor, where the limit is given by:
    $$
    \text{limit} = \sqrt{\frac{6}{\text{\# of input tensor count} + \text{\# of output tensor count}}}
    $$

    Args:
        tensor (Tensor): The tensor to be initialized. Its underlying data should be a NumPy array.

    Returns:
        Tensor: The same tensor after in-place initialization.

    Examples:
        >>> import cupy as np
        >>> from autograd.tensor import Tensor
        >>> # Create an uninitialized tensor with shape (3, 4)
        >>> tensor = Tensor(np.empty((3, 4)))
        >>> # Initialize the tensor using Xavier Uniform initialization
        >>> tensor = xavier_uniform(tensor)
    """
    input_tensor_count, output_tensor_count = compute_in_out_tensor_count(tensor.data)
    limit = np.sqrt(6.0 / (input_tensor_count + output_tensor_count))

    tensor.data[...] = np.random.uniform(
        low=-limit, high=limit, size=tensor.data.shape
    ).astype(tensor.data.dtype)
    return tensor


def compute_in_out_tensor_count(tensor: Tensor):
    r"""
    Computes the number of input and output tensor counts for the given tensor.

    For convolution kernels:
      - tensor.shape[0] is assumed to represent the number of output channels.
      - tensor.shape[1] is assumed to represent the number of input channels.
      - The remaining dimensions correspond to the spatial kernel sizes (e.g., kernel height, kernel width).

    The counts are computed as follows:
    $$
    \begin{align}
    \text{\# of input tensor count} &= \text{tensor.shape}[0] \times \prod_{i=2}^{n} \text{tensor.shape}[i] \\
    \text{\# of output tensor count} &= \text{tensor.shape}[-1] \times \prod_{i=2}^{n} \text{tensor.shape}[i]
    \end{align}
    $$
    where \( n \) is the number of dimensions in the tensor.

    Args:
        tensor (Tensor): The tensor for which to compute the number of input and output tensor counts.
                         The tensor must have at least 2 dimensions.

    Returns:
        Tuple[int, int]: A tuple containing (input_tensor_count, output_tensor_count).

    Raises:
        ValueError: If the tensor has fewer than 2 dimensions.

    Examples:
        >>> import cupy as np
        >>> from autograd.tensor import Tensor
        >>> # Example for a fully-connected layer weight matrix with shape (fan_in, fan_out)
        >>> tensor_fc = Tensor(np.empty((5, 10)))
        >>> compute_in_out_tensor_count(tensor_fc.data)
        (5, 10)
        >>> # Example for a convolution kernel with shape
        >>> # (output_channels, input_channels, kernel_height, kernel_width)
        >>> tensor_conv = Tensor(np.empty((16, 3, 3, 3)))
        >>> # The receptive field size is 3*3 = 9
        >>> # So, input tensor count = 16 * 9 = 144, output tensor count = 3 * 9 = 27
        >>> compute_in_out_tensor_count(tensor_conv.data)
        (144, 27)
    """
    tensor_shape = tensor.shape
    if len(tensor_shape) < 2:
        raise ValueError("Tensor must have at least 2 dimensions")

    receptive_field_size = 1
    if len(tensor_shape) > 2:
        for s in tensor_shape[2:]:
            receptive_field_size *= s

    input_tensor_count = tensor_shape[0] * receptive_field_size
    output_tensor_count = tensor_shape[-1] * receptive_field_size
    return input_tensor_count, output_tensor_count
