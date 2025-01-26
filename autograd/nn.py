import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from .functional import relu, sigmoid, softmax, tanh
from .init import xavier_uniform
from .tensor import Tensor

logger = logging.getLogger(__name__)


class Module:
    def __init__(self, *args, **kwargs) -> None:
        self._parameters: Dict[str, Tensor] = {}
        self._modules: Dict[str, "Module"] = {}
        self._states: Dict[str, Any] = {}
        self._is_training: Optional[bool] = None

    def zero_grad(self) -> None:
        # Zero gradients for parameters in current module
        for p in self._parameters.values():
            p.grad = 0

        # Recursively zero gradients in submodules
        for module in self._modules.values():
            module.zero_grad()

    def forward(self, x: Any) -> Tensor:
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        """
        Sometimes people like to call model = Module() then call model(x)
        as a forward pass. So this is an alias.
        """
        return self.forward(*args, **kwargs)

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Hook into attribute setting so that submodules / parameters
        are automatically registered.
        """
        if name.startswith("_"):
            # Bypass custom logic for private/internal attributes
            super().__setattr__(name, value)
            return

        if isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, Tensor):
            self._parameters[name] = value
        else:
            self._states[name] = value
            super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Any:
        if name in self._modules:
            return self._modules[name]
        if name in self._states:
            return self._states[name]
        raise AttributeError(
            f"Module {self.__class__.__name__} has no attribute {name}"
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        """
        Return a flattened dictionary of all Tensors (trainable parameters)
        from this module and submodules:
          { 'weight': <Tensor>, 'submodule1.weight': <Tensor>, ... }
        """
        return {
            k: v
            for k, v in self._get_attr_nested("_parameters").items()
            if isinstance(v, Tensor)
        }

    @property
    def states(self) -> Dict[str, Any]:
        """
        Return a flattened dictionary of all non-trainable states/buffers
        from this module and submodules:
          { 'some_state': <np.ndarray>, 'submodule1.running_var': <np.ndarray>, ... }
        """
        return {
            k: v
            for k, v in self._get_attr_nested("_states").items()
            if isinstance(v, np.ndarray)
        }

    def state_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a dict of:
          {
            'parameters': { 'weight': np.array(...), 'bias': np.array(...), ... },
            'states': { 'stateful_states': np.array(...), ... }
          }
        This is a flat representation (like PyTorch's state_dict).
        """
        # Convert Tensors to raw np.array
        param_arrays = {k: v.data for k, v in self.parameters.items()}
        state_arrays = self.states  # already np.ndarray
        return {"parameters": param_arrays, "states": state_arrays}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Inverse of state_dict(). For example:
            loaded = load_model('checkpoint.json', 'checkpoint.npz')
            model.load_state_dict(loaded)
        Expects a dict of the form:
        {
            'parameters': { 'weight': np.array(...), 'bias': np.array(...) },
            'states': { 'stateful_states': np.array(...), ... }
        }
        """
        # 1. Update parameters
        if "parameters" in state_dict:
            self._set_attr_nested(
                "_parameters", state_dict["parameters"], is_parameter=True
            )

        # 2. Update states
        if "states" in state_dict:
            self._set_attr_nested("_states", state_dict["states"], is_parameter=False)

    def num_parameters(self) -> int:
        """
        Returns the total number of parameters in the module and its submodules.
        """
        return sum(p.data.size for p in self.parameters.values())

    def train(self) -> None:
        for module in self._modules.values():
            module.train()
        self._is_training = True

    def eval(self) -> None:
        for module in self._modules.values():
            module.eval()
        self._is_training = False

    def _get_attr_nested(self, attr_name: str, prefix: str = "") -> Dict[str, Any]:
        """
        Recursively collect items (parameters or states) from self and submodules.
        Returns a flattened dict: { 'prefix.param_name': value, ... }
        """
        out = {}
        # 1. Collect this module's items (e.g. self._parameters or self._states)
        current_dict = getattr(self, attr_name)  # e.g. self._parameters
        for k, v in current_dict.items():
            full_name = f"{prefix}.{k}" if prefix else k
            out[full_name] = v

        # 2. Recursively collect submodules
        for sub_name, mod in self._modules.items():
            sub_prefix = f"{prefix}.{sub_name}" if prefix else sub_name
            out.update(mod._get_attr_nested(attr_name, prefix=sub_prefix))

        return out

    def _set_attr_nested(
        self, attr_name: str, flat_dict: Dict[str, Any], is_parameter: bool
    ) -> None:
        """
        Recursively update either `_parameters` or `_states` by descending into submodules.

        Args:
            flat_dict: is a mapping
                { "submodule1.sub_weight": np.array(...),  ... }
            attr_name: "_parameters" or "_states"
            is_parameter: indicates if these are trainable parameters (Tensor) or not.
        """
        for full_name, value in flat_dict.items():
            parts = full_name.split(".")  # e.g. ["submodule1", "sub_weight"]
            *path, var_name = parts
            module_ref = self
            # Descend into submodules
            for p in path:
                module_ref = module_ref._modules[p]

            # Now module_ref is the submodule owning var_name in its container
            container = getattr(module_ref, attr_name)

            if var_name not in container:
                # If it doesn't exist, create it
                container[var_name] = Tensor(value) if is_parameter else value
            else:
                # Important: In-place update for parameters to avoid breaking optimizer references
                # Otherwise, the model would not train because optimizer will be pointing to old parameters
                if is_parameter:
                    container[var_name].data[...] = value
                else:
                    # For states, just assign (or do an in-place copy if desired)
                    container[var_name] = value


class ModuleList(Module):
    """
    A list-like container of submodules.
    Each submodule is properly registered so that it
    appears in this container's .parameters, .modules, etc.
    """

    def __init__(self, modules=None):
        super().__init__()
        if modules is not None:
            for module in modules:
                self.append(module)

    def append(self, module: Module) -> None:
        """Append a module to the end of the list."""
        index = len(self._modules)  # how many modules we already have
        # Use __setattr__ with a string key so that it registers `module` as a submodule
        setattr(self, str(index), module)

    def __getitem__(self, idx: int) -> Module:
        """Retrieve the submodule at index `idx`."""
        return self._modules[str(idx)]

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self):
        """Allow iteration over submodules in a for-loop, e.g. for m in module_list: ..."""
        for idx in range(len(self)):
            yield self[idx]


class Linear(Module):
    def __init__(self, input_size: int, output_size: int, **kwargs: Any) -> None:
        super().__init__(input_size, output_size, **kwargs)

        # weight is a matrix of shape (input_size, output_size)
        self._parameters["weight"] = xavier_uniform(
            Tensor(np.zeros((input_size, output_size)))
        )

        # bias is always 1-dimensional
        self._parameters["bias"] = Tensor(np.random.rand(output_size))

    def forward(self, x: Union[Tensor, np.ndarray]) -> Tensor:
        if not isinstance(x, Tensor):
            x = Tensor(x)

        logger.debug(f"{x.data.shape=}")
        logger.debug(f"Linear forward {self._parameters['weight'].data.shape=}")

        # this is just a linear transformation (dot matrix multiplication)
        return x @ self._parameters["weight"] + self._parameters["bias"]


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding_mode: str = "valid",
        bias: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Applies a 2D convolution over an input tensor.
        The shape convention is the same as PyTorch:
            input_shape = (N, in_channels, H, W)
            output_shape = (N, out_channels, H', W')

        where:
        - N is the batch size
        - in_channels is the number of input channels
        - out_channels is the number of kernels, where each kernel is convolved with the input tensor in all input channels

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            stride (int, optional): Stride of the convolution. Defaults to 1.
            bias (bool, optional): Whether to add a bias term. Defaults to True.
            padding_mode (str, optional): The amount of padding_mode to add to the input. Defaults to 'valid'.
            - "valid" means no padding.
            - "same" means padding such that the output shape is the same as the input shape.
        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding_mode=padding_mode,
            bias=bias,
            **kwargs,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding_mode = padding_mode
        self.bias = bias

        # The layer contains N number of kernels, which is equivalent to the out_channels
        # Each kernel is of shape (in_channels, H, W)
        # Each kernel needs to be convolved with the input tensor
        # The resulting tensor will have the shape (out_channels, H', W')
        self._parameters["weight"] = xavier_uniform(
            Tensor(
                np.zeros(
                    (
                        self.out_channels,
                        self.in_channels,
                        self.kernel_size,
                        self.kernel_size,
                    )
                )
            )
        )
        if bias:
            self._parameters["bias"] = Tensor(
                np.random.rand(self.out_channels)
            )  # one bias per kernel

    def forward(self, x: Union[Tensor, np.ndarray]) -> Tensor:
        if not isinstance(x, Tensor):
            x = Tensor(x)

        batch_size, in_channels, H, W = x.data.shape
        windows, (H_out, W_out) = extract_windows(
            x, self.kernel_size, self.stride, self.padding_mode
        )
        # windows: (H_out, W_out, N, C, H, W)

        # Permute windows to (N, C, H, W, H_out, W_out)
        windows = windows.permute(2, 3, 4, 5, 0, 1)
        # Now shape is (N, C, H, W, H_out, W_out)

        # Reshape to (N, C * H * W, H_out*W_out)
        windows = windows.reshape(
            batch_size, in_channels * self.kernel_size * self.kernel_size, H_out * W_out
        )

        # Transpose to (N, H_out*W_out, C*H*W) if needed, or directly reshape:
        # Actually, PyTorch uses (N, C*H*W, H_out*W_out) and then it flattens differently.
        # We can directly flatten to (N*H_out*W_out, C*H*W) for multiplication:
        windows = windows.transpose(1, 2).reshape(
            batch_size * H_out * W_out,
            in_channels * self.kernel_size * self.kernel_size,
        )

        # Prepare kernels: (out_channels, in_channels*H*W) then transpose to (C*H*W, out_channels)
        kernel_flat = (
            self._parameters["weight"]
            .reshape(
                self.out_channels, in_channels * self.kernel_size * self.kernel_size
            )
            .transpose(1, 0)
        )

        # Matrix multiply: (N*H_out*W_out, out_channels)
        output = windows @ kernel_flat

        # Reshape and permute to (N, out_channels, H_out, W_out)
        output = output.reshape(batch_size, H_out, W_out, self.out_channels).permute(
            0, 3, 1, 2
        )

        # Add bias if present
        if self.bias:
            output += self._parameters["bias"].reshape(-1, 1, 1)

        return output


class MaxPool2d(Module):
    def __init__(
        self,
        kernel_size: int,
        stride: Optional[int] = None,
        padding_mode: str = "valid",
        **kwargs: Any,
    ) -> None:
        """
        2D Max Pooling Layer

        Args:
            kernel_size (int): Size of pooling window (2 means 2x2 window)
            stride (int, optional): How many pixels to move the window each time. Defaults to kernel_size.
            padding_mode (str, optional): The type of padding_mode to add to the input. Defaults to 'valid'.
            - "valid" means no padding.
            - "same" means padding such that the output shape is the same as the input shape.
        """
        super().__init__(kernel_size, **kwargs)
        self.kernel_size = kernel_size
        self.stride = (
            stride if stride is not None else kernel_size
        )  # Default to kernel_size
        self.padding_mode = padding_mode

    def forward(self, x: Union[Tensor, np.ndarray]) -> Tensor:
        if not isinstance(x, Tensor):
            x = Tensor(x)

        windows, (H_out, W_out) = extract_windows(
            x, self.kernel_size, self.stride, self.padding_mode
        )

        H_out, W_out, batch_size, in_channels, H_kernel, W_kernel = windows.shape

        # Reshape windows to match spatial layout
        windows = windows.permute(2, 3, 0, 1, 4, 5)
        # Reorder axes to (batch_size, in_channels, H_out, W_out, kH * kW)
        windows = windows.reshape(batch_size, in_channels, H_out, W_out, -1)

        pooled = windows.max(axis=-1, keepdims=True).reshape(
            batch_size, in_channels, H_out, W_out
        )
        return pooled


class ResidualBlock(Module):
    """
    Residual Block as described in Deep Residual Learning for Image Recognition
    The residual block as a whole implements both F(x) and x (identity mapping)
    The function that were trying to learn is H(x) = F(x) + x
    F(x) is the convolutional layer with ReLU activation
    x is the identity mapping

    Paper: https://arxiv.org/abs/1512.03385
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__(in_channels, out_channels, stride=stride)
        self.conv1 = Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding_mode="same"
        )
        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding_mode="same",
        )

        # Add projection shortcut when dimensions change
        self.shortcut = Conv2d(
            in_channels, out_channels, kernel_size=1, stride=stride, padding_mode="same"
        )

    def forward(self, x: Union[Tensor, np.ndarray]) -> Tensor:
        identity = self.shortcut(x)  # Match channels

        out = self.conv1(x)
        out = relu(out)
        out = self.conv2(out)
        return relu(out) + identity


class RecurrentBlock(Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        dropout_prob: Optional[float] = None,
    ) -> None:
        """
        Recurrent Neural Network (RNN)
        Paper: https://arxiv.org/abs/1308.0850

        Args:
            input_size (int): The size of the input
            hidden_size (int): The size of the hidden state
            output_size (int, optional): The size of the output. Defaults to None.
            If specified, the output will be a linear combination of final hidden state
            and output layer weights.
            dropout_prob (float, optional): Whether to apply dropout to non-recurrent connections

        W_xh: transforms the input into "hidden embedding"
        W_hh: transforms the hidden state into the next hidden state
        W_hy: transforms the hidden state into the output
        """
        super().__init__(
            input_size,
            hidden_size,
            output_size=output_size,
            dropout_prob=dropout_prob,
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # Apply dropout only to non-recurrent connections
        # Paper: https://arxiv.org/abs/1409.2329
        self.dropout = Dropout(p=dropout_prob) if dropout_prob else None

        self._parameters["W_xh"] = xavier_uniform(
            Tensor(np.zeros((input_size, hidden_size)))
        )
        self._parameters["W_hh"] = xavier_uniform(
            Tensor(np.zeros((hidden_size, hidden_size)))
        )
        self._parameters["bias"] = Tensor(np.zeros((hidden_size,)))

        if output_size:
            self._parameters["W_hy"] = xavier_uniform(
                Tensor(np.zeros((hidden_size, output_size)))
            )
            self._parameters["bias_y"] = Tensor(np.zeros((output_size,)))
        else:
            self._parameters["W_hy"] = None
            self._parameters["bias_y"] = None

    def forward(self, x: Union[Tensor, np.ndarray]) -> Tensor:
        """
        Forward pass of the RNN

        Args:
            x (Tensor): The input tensor of shape (batch_size, sequence_length, input_size)
        """
        if not isinstance(x, Tensor):
            x = Tensor(x)

        batch_size = x.shape[0]
        seq_length = x.shape[1]
        hidden_state = Tensor(np.zeros((batch_size, self.hidden_size)))

        # Iterate through the sequence (or time dimension)
        for t in range(seq_length):
            x_t = x[:, t, :]  # shape: (batch_size, input_size)

            # Only apply dropout to the non-recurrent connections
            if self.dropout:
                x_t = self.dropout(x_t)

            # Update the hidden state
            hidden_state = tanh(
                x_t @ self._parameters["W_xh"]  # (batch_size, hidden_size)
                + hidden_state @ self._parameters["W_hh"]  # (batch_size, hidden_size)
                + self._parameters["bias"]
            )

        # If we defined the output size, we will compute the final output
        if self._parameters["W_hy"] is not None:
            # Only apply dropout to the non-recurrent connections
            if self.dropout:
                hidden_state = self.dropout(hidden_state)

            # We will only use the final hidden state in the output calculation
            return hidden_state @ self._parameters["W_hy"] + self._parameters["bias_y"]
        # If there is no output size, we return the final hidden state
        else:
            return hidden_state


class LongShortTermMemoryBlock(Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        dropout_prob: Optional[float] = None,
    ) -> None:
        """
        Long Short-Term Memory (LSTM) Neural Network
        Paper: https://www.bioinf.jku.at/publications/older/2604.pdf

        Args:
            input_size (int): The size of the input
            hidden_size (int): The size of the hidden state
            output_size (int, optional): The size of the output. Defaults to None.
            If specified, the output will be a linear combination of final hidden state
            and output layer weights.
            dropout_prob (float, optional): Whether to apply dropout to non-recurrent connections

        States:
            - Cell state: Internal memory of LSTM. It flows down the chain (time steps)
            in a way that can be modified but not completely overwritten each time
            unless the gates decide to do so.
            - Hidden state: The output of the LSTM block at time t

        W_f/bias_f: weights for the forget gate
        W_i/bias_i: weights for the input gate
        W_c/bias_c: weights for the cell gate
        W_o/bias_o: weights for the output gate
        W_hy/bias_y: weights for the final output (if output_size is specified)
        """
        super().__init__(
            input_size, hidden_size, output_size=output_size, dropout_prob=dropout_prob
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Apply dropout only to non-recurrent connections
        # Paper: https://arxiv.org/abs/1409.2329
        self.dropout = Dropout(p=dropout_prob) if dropout_prob else None

        self._parameters["W_f"] = xavier_uniform(
            Tensor(np.zeros((input_size + hidden_size, hidden_size)))
        )
        self._parameters["W_i"] = xavier_uniform(
            Tensor(np.zeros((input_size + hidden_size, hidden_size)))
        )
        self._parameters["W_c"] = xavier_uniform(
            Tensor(np.zeros((input_size + hidden_size, hidden_size)))
        )
        self._parameters["W_o"] = xavier_uniform(
            Tensor(np.zeros((input_size + hidden_size, hidden_size)))
        )
        self._parameters["bias_f"] = Tensor(np.zeros((hidden_size,)))
        self._parameters["bias_i"] = Tensor(np.zeros((hidden_size,)))
        self._parameters["bias_c"] = Tensor(np.zeros((hidden_size,)))
        self._parameters["bias_o"] = Tensor(np.zeros((hidden_size,)))

        if output_size:
            self._parameters["W_hy"] = xavier_uniform(
                Tensor(np.zeros((hidden_size, output_size)))
            )
            self._parameters["bias_y"] = Tensor(np.zeros((output_size,)))
        else:
            self._parameters["W_hy"] = None
            self._parameters["bias_y"] = None

    def forward(
        self,
        x: Union[Tensor, np.ndarray],
        hidden_state: Optional[Tensor] = None,
        C_t: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the LSTM

        Args:
            x (Tensor): The input tensor of shape (batch_size, sequence_length, input_size)

            The following are optional. If you are explicitly "unrolling" the recursive time series structure by calling forward() one time step at a time

            hidden_state (Tensor): Starting hidden state
            C_t (Tensor): The starting cell state

            Returns:
                1. output (if output_size was specified when initializing the LSTM block), otherwise, the last hidden state
                2. Last cell state
        """
        if not isinstance(x, Tensor):
            x = Tensor(x)

        batch_size = x.shape[0]
        seq_length = x.shape[1]
        hidden_state = (
            hidden_state
            if hidden_state
            else Tensor(np.zeros((batch_size, self.hidden_size)))
        )
        C_t = C_t if C_t else Tensor(np.zeros((batch_size, self.hidden_size)))

        # Iterate through the sequence (or time dimension)
        for t in range(seq_length):
            x_t = x[:, t, :]  # shape: (batch_size, input_size)

            # Only apply dropout to the non-recurrent connections
            if self.dropout:
                x_t = self.dropout(x_t)

            xh_stacked = Tensor.cat(
                [x_t, hidden_state], axis=1
            )  # (batch_size, input_size + hidden_size)

            # Compute Forget Gate (how much previous cell state C_t to keep or forget)
            forget_gate = sigmoid(
                # (batch_size, hidden_size + input_size) @ (input_size + hidden_size, hidden_size)
                # yields (batch_size, hidden_size)
                xh_stacked @ self._parameters["W_f"] + self._parameters["bias_f"]
            )

            # Compute Input Gate (how much new info to add to cell state)
            input_gate = sigmoid(
                # (batch_size, hidden_size + input_size) @ (input_size + hidden_size, hidden_size)
                # yields (batch_size, hidden_size)
                xh_stacked @ self._parameters["W_i"] + self._parameters["bias_i"]
            )

            # Compute Cell Gate (new candidate values that could be added to cell state)
            cell_gate = tanh(
                # (batch_size, hidden_size + input_size) @ (input_size + hidden_size, hidden_size)
                # yields (batch_size, hidden_size)
                xh_stacked @ self._parameters["W_c"] + self._parameters["bias_c"]
            )

            # Update Cell State (all in batch_size x hidden_size shapes)
            # Do element-wise multiplication
            C_t = forget_gate * C_t + input_gate * cell_gate

            # Compute Output Gate
            output_gate = sigmoid(
                # (batch_size, hidden_size + input_size) @ (input_size + hidden_size, hidden_size)
                # yields (batch_size, hidden_size)
                xh_stacked @ self._parameters["W_o"] + self._parameters["bias_o"]
            )

            # Update the hidden state
            # (batch_size, hidden_size)
            hidden_state = output_gate * tanh(C_t)

        # If we defined the output size, we will compute the final output
        if self._parameters["W_hy"] is not None:
            # Only apply dropout to the non-recurrent connections
            if self.dropout:
                hidden_state = self.dropout(hidden_state)

            # We will only use the final hidden state in the output calculation
            return hidden_state @ self._parameters["W_hy"] + self._parameters[
                "bias_y"
            ], C_t
        # If there is no output size, we return the final hidden state
        else:
            return hidden_state, C_t


class Embedding(Module):
    """
    Embedding layer that projects an arbitrary input_size down to embedding_size
    """

    def __init__(self, input_size: int, embedding_size: int) -> None:
        super().__init__()

        # weight.shape: (input_size, embedding_size)
        self._parameters["weight"] = Tensor(
            np.random.randn(input_size, embedding_size) * 0.01,
            requires_grad=True,
        )

    def forward(self, x: Union[Tensor, np.ndarray]) -> Tensor:
        """
        x: shape (batch_size, seq_len), each entry is an integer index in [0..vocab_size-1].
        Returns: (batch_size, seq_len, embedding_size)
        """
        if not isinstance(x, Tensor):
            x = Tensor(x)

        # indices.shape: (batch_size, seq_len)
        # result.shape: (batch_size, seq_len, embedding_size)
        return self._parameters["weight"].gather(index=x.data.astype(np.int32))


class LayerNorm(Module):
    """
    Layer Normalization

    This layer computes mean and variance from all of the summed inputs to the neurons in a layer on a single training case

    Paper: https://arxiv.org/abs/1607.06450
    """

    def __init__(self, input_size: int, epsilon: float = 1e-5, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self._parameters["gain"] = Tensor(np.ones((input_size,)))
        self._parameters["bias"] = Tensor(np.zeros((input_size,)))

    def forward(self, x: Tensor) -> Tensor:
        # Equation 4 in section 3.1 in the paper
        mean = x.mean(
            axis=-1, keepdims=True
        )  # (batch_size, seq_len, 1), across "input_size" dimension
        var = ((x - mean) ** 2).mean(
            axis=-1, keepdims=True
        )  # (batch_size, seq_len, 1), across "input_size" dimension
        x_norm = (x - mean) / (
            var + self.epsilon
        ).sqrt()  # (batch_size, seq_len, input_size)

        # scale and shift
        output = x_norm * self._parameters["gain"].expand(
            x_norm.shape
        ) + self._parameters["bias"].expand(x_norm.shape)
        return output


class BatchNorm(Module):
    """
    Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

    This layer normalizes the input tensor by subtracting the batch mean and dividing by the batch standard deviation.

    Paper: http://arxiv.org/abs/1502.03167
    """

    def __init__(
        self,
        input_size: int,
        momentum: float = 0.1,
        epsilon: float = 1e-5,
        **kwargs: Any,
    ) -> None:
        super().__init__(input_size, momentum=momentum, epsilon=epsilon, **kwargs)

        self.momentum = momentum  # used in running mean and variance calculation
        self.epsilon = epsilon  # small constant for numeric stability

        # Running stats (used for inference)
        self.running_mean = np.zeros(input_size)
        self.running_var = np.ones(input_size)

        # gamma and beta are learnable parameters
        # gamma is responsible for scaling the normalized input
        # beta is responsible for shifting the normalized input
        self._parameters["weight"] = Tensor(np.ones(input_size, dtype=np.float32))
        self._parameters["bias"] = Tensor(np.zeros(input_size, dtype=np.float32))

    def forward(self, x: Tensor) -> Tensor:
        """
        Note that the backward pass is implemented via primitive operations in the Tensor class.
        The operations in the forward pass have all been implemented as Tensor-level operations.
        """
        if self._is_training:
            # Compute batch statistics using Tensor operations
            batch_mean = x.mean(axis=0)
            diff = x - batch_mean
            var = (diff**2).sum(axis=0)

            biased_batch_var = var / x.data.shape[0]
            # Unbiased variance (divide by N-1) is based on Bessel's correction
            unbiased_batch_var = var / (x.data.shape[0] - 1)
            std_dev = (biased_batch_var + self.epsilon) ** 0.5

            # Update running statistics
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * batch_mean.data
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * unbiased_batch_var.data

            normalized = diff / std_dev
        else:
            normalized = (x - self.running_mean) / np.sqrt(
                self.running_var + self.epsilon
            )

        # Scale and shift
        return normalized * self._parameters["weight"].expand(
            x.data.shape
        ) + self._parameters["bias"].expand(x.data.shape)


class Dropout(Module):
    def __init__(self, p: float = 0.5, **kwargs: Any) -> None:
        """
        The Dropout layer randomly sets a fraction of input units to 0 at each update during training time.

        "It prevents overfitting and provides a way of approximately combining exponentially many different
        neural network architectures efficiently."
        Paper: https://arxiv.org/abs/1207.0580

        Args:
            p (float, optional): Fraction of the input units to drop. Defaults to 0.5.
        """
        super().__init__(p=p, **kwargs)
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self._is_training:
            mask = np.random.binomial(1, 1 - self.p, size=x.data.shape)
            return (
                x
                * mask
                / (
                    1 - self.p
                )  # we scale the output by 1/(1-p) to keep the expected output the same
                if self.p < 1
                else x * 0  # when p=1, drop everything by multiplying by 0
            )
        return x


class ScaledDotProductAttention(Module):
    """
    Implements the Scaled Dot-Product Attention in Section 3.2.1 in the paper.
    Paper: https://arxiv.org/abs/1706.03762

    Attention(Q,K,V) = softmax(Q transpose(K) / sqrt(key_dim)) V
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        attention_size = Tensor(key.shape[-1])

        # scaled dot product
        # (batch_size, num_heads, sequence_len, sequence_len)
        att_score = (query @ key.transpose(2, 3)) / attention_size.sqrt()

        # mask (optional)
        if mask is not None:
            # broadcast across heads
            att_score = att_score + (mask * -1e9)
        att_score = softmax(att_score)
        return att_score @ value


class MultiHeadAttention(Module):
    """
    Implements the Multi-Head Attention in Section 3.2.2 in the paper.
    Paper: https://arxiv.org/abs/1706.03762

    Instead of performing a single attention with hidden_size keys, query, and values,
    we project them "num_heads" times with different learned linear projects
    """

    def __init__(self, num_heads: int, hidden_size: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.attention_size = (
            hidden_size // num_heads
        )  # We assume query, key, value all have the same dimension

        # Project query, key, value using linear layers before passing to attention
        self.q_linear = Linear(hidden_size, hidden_size)
        self.k_linear = Linear(hidden_size, hidden_size)
        self.v_linear = Linear(hidden_size, hidden_size)

        self.attention = ScaledDotProductAttention()
        self.fc = Linear(hidden_size, hidden_size)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        batch_size = query.shape[0]

        # We try to avoid explicitly splitting and combining the heads
        # So we are just using matrix multiplication to paralellize everything
        # Then we are going to reshape the resulting output to the correct
        # dimensions
        # 1. Linear Projections
        # (batch_size, num_heads, seq_len, input_size)
        query = (
            self.q_linear(query)
            .view(batch_size, -1, self.num_heads, self.attention_size)
            .permute(0, 2, 1, 3)
        )
        key = (
            self.k_linear(key)
            .view(batch_size, -1, self.num_heads, self.attention_size)
            .permute(0, 2, 1, 3)
        )
        value = (
            self.v_linear(value)
            .view(batch_size, -1, self.num_heads, self.attention_size)
            .permute(0, 2, 1, 3)
        )

        # 2. Apply Attention
        att_score = self.attention(query, key, value, mask=mask)

        att_score = att_score.permute(
            0, 2, 1, 3
        )  # (batch_size, num_heads , seq_len, input_size)
        # Expect (batch_size, seq_len, hidden_size)
        att_score = att_score.view(batch_size, -1, self.num_heads * self.attention_size)
        assert att_score.shape == (
            batch_size,
            query.shape[2],
            self.num_heads * query.shape[3],
        )

        del query, key, value

        return self.fc(att_score)


########### Utility Functions ###########
def extract_windows(
    x: Union[Tensor, np.ndarray],
    kernel_size: int,
    stride: int,
    padding_mode: str = "valid",
) -> Tuple[Tensor, Tuple[int, int]]:
    """
    Extract windows from input tensor while maintaining computational graph.

    Args:
        x (Tensor): Input tensor of shape (batch_size, channels, height, width)
        kernel_size (int): Size of the sliding window
        stride (int): Step size between windows
        padding_mode (str): Type of padding - "valid" or "same"

    Returns:
        tuple: (windows, output_shape)
            - windows: Stacked tensor of shape (H_out, W_out, batch_size, channels, kernel_size, kernel_size)
            - output_shape: (H_out, W_out) tuple for reshaping
    """
    if not isinstance(x, Tensor):
        x = Tensor(x)

    batch_size, in_channels, H, W = x.data.shape

    if padding_mode == "same":
        pad_h = pad_w = kernel_size // 2
        x_padded = x.pad(
            pad_width=((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
            mode="constant",
            constant_values=0,
        )
    else:
        x_padded = x

    # Get windows using strided_windows
    windows = x_padded.strided_windows(kernel_size, stride)

    # Calculate output dimensions
    H_out = (x_padded.shape[2] - kernel_size) // stride + 1
    W_out = (x_padded.shape[3] - kernel_size) // stride + 1

    return windows, (H_out, W_out)
