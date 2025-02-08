try:
    # drop-in replacement for numpy for GPU acceleration
    import cupy as np  # type: ignore

    _ = np.cuda.runtime.getDeviceCount()  # Check if a CUDA device is available
except Exception:
    import numpy as np

from autograd import functional, nn, optim
from autograd.tensor import Tensor
from autograd.tools import data, trainer
from autograd.tools.config_schema import GenericTrainingConfig

"""
Neural Turing Machines
Precursor to transformers architecture.
Paper: https://arxiv.org/abs/1410.5401
"""


class Memory:
    """
    Implements a simple memory module for a Neural Turing Machine.

    This class maintains a memory matrix for each batch with dimensions
    (batch_size, memory_length, memory_dim) and provides methods to reset,
    read from, and write to the memory.
    """

    def __init__(self, memory_length, memory_dim) -> None:
        """
        Initialize the Memory object.

        Args:
            memory_length (int): The number of memory locations.
            memory_dim (int): The dimensionality of each memory vector.
        """
        self.memory_length = memory_length
        self.memory_dim = memory_dim
        self.reset_memory()
        self._memory = None

    def reset_memory(self, batch_size=1):
        """
        Resets the memory by initializing a zero matrix for the given batch size.

        Args:
            batch_size (int): The number of samples in the batch. Defaults to 1.
        """
        self._memory = Tensor(
            data=np.zeros((batch_size, self.memory_length, self.memory_dim)),
            requires_grad=False,
        )

    def read(self):
        """
        Returns the current memory content.

        Returns:
            Tensor: The memory matrix of shape (batch_size, memory_length, memory_dim).
        """
        return self._memory

    def write(self, new_memory):
        """
        Overwrites the memory with new values.

        Args:
            new_memory (Tensor): A tensor with the same shape as the memory.
        """
        self._memory = new_memory


class ReadHead(nn.Module):
    """
    Implements the reading mechanism for the Neural Turing Machine.

    This module corresponds to section "3.1 Reading" in https://arxiv.org/abs/1410.5401.
    It uses provided read weights to compute a weighted sum over the memory,
    returning a read vector.
    """

    def __init__(self, memory: Memory):
        """
        Initialize the ReadHead.

        Args:
            memory (Memory): The Memory instance to read from.
        """
        super().__init__()
        self.memory = memory

    def forward(self, read_weights):
        """
        Computes the read vector from the memory using the read weights.

        r_t = \sum_i w_t(i) M_t(i)
        where:
            - r_t: Read vector at time t
            - M_t: N x M memory matrix at time t
            - N: the number of memory locations
            - w_t(i): Weights at time t

        Args:
            read_weights (Tensor): Weights of shape (batch_size, memory_length).

        Returns:
            r_t: read vector at time t of shape (batch_size, memory_dim)
        """
        read_weights = read_weights.view(
            read_weights.shape[0], read_weights.shape[1], 1
        )
        r_t = (self.memory.read() * read_weights).sum(axis=1, keepdims=True)
        # if you want to end up with shape (B, input_size), you can do a final squeeze(1)
        return r_t.view(r_t.shape[0], r_t.shape[2])  # => (B, input_size)


class WriteHead(nn.Module):
    """
    Implements the writing mechanism for the Neural Turing Machine.

    This module corresponds to section "3.2 Writing" in https://arxiv.org/abs/1410.5401
    It updates the memory based on the write weights, an erase vector, and an add vector.
    """

    def __init__(self, memory: Memory):
        """
        Initialize the WriteHead.

        Args:
            memory (Memory): The Memory instance to write to.
        """
        super().__init__()
        self.memory = memory

    def forward(self, write_weights, erase_vector, add_vector):
        """
        Writes to the memory using the given weights, erase vector, and add vector.

        The update rule is:
            M_t(i) = M_{t-1}(i) * (1 - w_t(i) * e_t) + w_t(i) * a_t
        where:
            - M_t(i): Memory vector at time t
            - w_t(i): Weights at time t
            - e_t: Erase vector at time t
            - a_t: Add vector at time t
            - 1: Row vector of all ones

        Args:
            write_weights (Tensor): Weights for writing of shape (batch_size, memory_length).
            erase_vector (Tensor): Erase vector of shape (batch_size, memory_dim).
            add_vector (Tensor): Add vector of shape (batch_size, memory_dim).

        If both erase vector and weight at the location i are both 0,
        then (1 - 1) will effectively "erase" the memory. Otherwise,
        the memory is left unchanged

        The function updates the memory in place.
        """
        old_memory = self.memory.read()  # (batch_size, memory_length, memory_dim)

        # Expand shapes for broadcasting
        w_t = write_weights.view(
            write_weights.shape[0], write_weights.shape[1], 1
        )  # (batch_size, memory_length, 1)
        e_t = erase_vector.view(
            erase_vector.shape[0], 1, erase_vector.shape[1]
        )  # (batch_size, 1, memory_dim)
        a_t = add_vector.view(
            add_vector.shape[0], 1, add_vector.shape[1]
        )  # (batch_size, 1, memory_dim)

        # 1) Erase
        memory_after_erase = old_memory * (1 - w_t * e_t)

        # 2) Add
        memory_after_add = memory_after_erase + (w_t * a_t)

        self.memory.write(memory_after_add)


class NeuralTuringMachine(nn.Module):
    """
    Implements a Neural Turing Machine (NTM).

    This model encapsulates the components of an NTM including an external memory,
    a controller implemented via an LSTM block, and both read and write heads.
    It follows the addressing mechanisms described in https://arxiv.org/abs/1410.5401 (Figure 2).

    Attributes:
        memory (Memory): The external memory module.
        lstm (nn.LongShortTermMemoryBlock): The LSTM-based controller.
        Various linear layers for computing addressing parameters (content key, shift, interpolation, etc.).
        read_head (ReadHead): Module for reading from memory.
        write_head (WriteHead): Module for writing to memory.
    """

    def __init__(
        self,
        input_size,
        memory_length,
        memory_dim,
        hidden_size,
        output_size,
    ):
        """
        Initialize the Neural Turing Machine.

        Args:
            input_size (int): Dimensionality of the input vector.
            memory_length (int): Number of memory slots.
            memory_dim (int): Dimensionality of each memory slot.
            hidden_size (int): Number of hidden units in the controller.
            output_size (int): Dimensionality of the model output.
        """
        super().__init__(
            input_size, memory_length, memory_dim, hidden_size, output_size
        )
        self.memory = Memory(
            memory_length=memory_length,
            memory_dim=memory_dim,
        )
        self.lstm = nn.LongShortTermMemoryBlock(
            input_size=input_size + memory_dim,
            hidden_size=hidden_size,
            output_size=None,  # return final hidden state
        )
        self.content_key_layer = nn.Linear(hidden_size, memory_dim)
        self.location_shift_layer = nn.Linear(
            hidden_size, 3
        )  # maps to 3 dims -> [shift - 1, shift, shift + 1]
        self.interpolation_gate_layer = nn.Linear(hidden_size, 1)
        self.sharpening_layer = nn.Linear(hidden_size, 1)
        self.key_strength_layer = nn.Linear(hidden_size, 1)
        self.erase_layer = nn.Linear(
            hidden_size, memory_dim
        )  # (batch_size, memory_dim)
        self.add_layer = nn.Linear(hidden_size, memory_dim)  # (batch_size, memory_dim)
        self.output_layer = nn.Linear(hidden_size, output_size)

        self.hidden_size = hidden_size
        self.read_head = ReadHead(self.memory)
        self.write_head = WriteHead(self.memory)
        self.memory_length = memory_length

    def forward(self, x):
        """
        Run the Neural Turing Machine on the input sequence.

        For each time step:
            1. Read from memory
            2. Combine (x_t + read_vector)
            3. Controller output shift logits and next token
            4. Convert shift logits to read/write weights
            5. Write the output to memory
            6. Update weights

        Additional Paramters:
            shift_logits:      (batch_size, seq_len, 3)
            content_keys:      (batch_size, seq_len, read_size)
            key_strength:      (batch_size, seq_len, 1)
            interpolation:     (batch_size, seq_len, 1)
            sharpening_factor: (batch_size, seq_len, 1)
            outputs:           (batch_size, seq_len, output_size)

        Args:
            x (np.ndarray): Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            Tensor: Stacked outputs from each time step, of shape (batch_size, sequence_length, output_size).
        """
        batch_size, seq_len, input_size = x.shape
        self.memory.reset_memory(batch_size)

        h_t = Tensor(np.zeros((batch_size, self.hidden_size)))
        cell_state = Tensor(np.zeros((batch_size, self.hidden_size)))
        read_weights = Tensor(
            np.ones((batch_size, self.memory_length)) / self.memory_length,
            requires_grad=False,
        )
        outputs = []

        for t in range(seq_len):
            # 1. Read from memory
            read_vector = self.read_head(read_weights)  # (batch_size, memory_dim)

            # 2. Combine (x_t + read_vector)
            combined_input = Tensor.cat(
                [Tensor(x[:, t, :]), read_vector], axis=1
            )  # (batch_size, 1, input_size + memory_dim)
            combined_input = combined_input.view(
                (combined_input.shape[0], 1, combined_input.shape[1])
            )

            # 3. LSTM (controller) forward. Controller output shift logits and next token
            # Pass in the current hidden state and cell state
            # Output new cell_state and next hidden layer after running for 1 time step
            # 3.4 Controller Network in paper: https://arxiv.org/abs/1410.5401
            h_t, cell_state = self.lstm(
                x=combined_input, hidden_state=h_t, C_t=cell_state
            )

            location_shift = self.location_shift_layer(h_t)
            content_key = self.content_key_layer(h_t)
            key_strength = self.key_strength_layer(h_t)
            interpolation_gate = self.interpolation_gate_layer(h_t)
            sharpening_factor = self.sharpening_layer(h_t)
            erase_vector = self.erase_layer(h_t)
            add_vector = self.add_layer(h_t)
            output = self.output_layer(h_t)
            outputs.append(output)

            # 4. Convert shift logits to read/write weights
            new_weights = self._content_addressing(content_key, key_strength)
            new_weights = self._location_addressing(
                content_weights=new_weights,
                sharpening_factor=sharpening_factor,
                interpolation_gate=interpolation_gate,
                old_weights=read_weights,
                shift_logits=location_shift,
            )

            # 5. Write to memory
            self.write_head(new_weights, erase_vector, add_vector)

            # 6. Update weights for next time step
            read_weights = new_weights

        return Tensor.stack(outputs, axis=1)  # (batch_size, seq_len, output_size)

    def _content_addressing(self, key: Tensor, key_strength: Tensor):
        """
        Perform content-based addressing to compute a normalized weight distribution.
        3.3.1 Focusing by Content in Paper: https://arxiv.org/abs/1410.5401

        This function computes cosine similarity between the key vector and each memory
        location, scales the similarity by key strength, and applies softmax to obtain weights.

        Args:
            key (Tensor): Key vector for addressing, shape (batch_size, memory_dim).
            key_strength (Tensor): Scalar or vector indicating the focus strength, shape (batch_size, 1).

        Returns:
            Tensor: A normalized weighting tensor of shape (batch_size, memory_length).
        """
        memory = self.memory.read()
        key_strength = functional.relu(key_strength) + 1e-8

        # Compute cosine similarity between current content and memory
        similarity = (memory * key.view(key.shape[0], 1, key.shape[1])).sum(
            axis=2
        )  # (batch_size, memory_length)

        # norms
        # \frac{key \cdot memory}{key norm \cdot memory norm}
        key_norm = (key**2).sum(axis=1).sqrt()  # batch_size
        key_norm = key_norm.view(key_norm.shape[0], 1)
        memory_norm = (memory**2).sum(axis=2).sqrt()  # (batch_size, memory_length)
        cosine_similarity = similarity / (key_norm * memory_norm + 1e-8)
        return functional.softmax(key_strength * cosine_similarity)

    def _location_addressing(
        self,
        content_weights,
        sharpening_factor,
        interpolation_gate,
        old_weights,
        shift_logits,
    ):
        """
        Performs location-based addressing to refine memory read weights.
        3.3.2 Focusing by Location in Paper: https://arxiv.org/abs/1410.5401

        This function interpolates between content-based weights and previous weights,
        applies a circular convolution via shifting, and sharpens the result.

        Args:
            content_weights (Tensor): Weights obtained from content addressing.
            sharpening_factor (Tensor): Factor for sharpening the distribution.
            interpolation_gate (Tensor): In the range of (0, 1). Gate to blend previous weights and current content weights.
            old_weights (Tensor): Weights from the previous time step.
            shift_logits (Tensor): Logits determining the shift in attention.

        Returns:
            Tensor: New, refined memory read weights.
        """
        interpolated_weights = (
            interpolation_gate * content_weights
            + (1 - interpolation_gate) * old_weights
        )
        shift_weights = self._shift_attention(interpolated_weights, shift_logits)
        read_weights = self._sharpen(shift_weights, sharpening_factor)
        return read_weights

    def _shift_attention(self, old_weights, shift_logits):
        """
        Adjusts attention weights using circular convolution.
        3.3.2 Focusing by Location in Paper: https://arxiv.org/abs/1410.5401

        This function computes a weighted sum of the previous weights shifted left and right
        based on shift probabilities obtained from shift_logits.

        Args:
            old_weights (Tensor): The previous memory read weights.
            shift_logits (Tensor): Logits used to compute a distribution over allowed shifts.

        Returns:
            Tensor: New attention weights after considering neighboring positions.
        """
        shift_probs = functional.softmax(shift_logits)  # (batch_size, 3)
        # shift_probs[:, 0] => p(shift=-1)
        # shift_probs[:, 1] => p(shift=0)
        # shift_probs[:, 2] => p(shift=+1)

        w_left = old_weights.roll(shifts=-1, dims=1)
        w_right = old_weights.roll(shifts=1, dims=1)
        p_left = shift_probs[:, 0:1]  # (batch_size, 1)
        p = shift_probs[:, 1:2]  # (batch_size, 1)
        p_right = shift_probs[:, 2:3]  # (batch_size, 1)

        new_weight = w_left * p_left + old_weights * p + w_right * p_right
        return new_weight

    def _sharpen(self, weights, gamma):
        r"""
        Sharpens the attention weights by raising them to a power.
        Raise weights to the power gamma to prevent leakage or dispersion of weights overtime if the shift weights are not sharp.

        3.3.2 Focusing by Location in paper: https://arxiv.org/abs/1410.5401

        $$w_t = w_t^{\gamma} / \sum_j w_t^{\gamma}$$

        This function prevents the dispersion of weights over time by raising the weights
        to a power gamma and then normalizing them.

        Args:
            weights (Tensor): The unsharpened attention weights.
            gamma (Tensor): The sharpening factor.

        Returns:
            Tensor: The sharpened, normalized attention weights.
        """
        gamma = functional.relu(gamma) + 1.0  # ensure gamma >= 1
        numerator = weights**gamma

        # Normalize
        denominator = numerator.sum(axis=1, keepdims=True) + 1e-8
        return numerator / denominator


def generate_copy_data(n_samples=100, seq_len=5, input_size=4):
    """
    Generate dummy data for the "Copy Task".
    Similar to the 4.1 Copy Experiment in the paper: https://arxiv.org/abs/1410.5401

    This helper function creates random integer sequences as input and uses the same
    sequences as target output. It is intended to simulate the copy task from the NTM paper.

    Args:
        n_samples (int): Number of sequences to generate.
        seq_len (int): Length of each sequence.
        input_size (int): Range of token IDs is [0, input_size-1].

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - X: A (n_samples, seq_len) array of input token IDs in [0..input_size-1]
            - Y: A copy of X, serving as the target output.
    """
    # Generate random integer sequences
    X_data = np.random.randint(0, input_size, size=(n_samples, seq_len))
    # For the copy task, the label is the same
    Y_data = X_data.copy()
    return X_data, Y_data


def to_one_hot(sequence_batch, vocab_size):
    """
    Convert a batch of token index sequences into one-hot encoded representations.

    Args:
        sequence_batch (np.ndarray): Array of shape (batch_size, seq_len) containing integer token IDs each entry is [0..vocab_size-1]
        vocab_size (int): Total number of tokens in the vocabulary.

    Returns:
        np.ndarray: One-hot encoded tensor of shape (batch_size, seq_len, vocab_size).
    """
    bsz, seq_len = sequence_batch.shape
    one_hot = np.zeros((bsz, seq_len, vocab_size), dtype=np.float32)
    for i in range(bsz):
        for t in range(seq_len):
            token = sequence_batch[i, t]
            one_hot[i, t, token] = 1.0
    return one_hot


class LSTM(nn.Module):
    """
    A simple LSTM network used for comparison against the Neural Turing Machine.

    This model processes an input sequence through an LSTM block and applies a final linear
    layer to produce outputs for each time step.
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the LSTM model.

        Args:
            input_size (int): The dimensionality of the input.
            hidden_size (int): The number of hidden units in the LSTM block.
            output_size (int): The dimensionality of the output.
        """
        super().__init__(input_size, hidden_size, output_size)
        self.lstm = nn.LongShortTermMemoryBlock(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, x):
        """
        Compute the forward pass of the LSTM network.

        Processes the input sequence one time step at a time, updating the hidden state
        and cell state, and collecting the output at each step.

        Args:
            x (np.ndarray): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            Tensor: Stacked outputs of shape (batch_size, seq_len, output_size).
        """
        batch_size, seq_len, input_size = x.shape
        outputs = []
        h_t = Tensor(np.zeros((batch_size, self.hidden_size)))
        cell_state = Tensor(np.zeros((batch_size, self.hidden_size)))

        for t in range(seq_len):
            x_t = Tensor(x[:, t, :])  # (batch_size, 1, input_size)
            x_t = x_t.view((x_t.shape[0], 1, x_t.shape[1]))
            h_t, cell_state = self.lstm(
                x=x_t, hidden_state=h_t, C_t=cell_state
            )  # hidden state, cell state for this timestep
            out = self.fc(h_t)
            outputs.append(out)
        return Tensor.stack(outputs, axis=1)


if __name__ == "__main__":
    """
    Main script for training and evaluating the Neural Turing Machine and LSTM models
    on a copy task.

    The script performs the following steps:
      1) Define hyperparameters and dimensions for the copy task.
      2) Generate dummy data using generate_copy_data and convert it to one-hot representations.
      3) Create training and validation data loaders.
      4) Configure and train a Neural Turing Machine (NTM) model using SimpleTrainer.
      5) Configure and train an LSTM model for comparison.
      6) The models are trained on the copy task, where the goal is to reproduce the input sequence.
    """
    # Suppose we have a batch of input sequences: (batch_size, seq_len, input_size)
    batch_size = 2
    seq_len = 40
    input_size = 24
    memory_length = 60
    memory_dim = 60
    hidden_size = 10
    output_size = 24  # one-hot dimension size
    epochs = 30

    # Generate dummy data
    X, y = generate_copy_data(n_samples=10, seq_len=seq_len, input_size=input_size)
    X = to_one_hot(X, input_size)

    # Generate a longer sequence to test generalization
    X_val, y_val = generate_copy_data(
        n_samples=10, seq_len=seq_len * 5, input_size=input_size
    )
    X_val = to_one_hot(X_val, input_size)

    print("------------- Neural Turing Machine ---------------")
    train_data_loader = data.SimpleDataLoader(X, y, batch_size=batch_size, shuffle=True)
    val_data_loader = data.SimpleDataLoader(
        X_val, y_val, batch_size=batch_size, shuffle=True
    )

    # Create training configuration for the Neural Turing Machine.
    config_ntm = GenericTrainingConfig(
        training_run_name="neural_turing_machine_copy_task",
        total_epochs=epochs,
        checkpoint_freq=epochs,
        model_kwargs={
            "input_size": input_size,
            "memory_length": memory_length,
            "memory_dim": memory_dim,
            "output_size": output_size,
            "hidden_size": hidden_size,
        },
        optimizer_kwargs={"lr": 5e-3},
    )

    ntm_trainer = trainer.SimpleTrainer(
        model_cls=NeuralTuringMachine,
        optimizer_cls=optim.Adam,
        loss_fn=functional.cross_entropy,
        output_type="logits",
        config=config_ntm,
    )
    ntm_trainer.fit(train_data_loader, val_data_loader)

    print("------------- Long-Short Memory Network ---------------")
    # Create training configuration for the LSTM.
    config_lstm = GenericTrainingConfig(
        training_run_name="lstm_copy_task",
        total_epochs=epochs,
        checkpoint_freq=epochs,
        model_kwargs={
            "input_size": input_size,
            "hidden_size": hidden_size,
            "output_size": output_size,
        },
        optimizer_kwargs={"lr": 5e-3},
    )

    lstm_trainer = trainer.SimpleTrainer(
        model_cls=LSTM,
        optimizer_cls=optim.Adam,
        loss_fn=functional.cross_entropy,
        output_type="logits",
        config=config_lstm,
    )
    lstm_trainer.fit(train_data_loader, val_data_loader)
