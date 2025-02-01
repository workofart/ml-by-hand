try:
    # drop-in replacement for numpy for GPU acceleration
    import cupy as np  # type: ignore

    _ = np.cuda.runtime.getDeviceCount()  # Check if a CUDA device is available
except Exception:
    import numpy as np

from autograd import functional, nn, optim
from autograd.tensor import Tensor
from autograd.tools import data, trainer

"""
Neural Turing Machines
Precursor to transformers architecture.
Paper: https://arxiv.org/abs/1410.5401
"""


class Memory:
    """
    (batch_size, memory_length, memory_dim)
    """

    def __init__(self, memory_length, memory_dim) -> None:
        self.memory_length = memory_length
        self.memory_dim = memory_dim
        self.reset_memory()
        self._memory = None

    def reset_memory(self, batch_size=1):
        self._memory = Tensor(
            data=np.zeros((batch_size, self.memory_length, self.memory_dim)),
            requires_grad=False,
        )

    def read(self):
        return self._memory

    def write(self, new_memory):
        self._memory = new_memory


class ReadHead(nn.Module):
    """
    "3.1 Reading" section in paper: https://arxiv.org/abs/1410.5401
    """

    def __init__(self, memory: Memory):
        super().__init__()
        self.memory = memory

    def forward(self, read_weights):
        """
        r_t = \sum_i w_t(i) M_t(i)
        where:
            - r_t: Read vector at time t
            - M_t: N x M memory matrix at time t
            - N: the number of memory locations
            - w_t(i): Weights at time t
        return:
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
    "3.2 Writing" section in paper: https://arxiv.org/abs/1410.5401
    """

    def __init__(self, memory: Memory):
        super().__init__()
        self.memory = memory

    def forward(self, write_weights, erase_vector, add_vector):
        """
        M_t(i) = (M_{t-1}(i)[1 - w_t(i) e_t]) + w_t(i) a_t
        where:
            - M_t(i): Memory vector at time t
            - w_t(i): Weights at time t
            - e_t: Erase vector at time t
            - a_t: Add vector at time t
            - 1: Row vector of all ones
        If both erase vector and weight at the location i are both 0,
        then (1 - 1) will effectively "erase" the memory. Otherwise,
        the memory is left unchanged
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
    The main model that encasulates all the components to build the
    Neural Turing Machine for any task.

    Refer to Figure 2: Flow Diagram of the Addressing Mechanism
    in paper: https://arxiv.org/abs/1410.5401
    """

    def __init__(
        self,
        input_size,
        memory_length,
        memory_dim,
        hidden_size,
        output_size,
    ):
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
        Run through the turing machine and LSTM model.

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
            x: (batch_size, sequence_length, input_size): The input vector
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

            # 6. Update weights
            read_weights = new_weights

        return Tensor.stack(outputs, axis=1)  # (batch_size, seq_len, output_size)

    def _content_addressing(self, key: Tensor, key_strength: Tensor):
        """
        3.3.1 Focusing by Content in Paper: https://arxiv.org/abs/1410.5401

        Args:
            key (Tensor): What index to focus on
            key_strength (Tensor): How much to focus on

        Returns:
            Tensor: Normalied weighting matrix based on content similarity
        """
        memory = self.memory.read()
        key_strength = functional.relu(key_strength) + 1e-8

        # Compute cosine similarity between current content and
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
        Location-based Addressing
        3.3.2 Focusing by Location in Paper: https://arxiv.org/abs/1410.5401

        Args:
            content_weights (Tensor): The output of content-based addressing, determined what to focus on
            sharpening_factor (Tensor): How much to sharpen the weights to combat shift weighting being not sharp
            interpolation_gate (Tensor): In the range of (0, 1), used to blend the content weights produced by timestep (t - 1) and content weights at timestep t
            old_weights (Tensor): Content weights read from memory produced by timestep (t-1)
            shift_logits (Tensor): Normalized distribution over the allowed integer shifts

        Returns:
            Tensor: Sharpened weight matrix based on content similarity and order and sequence
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
        Take some information from the neighboring sequence and order
        3.3.2 Focusing by Location in Paper: https://arxiv.org/abs/1410.5401
        It's called "circular convolution" in the paper.

        Args:
            old_weights (Tensor): Content weights read from memory produced by timestep (t-1)
            shift_logits (Tensor): Normalized distribution over the allowed integer shifts

        Returns:
            Tensor: New weights after considering neighboring memory locations
        """
        shift_probs = functional.softmax(shift_logits)  # (batch_size, 3)
        # shift_probs[:, 0] => p(shift=-1)
        # shift_probs[:, 1] => p(shift=0)
        # shift_probs[:, 2] => p(shift=+1)

        w_left = old_weights.roll(shifts=-1, dims=1)
        w_right = old_weights.roll(shifts=1, dims=1)
        p_left = shift_probs[:, 0:1]  # batch_size, 1
        p = shift_probs[:, 1:2]  # batch_size, 1
        p_right = shift_probs[:, 2:3]  # batch_size, 1

        new_weight = w_left * p_left + old_weights * p + w_right * p_right
        return new_weight

    def _sharpen(self, weights, gamma):
        """
        Raise weights to the power gamma to prevent leakage or dispersion of weights
        overtime if the shift weights are not sharp.

        3.3.2 Focusing by Location in paper: https://arxiv.org/abs/1410.5401

        w_t = w_t^gamma / \sum_j w_t^gamma

        Args:
            weights: original weights unsharpened at time t
            gamma: scalar at time t
        """
        gamma = functional.relu(gamma) + 1.0  # ensure gamma >= 1
        numerator = weights**gamma

        # Normalize
        denominator = numerator.sum(axis=1, keepdims=True) + 1e-8
        return numerator / denominator


def generate_copy_data(n_samples=100, seq_len=5, input_size=4):
    """
    Helper Function to generate dummy data for the "Copy Task"
    Similar to the 4.1 Copy Experiment in the paper: https://arxiv.org/abs/1410.5401
    Returns:
      X: (n_samples, seq_len) sparse int tokens  in [0..input_size-1]
      Y: (n_samples, seq_len) same as X, for the copy task
    """
    # Generate random integer sequences
    X_data = np.random.randint(0, input_size, size=(n_samples, seq_len))
    # For the copy task, the label is the same
    Y_data = X_data.copy()
    return X_data, Y_data


def to_one_hot(sequence_batch, vocab_size):
    """
    sequence_batch: shape (batch_size, seq_len), each entry is [0..vocab_size-1]
    Returns one_hot: shape (batch_size, seq_len, vocab_size)
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
    This is only used a comparison against Neural Turing Machine
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__(input_size, hidden_size, output_size)
        self.lstm = nn.LongShortTermMemoryBlock(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, x):
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

    # Generate a longer sequence to see if the model can generalize well
    X_val, y_val = generate_copy_data(
        n_samples=10, seq_len=seq_len * 5, input_size=input_size
    )
    X_val = to_one_hot(X_val, input_size)

    print("------------- Neural Turing Machine ---------------")
    ntm = NeuralTuringMachine(
        input_size=input_size,
        memory_length=memory_length,
        memory_dim=memory_dim,
        output_size=output_size,
        hidden_size=hidden_size,
    )
    train_data_loader = data.SimpleDataLoader(X, y, batch_size=batch_size, shuffle=True)
    val_data_loader = data.SimpleDataLoader(
        X_val, y_val, batch_size=batch_size, shuffle=True
    )
    t = trainer.SimpleTrainer(
        model=ntm,
        loss_fn=functional.cross_entropy,
        optimizer=optim.Adam(ntm.parameters, lr=5e-3),
        epochs=epochs,
        output_type="logits",
    )
    t.fit(train_data_loader, val_data_loader)

    print("------------- Long-Short Memory Network ---------------")
    lstm = LSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

    t = trainer.SimpleTrainer(
        model=lstm,
        loss_fn=functional.cross_entropy,
        optimizer=optim.Adam(lstm.parameters, lr=5e-3),
        epochs=epochs,
        output_type="logits",
    )

    t.fit(train_data_loader, val_data_loader)
