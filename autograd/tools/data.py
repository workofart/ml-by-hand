import os
from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional, Tuple, Union

try:
    # drop-in replacement for numpy for GPU acceleration
    import cupy as np  # type: ignore

    _ = np.cuda.runtime.getDeviceCount()  # Check if a CUDA device is available
except Exception:
    import numpy as np
import pyarrow.parquet as pq
import requests

from autograd.text import utils as text_utils


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits arrays or matrices into random train and test subsets.

    Args:
        X (np.ndarray): Feature array.
        y (np.ndarray): Labels array.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (Optional[int]): Seed for the random number generator.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            The training features, test features, training labels, and test labels.

    Examples:
        >>> import cupy as np
        >>> X = np.arange(100).reshape(50, 2)
        >>> y = np.arange(50)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        >>> X_train.shape, X_test.shape
        ((40, 2), (10, 2))
    """
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    num_test = int(len(indices) * test_size)
    X_train, X_test = X[indices[num_test:]], X[indices[:num_test]]
    y_train, y_test = y[indices[num_test:]], y[indices[:num_test]]
    return X_train, X_test, y_train, y_test


def load_data(
    url: str, filename: str, max_rows: Optional[int] = None
) -> Union[str, np.ndarray]:
    """
    Load data from a file, downloading (GET request) it first if it doesn't exist.
    Automatically handles parquet and text files based on extension.

    Args:
        url (str): URL to download the file from.
        filename (str): Local path to save/load the file.
        max_rows (Optional[int]): Maximum number of rows (only applies to parquet files).

    Returns:
        Union[str, np.ndarray]:
            - For text files: the file content as a string.
            - For parquet files: a numpy array containing the data.

    Examples:
        For a parquet file:
        >>> url = "http://example.com/data.parquet"
        >>> filename = "data.parquet"
        >>> data = load_data(url, filename)
        >>> isinstance(data, np.ndarray)
        True

        For a text file:
        >>> url = "http://example.com/data.txt"
        >>> filename = "data.txt"
        >>> text = load_data(url, filename)
        >>> isinstance(text, str)
        True
    """
    # Download if file doesn't exist
    if not os.path.exists(filename):
        response = requests.get(url)
        with open(filename, "wb") as f:
            f.write(response.content)

    # Read based on file extension
    if filename.endswith(".parquet"):
        data = pq.read_table(filename).to_pandas().to_numpy()
        return data[:max_rows] if max_rows else data
    else:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()


class AbstractDataLoader(ABC):
    """
    An abstract base class for DataLoaders.
    A base interface for DataLoaders that yield batches of data for training,
    with an optional per-epoch hook.
    """

    def __init__(self, batch_size: int, shuffle: bool = True) -> None:
        """
        Args:
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the data every epoch (implementation can vary).
        """
        self.batch_size = batch_size
        self.shuffle = shuffle

    def on_epoch_start(self) -> None:
        """
        Optional hook to perform actions at the start of an epoch.
        The default implementation does nothing.
        Subclasses (e.g. for shuffling) can override this method.
        """
        pass

    def __len__(self) -> int:
        """
        Returns the number of batches per epoch if defined,
        otherwise raises an error.
        """
        raise NotImplementedError(
            "This DataLoader does not support __len__ by default."
        )

    @abstractmethod
    def __iter__(self) -> Iterator[Any]:
        """
        Should yield one batch at a time.
        The batch format can vary depending on the type of data.
        Please implement this method in the subclasses.
        """
        pass


class SimpleDataLoader(AbstractDataLoader):
    """
    A basic DataLoader for supervised tasks (e.g., classification or regression).
    It handles:
      - In-memory storage of X and y.
      - Optional shuffling at the start of each epoch.
      - Batching data into (batch_X, batch_y) pairs.

    Examples:
        >>> import cupy as np
        >>> from your_module import SimpleDataLoader  # Replace 'your_module' with your actual module name.
        >>> # Create dummy data: 100 samples, each with 10 features.
        >>> X = np.random.rand(100, 10)
        >>> y = np.random.randint(0, 2, size=(100, 1))
        >>> # Instantiate the data loader with a batch size of 32.
        >>> loader = SimpleDataLoader(X, y, batch_size=32, shuffle=True)
        >>> # Iterate over batches.
        >>> for batch_X, batch_y in loader:
        ...     print(batch_X.shape, batch_y.shape)
        (32, 10) (32, 1)
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> None:
        super().__init__(batch_size=batch_size, shuffle=shuffle)
        self.X = X
        self.y = y
        self.num_samples = len(X)
        self.indices = np.arange(self.num_samples)

    def on_epoch_start(self) -> None:
        # Shuffle the index array if needed.
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        for start in range(0, self.num_samples, self.batch_size):
            batch_indices = self.indices[start : start + self.batch_size]
            yield self.X[batch_indices], self.y[batch_indices]

    def __len__(self) -> int:
        # Compute the number of batches per epoch.
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def preprocess(self, preprocess_func) -> None:
        """
        Optionally preprocess the data using the provided function.

        Args:
            preprocess_func: A function that takes (X, y) and returns preprocessed (X, y).

        Examples:
            >>> def dummy_preprocess(X, y):
            ...     return X * 2, y  # Example: double the features.
            >>> loader.preprocess(dummy_preprocess)
        """
        self.X, self.y = preprocess_func(self.X, self.y)


class LLMDataLoader(AbstractDataLoader):
    """
    A specialized DataLoader for language modeling or next-token tasks,
    where it samples random chunks of contiguous sequences of data.
    It handles:
      - A single tokenized array (e.g. integer IDs of text).
      - Random chunk sampling: each batch picks 'batch_size' random slices
        of length (seq_len+1).
      - Optional creation of a decoder_input by prepending a special <SOS> token.
      - Optional creation of a causal mask or other masks.
      - If a finite 'steps_per_epoch' is provided, each epoch produces that number of batches.
        Otherwise, batches can be yielded infinitely.

    Examples:
        >>> import cupy as np
        >>> from your_module import LLMDataLoader  # Replace 'your_module' with your actual module name.
        >>> # Define a dummy Byte Pair Encoder (BPE) with minimal encode/decode functionality.
        >>> class DummyBPE:
        ...     def encode(self, text, allowed_special=None):
        ...         # Simple encoding: return ASCII codes of characters.
        ...         return [ord(c) for c in text]
        ...     def decode(self, tokens):
        ...         return ''.join(chr(t) for t in tokens)
        >>> bpe = DummyBPE()
        >>>
        >>> # Create dummy token data: an array of token IDs (integers).
        >>> token_data = np.random.randint(0, 100, 1000)  # 1000 tokens.
        >>>
        >>> # Instantiate the LLMDataLoader.
        >>> loader = LLMDataLoader(
        ...     data=token_data,
        ...     bpe=bpe,
        ...     batch_size=4,
        ...     seq_len=10,
        ...     steps_per_epoch=5,  # Produce 5 batches per epoch.
        ...     include_decoder_input=True,
        ...     create_padding_masks=True,
        ...     sos_token="<SOS>",
        ...     pad_token="<PAD>"
        ... )
        >>>
        >>> # Iterate over one epoch of batches.
        >>> for batch in loader:
        ...     X_chunk, dec_inp, Y_chunk, smask, tmask, causal_mask = batch
        ...     print(X_chunk.shape, Y_chunk.shape)
        (4, 10) (4, 10)
    """

    def __init__(
        self,
        data: np.ndarray,  # array of token IDs
        bpe,
        batch_size: int,
        seq_len: int,
        shuffle: bool = True,
        steps_per_epoch: Optional[int] = 1000,
        include_decoder_input: bool = True,
        create_padding_masks: bool = True,
        sos_token: Union[str, bytes] = "<SOS>",
        pad_token: Union[str, bytes] = "<PAD>",
    ) -> None:
        """
        Args:
            data (np.ndarray): Tokenized integer IDs of the entire dataset.
            bpe: BytePairEncoder or other tokenizer with .encode() / .decode().
            batch_size (int): Number of sequences per batch.
            seq_len (int): Length of each sequence (X) fed to the model.
                           (seq_len+1 tokens are sliced so that position i can predict i+1).
            shuffle (bool): Whether to randomize each epoch's sampling.
            steps_per_epoch (Optional[int]): If given, produces exactly this many batches per epoch.
                                             Otherwise, batches can be yielded infinitely.
            include_decoder_input (bool): If True, creates a separate decoder input array.
            create_padding_masks (bool): If True, creates padding masks for variable-length sequences.
            sos_token (Union[str, bytes]): The start-of-sequence token for the decoder input.
            pad_token (Union[str, bytes]): The token for padding.
        """
        super().__init__(batch_size=batch_size, shuffle=shuffle)

        self.data = np.array(data)
        self.bpe = bpe
        self.seq_len = seq_len
        self.steps_per_epoch = steps_per_epoch
        self.include_decoder_input = include_decoder_input
        self.create_padding_masks = create_padding_masks

        # For ignoring or masking out pad tokens:
        self.pad_idx = bpe.encode(pad_token, allowed_special={pad_token})[0]
        if self.include_decoder_input:
            self.sos_idx = bpe.encode(sos_token, allowed_special={sos_token})[0]
        else:
            self.sos_idx = None

        self.data_size = len(self.data)

    def on_epoch_start(self) -> None:
        """
        Called at the start of each epoch.
        For random chunking, reseeds the RNG to ensure different random offsets each epoch if shuffle is True.
        """
        if self.shuffle:
            # Reseeding ensures different random offsets each epoch.
            np.random.seed()

    def __iter__(
        self,
    ) -> Iterator[
        Tuple[
            np.ndarray,  # X_chunk: (batch_size, seq_len)
            Optional[np.ndarray],  # dec_inp: (batch_size, seq_len) or None
            np.ndarray,  # Y_chunk: (batch_size, seq_len)
            Optional[np.ndarray],  # source mask (e.g., padding mask)
            Optional[np.ndarray],  # target mask (e.g., causal + padding)
            Optional[np.ndarray],  # causal mask
        ]
    ]:
        step = 0
        # Allow infinite iteration if steps_per_epoch is None.
        while self.steps_per_epoch is None or step < self.steps_per_epoch:
            step += 1

            max_offset = self.data_size - (self.seq_len + 1)
            if max_offset < 1:
                raise ValueError(
                    f"Dataset too small ({self.data_size} tokens) for seq_len={self.seq_len + 1}"
                )

            # Randomly choose starting offsets for each sequence in the batch.
            offsets = np.random.randint(low=0, high=max_offset, size=self.batch_size)
            # Extract chunks of (seq_len+1) tokens.
            batch_chunks = [self.data[o : o + self.seq_len + 1] for o in offsets]
            # Stack to shape: (batch_size, seq_len+1)
            batch = np.stack(batch_chunks, axis=0)

            # Prepare input (X) and target (Y) by shifting the sequence.
            X_chunk = batch[:, :-1]  # (batch_size, seq_len)
            Y_chunk = batch[:, 1:]  # (batch_size, seq_len)

            # Optionally create a decoder input by prepending the SOS token.
            if self.include_decoder_input:
                dec_inp = np.zeros_like(Y_chunk)
                dec_inp[:, 0] = self.sos_idx
                dec_inp[:, 1:] = Y_chunk[:, :-1]
            else:
                dec_inp = None

            # Create a causal mask (e.g., for next-token prediction).
            causal_mask = text_utils.create_causal_mask(
                seq_len=self.seq_len, batch_size=self.batch_size
            )

            smask, tmask = None, None
            if self.create_padding_masks:
                smask = text_utils.create_padding_mask(X_chunk, self.pad_idx)
                pmask = text_utils.create_padding_mask(Y_chunk, self.pad_idx)
                tmask = pmask + causal_mask if causal_mask is not None else pmask

            yield (
                X_chunk,  # (batch_size, seq_len)
                dec_inp,  # (batch_size, seq_len) or None
                Y_chunk,  # (batch_size, seq_len)
                smask,  # source mask or None
                tmask,  # target mask or None
                causal_mask,  # causal mask or None
            )

    def __len__(self) -> int:
        if self.steps_per_epoch is None:
            raise NotImplementedError("Infinite DataLoader does not support __len__.")
        return self.steps_per_epoch
