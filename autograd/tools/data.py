import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Optional, Tuple, Union

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
    if random_state is not None:
        np.random.seed(random_state)
    num_samples = len(X)
    num_test = int(num_samples * test_size)
    indices = np.random.permutation(num_samples)
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
        url: URL to download the file from
        filename: Local path to save/load the file
        max_rows: Maximum number of rows (only applies to parquet files)

    Returns:
        str for text files, numpy array for parquet files
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
        with open(filename, "r") as f:
            return f.read()


class AbstractDataLoader(ABC):
    """
    A base interface for DataLoaders that yield batches of data for training.
    """

    def __init__(self, batch_size: int, shuffle: bool = True) -> None:
        """
        Args:
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the data every epoch (implementation can vary).
        """
        self.batch_size = batch_size
        self.shuffle = shuffle

    @abstractmethod
    def on_epoch_start(self) -> None:
        """
        Hook that can be called at the start of each epoch.
        (e.g., for shuffling or resetting any internal state).
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of batches per epoch.
        For instance, if we have N data points and batch_size = B,
        this might return N // B (or ceil of it).
        """
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Any]:
        """
        Should yield one batch at a time.
        The batch format can vary depending on the type of data
        (e.g. (X, y) for classification, or (X, y, masks) for text tasks).
        """
        pass


class SimpleDataLoader(AbstractDataLoader):
    """
    A basic DataLoader for supervised tasks, e.g. classification or regression.
    It handles:
      - storing X and y in memory
      - optional shuffling each epoch
      - batching by batch_size
      - yielding (batch_X, batch_y) each iteration
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
        self._index_array = np.arange(self.num_samples)

        # We'll compute how many steps per epoch (rounding down by default)
        self.num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size

    def on_epoch_start(self) -> None:
        """Shuffle the index array if needed."""
        if self.shuffle:
            np.random.shuffle(self._index_array)

    def preprocess(self, preprocess_func) -> None:
        """Preprocess the data if needed."""
        self.X, self.y = preprocess_func(self.X, self.y)

    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        return self.num_batches

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Yields (batch_X, batch_y).
        """
        for batch_idx in range(self.num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, self.num_samples)
            indices = self._index_array[start_idx:end_idx]

            batch_X = self.X[indices]
            batch_y = self.y[indices]
            yield batch_X, batch_y


class LLMDataLoader(AbstractDataLoader):
    """
    A specialized DataLoader for language modeling or sequence-to-sequence tasks.
    It handles:
      - A single tokenized array (e.g. integer IDs of text).
      - Contiguous chunking into (X, y) (with optional shifting).
      - Optional creation of decoder_input by prepending a special <SOS> token.
      - Mask creation: source_mask, target_mask, etc.
      - Shuffling at row-level if desired (if we reshape to (batch_size, -1)).
    """

    def __init__(
        self,
        data: np.ndarray,
        vocab: Dict[Any, Any],
        batch_size: int,
        seq_len: int,
        shuffle: bool = True,
        include_decoder_input: bool = True,
        sos_token: Union[str, bytes] = b"<SOS>",
        pad_token: Union[str, bytes] = b"<PAD>",
    ) -> None:
        """
        Args:
            data (np.ndarray): Tokenized and encoded data (list of token IDs)
            vocab (dict): A mapping from token -> ID
            batch_size (int): Number of sequences in each batch.
            seq_len (int): Length of each sequence.
            shuffle (bool): Whether to shuffle row order at the start of each epoch (i.e., shuffle contiguous chunks).
            pad_token (str, bytes): The padding token to use.
            include_decoder_input (bool): Whether to include decoder input.
            sos_token (str, bytes): The start-of-sequence token to use.
        """
        super().__init__(batch_size=batch_size, shuffle=shuffle)
        self.data = data
        self.vocab = vocab
        self.seq_len = seq_len
        self.pad_idx = vocab[pad_token]
        self.include_decoder_input = include_decoder_input

        # If we need an <SOS> token, ensure it's in vocab
        if include_decoder_input:
            if sos_token not in vocab:
                raise ValueError(
                    f"SOS token {sos_token} not found in vocab. "
                    "Either add it or disable include_decoder_input."
                )
            self.sos_idx = vocab[sos_token]
        else:
            self.sos_idx = None

        # We'll prepare caches for each epoch
        self.batches_X = []
        self.batches_y = []
        self.batches_decoder_inp = []
        self.source_masks = []
        self.target_masks = []
        self.causal_masks = []

        self.num_batches = 0

    def on_epoch_start(self) -> None:
        """
        Called at the start of each epoch: shuffle at row level (if desired) after reshaping,
        then create contiguous chunks for each batch.
        """
        self._clear_cache()
        self._create_batches()

    def _clear_cache(self):
        self.batches_X.clear()
        self.batches_y.clear()
        self.batches_decoder_inp.clear()
        self.source_masks.clear()
        self.target_masks.clear()
        self.causal_masks.clear()

    def _create_batches(self) -> None:
        """
        Reshape data into (batch_size, -1), shuffle row order if needed,
        then chunk each row by seq_len.
        """
        data_length = len(self.data)
        total_tokens_per_epoch = (data_length // (self.batch_size * self.seq_len)) * (
            self.batch_size * self.seq_len
        )
        truncated_data = self.data[:total_tokens_per_epoch]

        # Reshape to (batch_size, -1)
        reshaped = truncated_data.reshape(self.batch_size, -1)

        if self.shuffle:
            np.random.shuffle(reshaped)

        width = reshaped.shape[1]
        num_steps = width // self.seq_len
        self.num_batches = num_steps

        for step in range(num_steps):
            X_chunk = reshaped[:, step * self.seq_len : (step + 1) * self.seq_len]
            # Typically, for a next-token LM, y is the same as X but shifted by 1 token in time.
            Y_chunk = X_chunk  # or some variant if you want a different target.

            if self.include_decoder_input:
                dec_inp = np.zeros_like(Y_chunk)
                dec_inp[:, 0] = self.sos_idx
                dec_inp[:, 1:] = Y_chunk[:, :-1]
            else:
                dec_inp = None

            # Build masks (example placeholders -- replace with your own calls)
            smask = text_utils.create_padding_mask(
                X_chunk, self.pad_idx
            )  # shape e.g. [batch_size, 1, 1, seq_len]
            cmask = text_utils.create_causal_mask(
                seq_len=self.seq_len, batch_size=self.batch_size
            )
            pmask = text_utils.create_padding_mask(Y_chunk, self.pad_idx)
            tmask = pmask + cmask

            self.batches_X.append(X_chunk)
            self.batches_y.append(Y_chunk)
            self.batches_decoder_inp.append(dec_inp)
            self.source_masks.append(smask)
            self.target_masks.append(tmask)
            self.causal_masks.append(cmask)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(
        self,
    ) -> Iterator[
        Tuple[
            np.ndarray,
            Optional[np.ndarray],
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
    ]:
        """
        Yields (X, decoder_inp, y, source_mask, target_mask, causal_mask).
        """
        for i in range(self.num_batches):
            yield (
                self.batches_X[i],
                self.batches_decoder_inp[i],
                self.batches_y[i],
                self.source_masks[i],
                self.target_masks[i],
                self.causal_masks[i],
            )
