import numpy as np
from typing import Union
import os
import requests
import pyarrow.parquet as pq
from autograd.text import utils as text_utils
from autograd.tensor import Tensor


def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    num_samples = len(X)
    num_test = int(num_samples * test_size)
    indices = np.random.permutation(num_samples)
    X_train, X_test = X[indices[num_test:]], X[indices[:num_test]]
    y_train, y_test = y[indices[num_test:]], y[indices[:num_test]]
    return X_train, X_test, y_train, y_test


def load_data(url: str, filename: str, max_rows: int = None) -> Union[str, np.ndarray]:
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


class DataLoader:
    """
    A simple data loader pipeline for generating batched data (X, y) plus
    precomputed masks (source_mask, target_mask). Optionally shuffles data every epoch.
    """

    def __init__(
        self,
        data: np.ndarray,
        vocab: dict,
        batch_size: int,
        seq_len: int,
        shuffle: bool = True,
        pad_idx: int = 0,
    ):
        """
        Args:
            data (np.ndarray): Tokenized data (list of token IDs or strings).
            vocab (dict): A mapping from token -> ID (used in text_utils).
            batch_size (int): Number of sequences in each batch.
            seq_len (int): Length of each sequence.
            shuffle (bool): Whether to shuffle row order at the start of each epoch
                            (i.e., shuffle contiguous chunks).
            pad_idx (int): Index to use for padding tokens if needed.
        """
        self.data = data
        self.vocab = vocab
        self.int2token = {v: k for k, v in self.vocab.items()}
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.pad_idx = pad_idx

        # We'll store how many batches we can produce from epoch to epoch
        self.num_batches = 0

        # We keep X, y, source_masks, target_masks in memory after each epoch creation
        self.batches_X = []
        self.batches_y = []
        self.source_masks = []
        self.target_masks = []

    def on_epoch_start(self):
        """
        Perform any epoch-begin tasks, e.g. row-level shuffling for contiguous sampling,
        then create the new epoch's batches.
        """
        # Just clear out previously cached data
        self.batches_X.clear()
        self.batches_y.clear()
        self.source_masks.clear()
        self.target_masks.clear()

        self._create_batches()

    def _create_batches(self):
        """
        Contiguously chunk the data into shape (batch_size, -1).
        Then partition each row into seq_len segments.

        If shuffle=True, we shuffle at the row level after reshaping, so each
        row is a different random chunk of text from the dataset (though still contiguous
        within the row).
        """
        data_length = len(self.data)

        # We must discard leftover tokens that don't fit exactly into (batch_size * seq_len).
        # We'll do it so each row has an integer multiple of seq_len.
        num_tokens_to_keep = (data_length // (self.batch_size * self.seq_len)) * (
            self.batch_size * self.seq_len
        )
        truncated_data = self.data[:num_tokens_to_keep]  # discard excess

        # Reshape to (batch_size, -1)
        reshaped = truncated_data.reshape(self.batch_size, -1)

        # Optionally shuffle the rows here (each row is a contiguous slice of text).
        if self.shuffle:
            np.random.shuffle(reshaped)

        # Now figure out how many seq_len blocks per row
        width = reshaped.shape[1]  # total tokens per row
        num_steps = width // self.seq_len
        self.num_batches = num_steps

        for step in range(num_steps):
            X_chunk = reshaped[:, step * self.seq_len : (step + 1) * self.seq_len]
            Y_chunk = reshaped[
                :, step * self.seq_len : (step + 1) * self.seq_len
            ]  # same if next-token

            # If your data is still string tokens, we convert them to indices.
            # If it's already integer IDs, you could skip this step.
            x = text_utils.token_batch_to_indices(X_chunk, self.vocab)
            y = text_utils.token_batch_to_indices(Y_chunk, self.vocab)

            # Create the masks
            smask = Tensor(
                text_utils.create_padding_mask(x, pad_idx=self.pad_idx),
                requires_grad=False,
            )
            pmask = text_utils.create_padding_mask(y, pad_idx=self.pad_idx)
            cmask = text_utils.create_causal_mask(self.seq_len, self.batch_size)
            tmask = Tensor(pmask + cmask, requires_grad=False)

            self.batches_X.append(x)
            self.batches_y.append(y)
            self.source_masks.append(smask)
            self.target_masks.append(tmask)

    def __len__(self):
        """
        Number of batches per epoch.
        """
        return self.num_batches

    def __iter__(self):
        """
        Iterate over precomputed batches (X, y, source_mask, target_mask).
        """
        for i in range(self.num_batches):
            yield (
                self.batches_X[i],
                self.batches_y[i],
                self.source_masks[i],
                self.target_masks[i],
            )

    def sample_random_sequence(
        self,
        seq_len: int,
        as_tokens: bool = False,
        return_dict: bool = False,
    ):
        """
        Sample a random contiguous chunk of length `seq_len` from `self.data`.
        Optionally convert it from integer IDs -> tokens (using the reverse of self.vocab).
        Optionally return it in a dict form, similar to `create_one_batch`.

        Args:
            seq_len (int): Number of tokens to slice.
            as_tokens (bool): If True, return the sampled sequence as a list of tokens
                              (rather than integer IDs).
            return_dict (bool): If True, return {"inputs": X, "labels": y} instead of (X, y).

        Returns:
            If return_dict = False:
                (X, y) where both are np.array of shape (1, seq_len).
            If return_dict = True:
                {"inputs": X, "labels": y} dict.
        """
        data_length = len(self.data)
        if data_length < seq_len + 1:
            raise ValueError(
                f"Data too short (length={data_length}) for sequence length={seq_len}."
            )

        # Pick a random start index that allows for X and y to both be length seq_len.
        start_idx = np.random.randint(0, data_length - seq_len - 1)

        # X is data[start_idx : start_idx + seq_len]
        # y is data[start_idx+1 : start_idx + seq_len + 1]
        X = self.data[start_idx : start_idx + seq_len]
        y = self.data[start_idx + 1 : start_idx + seq_len + 1]

        # Convert IDs -> tokens if desired
        if as_tokens:
            X = np.array(X)  # shape (seq_len, )
            y = np.array(y)  # shape (seq_len, )

        # Shape them into (seq_len, ) if still numeric
        # or a list of length seq_len if as_tokens=True
        if not as_tokens:
            # We'll assume vocab is {str -> int}, so we need a reverse mapping {int -> str}.
            X = [self.vocab[x_id] for x_id in X]
            y = [self.vocab[y_id] for y_id in y]

        if return_dict:
            return {"inputs": X, "labels": y}
        return (X, y)
