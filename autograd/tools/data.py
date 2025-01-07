import numpy as np
from typing import Union
import os
import requests
import pyarrow.parquet as pq


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
