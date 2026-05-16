import csv
import os
from typing import Any, Optional, Sequence, Union, cast
from urllib.request import urlopen

import numpy as np
from pyarrow import parquet as pq  # pyright: ignore[reportMissingImports]

from autograd.data.dataset import PairedMapDataset


def train_test_split(
    *arrays,
    test_size: float = 0.1,
    shuffle: bool = True,
    random_state: Optional[int] = None,
) -> list:
    """Split arrays into train and test subsets.

    Returns a flat list: [arr0_train, arr0_test, arr1_train, arr1_test, ...].
    """
    if not arrays:
        raise ValueError("need at least one array to split")
    n = len(arrays[0])
    split = int(n * test_size)
    result = []
    if shuffle:
        idx = np.random.RandomState(random_state).permutation(n)
        for arr in arrays:
            arr = np.asarray(arr)
            result.append(arr[idx[split:]])
            result.append(arr[idx[:split]])
    else:
        for arr in arrays:
            arr = np.asarray(arr)
            result.append(arr[split:])
            result.append(arr[:split])
    return result


def load_data(
    url: str, filename: str, max_rows: Optional[int] = None
) -> Union[str, list[dict[str, Any]], list[list[str]]]:
    """
    Load data from a file, downloading (GET request) it first if it doesn't exist.
    Automatically handles parquet and text files based on extension.
    """
    if not os.path.exists(filename):
        with urlopen(url) as response:
            content = response.read()
        parent_dir = os.path.dirname(filename)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(filename, "wb") as f:
            f.write(content)

    if filename.endswith(".parquet"):
        data = pq.read_table(filename).to_pylist()
        return data[:max_rows] if max_rows else data
    if filename.endswith(".csv"):
        with open(filename, "r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        return rows[1:] if rows else rows

    with open(filename, "r", encoding="utf-8") as f:
        return f.read()


def load_parquet_rows(
    url: str, filename: str, max_rows: Optional[int] = None
) -> list[dict[str, Any]]:
    if not filename.endswith(".parquet"):
        raise ValueError(f"filename must end with '.parquet', got {filename!r}")
    data = load_data(url, filename, max_rows=max_rows)
    if not isinstance(data, list) or any(not isinstance(row, dict) for row in data):
        raise TypeError("parquet data must contain row dictionaries")
    return cast(list[dict[str, Any]], data)


def build_seq2seq_dataset_from_text_pairs(
    text_pairs: Sequence[tuple[str, str]],
    bpe,
    *,
    target_suffix: str = "",
) -> PairedMapDataset:
    input_sequences = []
    label_sequences = []

    for source_text, target_text in text_pairs:
        source_tokens = np.array(bpe.encode(source_text), dtype=np.int32)
        target_tokens = np.array(bpe.encode(target_text), dtype=np.int32)
        if target_suffix:
            target_tokens = np.concatenate(
                [
                    target_tokens,
                    np.array(bpe.encode(target_suffix), dtype=np.int32),
                ],
                axis=0,
            )
        if len(source_tokens) == 0:
            raise ValueError("source text must encode to at least one token")
        if len(target_tokens) == 0:
            raise ValueError("target text must encode to at least one token")
        input_sequences.append(source_tokens)
        label_sequences.append(target_tokens)

    if not input_sequences:
        raise ValueError("text_pairs must contain at least one example")

    return PairedMapDataset(
        input_sequences,
        label_sequences,
        input_key="input_ids",
        target_key="labels",
    )
