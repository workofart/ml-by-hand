import csv
import os
from typing import Any, Optional, Sequence, Tuple, Union, cast
from urllib.request import urlopen

from pyarrow import parquet as pq  # pyright: ignore[reportMissingImports]

from autograd.backend import Array, xp
from autograd.data.dataset import PairedMapDataset


def train_test_split(
    X: Array,
    y: Array,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
) -> Tuple[Array, Array, Array, Array]:
    """
    Splits arrays or matrices into random train and test subsets.

    Args:
        X (Array): Feature array.
        y (Array): Labels array.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (Optional[int]): Seed for the random number generator.

    Returns:
        Tuple[Array, Array, Array, Array]:
            The training features, test features, training labels, and test labels.
    """
    if random_state is not None:
        xp.random.seed(random_state)
    indices = xp.random.permutation(X.shape[0])

    num_test = int(len(indices) * test_size)
    X_train, X_test = X[indices[num_test:]], X[indices[:num_test]]
    y_train, y_test = y[indices[num_test:]], y[indices[:num_test]]
    return X_train, X_test, y_train, y_test


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


def openai_chat_to_prompt_completion(chat_example: dict[str, Any]) -> dict[str, str]:
    """
    Converts one OpenAI chat-format SFT record into prompt/completion text.
    """
    messages = chat_example.get("messages")
    if not messages:
        raise ValueError("chat example must contain at least one message")
    if messages[-1].get("role") != "assistant":
        raise ValueError("chat example must end with an assistant message")

    prompt_parts = []
    for message in messages[:-1]:
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("message content must be a string")
        prompt_parts.append(content)

    response_content = messages[-1].get("content")
    if not isinstance(response_content, str):
        raise ValueError("message content must be a string")

    return {
        "prompt_text": "".join(prompt_parts),
        "completion_text": response_content,
    }


def tokenize_prompt_completion(
    prompt_completion_example: dict[str, str], bpe
) -> dict[str, Array]:
    """
    Tokenizes one prompt/completion example into LM tokens and a loss mask.
    """
    prompt_tokens = xp.array(
        bpe.encode(prompt_completion_example["prompt_text"]),
        dtype=xp.int32,
    )
    completion_tokens = xp.array(
        bpe.encode(prompt_completion_example["completion_text"]),
        dtype=xp.int32,
    )
    if len(completion_tokens) == 0:
        raise ValueError("assistant completion must contain at least one token")
    return {
        "tokens": xp.concatenate([prompt_tokens, completion_tokens], axis=0),
        "loss_mask": xp.concatenate(
            [
                xp.zeros(prompt_tokens.shape, dtype=xp.int32),
                xp.ones(completion_tokens.shape, dtype=xp.int32),
            ],
            axis=0,
        ),
    }


def build_seq2seq_dataset_from_text_pairs(
    text_pairs: Sequence[tuple[str, str]],
    bpe,
    *,
    target_suffix: str = "",
) -> PairedMapDataset:
    input_sequences = []
    label_sequences = []

    for source_text, target_text in text_pairs:
        source_tokens = xp.array(bpe.encode(source_text), dtype=xp.int32)
        target_tokens = xp.array(bpe.encode(target_text), dtype=xp.int32)
        if target_suffix:
            target_tokens = xp.concatenate(
                [
                    target_tokens,
                    xp.array(bpe.encode(target_suffix), dtype=xp.int32),
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
        dtype=xp.int32,
    )
