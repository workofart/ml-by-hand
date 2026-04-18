"""
Data pipeline boundary:

1. Per-sample transforms/tokenization
- dataset / transforms

2. Batch assembly and batch-time shaping
- collate_fn

3. Training-specific logic
- trainer / training loop / model forward
"""

import csv
import os
from abc import ABC, abstractmethod
from math import prod
from typing import Any, Callable, Iterator, Optional, Sequence, Tuple, Union
from urllib.request import urlopen

from pyarrow import parquet as pq  # pyright: ignore[reportMissingImports]

from autograd.backend import Array, xp
from autograd.text import utils as text_utils


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

    Examples:
        >>> from autograd.backend import xp
        >>> X = xp.arange(100).reshape(50, 2)
        >>> y = xp.arange(50)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        >>> X_train.shape, X_test.shape
        ((40, 2), (10, 2))
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

    Args:
        url (str): URL to download the file from.
        filename (str): Local path to save/load the file.
        max_rows (Optional[int]): Maximum number of rows (only applies to parquet files).

    Returns:
        Union[str, Any]:
            - For text files: the file content as a string.
            - For parquet files: a list of row dictionaries.
            - For CSV files: a list of rows without the header row.

    Examples:
        For a parquet file:
        >>> url = "http://example.com/data.parquet"
        >>> filename = "data.parquet"
        >>> data = load_data(url, filename)
        >>> hasattr(data, "shape")
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
        with urlopen(url) as response:
            content = response.read()
        with open(filename, "wb") as f:
            f.write(content)

    # Read based on file extension
    if filename.endswith(".parquet"):
        data = pq.read_table(filename).to_pylist()
        return data[:max_rows] if max_rows else data
    if filename.endswith(".csv"):
        with open(filename, "r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        return rows[1:] if rows else rows
    else:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()


class IterableDataset(ABC):
    """
    Abstract interface for iterable datasets that yield single examples.

    Each iteration should yield one example dictionary. The exact keys depend on
    the task, but examples should contain enough information for a downstream
    `collate_fn` to build the final trainer batch.

    Examples:
        >>> class DummyIterableDataset(IterableDataset):
        ...     def __iter__(self):
        ...         yield {"tokens": xp.array([1, 2, 3], dtype=xp.int32)}
        ...     def __len__(self):
        ...         return 1
        >>> dataset = DummyIterableDataset()
        >>> next(iter(dataset))["tokens"].shape
        (3,)
    """

    def on_epoch_start(self) -> None:
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[dict[str, Any]]:
        pass

    def __len__(self) -> int:
        raise TypeError(f"object of type '{self.__class__.__name__}' has no len()")


class PairedIterableDataset(IterableDataset):
    """
    Iterates over in-memory paired `(X, y)` data one example at a time.

    Examples:
        >>> X = xp.arange(6).reshape(3, 2)
        >>> y = xp.array([0, 1, 2])
        >>> dataset = PairedIterableDataset(X, y, shuffle=False)
        >>> example = next(iter(dataset))
        >>> example["inputs"].shape, int(example["targets"])
        ((2,), 0)
    """

    def __init__(
        self, X: Sequence[Any], y: Sequence[Any], shuffle: bool = True
    ) -> None:
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of examples")
        self.X = X
        self.y = y
        self.shuffle = shuffle
        self.num_samples = len(X)
        self.indices = xp.arange(self.num_samples)

    def on_epoch_start(self) -> None:
        if self.shuffle:
            self.indices = xp.random.permutation(self.num_samples)

    def __iter__(self) -> Iterator[dict[str, Array]]:
        for sample_idx in self.indices:
            idx = int(sample_idx)
            yield {"inputs": self.X[idx], "targets": self.y[idx]}

    def __len__(self) -> int:
        return self.num_samples


class TransformDataset(IterableDataset):
    """
    Wraps another dataset and applies per-example transforms during iteration.

    Examples:
        >>> dataset = PairedIterableDataset(
        ...     xp.arange(6).reshape(3, 2),
        ...     xp.array([0, 1, 2]),
        ...     shuffle=False,
        ... )
        >>> wrapped = TransformDataset(
        ...     dataset,
        ...     target_transform=lambda y: xp.array(int(y == 1), dtype=xp.int32),
        ... )
        >>> int(next(iter(wrapped))["targets"])
        0
    """

    def __init__(
        self,
        dataset: IterableDataset,
        transform: Optional[Callable[[Array], Array]] = None,
        target_transform: Optional[Callable[[Array], Array]] = None,
    ) -> None:
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def on_epoch_start(self) -> None:
        self.dataset.on_epoch_start()

    def __iter__(self) -> Iterator[dict[str, Array]]:
        for example in self.dataset:
            transformed_example = dict(example)
            if self.transform is not None:
                transformed_example["inputs"] = self.transform(
                    xp.array(example["inputs"])
                )
            if self.target_transform is not None:
                transformed_example["targets"] = self.target_transform(
                    xp.array(example["targets"])
                )
            yield transformed_example

    def __len__(self) -> int:
        return len(self.dataset)


class TokenSequenceDataset(IterableDataset):
    """
    Iterates over LM token sequences with per-token loss masks.

    This class supports two modes:
    - finite examples stored in memory
    - infinite random windows sampled from one flat token stream

    Examples:
        >>> dataset = TokenSequenceDataset(
        ...     token_sequences=[xp.array([10, 11, 20], dtype=xp.int32)],
        ...     loss_masks=[xp.array([0, 0, 1], dtype=xp.int32)],
        ...     shuffle=False,
        ... )
        >>> example = next(iter(dataset))
        >>> example["tokens"].tolist(), example["loss_mask"].tolist()
        ([10, 11, 20], [0, 0, 1])
        >>> random_window_dataset = TokenSequenceDataset(
        ...     data=xp.arange(32, dtype=xp.int32),
        ...     seq_len=4,
        ...     shuffle=False,
        ...     random_window=True,
        ... )
        >>> next(iter(random_window_dataset))["tokens"].shape
        (5,)
    """

    def __init__(
        self,
        token_sequences: Optional[Sequence[Array]] = None,
        loss_masks: Optional[Sequence[Array]] = None,
        *,
        data: Optional[Array] = None,
        seq_len: Optional[int] = None,
        shuffle: bool = True,
        random_window: bool = False,
    ) -> None:
        self.shuffle = shuffle
        self.random_window = random_window

        if self.random_window:
            if data is None or seq_len is None:
                raise ValueError("random_window datasets require both data and seq_len")
            if token_sequences is not None or loss_masks is not None:
                raise ValueError(
                    "random_window datasets do not accept token_sequences or loss_masks"
                )
            self.data = xp.array(data, dtype=xp.int32)
            self.seq_len = seq_len
            self.data_size = len(self.data)
            return

        if token_sequences is None:
            raise ValueError("token_sequences are required when random_window=False")
        if loss_masks is None:
            loss_masks = [
                xp.ones((len(tokens),), dtype=xp.int32) for tokens in token_sequences
            ]
        if len(token_sequences) != len(loss_masks):
            raise ValueError(
                "token_sequences and loss_masks must contain the same number of examples"
            )
        self.examples = []
        for tokens, loss_mask in zip(token_sequences, loss_masks):
            tokens_array = xp.array(tokens, dtype=xp.int32)
            loss_mask_array = xp.array(loss_mask, dtype=xp.int32)
            if len(tokens_array) != len(loss_mask_array):
                raise ValueError(
                    "token sequence and loss mask must have the same length"
                )
            self.examples.append({"tokens": tokens_array, "loss_mask": loss_mask_array})
        self.num_examples = len(self.examples)
        self.indices = xp.arange(self.num_examples)

    def on_epoch_start(self) -> None:
        if self.random_window:
            if self.shuffle:
                xp.random.seed(int.from_bytes(os.urandom(4), "big"))
            return
        if self.shuffle:
            self.indices = xp.random.permutation(self.num_examples)

    def __iter__(self) -> Iterator[dict[str, Array]]:
        if self.random_window:
            max_offset = self.data_size - (self.seq_len + 1)
            if max_offset < 1:
                raise ValueError(
                    f"Dataset too small ({self.data_size} tokens) for seq_len={self.seq_len + 1}"
                )
            while True:
                offset = int(xp.random.randint(0, max_offset, (), dtype=xp.int32))
                tokens = self.data[offset : offset + self.seq_len + 1]
                yield {
                    "tokens": tokens,
                    "loss_mask": xp.ones(tokens.shape, dtype=xp.int32),
                }
            return

        for sample_idx in self.indices:
            example = self.examples[int(sample_idx)]
            yield {
                "tokens": example["tokens"],
                "loss_mask": example["loss_mask"],
            }

    def __len__(self) -> int:
        if self.random_window:
            raise TypeError(f"object of type '{self.__class__.__name__}' has no len()")
        return self.num_examples


def openai_chat_to_prompt_completion(chat_example: dict[str, Any]) -> dict[str, str]:
    """
    Converts one OpenAI chat-format SFT record into prompt/completion text.

    Examples:
        >>> example = openai_chat_to_prompt_completion(
        ...     {
        ...         "messages": [
        ...             {"role": "user", "content": "ABC"},
        ...             {"role": "assistant", "content": "DE"},
        ...         ]
        ...     },
        ... )
        >>> example["prompt_text"], example["completion_text"]
        ('ABC', 'DE')
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

    Examples:
        >>> class DummyBPE:
        ...     def encode(self, text, allowed_special=None):
        ...         return [ord(char) for char in text]
        >>> example = tokenize_prompt_completion(
        ...     {"prompt_text": "ABC", "completion_text": "DE"},
        ...     DummyBPE(),
        ... )
        >>> example["tokens"].tolist(), example["loss_mask"].tolist()
        ([65, 66, 67, 68, 69], [0, 0, 0, 1, 1])
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


def pack_tokens(tokens: Array, max_tokens: int, pad_idx: int) -> Array:
    """
    Left-truncates and right-pads one token sequence to `max_tokens`.

    Examples:
        >>> packed = pack_tokens(
        ...     tokens=xp.array([10, 11, 12, 20, 21]),
        ...     max_tokens=6,
        ...     pad_idx=0,
        ... )
        >>> packed.shape
        (6,)
    """
    if len(tokens) > max_tokens:
        tokens = tokens[-max_tokens:]
    if len(tokens) < max_tokens:
        pad_width = max_tokens - len(tokens)
        tokens = xp.concatenate(
            [tokens, xp.full((pad_width,), pad_idx, dtype=xp.int32)],
            axis=0,
        )
    return tokens


class Collator(ABC):
    """
    Abstract interface for collators that turn example lists into batches.
    """

    @abstractmethod
    def __call__(self, examples: Sequence[dict[str, Array]]) -> Any:
        pass

    def batch_token_count(self, batch: Any) -> Optional[int]:
        return None


class PairedCollator(Collator):
    """
    Batches paired examples from `PairedIterableDataset`.

    Examples:
        >>> collator = PairedCollator()
        >>> batch_X, batch_y = collator(
        ...     [
        ...         {"inputs": xp.array([1, 2]), "targets": xp.array(0)},
        ...         {"inputs": xp.array([3, 4]), "targets": xp.array(1)},
        ...     ]
        ... )
        >>> batch_X.shape, batch_y.shape
        ((2, 2), (2,))
    """

    def __call__(self, examples: Sequence[dict[str, Array]]) -> Tuple[Array, Array]:
        batch_X = xp.stack(
            [xp.array(example["inputs"]) for example in examples], axis=0
        )
        batch_y = xp.stack(
            [xp.array(example["targets"]) for example in examples],
            axis=0,
        )
        return batch_X, batch_y


class OneHotCollator(Collator):
    """
    Batches token-id examples and materializes one-hot inputs.

    This keeps token IDs in memory and expands them to one-hot features at
    batch time.

    Examples:
        >>> collator = OneHotCollator(num_classes=4)
        >>> batch_X, batch_y = collator(
        ...     [
        ...         {"inputs": xp.array([0, 2]), "targets": xp.array(1)},
        ...         {"inputs": xp.array([1, 3]), "targets": xp.array(0)},
        ...     ]
        ... )
        >>> batch_X.shape, batch_y.shape
        ((2, 2, 4), (2,))
    """

    def __init__(self, num_classes: int, dtype: Array = xp.float32) -> None:
        self.num_classes = num_classes
        self.dtype = dtype

    def __call__(self, examples: Sequence[dict[str, Array]]) -> Tuple[Array, Array]:
        # TODO: replace batch-time one-hot with direct token-id model inputs.
        eye = xp.eye(self.num_classes, dtype=self.dtype)
        batch_tokens = xp.stack(
            [xp.array(example["inputs"], dtype=xp.int32) for example in examples],
            axis=0,
        )
        batch_y = xp.stack(
            [xp.array(example["targets"]) for example in examples],
            axis=0,
        )
        return eye[batch_tokens], batch_y


class LanguageModelingCollator(Collator):
    """
    Builds the full LM trainer batch tuple from token sequences and loss masks.

    Examples:
        >>> collator = LanguageModelingCollator(
        ...     max_tokens=4,
        ...     pad_idx=0,
        ...     sos_idx=1,
        ...     include_decoder_input=True,
        ...     create_padding_masks=False,
        ... )
        >>> batch = collator(
        ...     [
        ...         {
        ...             "tokens": xp.array([1, 2, 3, 4], dtype=xp.int32),
        ...             "loss_mask": xp.array([1, 1, 1, 1], dtype=xp.int32),
        ...         }
        ...     ],
        ... )
        >>> len(batch), batch[0].shape, batch[2].shape
        (6, (1, 3), (1, 3))
    """

    def __init__(
        self,
        max_tokens: int,
        pad_idx: int,
        sos_idx: Optional[int] = None,
        include_decoder_input: bool = True,
        create_padding_masks: bool = True,
        packer: Optional[Callable[[Array, int, int], Array]] = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.include_decoder_input = include_decoder_input
        self.create_padding_masks = create_padding_masks
        self.packer = packer or pack_tokens

    def __call__(
        self, examples: Sequence[dict[str, Array]]
    ) -> Tuple[
        Array, Optional[Array], Array, Optional[Array], Optional[Array], Optional[Array]
    ]:
        batch_inputs = []
        batch_targets = []

        for example in examples:
            tokens = xp.array(example["tokens"], dtype=xp.int32)
            loss_mask = xp.array(
                example.get(
                    "loss_mask",
                    xp.ones(tokens.shape, dtype=xp.int32),
                ),
                dtype=xp.int32,
            )
            if len(tokens) != len(loss_mask):
                raise ValueError("tokens and loss_mask must have the same length")

            packed_tokens = self.packer(tokens, self.max_tokens, self.pad_idx)
            packed_loss_mask = self.packer(loss_mask, self.max_tokens, 0)
            input_tokens = packed_tokens[:-1]
            target_tokens = xp.array(packed_tokens[1:])
            target_loss_mask = packed_loss_mask[1:]
            target_tokens[target_loss_mask == 0] = self.pad_idx

            batch_inputs.append(input_tokens)
            batch_targets.append(target_tokens)

        X_chunk = xp.stack(batch_inputs, axis=0)
        Y_chunk = xp.stack(batch_targets, axis=0)
        batch_size = X_chunk.shape[0]
        seq_len = X_chunk.shape[1]

        if self.include_decoder_input:
            assert self.sos_idx is not None
            sos_column = xp.full((batch_size, 1), self.sos_idx, dtype=Y_chunk.dtype)
            dec_inp = xp.concatenate([sos_column, Y_chunk[:, :-1]], axis=1)
        else:
            dec_inp = None

        causal_mask = text_utils.create_causal_mask(
            seq_len=seq_len, batch_size=batch_size
        )

        smask, tmask = None, None
        if self.create_padding_masks:
            smask = text_utils.create_padding_mask(X_chunk, self.pad_idx)
            pmask = text_utils.create_padding_mask(Y_chunk, self.pad_idx)
            tmask = pmask + causal_mask if causal_mask is not None else pmask

        return (
            X_chunk,
            dec_inp,
            Y_chunk,
            smask,
            tmask,
            causal_mask,
        )

    def batch_token_count(self, batch: Any) -> Optional[int]:
        X_chunk, _, _, _, _, _ = batch
        return int(prod(X_chunk.shape))


class DataLoader:
    """
    Thin generic data loader that batches examples from a dataset and applies a collator.

    Examples:
        >>> X = xp.arange(6).reshape(3, 2)
        >>> y = xp.array([0, 1, 2])
        >>> dataset = PairedIterableDataset(X, y, shuffle=False)
        >>> loader = DataLoader(dataset, batch_size=2, collate_fn=PairedCollator())
        >>> batch_X, batch_y = next(iter(loader))
        >>> batch_X.shape, batch_y.shape
        ((2, 2), (2,))
    """

    def __init__(
        self,
        dataset: IterableDataset,
        batch_size: int,
        collate_fn: Optional[Collator] = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.pad_idx = getattr(collate_fn, "pad_idx", None)

    def on_epoch_start(self) -> None:
        self.dataset.on_epoch_start()

    def batch_token_count(self, batch: Any) -> Optional[int]:
        if self.collate_fn is None:
            return None
        return self.collate_fn.batch_token_count(batch)

    def __iter__(self) -> Iterator[Any]:
        batch_examples = []
        for example in self.dataset:
            batch_examples.append(example)
            if len(batch_examples) < self.batch_size:
                continue

            yield self.collate_fn(batch_examples) if self.collate_fn else batch_examples
            batch_examples = []

        if batch_examples:
            yield self.collate_fn(batch_examples) if self.collate_fn else batch_examples

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
