from abc import ABC, abstractmethod
from typing import Any, Callable, Iterator, Literal, Optional, Sequence

import numpy as np

from autograd.backend import Array, xp
from autograd.data.types import TokenWindowExample


class IterableDataset(ABC):
    """
    Abstract interface for iterable datasets that yield training records.

    By default, each iteration yields one example dictionary.

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
    def __iter__(self) -> Iterator[Any]:
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
        ...     transform=lambda example: {
        ...         "inputs": example["inputs"],
        ...         "targets": xp.array(int(example["targets"] == 1), dtype=xp.int32),
        ...     },
        ... )
        >>> int(next(iter(wrapped))["targets"])
        0
    """

    def __init__(
        self,
        dataset: IterableDataset,
        transform: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        self.dataset = dataset
        self.transform = transform

    def on_epoch_start(self) -> None:
        self.dataset.on_epoch_start()

    def __iter__(self) -> Iterator[Any]:
        for example in self.dataset:
            if self.transform is None:
                yield example
            else:
                yield self.transform(example)

    def __len__(self) -> int:
        return len(self.dataset)


class Seq2SeqDataset(IterableDataset):
    """
    Iterates over in-memory encoder-decoder token pairs one example at a time.

    Examples:
        >>> dataset = Seq2SeqDataset(
        ...     input_sequences=[xp.array([1, 2], dtype=xp.int32)],
        ...     label_sequences=[xp.array([3, 4], dtype=xp.int32)],
        ...     shuffle=False,
        ... )
        >>> example = next(iter(dataset))
        >>> example["input_ids"].tolist(), example["labels"].tolist()
        ([1, 2], [3, 4])
    """

    def __init__(
        self,
        input_sequences: Sequence[Array],
        label_sequences: Sequence[Array],
        shuffle: bool = True,
    ) -> None:
        if len(input_sequences) != len(label_sequences):
            raise ValueError(
                "input_sequences and label_sequences must contain the same number of examples"
            )

        self.examples = []
        for input_ids, labels in zip(input_sequences, label_sequences):
            input_array = xp.array(input_ids, dtype=xp.int32)
            label_array = xp.array(labels, dtype=xp.int32)
            self.examples.append(
                {
                    "input_ids": input_array,
                    "labels": label_array,
                }
            )

        self.shuffle = shuffle
        self.num_examples = len(self.examples)
        self.indices = xp.arange(self.num_examples)

    def on_epoch_start(self) -> None:
        if self.shuffle:
            self.indices = xp.random.permutation(self.num_examples)

    def __iter__(self) -> Iterator[dict[str, Array]]:
        for sample_idx in self.indices:
            example = self.examples[int(sample_idx)]
            yield {
                "input_ids": example["input_ids"],
                "labels": example["labels"],
            }

    def __len__(self) -> int:
        return self.num_examples


class TokenSequenceDataset(IterableDataset):
    """
    Iterates over LM token sequences with per-token loss masks.

    Examples:
        >>> dataset = TokenSequenceDataset(
        ...     token_sequences=[xp.array([10, 11, 20], dtype=xp.int32)],
        ...     loss_masks=[xp.array([0, 0, 1], dtype=xp.int32)],
        ...     shuffle=False,
        ... )
        >>> example = next(iter(dataset))
        >>> example["tokens"].tolist(), example["loss_mask"].tolist()
        ([10, 11, 20], [0, 0, 1])
    """

    def __init__(
        self,
        token_sequences: Optional[Sequence[Array]] = None,
        loss_masks: Optional[Sequence[Array]] = None,
        shuffle: bool = True,
    ) -> None:
        self.shuffle = shuffle

        if token_sequences is None:
            raise ValueError("token_sequences are required")
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
        if self.shuffle:
            self.indices = xp.random.permutation(self.num_examples)

    def __iter__(self) -> Iterator[dict[str, Array]]:
        for sample_idx in self.indices:
            example = self.examples[int(sample_idx)]
            yield {
                "tokens": example["tokens"],
                "loss_mask": example["loss_mask"],
            }

    def __len__(self) -> int:
        return self.num_examples


class TokenWindowDataset(IterableDataset):
    """
    Yields one lazy token-window example at a time.

    One item is not a batch.
    One item represents one stream window:
        stream[offset : offset + window_len]
    """

    def __init__(
        self,
        data: Array,
        *,
        window_len: int,
        sampling: Literal["random", "sequential"] = "random",
        examples_per_epoch: int | None = None,
        offset_buffer_size: int = 4096,
    ) -> None:
        self.stream = xp.array(data, dtype=xp.int32)
        self.window_len = window_len
        self.sampling = sampling
        self.examples_per_epoch = examples_per_epoch
        self.offset_buffer_size = offset_buffer_size

        # The edge case where len(stream) == window length, valid_window_count == 1, offset 0 is valid
        self.valid_window_count = len(self.stream) - self.window_len + 1

        if self.sampling not in {"random", "sequential"}:
            raise ValueError(
                f"sampling must be 'random' or 'sequential', got {self.sampling!r}"
            )
        if self.offset_buffer_size < 1:
            raise ValueError("offset_buffer_size must be >= 1")
        if self.examples_per_epoch is not None and self.examples_per_epoch < 1:
            raise ValueError("examples_per_epoch must be >= 1 when provided")
        if self.sampling == "sequential" and self.examples_per_epoch is not None:
            raise ValueError("examples_per_epoch is only valid for sampling='random'")

        if self.valid_window_count < 1:
            raise ValueError(
                f"Need at least {self.window_len} tokens, got {len(self.stream)}"
            )

    def on_epoch_start(self) -> None:
        pass

    def __iter__(self):
        if self.sampling == "random":
            yielded = 0

            while self.examples_per_epoch is None or yielded < self.examples_per_epoch:
                remaining = (
                    self.offset_buffer_size
                    if self.examples_per_epoch is None
                    else min(self.offset_buffer_size, self.examples_per_epoch - yielded)
                )
                # Keeping offsets on CPU so we're using numpy explicitly

                offsets = np.random.randint(
                    0,
                    self.valid_window_count,  # exclusive high
                    size=remaining,
                    dtype=np.int32,
                )

                for offset in offsets:
                    yielded += 1
                    yield TokenWindowExample(
                        stream=self.stream,
                        offset=int(offset),
                        window_len=self.window_len,
                    )

            return

        for offset in range(self.valid_window_count):
            yield TokenWindowExample(
                stream=self.stream,
                offset=offset,
                window_len=self.window_len,
            )

    def __len__(self) -> int:
        if self.sampling == "random":
            if self.examples_per_epoch is None:
                raise TypeError("TokenWindowDataset with sampling='random' is infinite")
            return self.examples_per_epoch

        return self.valid_window_count
