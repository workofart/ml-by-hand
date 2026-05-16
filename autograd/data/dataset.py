from typing import Any, Iterator, Sequence, Union

import numpy as np

from autograd.data.types import TokenWindowExample

NumpyDType = Union[type[np.generic], np.dtype[Any], str]


class MapDataset:
    """
    Map-style dataset for already-shaped examples.

    Samplers operate on this interface: they yield indices, and DataLoader uses
    those indices to fetch examples with `dataset[index]`.

    Use `PairedMapDataset` when examples are built from two aligned fields.
    """

    def __init__(self, examples: Sequence[Any]) -> None:
        self.examples = list(examples)

    def on_epoch_start(self) -> None:
        pass

    def __getitem__(self, index: int) -> Any:
        return self.examples[index]

    def __iter__(self) -> Iterator[Any]:
        for index in range(len(self)):
            yield self[index]

    def __len__(self) -> int:
        return len(self.examples)


class PairedMapDataset(MapDataset):
    """
    Lazy paired `(X, y)` dataset — indexes into the source arrays on demand.

    The `inputs` and `targets` arguments can each be:
      - An array or sequence of arrays (stored as numpy)
      - A file path to a `.npy` file (memory-mapped, pages in on demand)

    Examples:
        >>> X = xp.arange(6).reshape(3, 2)
        >>> y = xp.array([0, 1, 2])
        >>> dataset = PairedMapDataset(X, y)
        >>> example = next(iter(dataset))
        >>> example["inputs"].shape, int(example["targets"])
        ((2,), 0)
        >>> seq2seq = PairedMapDataset(
        ...     [xp.array([1, 2])],
        ...     [xp.array([3, 4])],
        ...     input_key="input_ids",
        ...     target_key="labels",
        ... )
        >>> example = next(iter(seq2seq))
        >>> example["input_ids"].tolist(), example["labels"].tolist()
        ([1, 2], [3, 4])
        >>> causal_lm = PairedMapDataset(
        ...     [xp.array([10, 11, 12])],
        ...     [xp.array([0, 1, 1])],
        ...     input_key="tokens",
        ...     target_key="loss_mask",
        ... )
        >>> example = next(iter(causal_lm))
        >>> example["tokens"].tolist(), example["loss_mask"].tolist()
        ([10, 11, 12], [0, 1, 1])
    """

    @staticmethod
    def _load(data: Union[Sequence[Any], str]) -> Any:
        if isinstance(data, str):
            return np.load(data, mmap_mode="r")
        if isinstance(data, np.ndarray):
            return data
        # Ragged sequences (variable-length arrays) — keep as list
        return list(data)

    def __init__(
        self,
        inputs: Union[Sequence[Any], str],
        targets: Union[Sequence[Any], str],
        *,
        input_key: str = "inputs",
        target_key: str = "targets",
        dtype: NumpyDType | None = None,
    ) -> None:
        self.inputs = self._load(inputs)
        self.targets = self._load(targets)
        if len(self.inputs) != len(self.targets):
            raise ValueError(
                "inputs and targets must contain the same number of examples"
            )
        self.input_key = input_key
        self.target_key = target_key
        self.dtype = None if dtype is None else np.dtype(dtype)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            self.input_key: np.asarray(self.inputs[index], dtype=self.dtype),
            self.target_key: np.asarray(self.targets[index], dtype=self.dtype),
        }

    def __len__(self) -> int:
        return len(self.inputs)


class TokenWindowMapDataset(MapDataset):
    """
    Maps token-window offsets to lazy examples.

    One item is not a batch.
    One item represents one stream window:
        stream[offset : offset + window_len]
    The dataset owns the token stream and offset -> window mapping only.
    Samplers own traversal order.

    Use `SequentialSampler` for ordered windows, `RandomSampler` for one shuffled
    pass, and `RandomSampler(replacement=True, num_samples=...)` for fixed-size
    random-with-replacement epochs. Calling `DataLoader.on_epoch_start()` calls
    the sampler hook, so a `RandomSampler` gives a fresh random order each epoch.

    The `data` argument can be:
      - An array (loaded into memory as before)
      - A file path to a `.npy` file (memory-mapped, pages in on demand)
    """

    def __init__(
        self,
        data: Union[np.ndarray, str],
        *,
        window_len: int,
    ) -> None:
        if isinstance(data, str):
            self.stream = np.load(data, mmap_mode="r")
        else:
            self.stream = np.asarray(data, dtype=np.int32)
        self.window_len = window_len

        # The edge case where len(stream) == window length, valid_window_count == 1, offset 0 is valid
        self.valid_window_count = len(self.stream) - self.window_len + 1

        if self.valid_window_count < 1:
            raise ValueError(
                f"Need at least {self.window_len} tokens, got {len(self.stream)}"
            )

    def __getitem__(self, offset: int) -> TokenWindowExample:
        return TokenWindowExample(
            stream=self.stream,
            offset=offset,
            window_len=self.window_len,
        )

    def __len__(self) -> int:
        return self.valid_window_count
