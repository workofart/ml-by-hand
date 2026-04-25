from typing import Any, Iterator, Sequence

from autograd.backend import Array, xp
from autograd.data.types import TokenWindowExample


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
    Iterates over in-memory paired `(X, y)` data one example at a time.

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
        ...     dtype=xp.int32,
        ... )
        >>> example = next(iter(seq2seq))
        >>> example["input_ids"].tolist(), example["labels"].tolist()
        ([1, 2], [3, 4])
        >>> causal_lm = PairedMapDataset(
        ...     [xp.array([10, 11, 12])],
        ...     [xp.array([0, 1, 1])],
        ...     input_key="tokens",
        ...     target_key="loss_mask",
        ...     dtype=xp.int32,
        ... )
        >>> example = next(iter(causal_lm))
        >>> example["tokens"].tolist(), example["loss_mask"].tolist()
        ([10, 11, 12], [0, 1, 1])
    """

    def __init__(
        self,
        inputs: Sequence[Any],
        targets: Sequence[Any],
        *,
        input_key: str = "inputs",
        target_key: str = "targets",
        dtype: Any | None = None,
    ) -> None:
        if len(inputs) != len(targets):
            raise ValueError(
                "inputs and targets must contain the same number of examples"
            )

        examples = []
        for index in range(len(inputs)):
            input_value = inputs[index]
            target_value = targets[index]
            if dtype is not None:
                input_value = xp.array(input_value, dtype=dtype)
                target_value = xp.array(target_value, dtype=dtype)
            examples.append({input_key: input_value, target_key: target_value})
        super().__init__(examples)


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
    """

    def __init__(
        self,
        data: Array,
        *,
        window_len: int,
    ) -> None:
        self.stream = xp.array(data, dtype=xp.int32)
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
