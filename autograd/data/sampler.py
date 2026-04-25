from abc import ABC, abstractmethod
from typing import Iterator

import numpy as np

from autograd.data.dataset import MapDataset


class Sampler(ABC):
    """
    Produces dataset indices for one epoch.

    MapDataset owns examples. Sampler owns index order. DataLoader owns
    grouping ordered examples into batches.

    `len(sampler)` is the number of indices the sampler will yield for one
    epoch. It may differ from `len(dataset)` for samplers that draw subsets,
    repeat examples, or shard data.
    """

    def on_epoch_start(self) -> None:
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[int]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class SequentialSampler(Sampler):
    """Yields dataset indices in stored order."""

    def __init__(self, dataset: MapDataset) -> None:
        if not isinstance(dataset, MapDataset):
            raise TypeError("SequentialSampler requires MapDataset")
        self.dataset = dataset

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.dataset)))

    def __len__(self) -> int:
        return len(self.dataset)


class RandomSampler(Sampler):
    """
    Yields random dataset indices each epoch.

    Without replacement, each epoch is a shuffled finite pass unless
    `num_samples` asks for a shorter or repeated multi-pass sample. With
    replacement, each epoch draws `num_samples` independent indices. Call
    `on_epoch_start()` to resample, which `DataLoader.on_epoch_start()` does for
    attached samplers.
    """

    def __init__(
        self,
        dataset: MapDataset,
        *,
        replacement: bool = False,
        num_samples: int | None = None,
    ) -> None:
        if not isinstance(dataset, MapDataset):
            raise TypeError("RandomSampler requires MapDataset")
        if len(dataset) < 1:
            raise ValueError("RandomSampler requires a non-empty dataset")
        if num_samples is None:
            num_samples = len(dataset)
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1")
        self.dataset = dataset
        self.replacement = replacement
        self.num_samples = num_samples
        self.indices = self._sample_indices()

    def _sample_indices(self) -> list[int]:
        if self.replacement:
            return np.random.randint(
                0,
                len(self.dataset),
                size=self.num_samples,
                dtype=np.int32,
            ).tolist()

        indices = []
        while len(indices) < self.num_samples:
            permutation = list(range(len(self.dataset)))
            np.random.shuffle(permutation)
            indices.extend(permutation[: self.num_samples - len(indices)])
        return indices

    def on_epoch_start(self) -> None:
        self.indices = self._sample_indices()

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return self.num_samples


class TokenLengthGroupedRandomSampler(Sampler):
    """
    Yields indices so nearby examples have similar lengths.

    Use this with a dynamic-padding collator. The sampler does not pack examples:
    each sampled index still maps to one independent batch row.

    Args:
        dataset: MapDataset with a `"tokens"` field in each example.
        sort_buffer_size: Number of shuffled indices to sort by length at a
            time. Larger buffers reduce padding more but make ordering less
            random. The DataLoader batch size is still the number of examples
            passed to the collator at once.
    """

    def __init__(
        self,
        dataset: MapDataset,
        *,
        sort_buffer_size: int,
    ) -> None:
        if not isinstance(dataset, MapDataset):
            raise TypeError("TokenLengthGroupedRandomSampler requires MapDataset")
        if sort_buffer_size < 1:
            raise ValueError("sort_buffer_size must be >= 1")
        if len(dataset) > 0 and "tokens" not in dataset[0]:
            raise ValueError(
                "TokenLengthGroupedRandomSampler requires examples with a 'tokens' field"
            )
        self.dataset = dataset
        self.sort_buffer_size = sort_buffer_size
        self.indices = self._ordered_indices()

    def _ordered_indices(self) -> list[int]:
        shuffled = list(range(len(self.dataset)))
        np.random.shuffle(shuffled)
        ordered = []
        for start in range(0, len(shuffled), self.sort_buffer_size):
            buffer = shuffled[start : start + self.sort_buffer_size]
            ordered.extend(sorted(buffer, key=self._sequence_length))
        return ordered

    def _sequence_length(self, index: int) -> int:
        return len(self.dataset[index]["tokens"])

    def on_epoch_start(self) -> None:
        self.indices = self._ordered_indices()

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.dataset)
