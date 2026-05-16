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
        if not replacement:
            self.indices = self._sample_indices_no_replacement()

    def _sample_indices_no_replacement(self) -> list[int]:
        indices = []
        while len(indices) < self.num_samples:
            permutation = list(range(len(self.dataset)))
            np.random.shuffle(permutation)
            indices.extend(permutation[: self.num_samples - len(indices)])
        return indices

    def _iter_with_replacement(self) -> Iterator[int]:
        remaining = self.num_samples
        n = len(self.dataset)
        chunk_size = min(n, 1_000_000)
        while remaining > 0:
            batch = min(chunk_size, remaining)
            yield from np.random.randint(0, n, size=batch).tolist()
            remaining -= batch

    def on_epoch_start(self) -> None:
        if not self.replacement:
            self.indices = self._sample_indices_no_replacement()

    def __iter__(self) -> Iterator[int]:
        if self.replacement:
            return self._iter_with_replacement()
        return iter(self.indices)

    def __len__(self) -> int:
        return self.num_samples


class DistributedSamplerAdapter(Sampler):
    """
    Wraps a Sampler so each rank sees a disjoint slice of its indices.

    Two strategies, picked from the wrapped sampler's semantics:

    1. With-replacement (e.g. `RandomSampler(replacement=True)`): each rank
       draws independent indices from `range(len(dataset))` using an RNG
       seeded by `(seed, epoch, rank)`. No cross-rank coordination needed;
       per-rank `num_samples = global // world_size`.

    2. Without-replacement / sequential: the wrapped sampler is iterated to
       exhaustion to materialize the global permutation, padded to a length
       divisible by `world_size`, then strided so rank *r* sees positions
       `r, r + world_size, r + 2*world_size, ...`. Disjoint, full coverage,
       deterministic per `(seed, epoch)`.

    The "is replacement?" probe reads `sampler.replacement` (set by
    `RandomSampler`). Sampler classes without that attribute are treated
    as without-replacement.

    Args:
        sampler: The base Sampler whose index stream should be sharded.
        rank: This rank's id, in `[0, world_size)`.
        world_size: Total number of ranks participating.
        seed: Base RNG seed shared across ranks; per-rank streams mix in
            `epoch` and `rank` on top of this.

    Examples:
        Opt-in at the script level so DataLoader stays DDP-agnostic.

        >>> from autograd.distributed import is_distributed, rank, world_size
        >>> sampler = RandomSampler(
        ...     dataset, replacement=True, num_samples=len(dataset)
        ... )
        >>> if is_distributed():
        ...     sampler = DistributedSamplerAdapter(
        ...         sampler, rank=rank(), world_size=world_size()
        ...     )
        >>> loader = DataLoader(dataset, batch_size=32, sampler=sampler)
    """

    def __init__(
        self,
        sampler: Sampler,
        *,
        rank: int,
        world_size: int,
        seed: int = 0,
    ) -> None:
        if world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {world_size}")
        if not (0 <= rank < world_size):
            raise ValueError(f"rank must be in [0, {world_size}), got {rank}")

        self._sampler = sampler
        self._rank = rank
        self._world_size = world_size
        self._seed = int(seed)
        self._epoch = 0
        self._with_replacement = bool(getattr(sampler, "replacement", False))

        global_samples = len(sampler)
        if self._with_replacement:
            self._per_rank_samples = global_samples // world_size
        else:
            self._per_rank_samples = (global_samples + world_size - 1) // world_size

        self._indices: list[int] = []
        self._refresh_indices()

    def _refresh_indices(self) -> None:
        if self._with_replacement:
            dataset = getattr(self._sampler, "dataset", None)
            if dataset is None:
                raise RuntimeError(
                    "DistributedSamplerAdapter requires `sampler.dataset` "
                    "for with-replacement sampling so it can sample from "
                    "the full index range."
                )
            n = len(dataset)
            # Mix (seed, epoch, rank) into a single 64-bit seed. Naive
            # tuple hashing in some RNGs only uses the first element,
            # which would correlate streams across ranks.
            h = 0
            for s in (self._seed, self._epoch, self._rank):
                h = (h * 1_000_003 + int(s)) & 0xFFFF_FFFF_FFFF_FFFF
            rng = np.random.default_rng(h)
            self._indices = rng.integers(0, n, size=self._per_rank_samples).tolist()
            return

        # Without-replacement: materialize global order, pad to multiple
        # of world_size, then take every world_size-th index starting at
        # `rank`.
        full = list(iter(self._sampler))
        target = self._per_rank_samples * self._world_size
        if len(full) < target:
            full = full + full[: target - len(full)]
        elif len(full) > target:
            full = full[:target]
        self._indices = full[self._rank :: self._world_size]

    def on_epoch_start(self) -> None:
        self._epoch += 1
        # Forward to the wrapped sampler so its internal RNG advances
        # (e.g. RandomSampler reshuffles its without-replacement permutation).
        self._sampler.on_epoch_start()
        self._refresh_indices()

    def __iter__(self) -> Iterator[int]:
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)


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
