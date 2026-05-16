"""Tests for DistributedSamplerAdapter.

The adapter has two strategies (with-replacement vs without-replacement)
and the tests cover the contract on each:

- Without-replacement: the union of indices across ranks equals the global
  permutation; per-rank streams are disjoint; the partition is
  deterministic across calls with the same (seed, epoch).
- With-replacement: per-rank streams are independent (different seeds);
  the per-rank length equals `len(dataset) // world_size`; the same
  (seed, epoch, rank) is reproducible.
"""

from __future__ import annotations

import pytest

from autograd.data.dataset import MapDataset
from autograd.data.sampler import (
    DistributedSamplerAdapter,
    RandomSampler,
    SequentialSampler,
)


class _ListDataset(MapDataset):
    def __init__(self, n: int) -> None:
        self._n = n

    def __getitem__(self, index: int) -> int:
        return index

    def __len__(self) -> int:
        return self._n

    def on_epoch_start(self) -> None:
        pass


# ---- without-replacement / sequential -------------------------------------


def test_sequential_disjoint_full_coverage():
    """All ranks together cover the full index range with no overlap."""
    n, world_size = 12, 4
    dataset = _ListDataset(n)
    base = SequentialSampler(dataset)

    per_rank = [
        list(
            DistributedSamplerAdapter(
                SequentialSampler(dataset),
                rank=r,
                world_size=world_size,
            )
        )
        for r in range(world_size)
    ]

    # Each rank gets a quarter of the indices.
    assert all(len(s) == n // world_size for s in per_rank)

    # Disjoint: no index appears on two ranks.
    seen: set[int] = set()
    for shard in per_rank:
        overlap = seen.intersection(shard)
        assert not overlap, f"overlap across ranks: {overlap}"
        seen.update(shard)

    # Full coverage: every index is present exactly once.
    assert seen == set(range(n))
    # Wrapped base sampler still emits the full ordering for sanity.
    assert list(iter(base)) == list(range(n))


def test_sequential_padding_when_not_divisible():
    """When len(sampler) % world_size != 0, padding fills out the partition."""
    n, world_size = 10, 4  # 10 % 4 = 2, target = ceil(10/4)*4 = 12
    dataset = _ListDataset(n)
    per_rank = [
        list(
            DistributedSamplerAdapter(
                SequentialSampler(dataset),
                rank=r,
                world_size=world_size,
            )
        )
        for r in range(world_size)
    ]
    # All shards same length: ceil(10/4) = 3
    assert all(len(s) == 3 for s in per_rank)
    # Total emitted == 12 = 3 * 4; only the first (12 - 10) = 2 indices were
    # padded (head-repeat), so the multiset of values >= n // world_size copies
    # for each index except the head pair.
    flat = [i for s in per_rank for i in s]
    assert len(flat) == 12
    for idx in range(n):
        assert idx in flat


def test_sequential_deterministic_across_calls():
    """Two adapters with the same (seed, epoch) emit the same shards."""
    dataset = _ListDataset(16)
    s1 = DistributedSamplerAdapter(
        SequentialSampler(dataset), rank=1, world_size=4, seed=42
    )
    s2 = DistributedSamplerAdapter(
        SequentialSampler(dataset), rank=1, world_size=4, seed=42
    )
    assert list(s1) == list(s2)


# ---- with-replacement -----------------------------------------------------


def test_with_replacement_per_rank_length():
    """Each rank receives floor(global / world_size) samples."""
    n, world_size = 12, 4
    dataset = _ListDataset(n)
    base = RandomSampler(dataset, replacement=True, num_samples=n)
    adapter = DistributedSamplerAdapter(base, rank=0, world_size=world_size)
    assert len(adapter) == n // world_size


def test_with_replacement_independent_streams():
    """Different ranks use different RNG seeds, so their streams differ."""
    n, world_size = 128, 4
    dataset = _ListDataset(n)
    base = RandomSampler(dataset, replacement=True, num_samples=n)
    streams = [
        list(DistributedSamplerAdapter(base, rank=r, world_size=world_size, seed=7))
        for r in range(world_size)
    ]
    # The probability of two independent uniform streams over [0, 128) of
    # length 32 being exactly equal is vanishing (~128^-32). Use that as
    # the cheap "are they different?" check.
    for r1 in range(world_size):
        for r2 in range(r1 + 1, world_size):
            assert streams[r1] != streams[r2]


def test_with_replacement_deterministic_per_seed_rank_epoch():
    """Same (seed, epoch, rank) reproduces the same per-rank stream."""
    n = 64
    dataset = _ListDataset(n)
    base = RandomSampler(dataset, replacement=True, num_samples=n)
    a = DistributedSamplerAdapter(base, rank=2, world_size=4, seed=99)
    b = DistributedSamplerAdapter(base, rank=2, world_size=4, seed=99)
    assert list(a) == list(b)


def test_with_replacement_epoch_advances_stream():
    """on_epoch_start should re-seed the stream (different from epoch 0)."""
    n = 64
    dataset = _ListDataset(n)
    base = RandomSampler(dataset, replacement=True, num_samples=n)
    a = DistributedSamplerAdapter(base, rank=0, world_size=4, seed=1)
    e0 = list(a)
    a.on_epoch_start()
    e1 = list(a)
    assert e0 != e1


# ---- error paths ----------------------------------------------------------


def test_rejects_invalid_rank():
    dataset = _ListDataset(8)
    base = SequentialSampler(dataset)
    with pytest.raises(ValueError):
        DistributedSamplerAdapter(base, rank=4, world_size=4)
    with pytest.raises(ValueError):
        DistributedSamplerAdapter(base, rank=-1, world_size=4)


def test_rejects_invalid_world_size():
    dataset = _ListDataset(8)
    base = SequentialSampler(dataset)
    with pytest.raises(ValueError):
        DistributedSamplerAdapter(base, rank=0, world_size=0)
