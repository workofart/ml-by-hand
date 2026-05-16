"""
In-process threading-based mock backend for DDP correctness tests.

This module lives under `test/` because it is test infrastructure, not
production code: it lets us exercise the AllReduce / broadcast / barrier
code paths without spawning real NCCL processes or holding a multi-GPU
box. Each "rank" is one Python thread; the shared `MockComm` synchronizes
them through a `threading.Barrier`.

TODO(phase-2-cleanup): once real NCCL is wired in
`autograd.distributed._make_backend`, the multi-rank tests
(`test_allreduce.py`, `test_ddp_equivalence.py`) can spawn real processes
via the launcher and consume the real backend. At that point delete this
file entirely along with the thread-local plumbing in
`autograd/distributed.py`.

Usage from a test:

    from test.distributed.mock import run_mock_ranks

    def per_rank(...):
        ...   # this runs in a thread; rank()/world_size() reflect the thread

    run_mock_ranks(world_size=2, target=per_rank)

`run_mock_ranks` blocks until every rank finishes, and re-raises the first
exception observed. Exceptions abort the rendezvous so a misbehaving rank
cannot deadlock the others.

The real comm group lives in `autograd.distributed`; this module wires
per-thread rank/world_size into that module's thread-local hooks
(`_set_thread_local_rank` / `_clear_thread_local`) so production-side
helpers like `is_distributed()` / `rank()` see per-rank values without
needing to be aware of the test harness.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

from autograd import distributed as _dist
from autograd.backend import xp
from autograd.distributed import ReduceOp


def _snapshot(buf: Any) -> Any:
    """Backend-agnostic array copy.

    NumPy/CuPy arrays expose `.copy()`; MLX arrays do not but can be
    re-constructed via `xp.array(buf)`. We use this in MockComm so the
    rendezvous keeps a stable snapshot even if the caller mutates `buf`
    after publishing it (e.g. AllReduce writing the result back in-place).

    Caveat: MLX lazy graphs aren't safe to evaluate across threads. If a
    caller forces materialization (e.g. `xp.to_scalar`) on a freshly-
    AllReduced MLX array inside a rank-thread, the cross-thread `xp.eval`
    can segfault. Callers that scalarize results inline (e.g. trainer
    loss reporting) should either run those tests on numpy or skip them
    on MLX. Production DDP runs one process per rank so this is a
    mock-only concern.
    """
    if hasattr(buf, "copy"):
        return buf.copy()
    return xp.array(buf)


class MockComm:
    """Shared rendezvous state for one mock comm group.

    Threads coordinate through a single `threading.Barrier`. The result of
    each collective is computed by rank 0 (or by the broadcaster's writer)
    and dropped into a shared slot the other ranks read from.

    A `BrokenBarrierError` from any wait() means another rank has aborted;
    we surface it as a RuntimeError so tests get a useful traceback instead
    of a confusing "broken barrier" message.
    """

    def __init__(self, world_size: int) -> None:
        if world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {world_size}")
        self.world_size = world_size
        self._barrier = threading.Barrier(world_size)
        # AllReduce slots: each rank deposits, rank 0 reduces, all read.
        self._allreduce_inputs: list[Any] = [None] * world_size
        self._allreduce_result: Any = None
        # Broadcast slot: src deposits, others read.
        self._broadcast_value: Any = None

    def _sync(self) -> None:
        try:
            self._barrier.wait()
        except threading.BrokenBarrierError as exc:
            raise RuntimeError(
                "MockComm barrier broken — another rank likely failed."
            ) from exc

    def all_reduce(self, rank_: int, buf: Any, op: str) -> Any:
        # Each rank publishes a snapshot of its input. We snapshot because
        # the caller may rebind or mutate `buf` after we return; rank 0
        # reads from these snapshots to compute the reduction.
        self._allreduce_inputs[rank_] = _snapshot(buf)
        self._sync()

        if rank_ == 0:
            if op != ReduceOp.SUM:
                raise NotImplementedError(f"MockComm only implements SUM; got {op!r}")
            result = _snapshot(self._allreduce_inputs[0])
            for r in range(1, self.world_size):
                result = result + self._allreduce_inputs[r]
            self._allreduce_result = result
        self._sync()

        # Hand each rank its own snapshot so per-thread downstream
        # mutations (e.g. an in-place divide) cannot race.
        out = _snapshot(self._allreduce_result)
        self._sync()

        # Rank 0 clears the rendezvous state, then everyone syncs again
        # before returning. Without this final barrier, rank 1 can race
        # ahead into its next all_reduce and publish before rank 0 has
        # cleared the previous slots — wiping the new input.
        if rank_ == 0:
            self._allreduce_inputs = [None] * self.world_size
            self._allreduce_result = None
        self._sync()
        return out

    def broadcast(self, rank_: int, buf: Any, from_rank: int) -> Any:
        if not (0 <= from_rank < self.world_size):
            raise ValueError(
                f"from_rank must be in [0, {self.world_size}), got {from_rank}"
            )
        if rank_ == from_rank:
            self._broadcast_value = _snapshot(buf)
        self._sync()

        out = _snapshot(self._broadcast_value)
        self._sync()

        # Same final-barrier rationale as all_reduce: the broadcaster
        # clears the slot only after everyone has snapshotted, then we
        # barrier so the next call cannot publish into a half-cleared slot.
        if rank_ == from_rank:
            self._broadcast_value = None
        self._sync()
        return out

    def barrier(self) -> None:
        self._sync()


class MockBackend:
    """Per-thread `Backend` implementation bound to one rank of a `MockComm`.

    Each thread spawned by `run_mock_ranks` gets its own `MockBackend`; all
    instances in the group share the same `MockComm`. The split lets
    `Backend.rank` be a simple attribute instead of a thread-local lookup.
    """

    def __init__(self, comm: MockComm, rank_: int) -> None:
        if not (0 <= rank_ < comm.world_size):
            raise ValueError(f"rank must be in [0, {comm.world_size}), got {rank_}")
        self.comm = comm
        self._rank = rank_

    @property
    def world_size(self) -> int:
        return self.comm.world_size

    @property
    def rank(self) -> int:
        return self._rank

    def all_reduce(self, buf: Any, op: str = ReduceOp.SUM) -> Any:
        return self.comm.all_reduce(self._rank, buf, op)

    def broadcast(self, buf: Any, from_rank: int) -> Any:
        return self.comm.broadcast(self._rank, buf, from_rank)

    def barrier(self) -> None:
        self.comm.barrier()

    def teardown(self) -> None:
        # Threads have no external resources to release. We could poison
        # the shared state to make accidental reuse fail loudly, but tests
        # already create a fresh MockComm per call.
        pass


def run_mock_ranks(
    world_size: int,
    target: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> list[Any]:
    """Spawn `world_size` threads, each running `target(*args, **kwargs)`.

    Each thread is installed as a distinct rank with its own `MockBackend`,
    so calls to `autograd.distributed.rank()`/`world_size()` inside the
    target observe per-rank values rather than the process-wide env defaults.

    Returns the list of per-rank results in rank order. The first exception
    raised by any rank is re-raised after all threads have joined.
    """
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")

    comm = MockComm(world_size)
    results: list[Any] = [None] * world_size
    errors: list[BaseException | None] = [None] * world_size

    def _runner(rank_: int) -> None:
        backend = MockBackend(comm, rank_)
        _dist._set_thread_local_rank(
            rank_=rank_,
            world_size_=world_size,
            local_rank_=rank_,
            backend=backend,
        )
        try:
            results[rank_] = target(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 — re-raised below
            errors[rank_] = exc
            # Abort the rendezvous so peers waiting on a barrier wake up
            # instead of deadlocking. Other ranks will see BrokenBarrier
            # and re-raise via MockComm._sync.
            comm._barrier.abort()
        finally:
            _dist._clear_thread_local()

    threads = [
        threading.Thread(target=_runner, args=(r,), name=f"mock-rank-{r}")
        for r in range(world_size)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for exc in errors:
        if exc is not None:
            raise exc

    return results
