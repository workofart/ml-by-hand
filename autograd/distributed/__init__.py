"""
Single-node multi-GPU data-parallel training surface.

Everything DDP-related lives here so a reader can absorb the whole story
in one file. The structure of the module reads top to bottom:

1. Env-derived rank/world_size and thread-local overrides (test path).
2. Public API: rank(), world_size(), is_distributed(), barrier().
3. Backend protocol + lifecycle (`_NCCLBackend` wraps `cupyx.distributed`).
4. Collective operations on parameters: `allreduce_grads`,
   `broadcast_parameters`. These are the only DDP touchpoints that the
   optimizer needs.
5. bf16 hardware gate (`check_bf16_capability`).

When `WORLD_SIZE` is absent or `1`, `is_distributed()` returns False and
every DDP code path no-ops, so the single-node training path is
bit-identical to the pre-DDP code.

The thread-based test mock lives at `test/distributed/mock.py` — it
installs per-thread backends via the `_set_thread_local_rank` /
`_clear_thread_local` helpers below and never flows through the
process-wide backend singleton.
"""

from __future__ import annotations

import atexit
import os
import threading
from typing import Any, Mapping, Protocol

from autograd.backend import LOW_PRECISION_FLOAT_DTYPES, xp
from autograd.tensor import Tensor

# --- env-derived process-wide defaults ---------------------------------------
#
# Read once at import. The launcher (Phase 2 `autograd.distributed.launch`)
# sets WORLD_SIZE/RANK/LOCAL_RANK before the user script imports anything.

_env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
_env_rank = int(os.environ.get("RANK", "0"))
_env_local_rank = int(os.environ.get("LOCAL_RANK", "0"))


# --- thread-local overrides used by the test mock ----------------------------
#
# When tests spawn N threads to emulate N ranks in one Python process, each
# thread sets thread-local rank/world_size/backend via `_set_thread_local_rank`
# (called from `test/distributed/mock.py`). The accessors below prefer
# thread-local values when present.
#
# TODO(phase-2-cleanup): once real NCCL is wired in `_make_backend`, the
# multi-rank tests can spawn real processes via the launcher and we can drop:
#   - this `_thread_local` block
#   - the `_tls_get` branches in rank()/world_size()/local_rank()/get_backend()
#   - `_set_thread_local_rank` / `_clear_thread_local`
#   - `test/distributed/mock.py` entirely
# That removes ~25 lines from this module and the whole mock infrastructure.

_thread_local = threading.local()


def _tls_get(name: str, default: Any) -> Any:
    return getattr(_thread_local, name, default)


# --- public rank/world API ---------------------------------------------------


def world_size() -> int:
    """World size for the current rank (thread-local override beats env)."""
    return _tls_get("world_size", _env_world_size)


def rank() -> int:
    """Global rank of the caller (thread-local override beats env)."""
    return _tls_get("rank", _env_rank)


def local_rank() -> int:
    """Local rank of the caller within its node."""
    return _tls_get("local_rank", _env_local_rank)


def is_distributed() -> bool:
    """True when more than one rank is participating in this comm group."""
    return world_size() > 1


# --- backend protocol --------------------------------------------------------


class ReduceOp:
    """Op tags accepted by `Backend.all_reduce`. String tags keep the test
    mock serializable and avoid a dependency on a backend-specific enum."""

    SUM = "sum"


class Backend(Protocol):
    """Minimum collective surface needed by autograd's DDP path.

    Real implementation (Phase 2) wraps `cupyx.distributed.NCCLBackend`.
    Test implementation lives at `test/distributed/mock.py`.
    """

    @property
    def world_size(self) -> int: ...

    @property
    def rank(self) -> int: ...

    def all_reduce(self, buf: Any, op: str = ReduceOp.SUM) -> Any: ...

    def broadcast(self, buf: Any, from_rank: int) -> Any: ...

    def barrier(self) -> None: ...

    def teardown(self) -> None: ...


# --- backend lifecycle -------------------------------------------------------
#
# Two lookup paths share `get_backend()`:
#
# 1. Thread-local backend (test mock): per-thread, set by `_set_thread_local_rank`.
# 2. Process-wide backend (real NCCL): lazy singleton, one per process.

_process_backend: Backend | None = None
_init_lock = threading.Lock()


def init_process_group() -> Backend | None:
    """Eagerly initialize the process-wide backend (idempotent).

    Returns the backend, or None when not running distributed. Most callers
    use `get_backend()` (lazy) instead; this exists for tests/launchers that
    want to control init timing.
    """
    return _process_backend_singleton()


def get_backend() -> Backend:
    """Return the backend bound to the calling thread.

    Looks up the thread-local backend first (mock path), then the process
    singleton (real NCCL path). Raises if not running distributed.
    """
    tls_backend = _tls_get("backend", None)
    if tls_backend is not None:
        return tls_backend

    backend = _process_backend_singleton()
    if backend is None:
        raise RuntimeError(
            "get_backend() called but no comm group is initialized "
            "(world_size=1). Guard with is_distributed() before calling."
        )
    return backend


def _process_backend_singleton() -> Backend | None:
    """Lazy, double-checked init of the process-wide backend."""
    global _process_backend
    if _process_backend is not None:
        return _process_backend
    if not is_distributed():
        return None
    with _init_lock:
        if _process_backend is None:
            _process_backend = _make_backend()
            atexit.register(_safe_teardown)
    return _process_backend


class _NCCLBackend:
    """Real NCCL-backed collective ops for single-node multi-GPU DDP.

    Adapts `cupyx.distributed.NCCLBackend` to the `Backend` protocol:

    - `all_reduce`: NCCL takes separate (in_array, out_array). We allocate
      `out_array` here and return it so callers can rebind. Allocation is
      tiny relative to the reduction itself.
    - `broadcast`: NCCL is in-place via `root`. We return the same buffer
      so callers can use the result uniformly (in-place or not).

    Rendezvous works like PyTorch's: rank 0 binds a TCP server on
    `host:port`, the others connect, and they exchange the NCCL unique id.
    The launcher (`autograd.distributed.launch`) picks the port and
    propagates it via the `MASTER_PORT` env var.
    """

    def __init__(self, world_size: int, rank: int, host: str, port: int) -> None:
        # Imported here (not at module top) so the file stays importable on
        # MLX / numpy and on CuPy boxes without NCCL. _make_backend() only
        # runs when WORLD_SIZE > 1, which is gated by the launcher.
        from cupyx.distributed import NCCLBackend as _CupyxNCCL

        self._world_size = world_size
        self._rank = rank
        # Constructor blocks: rank 0 listens, others connect. All ranks
        # must reach this line within NCCL's bootstrap timeout.
        self._comm = _CupyxNCCL(world_size, rank, host=host, port=port)

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def rank(self) -> int:
        return self._rank

    def all_reduce(self, buf: Any, op: str = ReduceOp.SUM) -> Any:
        if op != ReduceOp.SUM:
            raise NotImplementedError(
                f"NCCL backend currently only implements SUM; got {op!r}"
            )
        import cupy as cp

        out = cp.empty_like(buf)
        self._comm.all_reduce(buf, out, op="sum")
        return out

    def broadcast(self, buf: Any, from_rank: int) -> Any:
        # NCCL broadcast mutates the buffer in place on every rank; the
        # root's contents win. We return `buf` so the call site uses the
        # same rebind pattern as `all_reduce`.
        self._comm.broadcast(buf, root=from_rank)
        return buf

    def barrier(self) -> None:
        self._comm.barrier()

    def teardown(self) -> None:
        self._comm.stop()


def _make_backend() -> Backend:
    """Construct the process-wide NCCL backend.

    Expects the launcher to have set MASTER_ADDR + MASTER_PORT in env. The
    error message guides the user to the launcher when those are missing —
    invoking a script directly with WORLD_SIZE manually set is unsupported.
    """
    host = os.environ.get("MASTER_ADDR", "127.0.0.1")
    port_str = os.environ.get("MASTER_PORT")
    if not port_str:
        raise RuntimeError(
            "MASTER_PORT must be set when running distributed. Use the "
            "launcher: python -m autograd.distributed.launch "
            "--nproc-per-node=N script.py"
        )
    return _NCCLBackend(
        world_size=world_size(),
        rank=rank(),
        host=host,
        port=int(port_str),
    )


def _safe_teardown() -> None:
    global _process_backend
    if _process_backend is not None:
        try:
            _process_backend.teardown()
        finally:
            _process_backend = None


def teardown() -> None:
    """Tear down the process-wide backend. Idempotent."""
    _safe_teardown()


def barrier() -> None:
    """Process-wide synchronization. No-op when not distributed."""
    if not is_distributed():
        return
    get_backend().barrier()


# --- thread-local helpers (test-mock plug point) -----------------------------


def _set_thread_local_rank(
    *,
    rank_: int,
    world_size_: int,
    local_rank_: int,
    backend: Backend,
) -> None:
    """Internal: install per-thread rank/world_size/backend.

    Called by `test/distributed/mock.py::run_mock_ranks` for each spawned
    rank-thread. Not intended for user code.
    """
    _thread_local.rank = rank_
    _thread_local.world_size = world_size_
    _thread_local.local_rank = local_rank_
    _thread_local.backend = backend


def _clear_thread_local() -> None:
    """Internal: wipe thread-local state after a mock rank-thread exits."""
    for name in ("rank", "world_size", "local_rank", "backend"):
        if hasattr(_thread_local, name):
            delattr(_thread_local, name)


# --- collective ops on model parameters --------------------------------------
#
# These are the only DDP touchpoints the optimizer needs. Both self-no-op
# when world_size==1, so callers can invoke them unconditionally — no
# `is_distributed()` guards required at the call site.


def allreduce_grads(parameters: Mapping[str, Tensor]) -> None:
    """AllReduce-mean every accumulated gradient across ranks.

    Result is mathematically equivalent to one forward+backward over the
    concatenated global batch (world_size * per_rank_batch rows).

    For low-precision gradients the reduction is performed in fp32: bf16's
    8-bit mantissa loses up to log2(N) bits per N-way sum, so reducing in
    bf16 would silently degrade gradient quality as world_size grows. The
    fp32 result is cast back to the original gradient dtype before being
    written back to `param.grad.data`.

    No-op when `world_size == 1`.
    """
    if not is_distributed():
        return

    backend = get_backend()
    n_ranks = world_size()

    for param in parameters.values():
        if param.grad is None:
            continue
        grad = param.grad.data
        if grad.dtype in LOW_PRECISION_FLOAT_DTYPES:
            g32 = grad.astype(xp.float32)
            summed = backend.all_reduce(g32, op=ReduceOp.SUM)
            averaged = summed / n_ranks
            param.grad.data = averaged.astype(grad.dtype)
        else:
            summed = backend.all_reduce(grad, op=ReduceOp.SUM)
            param.grad.data = summed / n_ranks


def broadcast_parameters(
    parameters: Mapping[str, Tensor],
    *,
    from_rank: int = 0,
) -> None:
    """Broadcast every parameter buffer from `from_rank` to all ranks.

    Belt-and-suspenders for the determinism story: even with a shared SEED
    every rank already constructs identical params under deterministic
    init, but platform-level non-determinism can drift them apart. One
    broadcast at startup eliminates that risk for a negligible one-time cost.

    No-op when `world_size == 1`.
    """
    if not is_distributed():
        return

    backend = get_backend()
    for param in parameters.values():
        param.data = backend.broadcast(param.data, from_rank=from_rank)


# --- bf16 hardware gate ------------------------------------------------------
#
# bf16 on CuPy requires CUDA compute capability >= 8.0 (Ampere). Lower-cc
# devices either don't expose `cupy.bfloat16` or fall back to software bf16
# emulation with surprise-slow kernels. We hard-fail with an actionable
# message instead.


def check_bf16_capability(parameter_dtype: str) -> None:
    """Hard-fail if `parameter_dtype == 'bfloat16'` on incompatible hardware.

    Runs unconditionally on CuPy (single-GPU and DDP). MLX has its own bf16
    path; numpy has no bf16 dtype. For non-CuPy backends, this is a no-op.
    """
    if parameter_dtype != "bfloat16":
        return

    # Look these up at call time (not import time) so test monkey-patches
    # of `autograd.backend.{IS_CUPY,xp}` take effect.
    from autograd.backend import IS_CUPY, xp

    if not IS_CUPY:
        return

    if not hasattr(xp, "bfloat16"):
        raise RuntimeError(
            'parameter_dtype="bfloat16" requires cupy.bfloat16, but the '
            "installed CuPy version does not expose it. Upgrade CuPy or "
            'change parameter_dtype to "float32" in the training config.'
        )

    device = xp.cuda.Device()
    cc_str = device.compute_capability
    name = (
        device.attributes.get("Name", "<unknown>")
        if hasattr(device, "attributes")
        else "<unknown>"
    )
    try:
        major = int(cc_str[:-1])
        minor = int(cc_str[-1])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Failed to parse compute capability {cc_str!r} from cupy device."
        ) from exc

    if (major, minor) < (8, 0):
        raise RuntimeError(
            'parameter_dtype="bfloat16" requires CUDA compute capability >= 8.0 '
            f"(Ampere or newer). Detected device: {name}, compute capability "
            f"{major}.{minor}. Either run on Ampere/Hopper hardware, or change "
            'parameter_dtype to "float32" in the training config.'
        )
