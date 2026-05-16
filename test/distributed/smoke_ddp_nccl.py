"""End-to-end Phase 2 smoke: real NCCL backend via the launcher.

Invoke with:

    AUTOGRAD_BACKEND=cupy .venv/bin/python -m autograd.distributed.launch \\
        --nproc-per-node=2 test/distributed/smoke_ddp_nccl.py

Each rank constructs a small tensor, calls `allreduce_grads` and
`broadcast_parameters`, and prints the result. We then assert manually
inside each rank that:

- After allreduce_grads, every rank's grad equals the host-side mean of
  the per-rank inputs.
- After broadcast_parameters from rank 0, every rank's param matches
  rank 0's.

If the asserts pass on every rank, the launcher exits 0.
"""

from __future__ import annotations

import numpy as np

from autograd.backend import xp
from autograd.distributed import (
    allreduce_grads,
    broadcast_parameters,
    get_backend,
    is_distributed,
    rank,
    world_size,
)


class _FakeGrad:
    def __init__(self, data):
        self.data = data


class _FakeParam:
    def __init__(self, data, grad=None):
        self.data = data
        self.grad = _FakeGrad(grad) if grad is not None else None


def main() -> None:
    if not is_distributed():
        raise RuntimeError(
            "smoke_ddp_nccl must be invoked via the launcher; world_size=1 here."
        )

    r = rank()
    ws = world_size()
    print(f"[rank {r}] starting; world_size={ws}", flush=True)

    # Force backend init (would also happen lazily on first collective).
    backend = get_backend()
    print(f"[rank {r}] backend ready: {type(backend).__name__}", flush=True)

    # --- allreduce_grads: each rank has a different grad; expect the mean. ---
    grad_vals = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32) * (r + 1)
    params = {
        "w": _FakeParam(
            data=xp.zeros(4, dtype=xp.float32),
            grad=xp.asarray(grad_vals),
        )
    }
    allreduce_grads(params)

    expected_mean = (
        sum(
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32) * (i + 1)
            for i in range(ws)
        )
        / ws
    )
    got = xp.to_numpy(params["w"].grad.data)
    np.testing.assert_allclose(got, expected_mean, rtol=1e-6, atol=1e-6)
    print(f"[rank {r}] allreduce_grads OK: {got}", flush=True)

    # --- broadcast_parameters: rank 0's data should propagate to all. ---
    rank0_data = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    starting = (
        rank0_data if r == 0 else np.array([99.0, 99.0, 99.0, 99.0], dtype=np.float32)
    )
    params = {"w": _FakeParam(data=xp.asarray(starting))}
    broadcast_parameters(params, from_rank=0)
    got = xp.to_numpy(params["w"].data)
    np.testing.assert_allclose(got, rank0_data, rtol=0, atol=0)
    print(f"[rank {r}] broadcast_parameters OK: {got}", flush=True)

    # --- barrier: every rank must reach this line. ---
    backend.barrier()
    print(f"[rank {r}] barrier OK; PASS", flush=True)


if __name__ == "__main__":
    main()
