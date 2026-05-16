"""End-to-end Phase 3 smoke: real NCCL exercising the new helpers.

Invoke with:

    AUTOGRAD_BACKEND=cupy .venv/bin/python -m autograd.distributed.launch \\
        --nproc-per-node=2 test/distributed/smoke_ddp_phase3.py

Verifies on a real multi-process / multi-GPU setup that:

- `broadcast_optimizer_state` copies rank-0's tensor-valued state to all
  ranks and leaves non-dict slots (e.g. integer `timestep`) untouched.
- `TrainingState.metrics_row` with `log_global_loss=True` produces the true global
  weighted mean across ranks, not the rank-local ratio.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from autograd.backend import xp
from autograd.distributed import (
    broadcast_optimizer_state,
    get_backend,
    is_distributed,
    rank,
    world_size,
)
from autograd.tools.trainer import TrainingState


def _test_broadcast_optimizer_state(r: int, ws: int) -> None:
    rank0_m_vals = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    rank0_v_vals = np.array([0.01, 0.02, 0.03], dtype=np.float32)

    m_val = xp.asarray(rank0_m_vals) if r == 0 else xp.zeros(3, dtype=xp.float32)
    v_val = xp.asarray(rank0_v_vals) if r == 0 else xp.zeros(3, dtype=xp.float32)

    fake_optimizer = SimpleNamespace(
        _states={
            "m": {"layer.w": m_val},
            "v": {"layer.w": v_val},
            "timestep": r * 100,
        }
    )

    broadcast_optimizer_state(fake_optimizer, from_rank=0)

    got_m = xp.to_numpy(fake_optimizer._states["m"]["layer.w"])
    got_v = xp.to_numpy(fake_optimizer._states["v"]["layer.w"])
    np.testing.assert_allclose(got_m, rank0_m_vals, rtol=0, atol=0)
    np.testing.assert_allclose(got_v, rank0_v_vals, rtol=0, atol=0)
    # timestep must NOT have been broadcast (rank-local int still in place).
    assert fake_optimizer._states["timestep"] == r * 100, (
        f"rank {r}: timestep was touched: {fake_optimizer._states['timestep']}"
    )
    print(f"[rank {r}] broadcast_optimizer_state OK", flush=True)


def _test_metrics_row_global_loss(r: int, ws: int) -> None:
    """Per-rank numerator/denominator → global mean of sums."""
    # Per rank: num = (r+1) * 2, den = (r+1). With ws=2: nums=[2,4] sums=6,
    # dens=[1,2] sums=3, expected_global = 2.0.
    state = TrainingState(report_started_at_s=0.0)
    state.record_loss(
        xp.asarray((r + 1) * 2.0, dtype=xp.float32),
        total_weight=xp.asarray(float(r + 1), dtype=xp.float32),
    )
    got = state.metrics_row(eval_state=None, log_global_loss=True)["train_loss"]

    nums = [(i + 1) * 2.0 for i in range(ws)]
    dens = [float(i + 1) for i in range(ws)]
    expected_global = sum(nums) / sum(dens)
    np.testing.assert_allclose(got, expected_global, rtol=1e-6, atol=1e-6)
    print(
        f"[rank {r}] metrics_row(log_global_loss=True) OK: "
        f"got={got:.6f} expected={expected_global:.6f}",
        flush=True,
    )


def main() -> None:
    if not is_distributed():
        raise RuntimeError(
            "smoke_ddp_phase3 must be invoked via the launcher; world_size=1 here."
        )

    r = rank()
    ws = world_size()
    print(f"[rank {r}] starting Phase 3 smoke; world_size={ws}", flush=True)

    backend = get_backend()
    print(f"[rank {r}] backend ready: {type(backend).__name__}", flush=True)

    _test_broadcast_optimizer_state(r, ws)
    _test_metrics_row_global_loss(r, ws)

    backend.barrier()
    print(f"[rank {r}] barrier OK; PASS", flush=True)


if __name__ == "__main__":
    main()
