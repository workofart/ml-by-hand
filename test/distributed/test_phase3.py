"""Phase 3 — checkpoint broadcast + global-loss AllReduce-mean.

Two pieces of Phase 3 plumbing get tested here against the mock backend:

- `broadcast_optimizer_state(optimizer)` walks `Optimizer._states`, copies
  every tensor-valued slot from rank 0 to all ranks, and leaves non-tensor
  slots (e.g. the integer `timestep`) untouched. The fp32-promotion path
  used by `broadcast_parameters` for bf16 also applies here.

- `TrainingState.metrics_row` with `log_global_loss=True` packs additive report
  counters into one vector, AllReduce-sums that vector, then reports the
  true global weighted mean. With the flag off, loss stays rank-local.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from autograd.backend import IS_MLX, LOW_PRECISION_FLOAT_DTYPES, xp
from autograd.distributed import (
    broadcast_optimizer_state,
    rank,
    world_size,
)
from autograd.tools.trainer import TrainingState

from .mock import run_mock_ranks

# ---- broadcast_optimizer_state ------------------------------------------


def test_broadcast_optimizer_state_syncs_from_rank_zero():
    """Every rank ends up with rank 0's `m` and `v` tensors; `timestep`
    (non-dict slot) is left alone on every rank."""
    ws = 3
    rank0_m = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    rank0_v = np.array([0.01, 0.02, 0.03], dtype=np.float32)

    def target():
        m_val = rank0_m.copy() if rank() == 0 else np.zeros(3, dtype=np.float32)
        v_val = rank0_v.copy() if rank() == 0 else np.zeros(3, dtype=np.float32)
        # `timestep` is intentionally rank-local here so we can assert that
        # broadcast_optimizer_state never overwrites it.
        fake_optimizer = SimpleNamespace(
            _states={
                "m": {"layer.w": m_val},
                "v": {"layer.w": v_val},
                "timestep": rank() * 100,
            }
        )
        broadcast_optimizer_state(fake_optimizer, from_rank=0)
        return fake_optimizer._states

    results = run_mock_ranks(ws, target)
    for r, out in enumerate(results):
        np.testing.assert_array_equal(
            out["m"]["layer.w"], rank0_m, err_msg=f"rank {r} m"
        )
        np.testing.assert_array_equal(
            out["v"]["layer.w"], rank0_v, err_msg=f"rank {r} v"
        )
        assert out["timestep"] == r * 100, f"rank {r} timestep was touched"


def test_broadcast_optimizer_state_noop_single_rank():
    """world_size==1 — the helper must not touch any state."""
    original_m = np.array([1.0, 2.0], dtype=np.float32)
    fake_optimizer = SimpleNamespace(
        _states={"m": {"layer.w": original_m.copy()}, "timestep": 7},
    )
    assert world_size() == 1
    broadcast_optimizer_state(fake_optimizer, from_rank=0)
    np.testing.assert_array_equal(fake_optimizer._states["m"]["layer.w"], original_m)
    assert fake_optimizer._states["timestep"] == 7


@pytest.mark.skipif(
    not LOW_PRECISION_FLOAT_DTYPES,
    reason="backend does not provide a low-precision float dtype",
)
def test_broadcast_optimizer_state_fp32_promotion_round_trip():
    """bf16/fp16 optimizer state survives the broadcast: dtype preserved,
    values match rank-0's bytes round-tripped through fp32."""
    low_dtype = LOW_PRECISION_FLOAT_DTYPES[0]
    ws = 3
    rank0_m_fp32 = np.array([0.25, 0.5, 0.75], dtype=np.float32)
    rank0_m = xp.asarray(rank0_m_fp32).astype(low_dtype)

    def target():
        m_val = xp.array(rank0_m) if rank() == 0 else xp.zeros(3, dtype=low_dtype)
        fake_optimizer = SimpleNamespace(_states={"m": {"layer.w": m_val}})
        broadcast_optimizer_state(fake_optimizer, from_rank=0)
        return fake_optimizer._states["m"]["layer.w"]

    results = run_mock_ranks(ws, target)
    for r, out in enumerate(results):
        assert out.dtype == low_dtype, f"rank {r} dtype drifted: {out.dtype}"
        np.testing.assert_allclose(
            xp.to_numpy(out.astype(xp.float32)),
            xp.to_numpy(rank0_m.astype(xp.float32)),
            atol=0,
            err_msg=f"rank {r}",
        )


# ---- report metric global allreduce path --------------------------------


def test_weighted_mean_is_local_math():
    """The ratio helper returns rank-local means and runs no collective."""
    ws = 4

    def target():
        # Each rank reports a different ratio. With the flag off, each
        # rank returns its own ratio unchanged.
        num = xp.array(2.0 * (rank() + 1), dtype=xp.float32)
        den = xp.array(1.0, dtype=xp.float32)
        return TrainingState()._weighted_mean(num, den)

    results = run_mock_ranks(ws, target)
    for r, out in enumerate(results):
        assert out == pytest.approx(2.0 * (r + 1)), f"rank {r}"


@pytest.mark.skipif(
    IS_MLX,
    reason=(
        "MLX lazy graphs aren't thread-safe: metrics_row's inline "
        "xp.to_scalar after AllReduce evaluates a cross-thread graph "
        "and segfaults under the in-process mock. Production DDP runs "
        "one process per rank, so this is a mock-only concern."
    ),
)
def test_metrics_row_global_allreduces_loss_when_flag_on():
    """With log_global_loss=True, every rank returns the global weighted
    mean: sum(numerators) / sum(denominators), not the per-rank ratio."""
    ws = 4
    numerators = [2.0, 4.0, 6.0, 8.0]  # sum = 20.0
    denominators = [1.0, 2.0, 1.0, 4.0]  # sum = 8.0
    expected_global = sum(numerators) / sum(denominators)  # 2.5

    def target():
        state = TrainingState(report_started_at_s=0.0)
        state.record_loss(
            xp.array(numerators[rank()], dtype=xp.float32),
            total_weight=xp.array(denominators[rank()], dtype=xp.float32),
        )
        return state.metrics_row(
            eval_state=None,
            log_global_loss=True,
        )["train_loss"]

    results = run_mock_ranks(ws, target)
    for r, out in enumerate(results):
        assert out == pytest.approx(expected_global, rel=1e-6), f"rank {r}"


def test_weighted_mean_none_numerator_returns_zero():
    """The early `numerator is None` short-circuit must remain — every
    rank returns 0.0 without entering the collective. (When None comes
    from `state.report_loss_sum`, all ranks observe it consistently, so
    no collective desync.)"""
    out = TrainingState()._weighted_mean(
        None,
        xp.array(1.0, dtype=xp.float32),
    )
    assert out == 0.0
