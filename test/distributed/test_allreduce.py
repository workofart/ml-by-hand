"""Mock-backend AllReduce / broadcast / barrier correctness tests.

These tests run multiple "ranks" as Python threads via `run_mock_ranks`,
verifying that:

- AllReduce summed across N ranks matches the host-side sum.
- The fp32-promotion path for low-precision gradients preserves the bf16
  end-to-end semantics — `param.grad.data` ends up with the original
  low-precision dtype and the average of the per-rank low-precision inputs
  rounded once at the very end.
- `broadcast` copies the `from_rank` rank's buffer to every other rank.
- `barrier` is a no-op aside from forcing rendezvous; we check that all
  threads finish a barrier-only target.
"""

from __future__ import annotations

import numpy as np
import pytest

from autograd.backend import LOW_PRECISION_FLOAT_DTYPES, xp
from autograd.distributed import (
    ReduceOp,
    allreduce_grads,
    broadcast_parameters,
    get_backend,
    rank,
    world_size,
)

from .mock import run_mock_ranks


# A minimal Tensor stand-in that mimics the (param.grad.data, param.data)
# attribute shape used by allreduce_grads / broadcast_parameters. We
# deliberately don't import autograd.tensor.Tensor so the test can run
# without setting up the full Module/Tensor stack.
class _FakeParam:
    def __init__(self, data, grad=None):
        self.data = data
        self.grad = _FakeGrad(grad) if grad is not None else None


class _FakeGrad:
    def __init__(self, data):
        self.data = data


# ---- raw collectives -----------------------------------------------------


def test_all_reduce_sum_matches_host_sum():
    """Sum of per-rank arrays equals the AllReduce result on every rank."""
    ws = 4
    per_rank_inputs = [
        np.array([1.0, 2.0, 3.0], dtype=np.float32) * (r + 1) for r in range(ws)
    ]
    expected = sum(per_rank_inputs)

    def target():
        backend = get_backend()
        # all_reduce returns the reduced array; caller binds the result
        # so the interface works for MLX (no in-place writes) and CuPy
        # alike.
        return backend.all_reduce(per_rank_inputs[rank()].copy(), op=ReduceOp.SUM)

    results = run_mock_ranks(ws, target)
    for r, out in enumerate(results):
        np.testing.assert_array_equal(out, expected, err_msg=f"rank {r}")


def test_broadcast_copies_from_rank_to_all():
    """Every rank ends up with the broadcaster's buffer contents."""
    ws = 3
    from_rank = 1
    broadcast_value = np.array([7.0, 8.0, 9.0], dtype=np.float32)

    def target():
        backend = get_backend()
        if rank() == from_rank:
            buf = broadcast_value.copy()
        else:
            buf = np.zeros(3, dtype=np.float32)
        return backend.broadcast(buf, from_rank=from_rank)

    results = run_mock_ranks(ws, target)
    for r, out in enumerate(results):
        np.testing.assert_array_equal(out, broadcast_value, err_msg=f"rank {r}")


def test_barrier_completes_on_all_ranks():
    """All threads return successfully after a barrier rendezvous."""
    ws = 4

    def target():
        backend = get_backend()
        backend.barrier()
        return rank()

    results = run_mock_ranks(ws, target)
    assert sorted(results) == list(range(ws))


# ---- hooks against fake-param shape --------------------------------------


def test_allreduce_grads_averages_across_ranks():
    """allreduce_grads divides the summed grads by world_size."""
    ws = 4
    base_grads = [
        np.array([1.0, 2.0, 3.0], dtype=np.float32) * (r + 1) for r in range(ws)
    ]
    expected_mean = sum(base_grads) / ws

    def target():
        params = {
            "w": _FakeParam(
                data=np.zeros(3, dtype=np.float32),
                grad=base_grads[rank()].copy(),
            )
        }
        allreduce_grads(params)
        return params["w"].grad.data

    results = run_mock_ranks(ws, target)
    for r, out in enumerate(results):
        np.testing.assert_allclose(out, expected_mean, err_msg=f"rank {r}")


@pytest.mark.skipif(
    not LOW_PRECISION_FLOAT_DTYPES,
    reason="backend does not provide a low-precision float dtype",
)
def test_allreduce_grads_fp32_promotion_for_low_precision():
    """bf16/fp16 grads reduce in fp32 then cast back; the final dtype
    matches the input, but the precision is what fp32-reduction gives."""
    low_dtype = LOW_PRECISION_FLOAT_DTYPES[0]
    ws = 4
    base_vals = [
        np.array([1.0, 2.0, 3.0], dtype=np.float32) * (r + 1) for r in range(ws)
    ]
    # Round-trip through the backend's low dtype so the expectation matches
    # what each rank actually starts with.
    base_grads = [xp.asarray(v).astype(low_dtype) for v in base_vals]
    # Expected: average in fp32, then cast to low_dtype once at the end.
    summed_fp32 = sum(xp.asarray(g).astype(xp.float32) for g in base_grads)
    expected = (summed_fp32 / ws).astype(low_dtype)

    def target():
        # xp.array(...) copies the backend array (works on MLX where
        # .copy() is missing). The fp32-promotion path inside
        # allreduce_grads is what's under test, not the buffer plumbing.
        params = {
            "w": _FakeParam(
                data=xp.zeros(3, dtype=low_dtype),
                grad=xp.array(base_grads[rank()]),
            )
        }
        allreduce_grads(params)
        return params["w"].grad.data

    results = run_mock_ranks(ws, target)
    for r, out in enumerate(results):
        assert out.dtype == low_dtype, f"rank {r} dtype: {out.dtype}"
        # Compare as fp32 to avoid bf16 stringification noise.
        np.testing.assert_allclose(
            xp.to_numpy(out.astype(xp.float32)),
            xp.to_numpy(expected.astype(xp.float32)),
            atol=0,  # the operation chain is bit-deterministic
        )


def test_broadcast_parameters_syncs_to_all_ranks():
    """broadcast_parameters copies rank-0 weights to every rank."""
    ws = 3
    rank0_weights = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    def target():
        # Each rank starts with a different weight init.
        params = {
            "w": _FakeParam(
                data=(
                    rank0_weights.copy()
                    if rank() == 0
                    else np.array([9.0, 9.0, 9.0, 9.0], dtype=np.float32)
                ),
                grad=None,
            )
        }
        broadcast_parameters(params, from_rank=0)
        return params["w"].data

    results = run_mock_ranks(ws, target)
    for r, out in enumerate(results):
        np.testing.assert_array_equal(out, rank0_weights, err_msg=f"rank {r}")


# ---- single-rank short-circuit -------------------------------------------


def test_allreduce_grads_noop_single_rank():
    """When world_size==1, allreduce_grads must not touch the gradient."""
    grad = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    params = {"w": _FakeParam(data=np.zeros(3), grad=grad.copy())}
    # Calling from the main thread, no run_mock_ranks => world_size == 1.
    assert world_size() == 1
    allreduce_grads(params)
    np.testing.assert_array_equal(params["w"].grad.data, grad)
