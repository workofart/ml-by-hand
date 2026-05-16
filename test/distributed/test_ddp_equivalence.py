"""
Headline DDP correctness proof: training under N mock ranks should reach the
same final parameters as a single-rank baseline that consumes the same
total data, when:

- the per-step grads on N ranks are the AllReduce-mean of independent
  microbatch grads, AND
- the baseline computes a grad of the same global batch in one shot.

This test materializes that equivalence with a tiny linear regression
problem so the check is bit-deterministic at fp32 (no nondeterminism from
threading because all reductions are summed in rank-0's serialized loop).
"""

from __future__ import annotations

import numpy as np
import pytest

from autograd import optim
from autograd.backend import IS_MLX, xp
from autograd.tensor import Tensor

from .mock import run_mock_ranks

# MLX's lazy eval is not thread-safe; calling materialize() (xp.eval) from
# multiple threads inside MockComm rendezvous segfaults the interpreter.
# DDP under MLX is not in scope for v1 (we run on CuPy or numpy), so we
# skip the multi-thread integration tests there.
pytestmark = pytest.mark.skipif(
    IS_MLX, reason="MLX lazy eval is not thread-safe; DDP path is CuPy/numpy only"
)


def _make_problem(seed: int = 0, n: int = 16, dim: int = 4):
    """A regression dataset of shape (n, dim) with a known linear target."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim)).astype(np.float32)
    true_w = rng.standard_normal((dim, 1)).astype(np.float32)
    y = X @ true_w + 0.1 * rng.standard_normal((n, 1)).astype(np.float32)
    return X, y


def _make_param(dim: int) -> Tensor:
    """Identical initialization across runs: zeros."""
    return Tensor(xp.zeros((dim, 1), dtype=xp.float32), requires_grad=True)


def _mse_loss_and_grad(W: Tensor, X: np.ndarray, y: np.ndarray) -> Tensor:
    """Compute MSE loss W^T X against y as an autograd-tracked Tensor.

    We use autograd's Tensor + backward so the gradient is what the real
    optimizer would consume, not a hand-derived approximation.
    """
    X_t = Tensor(xp.asarray(X))
    y_t = Tensor(xp.asarray(y))
    pred = X_t @ W
    diff = pred - y_t
    loss = (diff * diff).sum() / X.shape[0]
    return loss


def _get_or_make_sgd(params: dict, lr: float) -> optim.SGD:
    """SGD construction wraps a broadcast on non-rank-0; we want to make
    this once per test run, not once per step, so the broadcast is paid
    a single time."""
    return optim.SGD(params, lr=lr)


def test_ddp_2rank_matches_single_rank_baseline():
    """Two-rank mock DDP on (X1, X2) ≡ single-rank baseline on concat(X1, X2).

    Argument: each rank computes a per-half MSE gradient. After
    AllReduce-mean, the optimizer sees `(g1 + g2) / 2`. For MSE on
    disjoint halves of equal size,
        (g1 + g2) / 2 == grad(MSE on full batch)
    because MSE divides by per-batch count and the per-rank batches are
    the same size. So a single-rank baseline that uses the full batch
    must produce the same step under SGD.
    """
    X, y = _make_problem(seed=42, n=16, dim=4)
    half = X.shape[0] // 2
    X1, X2 = X[:half], X[half:]
    y1, y2 = y[:half], y[half:]
    lr = 0.1
    n_steps = 5

    # ---- baseline: 1 rank, full batch ---------------------------------
    W_baseline = _make_param(X.shape[1])
    params_baseline = {"W": W_baseline}
    optimizer_baseline = _get_or_make_sgd(params_baseline, lr)
    for _ in range(n_steps):
        W_baseline.grad = None
        loss = _mse_loss_and_grad(W_baseline, X, y)
        loss.backward()
        optimizer_baseline.step()

    # ---- DDP: 2 ranks, each takes one half ----------------------------
    def per_rank_train():
        # Match the baseline init (zeros) — no per-rank divergence.
        from autograd.distributed import rank

        W = _make_param(X.shape[1])
        params = {"W": W}
        optimizer = _get_or_make_sgd(params, lr)
        rank_X = X1 if rank() == 0 else X2
        rank_y = y1 if rank() == 0 else y2
        for _ in range(n_steps):
            W.grad = None
            loss = _mse_loss_and_grad(W, rank_X, rank_y)
            loss.backward()
            optimizer.step()
        return xp.to_numpy(W.data)

    rank_results = run_mock_ranks(2, per_rank_train)

    # Every rank ends up with the same params (AllReduce + identical init).
    for r in range(1, 2):
        np.testing.assert_allclose(
            rank_results[r],
            rank_results[0],
            err_msg=f"rank {r} disagrees with rank 0",
        )

    # And those params match the single-rank baseline.
    baseline_w = xp.to_numpy(W_baseline.data)
    np.testing.assert_allclose(
        rank_results[0],
        baseline_w,
        rtol=1e-5,
        atol=1e-5,
        err_msg="DDP 2-rank result differs from single-rank baseline",
    )


def test_broadcast_parameters_syncs_divergent_inits():
    """broadcast_parameters at optimizer construction makes rank-0's
    init the source of truth — even when rank > 0 starts with random
    weights, after Optimizer(...) constructs, every rank holds rank-0's
    weights."""
    from autograd.distributed import rank

    def per_rank():
        # Pretend init is non-deterministic by seeding per-rank.
        rng = np.random.default_rng(rank())
        W = Tensor(
            xp.asarray(rng.standard_normal((4, 1)).astype(np.float32)),
            requires_grad=True,
        )
        params = {"W": W}
        _get_or_make_sgd(params, lr=0.1)  # triggers broadcast_parameters
        return xp.to_numpy(W.data)

    results = run_mock_ranks(3, per_rank)
    for r in range(1, 3):
        np.testing.assert_array_equal(
            results[r],
            results[0],
            err_msg=f"rank {r} not synced to rank 0",
        )
