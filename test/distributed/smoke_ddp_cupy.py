"""End-to-end Phase 1 smoke on CuPy: tiny model trained under 2-rank mock
DDP must reach the same final params as a single-rank baseline using the
full global batch.

Not a pytest test — invoked as a module so relative imports work:

    AUTOGRAD_BACKEND=cupy .venv/bin/python -m test.distributed.smoke_ddp_cupy

This complements the in-tree pytest equivalence test by exercising real
CuPy buffers (not just numpy arrays in threads) through the same
hooks.allreduce_grads / broadcast_parameters code path that the real
NCCL backend will hit in Phase 2.
"""

from __future__ import annotations

import numpy as np

from autograd import optim
from autograd.backend import xp
from autograd.tensor import Tensor

from .mock import run_mock_ranks


def _make_problem(seed: int = 42, n: int = 16, dim: int = 4):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim)).astype(np.float32)
    true_w = rng.standard_normal((dim, 1)).astype(np.float32)
    y = X @ true_w + 0.1 * rng.standard_normal((n, 1)).astype(np.float32)
    return X, y


def _mse_loss(W: Tensor, X, y) -> Tensor:
    X_t = Tensor(xp.asarray(X))
    y_t = Tensor(xp.asarray(y))
    pred = X_t @ W
    diff = pred - y_t
    return (diff * diff).sum() / X.shape[0]


def _train(W: Tensor, X, y, *, steps: int, lr: float) -> Tensor:
    params = {"W": W}
    optimizer = optim.SGD(params, lr=lr)
    for _ in range(steps):
        W.grad = None
        loss = _mse_loss(W, X, y)
        loss.backward()
        optimizer.step()
    return W


def main() -> None:
    X, y = _make_problem(seed=42, n=16, dim=4)
    half = X.shape[0] // 2
    X1, X2 = X[:half], X[half:]
    y1, y2 = y[:half], y[half:]
    lr = 0.1
    n_steps = 5

    # ---- baseline ----
    W_base = Tensor(xp.zeros((X.shape[1], 1), dtype=xp.float32), requires_grad=True)
    _train(W_base, X, y, steps=n_steps, lr=lr)
    baseline = xp.to_numpy(W_base.data)
    print(f"single-rank baseline W = {baseline.ravel()}")

    # ---- DDP 2 ranks ----
    def per_rank():
        from autograd.distributed import rank

        W = Tensor(xp.zeros((X.shape[1], 1), dtype=xp.float32), requires_grad=True)
        rank_X = X1 if rank() == 0 else X2
        rank_y = y1 if rank() == 0 else y2
        _train(W, rank_X, rank_y, steps=n_steps, lr=lr)
        return xp.to_numpy(W.data)

    ddp_results = run_mock_ranks(2, per_rank)
    for r, W in enumerate(ddp_results):
        print(f"rank {r} final W = {W.ravel()}")

    # ---- equivalence check ----
    for r, W in enumerate(ddp_results):
        np.testing.assert_allclose(
            W,
            baseline,
            rtol=1e-4,
            atol=1e-4,
            err_msg=f"rank {r} disagrees with baseline",
        )
    # And ranks agree.
    np.testing.assert_allclose(ddp_results[0], ddp_results[1])
    print("PASS: 2-rank mock DDP on CuPy matches single-rank baseline")


if __name__ == "__main__":
    main()
