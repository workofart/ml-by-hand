"""Single-node multi-GPU launcher for autograd DDP.

Usage:

    python -m autograd.distributed.launch --nproc-per-node=N script.py [args...]

Spawns N child processes, one per GPU on this node. Each child sees these
env vars (which `autograd.distributed` and `autograd.backend` read at
import time):

- ``WORLD_SIZE``  total number of ranks (= N)
- ``RANK``        global rank, ``0..N-1`` (same as local on single node)
- ``LOCAL_RANK``  GPU index this rank pins to via ``cp.cuda.Device(LR).use()``
- ``MASTER_ADDR`` rendezvous host for the NCCL bootstrap (default 127.0.0.1)
- ``MASTER_PORT`` rendezvous port; auto-picks a free one if ``--master-port=0``
- ``SEED``        forwarded to the script for deterministic init

The launcher is intentionally simple: blocking ``subprocess.Popen`` per
rank, ``signal`` to propagate Ctrl+C, ``wait()`` to collect exit codes.
There is no fault tolerance and no elastic scaling — those belong in
torchrun's complexity bracket, not here. If a child fails, the launcher
waits for the rest to finish and exits with the worst observed code.

Why subprocess instead of fork/multiprocessing?
    Each rank needs its own CuPy/CUDA context. ``fork`` after CUDA init
    is undefined behaviour; ``multiprocessing.spawn`` works but adds a
    pickling-the-target layer. Plain ``subprocess`` is the cleanest
    one-step-removed model and matches how torchrun launches its workers.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
from typing import Sequence


def _find_free_port() -> int:
    """Bind a socket to port 0 (OS picks any free port), read it, release.

    Tiny race window: another process could bind the same port between the
    close here and the NCCL server bind in rank 0. In practice the window
    is sub-millisecond and the launcher is the only consumer on a typical
    training box.
    """
    s = socket.socket()
    try:
        s.bind(("", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def _spawn_rank(
    *,
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    seed: int,
    script: str,
    script_args: Sequence[str],
) -> subprocess.Popen:
    """Start one child process for ``rank``. Inherits stdin/stdout/stderr so
    rank-0 logs land in the launcher's terminal and non-zero ranks stay
    quiet (the trainer raises their log level via the rank-zero gate).
    """
    env = os.environ.copy()
    env["WORLD_SIZE"] = str(world_size)
    env["RANK"] = str(rank)
    env["LOCAL_RANK"] = str(rank)
    env["MASTER_ADDR"] = master_addr
    env["MASTER_PORT"] = str(master_port)
    env["SEED"] = str(seed)
    return subprocess.Popen(
        [sys.executable, script, *script_args],
        env=env,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m autograd.distributed.launch",
        description="Single-node multi-GPU launcher for autograd DDP.",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        required=True,
        help="Number of ranks (= number of GPUs to use on this node).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed forwarded to children via SEED env var (default: 42).",
    )
    parser.add_argument(
        "--master-addr",
        default="127.0.0.1",
        help="NCCL rendezvous host (default: 127.0.0.1; single-node only).",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=0,
        help="NCCL rendezvous port; 0 picks a free port (default: 0).",
    )
    parser.add_argument("script", help="Path to the training script.")
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded verbatim to the script.",
    )
    args = parser.parse_args(argv)

    if args.nproc_per_node < 1:
        parser.error(f"--nproc-per-node must be >= 1, got {args.nproc_per_node}")

    if args.master_port == 0:
        args.master_port = _find_free_port()

    children = [
        _spawn_rank(
            rank=r,
            world_size=args.nproc_per_node,
            master_addr=args.master_addr,
            master_port=args.master_port,
            seed=args.seed,
            script=args.script,
            script_args=args.script_args,
        )
        for r in range(args.nproc_per_node)
    ]

    print(
        f"[launcher] spawned {len(children)} ranks "
        f"on {args.master_addr}:{args.master_port}",
        file=sys.stderr,
        flush=True,
    )

    # Forward Ctrl+C / SIGTERM to every child so the user can abort cleanly.
    def _forward(signum: int, _frame: object) -> None:
        for proc in children:
            try:
                proc.send_signal(signum)
            except ProcessLookupError:
                # Child already exited; nothing to do.
                pass

    signal.signal(signal.SIGINT, _forward)
    signal.signal(signal.SIGTERM, _forward)

    exit_codes = [proc.wait() for proc in children]
    failures = [(r, c) for r, c in enumerate(exit_codes) if c != 0]
    if failures:
        for r, code in failures:
            print(
                f"[launcher] rank {r} exited with code {code}",
                file=sys.stderr,
                flush=True,
            )
        # Return the worst observed code; this becomes the launcher's exit.
        return max(c for _, c in failures)
    return 0


if __name__ == "__main__":
    sys.exit(main())
