#!/usr/bin/env python3
"""Run a short DYGKT profiling job and summarize bottlenecks.

This wrapper calls train.py with DYGKT and enables trainer-side profiling
(via --profile_batches) so you can quickly see whether data wait or model
compute dominates runtime.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile DYGKT training efficiency")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. ASSISTments12")
    parser.add_argument("--fold", type=int, default=0, help="Fold index")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs for quick profile run")
    parser.add_argument("--batch_size", type=int, default=2000, help="Batch size")
    parser.add_argument("--profile_batches", type=int, default=50, help="How many train batches to profile")
    parser.add_argument("--device", type=str, default="cuda", help="Device passed to train.py")
    parser.add_argument("--no_cache", action="store_true", help="Disable dataset cache")
    parser.add_argument("--cache_dir", type=str, default=None, help="Override DYGKT cache directory")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to train.py, place after '--'.",
    )
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "train.py",
        "-m",
        "DYGKT",
        "--dataset",
        args.dataset,
        "--fold",
        str(args.fold),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--profile_batches",
        str(args.profile_batches),
        "--device",
        args.device,
    ]

    if args.no_cache:
        cmd.append("--no_cache")
    if args.cache_dir:
        cmd.extend(["--cache_dir", args.cache_dir])
    if args.extra_args:
        cmd.extend(args.extra_args)

    return cmd


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    cmd = build_command(args)

    print("=" * 80)
    print("DYGKT profiling runner")
    print("=" * 80)
    print("Working directory:", repo_root)
    print("Command:", shlex.join(cmd))
    print("=" * 80)

    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=repo_root, check=False)
    elapsed = time.perf_counter() - start

    print("=" * 80)
    print(f"Finished with code: {proc.returncode}")
    print(f"Wall-clock time: {elapsed:.2f}s")
    print("=" * 80)
    print("Tip: check logs for lines containing 'DYGKT profile summary' and 'DYGKT profile breakdown'.")

    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
