"""Latency statistics for inference timing distributions."""

import numpy as np


def summarize_latencies(xs_ms: list[float]) -> dict:
    """Latency distribution statistics: mean/std/p50/p95/p99/min/max/cv."""
    if not xs_ms:
        return {
            "mean": 0.0,
            "std": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0,
            "cv": 0.0,
        }
    a = np.asarray(xs_ms, dtype=np.float64)
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if a.size > 1 else 0.0
    cv = std / mean if mean > 0 else 0.0
    return {
        "mean": mean,
        "std": std,
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "min": float(a.min()),
        "max": float(a.max()),
        "cv": cv,
    }
