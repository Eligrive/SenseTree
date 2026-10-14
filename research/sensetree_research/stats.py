"""Paired significance testing over queries.

Two systems evaluated on the same queries give paired per-query scores. We report the
mean difference with a percentile bootstrap confidence interval, and a two-sided
randomisation (sign-flip) test, which Smucker et al. (CIKM 2007) recommend for IR.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Comparison:
    mean_a: float
    mean_b: float
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    n: int


def _paired(a: dict[str, float], b: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    keys = sorted(set(a) & set(b))
    if len(keys) != len(a) or len(keys) != len(b):
        raise ValueError("systems were not evaluated on the same queries")
    return np.array([a[k] for k in keys]), np.array([b[k] for k in keys])


def bootstrap_ci(values: np.ndarray, n_resamples: int = 10_000, alpha: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_resamples, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def randomisation_test(diff: np.ndarray, n_resamples: int = 10_000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    observed = abs(diff.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_resamples, len(diff)))
    null = np.abs((signs * diff).mean(axis=1))
    return float((np.sum(null >= observed - 1e-12) + 1) / (n_resamples + 1))


def compare(a: dict[str, float], b: dict[str, float], n_resamples: int = 10_000) -> Comparison:
    """Compares system B against baseline A. ``delta`` is mean(B) - mean(A)."""
    va, vb = _paired(a, b)
    diff = vb - va
    lo, hi = bootstrap_ci(diff, n_resamples)
    return Comparison(
        mean_a=float(va.mean()),
        mean_b=float(vb.mean()),
        delta=float(diff.mean()),
        ci_low=lo,
        ci_high=hi,
        p_value=randomisation_test(diff, n_resamples),
        n=len(diff),
    )
