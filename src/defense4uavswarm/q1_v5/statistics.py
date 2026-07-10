from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def paired_bootstrap_delta(base: np.ndarray, method: np.ndarray, n_resamples: int = 10000, seed: int = 2026) -> dict[str, float]:
    base = np.asarray(base, dtype=float)
    method = np.asarray(method, dtype=float)
    if len(base) != len(method):
        raise ValueError("Paired bootstrap arrays must have the same length")
    rng = np.random.default_rng(seed)
    deltas = method - base
    samples = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, len(deltas), len(deltas))
        samples[i] = float(deltas[idx].mean())
    return {
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "ci95_low": float(np.percentile(samples, 2.5)),
        "ci95_high": float(np.percentile(samples, 97.5)),
    }


def noninferior(ci95_low: float, margin: float) -> bool:
    return float(ci95_low) > -float(margin)


def superiority_lower_is_better(ci95_high: float) -> bool:
    return float(ci95_high) < 0.0


def exact_sign_permutation_pvalue(deltas: np.ndarray) -> float:
    deltas = np.asarray(deltas, dtype=float)
    nonzero = deltas[np.abs(deltas) > 1e-12]
    if len(nonzero) == 0:
        return 1.0
    observed = abs(float(nonzero.mean()))
    count = 0
    extreme = 0
    for signs in itertools.product([-1.0, 1.0], repeat=len(nonzero)):
        val = abs(float((nonzero * np.asarray(signs)).mean()))
        count += 1
        if val >= observed - 1e-12:
            extreme += 1
    return extreme / max(1, count)

