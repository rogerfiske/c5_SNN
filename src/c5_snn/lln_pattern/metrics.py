"""Metrics for exclusion-set prediction (LLN-Pattern pipeline)."""

from __future__ import annotations

from collections import Counter

import numpy as np


def compute_n_wrong(
    actual_values: np.ndarray,
    excluded_values: np.ndarray,
) -> int:
    """Count how many actual values were incorrectly excluded.

    n_wrong = |actual_values intersection excluded_values|.
    A perfect prediction has n_wrong = 0.

    Args:
        actual_values: Array of 5 actual values (1-indexed, 1-39).
        excluded_values: Array of K excluded values (1-indexed, 1-39).

    Returns:
        Integer count of wrongly excluded values (0 = perfect).
    """
    return len(set(int(v) for v in actual_values) & set(int(v) for v in excluded_values))


def compute_n_wrong_distribution(
    all_n_wrong: list[int],
) -> dict:
    """Compute distribution statistics for N-wrong across holdout.

    Args:
        all_n_wrong: List of n_wrong values, one per holdout event.

    Returns:
        Dict with keys: total, zero_wrong_count, zero_wrong_pct,
        mean_wrong, distribution.
    """
    counter = Counter(all_n_wrong)
    total = len(all_n_wrong)
    zero_count = counter.get(0, 0)
    return {
        "total": total,
        "zero_wrong_count": zero_count,
        "zero_wrong_pct": (
            round(100.0 * zero_count / total, 2) if total > 0 else 0.0
        ),
        "mean_wrong": (
            round(float(np.mean(all_n_wrong)), 4) if total > 0 else 0.0
        ),
        "distribution": dict(sorted(counter.items())),
    }
