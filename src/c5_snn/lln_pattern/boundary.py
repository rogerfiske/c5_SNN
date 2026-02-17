"""Boundary-aware penalty for LLN exclusion scoring."""

from __future__ import annotations

import numpy as np

from c5_snn.lln_pattern.constants import (
    BOUNDARY_HIGH_RANGE,
    BOUNDARY_LOW_RANGE,
    DEFAULT_BOUNDARY_PENALTY,
)


def apply_boundary_penalty(
    scores: np.ndarray,
    penalty: float = DEFAULT_BOUNDARY_PENALTY,
    low_range: range = BOUNDARY_LOW_RANGE,
    high_range: range = BOUNDARY_HIGH_RANGE,
) -> np.ndarray:
    """Apply boundary penalty to frequency scores.

    Boundary values (1-7 and 33-39) appear in 79.6% of events, so
    excluding them is risky. We MULTIPLY their frequency scores by the
    penalty factor, boosting their apparent frequency and keeping them
    OUT of the exclusion set (which takes the LOWEST scores).

    Args:
        scores: (39,) frequency scores. Higher = appeared more recently.
        penalty: Multiplier for boundary values (>1 protects from exclusion).
        low_range: Range of low boundary values (default 1-7).
        high_range: Range of high boundary values (default 33-39).

    Returns:
        (39,) adjusted scores.
    """
    adjusted = scores.copy()
    for v in low_range:
        adjusted[v - 1] *= penalty
    for v in high_range:
        adjusted[v - 1] *= penalty
    return adjusted
