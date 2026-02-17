"""Recency-Weighted LLN Scorer for exclusion-based prediction."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from c5_snn.lln_pattern.constants import (
    DEFAULT_RECENCY_DECAY,
    NUM_VALUES,
    POSITION_PARAMS,
)

logger = logging.getLogger(__name__)


def compute_position_frequencies(
    values: np.ndarray,
    recent_n: int,
    decay: float = DEFAULT_RECENCY_DECAY,
) -> np.ndarray:
    """Compute recency-weighted frequency for one position.

    Args:
        values: 1D array of historical values for a single position
                (ints 1-39). The last element is the most recent.
        recent_n: Number of most recent events to use.
        decay: Exponential decay factor (0 < decay < 1). More recent
               events get higher weight.

    Returns:
        (39,) array of weighted frequency scores for values 1-39.
    """
    if recent_n <= 0 or len(values) == 0:
        return np.zeros(NUM_VALUES)

    recent_n = min(recent_n, len(values))
    recent_data = values[-recent_n:]

    # Weights: newest gets 1.0, oldest gets decay^(recent_n-1)
    weights = decay ** np.arange(recent_n - 1, -1, -1, dtype=float)

    frequencies = np.zeros(NUM_VALUES)
    for i, val in enumerate(recent_data):
        idx = int(val) - 1  # 1-indexed to 0-indexed
        if 0 <= idx < NUM_VALUES:
            frequencies[idx] += weights[i]

    return frequencies


def compute_lln_scores(
    df: pd.DataFrame,
    target_idx: int,
    position_params: dict | None = None,
    decay: float = DEFAULT_RECENCY_DECAY,
) -> np.ndarray:
    """Compute aggregated LLN likelihood scores across all 5 positions.

    For each position, computes recency-weighted frequency of each value.
    The aggregate score across all 5 positions measures how frequently
    a value has been appearing recently — higher = more recent activity.

    Args:
        df: Validated CA5_date DataFrame.
        target_idx: Index of the event to predict (uses data before this).
        position_params: Override POSITION_PARAMS.
        decay: Exponential decay factor.

    Returns:
        (39,) array of aggregated frequency scores.
    """
    params = position_params or POSITION_PARAMS
    aggregate = np.zeros(NUM_VALUES)

    for pos in range(1, 6):
        col = f"m_{pos}"
        pos_values = df[col].values[:target_idx]

        if len(pos_values) == 0:
            continue

        p = params.get(
            pos,
            {"recent_fraction": 0.08, "percentile_threshold": 80.0},
        )
        recent_n = max(1, int(p["recent_fraction"] * target_idx))

        pos_freq = compute_position_frequencies(pos_values, recent_n, decay)
        aggregate += pos_freq

    return aggregate


def select_exclusion_set(
    scores: np.ndarray,
    k: int = 20,
) -> np.ndarray:
    """Select the K values with LOWEST frequency scores for exclusion.

    Values with the lowest recent frequency are least likely to appear
    in the next event and are the best candidates for the exclusion set.

    Args:
        scores: (39,) array of recency-weighted frequency scores
                (higher = appeared more recently).
        k: Number of values to exclude.

    Returns:
        Sorted array of K value numbers (1-indexed, 1-39).
    """
    k = min(k, len(scores))
    indices_0based = np.argsort(scores)[:k]
    values_1indexed = indices_0based + 1
    return np.sort(values_1indexed)
