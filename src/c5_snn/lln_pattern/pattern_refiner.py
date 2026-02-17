"""Pattern-based refinement of LLN exclusion scores."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from c5_snn.lln_pattern.constants import HIGH_PMI_PAIRS, NUM_VALUES

logger = logging.getLogger(__name__)


def compute_repeat_scores(
    df: pd.DataFrame,
    target_idx: int,
    lookback: int = 5,
) -> np.ndarray:
    """Score values based on consecutive repeat patterns per position.

    If a value appeared in the same position repeatedly in recent events,
    it gets a likelihood boost (more likely to appear again).

    Args:
        df: Validated CA5_date DataFrame.
        target_idx: Predict for this index.
        lookback: Number of recent events to check for repeats.

    Returns:
        (39,) array of repeat-based likelihood adjustments.
    """
    scores = np.zeros(NUM_VALUES)
    start = max(0, target_idx - lookback)
    recent = df.iloc[start:target_idx]

    if len(recent) < 2:
        return scores

    for pos in range(1, 6):
        col = f"m_{pos}"
        vals = recent[col].values
        last_val = vals[-1]
        repeat_count = np.sum(vals == last_val)
        # Normalize by lookback window size
        scores[int(last_val) - 1] += repeat_count / len(vals)

    return scores


def compute_transition_scores(
    df: pd.DataFrame,
    target_idx: int,
    lookback: int = 20,
) -> np.ndarray:
    """Score values based on position-to-position transition patterns.

    For each adjacent position pair (p, p+1), compute conditional
    probabilities based on the most recent event's values.

    Args:
        df: Validated CA5_date DataFrame.
        target_idx: Predict for this index.
        lookback: Number of recent events for transition counting.

    Returns:
        (39,) array of transition-based likelihood adjustments.
    """
    scores = np.zeros(NUM_VALUES)
    start = max(0, target_idx - lookback)
    recent = df.iloc[start:target_idx]

    if len(recent) < 2:
        return scores

    last_event = df.iloc[target_idx - 1]

    for pos in range(1, 5):  # positions 1-4 (transition to pos+1)
        prev_col = f"m_{pos}"
        next_col = f"m_{pos + 1}"
        prev_val = int(last_event[prev_col])

        # Count transitions from prev_val in this position
        mask = recent[prev_col].values == prev_val
        n_matches = mask.sum()
        if n_matches > 0:
            following_vals = recent[next_col].values[mask]
            for v in following_vals:
                scores[int(v) - 1] += 1.0 / n_matches

    return scores


def compute_pmi_scores(
    df: pd.DataFrame,
    target_idx: int,
    pmi_pairs: list[tuple[int, int, int, int, float]] | None = None,
    lookback: int = 50,
) -> np.ndarray:
    """Score values using known high-PMI cross-position pairs.

    If a high-PMI partner appeared recently in its paired position,
    the counterpart value gets a likelihood boost.

    Args:
        df: Validated CA5_date DataFrame.
        target_idx: Predict for this index.
        pmi_pairs: List of (pos_a, val_a, pos_b, val_b, pmi_score).
        lookback: Number of recent events to check.

    Returns:
        (39,) array of PMI-based likelihood adjustments.
    """
    scores = np.zeros(NUM_VALUES)
    pairs = pmi_pairs if pmi_pairs is not None else HIGH_PMI_PAIRS
    start = max(0, target_idx - lookback)
    recent = df.iloc[start:target_idx]

    if len(recent) == 0:
        return scores

    last_event = df.iloc[target_idx - 1]

    for pos_a, val_a, pos_b, val_b, pmi in pairs:
        col_a = f"m_{pos_a}"
        col_b = f"m_{pos_b}"

        # If val_a appeared at pos_a in the last event, boost val_b
        if int(last_event[col_a]) == val_a:
            # Scale by normalized PMI
            scores[val_b - 1] += pmi / 10.0

        # Symmetric: if val_b at pos_b, boost val_a
        if int(last_event[col_b]) == val_b:
            scores[val_a - 1] += pmi / 10.0

    return scores


def refine_scores(
    lln_scores: np.ndarray,
    df: pd.DataFrame,
    target_idx: int,
    repeat_weight: float = 0.3,
    transition_weight: float = 0.2,
    pmi_weight: float = 0.5,
) -> np.ndarray:
    """Combine LLN base scores with pattern refinement signals.

    Higher final score = more likely to appear = LESS likely to be excluded.

    Args:
        lln_scores: (39,) base LLN likelihood scores.
        df: Validated DataFrame.
        target_idx: Predict for this index.
        repeat_weight: Weight for repeat signal.
        transition_weight: Weight for transition signal.
        pmi_weight: Weight for PMI signal.

    Returns:
        (39,) refined likelihood scores.
    """
    repeat = compute_repeat_scores(df, target_idx)
    transition = compute_transition_scores(df, target_idx)
    pmi = compute_pmi_scores(df, target_idx)

    return (
        lln_scores
        + repeat_weight * repeat
        + transition_weight * transition
        + pmi_weight * pmi
    )
