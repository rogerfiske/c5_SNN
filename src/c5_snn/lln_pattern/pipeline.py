"""LLN-Pattern prediction pipeline orchestrator."""

from __future__ import annotations

import logging

import pandas as pd

from c5_snn.lln_pattern.boundary import apply_boundary_penalty
from c5_snn.lln_pattern.constants import (
    DEFAULT_BOUNDARY_PENALTY,
    DEFAULT_K_EXCLUDE,
    DEFAULT_RECENCY_DECAY,
)
from c5_snn.lln_pattern.lln_scorer import compute_lln_scores, select_exclusion_set
from c5_snn.lln_pattern.pattern_refiner import refine_scores

logger = logging.getLogger(__name__)


def predict_exclusion_set(
    df: pd.DataFrame,
    target_idx: int,
    k_exclude: int = DEFAULT_K_EXCLUDE,
    use_pattern: bool = True,
    use_boundary: bool = True,
    boundary_penalty: float = DEFAULT_BOUNDARY_PENALTY,
    position_params: dict | None = None,
    decay: float = DEFAULT_RECENCY_DECAY,
    repeat_weight: float = 0.3,
    transition_weight: float = 0.2,
    pmi_weight: float = 0.5,
) -> dict:
    """Run the full LLN-Pattern pipeline for a single prediction.

    Pipeline stages:
      1. LLN scorer: recency-weighted frequency per position
      2. Pattern refiner: repeats + transitions + PMI (optional)
      3. Boundary penalty: boost boundary value likelihood (optional)
      4. Select exclusion set: K values with LOWEST final scores

    Args:
        df: Validated CA5_date DataFrame.
        target_idx: Index of the event to predict.
        k_exclude: Number of values to exclude (default 20).
        use_pattern: Whether to apply pattern refinement.
        use_boundary: Whether to apply boundary penalty.
        boundary_penalty: Multiplicative penalty for boundary values.
        position_params: Override per-position LLN params.
        decay: Recency decay factor.
        repeat_weight: Weight for repeat pattern signal.
        transition_weight: Weight for transition pattern signal.
        pmi_weight: Weight for PMI pattern signal.

    Returns:
        Dict with keys:
            excluded_values: sorted array of K excluded value numbers (1-indexed)
            scores: (39,) final likelihood scores
            lln_scores: (39,) raw LLN scores before refinement
    """
    # Stage 1: LLN base scores
    lln_scores = compute_lln_scores(
        df,
        target_idx,
        position_params=position_params,
        decay=decay,
    )

    # Stage 2: Pattern refinement
    if use_pattern:
        refined = refine_scores(
            lln_scores,
            df,
            target_idx,
            repeat_weight=repeat_weight,
            transition_weight=transition_weight,
            pmi_weight=pmi_weight,
        )
    else:
        refined = lln_scores.copy()

    # Stage 3: Boundary penalty
    if use_boundary:
        final = apply_boundary_penalty(refined, penalty=boundary_penalty)
    else:
        final = refined

    # Stage 4: Select exclusion set
    excluded_values = select_exclusion_set(final, k=k_exclude)

    return {
        "excluded_values": excluded_values,
        "scores": final,
        "lln_scores": lln_scores,
    }
