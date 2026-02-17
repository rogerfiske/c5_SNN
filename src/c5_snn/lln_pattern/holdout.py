"""Holdout evaluation for the LLN-Pattern pipeline."""

from __future__ import annotations

import logging

import pandas as pd

from c5_snn.lln_pattern.metrics import compute_n_wrong, compute_n_wrong_distribution
from c5_snn.lln_pattern.pipeline import predict_exclusion_set

logger = logging.getLogger(__name__)


def run_lln_holdout_test(
    df: pd.DataFrame,
    n_holdout: int | None = None,
    holdout_fraction: float = 0.10,
    k_exclude: int = 20,
    use_pattern: bool = True,
    use_boundary: bool = True,
    boundary_penalty: float = 1.7,
    **pipeline_kwargs: float,
) -> dict:
    """Run strict holdout evaluation on the last N events.

    No data leakage: for each holdout event, the pipeline only sees
    data BEFORE that event's index.

    Args:
        df: Full validated CA5_date DataFrame.
        n_holdout: Explicit number of holdout events. If None, uses
                   holdout_fraction of total.
        holdout_fraction: Fraction of data to hold out (default 10%).
        k_exclude: Size of exclusion set.
        use_pattern: Whether to use pattern refinement.
        use_boundary: Whether to use boundary penalty.
        boundary_penalty: Boundary penalty factor.
        **pipeline_kwargs: Passed to predict_exclusion_set.

    Returns:
        Dict with keys:
            summary: N-wrong distribution statistics
            per_sample: List of per-event results
            config: Pipeline configuration used
    """
    if n_holdout is None:
        n_holdout = max(1, int(len(df) * holdout_fraction))

    min_required = n_holdout + 50  # need some history for LLN
    if len(df) < min_required:
        raise ValueError(
            f"Need at least {min_required} rows, have {len(df)}."
        )

    m_cols = [f"m_{j}" for j in range(1, 6)]
    all_n_wrong: list[int] = []
    per_sample: list[dict] = []

    for i in range(n_holdout):
        offset = n_holdout - i
        target_idx = len(df) - offset

        # Actual values for this event
        actual = df.iloc[target_idx][m_cols].values.astype(int)

        # Predict
        result = predict_exclusion_set(
            df,
            target_idx,
            k_exclude=k_exclude,
            use_pattern=use_pattern,
            use_boundary=use_boundary,
            boundary_penalty=boundary_penalty,
            **pipeline_kwargs,
        )

        n_wrong = compute_n_wrong(actual, result["excluded_values"])
        all_n_wrong.append(n_wrong)

        per_sample.append(
            {
                "row_index": target_idx,
                "date": str(df["date"].iloc[target_idx]),
                "actual_values": actual.tolist(),
                "excluded_values": result["excluded_values"].tolist(),
                "n_wrong": n_wrong,
            }
        )

    summary = compute_n_wrong_distribution(all_n_wrong)

    config = {
        "n_holdout": n_holdout,
        "k_exclude": k_exclude,
        "use_pattern": use_pattern,
        "use_boundary": use_boundary,
        "boundary_penalty": boundary_penalty,
    }

    return {
        "summary": summary,
        "per_sample": per_sample,
        "config": config,
    }
