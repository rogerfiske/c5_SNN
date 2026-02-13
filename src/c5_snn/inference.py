"""Inference helpers: prediction windows, checkpoint loading, holdout tests."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from c5_snn.data.validation import P_COLUMNS
from c5_snn.models.base import get_model
from c5_snn.training.metrics import compute_all_metrics

logger = logging.getLogger("c5_snn")


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:
    """Load a trained model from a Trainer-format checkpoint.

    Reconstructs the model architecture from the stored config,
    loads the state_dict, moves to device, and sets eval mode.

    Args:
        checkpoint_path: Path to a ``best_model.pt`` file.
        device: Target device (cpu or cuda).

    Returns:
        (model, config) tuple.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "config" not in ckpt or "model_state_dict" not in ckpt:
        raise ValueError(
            f"Checkpoint {checkpoint_path} missing 'config' or "
            "'model_state_dict' keys. Expected Trainer-format checkpoint."
        )

    config = ckpt["config"]
    model = get_model(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    logger.info(
        "Loaded %s from %s (epoch %d)",
        config.get("model", {}).get("type", "unknown"),
        checkpoint_path,
        ckpt.get("epoch", -1),
    )
    return model, config


def build_prediction_window(
    df: pd.DataFrame,
    window_size: int,
    offset: int = 0,
) -> Tensor:
    """Build a single input window from the tail of the dataset.

    Args:
        df: Validated DataFrame with P_1..P_39 columns.
        window_size: W, the number of rows in the window.
        offset: Rows to skip from the end (0 = use the very last W rows).

    Returns:
        Tensor of shape (1, W, 39).

    Raises:
        ValueError: If there aren't enough rows.
    """
    end_idx = len(df) - offset
    start_idx = end_idx - window_size

    if start_idx < 0:
        raise ValueError(
            f"Not enough data: need {window_size + offset} rows, "
            f"have {len(df)}."
        )

    p_values = df[P_COLUMNS].iloc[start_idx:end_idx].values
    tensor = torch.tensor(p_values, dtype=torch.float32)
    return tensor.unsqueeze(0)  # (1, W, 39)


def format_top_k_prediction(
    logits: Tensor,
    k: int = 20,
) -> list[dict]:
    """Extract top-K predictions from logits.

    Args:
        logits: (1, 39) or (39,) tensor of raw scores.
        k: Number of top predictions.

    Returns:
        List of dicts with keys: rank, part, part_number, score.
    """
    scores = logits.squeeze(0).detach().cpu()
    k = min(k, scores.shape[0])
    topk_values, topk_indices = torch.topk(scores, k)

    results = []
    for rank, (idx, val) in enumerate(
        zip(topk_indices.tolist(), topk_values.tolist()), start=1
    ):
        part_number = idx + 1  # P_COLUMNS are 1-indexed
        results.append(
            {
                "rank": rank,
                "part": f"P_{part_number}",
                "part_number": part_number,
                "score": val,
            }
        )
    return results


def run_holdout_test(
    model: torch.nn.Module,
    df: pd.DataFrame,
    window_size: int,
    n_holdout: int,
    device: torch.device,
) -> dict:
    """Evaluate model on the last n_holdout rows as blind holdout.

    For each holdout row, a window of W rows ending just before
    that row is built, and the model predicts the holdout row.
    No data leakage: the target row is never in the input window.

    Args:
        model: A BaseModel in eval mode.
        df: Full validated DataFrame.
        window_size: W.
        n_holdout: Number of most recent rows to hold out.
        device: Torch device.

    Returns:
        Dict with 'metrics' (aggregate) and 'per_sample' (list).

    Raises:
        ValueError: If not enough data for the earliest holdout window.
    """
    min_rows_needed = window_size + n_holdout
    if len(df) < min_rows_needed:
        raise ValueError(
            f"Need at least {min_rows_needed} rows "
            f"(W={window_size} + n_holdout={n_holdout}), "
            f"have {len(df)}."
        )

    all_logits = []
    all_targets = []
    per_sample = []
    p_values = df[P_COLUMNS].values

    model.eval()
    with torch.no_grad():
        for i in range(n_holdout):
            # offset from end: n_holdout - i  (first iteration = farthest back)
            offset = n_holdout - i
            target_idx = len(df) - offset

            # Build window: W rows ending just before target_idx
            window_start = target_idx - window_size
            window_data = p_values[window_start:target_idx]
            x = torch.tensor(window_data, dtype=torch.float32).unsqueeze(0)
            x = x.to(device)

            logits = model(x).cpu()
            target = torch.tensor(
                p_values[target_idx], dtype=torch.float32
            ).unsqueeze(0)

            all_logits.append(logits)
            all_targets.append(target)

            # Per-sample details
            true_indices = target.squeeze(0).bool().nonzero(as_tuple=True)[0]
            true_parts = [f"P_{idx.item() + 1}" for idx in true_indices]
            top20_indices = torch.topk(logits.squeeze(0), 20).indices
            pred_parts = [f"P_{idx.item() + 1}" for idx in top20_indices]

            date_val = df["date"].iloc[target_idx]
            per_sample.append(
                {
                    "row_index": target_idx,
                    "date": str(date_val),
                    "true_parts": true_parts,
                    "predicted_top20": pred_parts,
                }
            )

    logits_cat = torch.cat(all_logits)
    targets_cat = torch.cat(all_targets)
    metrics = compute_all_metrics(logits_cat, targets_cat)

    # Add per-sample metric values
    for i, sample in enumerate(per_sample):
        single_logits = all_logits[i]
        single_target = all_targets[i]
        sample_metrics = compute_all_metrics(single_logits, single_target)
        sample.update(sample_metrics)

    return {"metrics": metrics, "per_sample": per_sample}


# ---------------------------------------------------------------------------
# Calendar-enhanced scoring (SplitExtra strategy)
# ---------------------------------------------------------------------------

_DEFAULT_CAL_CONFIG = {
    "base_window": 14,
    "base_decay": 0.85,
    "base_freq_weight": 1.0,
    "base_recency_weight": 2.0,
    "cal_weight": 10.0,
    "dom_weight": 5.0,
    "woy_weight": 10.0,
    "cal_core_weight": 2.0,
    "core_k": 8,
    "deficit_lookback": 10,
}


def _compute_base_scores(
    p_values: np.ndarray,
    target_idx: int,
    window: int = 14,
    decay: float = 0.85,
    freq_weight: float = 1.0,
    recency_weight: float = 2.0,
) -> np.ndarray:
    """Frequency + exponentially-decayed recency scores."""
    w_data = p_values[max(0, target_idx - window) : target_idx]
    freq = w_data.sum(axis=0)
    exp = np.arange(len(w_data) - 1, -1, -1, dtype=float)
    weights = decay**exp
    recency = (w_data * weights[:, None]).sum(axis=0)
    return freq_weight * freq + recency_weight * recency


def _calendar_ratios(
    p_values: np.ndarray,
    target_idx: int,
    cal_arr: np.ndarray,
    target_val: int,
    min_count: int = 5,
) -> np.ndarray:
    """Ratio of part frequency in calendar-matching rows vs overall."""
    overall = p_values[:target_idx].mean(axis=0) + 1e-10
    mask = cal_arr[:target_idx] == target_val
    if mask.sum() >= min_count:
        return p_values[:target_idx][mask].mean(axis=0) / overall
    return np.ones(39)


def calendar_enhanced_score(
    df: pd.DataFrame,
    target_idx: int,
    config: dict | None = None,
) -> np.ndarray:
    """Compute calendar-enhanced scores for a single prediction.

    Uses the SplitExtra strategy: deficit-based core selection with
    calendar-adjusted fill scoring.

    Args:
        df: Full validated DataFrame with P_1..P_39 and date columns.
        target_idx: Index of the target row (scores use data before this).
        config: Optional config dict overriding default parameters.

    Returns:
        (39,) array of scores for each part.
    """
    cfg = {**_DEFAULT_CAL_CONFIG, **(config or {})}
    p_values = df[P_COLUMNS].values.astype(float)
    dates = pd.to_datetime(df["date"])
    dow_arr = dates.dt.dayofweek.values
    month_arr = dates.dt.month.values
    dom_arr = dates.dt.day.values
    woy_arr = dates.dt.isocalendar().week.values.astype(int)

    # When target_idx == len(df) (predicting next event), estimate
    # the target date as last date + 1 day.
    if target_idx < len(dates):
        target_date = dates.iloc[target_idx]
    else:
        target_date = dates.iloc[-1] + pd.Timedelta(days=1)
    target_dow = target_date.dayofweek
    target_month = target_date.month
    target_dom = target_date.day
    target_woy = int(target_date.isocalendar().week)

    base = _compute_base_scores(
        p_values,
        target_idx,
        window=int(cfg["base_window"]),
        decay=float(cfg["base_decay"]),
        freq_weight=float(cfg["base_freq_weight"]),
        recency_weight=float(cfg["base_recency_weight"]),
    )

    # Calendar ratios: DOW + month
    dow_ratio = _calendar_ratios(
        p_values, target_idx, dow_arr, target_dow, min_count=10
    )
    month_ratio = _calendar_ratios(
        p_values, target_idx, month_arr, target_month, min_count=10
    )
    cal = dow_ratio + month_ratio

    # DOM and WOY ratios
    dom_r = _calendar_ratios(
        p_values, target_idx, dom_arr, target_dom
    )
    woy_r = _calendar_ratios(
        p_values, target_idx, woy_arr, target_woy
    )

    cal_wt = float(cfg["cal_weight"])
    dom_wt = float(cfg["dom_weight"])
    woy_wt = float(cfg["woy_weight"])
    adjusted = base + cal_wt * cal + dom_wt * dom_r + woy_wt * woy_r

    # Core selection: deficit + calendar signals
    lb = int(cfg["deficit_lookback"])
    ccw = float(cfg["cal_core_weight"])
    core_k = int(cfg["core_k"])
    recent_counts = p_values[target_idx - lb : target_idx].sum(axis=0)
    deficit = lb * 5.0 / 39.0 - recent_counts
    core_score = deficit + ccw * cal + dom_r + woy_r

    top_core = np.argsort(core_score)[::-1][:core_k]

    result = np.zeros(39)
    result[top_core] = 100.0
    mask = np.ones(39, dtype=bool)
    mask[top_core] = False
    result[mask] = adjusted[mask] * 0.01

    return result


def calendar_enhanced_predict(
    df: pd.DataFrame,
    config: dict | None = None,
) -> Tensor:
    """Predict scores for the next event using calendar-enhanced strategy.

    Args:
        df: Full validated DataFrame.
        config: Optional config dict overriding default parameters.

    Returns:
        (1, 39) tensor of scores.
    """
    scores = calendar_enhanced_score(df, len(df), config)
    return torch.tensor(scores, dtype=torch.float32).unsqueeze(0)


def run_calendar_holdout_test(
    df: pd.DataFrame,
    window_size: int,
    n_holdout: int,
    config: dict | None = None,
) -> dict:
    """Run holdout test using calendar-enhanced scoring.

    Args:
        df: Full validated DataFrame.
        window_size: W (used for minimum-data validation).
        n_holdout: Number of most recent rows to hold out.
        config: Optional config dict overriding default parameters.

    Returns:
        Dict with 'metrics' and 'per_sample' keys.
    """
    min_rows_needed = window_size + n_holdout
    if len(df) < min_rows_needed:
        raise ValueError(
            f"Need at least {min_rows_needed} rows "
            f"(W={window_size} + n_holdout={n_holdout}), "
            f"have {len(df)}."
        )

    p_values = df[P_COLUMNS].values
    all_logits = []
    all_targets = []
    per_sample = []

    for i in range(n_holdout):
        offset = n_holdout - i
        target_idx = len(df) - offset

        scores = calendar_enhanced_score(df, target_idx, config)
        logits = torch.tensor(scores, dtype=torch.float32).unsqueeze(0)
        target = torch.tensor(
            p_values[target_idx], dtype=torch.float32
        ).unsqueeze(0)

        all_logits.append(logits)
        all_targets.append(target)

        true_indices = target.squeeze(0).bool().nonzero(as_tuple=True)[0]
        true_parts = [f"P_{idx.item() + 1}" for idx in true_indices]
        top20_indices = torch.topk(logits.squeeze(0), 20).indices
        pred_parts = [f"P_{idx.item() + 1}" for idx in top20_indices]

        date_val = df["date"].iloc[target_idx]
        per_sample.append(
            {
                "row_index": target_idx,
                "date": str(date_val),
                "true_parts": true_parts,
                "predicted_top20": pred_parts,
            }
        )

    logits_cat = torch.cat(all_logits)
    targets_cat = torch.cat(all_targets)
    metrics = compute_all_metrics(logits_cat, targets_cat)

    for i, sample in enumerate(per_sample):
        sample_metrics = compute_all_metrics(all_logits[i], all_targets[i])
        sample.update(sample_metrics)

    return {"metrics": metrics, "per_sample": per_sample}
