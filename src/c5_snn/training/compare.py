"""Cross-model comparison report generation (Section 4.7)."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("c5_snn")

METRIC_KEYS = [
    "recall_at_5",
    "recall_at_20",
    "hit_at_5",
    "hit_at_20",
    "mrr",
]


def build_comparison(
    model_results: list[dict],
    window_size: int,
    test_split_size: int,
) -> dict:
    """Build a comparison report from multiple model evaluation results.

    Args:
        model_results: List of dicts, each with:
            - name: str (e.g., "frequency_baseline")
            - type: str ("heuristic" or "learned")
            - phase: str ("baseline")
            - seed_metrics: list[dict] — one metrics dict per seed run
            - training_time_s: float — total training time across seeds
            - environment: str ("local" or "runpod")
        window_size: Window size W used for all evaluations.
        test_split_size: Number of test samples.

    Returns:
        Comparison report dict following Section 4.7 schema.
    """
    models = []
    for result in model_results:
        seed_metrics = result["seed_metrics"]
        n_seeds = len(seed_metrics)

        metrics_mean = _compute_mean(seed_metrics)
        metrics_std = _compute_std(seed_metrics, metrics_mean)

        models.append({
            "name": result["name"],
            "type": result["type"],
            "phase": result["phase"],
            "metrics_mean": metrics_mean,
            "metrics_std": metrics_std,
            "n_seeds": n_seeds,
            "training_time_s": result.get("training_time_s", 0),
            "environment": result.get("environment", "local"),
        })

    report = {
        "models": models,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_size": window_size,
        "test_split_size": test_split_size,
    }

    logger.info(
        "Built comparison report for %d models (window=%d, test=%d)",
        len(models),
        window_size,
        test_split_size,
    )
    return report


def save_comparison(report: dict, output_path: str) -> Path:
    """Save comparison report to JSON file.

    Args:
        report: Comparison report dict from build_comparison.
        output_path: Path for the output JSON file.

    Returns:
        Path to the saved file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Saved comparison report to %s", path)
    return path


def format_comparison_table(report: dict) -> str:
    """Format a comparison report as a human-readable table.

    Args:
        report: Comparison report dict from build_comparison.

    Returns:
        Formatted table string.
    """
    lines = []
    lines.append("Baseline Comparison Results")
    lines.append("=" * 65)
    lines.append("")

    header = f"{'Model':<25} {'Recall@20':<15} {'Hit@20':<15} {'MRR':<15}"
    lines.append(header)
    lines.append("-" * 65)

    for model in report["models"]:
        name = model["name"]
        mean = model["metrics_mean"]
        std = model["metrics_std"]
        n_seeds = model["n_seeds"]

        if n_seeds > 1:
            r20 = f"{mean['recall_at_20']:.4f}+/-{std['recall_at_20']:.3f}"
            h20 = f"{mean['hit_at_20']:.4f}+/-{std['hit_at_20']:.3f}"
            mrr_val = f"{mean['mrr']:.4f}+/-{std['mrr']:.3f}"
        else:
            r20 = f"{mean['recall_at_20']:.4f}"
            h20 = f"{mean['hit_at_20']:.4f}"
            mrr_val = f"{mean['mrr']:.4f}"

        lines.append(f"{name:<25} {r20:<15} {h20:<15} {mrr_val:<15}")

    lines.append("")
    lines.append(
        f"Window size: {report['window_size']} | "
        f"Test samples: {report['test_split_size']}"
    )

    return "\n".join(lines)


def _compute_mean(seed_metrics: list[dict]) -> dict[str, float]:
    """Compute mean of metrics across seeds."""
    n = len(seed_metrics)
    if n == 0:
        return {k: 0.0 for k in METRIC_KEYS}

    result = {}
    for key in METRIC_KEYS:
        values = [m[key] for m in seed_metrics]
        result[key] = sum(values) / n
    return result


def _compute_std(
    seed_metrics: list[dict], mean: dict[str, float]
) -> dict[str, float]:
    """Compute standard deviation of metrics across seeds."""
    n = len(seed_metrics)
    if n <= 1:
        return {k: 0.0 for k in METRIC_KEYS}

    result = {}
    for key in METRIC_KEYS:
        values = [m[key] for m in seed_metrics]
        variance = sum((v - mean[key]) ** 2 for v in values) / n
        result[key] = variance**0.5
    return result
