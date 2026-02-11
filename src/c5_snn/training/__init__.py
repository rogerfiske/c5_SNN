"""Training, evaluation, and metrics for the c5_SNN pipeline."""

from c5_snn.training.compare import (
    build_comparison,
    format_comparison_table,
    save_comparison,
)
from c5_snn.training.evaluate import evaluate_model, export_results
from c5_snn.training.metrics import (
    compute_all_metrics,
    hit_at_k,
    mrr,
    recall_at_k,
)
from c5_snn.training.trainer import Trainer

__all__ = [
    "Trainer",
    "build_comparison",
    "compute_all_metrics",
    "evaluate_model",
    "export_results",
    "format_comparison_table",
    "hit_at_k",
    "mrr",
    "recall_at_k",
    "save_comparison",
]
