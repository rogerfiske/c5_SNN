"""Evaluation metrics for multi-label ranking (CA5 task).

All metrics operate on:
- logits: Tensor of shape (batch, 39) — raw model output scores
- targets: Tensor of shape (batch, 39) — multi-hot ground truth (5 ones per row)
"""

import logging

import torch
from torch import Tensor

logger = logging.getLogger("c5_snn")


def recall_at_k(logits: Tensor, targets: Tensor, k: int) -> float:
    """Recall@K averaged over the batch.

    For each sample, fraction of true positives in the top-K predictions.
    Recall@K = mean over samples of |top_K ∩ true_parts| / |true_parts|.

    Args:
        logits: (batch, 39) raw model scores.
        targets: (batch, 39) multi-hot ground truth.
        k: Number of top predictions to consider.

    Returns:
        Mean recall@K across the batch.
    """
    _, top_k_indices = torch.topk(logits, k, dim=1)
    target_bool = targets.bool()

    batch_size = logits.shape[0]
    recalls = []
    for i in range(batch_size):
        true_set = target_bool[i].nonzero(as_tuple=True)[0]
        n_true = true_set.numel()
        if n_true == 0:
            recalls.append(1.0)
            continue
        hits = target_bool[i][top_k_indices[i]].sum().item()
        recalls.append(hits / n_true)

    return sum(recalls) / len(recalls)


def hit_at_k(logits: Tensor, targets: Tensor, k: int) -> float:
    """Hit@K averaged over the batch.

    For each sample, 1 if at least one true positive is in the top-K,
    else 0.

    Args:
        logits: (batch, 39) raw model scores.
        targets: (batch, 39) multi-hot ground truth.
        k: Number of top predictions to consider.

    Returns:
        Mean hit@K across the batch.
    """
    _, top_k_indices = torch.topk(logits, k, dim=1)
    target_bool = targets.bool()

    batch_size = logits.shape[0]
    hits = []
    for i in range(batch_size):
        any_hit = target_bool[i][top_k_indices[i]].any().item()
        hits.append(1.0 if any_hit else 0.0)

    return sum(hits) / len(hits)


def mrr(logits: Tensor, targets: Tensor) -> float:
    """Mean Reciprocal Rank averaged over the batch.

    For each sample, find the rank of the highest-ranked true positive.
    Reciprocal Rank = 1 / rank (1-indexed).

    Args:
        logits: (batch, 39) raw model scores.
        targets: (batch, 39) multi-hot ground truth.

    Returns:
        Mean reciprocal rank across the batch.
    """
    n_classes = logits.shape[1]
    _, sorted_indices = torch.sort(logits, dim=1, descending=True)
    target_bool = targets.bool()

    batch_size = logits.shape[0]
    rr_values = []
    for i in range(batch_size):
        ranks = torch.zeros(n_classes, dtype=torch.long)
        ranks[sorted_indices[i]] = torch.arange(
            1, n_classes + 1, dtype=torch.long
        )
        true_positions = target_bool[i].nonzero(as_tuple=True)[0]
        if true_positions.numel() == 0:
            rr_values.append(0.0)
            continue
        best_rank = ranks[true_positions].min().item()
        rr_values.append(1.0 / best_rank)

    return sum(rr_values) / len(rr_values)


def compute_all_metrics(
    logits: Tensor, targets: Tensor
) -> dict[str, float]:
    """Compute all 5 standard metrics.

    Returns:
        Dict with keys: recall_at_5, recall_at_20, hit_at_5, hit_at_20,
        mrr.
    """
    results = {
        "recall_at_5": recall_at_k(logits, targets, k=5),
        "recall_at_20": recall_at_k(logits, targets, k=20),
        "hit_at_5": hit_at_k(logits, targets, k=5),
        "hit_at_20": hit_at_k(logits, targets, k=20),
        "mrr": mrr(logits, targets),
    }
    logger.debug("Computed metrics: %s", results)
    return results
