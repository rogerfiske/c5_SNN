"""Evaluation harness: inference + metric computation + result export."""

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from c5_snn.training.metrics import (
    compute_all_metrics,
    hit_at_k,
    recall_at_k,
)

logger = logging.getLogger("c5_snn")


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Run full evaluation: inference on all batches + aggregate metrics.

    Args:
        model: Any model following BaseModel interface
            (batch, W, 39) -> (batch, 39).
        dataloader: Test or validation DataLoader.
        device: Torch device for inference.

    Returns:
        Dict with:
          - "metrics": aggregate metric dict (5 keys)
          - "per_sample": list of per-sample result dicts
    """
    model.eval()
    model.to(device)

    all_logits = []
    all_targets = []

    logger.info("Starting evaluation on %s", device)

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            all_logits.append(logits.cpu())
            all_targets.append(batch_y.cpu())

    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)

    n_samples = logits_cat.shape[0]
    logger.info("Collected %d samples, computing metrics", n_samples)

    aggregate = compute_all_metrics(logits_cat, targets_cat)

    per_sample = _compute_per_sample(logits_cat, targets_cat)

    logger.info(
        "Evaluation complete: recall@20=%.4f, mrr=%.4f",
        aggregate["recall_at_20"],
        aggregate["mrr"],
    )

    return {"metrics": aggregate, "per_sample": per_sample}


def _compute_per_sample(
    logits: torch.Tensor, targets: torch.Tensor
) -> list[dict]:
    """Compute per-sample metric details."""
    results = []
    _, top20_indices = torch.topk(logits, 20, dim=1)

    for i in range(logits.shape[0]):
        true_parts = targets[i].bool().nonzero(as_tuple=True)[0].tolist()
        predicted_top20 = top20_indices[i].tolist()

        sample_logits = logits[i].unsqueeze(0)
        sample_targets = targets[i].unsqueeze(0)

        results.append({
            "sample_idx": i,
            "true_parts": true_parts,
            "predicted_top20": predicted_top20,
            "recall_at_5": recall_at_k(sample_logits, sample_targets, 5),
            "recall_at_20": recall_at_k(
                sample_logits, sample_targets, 20
            ),
            "hit_at_5": hit_at_k(sample_logits, sample_targets, 5),
            "hit_at_20": hit_at_k(sample_logits, sample_targets, 20),
        })

    return results


def export_results(
    metrics: dict,
    per_sample: list[dict],
    model_name: str,
    output_dir: str,
    checkpoint_path: str = "",
    split: str = "test",
) -> None:
    """Save test_metrics.json and test_per_sample.csv.

    Args:
        metrics: Aggregate metric dict from evaluate_model.
        per_sample: Per-sample results from evaluate_model.
        model_name: Name of the model (e.g., "freq_baseline_v1").
        output_dir: Directory to write output files.
        checkpoint_path: Path to model checkpoint (for metadata).
        split: Dataset split name (default "test").
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_payload = {
        "model_name": model_name,
        "checkpoint": checkpoint_path,
        "split": split,
        "n_samples": len(per_sample),
        "metrics": metrics,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    json_path = out / "test_metrics.json"
    with open(json_path, "w") as f:
        json.dump(json_payload, f, indent=2)
    logger.info("Saved metrics to %s", json_path)

    csv_path = out / "test_per_sample.csv"
    fieldnames = [
        "sample_idx",
        "true_parts",
        "predicted_top20",
        "recall_at_5",
        "recall_at_20",
        "hit_at_5",
        "hit_at_20",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_sample:
            writer.writerow(row)
    logger.info("Saved per-sample CSV to %s", csv_path)
