"""Generic training loop for any BaseModel."""

import csv
import logging
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from c5_snn.models.base import BaseModel
from c5_snn.training.metrics import compute_all_metrics

logger = logging.getLogger("c5_snn")


class Trainer:
    """Generic training loop for any BaseModel.

    Handles: forward -> BCEWithLogitsLoss -> backward -> optimizer step,
    validation after each epoch, early stopping, checkpointing, and
    experiment artifact saving.
    """

    def __init__(
        self,
        model: BaseModel,
        config: dict,
        dataloaders: dict[str, DataLoader],
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.config = config

        train_cfg = config.get("training", {})
        self.epochs = int(train_cfg.get("epochs", 100))
        self.lr = float(train_cfg.get("learning_rate", 0.001))
        self.patience = int(train_cfg.get("early_stopping_patience", 10))

        params = list(model.parameters())
        if params:
            self.optimizer = torch.optim.Adam(params, lr=self.lr)
        else:
            # Models with no learnable parameters (e.g., FrequencyBaseline)
            self.optimizer = None
            logger.info("Model has no learnable parameters; skipping optimizer")
        self.criterion = nn.BCEWithLogitsLoss()

        self.train_loader = dataloaders["train"]
        self.val_loader = dataloaders["val"]

        self.output_dir = Path(
            config.get("output", {}).get("dir", "results/default")
        )

    def run(self) -> dict:
        """Execute full training loop. Returns best metrics dict."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config snapshot (immutable — Section 13.3 Rule 8)
        self._save_config_snapshot()

        # Save pip freeze for reproducibility (NFR1)
        self._save_pip_freeze()

        best_val_recall: float = -1.0
        best_epoch: int = 0
        patience_counter: int = 0
        epoch_metrics: list[dict] = []
        start_time = time.time()

        for epoch in range(self.epochs):
            # --- Train step ---
            avg_loss = self._train_epoch()

            # --- Validation step ---
            val_metrics = self._validate()

            # --- Log ---
            row = {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "val_recall_at_20": val_metrics["recall_at_20"],
                "val_hit_at_20": val_metrics["hit_at_20"],
            }
            epoch_metrics.append(row)
            logger.info(
                "Epoch %d/%d — loss=%.4f val_recall@20=%.4f",
                epoch + 1,
                self.epochs,
                avg_loss,
                val_metrics["recall_at_20"],
            )

            # --- 2-epoch timing probe (NFR3) ---
            if epoch == 1:
                elapsed = time.time() - start_time
                projected = elapsed / 2 * self.epochs
                if projected > 20 * 60:
                    logger.warning(
                        "Projected training time: %.0f min. "
                        "Consider RunPod.",
                        projected / 60,
                    )

            # --- Early stopping + checkpoint ---
            if val_metrics["recall_at_20"] > best_val_recall:
                best_val_recall = val_metrics["recall_at_20"]
                best_epoch = epoch + 1
                patience_counter = 0
                self._save_checkpoint(best_epoch, best_val_recall)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(
                        "Early stopping at epoch %d (patience=%d)",
                        epoch + 1,
                        self.patience,
                    )
                    break

        # Save per-epoch metrics CSV (NFR4)
        self._save_metrics_csv(epoch_metrics)

        logger.info(
            "Training complete. Best val_recall@20=%.4f at epoch %d",
            best_val_recall,
            best_epoch,
        )

        return {
            "best_val_recall_at_20": best_val_recall,
            "best_epoch": best_epoch,
            "total_epochs": len(epoch_metrics),
        }

    def _train_epoch(self) -> float:
        """Run one training epoch. Returns average loss."""
        self.model.train()
        epoch_loss = 0.0
        n_samples = 0

        for batch_x, batch_y in self.train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            if self.optimizer is not None:
                self.optimizer.zero_grad()
            logits = self.model(batch_x)
            loss = self.criterion(logits, batch_y)
            if self.optimizer is not None:
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)
            n_samples += batch_x.size(0)

        return epoch_loss / n_samples if n_samples > 0 else 0.0

    def _validate(self) -> dict[str, float]:
        """Run validation and return metrics dict."""
        self.model.eval()
        all_logits = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x = batch_x.to(self.device)
                logits = self.model(batch_x)
                all_logits.append(logits.cpu())
                all_targets.append(batch_y)

        logits_cat = torch.cat(all_logits)
        targets_cat = torch.cat(all_targets)
        return compute_all_metrics(logits_cat, targets_cat)

    def _save_checkpoint(self, epoch: int, best_val_recall: float) -> None:
        """Save best-model checkpoint (Section 4.5 format)."""
        seed = self.config.get("experiment", {}).get("seed", 42)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": (
                self.optimizer.state_dict()
                if self.optimizer is not None
                else {}
            ),
            "epoch": epoch,
            "best_val_recall_at_20": best_val_recall,
            "config": self.config,
            "seed": seed,
        }
        path = self.output_dir / "best_model.pt"
        torch.save(checkpoint, path)
        logger.info("Saved checkpoint to %s (epoch %d)", path, epoch)

    def _save_config_snapshot(self) -> None:
        """Save frozen config snapshot at start of training."""
        path = self.output_dir / "config_snapshot.yaml"
        with open(path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        logger.info("Saved config snapshot to %s", path)

    def _save_pip_freeze(self) -> None:
        """Capture pip freeze for reproducibility."""
        path = self.output_dir / "pip_freeze.txt"
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            path.write_text(result.stdout)
            logger.info("Saved pip freeze to %s", path)
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning("Failed to capture pip freeze: %s", e)

    def _save_metrics_csv(self, epoch_metrics: list[dict]) -> None:
        """Save per-epoch metrics to CSV."""
        path = self.output_dir / "metrics.csv"
        fieldnames = [
            "epoch",
            "train_loss",
            "val_recall_at_20",
            "val_hit_at_20",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in epoch_metrics:
                writer.writerow(row)
        logger.info("Saved metrics CSV to %s (%d epochs)", path, len(epoch_metrics))
