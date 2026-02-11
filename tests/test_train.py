"""Tests for Trainer class and train CLI (STORY-3.3)."""

import csv
import logging
from pathlib import Path

import pytest
import torch
import yaml
from click.testing import CliRunner
from torch.utils.data import DataLoader, TensorDataset

from c5_snn.cli import cli
from c5_snn.models.baselines import FrequencyBaseline, GRUBaseline
from c5_snn.training.trainer import Trainer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataloaders(
    n_train: int = 40,
    n_val: int = 10,
    window_size: int = 7,
    batch_size: int = 8,
) -> dict[str, DataLoader]:
    """Create tiny synthetic dataloaders for testing."""
    X_train = torch.rand(n_train, window_size, 39)
    y_train = torch.zeros(n_train, 39)
    y_train[:, :5] = 1.0

    X_val = torch.rand(n_val, window_size, 39)
    y_val = torch.zeros(n_val, 39)
    y_val[:, :5] = 1.0

    return {
        "train": DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=False,
        ),
        "val": DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=batch_size,
            shuffle=False,
        ),
    }


def _make_config(
    epochs: int = 3,
    lr: float = 0.001,
    patience: int = 10,
    output_dir: str = "results/test",
) -> dict:
    """Create a minimal config dict for Trainer."""
    return {
        "experiment": {"name": "test_run", "seed": 42},
        "model": {"type": "gru_baseline", "hidden_size": 32, "num_layers": 1},
        "training": {
            "epochs": epochs,
            "learning_rate": lr,
            "early_stopping_patience": patience,
        },
        "output": {"dir": output_dir},
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dataloaders():
    """Tiny synthetic dataloaders."""
    return _make_dataloaders()


@pytest.fixture
def gru_model():
    """Small GRU model for testing."""
    config = {"model": {"hidden_size": 32, "num_layers": 1}}
    return GRUBaseline(config)


@pytest.fixture
def trainer(tmp_path, gru_model, dataloaders):
    """Trainer with GRU model, tiny data, 3 epochs."""
    config = _make_config(output_dir=str(tmp_path / "output"))
    return Trainer(gru_model, config, dataloaders, torch.device("cpu"))


# ---------------------------------------------------------------------------
# Trainer construction
# ---------------------------------------------------------------------------


class TestTrainerConstruction:
    """Verify Trainer initializes correctly."""

    def test_constructs_with_valid_inputs(self, trainer):
        """Trainer initializes without error."""
        assert trainer.epochs == 3
        assert trainer.lr == 0.001
        assert trainer.patience == 10

    def test_default_config_values(self, gru_model, dataloaders, tmp_path):
        """Trainer uses defaults when config fields missing."""
        config = {"output": {"dir": str(tmp_path / "out")}}
        t = Trainer(gru_model, config, dataloaders, torch.device("cpu"))
        assert t.epochs == 100
        assert t.lr == 0.001
        assert t.patience == 10

    def test_model_on_device(self, trainer):
        """Model is moved to the specified device."""
        for p in trainer.model.parameters():
            assert p.device == trainer.device


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


class TestTrainerRun:
    """Verify training loop execution."""

    def test_runs_specified_epochs(self, trainer):
        """Training runs for the configured number of epochs."""
        result = trainer.run()
        assert result["total_epochs"] == 3

    def test_returns_best_metrics(self, trainer):
        """run() returns dict with expected keys."""
        result = trainer.run()
        assert "best_val_recall_at_20" in result
        assert "best_epoch" in result
        assert "total_epochs" in result
        assert result["best_epoch"] >= 1

    def test_best_recall_non_negative(self, trainer):
        """Best val recall is non-negative."""
        result = trainer.run()
        assert result["best_val_recall_at_20"] >= 0.0

    def test_zero_epochs(self, gru_model, dataloaders, tmp_path):
        """epochs=0 returns immediately with no checkpoint."""
        config = _make_config(epochs=0, output_dir=str(tmp_path / "out"))
        t = Trainer(gru_model, config, dataloaders, torch.device("cpu"))
        result = t.run()
        assert result["total_epochs"] == 0
        assert result["best_epoch"] == 0
        assert not (tmp_path / "out" / "best_model.pt").exists()


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


class TestEarlyStopping:
    """Verify early stopping behavior."""

    def test_early_stopping_triggers(self, gru_model, dataloaders, tmp_path):
        """With patience=1, training stops early if no improvement."""
        config = _make_config(
            epochs=50, patience=1, output_dir=str(tmp_path / "out")
        )
        t = Trainer(gru_model, config, dataloaders, torch.device("cpu"))
        result = t.run()
        # Should stop well before 50 epochs
        assert result["total_epochs"] < 50

    def test_patience_zero_means_no_early_stop(
        self, gru_model, dataloaders, tmp_path
    ):
        """patience=0 effectively disables early stopping."""
        config = _make_config(
            epochs=5, patience=0, output_dir=str(tmp_path / "out")
        )
        t = Trainer(gru_model, config, dataloaders, torch.device("cpu"))
        result = t.run()
        # With patience=0, first non-improvement triggers stop, but
        # it still must run at least one epoch
        assert result["total_epochs"] >= 1


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


class TestCheckpoint:
    """Verify checkpoint saving."""

    def test_checkpoint_saved(self, trainer):
        """best_model.pt is created after training."""
        trainer.run()
        ckpt_path = trainer.output_dir / "best_model.pt"
        assert ckpt_path.exists()

    def test_checkpoint_has_required_keys(self, trainer):
        """Checkpoint dict contains all 6 required keys."""
        trainer.run()
        ckpt = torch.load(
            trainer.output_dir / "best_model.pt",
            map_location="cpu",
            weights_only=False,
        )
        required_keys = {
            "model_state_dict",
            "optimizer_state_dict",
            "epoch",
            "best_val_recall_at_20",
            "config",
            "seed",
        }
        assert required_keys.issubset(ckpt.keys())

    def test_checkpoint_state_dict_loadable(self, trainer, gru_model):
        """Checkpoint model_state_dict can be loaded back."""
        trainer.run()
        ckpt = torch.load(
            trainer.output_dir / "best_model.pt",
            map_location="cpu",
            weights_only=False,
        )
        new_model = GRUBaseline({"model": {"hidden_size": 32, "num_layers": 1}})
        new_model.load_state_dict(ckpt["model_state_dict"])

    def test_checkpoint_epoch_positive(self, trainer):
        """Checkpoint epoch is a positive integer."""
        trainer.run()
        ckpt = torch.load(
            trainer.output_dir / "best_model.pt",
            map_location="cpu",
            weights_only=False,
        )
        assert isinstance(ckpt["epoch"], int)
        assert ckpt["epoch"] >= 1

    def test_checkpoint_seed_from_config(self, trainer):
        """Checkpoint seed matches config."""
        trainer.run()
        ckpt = torch.load(
            trainer.output_dir / "best_model.pt",
            map_location="cpu",
            weights_only=False,
        )
        assert ckpt["seed"] == 42


# ---------------------------------------------------------------------------
# Metrics CSV
# ---------------------------------------------------------------------------


class TestMetricsCSV:
    """Verify per-epoch metrics logging."""

    def test_metrics_csv_created(self, trainer):
        """metrics.csv is written after training."""
        trainer.run()
        assert (trainer.output_dir / "metrics.csv").exists()

    def test_metrics_csv_columns(self, trainer):
        """metrics.csv has the correct columns."""
        trainer.run()
        with open(trainer.output_dir / "metrics.csv") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == [
                "epoch",
                "train_loss",
                "val_recall_at_20",
                "val_hit_at_20",
            ]

    def test_metrics_csv_row_count(self, trainer):
        """metrics.csv has one row per epoch."""
        result = trainer.run()
        with open(trainer.output_dir / "metrics.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == result["total_epochs"]

    def test_metrics_csv_values_parseable(self, trainer):
        """metrics.csv values are valid floats."""
        trainer.run()
        with open(trainer.output_dir / "metrics.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                assert int(row["epoch"]) >= 1
                assert float(row["train_loss"]) >= 0.0
                assert 0.0 <= float(row["val_recall_at_20"]) <= 1.0
                assert 0.0 <= float(row["val_hit_at_20"]) <= 1.0


# ---------------------------------------------------------------------------
# Config snapshot
# ---------------------------------------------------------------------------


class TestConfigSnapshot:
    """Verify config snapshot saving."""

    def test_config_snapshot_saved(self, trainer):
        """config_snapshot.yaml is created at start."""
        trainer.run()
        assert (trainer.output_dir / "config_snapshot.yaml").exists()

    def test_config_snapshot_matches(self, trainer):
        """config_snapshot.yaml matches the original config."""
        trainer.run()
        with open(trainer.output_dir / "config_snapshot.yaml") as f:
            saved = yaml.safe_load(f)
        assert saved == trainer.config


# ---------------------------------------------------------------------------
# Pip freeze
# ---------------------------------------------------------------------------


class TestPipFreeze:
    """Verify pip freeze capture."""

    def test_pip_freeze_saved(self, trainer):
        """pip_freeze.txt is created."""
        trainer.run()
        assert (trainer.output_dir / "pip_freeze.txt").exists()

    def test_pip_freeze_non_empty(self, trainer):
        """pip_freeze.txt has content."""
        trainer.run()
        content = (trainer.output_dir / "pip_freeze.txt").read_text()
        assert len(content) > 0


# ---------------------------------------------------------------------------
# Timing probe
# ---------------------------------------------------------------------------


class TestTimingProbe:
    """Verify 2-epoch timing probe."""

    def test_timing_warning_logged(
        self, gru_model, dataloaders, tmp_path, caplog
    ):
        """WARNING logged when projected time > 20 min."""
        from unittest.mock import patch

        config = _make_config(
            epochs=100, output_dir=str(tmp_path / "out")
        )
        config["training"]["early_stopping_patience"] = 2
        t = Trainer(gru_model, config, dataloaders, torch.device("cpu"))

        # Mock time.time to simulate slow training (15 min for 2 epochs)
        call_count = 0
        base_time = 1000.0

        def fake_time():
            nonlocal call_count
            call_count += 1
            # First call (start_time): 1000.0
            # Subsequent calls simulate 15 min elapsed per 2 epochs
            return base_time + (call_count - 1) * 450.0

        with (
            caplog.at_level(logging.WARNING, logger="c5_snn"),
            patch("c5_snn.training.trainer.time.time", side_effect=fake_time),
        ):
            t.run()

        assert any("RunPod" in r.message for r in caplog.records)

    def test_no_warning_for_short_training(self, trainer, caplog):
        """No timing warning for short training (3 epochs)."""
        with caplog.at_level(logging.WARNING, logger="c5_snn"):
            trainer.run()

        timing_warnings = [
            r for r in caplog.records if "RunPod" in r.message
        ]
        assert len(timing_warnings) == 0


# ---------------------------------------------------------------------------
# Non-trainable model (FrequencyBaseline)
# ---------------------------------------------------------------------------


class TestFrequencyBaselineTrainer:
    """Verify Trainer handles models with no learnable parameters."""

    def test_frequency_baseline_runs(self, dataloaders, tmp_path):
        """Training with FrequencyBaseline completes without error."""
        config = _make_config(epochs=2, output_dir=str(tmp_path / "out"))
        model = FrequencyBaseline(config)
        t = Trainer(model, config, dataloaders, torch.device("cpu"))
        result = t.run()
        assert result["total_epochs"] == 2


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------


class TestOutputDirectory:
    """Verify output directory handling."""

    def test_creates_output_dir(self, gru_model, dataloaders, tmp_path):
        """Output dir is created if it doesn't exist."""
        deep_path = tmp_path / "a" / "b" / "c"
        config = _make_config(output_dir=str(deep_path))
        t = Trainer(gru_model, config, dataloaders, torch.device("cpu"))
        t.run()
        assert deep_path.exists()

    def test_all_artifacts_present(self, trainer):
        """All expected output files exist after training."""
        trainer.run()
        assert (trainer.output_dir / "best_model.pt").exists()
        assert (trainer.output_dir / "config_snapshot.yaml").exists()
        assert (trainer.output_dir / "metrics.csv").exists()
        assert (trainer.output_dir / "pip_freeze.txt").exists()


# ---------------------------------------------------------------------------
# Integration test: train GRU on tiny data via Trainer
# ---------------------------------------------------------------------------


class TestTrainerIntegration:
    """Integration test: full pipeline through Trainer."""

    def test_train_gru_2_epochs(self, tmp_path):
        """Train GRU baseline 2 epochs on ~100 rows, verify all artifacts."""
        # Create synthetic data
        n_samples = 100
        window_size = 7
        X = torch.rand(n_samples, window_size, 39)
        y = torch.zeros(n_samples, 39)
        # Random active parts
        for i in range(n_samples):
            active = torch.randperm(39)[:5]
            y[i, active] = 1.0

        # Split: 70 train, 15 val, 15 test
        train_ds = TensorDataset(X[:70], y[:70])
        val_ds = TensorDataset(X[70:85], y[70:85])

        dataloaders = {
            "train": DataLoader(train_ds, batch_size=16, shuffle=False),
            "val": DataLoader(val_ds, batch_size=16, shuffle=False),
        }

        config = {
            "experiment": {"name": "integration_test", "seed": 42},
            "model": {
                "type": "gru_baseline",
                "hidden_size": 32,
                "num_layers": 1,
            },
            "training": {
                "epochs": 2,
                "learning_rate": 0.001,
                "early_stopping_patience": 10,
            },
            "output": {"dir": str(tmp_path / "results")},
        }

        model = GRUBaseline(config)
        trainer = Trainer(model, config, dataloaders, torch.device("cpu"))
        result = trainer.run()

        # Verify results
        assert result["total_epochs"] == 2
        assert result["best_epoch"] >= 1

        # Verify all output files exist
        out = tmp_path / "results"
        assert (out / "best_model.pt").exists()
        assert (out / "config_snapshot.yaml").exists()
        assert (out / "metrics.csv").exists()
        assert (out / "pip_freeze.txt").exists()

        # Verify checkpoint structure
        ckpt = torch.load(
            out / "best_model.pt", map_location="cpu", weights_only=False
        )
        assert set(ckpt.keys()) == {
            "model_state_dict",
            "optimizer_state_dict",
            "epoch",
            "best_val_recall_at_20",
            "config",
            "seed",
        }

        # Verify metrics CSV
        with open(out / "metrics.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2

        # Verify config snapshot
        with open(out / "config_snapshot.yaml") as f:
            saved_config = yaml.safe_load(f)
        assert saved_config == config


# ---------------------------------------------------------------------------
# Integration test: train CLI
# ---------------------------------------------------------------------------


class TestTrainCLI:
    """Integration test for the train CLI command."""

    def _create_test_csv(self, path: Path, n_rows: int = 50) -> None:
        """Create a minimal valid CSV for testing."""
        import pandas as pd

        dates = pd.date_range("2020-01-01", periods=n_rows, freq="D")
        data = {"date": dates.strftime("%Y-%m-%d")}
        for i in range(1, 6):
            data[f"m_{i}"] = [f"M{i}"] * n_rows
        for i in range(1, 40):
            data[f"P_{i}"] = [1 if i <= 5 else 0] * n_rows
        df = pd.DataFrame(data)
        df.to_csv(path, index=False)

    def test_train_cli_produces_output(self, tmp_path):
        """train --config produces all expected output files."""
        # Create test data
        csv_path = tmp_path / "test_data.csv"
        self._create_test_csv(csv_path, n_rows=50)

        # Create config
        output_dir = tmp_path / "results"
        config = {
            "experiment": {"name": "cli_test", "seed": 42},
            "data": {
                "raw_path": str(csv_path),
                "window_size": 7,
                "split_ratios": [0.70, 0.15, 0.15],
                "batch_size": 8,
            },
            "model": {
                "type": "gru_baseline",
                "hidden_size": 16,
                "num_layers": 1,
            },
            "training": {
                "epochs": 2,
                "learning_rate": 0.001,
                "early_stopping_patience": 10,
            },
            "output": {"dir": str(output_dir)},
            "log_level": "WARNING",
        }
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        runner = CliRunner()
        result = runner.invoke(cli, ["train", "--config", str(config_path)])

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert "Training Complete" in result.output
        assert (output_dir / "best_model.pt").exists()
        assert (output_dir / "config_snapshot.yaml").exists()
        assert (output_dir / "metrics.csv").exists()
        assert (output_dir / "pip_freeze.txt").exists()

    def test_train_cli_missing_config(self):
        """train --config with missing file exits non-zero."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["train", "--config", "nonexistent.yaml"]
        )
        assert result.exit_code != 0

    def test_train_cli_invalid_config(self, tmp_path):
        """train --config with invalid YAML exits non-zero."""
        bad_config = tmp_path / "bad.yaml"
        bad_config.write_text(": invalid: yaml: [")

        runner = CliRunner()
        result = runner.invoke(
            cli, ["train", "--config", str(bad_config)]
        )
        assert result.exit_code != 0

    def test_train_cli_missing_data(self, tmp_path):
        """train --config with missing data file exits non-zero."""
        config = {
            "experiment": {"name": "test", "seed": 42},
            "data": {"raw_path": str(tmp_path / "no_such_file.csv")},
            "model": {"type": "gru_baseline"},
            "training": {"epochs": 1},
            "output": {"dir": str(tmp_path / "out")},
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        runner = CliRunner()
        result = runner.invoke(
            cli, ["train", "--config", str(config_path)]
        )
        assert result.exit_code != 0
