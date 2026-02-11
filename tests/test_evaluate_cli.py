"""Integration tests for the CLI evaluate command (STORY-2.4)."""

import json

import pytest
import torch
from click.testing import CliRunner

from c5_snn.cli import cli
from c5_snn.data.splits import create_splits, save_splits

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyModel(torch.nn.Module):
    """Dummy model: returns last time-step as logits."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, -1, :]


@pytest.fixture
def cli_runner():
    """Click test runner."""
    return CliRunner()


@pytest.fixture
def eval_env(tmp_path):
    """Set up a complete evaluation environment.

    Creates windowed tensors, splits, and a dummy checkpoint
    in a temporary directory.
    """
    torch.manual_seed(42)
    n_samples = 20
    window_size = 7
    n_classes = 39

    # Create tensors where last step has high logits at true positions
    X = torch.rand(n_samples, window_size, n_classes)
    y = torch.zeros(n_samples, n_classes)
    for i in range(n_samples):
        true_parts = [(i * 5 + j) % n_classes for j in range(5)]
        for p in true_parts:
            y[i, p] = 1.0
            X[i, -1, p] = 100.0 + float(p)

    # Save tensors to data dir
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    torch.save(X, data_dir / f"X_w{window_size}.pt")
    torch.save(y, data_dir / f"y_w{window_size}.pt")

    # Create and save splits
    split_info = create_splits(
        n_samples, (0.70, 0.15, 0.15), window_size=window_size
    )
    save_splits(split_info, str(data_dir))

    # Create checkpoint
    model = _DummyModel()
    config = {
        "data": {
            "window_size": window_size,
            "batch_size": 8,
        },
    }
    ckpt = {
        "model": model,
        "model_name": "dummy_test_model",
        "config": config,
    }
    ckpt_path = tmp_path / "checkpoint.pt"
    torch.save(ckpt, ckpt_path)

    return {
        "tmp_path": tmp_path,
        "data_dir": str(data_dir),
        "ckpt_path": str(ckpt_path),
        "n_test": split_info.counts["test"],
    }


# ---------------------------------------------------------------------------
# evaluate command — happy path
# ---------------------------------------------------------------------------


class TestEvaluateCommand:
    """Verify CLI evaluate command end-to-end."""

    def test_produces_json_output(self, cli_runner, eval_env):
        """CLI creates test_metrics.json."""
        out_dir = str(eval_env["tmp_path"] / "results")
        result = cli_runner.invoke(
            cli,
            [
                "evaluate",
                "--checkpoint", eval_env["ckpt_path"],
                "--output-dir", out_dir,
                "--data-dir", eval_env["data_dir"],
            ],
        )
        assert result.exit_code == 0, result.output
        json_path = eval_env["tmp_path"] / "results" / "test_metrics.json"
        assert json_path.exists()

    def test_produces_csv_output(self, cli_runner, eval_env):
        """CLI creates test_per_sample.csv."""
        out_dir = str(eval_env["tmp_path"] / "results")
        result = cli_runner.invoke(
            cli,
            [
                "evaluate",
                "--checkpoint", eval_env["ckpt_path"],
                "--output-dir", out_dir,
                "--data-dir", eval_env["data_dir"],
            ],
        )
        assert result.exit_code == 0, result.output
        csv_path = eval_env["tmp_path"] / "results" / "test_per_sample.csv"
        assert csv_path.exists()

    def test_json_schema_correct(self, cli_runner, eval_env):
        """JSON output matches architecture Section 4.6 schema."""
        out_dir = str(eval_env["tmp_path"] / "results")
        result = cli_runner.invoke(
            cli,
            [
                "evaluate",
                "--checkpoint", eval_env["ckpt_path"],
                "--output-dir", out_dir,
                "--data-dir", eval_env["data_dir"],
            ],
        )
        assert result.exit_code == 0, result.output

        json_path = eval_env["tmp_path"] / "results" / "test_metrics.json"
        with open(json_path) as f:
            data = json.load(f)

        assert data["model_name"] == "dummy_test_model"
        assert data["checkpoint"] == eval_env["ckpt_path"]
        assert data["split"] == "test"
        assert data["n_samples"] == eval_env["n_test"]
        assert "metrics" in data
        assert "evaluated_at" in data

        metrics = data["metrics"]
        for key in (
            "recall_at_5", "recall_at_20",
            "hit_at_5", "hit_at_20", "mrr",
        ):
            assert key in metrics

    def test_prints_metrics_table(self, cli_runner, eval_env):
        """CLI prints formatted metrics table."""
        out_dir = str(eval_env["tmp_path"] / "results")
        result = cli_runner.invoke(
            cli,
            [
                "evaluate",
                "--checkpoint", eval_env["ckpt_path"],
                "--output-dir", out_dir,
                "--data-dir", eval_env["data_dir"],
            ],
        )
        assert result.exit_code == 0
        assert "Evaluation Results" in result.output
        assert "recall_at_5" in result.output
        assert "recall_at_20" in result.output
        assert "mrr" in result.output
        assert "Samples evaluated" in result.output

    def test_default_output_dir(self, cli_runner, eval_env):
        """Without --output-dir, saves to checkpoint parent dir."""
        result = cli_runner.invoke(
            cli,
            [
                "evaluate",
                "--checkpoint", eval_env["ckpt_path"],
                "--data-dir", eval_env["data_dir"],
            ],
        )
        assert result.exit_code == 0
        # Default output dir is checkpoint parent
        json_path = eval_env["tmp_path"] / "test_metrics.json"
        assert json_path.exists()

    def test_dummy_model_perfect_metrics(self, cli_runner, eval_env):
        """Dummy model with boosted logits gets perfect recall@5."""
        out_dir = str(eval_env["tmp_path"] / "results")
        result = cli_runner.invoke(
            cli,
            [
                "evaluate",
                "--checkpoint", eval_env["ckpt_path"],
                "--output-dir", out_dir,
                "--data-dir", eval_env["data_dir"],
            ],
        )
        assert result.exit_code == 0

        json_path = eval_env["tmp_path"] / "results" / "test_metrics.json"
        with open(json_path) as f:
            data = json.load(f)

        assert data["metrics"]["recall_at_5"] == 1.0
        assert data["metrics"]["hit_at_5"] == 1.0
        assert data["metrics"]["mrr"] == 1.0


# ---------------------------------------------------------------------------
# evaluate command — error cases
# ---------------------------------------------------------------------------


class TestEvaluateErrors:
    """Verify error handling in evaluate command."""

    def test_missing_checkpoint(self, cli_runner, eval_env):
        """Non-existent checkpoint gives non-zero exit."""
        result = cli_runner.invoke(
            cli,
            [
                "evaluate",
                "--checkpoint", "nonexistent.pt",
                "--data-dir", eval_env["data_dir"],
            ],
        )
        assert result.exit_code != 0
        assert "Checkpoint not found" in result.output

    def test_missing_data_dir(self, cli_runner, eval_env):
        """Missing data directory gives non-zero exit."""
        result = cli_runner.invoke(
            cli,
            [
                "evaluate",
                "--checkpoint", eval_env["ckpt_path"],
                "--data-dir", "nonexistent_dir",
            ],
        )
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_missing_splits_json(self, cli_runner, eval_env, tmp_path):
        """Missing splits.json gives non-zero exit."""
        # Create data dir with tensors but no splits
        bad_dir = tmp_path / "bad_data"
        bad_dir.mkdir()
        torch.save(torch.rand(10, 7, 39), bad_dir / "X_w7.pt")
        torch.save(torch.rand(10, 39), bad_dir / "y_w7.pt")

        result = cli_runner.invoke(
            cli,
            [
                "evaluate",
                "--checkpoint", eval_env["ckpt_path"],
                "--data-dir", str(bad_dir),
            ],
        )
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_checkpoint_required(self, cli_runner):
        """--checkpoint is required."""
        result = cli_runner.invoke(cli, ["evaluate"])
        assert result.exit_code != 0
        assert "checkpoint" in result.output.lower()
