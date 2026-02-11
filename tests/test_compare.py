"""Tests for comparison report generation and compare CLI (STORY-3.4)."""

import json
import math
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from c5_snn.cli import cli
from c5_snn.training.compare import (
    build_comparison,
    format_comparison_table,
    save_comparison,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

METRIC_KEYS = ["recall_at_5", "recall_at_20", "hit_at_5", "hit_at_20", "mrr"]


def _make_metrics(
    recall_at_5: float = 0.3,
    recall_at_20: float = 0.7,
    hit_at_5: float = 0.5,
    hit_at_20: float = 0.9,
    mrr: float = 0.4,
) -> dict:
    """Create a metrics dict with given values."""
    return {
        "recall_at_5": recall_at_5,
        "recall_at_20": recall_at_20,
        "hit_at_5": hit_at_5,
        "hit_at_20": hit_at_20,
        "mrr": mrr,
    }


def _make_model_result(
    name: str = "test_model",
    model_type: str = "learned",
    phase: str = "baseline",
    seed_metrics: list | None = None,
    training_time_s: float = 10.0,
    environment: str = "local",
) -> dict:
    """Create a model result dict for build_comparison input."""
    if seed_metrics is None:
        seed_metrics = [_make_metrics()]
    return {
        "name": name,
        "type": model_type,
        "phase": phase,
        "seed_metrics": seed_metrics,
        "training_time_s": training_time_s,
        "environment": environment,
    }


# ---------------------------------------------------------------------------
# build_comparison — schema
# ---------------------------------------------------------------------------


class TestBuildComparisonSchema:
    """Verify comparison report follows Section 4.7 schema."""

    def test_top_level_keys(self):
        """Report has all required top-level keys."""
        results = [_make_model_result()]
        report = build_comparison(results, window_size=21, test_split_size=100)
        assert "models" in report
        assert "generated_at" in report
        assert "window_size" in report
        assert "test_split_size" in report

    def test_model_entry_keys(self):
        """Each model entry has all required keys."""
        results = [_make_model_result()]
        report = build_comparison(results, window_size=21, test_split_size=100)
        model = report["models"][0]
        required = {
            "name",
            "type",
            "phase",
            "metrics_mean",
            "metrics_std",
            "n_seeds",
            "training_time_s",
            "environment",
        }
        assert required.issubset(model.keys())

    def test_metrics_mean_keys(self):
        """metrics_mean has all 5 metric keys."""
        results = [_make_model_result()]
        report = build_comparison(results, window_size=21, test_split_size=100)
        mean = report["models"][0]["metrics_mean"]
        for key in METRIC_KEYS:
            assert key in mean

    def test_metrics_std_keys(self):
        """metrics_std has all 5 metric keys."""
        results = [_make_model_result()]
        report = build_comparison(results, window_size=21, test_split_size=100)
        std = report["models"][0]["metrics_std"]
        for key in METRIC_KEYS:
            assert key in std

    def test_window_size_stored(self):
        """Window size is stored correctly."""
        report = build_comparison(
            [_make_model_result()], window_size=21, test_split_size=100
        )
        assert report["window_size"] == 21

    def test_test_split_size_stored(self):
        """Test split size is stored correctly."""
        report = build_comparison(
            [_make_model_result()], window_size=21, test_split_size=1752
        )
        assert report["test_split_size"] == 1752

    def test_generated_at_present(self):
        """generated_at timestamp is a non-empty string."""
        report = build_comparison(
            [_make_model_result()], window_size=21, test_split_size=100
        )
        assert len(report["generated_at"]) > 0

    def test_multiple_models(self):
        """Report handles multiple models."""
        results = [
            _make_model_result(name="model_a"),
            _make_model_result(name="model_b"),
        ]
        report = build_comparison(results, window_size=21, test_split_size=100)
        assert len(report["models"]) == 2
        assert report["models"][0]["name"] == "model_a"
        assert report["models"][1]["name"] == "model_b"


# ---------------------------------------------------------------------------
# build_comparison — mean/std computation
# ---------------------------------------------------------------------------


class TestBuildComparisonStats:
    """Verify mean and std computation."""

    def test_single_seed_mean_equals_value(self):
        """Single seed: mean equals the single metrics dict."""
        m = _make_metrics(recall_at_20=0.75)
        result = _make_model_result(seed_metrics=[m])
        report = build_comparison(
            [result], window_size=21, test_split_size=100
        )
        model = report["models"][0]
        assert model["metrics_mean"]["recall_at_20"] == pytest.approx(0.75)
        assert model["n_seeds"] == 1

    def test_single_seed_std_is_zero(self):
        """Single seed: std is 0 for all metrics."""
        result = _make_model_result(seed_metrics=[_make_metrics()])
        report = build_comparison(
            [result], window_size=21, test_split_size=100
        )
        std = report["models"][0]["metrics_std"]
        for key in METRIC_KEYS:
            assert std[key] == 0.0

    def test_multi_seed_mean_correct(self):
        """Multi-seed: mean is correctly computed."""
        m1 = _make_metrics(recall_at_20=0.6)
        m2 = _make_metrics(recall_at_20=0.8)
        m3 = _make_metrics(recall_at_20=0.7)
        result = _make_model_result(seed_metrics=[m1, m2, m3])
        report = build_comparison(
            [result], window_size=21, test_split_size=100
        )
        model = report["models"][0]
        assert model["metrics_mean"]["recall_at_20"] == pytest.approx(0.7)
        assert model["n_seeds"] == 3

    def test_multi_seed_std_correct(self):
        """Multi-seed: std is correctly computed (population std)."""
        # Values: 0.6, 0.8, 0.7 -> mean = 0.7
        # variance = ((0.1)^2 + (0.1)^2 + 0) / 3 = 0.02/3
        # std = sqrt(0.02/3)
        m1 = _make_metrics(recall_at_20=0.6)
        m2 = _make_metrics(recall_at_20=0.8)
        m3 = _make_metrics(recall_at_20=0.7)
        result = _make_model_result(seed_metrics=[m1, m2, m3])
        report = build_comparison(
            [result], window_size=21, test_split_size=100
        )
        std = report["models"][0]["metrics_std"]
        expected_std = math.sqrt(0.02 / 3)
        assert std["recall_at_20"] == pytest.approx(expected_std, abs=1e-6)

    def test_identical_seeds_zero_std(self):
        """Identical metrics across seeds: std is 0."""
        m = _make_metrics(recall_at_20=0.5)
        result = _make_model_result(seed_metrics=[m, m, m])
        report = build_comparison(
            [result], window_size=21, test_split_size=100
        )
        std = report["models"][0]["metrics_std"]
        assert std["recall_at_20"] == pytest.approx(0.0)

    def test_heuristic_model_properties(self):
        """Heuristic model with 1 seed: correct type and zero std."""
        result = _make_model_result(
            name="frequency_baseline",
            model_type="heuristic",
            seed_metrics=[_make_metrics()],
            training_time_s=0,
        )
        report = build_comparison(
            [result], window_size=21, test_split_size=100
        )
        model = report["models"][0]
        assert model["type"] == "heuristic"
        assert model["n_seeds"] == 1
        assert model["training_time_s"] == 0
        assert all(v == 0.0 for v in model["metrics_std"].values())


# ---------------------------------------------------------------------------
# save_comparison
# ---------------------------------------------------------------------------


class TestSaveComparison:
    """Verify comparison report persistence."""

    def test_saves_json(self, tmp_path):
        """Report is saved as valid JSON."""
        report = build_comparison(
            [_make_model_result()], window_size=21, test_split_size=100
        )
        path = save_comparison(report, str(tmp_path / "test.json"))
        assert path.exists()
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["window_size"] == 21

    def test_creates_parent_dirs(self, tmp_path):
        """Parent directories are created if needed."""
        deep_path = tmp_path / "a" / "b" / "report.json"
        report = build_comparison(
            [_make_model_result()], window_size=21, test_split_size=100
        )
        path = save_comparison(report, str(deep_path))
        assert path.exists()

    def test_roundtrip(self, tmp_path):
        """Report roundtrips through JSON correctly."""
        report = build_comparison(
            [_make_model_result(name="test")],
            window_size=21,
            test_split_size=500,
        )
        path = save_comparison(report, str(tmp_path / "r.json"))
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["models"][0]["name"] == "test"
        assert loaded["test_split_size"] == 500


# ---------------------------------------------------------------------------
# format_comparison_table
# ---------------------------------------------------------------------------


class TestFormatComparisonTable:
    """Verify formatted table output."""

    def test_contains_header(self):
        """Table contains 'Baseline Comparison Results' header."""
        report = build_comparison(
            [_make_model_result()], window_size=21, test_split_size=100
        )
        table = format_comparison_table(report)
        assert "Baseline Comparison Results" in table

    def test_contains_model_name(self):
        """Table contains the model name."""
        report = build_comparison(
            [_make_model_result(name="my_model")],
            window_size=21,
            test_split_size=100,
        )
        table = format_comparison_table(report)
        assert "my_model" in table

    def test_multi_seed_shows_plus_minus(self):
        """Multi-seed model shows +/- notation."""
        result = _make_model_result(
            seed_metrics=[_make_metrics(recall_at_20=0.6), _make_metrics(recall_at_20=0.8)]
        )
        report = build_comparison(
            [result], window_size=21, test_split_size=100
        )
        table = format_comparison_table(report)
        assert "+/-" in table

    def test_single_seed_no_plus_minus(self):
        """Single-seed model shows plain value without +/-."""
        result = _make_model_result(seed_metrics=[_make_metrics()])
        report = build_comparison(
            [result], window_size=21, test_split_size=100
        )
        table = format_comparison_table(report)
        # Should have a clean number without +/-
        assert "0.7000" in table  # recall_at_20 default
        # Should not have +/- for single seed
        lines = table.split("\n")
        model_line = [line for line in lines if "test_model" in line][0]
        assert "+/-" not in model_line

    def test_contains_window_and_samples(self):
        """Table footer shows window size and sample count."""
        report = build_comparison(
            [_make_model_result()], window_size=21, test_split_size=1752
        )
        table = format_comparison_table(report)
        assert "21" in table
        assert "1752" in table


# ---------------------------------------------------------------------------
# Integration test: compare CLI
# ---------------------------------------------------------------------------


class TestCompareCLI:
    """Integration test for the compare CLI command."""

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

    def test_compare_cli_produces_output(self, tmp_path):
        """compare --config produces comparison JSON and table."""
        csv_path = tmp_path / "test_data.csv"
        self._create_test_csv(csv_path, n_rows=50)

        output_json = tmp_path / "comparison.json"
        config = {
            "experiment": {"name": "cli_compare_test", "seed": 42},
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
            "output": {"dir": str(tmp_path / "results")},
            "log_level": "WARNING",
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "compare",
                "--config", str(config_path),
                "--seeds", "42,7",
                "--output", str(output_json),
            ],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert "Baseline Comparison Results" in result.output
        assert "frequency_baseline" in result.output
        assert "gru_baseline" in result.output

        # Verify JSON output
        assert output_json.exists()
        with open(output_json) as f:
            report = json.load(f)
        assert len(report["models"]) == 2
        assert report["models"][0]["name"] == "frequency_baseline"
        assert report["models"][1]["name"] == "gru_baseline"
        assert report["models"][0]["n_seeds"] == 1
        assert report["models"][1]["n_seeds"] == 2

    def test_compare_cli_missing_config(self):
        """compare --config with missing file exits non-zero."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["compare", "--config", "nonexistent.yaml"]
        )
        assert result.exit_code != 0

    def test_compare_cli_missing_data(self, tmp_path):
        """compare --config with missing data file exits non-zero."""
        config = {
            "experiment": {"name": "test", "seed": 42},
            "data": {"raw_path": str(tmp_path / "no_such.csv")},
            "model": {"type": "gru_baseline"},
            "training": {"epochs": 1},
            "output": {"dir": str(tmp_path / "out")},
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        runner = CliRunner()
        result = runner.invoke(
            cli, ["compare", "--config", str(config_path)]
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Integration test: phase-a CLI (STORY-4.4)
# ---------------------------------------------------------------------------


class TestPhaseACLI:
    """Integration tests for the phase-a CLI command."""

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

    def test_phase_a_command_exists(self):
        """phase-a command is registered and shows help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["phase-a", "--help"])
        assert result.exit_code == 0
        assert "Train Phase A SNN models" in result.output

    def test_phase_a_seeds_option(self):
        """phase-a --help shows --seeds option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["phase-a", "--help"])
        assert "--seeds" in result.output
        assert "42,123,7" in result.output

    def test_phase_a_output_option(self):
        """phase-a --help shows --output option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["phase-a", "--help"])
        assert "--output" in result.output
        assert "phase_a_comparison.json" in result.output

    def test_phase_a_produces_output(self, tmp_path, monkeypatch):
        """phase-a trains all models and produces comparison JSON."""
        csv_path = tmp_path / "test_data.csv"
        self._create_test_csv(csv_path, n_rows=50)

        output_json = tmp_path / "comparison.json"

        # Monkey-patch the hardcoded data path and training params
        # to use test data with tiny models and few epochs.
        import c5_snn.cli as cli_module

        original_phase_a = cli_module.phase_a.callback

        def patched_phase_a(seeds, output_path):
            """Wrapper that patches data path for testing."""

            from c5_snn.data.dataset import get_dataloaders
            from c5_snn.data.loader import load_csv
            from c5_snn.data.splits import create_splits
            from c5_snn.data.windowing import build_windows
            from c5_snn.models.base import get_model
            from c5_snn.models.baselines import FrequencyBaseline
            from c5_snn.training.compare import (
                build_comparison,
                format_comparison_table,
                save_comparison,
            )
            from c5_snn.training.evaluate import evaluate_model
            from c5_snn.training.trainer import Trainer
            from c5_snn.utils.device import get_device
            from c5_snn.utils.logging import setup_logging
            from c5_snn.utils.seed import set_global_seed

            setup_logging("WARNING")

            seed_list = [int(s.strip()) for s in seeds.split(",")]
            raw_path = str(csv_path)
            window_size = 7
            split_ratios = (0.70, 0.15, 0.15)
            batch_size = 8

            df = load_csv(raw_path)
            X, y = build_windows(df, window_size)
            split_info = create_splits(
                n_samples=X.shape[0],
                ratios=split_ratios,
                window_size=window_size,
                dates=df["date"] if "date" in df.columns else None,
            )
            dataloaders = get_dataloaders(split_info, X, y, batch_size)
            test_loader = dataloaders["test"]
            test_split_size = len(test_loader.dataset)
            device = get_device()

            model_results = []

            # FrequencyBaseline
            freq_model = FrequencyBaseline(
                {"model": {"type": "frequency_baseline"}}
            )
            freq_eval = evaluate_model(freq_model, test_loader, device)
            model_results.append({
                "name": "frequency_baseline",
                "type": "heuristic",
                "phase": "baseline",
                "seed_metrics": [freq_eval["metrics"]],
                "training_time_s": 0,
                "environment": "local",
            })

            # GRU (tiny, 2 epochs)
            gru_seed_metrics = []
            for seed in seed_list:
                set_global_seed(seed)
                gru_cfg = {
                    "experiment": {"name": f"gru_seed{seed}", "seed": seed},
                    "data": {
                        "raw_path": raw_path,
                        "window_size": window_size,
                        "split_ratios": list(split_ratios),
                        "batch_size": batch_size,
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
                    "output": {
                        "dir": str(tmp_path / f"gru_seed{seed}"),
                    },
                    "log_level": "WARNING",
                }
                gru_model = get_model(gru_cfg)
                trainer = Trainer(gru_model, gru_cfg, dataloaders, device)
                trainer.run()
                gru_eval = evaluate_model(gru_model, test_loader, device)
                gru_seed_metrics.append(gru_eval["metrics"])

            model_results.append({
                "name": "gru_baseline",
                "type": "learned",
                "phase": "baseline",
                "seed_metrics": gru_seed_metrics,
                "training_time_s": 0,
                "environment": "local",
            })

            # SpikingMLP (tiny, 2 epochs)
            mlp_seed_metrics = []
            for seed in seed_list:
                set_global_seed(seed)
                mlp_cfg = {
                    "experiment": {
                        "name": f"spiking_mlp_seed{seed}",
                        "seed": seed,
                    },
                    "data": {
                        "raw_path": raw_path,
                        "window_size": window_size,
                        "split_ratios": list(split_ratios),
                        "batch_size": batch_size,
                    },
                    "model": {
                        "type": "spiking_mlp",
                        "encoding": "direct",
                        "timesteps": 10,
                        "hidden_sizes": [32],
                        "beta": 0.95,
                    },
                    "training": {
                        "epochs": 2,
                        "learning_rate": 0.001,
                        "early_stopping_patience": 10,
                    },
                    "output": {
                        "dir": str(tmp_path / f"mlp_seed{seed}"),
                    },
                    "log_level": "WARNING",
                }
                mlp_model = get_model(mlp_cfg)
                trainer = Trainer(mlp_model, mlp_cfg, dataloaders, device)
                trainer.run()
                mlp_eval = evaluate_model(mlp_model, test_loader, device)
                mlp_seed_metrics.append(mlp_eval["metrics"])

            model_results.append({
                "name": "spiking_mlp",
                "type": "learned",
                "phase": "phase_a",
                "seed_metrics": mlp_seed_metrics,
                "training_time_s": 0,
                "environment": "local",
            })

            # SpikingCNN1D (tiny, 2 epochs)
            cnn_seed_metrics = []
            for seed in seed_list:
                set_global_seed(seed)
                cnn_cfg = {
                    "experiment": {
                        "name": f"spiking_cnn1d_seed{seed}",
                        "seed": seed,
                    },
                    "data": {
                        "raw_path": raw_path,
                        "window_size": window_size,
                        "split_ratios": list(split_ratios),
                        "batch_size": batch_size,
                    },
                    "model": {
                        "type": "spiking_cnn1d",
                        "encoding": "direct",
                        "timesteps": 10,
                        "channels": [16],
                        "kernel_sizes": [3],
                        "beta": 0.95,
                    },
                    "training": {
                        "epochs": 2,
                        "learning_rate": 0.001,
                        "early_stopping_patience": 10,
                    },
                    "output": {
                        "dir": str(tmp_path / f"cnn_seed{seed}"),
                    },
                    "log_level": "WARNING",
                }
                cnn_model = get_model(cnn_cfg)
                trainer = Trainer(cnn_model, cnn_cfg, dataloaders, device)
                trainer.run()
                cnn_eval = evaluate_model(cnn_model, test_loader, device)
                cnn_seed_metrics.append(cnn_eval["metrics"])

            model_results.append({
                "name": "spiking_cnn1d",
                "type": "learned",
                "phase": "phase_a",
                "seed_metrics": cnn_seed_metrics,
                "training_time_s": 0,
                "environment": "local",
            })

            report = build_comparison(
                model_results, window_size, test_split_size
            )
            save_comparison(report, output_path)
            import click as _click

            _click.echo(format_comparison_table(report))
            _click.echo(f"Results saved to: {output_path}")

        # Replace the callback temporarily
        cli_module.phase_a.callback = patched_phase_a
        try:
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "phase-a",
                    "--seeds", "42,7",
                    "--output", str(output_json),
                ],
            )
        finally:
            cli_module.phase_a.callback = original_phase_a

        assert result.exit_code == 0, (
            f"CLI failed:\n{result.output}\n{result.exception}"
        )
        assert "frequency_baseline" in result.output
        assert "gru_baseline" in result.output
        assert "spiking_mlp" in result.output
        assert "spiking_cnn1d" in result.output

        # Verify JSON output
        assert output_json.exists()
        with open(output_json) as f:
            report = json.load(f)
        assert len(report["models"]) == 4
        names = [m["name"] for m in report["models"]]
        assert "frequency_baseline" in names
        assert "gru_baseline" in names
        assert "spiking_mlp" in names
        assert "spiking_cnn1d" in names

        # Verify phase fields
        phases = {m["name"]: m["phase"] for m in report["models"]}
        assert phases["frequency_baseline"] == "baseline"
        assert phases["gru_baseline"] == "baseline"
        assert phases["spiking_mlp"] == "phase_a"
        assert phases["spiking_cnn1d"] == "phase_a"

        # Verify seed counts
        seeds_map = {m["name"]: m["n_seeds"] for m in report["models"]}
        assert seeds_map["frequency_baseline"] == 1
        assert seeds_map["gru_baseline"] == 2
        assert seeds_map["spiking_mlp"] == 2
        assert seeds_map["spiking_cnn1d"] == 2

        # Verify metrics exist
        for model in report["models"]:
            assert "recall_at_20" in model["metrics_mean"]
            assert "hit_at_20" in model["metrics_mean"]
            assert "mrr" in model["metrics_mean"]

    def test_phase_a_invalid_seeds(self):
        """phase-a with non-integer seeds exits non-zero."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["phase-a", "--seeds", "abc,def"]
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Integration test: phase-b-sweep CLI (STORY-5.2)
# ---------------------------------------------------------------------------


class TestPhaseBSweepCLI:
    """Integration tests for the phase-b-sweep CLI command."""

    def test_phase_b_sweep_command_exists(self):
        """phase-b-sweep command is registered and shows help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["phase-b-sweep", "--help"])
        assert result.exit_code == 0
        assert "Run Spike-GRU hyperparameter sweep" in result.output

    def test_phase_b_sweep_output_option(self):
        """phase-b-sweep --help shows --output option with default."""
        runner = CliRunner()
        result = runner.invoke(cli, ["phase-b-sweep", "--help"])
        assert "--output" in result.output
        assert "phase_b_sweep.csv" in result.output

    def test_phase_b_sweep_top_k_option(self):
        """phase-b-sweep --help shows --top-k option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["phase-b-sweep", "--help"])
        assert "--top-k" in result.output

    def test_phase_b_sweep_seeds_option(self):
        """phase-b-sweep --help shows --seeds option with default."""
        runner = CliRunner()
        result = runner.invoke(cli, ["phase-b-sweep", "--help"])
        assert "--seeds" in result.output
        assert "42,123,7" in result.output

    def test_phase_b_sweep_invalid_seeds(self):
        """phase-b-sweep with non-integer seeds exits non-zero."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["phase-b-sweep", "--seeds", "abc,def"]
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Integration test: phase-b CLI (STORY-5.3)
# ---------------------------------------------------------------------------


def _make_phase_a_json(path, models=None):
    """Create a minimal Phase A comparison JSON for testing."""
    if models is None:
        models = [
            {
                "name": "frequency_baseline",
                "type": "heuristic",
                "phase": "baseline",
                "metrics_mean": {
                    "recall_at_5": 0.13, "recall_at_20": 0.52,
                    "hit_at_5": 0.52, "hit_at_20": 0.98, "mrr": 0.31,
                },
                "metrics_std": {
                    "recall_at_5": 0.0, "recall_at_20": 0.0,
                    "hit_at_5": 0.0, "hit_at_20": 0.0, "mrr": 0.0,
                },
                "n_seeds": 1,
                "training_time_s": 0,
                "environment": "local",
            },
            {
                "name": "gru_baseline",
                "type": "learned",
                "phase": "baseline",
                "metrics_mean": {
                    "recall_at_5": 0.12, "recall_at_20": 0.51,
                    "hit_at_5": 0.51, "hit_at_20": 0.97, "mrr": 0.31,
                },
                "metrics_std": {
                    "recall_at_5": 0.002, "recall_at_20": 0.003,
                    "hit_at_5": 0.01, "hit_at_20": 0.003, "mrr": 0.002,
                },
                "n_seeds": 3,
                "training_time_s": 60.0,
                "environment": "local",
            },
            {
                "name": "spiking_mlp",
                "type": "learned",
                "phase": "phase_a",
                "metrics_mean": {
                    "recall_at_5": 0.13, "recall_at_20": 0.5125,
                    "hit_at_5": 0.52, "hit_at_20": 0.97, "mrr": 0.31,
                },
                "metrics_std": {
                    "recall_at_5": 0.002, "recall_at_20": 0.003,
                    "hit_at_5": 0.01, "hit_at_20": 0.001, "mrr": 0.007,
                },
                "n_seeds": 3,
                "training_time_s": 33.0,
                "environment": "local",
            },
            {
                "name": "spiking_cnn1d",
                "type": "learned",
                "phase": "phase_a",
                "metrics_mean": {
                    "recall_at_5": 0.12, "recall_at_20": 0.515,
                    "hit_at_5": 0.50, "hit_at_20": 0.98, "mrr": 0.30,
                },
                "metrics_std": {
                    "recall_at_5": 0.002, "recall_at_20": 0.002,
                    "hit_at_5": 0.004, "hit_at_20": 0.002, "mrr": 0.001,
                },
                "n_seeds": 3,
                "training_time_s": 33.0,
                "environment": "local",
            },
        ]
    report = {
        "models": models,
        "generated_at": "2026-02-11T00:00:00+00:00",
        "window_size": 21,
        "test_split_size": 1753,
    }
    with open(path, "w") as f:
        json.dump(report, f)


def _make_phase_b_json(path, models=None):
    """Create a minimal Phase B top-3 JSON for testing."""
    if models is None:
        models = [
            {
                "name": "spike_gru_top1",
                "type": "learned",
                "phase": "phase_b",
                "metrics_mean": {
                    "recall_at_5": 0.127, "recall_at_20": 0.514,
                    "hit_at_5": 0.505, "hit_at_20": 0.98, "mrr": 0.311,
                },
                "metrics_std": {
                    "recall_at_5": 0.0, "recall_at_20": 0.0,
                    "hit_at_5": 0.0, "hit_at_20": 0.0, "mrr": 0.0,
                },
                "n_seeds": 3,
                "training_time_s": 114.0,
                "environment": "local",
            },
            {
                "name": "spike_gru_top2",
                "type": "learned",
                "phase": "phase_b",
                "metrics_mean": {
                    "recall_at_5": 0.127, "recall_at_20": 0.513,
                    "hit_at_5": 0.505, "hit_at_20": 0.98, "mrr": 0.311,
                },
                "metrics_std": {
                    "recall_at_5": 0.0, "recall_at_20": 0.0,
                    "hit_at_5": 0.0, "hit_at_20": 0.0, "mrr": 0.0,
                },
                "n_seeds": 3,
                "training_time_s": 860.0,
                "environment": "local",
            },
            {
                "name": "spike_gru_top3",
                "type": "learned",
                "phase": "phase_b",
                "metrics_mean": {
                    "recall_at_5": 0.127, "recall_at_20": 0.512,
                    "hit_at_5": 0.505, "hit_at_20": 0.98, "mrr": 0.311,
                },
                "metrics_std": {
                    "recall_at_5": 0.0, "recall_at_20": 0.0,
                    "hit_at_5": 0.0, "hit_at_20": 0.0, "mrr": 0.0,
                },
                "n_seeds": 3,
                "training_time_s": 100.0,
                "environment": "local",
            },
        ]
    report = {
        "models": models,
        "generated_at": "2026-02-11T00:00:00+00:00",
        "window_size": 21,
        "test_split_size": 1753,
    }
    with open(path, "w") as f:
        json.dump(report, f)


class TestPhaseBCLI:
    """Integration tests for the phase-b CLI command (STORY-5.3)."""

    def test_phase_b_command_exists(self):
        """phase-b command is registered and shows help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["phase-b", "--help"])
        assert result.exit_code == 0
        assert "Build cumulative comparison" in result.output

    def test_phase_b_output_option(self):
        """phase-b --help shows --output option with default."""
        runner = CliRunner()
        result = runner.invoke(cli, ["phase-b", "--help"])
        assert "--output" in result.output
        assert "cumulative_comparison.json" in result.output

    def test_phase_b_phase_a_option(self):
        """phase-b --help shows --phase-a option with default."""
        runner = CliRunner()
        result = runner.invoke(cli, ["phase-b", "--help"])
        assert "--phase-a" in result.output
        assert "phase_a_comparison.json" in result.output

    def test_phase_b_phase_b_top_option(self):
        """phase-b --help shows --phase-b-top option with default."""
        runner = CliRunner()
        result = runner.invoke(cli, ["phase-b", "--help"])
        assert "--phase-b-top" in result.output
        assert "phase_b_top3.json" in result.output

    def test_phase_b_missing_phase_a(self, tmp_path):
        """phase-b with missing Phase A file exits non-zero."""
        phase_b_json = tmp_path / "phase_b_top3.json"
        _make_phase_b_json(phase_b_json)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "phase-b",
            "--phase-a", str(tmp_path / "nonexistent.json"),
            "--phase-b-top", str(phase_b_json),
        ])
        assert result.exit_code != 0

    def test_phase_b_missing_phase_b(self, tmp_path):
        """phase-b with missing Phase B file exits non-zero."""
        phase_a_json = tmp_path / "phase_a.json"
        _make_phase_a_json(phase_a_json)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "phase-b",
            "--phase-a", str(phase_a_json),
            "--phase-b-top", str(tmp_path / "nonexistent.json"),
        ])
        assert result.exit_code != 0

    def test_phase_b_produces_output(self, tmp_path):
        """phase-b merges Phase A + B and produces cumulative JSON."""
        phase_a_json = tmp_path / "phase_a.json"
        phase_b_json = tmp_path / "phase_b_top3.json"
        output_json = tmp_path / "cumulative.json"

        _make_phase_a_json(phase_a_json)
        _make_phase_b_json(phase_b_json)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "phase-b",
            "--phase-a", str(phase_a_json),
            "--phase-b-top", str(phase_b_json),
            "--output", str(output_json),
        ])

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert "Cumulative Leaderboard" in result.output
        assert "Analysis:" in result.output
        assert "spike_gru" in result.output

        # Verify JSON output
        assert output_json.exists()
        with open(output_json) as f:
            report = json.load(f)
        assert len(report["models"]) == 5
        names = [m["name"] for m in report["models"]]
        assert "frequency_baseline" in names
        assert "gru_baseline" in names
        assert "spiking_mlp" in names
        assert "spiking_cnn1d" in names
        assert "spike_gru" in names
        assert report["window_size"] == 21
        assert report["test_split_size"] == 1753

    def test_phase_b_selects_best_spike_gru(self, tmp_path):
        """phase-b selects highest recall@20 from Phase B top-3."""
        phase_a_json = tmp_path / "phase_a.json"
        phase_b_json = tmp_path / "phase_b_top3.json"
        output_json = tmp_path / "cumulative.json"

        _make_phase_a_json(phase_a_json)
        _make_phase_b_json(phase_b_json)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "phase-b",
            "--phase-a", str(phase_a_json),
            "--phase-b-top", str(phase_b_json),
            "--output", str(output_json),
        ])

        assert result.exit_code == 0
        with open(output_json) as f:
            report = json.load(f)

        spike_gru = [
            m for m in report["models"] if m["name"] == "spike_gru"
        ][0]
        # Best from Phase B top-3 has recall@20=0.514
        assert spike_gru["metrics_mean"]["recall_at_20"] == pytest.approx(
            0.514
        )
