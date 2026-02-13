"""CLI integration tests for predict and holdout-test commands."""

import pytest
from click.testing import CliRunner

from c5_snn.cli import cli
from tests.conftest import _make_valid_df


@pytest.fixture
def csv_file(tmp_path):
    """Write a 50-row valid CSV for testing."""
    df = _make_valid_df(50)
    path = tmp_path / "test_data.csv"
    df.to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# predict command — existence and options
# ---------------------------------------------------------------------------


class TestPredictCommand:
    def test_command_exists(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["predict", "--help"])
        assert result.exit_code == 0
        assert "Predict Top-K" in result.output

    def test_has_checkpoint_option(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["predict", "--help"])
        assert "--checkpoint" in result.output

    def test_has_model_type_option(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["predict", "--help"])
        assert "--model-type" in result.output

    def test_has_top_k_option(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["predict", "--help"])
        assert "--top-k" in result.output

    def test_has_asof_option(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["predict", "--help"])
        assert "--asof" in result.output

    def test_requires_model_source(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["predict"])
        assert result.exit_code != 0

    def test_rejects_both_sources(self, csv_file):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "predict",
                "--checkpoint",
                "fake.pt",
                "--model-type",
                "frequency_baseline",
                "--data-path",
                csv_file,
            ],
        )
        assert result.exit_code != 0
        assert "not both" in result.output

    def test_frequency_baseline_prediction(self, csv_file):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "predict",
                "--model-type",
                "frequency_baseline",
                "--data-path",
                csv_file,
                "--window-size",
                "7",
            ],
        )
        assert result.exit_code == 0
        assert "Prediction for next event" in result.output
        assert "P_" in result.output

    def test_top_k_5(self, csv_file):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "predict",
                "--model-type",
                "frequency_baseline",
                "--data-path",
                csv_file,
                "--window-size",
                "7",
                "--top-k",
                "5",
            ],
        )
        assert result.exit_code == 0
        # Count lines with "P_" — should be exactly 5
        p_lines = [
            ln
            for ln in result.output.split("\n")
            if "P_" in ln and "Part" not in ln
        ]
        assert len(p_lines) == 5

    def test_missing_checkpoint_errors(self, csv_file):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "predict",
                "--checkpoint",
                "nonexistent.pt",
                "--data-path",
                csv_file,
            ],
        )
        assert result.exit_code != 0
        assert "not found" in result.output


# ---------------------------------------------------------------------------
# holdout-test command — existence and options
# ---------------------------------------------------------------------------


class TestHoldoutTestCommand:
    def test_command_exists(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["holdout-test", "--help"])
        assert result.exit_code == 0
        assert "holdout" in result.output.lower()

    def test_has_n_holdout_option(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["holdout-test", "--help"])
        assert "--n-holdout" in result.output

    def test_requires_model_source(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["holdout-test"])
        assert result.exit_code != 0

    def test_frequency_baseline_single_holdout(self, csv_file):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "holdout-test",
                "--model-type",
                "frequency_baseline",
                "--data-path",
                csv_file,
                "--window-size",
                "7",
                "--n-holdout",
                "1",
            ],
        )
        assert result.exit_code == 0
        assert "Holdout Test Results" in result.output
        assert "recall_at_20" in result.output

    def test_frequency_baseline_multi_holdout(self, csv_file):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "holdout-test",
                "--model-type",
                "frequency_baseline",
                "--data-path",
                csv_file,
                "--window-size",
                "7",
                "--n-holdout",
                "5",
            ],
        )
        assert result.exit_code == 0
        assert "Holdout Test Results" in result.output
        assert "Summary:" in result.output

    def test_too_many_holdout_errors(self, csv_file):
        """50-row CSV with W=7 can hold out at most 43 rows."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "holdout-test",
                "--model-type",
                "frequency_baseline",
                "--data-path",
                csv_file,
                "--window-size",
                "7",
                "--n-holdout",
                "50",
            ],
        )
        assert result.exit_code != 0
        assert "Need at least" in result.output


# ---------------------------------------------------------------------------
# calendar_enhanced model type — predict and holdout-test
# ---------------------------------------------------------------------------


class TestCalendarEnhancedPredict:
    def test_prediction_runs(self, csv_file):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "predict",
                "--model-type",
                "calendar_enhanced",
                "--data-path",
                csv_file,
            ],
        )
        assert result.exit_code == 0
        assert "Prediction for next event" in result.output
        assert "calendar_enhanced" in result.output
        assert "P_" in result.output

    def test_top_k_5(self, csv_file):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "predict",
                "--model-type",
                "calendar_enhanced",
                "--data-path",
                csv_file,
                "--top-k",
                "5",
            ],
        )
        assert result.exit_code == 0
        p_lines = [
            ln
            for ln in result.output.split("\n")
            if "P_" in ln and "Part" not in ln and "calendar" not in ln
        ]
        assert len(p_lines) == 5


class TestCalendarEnhancedHoldout:
    def test_single_holdout(self, csv_file):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "holdout-test",
                "--model-type",
                "calendar_enhanced",
                "--data-path",
                csv_file,
                "--window-size",
                "7",
                "--n-holdout",
                "1",
            ],
        )
        assert result.exit_code == 0
        assert "Holdout Test Results" in result.output
        assert "recall_at_20" in result.output

    def test_multi_holdout(self, csv_file):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "holdout-test",
                "--model-type",
                "calendar_enhanced",
                "--data-path",
                csv_file,
                "--window-size",
                "7",
                "--n-holdout",
                "5",
            ],
        )
        assert result.exit_code == 0
        assert "Holdout Test Results" in result.output
        assert "Summary:" in result.output
