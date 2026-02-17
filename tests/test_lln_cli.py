"""Tests for the LLN-Pattern CLI commands."""

import numpy as np
import pandas as pd
from click.testing import CliRunner

from c5_snn.cli import cli


def _make_date_csv(tmp_path, n: int = 200) -> str:
    """Write a synthetic CA5_date CSV and return the path."""
    rng = np.random.RandomState(42)
    base = pd.Timestamp("2020-01-01")
    rows = []
    for i in range(n):
        parts = sorted(rng.choice(range(1, 40), size=5, replace=False))
        dt = base + pd.Timedelta(days=i)
        rows.append(
            {
                "date": f"{dt.month}/{dt.day}/{dt.year}",
                "m_1": parts[0],
                "m_2": parts[1],
                "m_3": parts[2],
                "m_4": parts[3],
                "m_5": parts[4],
            }
        )
    df = pd.DataFrame(rows)
    path = tmp_path / "CA5_date.csv"
    df.to_csv(path, index=False)
    return str(path)


class TestLLNPredictCLI:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["lln-predict", "--help"])
        assert result.exit_code == 0
        assert "exclusion set" in result.output.lower()

    def test_runs_with_synthetic_data(self, tmp_path):
        csv_path = _make_date_csv(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["lln-predict", "--data-path", csv_path]
        )
        assert result.exit_code == 0
        assert "Exclusion set" in result.output

    def test_no_pattern_flag(self, tmp_path):
        csv_path = _make_date_csv(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["lln-predict", "--data-path", csv_path, "--no-pattern"]
        )
        assert result.exit_code == 0
        assert "LLN" in result.output

    def test_no_boundary_flag(self, tmp_path):
        csv_path = _make_date_csv(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["lln-predict", "--data-path", csv_path, "--no-boundary"]
        )
        assert result.exit_code == 0

    def test_custom_k_exclude(self, tmp_path):
        csv_path = _make_date_csv(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["lln-predict", "--data-path", csv_path, "--k-exclude", "15"],
        )
        assert result.exit_code == 0
        assert "15 values" in result.output

    def test_missing_file_error(self):
        runner = CliRunner()
        result = runner.invoke(
            cli, ["lln-predict", "--data-path", "nonexistent.csv"]
        )
        assert result.exit_code != 0


class TestLLNHoldoutCLI:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["lln-holdout", "--help"])
        assert result.exit_code == 0
        assert "holdout" in result.output.lower()

    def test_runs_with_synthetic_data(self, tmp_path):
        csv_path = _make_date_csv(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["lln-holdout", "--data-path", csv_path, "--n-holdout", "10"],
        )
        assert result.exit_code == 0
        assert "0 wrong" in result.output

    def test_output_format(self, tmp_path):
        csv_path = _make_date_csv(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["lln-holdout", "--data-path", csv_path, "--n-holdout", "5"],
        )
        assert result.exit_code == 0
        assert "Mean wrong" in result.output
        assert "LLN-Pattern Holdout Test" in result.output

    def test_no_pattern_flag(self, tmp_path):
        csv_path = _make_date_csv(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "lln-holdout",
                "--data-path",
                csv_path,
                "--n-holdout",
                "5",
                "--no-pattern",
            ],
        )
        assert result.exit_code == 0

    def test_custom_boundary_penalty(self, tmp_path):
        csv_path = _make_date_csv(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "lln-holdout",
                "--data-path",
                csv_path,
                "--n-holdout",
                "5",
                "--boundary-penalty",
                "2.0",
            ],
        )
        assert result.exit_code == 0

    def test_missing_file_error(self):
        runner = CliRunner()
        result = runner.invoke(
            cli, ["lln-holdout", "--data-path", "nonexistent.csv"]
        )
        assert result.exit_code != 0
