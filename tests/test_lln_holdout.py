"""Tests for the LLN-Pattern holdout evaluation."""

import numpy as np
import pandas as pd
import pytest

from c5_snn.lln_pattern.holdout import run_lln_holdout_test


def _make_date_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    rows = []
    for i in range(n):
        parts = sorted(rng.choice(range(1, 40), size=5, replace=False))
        rows.append(
            {
                "date": f"{(i % 12) + 1}/{(i % 28) + 1}/2020",
                "m_1": parts[0],
                "m_2": parts[1],
                "m_3": parts[2],
                "m_4": parts[3],
                "m_5": parts[4],
            }
        )
    return pd.DataFrame(rows)


class TestRunLLNHoldoutTest:
    def test_returns_expected_keys(self):
        df = _make_date_df(200)
        result = run_lln_holdout_test(df, n_holdout=10)
        assert "summary" in result
        assert "per_sample" in result
        assert "config" in result

    def test_sample_count_matches_n_holdout(self):
        df = _make_date_df(200)
        result = run_lln_holdout_test(df, n_holdout=10)
        assert len(result["per_sample"]) == 10

    def test_n_wrong_in_range(self):
        df = _make_date_df(200)
        result = run_lln_holdout_test(df, n_holdout=10)
        for sample in result["per_sample"]:
            assert 0 <= sample["n_wrong"] <= 5

    def test_too_few_rows_raises(self):
        df = _make_date_df(30)
        with pytest.raises(ValueError, match="Need at least"):
            run_lln_holdout_test(df, n_holdout=10)

    def test_holdout_fraction_default(self):
        df = _make_date_df(200)
        result = run_lln_holdout_test(df, n_holdout=None)
        # 10% of 200 = 20
        assert result["config"]["n_holdout"] == 20
        assert len(result["per_sample"]) == 20

    def test_summary_distribution(self):
        df = _make_date_df(200)
        result = run_lln_holdout_test(df, n_holdout=10)
        summary = result["summary"]
        assert summary["total"] == 10
        assert sum(summary["distribution"].values()) == 10
        assert 0 <= summary["zero_wrong_pct"] <= 100.0

    def test_per_sample_has_required_fields(self):
        df = _make_date_df(200)
        result = run_lln_holdout_test(df, n_holdout=5)
        for sample in result["per_sample"]:
            assert "row_index" in sample
            assert "date" in sample
            assert "actual_values" in sample
            assert "excluded_values" in sample
            assert "n_wrong" in sample
            assert len(sample["actual_values"]) == 5
            assert len(sample["excluded_values"]) == 20

    def test_no_pattern_mode(self):
        df = _make_date_df(200)
        result = run_lln_holdout_test(df, n_holdout=5, use_pattern=False)
        assert result["config"]["use_pattern"] is False

    def test_no_data_leakage(self):
        """Target row index should always be >= first holdout index."""
        df = _make_date_df(200)
        result = run_lln_holdout_test(df, n_holdout=10)
        first_holdout_idx = len(df) - 10
        for sample in result["per_sample"]:
            assert sample["row_index"] >= first_holdout_idx
