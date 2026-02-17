"""Tests for the LLN-Pattern date CSV loader."""

import numpy as np
import pandas as pd
import pytest

from c5_snn.lln_pattern.loader import (
    EXPECTED_COLUMNS,
    M_COLUMNS,
    load_date_csv,
    validate_date_df,
)
from c5_snn.utils.exceptions import DataValidationError


def _make_valid_date_df(n: int = 50) -> pd.DataFrame:
    """Create a valid CA5_date-format DataFrame with n rows."""
    rng = np.random.RandomState(42)
    base = pd.Timestamp("2000-01-01")
    rows = []
    for i in range(n):
        dt = base + pd.Timedelta(days=i)
        parts = sorted(
            rng.choice(range(1, 40), size=5, replace=False).tolist()
        )
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
    return pd.DataFrame(rows)


@pytest.fixture
def valid_date_df():
    return _make_valid_date_df(50)


@pytest.fixture
def valid_date_csv_file(valid_date_df, tmp_path):
    path = tmp_path / "CA5_date.csv"
    valid_date_df.to_csv(path, index=False)
    return str(path)


class TestLoadDateCSV:
    def test_load_real_file(self):
        """Loads actual CA5_date.csv if it exists."""
        import os

        path = "data/raw/CA5_date.csv"
        if not os.path.exists(path):
            pytest.skip("CA5_date.csv not found")
        df = load_date_csv(path)
        assert len(df) > 11000
        assert list(df.columns) == EXPECTED_COLUMNS

    def test_load_synthetic(self, valid_date_csv_file):
        df = load_date_csv(valid_date_csv_file)
        assert len(df) == 50
        assert list(df.columns) == EXPECTED_COLUMNS

    def test_missing_file_raises(self):
        with pytest.raises(DataValidationError, match="not found"):
            load_date_csv("nonexistent/path.csv")


class TestValidateDateDF:
    def test_valid_df_passes(self, valid_date_df):
        validate_date_df(valid_date_df)

    def test_wrong_columns_raises(self, valid_date_df):
        df = valid_date_df.rename(columns={"m_1": "bad_col"})
        with pytest.raises(DataValidationError, match="Expected columns"):
            validate_date_df(df)

    def test_nan_values_raises(self, valid_date_df):
        df = valid_date_df.copy()
        df.loc[0, "m_1"] = np.nan
        with pytest.raises(DataValidationError, match="NaN"):
            validate_date_df(df)

    def test_out_of_range_raises(self, valid_date_df):
        df = valid_date_df.copy()
        df.loc[0, "m_5"] = 40
        with pytest.raises(DataValidationError, match="out of range"):
            validate_date_df(df)

    def test_unsorted_order_raises(self):
        df = pd.DataFrame(
            {
                "date": ["1/1/2000"],
                "m_1": [10],
                "m_2": [5],
                "m_3": [20],
                "m_4": [25],
                "m_5": [30],
            }
        )
        with pytest.raises(DataValidationError, match="m_1 < m_2"):
            validate_date_df(df)

    def test_sorted_order_passes(self):
        df = pd.DataFrame(
            {
                "date": ["1/1/2000"],
                "m_1": [1],
                "m_2": [5],
                "m_3": [10],
                "m_4": [25],
                "m_5": [39],
            }
        )
        validate_date_df(df)

    def test_real_data_sorted(self):
        """Verify actual CA5_date.csv has sorted m values."""
        import os

        path = "data/raw/CA5_date.csv"
        if not os.path.exists(path):
            pytest.skip("CA5_date.csv not found")
        df = load_date_csv(path)
        m_vals = df[M_COLUMNS].values
        diffs = np.diff(m_vals, axis=1)
        assert (diffs > 0).all()
