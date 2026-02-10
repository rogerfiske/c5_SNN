"""Tests for CSV data loading."""

import pytest

from c5_snn.data.loader import load_csv
from c5_snn.utils.exceptions import DataValidationError


class TestLoadCsv:
    def test_load_valid_csv(self, tiny_csv_file):
        """load_csv on a valid CSV returns a DataFrame with correct shape."""
        df = load_csv(tiny_csv_file)
        assert df.shape == (20, 45)
        assert list(df.columns)[0] == "date"
        assert list(df.columns)[-1] == "P_39"

    def test_load_nonexistent_raises(self):
        """load_csv on a non-existent path raises DataValidationError."""
        with pytest.raises(DataValidationError, match="Data file not found"):
            load_csv("/nonexistent/path/data.csv")

    def test_date_column_is_string(self, tiny_csv_file):
        """The date column is loaded as string, not datetime."""
        df = load_csv(tiny_csv_file)
        assert df["date"].dtype == object  # pandas string dtype
