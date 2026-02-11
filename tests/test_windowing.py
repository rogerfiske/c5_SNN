"""Tests for sliding window tensor construction."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from c5_snn.data.validation import P_COLUMNS
from c5_snn.data.windowing import build_windows, save_tensors
from c5_snn.utils.exceptions import ConfigError, DataValidationError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_valid_df(n: int = 30) -> pd.DataFrame:
    """Create a valid CA5-format DataFrame with n rows."""
    rng = np.random.RandomState(42)
    rows = []
    base_year = 2000
    for i in range(n):
        date = f"{base_year + i // 12}/{(i % 12) + 1}/15"
        parts = sorted(rng.choice(range(1, 40), size=5, replace=False).tolist())
        row: dict = {"date": date}
        for j, p in enumerate(parts, 1):
            row[f"m_{j}"] = p
        for k in range(1, 40):
            row[f"P_{k}"] = 1 if k in parts else 0
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def df30() -> pd.DataFrame:
    """30-row valid DataFrame for windowing tests."""
    return _make_valid_df(30)


@pytest.fixture
def df10() -> pd.DataFrame:
    """10-row valid DataFrame (edge-case testing)."""
    return _make_valid_df(10)


# ---------------------------------------------------------------------------
# build_windows — shape tests
# ---------------------------------------------------------------------------

class TestBuildWindowsShapes:
    """Verify output tensor shapes for various window sizes."""

    def test_default_window(self, df30):
        """W=7 on 30 rows -> N=23 samples."""
        X, y = build_windows(df30, window_size=7)
        assert X.shape == (23, 7, 39)
        assert y.shape == (23, 39)

    def test_window_10(self, df30):
        """W=10 on 30 rows -> N=20 samples."""
        X, y = build_windows(df30, window_size=10)
        assert X.shape == (20, 10, 39)
        assert y.shape == (20, 39)

    def test_single_sample(self, df10):
        """W=9 on 10 rows -> exactly 1 sample."""
        X, y = build_windows(df10, window_size=9)
        assert X.shape == (1, 9, 39)
        assert y.shape == (1, 39)

    def test_dtype_float32(self, df30):
        """Tensors must be float32."""
        X, y = build_windows(df30, window_size=7)
        assert X.dtype == torch.float32
        assert y.dtype == torch.float32

    def test_n_features_is_39(self, df30):
        """Only P-columns (39 features), not date or m-columns."""
        X, y = build_windows(df30, window_size=7)
        assert X.shape[2] == 39
        assert y.shape[1] == 39

    def test_n_samples_formula(self, df30):
        """N = len(df) - W for various window sizes."""
        for w in [7, 10, 15, 20]:
            X, y = build_windows(df30, window_size=w)
            expected_n = len(df30) - w
            assert X.shape[0] == expected_n
            assert y.shape[0] == expected_n


# ---------------------------------------------------------------------------
# build_windows — alignment & no-leakage tests
# ---------------------------------------------------------------------------

class TestBuildWindowsAlignment:
    """Verify first/last window alignment and no data leakage."""

    def test_first_window_alignment(self, df30):
        """X[0] should contain the first W rows of P-columns."""
        W = 7
        X, y = build_windows(df30, window_size=W)

        expected_x0 = torch.tensor(df30[P_COLUMNS].values[:W], dtype=torch.float32)
        assert torch.equal(X[0], expected_x0)

    def test_first_target_alignment(self, df30):
        """y[0] should be the P-vector at index W (row after the first window)."""
        W = 7
        X, y = build_windows(df30, window_size=W)

        expected_y0 = torch.tensor(df30[P_COLUMNS].iloc[W].values, dtype=torch.float32)
        assert torch.equal(y[0], expected_y0)

    def test_last_window_alignment(self, df30):
        """X[-1] should contain the last W rows before the final row."""
        W = 7
        X, y = build_windows(df30, window_size=W)
        n = len(df30)

        # Last window: rows [n-W-1 .. n-2], target: row n-1
        expected_x_last = torch.tensor(
            df30[P_COLUMNS].values[n - W - 1 : n - 1], dtype=torch.float32
        )
        assert torch.equal(X[-1], expected_x_last)

    def test_last_target_is_last_row(self, df30):
        """y[-1] should be the P-vector of the very last row in the DataFrame."""
        W = 7
        X, y = build_windows(df30, window_size=W)

        expected_y_last = torch.tensor(
            df30[P_COLUMNS].iloc[-1].values, dtype=torch.float32
        )
        assert torch.equal(y[-1], expected_y_last)

    def test_no_input_target_overlap(self, df30):
        """Target row index (t+W) is strictly after the last input row (t+W-1)."""
        W = 7
        X, y = build_windows(df30, window_size=W)
        p_all = torch.tensor(df30[P_COLUMNS].values, dtype=torch.float32)

        for t in range(len(X)):
            # Input window covers rows [t, t+W)
            assert torch.equal(X[t], p_all[t : t + W])
            # Target is row t+W (strictly after window)
            assert torch.equal(y[t], p_all[t + W])

    def test_consecutive_windows_shift_by_one(self, df30):
        """Consecutive windows overlap by W-1 rows."""
        W = 7
        X, y = build_windows(df30, window_size=W)

        # X[1] rows [1..7] should share rows [1..6] with X[0] rows [0..6]
        assert torch.equal(X[0][1:], X[1][:-1])

    def test_target_row_sums(self, df30):
        """Each target vector should sum to 5 (multi-hot with 5 ones)."""
        W = 7
        _, y = build_windows(df30, window_size=W)

        row_sums = y.sum(dim=1)
        assert torch.all(row_sums == 5.0)


# ---------------------------------------------------------------------------
# build_windows — error cases
# ---------------------------------------------------------------------------

class TestBuildWindowsErrors:
    """Verify proper errors for invalid inputs."""

    def test_window_too_small(self, df30):
        """W < 7 should raise ConfigError."""
        with pytest.raises(ConfigError, match="window_size must be an integer"):
            build_windows(df30, window_size=6)

    def test_window_too_large(self, df30):
        """W > 90 should raise ConfigError."""
        with pytest.raises(ConfigError, match="window_size must be an integer"):
            build_windows(df30, window_size=91)

    def test_window_zero(self, df30):
        """W=0 should raise ConfigError."""
        with pytest.raises(ConfigError):
            build_windows(df30, window_size=0)

    def test_window_negative(self, df30):
        """W=-1 should raise ConfigError."""
        with pytest.raises(ConfigError):
            build_windows(df30, window_size=-1)

    def test_dataframe_too_small(self, df10):
        """10 rows with W=10 -> 0 samples -> DataValidationError."""
        with pytest.raises(DataValidationError, match="need at least 11"):
            build_windows(df10, window_size=10)

    def test_dataframe_exact_window_size(self):
        """Exactly W rows -> 0 samples -> DataValidationError."""
        df = _make_valid_df(7)
        with pytest.raises(DataValidationError, match="need at least 8"):
            build_windows(df, window_size=7)

    def test_dataframe_one_more_than_window(self):
        """W+1 rows -> exactly 1 sample (valid)."""
        df = _make_valid_df(8)
        X, y = build_windows(df, window_size=7)
        assert X.shape[0] == 1

    def test_window_boundary_min(self, df30):
        """W=7 (minimum) should work."""
        X, y = build_windows(df30, window_size=7)
        assert X.shape[1] == 7

    def test_window_boundary_max(self):
        """W=90 (maximum) should work if enough data."""
        df = _make_valid_df(100)
        X, y = build_windows(df, window_size=90)
        assert X.shape == (10, 90, 39)


# ---------------------------------------------------------------------------
# save_tensors — persistence tests
# ---------------------------------------------------------------------------

class TestSaveTensors:
    """Verify tensor and metadata persistence."""

    def test_files_created(self, df30, tmp_path):
        """save_tensors creates X.pt, y.pt, and metadata JSON."""
        X, y = build_windows(df30, window_size=7)

        # Write a source CSV so we can hash it
        csv_path = tmp_path / "source.csv"
        df30.to_csv(csv_path, index=False)

        save_tensors(X, y, 7, str(csv_path), df30, str(tmp_path / "out"))

        assert (tmp_path / "out" / "X_w7.pt").exists()
        assert (tmp_path / "out" / "y_w7.pt").exists()
        assert (tmp_path / "out" / "tensor_meta_w7.json").exists()

    def test_tensors_roundtrip(self, df30, tmp_path):
        """Saved tensors can be loaded and match originals."""
        X, y = build_windows(df30, window_size=7)

        csv_path = tmp_path / "source.csv"
        df30.to_csv(csv_path, index=False)

        out_dir = str(tmp_path / "out")
        save_tensors(X, y, 7, str(csv_path), df30, out_dir)

        X_loaded = torch.load(tmp_path / "out" / "X_w7.pt", weights_only=True)
        y_loaded = torch.load(tmp_path / "out" / "y_w7.pt", weights_only=True)

        assert torch.equal(X, X_loaded)
        assert torch.equal(y, y_loaded)

    def test_metadata_fields(self, df30, tmp_path):
        """Metadata JSON has all required fields."""
        X, y = build_windows(df30, window_size=7)

        csv_path = tmp_path / "source.csv"
        df30.to_csv(csv_path, index=False)

        meta = save_tensors(X, y, 7, str(csv_path), df30, str(tmp_path / "out"))

        assert meta["source_file"] == str(csv_path)
        assert meta["source_hash"].startswith("sha256:")
        assert len(meta["source_hash"]) == len("sha256:") + 64  # SHA-256 hex
        assert meta["window_size"] == 7
        assert meta["n_samples"] == 23
        assert meta["n_features"] == 39
        assert isinstance(meta["date_range"], list)
        assert len(meta["date_range"]) == 2
        assert "created_at" in meta

    def test_metadata_json_roundtrip(self, df30, tmp_path):
        """Metadata JSON can be loaded and parsed."""
        X, y = build_windows(df30, window_size=7)

        csv_path = tmp_path / "source.csv"
        df30.to_csv(csv_path, index=False)

        save_tensors(X, y, 7, str(csv_path), df30, str(tmp_path / "out"))

        meta_path = tmp_path / "out" / "tensor_meta_w7.json"
        with open(meta_path) as f:
            loaded = json.load(f)

        assert loaded["window_size"] == 7
        assert loaded["n_samples"] == 23
        assert loaded["n_features"] == 39

    def test_creates_output_dir(self, df30, tmp_path):
        """save_tensors creates the output directory if it doesn't exist."""
        X, y = build_windows(df30, window_size=7)

        csv_path = tmp_path / "source.csv"
        df30.to_csv(csv_path, index=False)

        deep_dir = str(tmp_path / "a" / "b" / "c")
        save_tensors(X, y, 7, str(csv_path), df30, deep_dir)

        assert (tmp_path / "a" / "b" / "c" / "X_w7.pt").exists()

    def test_different_window_sizes_separate_files(self, df30, tmp_path):
        """Different W values produce separate file sets."""
        csv_path = tmp_path / "source.csv"
        df30.to_csv(csv_path, index=False)
        out_dir = str(tmp_path / "out")

        X7, y7 = build_windows(df30, window_size=7)
        save_tensors(X7, y7, 7, str(csv_path), df30, out_dir)

        X10, y10 = build_windows(df30, window_size=10)
        save_tensors(X10, y10, 10, str(csv_path), df30, out_dir)

        assert (tmp_path / "out" / "X_w7.pt").exists()
        assert (tmp_path / "out" / "X_w10.pt").exists()
        assert (tmp_path / "out" / "tensor_meta_w7.json").exists()
        assert (tmp_path / "out" / "tensor_meta_w10.json").exists()
