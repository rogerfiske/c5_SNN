"""Tests for time-based splitting and DataLoader creation."""

import json

import pandas as pd
import pytest
import torch

from c5_snn.data.dataset import CA5Dataset, get_dataloaders
from c5_snn.data.splits import create_splits, load_splits, save_splits
from c5_snn.utils.exceptions import ConfigError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_tensors():
    """100-sample windowed tensors with W=7, 39 features."""
    torch.manual_seed(42)
    X = torch.rand(100, 7, 39)
    y = torch.rand(100, 39)
    return X, y


@pytest.fixture
def sample_dates():
    """Date series with 107 entries (100 samples + W=7)."""
    return pd.Series(
        pd.date_range("2000-01-01", periods=107, freq="D").strftime("%Y-%m-%d")
    )


@pytest.fixture
def default_split(sample_dates):
    """SplitInfo with default 70/15/15 ratios on 100 samples."""
    return create_splits(100, (0.70, 0.15, 0.15), window_size=7, dates=sample_dates)


# ---------------------------------------------------------------------------
# create_splits — basic behavior
# ---------------------------------------------------------------------------


class TestCreateSplits:
    """Verify split boundary computation."""

    def test_default_ratios(self, default_split):
        """70/15/15 on 100 samples -> 70/15/15."""
        assert default_split.counts["train"] == 70
        assert default_split.counts["val"] == 15
        assert default_split.counts["test"] == 15

    def test_indices_contiguous(self, default_split):
        """Split indices cover [0, N) with no gaps."""
        assert default_split.indices["train"] == [0, 70]
        assert default_split.indices["val"] == [70, 85]
        assert default_split.indices["test"] == [85, 100]

    def test_counts_sum_to_total(self, default_split):
        """All counts must sum to n_samples."""
        total = sum(default_split.counts.values())
        assert total == 100

    def test_ratios_stored(self, default_split):
        """Ratios dict matches input."""
        assert default_split.ratios == {
            "train": 0.70,
            "val": 0.15,
            "test": 0.15,
        }

    def test_window_size_stored(self, default_split):
        """Window size metadata preserved."""
        assert default_split.window_size == 7

    def test_remainder_goes_to_test(self):
        """When int truncation leaves remainder, test gets it."""
        # 33 * 0.7 = 23.1 -> 23, 33 * 0.15 = 4.95 -> 4, test = 33-23-4 = 6
        info = create_splits(33, (0.70, 0.15, 0.15), window_size=7)
        assert info.counts["train"] == 23
        assert info.counts["val"] == 4
        assert info.counts["test"] == 6
        assert sum(info.counts.values()) == 33

    def test_80_10_10_ratios(self):
        """Different ratio config produces valid splits."""
        info = create_splits(100, (0.80, 0.10, 0.10), window_size=7)
        assert info.counts["train"] == 80
        assert info.counts["val"] == 10
        assert info.counts["test"] == 10

    def test_60_20_20_ratios(self):
        """60/20/20 split on 50 samples."""
        info = create_splits(50, (0.60, 0.20, 0.20), window_size=7)
        assert info.counts["train"] == 30
        assert info.counts["val"] == 10
        assert info.counts["test"] == 10


# ---------------------------------------------------------------------------
# create_splits — strict ordering (no leakage)
# ---------------------------------------------------------------------------


class TestSplitOrdering:
    """Verify strict chronological ordering."""

    def test_train_before_val(self, default_split):
        """max(train) < min(val)."""
        train_end = default_split.indices["train"][1]
        val_start = default_split.indices["val"][0]
        assert train_end <= val_start

    def test_val_before_test(self, default_split):
        """max(val) < min(test)."""
        val_end = default_split.indices["val"][1]
        test_start = default_split.indices["test"][0]
        assert val_end <= test_start

    def test_no_overlap_train_val(self, default_split):
        """Train and val index ranges do not overlap."""
        train_range = set(range(*default_split.indices["train"]))
        val_range = set(range(*default_split.indices["val"]))
        assert train_range.isdisjoint(val_range)

    def test_no_overlap_val_test(self, default_split):
        """Val and test index ranges do not overlap."""
        val_range = set(range(*default_split.indices["val"]))
        test_range = set(range(*default_split.indices["test"]))
        assert val_range.isdisjoint(test_range)

    def test_no_overlap_train_test(self, default_split):
        """Train and test index ranges do not overlap."""
        train_range = set(range(*default_split.indices["train"]))
        test_range = set(range(*default_split.indices["test"]))
        assert train_range.isdisjoint(test_range)

    def test_all_indices_covered(self, default_split):
        """Union of all splits covers [0, N)."""
        all_indices = set()
        for start, end in default_split.indices.values():
            all_indices.update(range(start, end))
        assert all_indices == set(range(100))


# ---------------------------------------------------------------------------
# create_splits — determinism
# ---------------------------------------------------------------------------


class TestSplitDeterminism:
    """Same inputs always produce identical splits."""

    def test_repeated_calls_identical(self):
        """Two calls with same args return same result."""
        s1 = create_splits(100, (0.70, 0.15, 0.15), window_size=7)
        s2 = create_splits(100, (0.70, 0.15, 0.15), window_size=7)
        assert s1.indices == s2.indices
        assert s1.counts == s2.counts

    def test_different_n_gives_different_splits(self):
        """Different n_samples produces different boundaries."""
        s1 = create_splits(100, (0.70, 0.15, 0.15), window_size=7)
        s2 = create_splits(200, (0.70, 0.15, 0.15), window_size=7)
        assert s1.indices != s2.indices


# ---------------------------------------------------------------------------
# create_splits — date ranges
# ---------------------------------------------------------------------------


class TestSplitDateRanges:
    """Verify date range mapping."""

    def test_date_ranges_present(self, default_split):
        """Date ranges populated for all splits."""
        for name in ("train", "val", "test"):
            assert len(default_split.date_ranges[name]) == 2
            assert default_split.date_ranges[name][0] != ""
            assert default_split.date_ranges[name][1] != ""

    def test_date_ranges_chronological(self, default_split):
        """Train dates < val dates < test dates."""
        train_last = default_split.date_ranges["train"][1]
        val_first = default_split.date_ranges["val"][0]
        test_first = default_split.date_ranges["test"][0]
        assert train_last <= val_first
        assert val_first <= test_first

    def test_no_dates_gives_empty_strings(self):
        """When dates=None, date_ranges are empty strings."""
        info = create_splits(100, (0.70, 0.15, 0.15), window_size=7)
        for name in ("train", "val", "test"):
            assert info.date_ranges[name] == ["", ""]


# ---------------------------------------------------------------------------
# create_splits — error cases
# ---------------------------------------------------------------------------


class TestCreateSplitsErrors:
    """Verify proper errors for invalid inputs."""

    def test_ratios_dont_sum_to_one(self):
        """Ratios summing to != 1.0 raise ConfigError."""
        with pytest.raises(ConfigError, match="must sum to 1.0"):
            create_splits(100, (0.50, 0.20, 0.20), window_size=7)

    def test_negative_ratio(self):
        """Negative ratio raises ConfigError."""
        with pytest.raises(ConfigError, match="must be > 0"):
            create_splits(100, (0.80, -0.10, 0.30), window_size=7)

    def test_zero_ratio(self):
        """Zero ratio raises ConfigError."""
        with pytest.raises(ConfigError, match="must be > 0"):
            create_splits(100, (0.70, 0.0, 0.30), window_size=7)

    def test_too_few_samples(self):
        """2 samples with 3 splits -> empty split -> ConfigError."""
        with pytest.raises(ConfigError, match="empty split"):
            create_splits(2, (0.70, 0.15, 0.15), window_size=7)

    def test_minimum_viable(self):
        """10 samples with 40/30/30 works (4/3/3)."""
        info = create_splits(10, (0.40, 0.30, 0.30), window_size=7)
        assert info.counts["train"] >= 1
        assert info.counts["val"] >= 1
        assert info.counts["test"] >= 1
        assert sum(info.counts.values()) == 10


# ---------------------------------------------------------------------------
# save_splits / load_splits — round-trip
# ---------------------------------------------------------------------------


class TestSplitPersistence:
    """Verify JSON save/load round-trip."""

    def test_save_creates_file(self, default_split, tmp_path):
        """save_splits creates splits.json."""
        path = save_splits(default_split, str(tmp_path))
        assert path.exists()
        assert path.name == "splits.json"

    def test_round_trip(self, default_split, tmp_path):
        """Save then load produces identical SplitInfo."""
        save_splits(default_split, str(tmp_path))
        loaded = load_splits(str(tmp_path / "splits.json"))

        assert loaded.window_size == default_split.window_size
        assert loaded.ratios == default_split.ratios
        assert loaded.indices == default_split.indices
        assert loaded.date_ranges == default_split.date_ranges
        assert loaded.counts == default_split.counts

    def test_json_schema(self, default_split, tmp_path):
        """JSON file contains all required fields per architecture 4.3."""
        save_splits(default_split, str(tmp_path))
        with open(tmp_path / "splits.json") as f:
            data = json.load(f)

        assert "window_size" in data
        assert "ratios" in data
        assert "indices" in data
        assert "date_ranges" in data
        assert "counts" in data

        # Indices are half-open ranges [start, end)
        for name in ("train", "val", "test"):
            assert len(data["indices"][name]) == 2
            assert data["indices"][name][0] < data["indices"][name][1]

    def test_load_nonexistent_raises(self, tmp_path):
        """Loading from missing file raises ConfigError."""
        with pytest.raises(ConfigError, match="not found"):
            load_splits(str(tmp_path / "nope.json"))

    def test_creates_output_dir(self, default_split, tmp_path):
        """save_splits creates the output directory if needed."""
        deep = str(tmp_path / "a" / "b")
        save_splits(default_split, deep)
        assert (tmp_path / "a" / "b" / "splits.json").exists()


# ---------------------------------------------------------------------------
# CA5Dataset
# ---------------------------------------------------------------------------


class TestCA5Dataset:
    """Verify PyTorch Dataset wrapper."""

    def test_length(self, sample_tensors):
        """Dataset length matches tensor length."""
        X, y = sample_tensors
        ds = CA5Dataset(X, y)
        assert len(ds) == 100

    def test_getitem_shapes(self, sample_tensors):
        """__getitem__ returns (W, 39) and (39,) tensors."""
        X, y = sample_tensors
        ds = CA5Dataset(X, y)
        xi, yi = ds[0]
        assert xi.shape == (7, 39)
        assert yi.shape == (39,)

    def test_getitem_values(self, sample_tensors):
        """__getitem__ returns the correct slice."""
        X, y = sample_tensors
        ds = CA5Dataset(X, y)
        xi, yi = ds[5]
        assert torch.equal(xi, X[5])
        assert torch.equal(yi, y[5])

    def test_sliced_dataset(self, sample_tensors):
        """Dataset works on a tensor slice (as used by get_dataloaders)."""
        X, y = sample_tensors
        ds = CA5Dataset(X[10:20], y[10:20])
        assert len(ds) == 10
        xi, yi = ds[0]
        assert torch.equal(xi, X[10])
        assert torch.equal(yi, y[10])


# ---------------------------------------------------------------------------
# get_dataloaders
# ---------------------------------------------------------------------------


class TestGetDataloaders:
    """Verify DataLoader factory."""

    def test_returns_three_loaders(self, default_split, sample_tensors):
        """Returns dict with train, val, test keys."""
        X, y = sample_tensors
        loaders = get_dataloaders(default_split, X, y, batch_size=16)
        assert set(loaders.keys()) == {"train", "val", "test"}

    def test_loader_sizes(self, default_split, sample_tensors):
        """Each loader has the correct number of samples."""
        X, y = sample_tensors
        loaders = get_dataloaders(default_split, X, y, batch_size=16)

        train_samples = sum(xb.shape[0] for xb, _ in loaders["train"])
        val_samples = sum(xb.shape[0] for xb, _ in loaders["val"])
        test_samples = sum(xb.shape[0] for xb, _ in loaders["test"])

        assert train_samples == 70
        assert val_samples == 15
        assert test_samples == 15

    def test_no_shuffle(self, default_split, sample_tensors):
        """DataLoaders preserve order across multiple iterations."""
        X, y = sample_tensors
        loaders = get_dataloaders(default_split, X, y, batch_size=16)

        # Iterate train loader twice — should get same order
        first_run = torch.cat([xb for xb, _ in loaders["train"]])
        second_run = torch.cat([xb for xb, _ in loaders["train"]])
        assert torch.equal(first_run, second_run)

    def test_batch_shapes(self, default_split, sample_tensors):
        """Batch tensors have correct shapes."""
        X, y = sample_tensors
        loaders = get_dataloaders(default_split, X, y, batch_size=16)

        xb, yb = next(iter(loaders["train"]))
        assert xb.shape == (16, 7, 39)
        assert yb.shape == (16, 39)

    def test_data_matches_original(self, default_split, sample_tensors):
        """Loader data matches the corresponding slice of the original tensors."""
        X, y = sample_tensors
        loaders = get_dataloaders(default_split, X, y, batch_size=100)

        # Train loader should contain X[0:70]
        train_x, train_y = next(iter(loaders["train"]))
        assert torch.equal(train_x, X[0:70])
        assert torch.equal(train_y, y[0:70])

    def test_large_batch_size(self, default_split, sample_tensors):
        """Batch size larger than split works (single batch)."""
        X, y = sample_tensors
        loaders = get_dataloaders(default_split, X, y, batch_size=1000)

        batches = list(loaders["val"])
        assert len(batches) == 1
        assert batches[0][0].shape[0] == 15
