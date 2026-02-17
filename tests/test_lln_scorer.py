"""Tests for the LLN scorer module."""

import numpy as np
import pandas as pd

from c5_snn.lln_pattern.lln_scorer import (
    compute_lln_scores,
    compute_position_frequencies,
    select_exclusion_set,
)


def _make_date_df(n: int = 100) -> pd.DataFrame:
    """Create synthetic CA5_date DataFrame."""
    rng = np.random.RandomState(42)
    rows = []
    for i in range(n):
        parts = sorted(rng.choice(range(1, 40), size=5, replace=False))
        rows.append(
            {
                "date": f"1/{(i % 28) + 1}/2020",
                "m_1": parts[0],
                "m_2": parts[1],
                "m_3": parts[2],
                "m_4": parts[3],
                "m_5": parts[4],
            }
        )
    return pd.DataFrame(rows)


class TestComputePositionFrequencies:
    def test_shape(self):
        values = np.array([1, 5, 10, 15, 20])
        result = compute_position_frequencies(values, recent_n=5)
        assert result.shape == (39,)

    def test_nonnegative(self):
        values = np.array([1, 5, 10, 15, 20])
        result = compute_position_frequencies(values, recent_n=5)
        assert (result >= 0).all()

    def test_recent_value_high(self):
        # Value 5 appears 3 times in last 5 events
        values = np.array([5, 5, 5, 10, 20])
        result = compute_position_frequencies(values, recent_n=5)
        assert result[4] > result[9]  # 5 (idx 4) > 10 (idx 9)

    def test_decay_weighting(self):
        # Same value twice: once at start, once at end of window
        # Newest should contribute more weight
        values = np.array([7, 1, 1, 1, 7])
        result = compute_position_frequencies(values, recent_n=5, decay=0.5)
        # Value 7 at position 0 (oldest) and 4 (newest)
        # Weight of newest = 0.5^0 = 1.0
        # Weight of oldest = 0.5^4 = 0.0625
        # Total for 7 = 1.0625
        assert result[6] > 0  # value 7 is at index 6

    def test_empty_returns_zeros(self):
        result = compute_position_frequencies(np.array([]), recent_n=5)
        assert (result == 0).all()

    def test_zero_recent_n_returns_zeros(self):
        result = compute_position_frequencies(np.array([1, 2, 3]), recent_n=0)
        assert (result == 0).all()


class TestComputeLLNScores:
    def test_shape(self):
        df = _make_date_df(100)
        scores = compute_lln_scores(df, target_idx=80)
        assert scores.shape == (39,)

    def test_no_leakage(self):
        """Scores should only depend on data before target_idx."""
        df = _make_date_df(100)
        # Modify data after target to verify it doesn't affect scores
        df2 = df.copy()
        df2.loc[80:, "m_1"] = 1
        scores1 = compute_lln_scores(df, target_idx=80)
        scores2 = compute_lln_scores(df2, target_idx=80)
        np.testing.assert_array_equal(scores1, scores2)

    def test_different_targets(self):
        df = _make_date_df(100)
        s1 = compute_lln_scores(df, target_idx=50)
        s2 = compute_lln_scores(df, target_idx=90)
        assert not np.array_equal(s1, s2)

    def test_custom_params(self):
        df = _make_date_df(100)
        custom = {
            p: {"recent_fraction": 0.5, "percentile_threshold": 50.0}
            for p in range(1, 6)
        }
        scores = compute_lln_scores(
            df, target_idx=80, position_params=custom
        )
        assert scores.shape == (39,)


class TestSelectExclusionSet:
    def test_correct_size(self):
        scores = np.random.RandomState(42).rand(39)
        result = select_exclusion_set(scores, k=20)
        assert len(result) == 20

    def test_lowest_scores_excluded(self):
        scores = np.arange(1.0, 40.0)  # 1..39
        result = select_exclusion_set(scores, k=20)
        # Lowest scores at indices 0..19 -> values 1..20
        np.testing.assert_array_equal(result, np.arange(1, 21))

    def test_sorted_output(self):
        scores = np.random.RandomState(42).rand(39)
        result = select_exclusion_set(scores, k=20)
        assert (np.diff(result) >= 0).all()

    def test_values_in_range(self):
        scores = np.random.RandomState(42).rand(39)
        result = select_exclusion_set(scores, k=20)
        assert result.min() >= 1
        assert result.max() <= 39

    def test_no_duplicates(self):
        scores = np.random.RandomState(42).rand(39)
        result = select_exclusion_set(scores, k=20)
        assert len(set(result)) == len(result)

    def test_k_larger_than_values(self):
        scores = np.random.RandomState(42).rand(39)
        result = select_exclusion_set(scores, k=50)
        assert len(result) == 39
