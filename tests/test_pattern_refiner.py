"""Tests for the pattern refiner module."""

import numpy as np
import pandas as pd

from c5_snn.lln_pattern.pattern_refiner import (
    compute_pmi_scores,
    compute_repeat_scores,
    compute_transition_scores,
    refine_scores,
)


def _make_date_df(n: int = 100) -> pd.DataFrame:
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


class TestComputeRepeatScores:
    def test_shape(self):
        df = _make_date_df(50)
        scores = compute_repeat_scores(df, target_idx=40)
        assert scores.shape == (39,)

    def test_nonnegative(self):
        df = _make_date_df(50)
        scores = compute_repeat_scores(df, target_idx=40)
        assert (scores >= 0).all()

    def test_boosted_for_repeat(self):
        """A value repeated in the same position should get a boost."""
        rows = []
        for i in range(10):
            rows.append(
                {
                    "date": f"1/{i + 1}/2020",
                    "m_1": 5,
                    "m_2": 10,
                    "m_3": 20,
                    "m_4": 25,
                    "m_5": 30,
                }
            )
        df = pd.DataFrame(rows)
        scores = compute_repeat_scores(df, target_idx=10, lookback=5)
        assert scores[4] > 0  # value 5 at index 4

    def test_too_little_data(self):
        df = _make_date_df(5)
        scores = compute_repeat_scores(df, target_idx=1)
        assert scores.shape == (39,)


class TestComputeTransitionScores:
    def test_shape(self):
        df = _make_date_df(50)
        scores = compute_transition_scores(df, target_idx=40)
        assert scores.shape == (39,)

    def test_nonnegative(self):
        df = _make_date_df(50)
        scores = compute_transition_scores(df, target_idx=40)
        assert (scores >= 0).all()


class TestComputePMIScores:
    def test_shape(self):
        df = _make_date_df(50)
        scores = compute_pmi_scores(df, target_idx=40)
        assert scores.shape == (39,)

    def test_empty_pairs_returns_zeros(self):
        df = _make_date_df(50)
        scores = compute_pmi_scores(df, target_idx=40, pmi_pairs=[])
        assert (scores == 0).all()


class TestRefineScores:
    def test_shape(self):
        df = _make_date_df(50)
        lln_scores = np.ones(39)
        result = refine_scores(lln_scores, df, target_idx=40)
        assert result.shape == (39,)

    def test_zero_weights_equals_lln(self):
        df = _make_date_df(50)
        lln_scores = np.random.RandomState(42).rand(39)
        result = refine_scores(
            lln_scores, df, target_idx=40,
            repeat_weight=0.0,
            transition_weight=0.0,
            pmi_weight=0.0,
        )
        np.testing.assert_array_equal(result, lln_scores)

    def test_refinement_changes_scores(self):
        df = _make_date_df(50)
        lln_scores = np.ones(39)
        result = refine_scores(lln_scores, df, target_idx=40)
        # At least some scores should change from the base
        assert not np.array_equal(result, lln_scores)
