"""Tests for LLN-Pattern N-wrong metrics."""

import numpy as np
import pytest

from c5_snn.lln_pattern.metrics import (
    compute_n_wrong,
    compute_n_wrong_distribution,
)


class TestComputeNWrong:
    def test_zero_overlap(self):
        actual = np.array([1, 2, 3, 4, 5])
        excluded = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                             16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
        assert compute_n_wrong(actual, excluded) == 0

    def test_full_overlap(self):
        actual = np.array([1, 2, 3, 4, 5])
        excluded = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                             11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        assert compute_n_wrong(actual, excluded) == 5

    def test_partial_overlap(self):
        actual = np.array([1, 2, 3, 4, 5])
        excluded = np.array([3, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                             14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
        assert compute_n_wrong(actual, excluded) == 2

    def test_single_overlap(self):
        actual = np.array([10, 20, 30, 35, 39])
        excluded = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                             11, 12, 13, 14, 15, 16, 17, 18, 19, 21])
        assert compute_n_wrong(actual, excluded) == 1


class TestComputeNWrongDistribution:
    def test_all_zero_wrong(self):
        result = compute_n_wrong_distribution([0, 0, 0, 0, 0])
        assert result["total"] == 5
        assert result["zero_wrong_count"] == 5
        assert result["zero_wrong_pct"] == 100.0
        assert result["mean_wrong"] == 0.0

    def test_mixed_distribution(self):
        data = [0, 0, 1, 1, 2, 3]
        result = compute_n_wrong_distribution(data)
        assert result["total"] == 6
        assert result["zero_wrong_count"] == 2
        assert result["zero_wrong_pct"] == pytest.approx(33.33, abs=0.01)
        assert result["mean_wrong"] == pytest.approx(
            np.mean(data), abs=0.0001
        )
        assert result["distribution"] == {0: 2, 1: 2, 2: 1, 3: 1}

    def test_empty_list(self):
        result = compute_n_wrong_distribution([])
        assert result["total"] == 0
        assert result["zero_wrong_count"] == 0
        assert result["zero_wrong_pct"] == 0.0

    def test_distribution_sums_to_total(self):
        data = [0, 1, 2, 0, 1, 3, 0, 2, 1, 0]
        result = compute_n_wrong_distribution(data)
        dist_sum = sum(result["distribution"].values())
        assert dist_sum == result["total"]
