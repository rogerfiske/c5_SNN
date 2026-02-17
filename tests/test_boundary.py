"""Tests for the boundary penalty module."""

import numpy as np
import pytest

from c5_snn.lln_pattern.boundary import apply_boundary_penalty
from c5_snn.lln_pattern.constants import BOUNDARY_HIGH_RANGE, BOUNDARY_LOW_RANGE


class TestApplyBoundaryPenalty:
    def test_shape(self):
        scores = np.ones(39)
        result = apply_boundary_penalty(scores)
        assert result.shape == (39,)

    def test_boundary_values_boosted(self):
        scores = np.ones(39)
        result = apply_boundary_penalty(scores, penalty=1.7)
        for v in BOUNDARY_LOW_RANGE:
            assert result[v - 1] == pytest.approx(1.7)
        for v in BOUNDARY_HIGH_RANGE:
            assert result[v - 1] == pytest.approx(1.7)

    def test_non_boundary_unchanged(self):
        scores = np.ones(39) * 2.0
        result = apply_boundary_penalty(scores, penalty=1.7)
        for v in range(8, 33):  # values 8-32
            assert result[v - 1] == pytest.approx(2.0)

    def test_penalty_1_is_noop(self):
        scores = np.random.RandomState(42).rand(39)
        result = apply_boundary_penalty(scores, penalty=1.0)
        np.testing.assert_array_equal(result, scores)

    def test_does_not_mutate_input(self):
        scores = np.ones(39)
        original = scores.copy()
        apply_boundary_penalty(scores, penalty=2.0)
        np.testing.assert_array_equal(scores, original)

    def test_custom_ranges(self):
        scores = np.ones(39)
        result = apply_boundary_penalty(
            scores,
            penalty=2.0,
            low_range=range(1, 4),  # values 1-3 only
            high_range=range(37, 40),  # values 37-39 only
        )
        assert result[0] == pytest.approx(2.0)  # value 1
        assert result[2] == pytest.approx(2.0)  # value 3
        assert result[3] == pytest.approx(1.0)  # value 4 (not in range)
        assert result[36] == pytest.approx(2.0)  # value 37
