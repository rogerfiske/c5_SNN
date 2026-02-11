"""Tests for BaseModel, MODEL_REGISTRY, and FrequencyBaseline (STORY-3.1)."""

import pytest
import torch

from c5_snn.models.base import MODEL_REGISTRY, BaseModel, get_model
from c5_snn.models.baselines import FrequencyBaseline
from c5_snn.utils.exceptions import ConfigError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {"model": {"type": "frequency_baseline"}}


@pytest.fixture
def model():
    """FrequencyBaseline with default config."""
    return FrequencyBaseline(DEFAULT_CONFIG)


# ---------------------------------------------------------------------------
# BaseModel ABC
# ---------------------------------------------------------------------------


class TestBaseModel:
    """Verify BaseModel abstract base class."""

    def test_cannot_instantiate(self):
        """BaseModel is abstract — cannot instantiate directly."""
        with pytest.raises(TypeError, match="abstract"):
            BaseModel()

    def test_subclass_must_implement_forward(self):
        """Subclass without forward() raises TypeError."""

        class BadModel(BaseModel):
            pass

        with pytest.raises(TypeError, match="abstract"):
            BadModel()

    def test_subclass_with_forward_works(self):
        """Subclass with forward() can be instantiated."""

        class GoodModel(BaseModel):
            def forward(self, x):
                return x[:, -1, :]

        m = GoodModel()
        assert isinstance(m, BaseModel)


# ---------------------------------------------------------------------------
# MODEL_REGISTRY and get_model
# ---------------------------------------------------------------------------


class TestModelRegistry:
    """Verify model registry and factory function."""

    def test_frequency_baseline_registered(self):
        """FrequencyBaseline is in MODEL_REGISTRY."""
        assert "frequency_baseline" in MODEL_REGISTRY
        assert MODEL_REGISTRY["frequency_baseline"] is FrequencyBaseline

    def test_get_model_returns_correct_type(self):
        """get_model returns FrequencyBaseline for correct config."""
        m = get_model(DEFAULT_CONFIG)
        assert isinstance(m, FrequencyBaseline)
        assert isinstance(m, BaseModel)

    def test_get_model_unknown_type_raises(self):
        """get_model raises ConfigError for unknown model type."""
        with pytest.raises(ConfigError, match="Unknown model type"):
            get_model({"model": {"type": "nonexistent_model"}})

    def test_get_model_missing_type_raises(self):
        """get_model raises ConfigError when model.type is missing."""
        with pytest.raises(ConfigError, match="Unknown model type"):
            get_model({"model": {}})

    def test_get_model_empty_config_raises(self):
        """get_model raises ConfigError for empty config."""
        with pytest.raises(ConfigError, match="Unknown model type"):
            get_model({})


# ---------------------------------------------------------------------------
# FrequencyBaseline — output shape
# ---------------------------------------------------------------------------


class TestFrequencyBaselineShape:
    """Verify output shapes for various inputs."""

    def test_basic_shape(self, model):
        """Standard input (4, 21, 39) -> (4, 39)."""
        x = torch.rand(4, 21, 39)
        out = model(x)
        assert out.shape == (4, 39)

    def test_single_sample(self, model):
        """Batch size 1."""
        x = torch.rand(1, 21, 39)
        out = model(x)
        assert out.shape == (1, 39)

    def test_window_size_3(self, model):
        """Small window W=3."""
        x = torch.rand(8, 3, 39)
        out = model(x)
        assert out.shape == (8, 39)

    def test_window_size_1(self, model):
        """Minimal window W=1."""
        x = torch.rand(2, 1, 39)
        out = model(x)
        assert out.shape == (2, 39)

    def test_large_batch(self, model):
        """Large batch size."""
        x = torch.rand(128, 7, 39)
        out = model(x)
        assert out.shape == (128, 39)


# ---------------------------------------------------------------------------
# FrequencyBaseline — scoring correctness
# ---------------------------------------------------------------------------


class TestFrequencyBaselineScoring:
    """Verify scoring with hand-computed values."""

    def test_dominant_part_ranked_highest(self, model):
        """Part active in all steps should have highest score."""
        x = torch.zeros(1, 3, 39)
        # Part 5 active in all 3 steps
        x[0, :, 5] = 1.0
        out = model(x)
        # Part 5 should have the highest score
        assert out[0, 5] == out[0].max()
        # All other parts should be 0
        mask = torch.ones(39, dtype=torch.bool)
        mask[5] = False
        assert (out[0, mask] == 0.0).all()

    def test_dominant_part_hand_computed(self):
        """Verify exact score for dominant part (decay=0.9, W=3).

        freq[5] = 3
        recency[5] = 0.9^2 + 0.9^1 + 0.9^0 = 0.81 + 0.9 + 1.0 = 2.71
        score = 1.0 * 3 + 1.0 * 2.71 = 5.71
        """
        config = {
            "model": {
                "type": "frequency_baseline",
                "freq_weight": 1.0,
                "recency_weight": 1.0,
                "decay": 0.9,
            }
        }
        m = FrequencyBaseline(config)
        x = torch.zeros(1, 3, 39)
        x[0, :, 5] = 1.0
        out = m(x)
        assert out[0, 5].item() == pytest.approx(5.71, abs=1e-4)

    def test_recency_recent_beats_old(self, model):
        """Part active only at newest step ranks higher than oldest.

        Part A (index 0): active at t=0 only (oldest)
        Part B (index 1): active at t=W-1 only (newest)
        Both freq=1, but B has higher recency -> B > A.
        """
        W = 5
        x = torch.zeros(1, W, 39)
        x[0, 0, 0] = 1.0    # Part 0 at oldest step
        x[0, W - 1, 1] = 1.0  # Part 1 at newest step
        out = model(x)
        assert out[0, 1] > out[0, 0]

    def test_recency_values_hand_computed(self):
        """Verify recency values for old vs new (decay=0.9, W=5).

        Part A at t=0: recency = 0.9^4 = 0.6561
        Part B at t=4: recency = 0.9^0 = 1.0
        score_A = 1.0 * 1 + 1.0 * 0.6561 = 1.6561
        score_B = 1.0 * 1 + 1.0 * 1.0 = 2.0
        """
        config = {
            "model": {
                "type": "frequency_baseline",
                "freq_weight": 1.0,
                "recency_weight": 1.0,
                "decay": 0.9,
            }
        }
        m = FrequencyBaseline(config)
        x = torch.zeros(1, 5, 39)
        x[0, 0, 0] = 1.0  # Part 0 at oldest
        x[0, 4, 1] = 1.0  # Part 1 at newest
        out = m(x)
        assert out[0, 0].item() == pytest.approx(1.6561, abs=1e-4)
        assert out[0, 1].item() == pytest.approx(2.0, abs=1e-4)

    def test_all_inactive_all_zero(self, model):
        """All-zero input produces all-zero output."""
        x = torch.zeros(2, 7, 39)
        out = model(x)
        assert (out == 0.0).all()

    def test_frequency_only_weight(self):
        """With recency_weight=0, only frequency matters."""
        config = {
            "model": {
                "type": "frequency_baseline",
                "freq_weight": 1.0,
                "recency_weight": 0.0,
                "decay": 0.9,
            }
        }
        m = FrequencyBaseline(config)
        x = torch.zeros(1, 5, 39)
        # Part 0 active 3 times, Part 1 active 1 time
        x[0, 0, 0] = 1.0
        x[0, 2, 0] = 1.0
        x[0, 4, 0] = 1.0
        x[0, 4, 1] = 1.0
        out = m(x)
        assert out[0, 0].item() == pytest.approx(3.0)
        assert out[0, 1].item() == pytest.approx(1.0)

    def test_batch_independence(self, model):
        """Each sample in batch scored independently."""
        x = torch.zeros(2, 3, 39)
        # Sample 0: part 10 dominant
        x[0, :, 10] = 1.0
        # Sample 1: part 20 dominant
        x[1, :, 20] = 1.0

        out = model(x)
        assert out[0, 10] == out[0].max()
        assert out[1, 20] == out[1].max()
        # Cross-check: sample 0 has 0 for part 20, sample 1 has 0 for 10
        assert out[0, 20] == 0.0
        assert out[1, 10] == 0.0


# ---------------------------------------------------------------------------
# FrequencyBaseline — properties
# ---------------------------------------------------------------------------


class TestFrequencyBaselineProperties:
    """Verify model properties."""

    def test_no_learnable_parameters(self, model):
        """Model has no learnable parameters."""
        params = list(model.parameters())
        assert len(params) == 0

    def test_is_base_model(self, model):
        """FrequencyBaseline is a BaseModel."""
        assert isinstance(model, BaseModel)

    def test_is_nn_module(self, model):
        """FrequencyBaseline is an nn.Module."""
        assert isinstance(model, torch.nn.Module)

    def test_default_hyperparameters(self):
        """Default config uses expected hyperparameters."""
        m = FrequencyBaseline({"model": {}})
        assert m.freq_weight == 1.0
        assert m.recency_weight == 1.0
        assert m.decay == 0.9

    def test_custom_hyperparameters(self):
        """Custom config overrides defaults."""
        config = {
            "model": {
                "freq_weight": 2.0,
                "recency_weight": 0.5,
                "decay": 0.8,
            }
        }
        m = FrequencyBaseline(config)
        assert m.freq_weight == 2.0
        assert m.recency_weight == 0.5
        assert m.decay == 0.8

    def test_eval_mode_works(self, model):
        """Model can be set to eval mode."""
        model.eval()
        x = torch.rand(2, 7, 39)
        out = model(x)
        assert out.shape == (2, 39)
