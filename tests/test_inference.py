"""Unit tests for c5_snn.inference module."""

import numpy as np
import pytest
import torch

from c5_snn.inference import (
    build_prediction_window,
    calendar_enhanced_predict,
    calendar_enhanced_score,
    format_top_k_prediction,
    load_model_from_checkpoint,
    run_calendar_holdout_test,
    run_holdout_test,
)
from tests.conftest import _make_valid_df

# ---------------------------------------------------------------------------
# build_prediction_window
# ---------------------------------------------------------------------------


class TestBuildPredictionWindow:
    def test_shape_default(self):
        df = _make_valid_df(30)
        x = build_prediction_window(df, window_size=7)
        assert x.shape == (1, 7, 39)

    def test_shape_larger_window(self):
        df = _make_valid_df(30)
        x = build_prediction_window(df, window_size=21)
        assert x.shape == (1, 21, 39)

    def test_dtype(self):
        df = _make_valid_df(30)
        x = build_prediction_window(df, window_size=7)
        assert x.dtype == torch.float32

    def test_offset_skips_last_rows(self):
        df = _make_valid_df(30)
        x0 = build_prediction_window(df, window_size=7, offset=0)
        x1 = build_prediction_window(df, window_size=7, offset=1)
        # offset=1 should skip the last row, giving a different window
        assert not torch.equal(x0, x1)

    def test_offset_1_uses_penultimate(self):
        df = _make_valid_df(30)
        x = build_prediction_window(df, window_size=7, offset=1)
        # The window should end at df.iloc[-2], not df.iloc[-1]
        from c5_snn.data.validation import P_COLUMNS

        expected = df[P_COLUMNS].iloc[-8:-1].values
        expected_tensor = torch.tensor(expected, dtype=torch.float32)
        assert torch.equal(x.squeeze(0), expected_tensor)

    def test_too_short_raises(self):
        df = _make_valid_df(10)
        with pytest.raises(ValueError, match="Not enough data"):
            build_prediction_window(df, window_size=11)

    def test_too_short_with_offset_raises(self):
        df = _make_valid_df(10)
        with pytest.raises(ValueError, match="Not enough data"):
            build_prediction_window(df, window_size=7, offset=4)

    def test_values_match_last_rows(self):
        df = _make_valid_df(20)
        from c5_snn.data.validation import P_COLUMNS

        x = build_prediction_window(df, window_size=7)
        expected = df[P_COLUMNS].iloc[-7:].values
        expected_tensor = torch.tensor(expected, dtype=torch.float32)
        assert torch.equal(x.squeeze(0), expected_tensor)


# ---------------------------------------------------------------------------
# format_top_k_prediction
# ---------------------------------------------------------------------------


class TestFormatTopK:
    def test_length(self):
        logits = torch.randn(1, 39)
        result = format_top_k_prediction(logits, k=20)
        assert len(result) == 20

    def test_length_k5(self):
        logits = torch.randn(1, 39)
        result = format_top_k_prediction(logits, k=5)
        assert len(result) == 5

    def test_ordering_descending(self):
        logits = torch.randn(1, 39)
        result = format_top_k_prediction(logits, k=20)
        scores = [p["score"] for p in result]
        assert scores == sorted(scores, reverse=True)

    def test_rank_values(self):
        logits = torch.randn(1, 39)
        result = format_top_k_prediction(logits, k=10)
        ranks = [p["rank"] for p in result]
        assert ranks == list(range(1, 11))

    def test_part_names_1_indexed(self):
        logits = torch.randn(1, 39)
        result = format_top_k_prediction(logits, k=39)
        part_numbers = sorted(p["part_number"] for p in result)
        assert part_numbers == list(range(1, 40))

    def test_part_name_format(self):
        logits = torch.randn(1, 39)
        result = format_top_k_prediction(logits, k=5)
        for p in result:
            assert p["part"].startswith("P_")
            assert p["part"] == f"P_{p['part_number']}"

    def test_known_input(self):
        """When logits are all zeros except one, that one should be rank 1."""
        logits = torch.zeros(39)
        logits[13] = 10.0  # P_14 should be #1
        result = format_top_k_prediction(logits, k=5)
        assert result[0]["part"] == "P_14"
        assert result[0]["score"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# load_model_from_checkpoint
# ---------------------------------------------------------------------------


class TestLoadModelFromCheckpoint:
    def test_roundtrip(self, tmp_path):
        """Save a GRU checkpoint and reload it."""
        from c5_snn.models.base import get_model

        config = {
            "model": {
                "type": "gru_baseline",
                "hidden_size": 32,
                "num_layers": 1,
                "dropout": 0.0,
            },
            "data": {"window_size": 7},
            "experiment": {"seed": 42},
        }
        model = get_model(config)
        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "epoch": 5,
            "best_val_recall_at_20": 0.5,
            "config": config,
            "seed": 42,
        }
        path = tmp_path / "test_ckpt.pt"
        torch.save(ckpt, path)

        loaded_model, loaded_config = load_model_from_checkpoint(
            path, torch.device("cpu")
        )
        assert loaded_config["model"]["type"] == "gru_baseline"
        assert not loaded_model.training  # eval mode

    def test_forward_matches(self, tmp_path):
        """Loaded model produces same output as original."""
        from c5_snn.models.base import get_model

        config = {
            "model": {
                "type": "gru_baseline",
                "hidden_size": 32,
                "num_layers": 1,
                "dropout": 0.0,
            },
            "data": {"window_size": 7},
            "experiment": {"seed": 42},
        }
        model = get_model(config)
        model.eval()
        x = torch.randn(2, 7, 39)

        with torch.no_grad():
            original_out = model(x)

        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "epoch": 5,
            "best_val_recall_at_20": 0.5,
            "config": config,
            "seed": 42,
        }
        path = tmp_path / "test_ckpt.pt"
        torch.save(ckpt, path)

        loaded_model, _ = load_model_from_checkpoint(
            path, torch.device("cpu")
        )
        with torch.no_grad():
            loaded_out = loaded_model(x)

        assert torch.allclose(original_out, loaded_out, atol=1e-6)

    def test_missing_keys_raises(self, tmp_path):
        path = tmp_path / "bad_ckpt.pt"
        torch.save({"foo": "bar"}, path)
        with pytest.raises(ValueError, match="missing"):
            load_model_from_checkpoint(path, torch.device("cpu"))


# ---------------------------------------------------------------------------
# run_holdout_test
# ---------------------------------------------------------------------------


class TestRunHoldoutTest:
    def _make_model(self):
        from c5_snn.models.base import get_model

        config = {"model": {"type": "frequency_baseline"}}
        model = get_model(config)
        model.eval()
        return model

    def test_single_holdout_keys(self):
        df = _make_valid_df(30)
        model = self._make_model()
        result = run_holdout_test(
            model, df, window_size=7, n_holdout=1, device=torch.device("cpu")
        )
        assert "metrics" in result
        assert "per_sample" in result
        assert len(result["per_sample"]) == 1
        assert "recall_at_20" in result["metrics"]

    def test_multi_holdout(self):
        df = _make_valid_df(30)
        model = self._make_model()
        result = run_holdout_test(
            model, df, window_size=7, n_holdout=5, device=torch.device("cpu")
        )
        assert len(result["per_sample"]) == 5

    def test_per_sample_has_metrics(self):
        df = _make_valid_df(30)
        model = self._make_model()
        result = run_holdout_test(
            model, df, window_size=7, n_holdout=3, device=torch.device("cpu")
        )
        for s in result["per_sample"]:
            assert "recall_at_20" in s
            assert "hit_at_20" in s
            assert "mrr" in s
            assert "true_parts" in s
            assert "predicted_top20" in s
            assert "date" in s

    def test_not_enough_data_raises(self):
        df = _make_valid_df(10)
        model = self._make_model()
        with pytest.raises(ValueError, match="Need at least"):
            run_holdout_test(
                model,
                df,
                window_size=7,
                n_holdout=5,
                device=torch.device("cpu"),
            )

    def test_no_data_leakage(self):
        """The target row must NOT be in the prediction window."""
        df = _make_valid_df(30)
        model = self._make_model()
        result = run_holdout_test(
            model, df, window_size=7, n_holdout=1, device=torch.device("cpu")
        )
        # The holdout row is the last row (index 29)
        sample = result["per_sample"][0]
        assert sample["row_index"] == 29


# ---------------------------------------------------------------------------
# calendar_enhanced_score
# ---------------------------------------------------------------------------


class TestCalendarEnhancedScore:
    def test_returns_39_scores(self):
        df = _make_valid_df(30)
        scores = calendar_enhanced_score(df, target_idx=25)
        assert scores.shape == (39,)

    def test_scores_are_finite(self):
        df = _make_valid_df(30)
        scores = calendar_enhanced_score(df, target_idx=25)
        assert np.all(np.isfinite(scores))

    def test_core_parts_have_high_scores(self):
        """The top 8 'core' parts should have score 100."""
        df = _make_valid_df(30)
        scores = calendar_enhanced_score(df, target_idx=25)
        top8 = np.sort(scores)[::-1][:8]
        assert np.all(top8 == 100.0)

    def test_non_core_parts_have_low_scores(self):
        """Non-core parts should have scores << 100."""
        df = _make_valid_df(30)
        scores = calendar_enhanced_score(df, target_idx=25)
        non_core = np.sort(scores)[::-1][8:]
        assert np.all(non_core < 10.0)

    def test_config_override(self):
        """Custom config changes core_k."""
        df = _make_valid_df(30)
        scores = calendar_enhanced_score(
            df, target_idx=25, config={"core_k": 5}
        )
        n_core = np.sum(scores == 100.0)
        assert n_core == 5

    def test_different_targets_give_different_scores(self):
        df = _make_valid_df(30)
        s1 = calendar_enhanced_score(df, target_idx=20)
        s2 = calendar_enhanced_score(df, target_idx=25)
        assert not np.array_equal(s1, s2)


# ---------------------------------------------------------------------------
# calendar_enhanced_predict
# ---------------------------------------------------------------------------


class TestCalendarEnhancedPredict:
    def test_returns_tensor(self):
        df = _make_valid_df(30)
        logits = calendar_enhanced_predict(df)
        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (1, 39)

    def test_dtype_float32(self):
        df = _make_valid_df(30)
        logits = calendar_enhanced_predict(df)
        assert logits.dtype == torch.float32


# ---------------------------------------------------------------------------
# run_calendar_holdout_test
# ---------------------------------------------------------------------------


class TestRunCalendarHoldoutTest:
    def test_single_holdout_keys(self):
        df = _make_valid_df(30)
        result = run_calendar_holdout_test(
            df, window_size=7, n_holdout=1
        )
        assert "metrics" in result
        assert "per_sample" in result
        assert len(result["per_sample"]) == 1
        assert "recall_at_20" in result["metrics"]

    def test_multi_holdout(self):
        df = _make_valid_df(30)
        result = run_calendar_holdout_test(
            df, window_size=7, n_holdout=5
        )
        assert len(result["per_sample"]) == 5

    def test_per_sample_has_metrics(self):
        df = _make_valid_df(30)
        result = run_calendar_holdout_test(
            df, window_size=7, n_holdout=3
        )
        for s in result["per_sample"]:
            assert "recall_at_20" in s
            assert "hit_at_20" in s
            assert "mrr" in s
            assert "true_parts" in s
            assert "predicted_top20" in s
            assert "date" in s

    def test_not_enough_data_raises(self):
        df = _make_valid_df(10)
        with pytest.raises(ValueError, match="Need at least"):
            run_calendar_holdout_test(
                df, window_size=7, n_holdout=5
            )

    def test_no_data_leakage(self):
        """The target row must NOT be in the prediction window."""
        df = _make_valid_df(30)
        result = run_calendar_holdout_test(
            df, window_size=7, n_holdout=1
        )
        sample = result["per_sample"][0]
        assert sample["row_index"] == 29
