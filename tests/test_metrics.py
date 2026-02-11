"""Tests for evaluation metrics and harness (STORY-2.3)."""

import json

import pytest
import torch
from torch.utils.data import DataLoader

from c5_snn.training.evaluate import evaluate_model, export_results
from c5_snn.training.metrics import (
    compute_all_metrics,
    hit_at_k,
    mrr,
    recall_at_k,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_targets(*part_indices, n_classes=39):
    """Create a single multi-hot target row with 1s at given indices."""
    t = torch.zeros(n_classes)
    for idx in part_indices:
        t[idx] = 1.0
    return t


def _make_logits_ranking(*ordered_indices, n_classes=39):
    """Create logits where ordered_indices[0] gets highest score, etc.

    Positions not in ordered_indices get scores below all listed ones.
    """
    logits = torch.zeros(n_classes)
    n = len(ordered_indices)
    for rank, idx in enumerate(ordered_indices):
        logits[idx] = float(n - rank)
    return logits


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------


class TestRecallAtK:
    """Verify recall@K with hand-computed values."""

    def test_perfect_prediction_at_5(self):
        """All 5 true parts in top-5 -> recall@5 = 1.0."""
        targets = _make_targets(1, 5, 10, 20, 30).unsqueeze(0)
        logits = _make_logits_ranking(1, 5, 10, 20, 30).unsqueeze(0)
        assert recall_at_k(logits, targets, k=5) == pytest.approx(1.0)

    def test_zero_overlap_at_5(self):
        """No true parts in top-5 -> recall@5 = 0.0."""
        targets = _make_targets(0, 1, 2, 3, 4).unsqueeze(0)
        # Top-5 are indices 35-38 and 34
        logits = _make_logits_ranking(35, 36, 37, 38, 34).unsqueeze(0)
        assert recall_at_k(logits, targets, k=5) == pytest.approx(0.0)

    def test_partial_overlap_at_5(self):
        """2 of 5 true parts in top-5 -> recall@5 = 0.4."""
        targets = _make_targets(1, 5, 10, 20, 30).unsqueeze(0)
        # Top-5: 1, 5, 11, 21, 31 -> overlap = {1, 5} = 2/5 = 0.4
        logits = _make_logits_ranking(1, 5, 11, 21, 31).unsqueeze(0)
        assert recall_at_k(logits, targets, k=5) == pytest.approx(0.4)

    def test_recall_at_20_perfect(self):
        """All 5 true parts in top-20 -> recall@20 = 1.0."""
        targets = _make_targets(1, 5, 10, 20, 30).unsqueeze(0)
        # Top-20 must include all true parts: 1, 5, 10, 20, 30
        ranking = [1, 5, 10, 20, 30, 0, 2, 3, 4, 6,
                   7, 8, 9, 11, 12, 13, 14, 15, 16, 17]
        logits = _make_logits_ranking(*ranking).unsqueeze(0)
        assert recall_at_k(logits, targets, k=20) == pytest.approx(1.0)

    def test_recall_at_20_partial(self):
        """3 of 5 true parts in top-20 -> recall@20 = 0.6."""
        targets = _make_targets(0, 1, 2, 3, 4).unsqueeze(0)
        # Top-20 = indices 5-24 -> only 0,1,2,3,4 are true
        # None in 5-24 are true... let's be explicit
        # Top-20: indices 0, 1, 2, 10, 11, 12, ..., 19
        # True parts: 0, 1, 2, 3, 4 -> overlap = {0, 1, 2} = 3/5
        ranking = [0, 1, 2, 10, 11, 12, 13, 14, 15, 16,
                   17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
        logits = _make_logits_ranking(*ranking).unsqueeze(0)
        assert recall_at_k(logits, targets, k=20) == pytest.approx(0.6)

    def test_batch_averaging(self):
        """Batch of 2: one perfect, one zero -> average = 0.5."""
        t1 = _make_targets(0, 1, 2, 3, 4)
        t2 = _make_targets(0, 1, 2, 3, 4)

        # Sample 1: perfect top-5
        l1 = _make_logits_ranking(0, 1, 2, 3, 4)
        # Sample 2: zero overlap top-5
        l2 = _make_logits_ranking(35, 36, 37, 38, 34)

        logits = torch.stack([l1, l2])
        targets = torch.stack([t1, t2])
        assert recall_at_k(logits, targets, k=5) == pytest.approx(0.5)

    def test_single_sample_batch(self):
        """Single sample batch works correctly."""
        targets = _make_targets(10, 20, 30, 35, 38).unsqueeze(0)
        logits = _make_logits_ranking(10, 20, 30, 35, 38).unsqueeze(0)
        assert recall_at_k(logits, targets, k=5) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# hit_at_k
# ---------------------------------------------------------------------------


class TestHitAtK:
    """Verify hit@K with hand-computed values."""

    def test_perfect_prediction(self):
        """At least one true part in top-5 -> hit@5 = 1.0."""
        targets = _make_targets(1, 5, 10, 20, 30).unsqueeze(0)
        logits = _make_logits_ranking(1, 5, 10, 20, 30).unsqueeze(0)
        assert hit_at_k(logits, targets, k=5) == pytest.approx(1.0)

    def test_zero_overlap(self):
        """No true parts in top-5 -> hit@5 = 0.0."""
        targets = _make_targets(0, 1, 2, 3, 4).unsqueeze(0)
        logits = _make_logits_ranking(35, 36, 37, 38, 34).unsqueeze(0)
        assert hit_at_k(logits, targets, k=5) == pytest.approx(0.0)

    def test_one_hit(self):
        """One true part in top-5 -> hit@5 = 1.0."""
        targets = _make_targets(0, 1, 2, 3, 4).unsqueeze(0)
        # Only index 0 in top-5
        logits = _make_logits_ranking(0, 35, 36, 37, 38).unsqueeze(0)
        assert hit_at_k(logits, targets, k=5) == pytest.approx(1.0)

    def test_batch_mixed(self):
        """Batch of 3: two hits, one miss -> hit@5 = 2/3."""
        t1 = _make_targets(0, 1, 2, 3, 4)
        t2 = _make_targets(0, 1, 2, 3, 4)
        t3 = _make_targets(0, 1, 2, 3, 4)

        l1 = _make_logits_ranking(0, 1, 2, 3, 4)       # hit
        l2 = _make_logits_ranking(35, 36, 37, 38, 34)   # miss
        l3 = _make_logits_ranking(0, 35, 36, 37, 38)    # hit

        logits = torch.stack([l1, l2, l3])
        targets = torch.stack([t1, t2, t3])
        result = hit_at_k(logits, targets, k=5)
        assert result == pytest.approx(2.0 / 3.0)

    def test_hit_at_20_easier(self):
        """Hit@20 is easier than hit@5 — one true part in top-20."""
        targets = _make_targets(15, 16, 17, 18, 19).unsqueeze(0)
        # Top-20 includes indices 0-19
        ranking = list(range(20))
        logits = _make_logits_ranking(*ranking).unsqueeze(0)
        assert hit_at_k(logits, targets, k=20) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# mrr
# ---------------------------------------------------------------------------


class TestMRR:
    """Verify MRR with hand-computed values."""

    def test_perfect_rank_1(self):
        """Best true positive at rank 1 -> RR = 1.0."""
        targets = _make_targets(10, 20, 30, 35, 38).unsqueeze(0)
        # Index 10 is ranked #1
        logits = _make_logits_ranking(10, 0, 1, 2, 3).unsqueeze(0)
        assert mrr(logits, targets) == pytest.approx(1.0)

    def test_best_true_at_rank_2(self):
        """Best true positive at rank 2 -> RR = 0.5."""
        targets = _make_targets(10, 20, 30, 35, 38).unsqueeze(0)
        # Rank 1 = index 5 (not true), rank 2 = index 20 (true)
        logits = _make_logits_ranking(5, 20, 0, 1, 2).unsqueeze(0)
        assert mrr(logits, targets) == pytest.approx(0.5)

    def test_best_true_at_rank_3(self):
        """Best true positive at rank 3 -> RR = 1/3."""
        targets = _make_targets(10, 20, 30, 35, 38).unsqueeze(0)
        # Rank 1=5, rank 2=6, rank 3=10 (true)
        logits = _make_logits_ranking(5, 6, 10, 0, 1).unsqueeze(0)
        assert mrr(logits, targets) == pytest.approx(1.0 / 3.0)

    def test_story_example(self):
        """MRR from STORY-2.3 hand-computed example.

        Target: {10, 20, 30, 35, 38}
        Sorted logits: rank1=5, rank2=20, rank3=10, ...
        Best true positive at rank 2 (pos 20) -> RR = 0.5.
        """
        targets = _make_targets(10, 20, 30, 35, 38).unsqueeze(0)
        logits = _make_logits_ranking(5, 20, 10, 8, 30).unsqueeze(0)
        assert mrr(logits, targets) == pytest.approx(0.5)

    def test_batch_averaging(self):
        """Batch of 2: RR=1.0 and RR=0.5 -> MRR = 0.75."""
        t1 = _make_targets(10, 20, 30, 35, 38)
        t2 = _make_targets(10, 20, 30, 35, 38)

        # Sample 1: true positive at rank 1 -> RR = 1.0
        l1 = _make_logits_ranking(10, 0, 1, 2, 3)
        # Sample 2: true positive at rank 2 -> RR = 0.5
        l2 = _make_logits_ranking(5, 20, 0, 1, 2)

        logits = torch.stack([l1, l2])
        targets = torch.stack([t1, t2])
        assert mrr(logits, targets) == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# compute_all_metrics
# ---------------------------------------------------------------------------


class TestComputeAllMetrics:
    """Verify compute_all_metrics returns all 5 keys."""

    def test_returns_all_keys(self):
        """All 5 metric keys present."""
        targets = _make_targets(0, 1, 2, 3, 4).unsqueeze(0)
        logits = _make_logits_ranking(0, 1, 2, 3, 4).unsqueeze(0)
        result = compute_all_metrics(logits, targets)
        expected_keys = {
            "recall_at_5",
            "recall_at_20",
            "hit_at_5",
            "hit_at_20",
            "mrr",
        }
        assert set(result.keys()) == expected_keys

    def test_perfect_all_ones(self):
        """Perfect predictions -> all metrics = 1.0."""
        targets = _make_targets(0, 1, 2, 3, 4).unsqueeze(0)
        logits = _make_logits_ranking(0, 1, 2, 3, 4).unsqueeze(0)
        result = compute_all_metrics(logits, targets)
        for key, value in result.items():
            assert value == pytest.approx(1.0), f"{key} should be 1.0"

    def test_zero_overlap_metrics(self):
        """Zero overlap -> recall and hit = 0.0."""
        targets = _make_targets(0, 1, 2, 3, 4).unsqueeze(0)
        # All true parts ranked last (positions 34-38 are top-5)
        logits = _make_logits_ranking(
            34, 35, 36, 37, 38,
            30, 31, 32, 33, 29,
            25, 26, 27, 28, 24,
            20, 21, 22, 23, 19,
        ).unsqueeze(0)
        result = compute_all_metrics(logits, targets)
        assert result["recall_at_5"] == pytest.approx(0.0)
        assert result["hit_at_5"] == pytest.approx(0.0)
        assert result["recall_at_20"] == pytest.approx(0.0)
        assert result["hit_at_20"] == pytest.approx(0.0)

    def test_values_are_floats(self):
        """All returned values are Python floats."""
        targets = _make_targets(0, 1, 2, 3, 4).unsqueeze(0)
        logits = _make_logits_ranking(0, 1, 2, 3, 4).unsqueeze(0)
        result = compute_all_metrics(logits, targets)
        for key, value in result.items():
            assert isinstance(value, float), f"{key} should be float"


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------


class DummyModel(torch.nn.Module):
    """Dummy model that returns the last time-step of input as logits."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, W, 39) -> return last step: (batch, 39)
        return x[:, -1, :]


class TestEvaluateModel:
    """Verify evaluation harness with a dummy model."""

    @pytest.fixture
    def dummy_setup(self):
        """Create a dummy model, data, and DataLoader."""
        torch.manual_seed(42)
        n_samples = 10
        window_size = 7
        n_classes = 39

        # Create X where last step has highest values at true positions
        X = torch.rand(n_samples, window_size, n_classes)
        y = torch.zeros(n_samples, n_classes)

        # For each sample, set 5 true parts and boost those in last step
        for i in range(n_samples):
            true_parts = [(i * 5 + j) % n_classes for j in range(5)]
            for p in true_parts:
                y[i, p] = 1.0
                X[i, -1, p] = 100.0 + float(p)

        from c5_snn.data.dataset import CA5Dataset

        dataset = CA5Dataset(X, y)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        model = DummyModel()
        return model, loader

    def test_returns_metrics_and_per_sample(self, dummy_setup):
        """evaluate_model returns dict with metrics and per_sample."""
        model, loader = dummy_setup
        result = evaluate_model(model, loader, torch.device("cpu"))
        assert "metrics" in result
        assert "per_sample" in result

    def test_metrics_keys(self, dummy_setup):
        """Aggregate metrics contain all 5 keys."""
        model, loader = dummy_setup
        result = evaluate_model(model, loader, torch.device("cpu"))
        expected = {
            "recall_at_5",
            "recall_at_20",
            "hit_at_5",
            "hit_at_20",
            "mrr",
        }
        assert set(result["metrics"].keys()) == expected

    def test_per_sample_count(self, dummy_setup):
        """Per-sample list has correct length."""
        model, loader = dummy_setup
        result = evaluate_model(model, loader, torch.device("cpu"))
        assert len(result["per_sample"]) == 10

    def test_perfect_dummy_model(self, dummy_setup):
        """Dummy model with boosted logits gets perfect metrics."""
        model, loader = dummy_setup
        result = evaluate_model(model, loader, torch.device("cpu"))
        assert result["metrics"]["recall_at_5"] == pytest.approx(1.0)
        assert result["metrics"]["hit_at_5"] == pytest.approx(1.0)
        assert result["metrics"]["mrr"] == pytest.approx(1.0)

    def test_per_sample_fields(self, dummy_setup):
        """Each per-sample dict has required fields."""
        model, loader = dummy_setup
        result = evaluate_model(model, loader, torch.device("cpu"))
        sample = result["per_sample"][0]
        assert "sample_idx" in sample
        assert "true_parts" in sample
        assert "predicted_top20" in sample
        assert "recall_at_5" in sample
        assert "recall_at_20" in sample
        assert "hit_at_5" in sample
        assert "hit_at_20" in sample


# ---------------------------------------------------------------------------
# export_results
# ---------------------------------------------------------------------------


class TestExportResults:
    """Verify JSON and CSV export."""

    @pytest.fixture
    def sample_results(self):
        """Minimal metrics and per-sample data for export."""
        metrics = {
            "recall_at_5": 0.8,
            "recall_at_20": 0.95,
            "hit_at_5": 0.9,
            "hit_at_20": 1.0,
            "mrr": 0.7,
        }
        per_sample = [
            {
                "sample_idx": 0,
                "true_parts": [1, 2, 3, 4, 5],
                "predicted_top20": list(range(20)),
                "recall_at_5": 0.8,
                "recall_at_20": 1.0,
                "hit_at_5": 1.0,
                "hit_at_20": 1.0,
            },
            {
                "sample_idx": 1,
                "true_parts": [10, 20, 30, 35, 38],
                "predicted_top20": list(range(20)),
                "recall_at_5": 0.2,
                "recall_at_20": 0.4,
                "hit_at_5": 1.0,
                "hit_at_20": 1.0,
            },
        ]
        return metrics, per_sample

    def test_creates_json(self, sample_results, tmp_path):
        """export_results creates test_metrics.json."""
        metrics, per_sample = sample_results
        export_results(metrics, per_sample, "test_model", str(tmp_path))
        assert (tmp_path / "test_metrics.json").exists()

    def test_creates_csv(self, sample_results, tmp_path):
        """export_results creates test_per_sample.csv."""
        metrics, per_sample = sample_results
        export_results(metrics, per_sample, "test_model", str(tmp_path))
        assert (tmp_path / "test_per_sample.csv").exists()

    def test_json_schema(self, sample_results, tmp_path):
        """JSON matches architecture Section 4.6 schema."""
        metrics, per_sample = sample_results
        export_results(
            metrics,
            per_sample,
            "test_model",
            str(tmp_path),
            checkpoint_path="best.pt",
            split="test",
        )
        with open(tmp_path / "test_metrics.json") as f:
            data = json.load(f)

        assert data["model_name"] == "test_model"
        assert data["checkpoint"] == "best.pt"
        assert data["split"] == "test"
        assert data["n_samples"] == 2
        assert "metrics" in data
        assert data["metrics"]["recall_at_5"] == 0.8
        assert "evaluated_at" in data

    def test_csv_row_count(self, sample_results, tmp_path):
        """CSV has header + correct number of rows."""
        metrics, per_sample = sample_results
        export_results(metrics, per_sample, "test_model", str(tmp_path))
        csv_path = tmp_path / "test_per_sample.csv"
        lines = csv_path.read_text().strip().split("\n")
        assert len(lines) == 3  # header + 2 rows

    def test_creates_output_dir(self, sample_results, tmp_path):
        """export_results creates output directory if needed."""
        metrics, per_sample = sample_results
        deep = str(tmp_path / "a" / "b")
        export_results(metrics, per_sample, "test_model", deep)
        assert (tmp_path / "a" / "b" / "test_metrics.json").exists()
        assert (tmp_path / "a" / "b" / "test_per_sample.csv").exists()
