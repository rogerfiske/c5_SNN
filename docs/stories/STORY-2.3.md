# STORY-2.3: Evaluation Harness & Metrics

**Epic:** Epic 2 — Data Pipeline & Evaluation Harness
**Priority:** Must Have
**Story Points:** 5
**Status:** Completed
**Assigned To:** ai_dev_agent
**Created:** 2026-02-11
**Sprint:** 2

---

## User Story

As a researcher,
I want a reusable evaluation module that computes all defined metrics given model predictions and ground truth,
so that every model is measured consistently.

---

## Description

### Background

With windowed tensors (STORY-2.1) and time-based splits with DataLoaders (STORY-2.2) now in place, the project needs a standardized way to measure model quality. Every model in the project — from the frequency heuristic baseline through to the Spiking Transformer — must be evaluated using the same metrics computed the same way, on the same test split.

This story creates the evaluation infrastructure: individual metric functions (Recall@K, Hit@K, MRR), a full evaluation harness that runs inference and computes all metrics, and export of results as both a JSON summary and a per-sample CSV. This ensures fair, reproducible comparison across all models.

The CA5 task is a multi-label ranking problem: given a window of past events, predict which 5 of 39 parts will appear in the next event. The model outputs 39 logits (one per part), and we evaluate by ranking those logits and checking overlap with the 5 true parts.

### Scope

**In scope:**
- Individual metric functions: `recall_at_k`, `hit_at_k`, `mrr`
- Full evaluation harness: `evaluate_model(model, dataloader, device) -> MetricsDict`
- JSON summary export (`test_metrics.json`) matching architecture Section 4.6 schema
- Per-sample CSV export (`test_per_sample.csv`)
- Device-agnostic evaluation via `torch.device`
- Unit tests with hand-computed expected values

**Out of scope:**
- CLI `evaluate` command integration (STORY-2.4)
- Training loop and checkpoint management (STORY-3.3)
- Cross-model comparison reports (STORY-3.4)
- Any specific model implementations

### User Flow

1. User has a trained model and a test DataLoader
2. User calls `evaluate_model(model, test_loader, device)`
3. System runs inference batch-by-batch, collecting all logits and targets
4. System computes Recall@5, Recall@20, Hit@5, Hit@20, and MRR across all samples
5. System returns a `MetricsDict` with aggregate results
6. User optionally calls `export_results()` to save JSON summary + per-sample CSV

---

## Acceptance Criteria

- [ ] `src/c5_snn/training/metrics.py` implements `recall_at_k(logits, targets, k) -> float` — fraction of true positives in the top-K ranked predictions, averaged across the batch
- [ ] `src/c5_snn/training/metrics.py` implements `hit_at_k(logits, targets, k) -> float` — fraction of samples where at least one true positive is in the top-K
- [ ] `src/c5_snn/training/metrics.py` implements `mrr(logits, targets) -> float` — Mean Reciprocal Rank: average of `1/rank` of the highest-ranked true positive
- [ ] `src/c5_snn/training/metrics.py` implements `compute_all_metrics(logits, targets) -> dict` — returns all 5 metrics (Recall@5, Recall@20, Hit@5, Hit@20, MRR)
- [ ] `src/c5_snn/training/evaluate.py` implements `evaluate_model(model, dataloader, device) -> MetricsDict` — runs inference on all batches, concatenates results, computes aggregate metrics
- [ ] `evaluate_model` is device-agnostic via `torch.device` — model and tensors moved to device before inference, results on CPU
- [ ] `evaluate_model` calls `model.eval()` and uses `torch.no_grad()` during inference
- [ ] `src/c5_snn/training/evaluate.py` implements `export_results(metrics, per_sample, model_name, output_dir)` — saves `test_metrics.json` and `test_per_sample.csv`
- [ ] `test_metrics.json` schema matches architecture Section 4.6: `model_name`, `checkpoint`, `split`, `n_samples`, `metrics` dict, `evaluated_at`
- [ ] Unit tests verify metrics with hand-computed values:
  - Perfect predictions → all metrics = 1.0
  - Zero overlap → Recall@K = 0.0, Hit@K = 0.0
  - Known partial overlap → exact expected values
  - MRR hand-computed for a specific example
  - Batch of mixed results → correct average
- [ ] Uses logging from STORY-1.2 (no `print()` statements)

---

## Technical Notes

### Components

- **New file:** `src/c5_snn/training/metrics.py` — individual metric functions
- **New file:** `src/c5_snn/training/evaluate.py` — evaluation harness + export
- **Modified file:** `src/c5_snn/training/__init__.py` — export new functions
- **New test file:** `tests/test_metrics.py`

### Metric Definitions

All metrics operate on:
- `logits`: Tensor of shape `(batch, 39)` — raw model output scores
- `targets`: Tensor of shape `(batch, 39)` — multi-hot ground truth (exactly 5 ones per row)

**Recall@K:**
For each sample, take the top-K indices by logit score. Count how many of the 5 true parts are in those K predictions. Divide by 5 (the total number of true parts). Average across all samples.
```
Recall@K = mean over samples of: |top_K ∩ true_parts| / |true_parts|
```
- Recall@5: best case 1.0 (all 5 in top 5), worst 0.0
- Recall@20: best case 1.0, worst 0.0 — primary metric for early stopping

**Hit@K:**
For each sample, 1 if at least one true part is in the top-K, else 0. Average across samples.
```
Hit@K = mean over samples of: 1 if |top_K ∩ true_parts| > 0 else 0
```

**MRR (Mean Reciprocal Rank):**
For each sample, find the rank of the highest-ranked true positive. Reciprocal Rank = 1/rank. Average across samples.
```
MRR = mean over samples of: 1 / rank_of_best_true_positive
```
Rank is 1-indexed (best = rank 1 → RR = 1.0).

### Key Functions

```python
# metrics.py
def recall_at_k(logits: Tensor, targets: Tensor, k: int) -> float:
    """Recall@K averaged over the batch."""

def hit_at_k(logits: Tensor, targets: Tensor, k: int) -> float:
    """Hit@K averaged over the batch."""

def mrr(logits: Tensor, targets: Tensor) -> float:
    """Mean Reciprocal Rank averaged over the batch."""

def compute_all_metrics(logits: Tensor, targets: Tensor) -> dict[str, float]:
    """Compute all 5 metrics: recall@5, recall@20, hit@5, hit@20, mrr."""

# evaluate.py
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Run full evaluation: inference + all metrics.

    Returns dict with 'metrics' (aggregate) and 'per_sample' (per-row details).
    """

def export_results(
    metrics: dict,
    per_sample: list[dict],
    model_name: str,
    output_dir: str,
    checkpoint_path: str = "",
    split: str = "test",
) -> None:
    """Save test_metrics.json and test_per_sample.csv."""
```

### Implementation Details

1. **Top-K extraction:** Use `torch.topk(logits, k, dim=1)` to get the top-K indices per sample. Compare against `targets.bool()` or find true indices via `targets.nonzero()`.

2. **Batch accumulation in evaluate_model:** Iterate through the DataLoader, collecting all logits and targets into lists, then `torch.cat` before computing metrics. This ensures metrics are computed over the full split, not averaged per-batch.

3. **Per-sample results:** For each sample, store: `sample_idx`, `true_parts` (list of 5 part indices), `predicted_top20` (list of 20 ranked indices), and individual metric values.

4. **Model.eval() + no_grad:** Always call `model.eval()` before inference and wrap in `torch.no_grad()` context. Restore model to previous training state is not needed (caller manages).

5. **Architecture references:**
   - Section 4.6 (Evaluation Artifacts — JSON + CSV schemas)
   - Section 5.3 (Training Module — `evaluate_model`, `compute_recall_at_k`)
   - Section 7.3 (evaluate workflow sequence diagram)
   - FR9 (evaluation harness + metrics)
   - FR12 (Top-20 ranked output)

### Hand-Computed Test Cases

**Perfect prediction (Recall@5 = 1.0):**
- Target: parts {1, 5, 10, 20, 30} (positions where value=1)
- Logits: highest at positions 1, 5, 10, 20, 30
- Top-5 = {1, 5, 10, 20, 30} → overlap = 5/5 = 1.0

**Zero overlap (Recall@K = 0.0):**
- Target: parts {1, 2, 3, 4, 5}
- Logits: highest at positions 35, 36, 37, 38, 39
- Top-5 = {35,36,37,38,39} → overlap = 0/5 = 0.0

**Partial overlap (Recall@5 = 0.4):**
- Target: parts {1, 5, 10, 20, 30}
- Top-5 = {1, 5, 11, 21, 31} → overlap = 2/5 = 0.4

**MRR example:**
- Target: parts {10, 20, 30, 35, 39}
- Sorted logits: position 5, 20, 10, 8, 30, ... (rank 1=pos5, rank 2=pos20, rank 3=pos10)
- Best true positive is at rank 2 (pos 20) → RR = 1/2 = 0.5

### Edge Cases

- Single sample batch
- Batch size = 1 with perfect/zero predictions
- All 39 logits identical (ties) — `topk` breaks ties arbitrarily, acceptable
- Model outputs very large/small logits — no numerical issues with `topk`

---

## Dependencies

**Prerequisite Stories:**
- STORY-2.2: Time-Based Splits (provides DataLoaders for evaluation)
- STORY-1.2: Logging, Seed Management & Config (logging infrastructure)

**Blocked Stories:**
- STORY-2.4: CLI evaluate Command (needs `evaluate_model` and `export_results`)
- STORY-3.3: Training Loop (uses metrics for val monitoring and early stopping)
- STORY-3.4: Baseline Results & Comparison (needs evaluation results)

**External Dependencies:**
- None

---

## Definition of Done

- [ ] `src/c5_snn/training/metrics.py` implemented with `recall_at_k`, `hit_at_k`, `mrr`, `compute_all_metrics`
- [ ] `src/c5_snn/training/evaluate.py` implemented with `evaluate_model`, `export_results`
- [ ] `src/c5_snn/training/__init__.py` updated with exports
- [ ] Unit tests in `tests/test_metrics.py` written and passing:
  - [ ] Perfect predictions → all metrics = 1.0
  - [ ] Zero overlap → metrics = 0.0
  - [ ] Partial overlap with hand-computed expected values
  - [ ] MRR hand-computed for specific ranking
  - [ ] Batch averaging works correctly
  - [ ] `compute_all_metrics` returns all 5 keys
  - [ ] `evaluate_model` produces correct results with a dummy model
  - [ ] `export_results` saves valid JSON and CSV
- [ ] `ruff check src/ tests/` passes with zero errors
- [ ] `pytest tests/ -v` passes (all existing + new tests)
- [ ] CI green on GitHub Actions
- [ ] Acceptance criteria validated (all checked)
- [ ] Code committed to `main` branch and pushed

---

## Story Points Breakdown

- **Metric functions (recall, hit, mrr):** 2 points (ranking logic, batch handling)
- **Evaluation harness (inference + export):** 1.5 points
- **Testing (hand-computed values, edge cases):** 1.5 points
- **Total:** 5 points

**Rationale:** Moderate complexity — the metric math is well-defined but requires careful implementation (especially MRR). The evaluation harness is straightforward batch accumulation. Most effort goes into thorough testing with hand-computed expected values to ensure correctness.

---

## Additional Notes

- This story completes the evaluation side of the pipeline — after this, any model can be measured
- `recall_at_20` is the primary metric used for early stopping in training (STORY-3.3) and for the final leaderboard
- The `compute_all_metrics` function returns a flat dict, not nested — easy to log to CSV or JSON
- Per-sample CSV enables debugging: which events does the model get right/wrong?
- The evaluation harness must work with ANY model that follows the `BaseModel` interface: `(batch, W, 39) -> (batch, 39)` — tested with a dummy model in this story
- For the full test split (1,752 samples), evaluation is fast (< 1 second on CPU)

---

## Progress Tracking

**Status History:**
- 2026-02-11: Created by Scrum Master (AI)
- 2026-02-11: Implemented and completed by Developer (AI)

**Actual Effort:** 5 points (matched estimate)

**Implementation Notes:**
- `src/c5_snn/training/metrics.py`: `recall_at_k`, `hit_at_k`, `mrr`, `compute_all_metrics`
- `src/c5_snn/training/evaluate.py`: `evaluate_model`, `export_results`
- `tests/test_metrics.py`: 31 tests across 7 test classes
- All 11 acceptance criteria validated
- Ruff clean, all 124 tests passing

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
