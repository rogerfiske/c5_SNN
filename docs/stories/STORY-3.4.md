# STORY-3.4: Baseline Results & Comparison

**Epic:** Epic 3 — Baseline Models
**Priority:** Must Have
**Story Points:** 3
**Status:** Completed
**Assigned To:** ai_dev_agent
**Created:** 2026-02-11
**Sprint:** 3

---

## User Story

As a researcher,
I want documented baseline benchmarks on the full dataset,
so that I have quantified targets before starting SNN work.

---

## Description

### Background

With the FrequencyBaseline (STORY-3.1), GRUBaseline (STORY-3.2), and training infrastructure (STORY-3.3) all complete, the project needs its first real benchmark numbers. This story trains the GRU baseline with 3 different seeds, evaluates all models on the identical test split, produces a structured comparison JSON, and prints a formatted console leaderboard table.

This is the final story in Sprint 3 and marks the transition from infrastructure to SNN research. The baseline numbers established here become the "beat this" targets for all SNN models in Sprints 4-6.

### Scope

**In scope:**
- `compare.py` module in `src/c5_snn/training/` with `build_comparison()` function
- `compare` CLI subcommand that orchestrates: train (if needed) -> evaluate -> compare -> report
- Train GRU baseline with 3 seeds (42, 123, 7) to measure variance
- Evaluate FrequencyBaseline on the test split (single seed — deterministic, no variance)
- Produce `results/baseline_comparison.json` following the Section 4.7 schema
- Print a formatted comparison table to the console
- Update `docs/project-memory.md` with baseline results

**Out of scope:**
- Training hyperparameter tuning (Sprint 5)
- SNN model evaluation (Sprint 4+)
- Interactive dashboards or plots
- RunPod deployment automation (manual if needed)

### User Flow

1. User runs `python -m c5_snn.cli compare --config configs/baseline_gru.yaml --seeds 42,123,7`
2. System trains GRU with seed=42, seed=123, seed=7 (or loads existing checkpoints)
3. System evaluates each GRU checkpoint on the test split
4. System evaluates FrequencyBaseline on the same test split (deterministic, 1 run)
5. System computes mean ± std of GRU metrics across 3 seeds
6. System writes `results/baseline_comparison.json` with all model results
7. System prints formatted comparison table to console
8. User manually updates `docs/project-memory.md` with the results (or story does it)

---

## Acceptance Criteria

- [ ] `build_comparison()` function in `src/c5_snn/training/compare.py` accepts a list of model result dicts and produces a comparison report dict
- [ ] Comparison report follows Section 4.7 schema: `models` array with `name`, `type`, `phase`, `metrics_mean`, `metrics_std`, `n_seeds`, `training_time_s`, `environment`; plus `generated_at`, `window_size`, `test_split_size`
- [ ] Both models (FrequencyBaseline and GRU) evaluated on the identical test split
- [ ] `results/baseline_comparison.json` written with all metrics per model
- [ ] GRU trained with 3 seeds (42, 123, 7); report shows mean ± std for each metric
- [ ] FrequencyBaseline reported with `n_seeds: 1` and `metrics_std` all zeros (deterministic)
- [ ] `compare` CLI subcommand accepts `--config` and `--seeds` options
- [ ] Console prints formatted comparison table with model name, Recall@20, Hit@20, MRR columns
- [ ] `docs/project-memory.md` updated with baseline benchmark numbers
- [ ] All models evaluated using existing `evaluate_model()` from STORY-2.3
- [ ] `set_global_seed()` called before each seed run for reproducibility
- [ ] Uses logging from STORY-1.2 (no `print()` in library code; CLI uses `click.echo`)

---

## Technical Notes

### Components

- **New file:** `src/c5_snn/training/compare.py` — `build_comparison()` function
- **Modified file:** `src/c5_snn/training/__init__.py` — add `build_comparison` export
- **Modified file:** `src/c5_snn/cli.py` — add `compare` CLI subcommand
- **Modified file:** `docs/project-memory.md` — add baseline results
- **New test file:** `tests/test_compare.py` — comparison report tests
- **Existing deps:** `training/trainer.py`, `training/evaluate.py`, `training/metrics.py`, `models/base.py`

### `build_comparison()` Function

```python
def build_comparison(
    model_results: list[dict],
    window_size: int,
    test_split_size: int,
) -> dict:
    """Build a comparison report from multiple model evaluation results.

    Args:
        model_results: List of dicts, each with:
            - name: str (e.g., "frequency_baseline")
            - type: str ("heuristic" or "learned")
            - phase: str ("baseline")
            - seed_metrics: list[dict] — one metrics dict per seed
            - training_time_s: float
            - environment: str ("local" or "runpod")
        window_size: Window size W used for all evaluations.
        test_split_size: Number of test samples.

    Returns:
        Comparison report dict following Section 4.7 schema.
    """
```

**For each model:**
- If `n_seeds == 1`: `metrics_mean = seed_metrics[0]`, `metrics_std = {all zeros}`
- If `n_seeds > 1`: compute `mean` and `std` across seed_metrics for each metric key

### Comparison Report Schema (Section 4.7)

```json
{
  "models": [
    {
      "name": "frequency_baseline",
      "type": "heuristic",
      "phase": "baseline",
      "metrics_mean": {
        "recall_at_5": 0.xx,
        "recall_at_20": 0.xx,
        "hit_at_5": 0.xx,
        "hit_at_20": 0.xx,
        "mrr": 0.xx
      },
      "metrics_std": {
        "recall_at_5": 0.0,
        "recall_at_20": 0.0,
        "hit_at_5": 0.0,
        "hit_at_20": 0.0,
        "mrr": 0.0
      },
      "n_seeds": 1,
      "training_time_s": 0,
      "environment": "local"
    },
    {
      "name": "gru_baseline",
      "type": "learned",
      "phase": "baseline",
      "metrics_mean": { ... },
      "metrics_std": { ... },
      "n_seeds": 3,
      "training_time_s": 120.5,
      "environment": "local"
    }
  ],
  "generated_at": "2026-02-11T12:00:00Z",
  "window_size": 21,
  "test_split_size": 1752
}
```

### Console Comparison Table Format

```
Baseline Comparison Results
===========================

Model                 Recall@20       Hit@20          MRR
─────────────────     ──────────      ──────────      ──────────
frequency_baseline    0.6500          0.8200          0.4100
gru_baseline          0.7800±0.012   0.9100±0.008   0.5200±0.015

Window size: 21 | Test samples: 1752
Results saved to: results/baseline_comparison.json
```

### `compare` CLI Command

```python
@cli.command("compare")
@click.option("--config", "config_path", required=True,
              help="Path to experiment config YAML.")
@click.option("--seeds", default="42,123,7",
              help="Comma-separated seeds for multi-seed training.")
def compare(config_path: str, seeds: str) -> None:
    """Train with multiple seeds, evaluate, and compare baselines."""
```

**CLI orchestration:**
1. Load config, set up logging
2. Load raw data -> build windows -> create splits -> create dataloaders
3. For each seed: train GRU, evaluate on test split, collect metrics + timing
4. Evaluate FrequencyBaseline on same test split (one run)
5. Call `build_comparison()` to assemble report
6. Save `results/baseline_comparison.json`
7. Print formatted table

### Multi-Seed Training Strategy

For GRU with 3 seeds:
1. For each seed in [42, 123, 7]:
   a. Call `set_global_seed(seed)`
   b. Instantiate fresh `GRUBaseline(config)`
   c. Create `Trainer(model, config, dataloaders, device)`
   d. Override `config["experiment"]["seed"]` and `config["output"]["dir"]` per seed
   e. Call `trainer.run()` — get training time
   f. Evaluate on test split via `evaluate_model(model, test_loader, device)`
   g. Collect metrics dict

**Note:** Each seed gets its own output dir (e.g., `results/baseline_gru_seed42/`, `results/baseline_gru_seed123/`, `results/baseline_gru_seed7/`).

### Architecture References

- Section 4.6 (Evaluation Artifacts — test_metrics.json schema)
- Section 4.7 (Comparison Report — baseline_comparison.json schema)
- Section 5.3 (Training Module — `compare.py` listed as internal file)
- Section 7.3 (Evaluate workflow)
- Section 10 (Source tree — `results/baseline_comparison.json`)
- Section 13.3 Rule 3 (Seed before everything)
- Section 13.3 Rule 5 (Common Model Interface)

### Edge Cases

- **GRU training >20 min:** Trainer already has 2-epoch timing probe; log RunPod warning
- **Checkpoint already exists:** If `best_model.pt` already exists for a seed, could skip retraining (optimization for future — not required for this story)
- **FrequencyBaseline has no checkpoint:** Evaluate directly from model instance (no training step)
- **Identical metrics across seeds:** Valid — std will be 0.0
- **Empty test split:** Should error clearly (handled by splits module)

---

## Dependencies

**Prerequisite Stories:**
- STORY-3.1: Frequency/Recency Heuristic (FrequencyBaseline model)
- STORY-3.2: ANN GRU Baseline (GRUBaseline model)
- STORY-3.3: Training Loop & train CLI (Trainer class, train infrastructure)
- STORY-2.3: Evaluation Harness & Metrics (`evaluate_model`, `compute_all_metrics`)
- STORY-2.1: Windowed Tensor Construction (data pipeline)
- STORY-2.2: Time-Based Splits (DataLoaders)

**Blocked Stories:**
- STORY-4.4: Phase A Training & Comparison (extends comparison with SNN models)
- STORY-5.3: Phase B Evaluation & Comparison
- STORY-6.4: Final Comprehensive Comparison

**External Dependencies:**
- CA5 dataset available at `data/raw/CA5_matrix_binary.csv`
- RunPod B200 access (if GRU training >20 min locally)

---

## Definition of Done

- [ ] `build_comparison()` implemented in `src/c5_snn/training/compare.py`
- [ ] `compare` CLI subcommand implemented in `src/c5_snn/cli.py`
- [ ] `src/c5_snn/training/__init__.py` updated with `build_comparison` export
- [ ] Unit tests in `tests/test_compare.py`:
  - [ ] `build_comparison` produces correct schema
  - [ ] Mean/std computed correctly for multi-seed
  - [ ] Single-seed model has zero std
  - [ ] Generated timestamp present
- [ ] Integration test:
  - [ ] Compare 2 models (frequency + GRU) on tiny data with 2 seeds
  - [ ] JSON output file exists with correct structure
  - [ ] CLI compare command produces formatted table
- [ ] `docs/project-memory.md` updated with baseline numbers
- [ ] `ruff check src/ tests/` passes with zero errors
- [ ] `pytest tests/ -v` passes (all existing + new tests)
- [ ] CI green on GitHub Actions
- [ ] Acceptance criteria validated (all checked)
- [ ] Code committed to `main` branch and pushed

---

## Story Points Breakdown

- **`compare.py` module (build_comparison + helpers):** 1 point
- **`compare` CLI command (multi-seed orchestration):** 1 point
- **Testing + project-memory update:** 1 point
- **Total:** 3 points

**Rationale:** Low complexity. The heavy lifting (Trainer, evaluate, metrics) is already built. This story orchestrates existing components and adds a comparison report. The multi-seed loop is straightforward. Main work is in the CLI orchestration and formatted output.

---

## Additional Notes

- This is the capstone story for Sprint 3 — completing it means all baseline infrastructure is in place
- The `baseline_comparison.json` schema is reused in STORY-4.4, 5.3, and 6.4 for progressive comparisons
- FrequencyBaseline requires no training — just instantiate and evaluate
- GRU multi-seed training validates that the training infrastructure handles reproducibility correctly
- The comparison table format will be reused as the project "leaderboard" throughout Sprints 4-6
- If GRU training is slow locally, the user should train on RunPod and bring checkpoints back for evaluation

---

## Progress Tracking

**Status History:**
- 2026-02-11: Created by Scrum Master (AI)
- 2026-02-11: Started by Developer (AI)
- 2026-02-11: Completed by Developer (AI)

**Actual Effort:** 3 points (matched estimate)

**Implementation Notes:**
- Created `src/c5_snn/training/compare.py` with `build_comparison()`, `save_comparison()`, `format_comparison_table()`
- Added `compare` CLI subcommand to `src/c5_snn/cli.py` with `--config`, `--seeds`, `--output` options
- Uses population std (n divisor) for multi-seed metrics
- 25 new tests (235 total), all passing
- All 12 acceptance criteria validated

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
