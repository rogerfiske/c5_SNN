# STORY-6.2: Window Size Tuning Experiment

**Epic:** Epic 6 — SNN Phase C & Final Report
**Priority:** Must Have
**Story Points:** 5
**Status:** Completed
**Assigned To:** ai_dev_agent
**Created:** 2026-02-11
**Sprint:** 6

---

## User Story

As a researcher,
I want the spiking transformer evaluated across W=7 to W=90,
so that I find the optimal temporal context length.

---

## Description

### Background

All experiments in Sprints 3-5 used a fixed window size W=21. Phase B findings showed that all learned models cluster around Recall@20 ~ 0.51 regardless of architecture, suggesting that **window size may be a stronger lever than architecture choice**. This was explicitly called out as a Phase C recommendation.

The Spiking Transformer (STORY-6.1) supports variable window sizes W=7 through W=90 without architectural changes thanks to its learnable positional encoding (`nn.Parameter(1, max_window_size=100, d_model)` sliced to actual W). This makes it the ideal architecture for a systematic window size sweep.

This story implements a CLI command and sweep infrastructure to train the Spiking Transformer at 7 different window sizes, identify the optimal W, and feed that result into the HP sweep (STORY-6.3).

### Scope

**In scope:**
- New CLI command `c5_snn window-tune` that runs a two-phase window size experiment
- Phase 1 (Screening): Train SpikingTransformer at each of 7 window sizes with a single seed
- Phase 2 (Validation): Re-train top-3 window sizes with 3 seeds, evaluate on test set
- `results/window_tuning.csv` with screening results (all 7 window sizes)
- `results/window_tuning_top3.json` with multi-seed results (top 3 window sizes)
- Console leaderboard and analysis summary
- Update `docs/project-memory.md` with optimal W and findings

**Out of scope:**
- HP sweep at optimal W (STORY-6.3)
- Tuning other models (only SpikingTransformer)
- Window sizes outside 7-90 range
- Modifying the SpikingTransformer architecture
- RunPod execution (compute environment is the user's choice)

### User Flow

1. Researcher ensures `configs/snn_phase_c.yaml` exists (base config for SpikingTransformer)
2. Researcher runs `c5_snn window-tune --config configs/snn_phase_c.yaml`
3. **Phase 1 (Screening):** For each W in {7, 14, 21, 30, 45, 60, 90}:
   a. Rebuild windows from raw CSV with window_size=W
   b. Recompute train/val/test splits
   c. Train SpikingTransformer with a single seed
   d. Record val_recall_at_20, training_time_s, best_epoch, sample counts
4. Console prints screening leaderboard sorted by val_recall_at_20
5. **Phase 2 (Top-3):** Re-train top-3 window sizes with seeds 42, 123, 7
   a. For each top W × seed: full train + test evaluation
   b. Aggregate mean/std of test metrics across seeds
6. Console prints top-3 leaderboard with test metrics (mean ± std)
7. Saves `results/window_tuning.csv` (Phase 1) and `results/window_tuning_top3.json` (Phase 2)
8. Console prints analysis: optimal W, effect of window size on recall, training time trends

---

## Acceptance Criteria

- [ ] New CLI command `c5_snn window-tune` exists and runs end-to-end
- [ ] Command accepts `--config` (required), `--windows` (default `"7,14,21,30,45,60,90"`), `--seeds` (default `"42,123,7"`), `--top-k` (default 3), `--output` (default `"results/window_tuning.csv"`), `--screening-seed` (default 42)
- [ ] Phase 1: trains SpikingTransformer at each window size with single screening seed
- [ ] Phase 1: for each W, data is re-windowed and splits recomputed (different n_samples per W)
- [ ] `results/window_tuning.csv` contains columns: `window_size`, `n_train`, `n_val`, `n_test`, `val_recall_at_20`, `val_hit_at_20`, `val_mrr`, `training_time_s`, `best_epoch`
- [ ] Phase 2: top-K window sizes re-trained with multiple seeds, evaluated on test set
- [ ] `results/window_tuning_top3.json` contains multi-seed results following comparison report schema
- [ ] Console prints Phase 1 screening leaderboard sorted by val_recall_at_20
- [ ] Console prints Phase 2 test leaderboard with mean ± std
- [ ] Console prints analysis: optimal W, dataset size effect, training time per W
- [ ] Unit tests in `tests/test_compare.py` (or `tests/test_window_tune.py`):
  - [ ] `window-tune` command exists with correct help text
  - [ ] `--windows` and `--seeds` options parse correctly
- [ ] `ruff check src/ tests/` passes with zero errors
- [ ] `pytest tests/ -v` passes (all existing + new tests)

---

## Technical Notes

### Components

- **New/modified file:** `src/c5_snn/cli.py` — add `window-tune` CLI command
- **New/modified file:** `tests/test_compare.py` (or `tests/test_window_tune.py`) — unit tests
- **Modified file:** `docs/project-memory.md` — optimal W findings
- **New outputs:** `results/window_tuning.csv`, `results/window_tuning_top3.json`
- **Reused (no changes):** `src/c5_snn/models/snn_phase_c.py`, `src/c5_snn/data/loader.py`, `src/c5_snn/data/windowing.py`, `src/c5_snn/data/splits.py`, `src/c5_snn/data/dataset.py`, `src/c5_snn/training/trainer.py`, `src/c5_snn/training/evaluate.py`

### CLI Command Design

```python
@cli.command("window-tune")
@click.option("--config", "config_path", required=True,
              help="Path to base YAML config (must be spiking_transformer).")
@click.option("--windows", "window_str", default="7,14,21,30,45,60,90",
              help="Comma-separated window sizes to test.", show_default=True)
@click.option("--seeds", "seed_str", default="42,123,7",
              help="Comma-separated seeds for Phase 2 re-runs.", show_default=True)
@click.option("--top-k", "top_k", default=3,
              help="Number of top window sizes to re-run with seeds.", show_default=True)
@click.option("--output", "output_path", default="results/window_tuning.csv",
              help="Path for screening CSV.", show_default=True)
@click.option("--screening-seed", "screening_seed", default=42,
              help="Seed for Phase 1 screening runs.", show_default=True)
def window_tune(config_path, window_str, seed_str, top_k, output_path, screening_seed):
    """Sweep window sizes for SpikingTransformer (Phase C)."""
```

### Implementation Strategy

Follow the `phase-b-sweep` pattern closely with key differences:

**Phase 1 (Screening):**
```python
window_sizes = [int(w) for w in window_str.split(",")]
# For each window_size W:
#   1. load_csv(raw_path)
#   2. build_windows(df, W) -> (n_samples, W, 39) tensors
#   3. create_splits(n_samples, ratios, W, dates)
#   4. get_dataloaders(split_info, X, y, batch_size)
#   5. Override config: config["data"]["window_size"] = W
#   6. get_model(config) -> SpikingTransformer
#   7. Trainer(model, config, dataloaders, device).run()
#   8. Record: val_recall_at_20, training_time, best_epoch, sample counts
```

**Key difference from phase-b-sweep:** The data pipeline (windowing + splits) must be regenerated for each window size because changing W changes the number of samples. In phase-b-sweep, data was built once and reused because only model hyperparameters varied.

**Phase 2 (Top-3):**
```python
# Sort Phase 1 by val_recall_at_20 descending
# Take top-K window sizes
# For each top W:
#   For each seed in seeds:
#     1. Rebuild data pipeline for W
#     2. set_global_seed(seed)
#     3. Train + evaluate on test set
#     4. Record test metrics
#   Aggregate: mean/std of test_recall_at_20, test_hit_at_20, test_mrr
```

**Phase 2 output schema** (follows existing comparison report format):
```python
{
    "window_sizes": [
        {
            "window_size": W,
            "n_train": int,
            "n_val": int,
            "n_test": int,
            "metrics_mean": {"recall_at_20": ..., "hit_at_20": ..., "mrr": ...},
            "metrics_std": {"recall_at_20": ..., "hit_at_20": ..., "mrr": ...},
            "n_seeds": 3,
            "training_time_s": float  # average
        }
    ],
    "optimal_window_size": int,
    "generated_at": "ISO datetime",
    "model_type": "spiking_transformer",
    "model_config": {d_model, n_heads, n_layers, ...}
}
```

### Sample Counts per Window Size

Window size affects sample count: `n_samples = len(df) - W`. With 11,702 rows:

| W | n_samples | n_train (70%) | n_val (15%) | n_test (15%) |
|---|-----------|---------------|-------------|--------------|
| 7 | 11,695 | 8,186 | 1,754 | 1,755 |
| 14 | 11,688 | 8,181 | 1,753 | 1,754 |
| 21 | 11,681 | 8,176 | 1,752 | 1,753 |
| 30 | 11,672 | 8,170 | 1,750 | 1,752 |
| 45 | 11,657 | 8,159 | 1,748 | 1,750 |
| 60 | 11,642 | 8,149 | 1,746 | 1,747 |
| 90 | 11,612 | 8,128 | 1,741 | 1,743 |

Note: Sample counts vary only slightly (~0.7%) across window sizes, so splits are reasonably comparable. The test set size varies by at most 12 samples, which should not significantly affect metric comparisons.

### Expected Console Output

```
Window Size Tuning — Phase C (SpikingTransformer)
===================================================

Config: configs/snn_phase_c.yaml
Model: spiking_transformer (d_model=128, n_heads=4, n_layers=2)
Window sizes: [7, 14, 21, 30, 45, 60, 90]
Screening seed: 42

Phase 1: Screening (7 window sizes, 1 seed each)
--------------------------------------------------
[1/7] W=7  ... val_recall@20=0.XXXX  time=XX.Xs  epoch=XX
[2/7] W=14 ... val_recall@20=0.XXXX  time=XX.Xs  epoch=XX
...

Screening Leaderboard (sorted by val_recall@20):
W    n_samples  val_recall@20  val_hit@20  val_mrr   time(s)  epoch
---  ---------  -------------  ----------  --------  -------  -----
30   11672      0.XXXX         0.XXXX      0.XXXX    XX.X     XX
21   11681      0.XXXX         0.XXXX      0.XXXX    XX.X     XX
...

Phase 2: Top-3 Re-run (3 seeds each, test evaluation)
------------------------------------------------------
[1/9] W=30  seed=42  ... test_recall@20=0.XXXX
[2/9] W=30  seed=123 ... test_recall@20=0.XXXX
...

Top-3 Test Results (mean ± std):
W    test_recall@20      test_hit@20         test_mrr            seeds
---  ------------------  ------------------  ------------------  -----
30   0.XXXX +/- 0.XXX   0.XXXX +/- 0.XXX   0.XXXX +/- 0.XXX   3
21   0.XXXX +/- 0.XXX   0.XXXX +/- 0.XXX   0.XXXX +/- 0.XXX   3
14   0.XXXX +/- 0.XXX   0.XXXX +/- 0.XXX   0.XXXX +/- 0.XXX   3

Analysis:
  Optimal window size:    W=30
  vs W=21 (default):      +0.XXXX (+X.XX%)
  vs W=7 (smallest):      +0.XXXX (+X.XX%)
  Training time trend:    Larger W = longer training (W=7: XXs, W=90: XXs)
  Recommendation:         Use W=30 for STORY-6.3 HP sweep

Results saved to:
  Screening CSV:  results/window_tuning.csv
  Top-3 JSON:     results/window_tuning_top3.json
```

### Edge Cases

- **W > len(df) - 1:** Would produce 0 samples. `build_windows()` already raises `ConfigError` for this. Print error and skip that window size.
- **All window sizes produce similar recall:** This is a valid finding — document that window size doesn't matter much.
- **Very large W (90) with large d_model:** Attention matrix is 90×90 per head. Memory should be fine at batch_size=64 but monitor for OOM.
- **Training time scaling:** W=90 processes 90 positions per T-loop iteration vs W=7. Expect ~10-13x longer per epoch. Phase 1 screening should finish in <30 min on CPU for direct encoding.
- **Rate-coded encoding slowdown:** If base config uses `rate_coded` with T>1, screening could be very slow. Default config uses `direct` (T=1).

### Architecture References

- Section 5.2: Models module — SpikingTransformer supports variable W
- Section 4.3: Data windowing — `build_windows(df, W)` produces (N-W, W, 39) tensors
- Section 4.4: Splits — `create_splits()` handles variable n_samples
- Phase B sweep pattern: `cli.py` phase-b-sweep command (lines ~773-1108)

---

## Dependencies

**Prerequisite Stories:**
- STORY-6.1: Spiking Transformer Architecture (SpikingTransformer model class, variable W support)
- STORY-4.1: snnTorch Integration (SpikeEncoder)
- STORY-3.3: Training Loop & train CLI (Trainer class)
- STORY-2.1: Windowed Tensor Construction (build_windows)
- STORY-2.2: Time-Based Splits (create_splits)

**Blocked Stories:**
- STORY-6.3: HP Sweep & Best Model (needs optimal W from this story)

**External Dependencies:**
- None (all inputs are local CSV and config files)

---

## Definition of Done

- [ ] `window-tune` CLI command implemented in `src/c5_snn/cli.py`
- [ ] Phase 1: screens 7 window sizes with single seed
- [ ] Phase 2: re-trains top-3 with 3 seeds, evaluates on test set
- [ ] `results/window_tuning.csv` generated with 7 rows (one per W)
- [ ] `results/window_tuning_top3.json` generated with top-3 multi-seed results
- [ ] Console prints screening leaderboard, top-3 test leaderboard, and analysis
- [ ] Optimal W identified and documented in `docs/project-memory.md`
- [ ] Unit tests:
  - [ ] `window-tune` command exists with correct help text
  - [ ] `--windows` and `--seeds` options parse correctly
- [ ] `ruff check src/ tests/` passes with zero errors
- [ ] `pytest tests/ -v` passes (all existing + new tests)
- [ ] Acceptance criteria validated (all checked)
- [ ] Code committed to `main` branch and pushed

---

## Story Points Breakdown

- **CLI command + Phase 1 screening loop:** 1.5 points
- **Phase 2 top-K re-run with test evaluation:** 1.0 points
- **CSV/JSON output + console formatting:** 0.5 points
- **Testing:** 0.5 points
- **Actual sweep execution + analysis:** 1.0 points
- **Documentation (project-memory.md):** 0.5 points
- **Total:** 5 points

**Rationale:** This story follows the phase-b-sweep pattern closely, so the CLI infrastructure is well-understood. The key difference is that data must be re-windowed for each W (vs reusing data in phase-b-sweep). The sweep itself is straightforward but GPU-intensive — 7 screening runs + up to 9 top-3 runs = 16 total training runs. The 5-point estimate accounts for implementation time plus actual execution. Matches sprint plan estimate.

---

## Additional Notes

- The `window-tune` command does NOT modify the SpikingTransformer architecture. It only varies `config["data"]["window_size"]` across runs. The model's learnable positional encoding handles variable W automatically.
- The data pipeline (load CSV → build_windows → create_splits → get_dataloaders) must be rebuilt for each window size because changing W changes n_samples. This is the key difference from phase-b-sweep where data was built once.
- Phase B showed encoding mode doesn't matter. The base config should use `direct` encoding for fast screening. If the user wants to also test `rate_coded`, they can modify the config.
- Window sizes {7, 14, 21, 30, 45, 60, 90} span from ~1 week to ~6 months of historical events. This covers a wide range of temporal context.
- If all window sizes produce similar recall (as was the case for architecture choice), that's a valuable finding: it means the prediction task is dominated by recent events and long-range context doesn't help.
- The output files (`results/window_tuning.csv`, `results/window_tuning_top3.json`) are in `.gitignore` and won't be committed. The findings are documented in `project-memory.md`.
- Training time will scale roughly linearly with W for the SpikingTransformer (attention is O(W²) but the T-loop is the dominant cost at T=1).

---

## Progress Tracking

**Status History:**
- 2026-02-11: Created by Scrum Master (AI)
- 2026-02-11: Implemented by Developer (AI)
- 2026-02-11: Completed — all 14 AC validated

**Actual Effort:** 5 points (matched estimate)

**Implementation Notes:**
- CLI command `window-tune` added to `src/c5_snn/cli.py`
- Phase 1 screened 7 window sizes (W=7,14,21,30,45,60,90) with seed=42
- Phase 2 re-ran top-3 (W=90, W=7, W=14) with 3 seeds (42, 123, 7) on test set
- **Optimal W=90** — test_recall@20 = 0.5107 +/- 0.003
- W=90 is the only window that trains beyond epoch 1 (reaches best at epoch 20)
- All W=7-60 cluster at ~0.507 val_recall@20, early-stopping at epoch 11
- 11 new unit tests added to `tests/test_compare.py` (475 total tests passing)
- Total sweep time: ~32 min on CPU (Phase 1: 14 min, Phase 2: 18 min)

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
