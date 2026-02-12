# STORY-6.3: Spiking Transformer HP Sweep & Best Model

**Epic:** Epic 6 — SNN Phase C & Final Report
**Priority:** Must Have
**Story Points:** 5
**Status:** Completed
**Assigned To:** ai_dev_agent
**Completion Date:** 2026-02-12
**Created:** 2026-02-11
**Sprint:** 6

---

## User Story

As a researcher,
I want a structured HP sweep at the optimal window size,
so that the final SNN model is the best it can be.

---

## Description

### Background

STORY-6.1 delivered the SpikingTransformer architecture (Spikformer-style spike-form self-attention, spiking FFN with LIF neurons). STORY-6.2 established the optimal window size W=90 through a systematic window tuning experiment, finding that W=90 is the only window size where the transformer trains beyond epoch 1 and achieves +0.79% improvement over the default W=21.

Now the SpikingTransformer needs a full hyperparameter sweep at W=90 to find the best possible configuration. The sweep covers the architecture's key dimensions: number of transformer layers, attention heads, model dimension, LIF decay rate (beta), and encoding strategy. This is the final model training phase before the comprehensive comparison in STORY-6.4.

The sweep follows the proven two-phase approach from STORY-5.2 (Phase B): (1) broad single-seed screening across all grid combinations, (2) top-5 configs re-run with 3 seeds for statistical robustness. Since W=90 training is substantially slower (~300-430s per run vs ~30-60s at smaller windows), the total sweep time will be significant — RunPod B200 should be used for the Phase 1 screening pass.

### Scope

**In scope:**
- New CLI command `c5_snn phase-c-sweep` that runs the full HP sweep at W=90
- Sweep grid: `n_layers` [2, 4, 6] × `n_heads` [2, 4] × `d_model` [64, 128] × `beta` [0.5, 0.8, 0.95] × `encoding` [direct, rate_coded] = 72 configs
- `d_ffn` set to `2 × d_model` for each config (128 or 256)
- Single-seed (seed=42) screening pass for all 72 configs on validation set
- Top-5 configs (by `val_recall_at_20`) re-run with 3 seeds (42, 123, 7) on test set
- Sweep results logged to `results/phase_c_sweep.csv` (all 72 runs)
- Top-5 multi-seed results saved to `results/phase_c_top5.json` (comparison format)
- Best checkpoint saved with config + pip freeze provenance
- Console progress output during sweep
- Unit tests for the `phase-c-sweep` CLI command

**Out of scope:**
- Sweeping `timesteps` (fixed at T=10 for rate_coded)
- Sweeping `learning_rate` or `optimizer` (keep at 0.001/adam)
- Sweeping `dropout` (keep at 0.1)
- Sweeping `window_size` (fixed at W=90 from STORY-6.2)
- Sweeping `max_window_size` (fixed at 100)
- Final comprehensive comparison against all phases (STORY-6.4)
- Modifying the SpikingTransformer model class or Trainer class

### User Flow

1. Researcher ensures `data/raw/CA5_matrix_binary.csv` exists
2. Researcher runs `c5_snn phase-c-sweep --config configs/snn_phase_c.yaml --output results/phase_c_sweep.csv`
3. CLI loads data pipeline at W=90 (rebuilds windows/splits/dataloaders for optimal window)
4. Sweep Phase 1: 72 configs trained with seed=42, each evaluated on validation set
5. Console prints progress: `[1/72] spiking_transformer d=64 h=2 l=2 b=0.50 enc=direct ...`
6. After all 72 runs, sweep CSV saved and console prints sorted leaderboard
7. Sweep Phase 2: Top-5 configs identified by `val_recall_at_20`
8. Each top-5 config re-run with 3 seeds, evaluated on test set
9. Top-5 multi-seed results saved to `results/phase_c_top5.json`
10. Console prints comparison table with mean +/- std for top-5
11. Best checkpoint preserved in `results/phase_c_best/`

---

## Acceptance Criteria

- [x] New CLI command `c5_snn phase-c-sweep` exists and runs end-to-end
- [x] Command accepts `--config` (required), `--output` (default `"results/phase_c_sweep.csv"`), `--top-k` (default `5`), `--seeds` (default `"42,123,7"`), and `--screening-seed` (default `42`) options
- [x] Sweep grid covers: n_layers [2, 4, 6] × n_heads [2, 4] × d_model [64, 128] × beta [0.5, 0.8, 0.95] × encoding [direct, rate_coded] = 72 configs
- [x] `d_ffn` automatically set to `2 × d_model` for each config
- [x] Window size W=90 used for all runs (from STORY-6.2 optimal result)
- [x] Phase 1 (screening): All 72 configs trained with seed=42, evaluated on validation set
- [x] `results/phase_c_sweep.csv` contains all 72 rows with columns: config_id, d_model, n_heads, n_layers, d_ffn, beta, encoding, timesteps, val_recall_at_20, val_hit_at_20, val_mrr, training_time_s, best_epoch
- [x] Console prints numbered progress during sweep: `[N/72] ...`
- [x] After Phase 1, console prints sorted leaderboard of all 72 configs by val_recall_at_20
- [x] Phase 2 (top-K re-run): Top-5 configs re-run with 3 seeds each, evaluated on **test** set
- [x] `results/phase_c_top5.json` contains top-5 models in comparison report format (metrics_mean, metrics_std, n_seeds)
- [x] Best overall checkpoint (highest mean test recall_at_20 across seeds) saved to `results/phase_c_best/` with config_snapshot.yaml
- [x] `set_global_seed()` called before each training run
- [x] Data pipeline rebuilt at W=90 (windows, splits, dataloaders)
- [x] Unit tests for `phase-c-sweep` CLI command (command exists, help text, all options)
- [x] `ruff check` passes and `pytest` all green

---

## Technical Notes

### Components

- **Modified file:** `src/c5_snn/cli.py` — add `phase-c-sweep` CLI command
- **Modified test file:** `tests/test_compare.py` — add phase-c-sweep command tests
- **Reused (no changes):** `src/c5_snn/models/snn_phase_c.py` — SpikingTransformer model
- **Reused (no changes):** `src/c5_snn/training/trainer.py` — Trainer (model-agnostic)
- **Reused (no changes):** `src/c5_snn/training/compare.py` — comparison report functions
- **Reused (no changes):** `src/c5_snn/training/evaluate.py` — `evaluate_model`
- **New output:** `results/phase_c_sweep.csv`, `results/phase_c_top5.json`, `results/phase_c_best/`

### CLI Command Design

```python
@cli.command("phase-c-sweep")
@click.option("--config", "config_path", required=True,
              type=click.Path(exists=True), help="Base config YAML.")
@click.option("--output", "output_path", default="results/phase_c_sweep.csv",
              help="Path for sweep results CSV.")
@click.option("--top-k", default=5, type=int,
              help="Number of top configs to re-run with multi-seed.")
@click.option("--seeds", default="42,123,7",
              help="Comma-separated seeds for top-K re-runs.")
@click.option("--screening-seed", default=42, type=int,
              help="Seed for Phase 1 screening.")
def phase_c_sweep(config_path, output_path, top_k, seeds, screening_seed):
    """Run Spiking Transformer hyperparameter sweep at optimal W."""
```

### Sweep Grid

```python
import itertools

SWEEP_GRID = {
    "n_layers": [2, 4, 6],
    "n_heads": [2, 4],
    "d_model": [64, 128],
    "beta": [0.5, 0.8, 0.95],
    "encoding": ["direct", "rate_coded"],
}

# Generate all combinations: 3 × 2 × 2 × 3 × 2 = 72
combos = list(itertools.product(
    SWEEP_GRID["n_layers"],
    SWEEP_GRID["n_heads"],
    SWEEP_GRID["d_model"],
    SWEEP_GRID["beta"],
    SWEEP_GRID["encoding"],
))
```

### Config Construction per Sweep Run

```python
def _make_phase_c_sweep_config(
    n_layers, n_heads, d_model, beta, encoding,
    seed, config_id, base_config,
):
    return {
        "experiment": {"name": f"spiking_transformer_sweep_{config_id:03d}", "seed": seed},
        "data": {
            "raw_path": base_config["data"]["raw_path"],
            "window_size": 90,  # Optimal W from STORY-6.2
            "split_ratios": base_config["data"]["split_ratios"],
            "batch_size": base_config["data"]["batch_size"],
        },
        "model": {
            "type": "spiking_transformer",
            "encoding": encoding,
            "timesteps": 10,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "d_ffn": 2 * d_model,  # 2x d_model
            "beta": beta,
            "dropout": 0.1,
            "max_window_size": 100,
            "surrogate": "fast_sigmoid",
        },
        "training": {
            "epochs": 100,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "early_stopping_patience": 10,
            "early_stopping_metric": "val_recall_at_20",
        },
        "output": {"dir": f"results/phase_c_sweep_{config_id:03d}"},
        "log_level": "WARNING",  # Reduce noise during sweep
    }
```

### Data Pipeline (W=90)

Unlike the phase-b-sweep (which uses fixed W=21), this sweep MUST rebuild the data pipeline at W=90:

```python
# Load raw CSV once
df = load_csv(config["data"]["raw_path"])

# Build windows at W=90
X, y = build_windows(df, window_size=90)

# Create splits
n_samples = X.shape[0]
split_info = create_splits(
    n_samples=n_samples,
    ratios=tuple(config["data"]["split_ratios"]),
    window_size=90,
    dates=df["date"],
)

# Create dataloaders (shared across all 72 runs)
dataloaders = get_dataloaders(X, y, split_info, batch_size=64)
```

### Expected Sweep CSV Format

```csv
config_id,d_model,n_heads,n_layers,d_ffn,beta,encoding,timesteps,val_recall_at_20,val_hit_at_20,val_mrr,training_time_s,best_epoch
0,64,2,2,128,0.50,direct,1,0.5080,0.9755,0.3100,180.2,15
1,64,2,2,128,0.50,rate_coded,10,0.5090,0.9760,0.3120,350.4,18
...
71,128,4,6,256,0.95,rate_coded,10,0.5160,0.9785,0.3180,520.1,22
```

### Timing Estimates

- **Phase 1 (72 configs, single seed, W=90):**
  - Direct encoding configs (~36): ~150-300s each → ~90-180 min
  - Rate-coded encoding configs (~36): ~300-500s each → ~180-300 min
  - **Total Phase 1: ~4.5-8 hours locally** (RunPod recommended)
- **Phase 2 (top-5, 3 seeds each = 15 runs):**
  - ~200-500s each → ~50-125 min
- **Total estimated: ~6-10 hours locally, ~1-2 hours on RunPod B200**

### Compute Strategy

Given the estimated sweep time, this story should use RunPod B200:
1. Push code with `phase-c-sweep` command to GitHub
2. Clone on RunPod, install dependencies
3. Run sweep on GPU
4. Download results (`phase_c_sweep.csv`, `phase_c_top5.json`, `phase_c_best/`)
5. Commit results locally

### Edge Cases

- **n_heads must divide d_model:** Both d_model=64 and d_model=128 are divisible by 2 and 4, so all 72 configs are valid
- **OOM with d_model=128 + n_layers=6 + rate_coded at W=90:** Large model with long sequence. If OOM, reduce batch_size to 32 for that config
- **All 72 configs produce similar results:** Would confirm that the SpikingTransformer (like all other models) clusters around Recall@20 ~0.51 regardless of hyperparameters
- **Rate-coded consistently outperforms direct:** Would validate encoding hypothesis (not seen in Phase B)
- **6-layer models overfit or fail to converge:** Deeper transformers may need different learning rates; early stopping will terminate

### Architecture References

- Section 4.4: Experiment config schema
- Section 4.7: Comparison report schema
- Section 5.2: Models module — Phase C models
- Section 13.3 Rule #4: Config immutable at runtime
- Section 13.4: snnTorch guidance

---

## Dependencies

**Prerequisite Stories:**
- STORY-6.1: Spiking Transformer Architecture (SpikingTransformer model class)
- STORY-6.2: Window Size Tuning (optimal W=90)
- STORY-3.3: Training Loop & train CLI (Trainer class)
- STORY-5.2: HP Sweep (phase-b-sweep pattern to follow)

**Blocked Stories:**
- STORY-6.4: Final Comprehensive Comparison (needs Phase C best model)

**External Dependencies:**
- snnTorch 0.9.1 installed
- CA5 data file at `data/raw/CA5_matrix_binary.csv`
- RunPod B200 access for GPU-accelerated sweep

---

## Definition of Done

- [x] `phase-c-sweep` CLI command implemented in `src/c5_snn/cli.py`
- [x] Command runs 72-config sweep (Phase 1) + top-K multi-seed re-run (Phase 2)
- [x] `results/phase_c_sweep.csv` generated with all 72 configs and metrics
- [x] `results/phase_c_top5.json` generated with top-5 multi-seed results
- [x] Best checkpoint saved to `results/phase_c_best/`
- [x] Console prints progress, screening leaderboard, and top-K comparison table
- [x] Unit tests in `tests/test_compare.py`:
  - [x] `phase-c-sweep` command exists and has correct help text
  - [x] `--config`, `--output`, `--top-k`, `--seeds`, `--screening-seed` options work
- [x] `ruff check src/ tests/` passes with zero errors
- [x] `pytest tests/ -v` passes (all existing + new tests)
- [x] Acceptance criteria validated (all checked)
- [x] Code committed to `main` branch and pushed

---

## Story Points Breakdown

- **CLI command implementation (sweep grid + two-phase loop):** 2.0 points
- **CSV/JSON output generation and progress reporting:** 1.0 points
- **Running the actual sweep and analyzing results:** 1.5 points
- **Testing (CLI tests + manual validation):** 0.5 points
- **Total:** 5 points

**Rationale:** Structurally identical to STORY-5.2 (Phase B sweep) but with a larger grid (72 vs 36 configs) and longer per-run training times (W=90). The implementation pattern is well-established. Most effort is sweep execution time and results analysis. 5 points matches the sprint plan estimate.

---

## Additional Notes

- The sweep uses `val_recall_at_20` for screening (Phase 1) but `test set` metrics for the top-K re-run (Phase 2). This prevents data leakage.
- Fixed parameters across all sweep runs: `learning_rate=0.001`, `optimizer=adam`, `early_stopping_patience=10`, `dropout=0.1`, `timesteps=10` (for rate_coded), `window_size=90`, `batch_size=64`, `max_window_size=100`, `surrogate=fast_sigmoid`.
- `d_ffn` is automatically derived as `2 × d_model` (not independently swept) to keep the grid manageable.
- The `phase_c_top5.json` follows the same schema as `phase_b_top3.json` (Section 4.7) with `phase: "phase_c"`, making it directly compatible with STORY-6.4's final comparison.
- STORY-6.2 showed that W=90 trains for ~20 epochs (unlike W=7-60 which plateau at epoch 1), so expect meaningful training dynamics across the sweep.

---

## Progress Tracking

**Status History:**
- 2026-02-11: Created by Scrum Master (AI)
- 2026-02-11: Implementation started (CLI command + tests)
- 2026-02-12: Phase 1+2 sweep completed on RunPod B200 (10.3 hours total)
- 2026-02-12: Results verified, all acceptance criteria validated, story completed

**Actual Effort:** 5 points (matched estimate). Phase 1 screening: 7.6h, Phase 2 top-5: 2.7h. Total GPU time: 10.3h on NVIDIA B200.

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
