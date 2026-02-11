# STORY-4.4: Phase A Training & Comparison

**Epic:** Epic 4 — SNN Phase A
**Priority:** Must Have
**Story Points:** 3
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-02-11
**Sprint:** 4

---

## User Story

As a researcher,
I want both Phase A SNN models trained and compared against baselines,
so that I can see if the spiking approach shows promise for CA5 prediction.

---

## Description

### Background

With SpikingMLP (STORY-4.2) and SpikingCNN1D (STORY-4.3) implemented and unit-tested, the project is ready to run them on real CA5 data and measure actual performance. This is the first time SNN models face real evaluation — all prior work was structural (code, tests, shapes). STORY-4.4 answers the central question: **do spiking networks show promise on this task?**

The comparison must be fair: same data splits, same seeds, same evaluation metrics, same early-stopping strategy. The existing `compare` CLI command (STORY-3.4) trains only baseline models. This story extends the comparison pipeline to include Phase A SNN models alongside the baselines, producing a cumulative `phase_a_comparison.json` leaderboard.

### Scope

**In scope:**
- New CLI command `c5_snn phase-a` that trains both SNN models with 3 seeds each
- Reuse existing `compare` infrastructure (`build_comparison`, `format_comparison_table`, `save_comparison`)
- Baseline results included in comparison (re-run FrequencyBaseline + GRU with same seeds)
- `results/phase_a_comparison.json` containing all 4 models (freq, GRU, SpikingMLP, SpikingCNN1D)
- Console table with cumulative comparison across all models
- Update `docs/project-memory.md` with Phase A results and encoding recommendation
- RunPod detection: if Trainer timing probe projects >20 min locally, log warning
- Per-seed output directories: `results/snn_phase_a_mlp_seed{S}/`, `results/snn_phase_a_cnn_seed{S}/`

**Out of scope:**
- Hyperparameter tuning (Sprint 5)
- Window size tuning (Sprint 6)
- Modifying existing model architectures
- Modifying the Trainer class
- Rate-coded encoding experiments (future — use `direct` encoding for Phase A)
- RunPod deployment (just log warning if needed; actual RunPod use is manual)

### User Flow

1. Researcher ensures `data/raw/CA5_matrix_binary.csv` exists
2. Researcher runs `c5_snn phase-a --seeds 42,123,7 --output results/phase_a_comparison.json`
3. CLI loads data pipeline (same as `train` and `compare` commands)
4. FrequencyBaseline evaluated (deterministic, 1 run)
5. GRU baseline trained and evaluated with each seed
6. SpikingMLP trained and evaluated with each seed
7. SpikingCNN1D trained and evaluated with each seed
8. Comparison report built with all 4 models
9. Console prints formatted comparison table with mean ± std
10. JSON report saved to `results/phase_a_comparison.json`
11. Researcher updates `project-memory.md` with results and observations

---

## Acceptance Criteria

- [ ] New CLI command `c5_snn phase-a` exists and runs end-to-end
- [ ] Command accepts `--seeds` (default `"42,123,7"`) and `--output` (default `"results/phase_a_comparison.json"`) options
- [ ] FrequencyBaseline evaluated (deterministic, single run) and included in comparison
- [ ] GRU baseline trained with 3 seeds, evaluated on test split, metrics aggregated (mean ± std)
- [ ] SpikingMLP trained with 3 seeds, evaluated on test split, metrics aggregated (mean ± std)
- [ ] SpikingCNN1D trained with 3 seeds, evaluated on test split, metrics aggregated (mean ± std)
- [ ] Per-seed checkpoint bundles saved: `results/{model}_seed{S}/best_model.pt`, `config_snapshot.yaml`, `metrics.csv`, `pip_freeze.txt`
- [ ] `results/phase_a_comparison.json` contains all 4 models with `metrics_mean`, `metrics_std`, `n_seeds`, `training_time_s`, `environment`, `phase` fields
- [ ] Phase field values: `"baseline"` for freq/GRU, `"phase_a"` for SNN models
- [ ] Console prints formatted comparison table showing Recall@20, Hit@20, MRR for all models
- [ ] Same data splits used for all models (loaded once, shared across all training runs)
- [ ] `set_global_seed()` called before each seed's training run
- [ ] Unit tests for the `phase-a` CLI command (at minimum: command exists, help text works)
- [ ] `ruff check` passes and `pytest` all green
- [ ] `project-memory.md` updated with Phase A results summary and encoding recommendation

---

## Technical Notes

### Components

- **Modified file:** `src/c5_snn/cli.py` — add `phase-a` CLI command
- **Modified file:** `docs/project-memory.md` — add Phase A results section
- **Reused (no changes):** `src/c5_snn/training/trainer.py` — Trainer is model-agnostic
- **Reused (no changes):** `src/c5_snn/training/compare.py` — `build_comparison`, `format_comparison_table`, `save_comparison`
- **Reused (no changes):** `src/c5_snn/training/evaluate.py` — `evaluate_model`
- **Reused (no changes):** All model classes, SpikeEncoder, registry
- **Modified test file:** `tests/test_cli.py` — add phase-a command tests
- **New output:** `results/phase_a_comparison.json`

### CLI Command Design

```python
@cli.command("phase-a")
@click.option("--seeds", default="42,123,7", help="Comma-separated seeds.")
@click.option("--output", "output_path", default="results/phase_a_comparison.json")
def phase_a(seeds: str, output_path: str) -> None:
    """Train Phase A SNN models and compare against baselines."""
```

The command follows the same pattern as the existing `compare` command but extends it to 4 models:

```python
# Model training order:
# 1. FrequencyBaseline (deterministic, no training, 1 run)
# 2. GRU Baseline (3 seeds, trained with early stopping)
# 3. SpikingMLP (3 seeds, trained with early stopping)
# 4. SpikingCNN1D (3 seeds, trained with early stopping)
```

### Config Construction Pattern

Each SNN model needs its config constructed inline (similar to how `compare` builds GRU configs):

```python
def _make_snn_config(base_data_cfg: dict, model_type: str, seed: int, model_params: dict) -> dict:
    """Build a complete config dict for an SNN model run."""
    return {
        "experiment": {"name": f"{model_type}_seed{seed}", "seed": seed},
        "data": base_data_cfg,
        "model": {"type": model_type, **model_params},
        "training": {
            "epochs": 100,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "early_stopping_patience": 10,
            "early_stopping_metric": "val_recall_at_20",
        },
        "output": {"dir": f"results/{model_type}_seed{seed}"},
        "log_level": "INFO",
    }
```

### SNN Model Parameters (Phase A defaults)

```python
SNN_MODELS = {
    "spiking_mlp": {
        "encoding": "direct",
        "timesteps": 10,
        "hidden_sizes": [256, 128],
        "beta": 0.95,
        "surrogate": "fast_sigmoid",
    },
    "spiking_cnn1d": {
        "encoding": "direct",
        "timesteps": 10,
        "channels": [64, 64],
        "kernel_sizes": [3, 3],
        "beta": 0.95,
        "surrogate": "fast_sigmoid",
    },
}
```

### GRU Baseline Config

Reuse existing `baseline_gru.yaml` parameters — `hidden_size: 128`, `num_layers: 1`, `dropout: 0.0`.

### Comparison Report Schema

The `phase_a_comparison.json` follows the same schema as `baseline_comparison.json` (architecture Section 4.7), but with 4 models and `phase` values of `"baseline"` or `"phase_a"`:

```json
{
  "models": [
    {"name": "frequency_baseline", "phase": "baseline", ...},
    {"name": "gru_baseline", "phase": "baseline", ...},
    {"name": "spiking_mlp", "phase": "phase_a", ...},
    {"name": "spiking_cnn1d", "phase": "phase_a", ...}
  ],
  "generated_at": "...",
  "window_size": 21,
  "test_split_size": 1752
}
```

### Expected Console Output

```
Phase A Comparison Results
===========================

Model                 Phase     Recall@20       Hit@20          MRR
─────────────────     ───────   ──────────      ──────────      ──────────
frequency_baseline    baseline  0.6500          0.8200          0.4100
gru_baseline          baseline  0.7800±0.012   0.9100±0.008   0.5200±0.015
spiking_mlp           phase_a   0.7200±0.018   0.8800±0.011   0.4800±0.020
spiking_cnn1d         phase_a   0.7400±0.015   0.8900±0.009   0.4900±0.017

Window size: 21 | Test samples: 1752
Results saved to: results/phase_a_comparison.json
```

### Data Pipeline (Loaded Once, Shared)

```python
# Load data once — all models use same splits
df = load_csv(raw_path)
X, y = build_windows(df, window_size)
split_info = create_splits(n_samples=X.shape[0], ratios=(0.70, 0.15, 0.15), ...)
dataloaders = get_dataloaders(split_info, X, y, batch_size=64)
```

### Timing and RunPod Warning

The Trainer already has a timing probe (after epoch 2, projects total time). If projected >20 min, it logs a RunPod warning. No additional timing code needed for the warning — it's built into Trainer.

The `phase-a` command does track total training time per model for the comparison report.

### Edge Cases

- **SNN models fail to converge:** Early stopping will terminate after patience exhausted. Report whatever metrics are achieved.
- **All-zero predictions:** If an SNN model produces all zeros (no spikes), metrics will be low. This is useful diagnostic information.
- **Training time variance:** SNN models may train slower per epoch due to temporal loop. The Trainer's timing probe handles this.
- **Memory pressure:** SpikingMLP with 256+128 hidden units and T=10 timesteps uses more memory than GRU. If OOM on local GPU, reduce batch size or use CPU.

### Architecture References

- Section 4.7: Comparison report schema
- Section 5.2: Models module — Phase A models
- Section 10: Source tree — results directory structure
- Section 13.3 Rule #5: Common model interface
- Section 13.4: snnTorch guidance

---

## Dependencies

**Prerequisite Stories:**
- STORY-4.1: snnTorch Integration & Spike Encoding (SpikeEncoder class)
- STORY-4.2: Spiking MLP Model (SpikingMLP in registry)
- STORY-4.3: Spiking 1D-CNN Model (SpikingCNN1D in registry)
- STORY-3.3: Training Loop & train CLI (Trainer class)
- STORY-3.4: Baseline Results & Comparison (compare framework)

**Blocked Stories:**
- STORY-5.1: Spike-GRU Architecture (needs Phase A results for context)
- STORY-5.3: Phase B Evaluation & Cumulative Comparison (extends comparison)

**External Dependencies:**
- snnTorch 0.9.1 installed
- CA5 data file at `data/raw/CA5_matrix_binary.csv`

---

## Definition of Done

- [ ] `phase-a` CLI command implemented in `src/c5_snn/cli.py`
- [ ] Command trains FrequencyBaseline (1 run), GRU (3 seeds), SpikingMLP (3 seeds), SpikingCNN1D (3 seeds)
- [ ] Per-seed checkpoint bundles saved to `results/{model}_seed{S}/`
- [ ] `results/phase_a_comparison.json` generated with all 4 models
- [ ] Console prints formatted comparison table
- [ ] Unit tests in `tests/test_cli.py`:
  - [ ] `phase-a` command exists and has correct help text
  - [ ] `--seeds` and `--output` options work
- [ ] `docs/project-memory.md` updated with Phase A results
- [ ] `ruff check src/ tests/` passes with zero errors
- [ ] `pytest tests/ -v` passes (all existing + new tests)
- [ ] Acceptance criteria validated (all checked)
- [ ] Code committed to `main` branch and pushed

---

## Story Points Breakdown

- **CLI command implementation (config construction + training loop):** 1.5 points
- **Testing (CLI tests + manual validation):** 0.5 points
- **Run training & generate comparison report:** 0.5 points
- **Documentation (project-memory.md update):** 0.5 points
- **Total:** 3 points

**Rationale:** Low code complexity — the `phase-a` command is structurally identical to the existing `compare` command, just extended to 4 models instead of 2. The infrastructure (Trainer, compare, evaluate, models) is all built and tested. Most effort is the actual training runs and analyzing results. 3 points matches the sprint plan estimate.

---

## Additional Notes

- The `phase-a` command does NOT load a YAML config file — it constructs configs programmatically. This avoids needing a "phase A meta-config" and keeps the command self-contained.
- Data pipeline parameters (window_size=21, split_ratios=[0.70, 0.15, 0.15], batch_size=64) are hardcoded in the command to match the project's standard settings. This ensures all models see exactly the same data.
- The command uses `direct` encoding (T=1 effective) for Phase A. Rate-coded encoding (T=10+ timesteps) is a Phase B/HP-sweep concern.
- If neither SNN model beats GRU baseline on Recall@20, the story doc and project-memory.md should document hypotheses for Phase B adjustments (as noted in sprint plan).
- The `format_comparison_table` function from STORY-3.4 already handles multi-model display with mean ± std formatting.
- Per the architecture (Section 4.7), comparison JSON includes `phase` field — `"baseline"` for freq/GRU, `"phase_a"` for SNN models.

---

## Progress Tracking

**Status History:**
- 2026-02-11: Created by Scrum Master (AI)

**Actual Effort:** TBD (will be filled during/after implementation)

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
