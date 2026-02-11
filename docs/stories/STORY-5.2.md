# STORY-5.2: Spike-GRU Hyperparameter Sweep

**Epic:** Epic 5 — SNN Phase B
**Priority:** Must Have
**Story Points:** 5
**Status:** Completed
**Assigned To:** ai_dev_agent
**Created:** 2026-02-11
**Sprint:** 5

---

## User Story

As a researcher,
I want a structured HP sweep for the Spike-GRU,
so that the architecture gets a fair shot.

---

## Description

### Background

STORY-5.1 delivered the SpikeGRU model with a single default configuration (hidden_size=128, num_layers=1, beta=0.95, direct encoding). Phase A showed all models clustering around Recall@20 ~0.51 with direct encoding — the key finding was that direct encoding (T=1) limits SNN potential by collapsing the temporal loop to a single timestep.

Before declaring whether recurrent spiking networks add value, the Spike-GRU deserves a systematic hyperparameter sweep across its most impactful dimensions. The sweep must include **rate-coded encoding** (T=10), which was identified as the most likely lever for SNN improvement. Without this sweep, we cannot fairly compare Spike-GRU against the GRU baseline.

The sweep follows a two-phase approach: (1) broad single-seed sweep across all 36 grid combinations to identify promising regions, (2) top-3 configs re-run with 3 seeds each for statistical robustness. Results feed directly into STORY-5.3's cumulative comparison.

### Scope

**In scope:**
- New CLI command `c5_snn phase-b-sweep` that runs the full HP sweep
- Sweep grid: `hidden_size` [64, 128, 256] × `num_layers` [1, 2] × `beta` [0.5, 0.8, 0.95] × `encoding` [direct, rate_coded] = 36 configs
- Single-seed (seed=42) screening pass for all 36 configs
- Top-3 configs (by `val_recall_at_20`) re-run with 3 seeds (42, 123, 7)
- Sweep results logged to `results/phase_b_sweep.csv` (all 36 runs)
- Top-3 multi-seed results saved to `results/phase_b_top3.json` (comparison format)
- Best checkpoint saved with config provenance
- Console progress output during sweep
- RunPod warning if any individual run projects >20 min (handled by Trainer's built-in timing probe)
- Unit tests for the `phase-b-sweep` CLI command

**Out of scope:**
- Sweeping `timesteps` (fixed at T=10 for rate_coded) — can be explored in Sprint 6
- Sweeping `learning_rate` or `optimizer` — keep at 0.001/adam
- Sweeping `dropout` — keep at 0.0 for now (num_layers=2 runs get dropout=0.0)
- Phase B cumulative comparison against all prior models (STORY-5.3)
- Spiking Transformer (Sprint 6)
- Modifying the Trainer class or existing models
- RunPod deployment (just log warning; actual RunPod use is manual)

### User Flow

1. Researcher ensures `data/raw/CA5_matrix_binary.csv` exists
2. Researcher runs `c5_snn phase-b-sweep --output results/phase_b_sweep.csv`
3. CLI loads data pipeline (same as `phase-a` command)
4. Sweep Phase 1: 36 configs trained with seed=42, each evaluated on validation set
5. Console prints progress: `[1/36] spike_gru h=64 l=1 b=0.50 enc=direct ...`
6. After all 36 runs, sweep CSV saved and console prints sorted leaderboard
7. Sweep Phase 2: Top-3 configs identified by `val_recall_at_20`
8. Each top-3 config re-run with 3 seeds, evaluated on test set
9. Top-3 multi-seed results saved to `results/phase_b_top3.json`
10. Console prints comparison table with mean +/- std for top-3
11. Best checkpoint preserved in `results/phase_b_best/`

---

## Acceptance Criteria

- [ ] New CLI command `c5_snn phase-b-sweep` exists and runs end-to-end
- [ ] Command accepts `--output` (default `"results/phase_b_sweep.csv"`) and `--top-k` (default `3`) and `--seeds` (default `"42,123,7"`) options
- [ ] Sweep grid covers: hidden_size [64, 128, 256] × num_layers [1, 2] × beta [0.5, 0.8, 0.95] × encoding [direct, rate_coded] = 36 configs
- [ ] Phase 1 (screening): All 36 configs trained with seed=42, evaluated on validation set
- [ ] `results/phase_b_sweep.csv` contains all 36 rows with columns: config_id, hidden_size, num_layers, beta, encoding, timesteps, val_recall_at_20, val_hit_at_20, val_mrr, training_time_s, best_epoch
- [ ] Console prints numbered progress during sweep: `[N/36] ...`
- [ ] After Phase 1, console prints sorted leaderboard of all 36 configs by val_recall_at_20
- [ ] Phase 2 (top-K re-run): Top-3 configs re-run with 3 seeds each, evaluated on **test** set
- [ ] `results/phase_b_top3.json` contains top-3 models in comparison report format (metrics_mean, metrics_std, n_seeds)
- [ ] Best overall checkpoint (highest mean test recall_at_20 across seeds) saved to `results/phase_b_best/` with config_snapshot.yaml
- [ ] `set_global_seed()` called before each training run
- [ ] Same data splits used for all runs (loaded once, shared)
- [ ] Rate-coded runs use `timesteps=10` (fixed)
- [ ] Unit tests for `phase-b-sweep` CLI command (at minimum: command exists, help text, options)
- [ ] `ruff check` passes and `pytest` all green

---

## Technical Notes

### Components

- **Modified file:** `src/c5_snn/cli.py` — add `phase-b-sweep` CLI command
- **Modified test file:** `tests/test_compare.py` — add phase-b-sweep command tests
- **Reused (no changes):** `src/c5_snn/models/snn_phase_b.py` — SpikeGRU model
- **Reused (no changes):** `src/c5_snn/training/trainer.py` — Trainer (model-agnostic)
- **Reused (no changes):** `src/c5_snn/training/compare.py` — comparison report functions
- **Reused (no changes):** `src/c5_snn/training/evaluate.py` — `evaluate_model`
- **New output:** `results/phase_b_sweep.csv`, `results/phase_b_top3.json`

### CLI Command Design

```python
@cli.command("phase-b-sweep")
@click.option("--output", "output_path", default="results/phase_b_sweep.csv",
              help="Path for sweep results CSV.")
@click.option("--top-k", default=3, type=int,
              help="Number of top configs to re-run with multi-seed.")
@click.option("--seeds", default="42,123,7",
              help="Comma-separated seeds for top-K re-runs.")
def phase_b_sweep(output_path: str, top_k: int, seeds: str) -> None:
    """Run Spike-GRU hyperparameter sweep."""
```

### Sweep Grid

```python
import itertools

SWEEP_GRID = {
    "hidden_size": [64, 128, 256],
    "num_layers": [1, 2],
    "beta": [0.5, 0.8, 0.95],
    "encoding": ["direct", "rate_coded"],
}

# Generate all combinations: 3 × 2 × 3 × 2 = 36
combos = list(itertools.product(
    SWEEP_GRID["hidden_size"],
    SWEEP_GRID["num_layers"],
    SWEEP_GRID["beta"],
    SWEEP_GRID["encoding"],
))
```

### Config Construction per Sweep Run

```python
def _make_sweep_config(
    hidden_size: int, num_layers: int, beta: float,
    encoding: str, seed: int, config_id: int,
    base_data_cfg: dict,
) -> dict:
    return {
        "experiment": {"name": f"spike_gru_sweep_{config_id:03d}", "seed": seed},
        "data": base_data_cfg,
        "model": {
            "type": "spike_gru",
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "beta": beta,
            "encoding": encoding,
            "timesteps": 10,
            "dropout": 0.0,
            "surrogate": "fast_sigmoid",
        },
        "training": {
            "epochs": 100,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "early_stopping_patience": 10,
            "early_stopping_metric": "val_recall_at_20",
        },
        "output": {"dir": f"results/phase_b_sweep_{config_id:03d}"},
        "log_level": "INFO",
    }
```

### Phase 1: Screening (Single Seed)

```python
sweep_results = []
screening_seed = 42

for i, (h, n, b, e) in enumerate(combos):
    click.echo(f"[{i+1}/{len(combos)}] spike_gru h={h} l={n} b={b:.2f} enc={e}")
    set_global_seed(screening_seed)
    config = _make_sweep_config(h, n, b, e, screening_seed, i, base_data_cfg)
    model = get_model(config)
    trainer = Trainer(model, config, dataloaders, device)

    t0 = time.time()
    result = trainer.run()
    elapsed = time.time() - t0

    # Evaluate on validation set for screening
    val_eval = evaluate_model(model, dataloaders["val"], device)

    sweep_results.append({
        "config_id": i,
        "hidden_size": h,
        "num_layers": n,
        "beta": b,
        "encoding": e,
        "timesteps": 10 if e == "rate_coded" else 1,
        "val_recall_at_20": val_eval["metrics"]["recall_at_20"],
        "val_hit_at_20": val_eval["metrics"]["hit_at_20"],
        "val_mrr": val_eval["metrics"]["mrr"],
        "training_time_s": round(elapsed, 1),
        "best_epoch": result["best_epoch"],
    })
```

### Phase 2: Top-K Multi-Seed Re-Run

```python
# Sort by val_recall_at_20, take top-K
sorted_results = sorted(sweep_results, key=lambda r: r["val_recall_at_20"], reverse=True)
top_configs = sorted_results[:top_k]

# Re-run each top config with 3 seeds, evaluate on TEST set
model_results = []
for rank, top in enumerate(top_configs):
    seed_metrics = []
    total_time = 0.0
    for seed in seed_list:
        set_global_seed(seed)
        config = _make_sweep_config(
            top["hidden_size"], top["num_layers"], top["beta"],
            top["encoding"], seed, top["config_id"], base_data_cfg,
        )
        config["output"]["dir"] = f"results/phase_b_top{rank+1}_seed{seed}"
        model = get_model(config)
        trainer = Trainer(model, config, dataloaders, device)
        t0 = time.time()
        trainer.run()
        total_time += time.time() - t0

        test_eval = evaluate_model(model, test_loader, device)
        seed_metrics.append(test_eval["metrics"])

    model_results.append({
        "name": f"spike_gru_top{rank+1}",
        "type": "learned",
        "phase": "phase_b",
        "seed_metrics": seed_metrics,
        "training_time_s": round(total_time, 1),
        "environment": "local",
        "config": {k: top[k] for k in ["hidden_size", "num_layers", "beta", "encoding"]},
    })
```

### Expected Sweep CSV Format

```csv
config_id,hidden_size,num_layers,beta,encoding,timesteps,val_recall_at_20,val_hit_at_20,val_mrr,training_time_s,best_epoch
0,64,1,0.50,direct,1,0.5102,0.9780,0.3050,12.3,35
1,64,1,0.50,rate_coded,10,0.5220,0.9810,0.3180,85.4,28
...
35,256,2,0.95,rate_coded,10,0.5180,0.9790,0.3120,142.1,22
```

### Expected Console Output

```
Phase B Spike-GRU HP Sweep
===========================

Phase 1: Screening (36 configs, seed=42)
[1/36] spike_gru h=64 l=1 b=0.50 enc=direct → val_recall@20=0.5102 (12.3s)
[2/36] spike_gru h=64 l=1 b=0.50 enc=rate_coded → val_recall@20=0.5220 (85.4s)
...
[36/36] spike_gru h=256 l=2 b=0.95 enc=rate_coded → val_recall@20=0.5180 (142.1s)

Sweep results saved to: results/phase_b_sweep.csv

Screening Leaderboard (top 10):
Rank  Config  Hidden  Layers  Beta  Encoding     val_R@20  Time(s)
  1      17     128       1  0.80  rate_coded    0.5280    90.2
  2       5     64        2  0.50  rate_coded    0.5250    72.1
  ...

Phase 2: Top-3 re-run with 3 seeds (42, 123, 7)
[1/3] Top-1 config: h=128, l=1, b=0.80, enc=rate_coded
  Seed 42: test_recall@20=0.5260 (88.1s)
  Seed 123: test_recall@20=0.5240 (91.3s)
  Seed 7: test_recall@20=0.5270 (87.5s)
...

Top-3 Multi-Seed Results:
Model           Hidden  Layers  Beta  Encoding     Recall@20        Hit@20
spike_gru_top1  128     1       0.80  rate_coded   0.5257±0.002    0.9810±0.001
spike_gru_top2  64      2       0.50  rate_coded   0.5230±0.003    0.9795±0.002
spike_gru_top3  128     2       0.95  rate_coded   0.5210±0.004    0.9788±0.002

Results saved to: results/phase_b_top3.json
Best checkpoint: results/phase_b_best/
```

### Timing Estimates

- **Phase 1 (36 configs, single seed):** 18 direct (~15s each) + 18 rate_coded (~90s each) = ~32 min
- **Phase 2 (top-3, 3 seeds each):** 9 runs × ~90s = ~14 min (assuming top-3 are rate_coded)
- **Total estimated:** ~45-60 min locally
- **RunPod threshold:** Per the Trainer's timing probe, individual runs projecting >20 min will log warnings. Most runs will complete in 1-3 min with early stopping.

### Edge Cases

- **All 36 configs produce similar results:** This would confirm that the architecture (rather than hyperparameters) is the limiting factor. Document this finding.
- **Rate-coded configs all fail to converge:** Early stopping will terminate; report whatever metrics are achieved.
- **OOM with hidden_size=256 + rate_coded:** 256-dim hidden with T=10 and W=21 uses significant memory. If OOM, reduce batch_size to 32 for that config or skip.
- **Sweep takes >2 hours:** The Trainer's timing probe warns per-run; cumulative time may be acceptable locally. If not, reduce grid (e.g., drop hidden_size=256 or beta=0.5).
- **All top-3 are rate_coded:** This would strongly validate the encoding hypothesis from Phase A.
- **All top-3 are direct:** This would suggest recurrence matters more than encoding.

### Architecture References

- Section 4.4: Experiment config schema
- Section 4.7: Comparison report schema
- Section 5.2: Models module — Phase B models
- Section 13.3 Rule #4: Config immutable at runtime
- Section 13.4: snnTorch guidance

---

## Dependencies

**Prerequisite Stories:**
- STORY-5.1: Spike-GRU Architecture (SpikeGRU model class and registry)
- STORY-3.3: Training Loop & train CLI (Trainer class)
- STORY-3.4: Baseline Results & Comparison (compare framework)
- STORY-4.4: Phase A Training & Comparison (phase-a command pattern)

**Blocked Stories:**
- STORY-5.3: Phase B Evaluation & Cumulative Comparison (needs top-3 results from sweep)

**External Dependencies:**
- snnTorch 0.9.1 installed
- CA5 data file at `data/raw/CA5_matrix_binary.csv`

---

## Definition of Done

- [ ] `phase-b-sweep` CLI command implemented in `src/c5_snn/cli.py`
- [ ] Command runs 36-config sweep (Phase 1) + top-K multi-seed re-run (Phase 2)
- [ ] `results/phase_b_sweep.csv` generated with all 36 configs and metrics
- [ ] `results/phase_b_top3.json` generated with top-3 multi-seed results
- [ ] Best checkpoint saved to `results/phase_b_best/`
- [ ] Console prints progress, screening leaderboard, and top-K comparison table
- [ ] Unit tests in `tests/test_compare.py`:
  - [ ] `phase-b-sweep` command exists and has correct help text
  - [ ] `--output`, `--top-k`, and `--seeds` options work
- [ ] `ruff check src/ tests/` passes with zero errors
- [ ] `pytest tests/ -v` passes (all existing + new tests)
- [ ] Acceptance criteria validated (all checked)
- [ ] Code committed to `main` branch and pushed

---

## Story Points Breakdown

- **CLI command implementation (sweep grid + two-phase loop):** 2.0 points
- **CSV/JSON output generation and progress reporting:** 1.0 points
- **Running the actual sweep and analyzing results:** 1.5 points
- **Testing (CLI tests + manual validation):** 0.5 points
- **Total:** 5 points

**Rationale:** The sweep command is structurally similar to `phase-a` but larger — 36 configs vs 4 models, plus a two-phase approach (screening + top-K re-run). The infrastructure (Trainer, compare, evaluate) is all built. Most effort is the sweep execution time (~45-60 min) and analyzing the results. 5 points matches the sprint plan estimate.

---

## Additional Notes

- The sweep uses `val_recall_at_20` for screening (Phase 1) but `test set` metrics for the top-K re-run (Phase 2). This prevents data leakage — the validation set selects the architecture, the test set provides the final unbiased estimate.
- Fixed parameters across all sweep runs: `learning_rate=0.001`, `optimizer=adam`, `early_stopping_patience=10`, `dropout=0.0`, `timesteps=10` (for rate_coded), `window_size=21`, `batch_size=64`.
- The encoding dimension (`direct` vs `rate_coded`) is expected to be the most impactful hyperparameter based on Phase A findings. If all top-3 configs are rate_coded, this strongly validates the encoding hypothesis.
- Per-sweep-run output directories (`results/phase_b_sweep_NNN/`) are created by the Trainer but can be cleaned up after the sweep CSV is saved. Only the top-K and best checkpoint directories need to persist.
- The `phase_b_top3.json` follows the same schema as `phase_a_comparison.json` (Section 4.7) with `phase: "phase_b"`, making it directly compatible with STORY-5.3's cumulative comparison.

---

## Progress Tracking

**Status History:**
- 2026-02-11: Created by Scrum Master (AI)
- 2026-02-11: Started by Developer (AI)
- 2026-02-11: Completed by Developer (AI)

**Actual Effort:** 5 points (matched estimate)

**Implementation Notes:**
- Added `phase-b-sweep` CLI command to `src/c5_snn/cli.py` (~250 lines)
- Two-phase approach: Phase 1 screens 36 configs (single seed=42), Phase 2 re-runs top-3 with 3 seeds
- Sweep grid: hidden_size [64,128,256] x num_layers [1,2] x beta [0.5,0.8,0.95] x encoding [direct,rate_coded]
- CSV output (`results/phase_b_sweep.csv`) with 36 rows, 11 columns
- JSON comparison (`results/phase_b_top3.json`) with top-3 multi-seed results
- Best checkpoint saved to `results/phase_b_best/`
- 5 new tests (405 total), all passing
- All 15 acceptance criteria validated

**Sweep Results Summary:**
- Total sweep time: ~2.5 hours (Phase 1 ~2h, Phase 2 ~30 min)
- Top-8 configs all achieved val_recall@20=0.5201 (all h=256, beta>=0.80)
- Test set Recall@20 for all top-3: 0.5137 (std=0.000 across 3 seeds)
- Direct and rate_coded encoding produce equivalent results
- hidden_size=256 > 128 > 64 (strongest hyperparameter effect)
- num_layers: 1 and 2 layers both reach 0.5201 for h=256
- Spike-GRU test Recall@20 (0.5137) slightly above GRU baseline (0.510) and Phase A SNNs (~0.51)

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
