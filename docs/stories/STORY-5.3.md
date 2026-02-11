# STORY-5.3: Phase B Evaluation & Cumulative Comparison

**Epic:** Epic 5 — SNN Phase B
**Priority:** Must Have
**Story Points:** 3
**Status:** Completed
**Assigned To:** ai_dev_agent
**Created:** 2026-02-11
**Sprint:** 5

---

## User Story

As a researcher,
I want the best Spike-GRU compared against all prior models,
so that I can quantify the value of recurrence and inform Phase C.

---

## Description

### Background

Sprint 5 has delivered the Spike-GRU architecture (STORY-5.1) and a 36-config hyperparameter sweep (STORY-5.2). The sweep results are captured in `results/phase_b_top3.json` (top-3 configs with 3 seeds each) and `results/phase_b_sweep.csv` (all 36 screening results).

Phase A comparison (`results/phase_a_comparison.json`) already contains 4 models:
- `frequency_baseline`: Recall@20 = 0.5232 (heuristic, 1 seed)
- `gru_baseline`: Recall@20 = 0.5099 +/- 0.003 (3 seeds)
- `spiking_mlp`: Recall@20 = 0.5125 +/- 0.003 (3 seeds)
- `spiking_cnn1d`: Recall@20 = 0.5152 +/- 0.002 (3 seeds)

Phase B sweep found the best Spike-GRU at:
- `spike_gru_top1` (h=256, l=1, b=0.80, direct): Test Recall@20 = 0.5137 +/- 0.000 (3 seeds)

This story merges Phase A and Phase B results into a single cumulative leaderboard, adds a CLI command for easy regeneration, and documents findings to inform Sprint 6 (Spiking Transformer).

### Scope

**In scope:**
- New CLI command `c5_snn phase-b` that produces the cumulative comparison
- Merge Phase A results (4 models) + Phase B best Spike-GRU (1 model) into `results/cumulative_comparison.json`
- Console prints full leaderboard sorted by Recall@20
- Analysis summary printed: does Spike-GRU beat Phase A? Beat GRU baseline?
- Update `docs/project-memory.md` with Phase B findings and Phase C recommendations
- Unit tests for the `phase-b` CLI command

**Out of scope:**
- Re-training any models (use existing JSON results from Phase A and Phase B)
- Spiking Transformer (Sprint 6)
- Modifying `compare.py` or existing comparison infrastructure
- Full statistical significance tests (just mean +/- std comparison)

### User Flow

1. Researcher ensures `results/phase_a_comparison.json` and `results/phase_b_top3.json` exist
2. Researcher runs `c5_snn phase-b --output results/cumulative_comparison.json`
3. CLI loads both JSON files, merges models into a unified list
4. Selects the best Spike-GRU variant (highest mean test recall@20) from Phase B top-3
5. Builds cumulative comparison report using `build_comparison()`
6. Saves to `results/cumulative_comparison.json`
7. Console prints full leaderboard (5 models) sorted by Recall@20
8. Console prints analysis summary with key findings

---

## Acceptance Criteria

- [ ] New CLI command `c5_snn phase-b` exists and runs end-to-end
- [ ] Command accepts `--output` (default `"results/cumulative_comparison.json"`) and `--phase-a` (default `"results/phase_a_comparison.json"`) and `--phase-b-top` (default `"results/phase_b_top3.json"`) options
- [ ] Loads Phase A comparison JSON and Phase B top-3 JSON without re-training
- [ ] Selects the single best Spike-GRU config from Phase B top-3 (by mean recall@20)
- [ ] `results/cumulative_comparison.json` contains 5 models: frequency_baseline, gru_baseline, spiking_mlp, spiking_cnn1d, spike_gru (best)
- [ ] Output JSON follows existing comparison schema (models, generated_at, window_size, test_split_size)
- [ ] Console prints leaderboard sorted by Recall@20 (descending)
- [ ] Console prints analysis section: (a) best overall model, (b) does Spike-GRU beat GRU baseline?, (c) does it beat Phase A SNNs?, (d) encoding finding, (e) recommendation for Phase C
- [ ] `docs/project-memory.md` updated with Phase B results and Phase C recommendations
- [ ] Unit tests in `tests/test_compare.py`:
  - [ ] `phase-b` command exists with correct help text
  - [ ] `--output`, `--phase-a`, `--phase-b-top` options work
- [ ] `ruff check src/ tests/` passes with zero errors
- [ ] `pytest tests/ -v` passes (all existing + new tests)

---

## Technical Notes

### Components

- **Modified file:** `src/c5_snn/cli.py` — add `phase-b` CLI command
- **Modified file:** `tests/test_compare.py` — add `phase-b` command tests
- **Modified file:** `docs/project-memory.md` — Phase B findings
- **New output:** `results/cumulative_comparison.json`
- **Reused (no changes):** `src/c5_snn/training/compare.py` — `build_comparison`, `format_comparison_table`, `save_comparison`
- **Input files:** `results/phase_a_comparison.json`, `results/phase_b_top3.json`

### CLI Command Design

```python
@cli.command("phase-b")
@click.option("--output", "output_path", default="results/cumulative_comparison.json",
              help="Path for cumulative comparison JSON.")
@click.option("--phase-a", "phase_a_path", default="results/phase_a_comparison.json",
              help="Path to Phase A comparison JSON.")
@click.option("--phase-b-top", "phase_b_path", default="results/phase_b_top3.json",
              help="Path to Phase B top-3 JSON.")
def phase_b(output_path: str, phase_a_path: str, phase_b_path: str) -> None:
    """Build cumulative comparison: baselines + Phase A + Phase B."""
```

### Implementation Strategy

The key insight is that we do **not** need to re-train anything. Both Phase A and Phase B results already exist in JSON files that follow the Section 4.7 comparison schema. The `phase-b` command simply:

1. **Loads** both JSON files
2. **Merges** the model lists — Phase A contributes 4 models, Phase B contributes the best 1
3. **Selects** the best Spike-GRU from Phase B top-3 by highest `metrics_mean["recall_at_20"]`
4. **Renames** the selected model to `"spike_gru"` (dropping the `_top1` suffix) for clean leaderboard
5. **Reconstructs** the model_results list in the format `build_comparison()` expects:

```python
# Each model from existing JSON already has metrics_mean/metrics_std/n_seeds.
# We can pass these directly into the cumulative report WITHOUT re-computing.

cumulative_models = []

# Phase A models: copy directly from phase_a_comparison.json["models"]
for model in phase_a_report["models"]:
    cumulative_models.append(model)

# Phase B best: select from phase_b_top3.json["models"]
best_b = max(phase_b_report["models"],
             key=lambda m: m["metrics_mean"]["recall_at_20"])
best_b["name"] = "spike_gru"  # Clean name
cumulative_models.append(best_b)

# Build cumulative report (models already have metrics_mean/std)
report = {
    "models": cumulative_models,
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "window_size": phase_a_report["window_size"],
    "test_split_size": phase_a_report["test_split_size"],
}
```

Note: Since models already have `metrics_mean` and `metrics_std` computed, we do NOT need to call `build_comparison()` (which expects raw `seed_metrics` lists). Instead, assemble the report dict directly.

### Expected Console Output

```
Phase B Cumulative Comparison
=============================

Loading Phase A results: results/phase_a_comparison.json (4 models)
Loading Phase B results: results/phase_b_top3.json (3 configs)
Selected best Spike-GRU: h=256, l=1, b=0.80, enc=direct

Cumulative Leaderboard (sorted by Recall@20):
Model                     Recall@20       Hit@20          MRR            Seeds
---------------------------------------------------------------------------
frequency_baseline        0.5232          0.9840          0.3125         1
spiking_cnn1d             0.5152+/-0.002  0.9795+/-0.002  0.3053+/-0.001 3
spike_gru                 0.5137+/-0.000  0.9812+/-0.000  0.3115+/-0.000 3
spiking_mlp               0.5125+/-0.003  0.9759+/-0.001  0.3101+/-0.007 3
gru_baseline              0.5099+/-0.003  0.9789+/-0.003  0.3103+/-0.002 3

Analysis:
  Best learned model:     spike_gru (Recall@20=0.5137)
  vs GRU baseline:        +0.0038 (+0.75%) — slight improvement
  vs SpikingCNN1D:        -0.0015 (-0.29%) — slightly below best Phase A
  vs SpikingMLP:          +0.0012 (+0.23%) — slight improvement
  Encoding finding:       direct == rate_coded for Spike-GRU (no benefit from T>1)
  Recurrence finding:     Spiking recurrence adds marginal value vs feedforward SNNs
  Phase C recommendation: Spiking Transformer with attention may capture patterns
                          that recurrence alone cannot. Consider window size tuning.

Results saved to: results/cumulative_comparison.json
```

### Analysis Logic

```python
# Find key models for comparison
models_by_name = {m["name"]: m for m in cumulative_models}
spike_gru = models_by_name["spike_gru"]
gru = models_by_name["gru_baseline"]
best_phase_a = max(
    (m for m in cumulative_models if m.get("phase") == "phase_a"),
    key=lambda m: m["metrics_mean"]["recall_at_20"],
)

# Compute deltas
gru_delta = spike_gru["metrics_mean"]["recall_at_20"] - gru["metrics_mean"]["recall_at_20"]
phase_a_delta = spike_gru["metrics_mean"]["recall_at_20"] - best_phase_a["metrics_mean"]["recall_at_20"]
```

### project-memory.md Updates

Add to the Phase B section:
- Spike-GRU best config: h=256, l=1, b=0.80, encoding=direct
- Test Recall@20: 0.5137 (consistent across 3 seeds, std=0.000)
- Ranking: 3rd among learned models (behind frequency_baseline and spiking_cnn1d)
- Encoding finding: direct and rate_coded produce identical results for Spike-GRU
- hidden_size is the dominant hyperparameter (256 >> 128 >> 64)
- Phase C recommendation: try Spiking Transformer + window size tuning

### Edge Cases

- **Phase A JSON missing:** Print error and exit non-zero
- **Phase B JSON missing:** Print error and exit non-zero
- **All Phase B top-3 have identical recall:** Pick the first (fastest training time)
- **Phase A and Phase B have different window_size/test_split_size:** This should not happen (same data pipeline), but validate and warn if mismatched

### Architecture References

- Section 4.7: Comparison report schema
- Section 5.3: Training module — compare.py
- Section 7.1: Output directory structure — `results/cumulative_comparison.json`

---

## Dependencies

**Prerequisite Stories:**
- STORY-5.1: Spike-GRU Architecture (SpikeGRU model class)
- STORY-5.2: HP Sweep (produces `results/phase_b_top3.json`)
- STORY-4.4: Phase A Training & Comparison (produces `results/phase_a_comparison.json`)
- STORY-3.4: Baseline Results & Comparison (comparison framework)

**Blocked Stories:**
- STORY-6.1: Spiking Transformer Architecture (uses Phase B findings to inform design)

**External Dependencies:**
- None (all inputs are local JSON files)

---

## Definition of Done

- [ ] `phase-b` CLI command implemented in `src/c5_snn/cli.py`
- [ ] Loads Phase A and Phase B JSON results without re-training
- [ ] `results/cumulative_comparison.json` generated with 5 models
- [ ] Console prints sorted leaderboard and analysis summary
- [ ] `docs/project-memory.md` updated with Phase B findings and Phase C recommendations
- [ ] Unit tests in `tests/test_compare.py`:
  - [ ] `phase-b` command exists and has correct help text
  - [ ] `--output`, `--phase-a`, `--phase-b-top` options work
- [ ] `ruff check src/ tests/` passes with zero errors
- [ ] `pytest tests/ -v` passes (all existing + new tests)
- [ ] Acceptance criteria validated (all checked)
- [ ] Code committed to `main` branch and pushed

---

## Story Points Breakdown

- **CLI command implementation (load + merge + format):** 1.0 points
- **Analysis logic and console output:** 0.5 points
- **project-memory.md update:** 0.5 points
- **Testing:** 0.5 points
- **Validation and cleanup:** 0.5 points
- **Total:** 3 points

**Rationale:** This story is primarily an assembly task — no model training, no new algorithms. It loads two existing JSON files, merges them, and produces a leaderboard. The analysis section adds modest complexity. 3 points matches the sprint plan estimate and is appropriate for the scope.

---

## Additional Notes

- The `phase-b` command does NOT re-train any models. It reads existing result JSON files. This makes it fast (<1 second) and idempotent.
- The cumulative comparison includes exactly 5 models: 2 baselines (frequency, GRU) + 2 Phase A SNNs (MLP, CNN1D) + 1 Phase B SNN (Spike-GRU best). Future stories (Sprint 6) will add the Spiking Transformer.
- The `format_comparison_table()` from `compare.py` already handles the table formatting. The analysis section is new console output specific to this command.
- If all models cluster near 0.51 Recall@20 (which they do), the analysis should honestly acknowledge this and focus on what Phase C (Spiking Transformer + window tuning) might change.
- The cumulative_comparison.json follows the exact same schema as phase_a_comparison.json, making it directly compatible with Sprint 6's final comparison (STORY-6.4).
- Since we are assembling pre-computed results rather than calling `build_comparison()` with raw seed_metrics, the report dict is constructed directly. This is fine because all input models already have `metrics_mean`, `metrics_std`, and `n_seeds`.

---

## Progress Tracking

**Status History:**
- 2026-02-11: Created by Scrum Master (AI)
- 2026-02-11: Started by Developer (AI)
- 2026-02-11: Completed by Developer (AI)

**Actual Effort:** 3 points (matched estimate)

**Implementation Notes:**
- Added `phase-b` CLI command to `src/c5_snn/cli.py` (~150 lines)
- Loads Phase A + Phase B JSON files, merges into cumulative report (no re-training)
- Selects best Spike-GRU (top1: h=256, l=1, b=0.80, direct, Recall@20=0.5137)
- Cumulative leaderboard: 5 models sorted by Recall@20
- Analysis section: vs GRU (+0.75%), vs CNN1D (-0.29%), encoding/recurrence findings
- 8 new tests (413 total), all passing
- All 12 acceptance criteria validated
- `results/cumulative_comparison.json` generated with correct schema
- `docs/project-memory.md` updated with Phase B findings and Phase C recommendations

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
