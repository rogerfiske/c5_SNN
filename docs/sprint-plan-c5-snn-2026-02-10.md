# Sprint Plan: c5_SNN

**Date:** 2026-02-10
**Scrum Master:** Steve (AI)
**Project Level:** 2
**Total Stories:** 22
**Total Points:** 103
**Planned Sprints:** 7 (one-week sprints)
**Team:** 1 AI dev agent
**Target Completion:** Week of 2026-03-31

---

## Executive Summary

The c5_SNN project is a 7-week implementation plan delivering a complete SNN time-series forecasting pipeline. Sprints 1-2 build the foundation (scaffold, data pipeline, evaluation harness). Sprint 3 establishes baseline benchmarks. Sprints 4-6 progressively implement three SNN architecture phases (MLP/CNN, Spike-GRU, Spiking Transformer). Sprint 7 produces the final comparison report and reproducible runbook.

**Key Metrics:**

- Total Stories: 22
- Total Points: 103
- Sprints: 7 (1-week each)
- Team Capacity: ~18-22 points per sprint
- Target Completion: Week 7

---

## Story Inventory

### Epic 1: Foundation & Data Validation (13 pts)

#### STORY-1.1: Repository & Package Scaffold

**Epic:** Epic 1 — Foundation & Data Validation
**Priority:** Must Have
**Points:** 3

**User Story:**
As a developer,
I want a properly initialized git repository with a pip-installable Python package structure,
so that all subsequent work has a clean, standard foundation to build on.

**Acceptance Criteria:**

- [ ] Git repository initialized with `.gitignore` (Python, IDE files, data artifacts, `__pycache__`, `.mypy_cache`)
- [ ] `pyproject.toml` with project metadata (`c5-snn`, version `0.1.0`), Python 3.11+ requirement, and `[project.dependencies]` including `torch`, `snntorch`, `pandas`, `numpy`, `pytest`
- [ ] Package installable via `pip install -e .` with `src/c5_snn/` layout and `__init__.py`
- [ ] Directory structure created per architecture Section 10 (Source Tree)
- [ ] `README.md` with project name, one-line description, and setup instructions
- [ ] Initial commit pushed to `https://github.com/rogerfiske/c5_SNN`

**Technical Notes:**
Use pinned versions from architecture Section 3.2. Include ROCm install instructions in README. Smoke test `import torch; print(torch.cuda.is_available())` with `HSA_OVERRIDE_GFX_VERSION=10.3.0` on local RX 6600M.

**Dependencies:** None (first story)

---

#### STORY-1.2: Logging, Seed Management & Configuration Infrastructure

**Epic:** Epic 1 — Foundation & Data Validation
**Priority:** Must Have
**Points:** 3

**User Story:**
As a developer,
I want centralized logging and deterministic seed management available from the start,
so that every subsequent story has reproducibility and observability built in from day one.

**Acceptance Criteria:**

- [ ] `src/c5_snn/utils/seed.py` provides `set_global_seed(seed: int)` setting `random`, `numpy`, `torch.manual_seed`, `torch.cuda.manual_seed_all`
- [ ] `src/c5_snn/utils/logging.py` provides `setup_logging(level, log_file)` configuring Python logging with timestamped output
- [ ] `src/c5_snn/utils/config.py` provides `load_config(path) -> dict` for YAML parsing
- [ ] `src/c5_snn/utils/device.py` provides `get_device() -> torch.device` with CPU/CUDA/ROCm auto-detection
- [ ] Base config `configs/default.yaml` with `seed`, `log_level`, placeholder sections
- [ ] Unit tests verify seed determinism and logging output
- [ ] All utilities importable via `from c5_snn.utils import ...`

**Technical Notes:**
Follow architecture Section 5.5 (Utils Module) and coding standard #3 (seed before everything).

**Dependencies:** STORY-1.1

---

#### STORY-1.3: Data Validation CLI Command

**Epic:** Epic 1 — Foundation & Data Validation
**Priority:** Must Have
**Points:** 5

**User Story:**
As a researcher,
I want to run `python -m c5_snn.cli validate-data` and get a clear pass/fail report on the raw dataset,
so that I can confirm data integrity before any modeling work begins.

**Acceptance Criteria:**

- [ ] CLI entry point at `src/c5_snn/cli.py` using Click, with `validate-data` subcommand accepting `--data-path` (default: `data/raw/CA5_matrix_binary.csv`)
- [ ] Validation checks: file loads, 45 columns present, dates monotonic, `sum(P_1..P_39)==5` per row, m/P cross-check
- [ ] CLI outputs summary: total rows, date range, unique parts, pass/fail per check
- [ ] Non-zero exit code on any validation failure
- [ ] Unit tests using fixture CSV: valid passes, missing column fails, broken row-sum fails, m/P mismatch fails
- [ ] Uses logging from STORY-1.2

**Technical Notes:**
Follow architecture Section 5.1 (Data Module) and Section 7.1 (validate-data workflow). Implements FR1.

**Dependencies:** STORY-1.2

---

#### STORY-1.4: CI Skeleton & First Green Build

**Epic:** Epic 1 — Foundation & Data Validation
**Priority:** Must Have
**Points:** 2

**User Story:**
As a developer,
I want a GitHub Actions workflow that runs linting and tests on every push,
so that code quality is enforced from the first commit.

**Acceptance Criteria:**

- [ ] `.github/workflows/ci.yml` runs on push and PR to `main`
- [ ] Steps: checkout, Python 3.11, `pip install -e ".[dev]"` (CPU torch), `ruff check src/`, `pytest tests/ -v`
- [ ] Dev dependencies (`pytest`, `ruff`, `pyright`) in `pyproject.toml` `[project.optional-dependencies.dev]`
- [ ] CI passes on current codebase
- [ ] Status check visible on GitHub repo

**Technical Notes:**
Follow architecture Section 11.2 (CI Pipeline). Use CPU-only PyTorch wheel in CI.

**Dependencies:** STORY-1.3

---

### Epic 2: Data Pipeline & Evaluation Harness (16 pts)

#### STORY-2.1: Windowed Tensor Construction

**Epic:** Epic 2 — Data Pipeline & Evaluation Harness
**Priority:** Must Have
**Points:** 5

**User Story:**
As a researcher,
I want validated raw data transformed into windowed tensors of shape `(W, 39)` with corresponding next-event targets,
so that I have training-ready inputs for all models.

**Acceptance Criteria:**

- [ ] `src/c5_snn/data/windowing.py` produces `X (N, W, 39)` and `y (N, 39)` from validated DataFrame
- [ ] `X[t]` = events `[t..t+W-1]`, `y[t]` = event `t+W` — no leakage
- [ ] Window size `W` configurable via YAML (default 21, range 7-90)
- [ ] Tensors saved as `.pt` files with `tensor_meta_w{W}.json` metadata
- [ ] Unit tests: correct shapes, first/last window alignment, no input-target overlap, empty result for tiny data

**Technical Notes:**
Follow architecture Section 4.2 (Windowed Tensor) and coding standard #1 (no data leakage).

**Dependencies:** STORY-1.3

---

#### STORY-2.2: Time-Based Train/Validation/Test Splits

**Epic:** Epic 2 — Data Pipeline & Evaluation Harness
**Priority:** Must Have
**Points:** 3

**User Story:**
As a researcher,
I want the windowed samples split into train/val/test sets by chronological order,
so that evaluation is temporally valid with zero data leakage.

**Acceptance Criteria:**

- [ ] `src/c5_snn/data/splits.py` splits by time order, configurable ratios (default 70/15/15)
- [ ] Split indices persisted to `data/processed/splits.json` with ratios, index ranges, date ranges
- [ ] Deterministic — same config produces identical splits
- [ ] No shuffling — train < val < test indices
- [ ] Convenience function returns PyTorch DataLoaders per split
- [ ] Unit tests: no overlap, strict ordering, JSON round-trip, ratio changes valid

**Technical Notes:**
Follow architecture Section 4.3 (Split Index). Implements FR3.

**Dependencies:** STORY-2.1

---

#### STORY-2.3: Evaluation Harness & Metrics

**Epic:** Epic 2 — Data Pipeline & Evaluation Harness
**Priority:** Must Have
**Points:** 5

**User Story:**
As a researcher,
I want a reusable evaluation module that computes all defined metrics given model predictions and ground truth,
so that every model is measured consistently.

**Acceptance Criteria:**

- [ ] `src/c5_snn/training/metrics.py` provides: Recall@K, Hit@K (K=5,20), MRR
- [ ] `src/c5_snn/training/evaluate.py` provides `evaluate_model(model, dataloader, device) -> MetricsDict`
- [ ] Results exported as JSON summary + per-sample CSV
- [ ] Device-agnostic via `torch.device`
- [ ] Unit tests: perfect=1.0, zero=0.0, known partial overlap, MRR hand-computed

**Technical Notes:**
Follow architecture Section 5.3 (Training Module). Implements FR9.

**Dependencies:** STORY-2.2

---

#### STORY-2.4: CLI evaluate Command Integration

**Epic:** Epic 2 — Data Pipeline & Evaluation Harness
**Priority:** Must Have
**Points:** 3

**User Story:**
As a researcher,
I want to run `python -m c5_snn.cli evaluate --checkpoint <path>` and get a printed metrics summary plus saved report files,
so that I can assess any trained model with a single command.

**Acceptance Criteria:**

- [ ] `evaluate` subcommand added to CLI, accepting `--checkpoint` and `--output-dir`
- [ ] Loads checkpoint, loads test split, runs evaluation harness, prints metrics table
- [ ] JSON and CSV reports saved to output directory
- [ ] Non-zero exit code if checkpoint not found
- [ ] Integration test: dummy model, verify CLI produces output files with correct structure

**Technical Notes:**
Follow architecture Section 7.3 (evaluate workflow).

**Dependencies:** STORY-2.3

---

### Epic 3: Baseline Models (19 pts)

#### STORY-3.1: Frequency/Recency Heuristic Baseline

**Epic:** Epic 3 — Baseline Models
**Priority:** Must Have
**Points:** 3

**User Story:**
As a researcher,
I want a non-learned baseline that ranks parts by frequency and recency within the input window,
so that I have a simple lower bound for comparison.

**Acceptance Criteria:**

- [ ] `FrequencyBaseline` in `src/c5_snn/models/baselines.py` with `predict(X) -> scores`
- [ ] Scoring: weighted frequency + exponentially-decayed recency
- [ ] Conforms to `BaseModel` interface: `(batch, W, 39) -> (batch, 39)`
- [ ] Registered in `MODEL_REGISTRY` as `"frequency_baseline"`
- [ ] Unit tests: dominant part ranked highest, recency weighting correct, output shape valid

**Technical Notes:**
Follow architecture Section 5.2 (Models Module). Implements FR4. Also create `base.py` with `BaseModel` ABC and `MODEL_REGISTRY` dict.

**Dependencies:** STORY-2.3

---

#### STORY-3.2: ANN Baseline — GRU Model

**Epic:** Epic 3 — Baseline Models
**Priority:** Must Have
**Points:** 5

**User Story:**
As a researcher,
I want a conventional GRU-based neural network baseline,
so that I have a strong learned-model benchmark for SNN comparison.

**Acceptance Criteria:**

- [ ] `GRUBaseline(nn.Module)` in `baselines.py` with configurable hidden size, layers, dropout
- [ ] Architecture: GRU encoder `(W, 39)` -> final hidden -> linear -> 39 logits
- [ ] Uses `BCEWithLogitsLoss`, Adam optimizer, configurable via YAML
- [ ] Conforms to `BaseModel` interface, registered as `"gru_baseline"`
- [ ] Device-agnostic via `torch.device`
- [ ] Unit test: forward `(4, 21, 39)` -> `(4, 39)`, backward completes

**Technical Notes:**
Config file `configs/baseline_gru.yaml`. Implements FR4.

**Dependencies:** STORY-3.1 (needs BaseModel/registry)

---

#### STORY-3.3: Training Loop & train CLI Command

**Epic:** Epic 3 — Baseline Models
**Priority:** Must Have
**Points:** 8

**User Story:**
As a researcher,
I want to run `python -m c5_snn.cli train --config configs/baseline_gru.yaml` and get a trained model,
so that I can train any model with a single command.

**Acceptance Criteria:**

- [ ] `Trainer` class in `src/c5_snn/training/trainer.py` accepting any `BaseModel`
- [ ] Training loop: epoch -> train step -> val metrics -> early stopping -> checkpoint on best val Recall@20
- [ ] Per-epoch metrics logged to CSV + console
- [ ] `train` subcommand added to CLI accepting `--config`
- [ ] 2-epoch timing probe: if projected >20 min, log RunPod warning
- [ ] `pip freeze` captured to experiment output dir (NFR1)
- [ ] Config snapshot saved alongside checkpoint (FR11)
- [ ] Integration test: train GRU 2 epochs on 100 rows, verify checkpoint + metrics exist

**Technical Notes:**
Follow architecture Sections 5.3 and 7.2. This is the highest-value infrastructure story — unlocks all model training. Implements FR8, FR11, NFR1, NFR3.

**Dependencies:** STORY-3.2, STORY-2.4

---

#### STORY-3.4: Baseline Results & First Comparison

**Epic:** Epic 3 — Baseline Models
**Priority:** Must Have
**Points:** 3

**User Story:**
As a researcher,
I want documented baseline benchmarks on the full dataset,
so that I have quantified targets before starting SNN work.

**Acceptance Criteria:**

- [ ] Both models evaluated on identical test split
- [ ] `results/baseline_comparison.json` with all metrics per model
- [ ] GRU trained with 3 seeds, report mean +/- std
- [ ] Console prints formatted comparison table
- [ ] `project-memory.md` updated with baseline results

**Technical Notes:**
Follow architecture Section 5.3 (`compare.py`). If GRU training >20 min locally, use RunPod.

**Dependencies:** STORY-3.3

---

### Epic 4: SNN Phase A — Spiking MLP/CNN (18 pts)

#### STORY-4.1: snnTorch Integration & Spike Encoding Layer

**Epic:** Epic 4 — SNN Phase A
**Priority:** Must Have
**Points:** 5

**User Story:**
As a researcher,
I want a reusable spike encoding layer that converts windowed multi-hot input into spike trains,
so that all SNN models share a consistent, configurable encoding.

**Acceptance Criteria:**

- [ ] `SpikeEncoder` in `src/c5_snn/models/encoding.py` with direct and rate-coded modes
- [ ] Direct mode: passes binary values as spikes
- [ ] Rate-coded mode: Poisson-like encoding with configurable time steps
- [ ] Output shapes documented and tested for both modes
- [ ] Encoding deterministic given same seed
- [ ] Verify `import snntorch` works with PyTorch 2.5.1

**Technical Notes:**
Follow architecture Section 5.2. Implements FR5.

**Dependencies:** STORY-3.3 (needs working trainer)

---

#### STORY-4.2: Spiking MLP Model

**Epic:** Epic 4 — SNN Phase A
**Priority:** Must Have
**Points:** 5

**User Story:**
As a researcher,
I want a spiking MLP using LIF neurons and surrogate gradients,
so that I can evaluate the simplest SNN on this task.

**Acceptance Criteria:**

- [ ] `SpikingMLP(BaseModel)` in `src/c5_snn/models/snn_phase_a.py`
- [ ] Architecture: SpikeEncoder -> flatten -> spiking FC layers (snnTorch `Leaky`) -> linear -> 39 logits
- [ ] Configurable: beta, hidden sizes, layers, surrogate gradient method
- [ ] BCEWithLogitsLoss + optional cardinality penalty
- [ ] Registered as `"spiking_mlp"`, config at `configs/snn_phase_a_mlp.yaml`
- [ ] Unit test: forward `(4, 21, 39)` -> `(4, 39)`, backward completes

**Technical Notes:**
Follow SeqSNN paper guidance: beta=0.95, fast_sigmoid surrogate.

**Dependencies:** STORY-4.1

---

#### STORY-4.3: Spiking 1D-CNN Model

**Epic:** Epic 4 — SNN Phase A
**Priority:** Must Have
**Points:** 5

**User Story:**
As a researcher,
I want a spiking 1D-CNN with temporal convolutions and LIF neurons,
so that I can compare against the spiking MLP.

**Acceptance Criteria:**

- [ ] `SpikingCNN1D(BaseModel)` in `snn_phase_a.py`
- [ ] Architecture: SpikeEncoder -> 1D conv layers (temporal dim) + LIF -> pool -> linear -> 39 logits
- [ ] Configurable: kernel sizes, channels, conv layers, beta
- [ ] Registered as `"spiking_cnn1d"`, config at `configs/snn_phase_a_cnn.yaml`
- [ ] Unit test: forward + backward shape checks

**Technical Notes:**
Same surrogate/cardinality options as STORY-4.2.

**Dependencies:** STORY-4.1

---

#### STORY-4.4: SNN Phase A Training, Evaluation & Comparison

**Epic:** Epic 4 — SNN Phase A
**Priority:** Must Have
**Points:** 3

**User Story:**
As a researcher,
I want both Phase A models trained and compared against baselines,
so that I can see if SNN shows promise.

**Acceptance Criteria:**

- [ ] Both models trained with 3 seeds each
- [ ] RunPod B200 if projected >20 min locally
- [ ] `results/phase_a_comparison.json` with all models to date
- [ ] Console prints cumulative comparison table
- [ ] `project-memory.md` updated with Phase A results and encoding recommendation

**Technical Notes:**
If neither Phase A model beats GRU, document hypotheses for Phase B adjustments.

**Dependencies:** STORY-4.2, STORY-4.3

---

### Epic 5: SNN Phase B — Spike-GRU (13 pts)

#### STORY-5.1: Spike-GRU Architecture

**Epic:** Epic 5 — SNN Phase B
**Priority:** Must Have
**Points:** 5

**User Story:**
As a researcher,
I want a spiking GRU that processes the window recurrently with LIF neurons,
so that I can evaluate whether recurrence helps in the spiking domain.

**Acceptance Criteria:**

- [ ] `SpikeGRU(BaseModel)` in `src/c5_snn/models/snn_phase_b.py`
- [ ] Uses snnTorch `RLeaky`/`RSynaptic` for recurrent spiking layers
- [ ] Processes window event-by-event with accumulating membrane state
- [ ] Configurable: hidden size, recurrent layers, beta, dropout, surrogate
- [ ] Registered as `"spike_gru"`, config at `configs/snn_phase_b.yaml`
- [ ] Unit test: forward + backward, hidden state shape check

**Technical Notes:**
Informed by Phase A learnings on encoding mode (direct vs rate-coded).

**Dependencies:** STORY-4.4

---

#### STORY-5.2: Spike-GRU Hyperparameter Sweep

**Epic:** Epic 5 — SNN Phase B
**Priority:** Must Have
**Points:** 5

**User Story:**
As a researcher,
I want a structured HP sweep for the Spike-GRU,
so that the architecture gets a fair shot.

**Acceptance Criteria:**

- [ ] Sweep: hidden_size [64,128,256], layers [1,2], beta [0.5,0.8,0.95], encoding [direct,rate_coded]
- [ ] Top-3 configs re-run with 3 seeds
- [ ] RunPod B200 if any run >20 min projected
- [ ] Results logged to `results/phase_b_sweep.csv`
- [ ] Best checkpoint saved with config provenance

**Technical Notes:**
Generate sweep configs programmatically or use multiple YAML files.

**Dependencies:** STORY-5.1

---

#### STORY-5.3: Phase B Evaluation & Cumulative Comparison

**Epic:** Epic 5 — SNN Phase B
**Priority:** Must Have
**Points:** 3

**User Story:**
As a researcher,
I want best Spike-GRU compared against all prior models,
so that I can quantify the value of recurrence and inform Phase C.

**Acceptance Criteria:**

- [ ] `results/cumulative_comparison.json` with all models, all metrics, mean +/- std
- [ ] Console prints full leaderboard sorted by Recall@20
- [ ] `project-memory.md` updated: does Spike-GRU beat Phase A? Beat GRU? Recommendations for Phase C.

**Technical Notes:**
If Spike-GRU underperforms ANN GRU, document specific hypotheses.

**Dependencies:** STORY-5.2

---

### Epic 6: SNN Phase C — Spiking Transformer & Final Report (24 pts)

#### STORY-6.1: Spiking Transformer Architecture

**Epic:** Epic 6 — SNN Phase C & Final Report
**Priority:** Must Have
**Points:** 8

**User Story:**
As a researcher,
I want a spiking transformer with spike-domain self-attention and positional encoding,
so that I can capture long-range temporal dependencies.

**Acceptance Criteria:**

- [ ] `SpikingTransformer(BaseModel)` in `src/c5_snn/models/snn_phase_c.py`
- [ ] Architecture: SpikeEncoder -> positional encoding -> N spiking transformer layers (SSA + spiking FFN with LIF) -> temporal aggregation -> linear -> 39 logits
- [ ] SSA: attention from membrane potentials/spike counts, LIF-gated output
- [ ] Configurable: encoder layers, heads, hidden dim, FFN dim, beta, dropout, PE type
- [ ] Supports W=7-90 without architectural changes
- [ ] Registered as `"spiking_transformer"`, config at `configs/snn_phase_c.yaml`
- [ ] Unit tests: forward on `(4, 21, 39)` and `(4, 60, 39)` both -> `(4, 39)`, backward completes

**Technical Notes:**
Inspired by SeqSNN/Spikformer. SSA threshold ~0.25 per paper. Most complex model in the project.

**Dependencies:** STORY-5.3

---

#### STORY-6.2: Window Size Tuning Experiment

**Epic:** Epic 6 — SNN Phase C & Final Report
**Priority:** Must Have
**Points:** 5

**User Story:**
As a researcher,
I want the spiking transformer evaluated across W=7 to W=90,
so that I find the optimal temporal context length.

**Acceptance Criteria:**

- [ ] Window sizes: W in {7, 14, 21, 30, 45, 60, 90}
- [ ] For each W: regenerate tensors, recompute splits, train spiking transformer
- [ ] RunPod B200 for any run >20 min
- [ ] `results/window_tuning.csv` with W, all metrics, training time, sample count
- [ ] Optimal W identified and documented in `project-memory.md`

**Technical Notes:**
GPU-heavy story. 7 full training runs.

**Dependencies:** STORY-6.1

---

#### STORY-6.3: Spiking Transformer HP Sweep & Best Model

**Epic:** Epic 6 — SNN Phase C & Final Report
**Priority:** Must Have
**Points:** 5

**User Story:**
As a researcher,
I want a structured HP sweep at the optimal window size,
so that the final SNN model is the best it can be.

**Acceptance Criteria:**

- [ ] Sweep at optimal W: layers [2,4,6], heads [2,4], hidden [64,128], beta [0.5,0.8,0.95], encoding [direct,rate_coded]
- [ ] Top-5 re-run with 3 seeds
- [ ] `results/phase_c_sweep.csv` with all results
- [ ] Best checkpoint saved with config + pip freeze
- [ ] RunPod B200 as needed

**Technical Notes:**
This is the final model training. Ensure best checkpoint has full provenance.

**Dependencies:** STORY-6.2

---

#### STORY-6.4: Final Comprehensive Comparison & Report

**Epic:** Epic 6 — SNN Phase C & Final Report
**Priority:** Must Have
**Points:** 3

**User Story:**
As a researcher,
I want a comprehensive final report comparing every model,
so that I have a publishable-quality experimental record.

**Acceptance Criteria:**

- [ ] All 6 model types evaluated on identical test split
- [ ] `results/final_comparison.json` with all metrics, mean +/- std
- [ ] `results/final_report.md`: executive summary, leaderboard, per-phase analysis, window tuning findings, encoding analysis, efficiency, recommendations
- [ ] `project-memory.md` updated with final conclusions
- [ ] `python -m c5_snn.cli evaluate --final` prints final leaderboard

**Technical Notes:**
Follow architecture Data Model 4.7 (Comparison Report) schema.

**Dependencies:** STORY-6.3

---

#### STORY-6.5: Reproducible Runbook & Project Closure

**Epic:** Epic 6 — SNN Phase C & Final Report
**Priority:** Must Have
**Points:** 3

**User Story:**
As a researcher,
I want a documented runbook that allows anyone to reproduce the entire experiment,
so that the work is scientifically valid and extensible.

**Acceptance Criteria:**

- [ ] `README.md` updated: setup, data location, validate, train each model, evaluate, generate report
- [ ] `configs/` contains final best-run configs for every model
- [ ] All best checkpoints have config snapshots + pip freeze
- [ ] Single documented command sequence reproduces full pipeline
- [ ] All docs reflect final state
- [ ] All code passes CI (tests green, ruff clean)

**Technical Notes:**
Final cleanup story. Verify everything works end-to-end.

**Dependencies:** STORY-6.4

---

## Sprint Allocation

### Sprint 1 (Week 1) — 13/~20 pts

**Goal:** Working project scaffold with `validate-data` CLI and green CI

**Stories:**

| Story | Title | Points | Priority |
| --- | --- | --- | --- |
| STORY-1.1 | Repository & Package Scaffold | 3 | Must Have |
| STORY-1.2 | Logging, Seed Management & Config | 3 | Must Have |
| STORY-1.3 | Data Validation CLI Command | 5 | Must Have |
| STORY-1.4 | CI Skeleton & First Green Build | 2 | Must Have |

**Total:** 13 points (65% utilization — buffer for ROCm smoke test and initial setup friction)

**Risks:**

- ROCm may not work on RX 6600M — fallback to CPU-only local dev
- snnTorch install may conflict with PyTorch 2.5.1 — validate early

**Deliverable:** `python -m c5_snn.cli validate-data` works, CI green on GitHub

---

### Sprint 2 (Week 2) — 16/~20 pts

**Goal:** Complete data pipeline and evaluation harness ready for any model

**Stories:**

| Story | Title | Points | Priority |
| --- | --- | --- | --- |
| STORY-2.1 | Windowed Tensor Construction | 5 | Must Have |
| STORY-2.2 | Time-Based Splits | 3 | Must Have |
| STORY-2.3 | Evaluation Harness & Metrics | 5 | Must Have |
| STORY-2.4 | CLI evaluate Command | 3 | Must Have |

**Total:** 16 points (80% utilization)

**Risks:**

- Tensor shapes and leakage logic require careful testing

**Deliverable:** `data/processed/` populated, `evaluate` CLI works with dummy model

---

### Sprint 3 (Week 3) — 19/~20 pts

**Goal:** Baselines trained with first real benchmark numbers on Recall@20

**Stories:**

| Story | Title | Points | Priority |
| --- | --- | --- | --- |
| STORY-3.1 | Frequency/Recency Heuristic | 3 | Must Have |
| STORY-3.2 | ANN GRU Baseline | 5 | Must Have |
| STORY-3.3 | Training Loop & train CLI | 8 | Must Have |
| STORY-3.4 | Baseline Results & Comparison | 3 | Must Have |

**Total:** 19 points (95% utilization — heaviest sprint, Trainer is critical path)

**Risks:**

- Trainer (8 pts) is the most complex infrastructure piece
- GRU training may need RunPod if >20 min locally

**Deliverable:** `results/baseline_comparison.json` with real metrics, `train` CLI works

---

### Sprint 4 (Week 4) — 18/~20 pts

**Goal:** First SNN models running, Phase A vs baseline comparison

**Stories:**

| Story | Title | Points | Priority |
| --- | --- | --- | --- |
| STORY-4.1 | snnTorch Integration & Spike Encoding | 5 | Must Have |
| STORY-4.2 | Spiking MLP Model | 5 | Must Have |
| STORY-4.3 | Spiking 1D-CNN Model | 5 | Must Have |
| STORY-4.4 | Phase A Training & Comparison | 3 | Must Have |

**Total:** 18 points (90% utilization)

**Risks:**

- snnTorch surrogate gradient integration may require debugging
- SNN models may train slower than ANN (more compute per forward pass)

**Deliverable:** `results/phase_a_comparison.json`, first SNN-vs-baseline numbers

---

### Sprint 5 (Week 5) — 13/~20 pts

**Goal:** Spike-GRU with HP sweep, cumulative leaderboard through Phase B

**Stories:**

| Story | Title | Points | Priority |
| --- | --- | --- | --- |
| STORY-5.1 | Spike-GRU Architecture | 5 | Must Have |
| STORY-5.2 | HP Sweep | 5 | Must Have |
| STORY-5.3 | Phase B Evaluation & Comparison | 3 | Must Have |

**Total:** 13 points (65% utilization — buffer for HP sweep GPU time)

**Risks:**

- HP sweep generates many training runs — RunPod likely needed
- Recurrent spiking dynamics may be unstable (NaN gradients)

**Deliverable:** `results/cumulative_comparison.json` with all models through Phase B

---

### Sprint 6 (Week 6) — 18/~20 pts

**Goal:** Spiking Transformer built, window tuning and HP sweep completed

**Stories:**

| Story | Title | Points | Priority |
| --- | --- | --- | --- |
| STORY-6.1 | Spiking Transformer Architecture | 8 | Must Have |
| STORY-6.2 | Window Size Tuning | 5 | Must Have |
| STORY-6.3 | HP Sweep & Best Model | 5 | Must Have |

**Total:** 18 points (90% utilization)

**Risks:**

- Most complex model + most GPU-intensive sprint
- 7 window sizes + HP sweep = many RunPod hours
- SSA implementation may require iteration

**Deliverable:** Best spiking transformer checkpoint with optimal W and hyperparameters

---

### Sprint 7 (Week 7) — 6/~20 pts

**Goal:** Final comparison report and reproducible runbook — project closure

**Stories:**

| Story | Title | Points | Priority |
| --- | --- | --- | --- |
| STORY-6.4 | Final Comprehensive Comparison | 3 | Must Have |
| STORY-6.5 | Reproducible Runbook & Closure | 3 | Must Have |

**Total:** 6 points (30% utilization — intentional buffer for Sprint 6 spillover)

**Risks:**

- Sprint 6 training runs may spill into Week 7

**Deliverable:** `results/final_report.md`, updated README, all docs finalized

---

## Epic Traceability

| Epic | Epic Name | Stories | Total Points | Sprint |
| --- | --- | --- | --- | --- |
| Epic 1 | Foundation & Data Validation | 1.1, 1.2, 1.3, 1.4 | 13 | Sprint 1 |
| Epic 2 | Data Pipeline & Evaluation Harness | 2.1, 2.2, 2.3, 2.4 | 16 | Sprint 2 |
| Epic 3 | Baseline Models | 3.1, 3.2, 3.3, 3.4 | 19 | Sprint 3 |
| Epic 4 | SNN Phase A | 4.1, 4.2, 4.3, 4.4 | 18 | Sprint 4 |
| Epic 5 | SNN Phase B | 5.1, 5.2, 5.3 | 13 | Sprint 5 |
| Epic 6 | SNN Phase C & Final Report | 6.1, 6.2, 6.3, 6.4, 6.5 | 24 | Sprint 6-7 |

---

## Requirements Coverage

| FR/NFR | Requirement | Story | Sprint |
| --- | --- | --- | --- |
| FR1 | Validate CA5 data | STORY-1.3 | 1 |
| FR2 | Windowed tensors `(W, 39)` | STORY-2.1 | 2 |
| FR3 | Time-based splits | STORY-2.2 | 2 |
| FR4 | Baseline models | STORY-3.1, 3.2 | 3 |
| FR5 | SNN Phase A (MLP/CNN) | STORY-4.1, 4.2, 4.3 | 4 |
| FR6 | SNN Phase B (Spike-GRU) | STORY-5.1 | 5 |
| FR7 | SNN Phase C (Transformer) | STORY-6.1 | 6 |
| FR8 | BCEWithLogitsLoss + penalty | STORY-3.3 | 3 |
| FR9 | Evaluation harness + metrics | STORY-2.3, 2.4 | 2 |
| FR10 | CLI commands | STORY-1.3, 2.4, 3.3 | 1-3 |
| FR11 | Checkpoint + config snapshot | STORY-3.3 | 3 |
| FR12 | Top-20 ranked output | STORY-2.3 | 2 |
| NFR1 | Reproducibility (seeds, pip freeze) | STORY-1.2, 3.3 | 1, 3 |
| NFR2 | No data leakage | STORY-2.1, 2.2 | 2 |
| NFR3 | 20-min timing probe | STORY-3.3 | 3 |
| NFR4 | Experiment logging (CSV/JSON) | STORY-3.3 | 3 |
| NFR5 | Testing (unit + integration) | All stories | 1-7 |
| NFR6 | ONNX export | Deferred to v1.1+ | — |

---

## Risks and Mitigation

**High:**

- **ROCm + snnTorch compatibility** on consumer RX 6600M — Mitigation: smoke test in Sprint 1 Story 1.1; fallback to CPU-only local dev + RunPod for GPU
- **Spiking Transformer complexity** (Sprint 6) — Mitigation: SSA implementation informed by SeqSNN paper; extra buffer in Sprint 7

**Medium:**

- **RunPod GPU availability** for HP sweeps — Mitigation: plan GPU time in advance; keep local CPU training as backup for small experiments
- **NaN gradients in SNN training** — Mitigation: surrogate gradient clipping, beta tuning; documented in coding standards

**Low:**

- **GRU baseline may be hard to beat** — Mitigation: this is a valid research outcome; document it in final report
- **Dataset too small for transformer** (11,700 events) — Mitigation: regularization, dropout, early stopping; window tuning will reveal optimal context

---

## Definition of Done

For a story to be considered complete:

- [ ] Code implemented and committed to `main` branch
- [ ] Unit tests written and passing
- [ ] `ruff check src/` passes with zero errors
- [ ] Integration test passing (where applicable)
- [ ] Config files created (where applicable)
- [ ] Acceptance criteria validated
- [ ] CI green on GitHub Actions

---

## Next Steps

**Immediate:** Begin Sprint 1

Run `/bmad:dev-story` for STORY-1.1 to start implementing the repository scaffold, or run `/bmad:create-story STORY-1.1` to generate a detailed story document first.

**Sprint cadence:**

- Sprint length: 1 week
- Sprint start: Monday
- Sprint review: Friday
- Story order: sequential within each sprint (dependencies respected)

---

**This plan was created using BMAD Method v6 — Phase 4 (Implementation Planning)**
