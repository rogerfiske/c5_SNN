# c5_SNN Product Requirements Document (PRD)

## 1. Goals and Background Context

### 1.1 Goals

- **G1**: Produce a ranked list of the 20 most likely part IDs (from 39) for the next event, given a sliding window of historical events
- **G2**: Achieve measurable lift over non-SNN baselines on **Recall@20** and **Hit@20** averaged over a held-out time-based test set
- **G3**: Deliver a fully reproducible, one-command training and evaluation pipeline with fixed seeds, deterministic splits, and documented experiment configs
- **G4**: Progressively explore SNN architectures (spiking MLP/CNN → Spike-GRU → spiking transformer) to determine which SNN topology best suits this multi-label temporal ranking task
- **G5**: Provide a CLI tool that outputs Top-20 predictions for any given "as-of" date

### 1.2 Background Context

The CA5 dataset captures 11,700 historical events (1992–2026), each containing exactly 5 unique parts drawn from a pool of 39. The data is represented as multi-hot binary vectors (P_1...P_39), which maps naturally to spike-coded inputs for Spiking Neural Networks.

This project investigates whether SNN architectures — leveraging their inherent temporal dynamics and event-driven processing — can outperform conventional ANN baselines (GRU, TCN) and frequency heuristics on this multi-label ranking task. Strong non-SNN baselines must be established first to ensure any SNN lift is genuine and not an artifact of data leakage or weak comparison.

### 1.3 Change Log

| Date       | Version | Description                            | Author             |
| ---------- | ------- | -------------------------------------- | ------------------ |
| 2026-02-09 | 0.1     | Initial ChatGPT draft PRD              | ChatGPT / Roger    |
| 2026-02-09 | 0.2     | BMad Master audit & PM-HANDOFF created | BMad Master        |
| 2026-02-10 | 1.0     | BMAD restructured PRD                  | PM Agent (John)    |

---

## 2. Requirements

### 2.1 Functional Requirements

- **FR1**: The system shall validate `CA5_matrix_binary.csv` on load — asserting monotonic dates, exactly 5 active parts per event (`sum(P_1..P_39)==5`), and consistency between multi-hot columns and `m_1..m_5` columns
- **FR2**: The system shall construct windowed training samples of shape `(W, 39)` from consecutive events, with a configurable window size `W` (default 21, range 7–90)
- **FR3**: The system shall split data into train/validation/test sets using **time-based ordering only** (no shuffling), with split indices persisted to `data/processed/splits.json`
- **FR4**: The system shall implement baseline models: (a) frequency/recency heuristic over the last K events, and (b) ANN baseline (GRU or TCN) with identical windowing and output head
- **FR5**: The system shall implement SNN Phase A models (spiking MLP and/or spiking 1D-CNN) using PyTorch + snnTorch with surrogate gradients and LIF neurons
- **FR6**: The system shall implement SNN Phase B model (Spike-GRU) with configurable encoding (direct spike or rate-coded)
- **FR7**: The system shall implement SNN Phase C model (spiking transformer / SeqSNN-style temporal model with positional encoding and spike-domain self-attention)
- **FR8**: The system shall train all models using `BCEWithLogitsLoss` with optional cardinality penalty, applying early stopping on validation Recall@20
- **FR9**: The system shall provide an evaluation harness that computes Recall@20, Hit@20 (primary), and Recall@5, Hit@5, MRR (secondary) over the test set and outputs a JSON/CSV evaluation report
- **FR10**: The system shall provide a CLI with four commands: `validate-data`, `train --config <path>`, `evaluate --checkpoint <path>`, and `predict --asof <YYYY-MM-DD>`
- **FR11**: The system shall save model checkpoints alongside the experiment config snapshot used to produce them
- **FR12**: The system shall output a ranked list of **20 unique part IDs** with optional associated scores for each prediction

### 2.2 Non-Functional Requirements

- **NFR1**: **Reproducibility** — All training runs shall use fixed random seeds and deterministic flags; a `pip freeze` output shall be captured per run
- **NFR2**: **No data leakage** — All feature engineering and windowing must use only history up to time `t` when predicting `t+1`; no future information may be accessible during training or validation
- **NFR3**: **Local training probe** — A 2-epoch timing probe shall complete in under 20 minutes on the developer workstation (Ryzen 9 6900HX, 64GB RAM, RX 6600M 8GB); if projected full training exceeds this threshold, training shall be moved to RunPod B200 GPU
- **NFR4**: **Experiment logging** — Each training run shall produce a metrics CSV and JSON summary persisted alongside the checkpoint, sufficient for comparing runs without re-execution
- **NFR5**: **Testing** — Unit tests (pytest) shall cover data validation, window construction (shapes/alignment), no-leakage constraints, and deterministic split reproduction; an integration test shall validate a tiny end-to-end run on the first N rows
- **NFR6**: **Export (optional, v1.1+)** — Models may optionally be exported to ONNX or TorchScript format for future deployment scenarios

---

## 3. User Interface Design Goals

**N/A** — This is a CLI-only ML research pipeline. The command-line interface is specified in FR10.

---

## 4. Technical Assumptions

### 4.1 Repository Structure: Monorepo

Single repository containing all source code, data references, configs, tests, and documentation. The project is a self-contained ML research pipeline with no service decomposition needed.

### 4.2 Service Architecture

**Single-service ML research pipeline** — a single Python package (`c5_snn`) with a CLI entry point. No API server, no microservices, no serverless functions.

Components:

- Data module (loading, validation, windowing, splits)
- Models module (baselines, SNN phases A/B/C)
- Training module (trainer loop, metrics, callbacks, checkpointing)
- CLI module (argparse/click entry points)
- Config-driven experiments (YAML configs in `configs/`)

### 4.3 Testing Requirements: Unit + Integration

- **Unit tests** (pytest): Data validation logic, window builder, no-leakage constraints, deterministic split reproduction, metric computation correctness
- **Integration test**: End-to-end tiny run on first N rows — load → window → train 1 epoch → evaluate → produce Top-20 output
- No E2E/UI testing needed (no UI)

### 4.4 Additional Technical Assumptions

- **Language**: Python 3.11+
- **Core frameworks**: PyTorch (deep learning), snnTorch (spiking neural network layers, surrogate gradients, LIF neurons)
- **Data handling**: pandas, numpy
- **CLI framework**: argparse or click (developer's choice at implementation time)
- **Build system**: `pyproject.toml` with pip-installable package (`pip install -e .`)
- **Dependency management**: Pinned versions in `pyproject.toml` `[project.dependencies]`
- **Compute**: Local development on Ryzen 9 6900HX / 64GB RAM / RX 6600M 8GB GPU; RunPod B200 GPU for training runs exceeding 20 minutes locally
- **GPU note**: Local GPU is AMD RX 6600M (ROCm). RunPod uses NVIDIA (CUDA). Code must be device-agnostic via `torch.device` abstraction
- **Random seed management**: Global seed utility setting `random`, `numpy`, `torch` seeds consistently
- **Config format**: YAML experiment configs (model hyperparams, window size, learning rate, epochs, etc.)
- **Logging**: Python `logging` module for runtime logs; CSV/JSON for experiment metrics
- **Version control**: Git, GitHub repo at <https://github.com/rogerfiske/c5_SNN>
- **CI**: GitHub Actions — run pytest + linting on push (details deferred to Architect)

---

## 5. Epic List

| #     | Epic                                                 | Goal                                                         |
| ----- | ---------------------------------------------------- | ------------------------------------------------------------ |
| **1** | Foundation & Data Validation                         | Project infrastructure + working `validate-data` CLI         |
| **2** | Data Pipeline & Evaluation Harness                   | Windowing, splits, metrics framework                         |
| **3** | Baseline Models                                      | Frequency heuristic + ANN baseline benchmarks                |
| **4** | SNN Phase A — Spiking MLP/CNN                        | First SNN model, validate snnTorch integration               |
| **5** | SNN Phase B — Spike-GRU                              | Sequence-aware SNN, compare against Phase A                  |
| **6** | SNN Phase C — Spiking Transformer & Final Report     | Expected best-performing model + comprehensive comparison    |

---

## 6. Epic Details

### Epic 1: Foundation & Data Validation

**Goal**: Establish the complete project scaffold — git repository, Python package structure, build system, logging, seed management, and cross-cutting infrastructure — then deliver the first functional increment: a CLI command that validates the raw CA5 dataset integrity.

#### Story 1.1: Repository & Package Scaffold

> As a **developer**,
> I want a properly initialized git repository with a pip-installable Python package structure,
> so that all subsequent work has a clean, standard foundation to build on.

**Acceptance Criteria:**

1. Git repository initialized with `.gitignore` (Python, IDE files, data artifacts, `__pycache__`, `.mypy_cache`)
2. `pyproject.toml` with project metadata (`c5-snn`, version `0.1.0`), Python 3.11+ requirement, and `[project.dependencies]` including `torch`, `snntorch`, `pandas`, `numpy`, `pytest`
3. Package installable via `pip install -e .` with `src/c5_snn/` layout and `__init__.py`
4. Directory structure created: `src/c5_snn/data/`, `src/c5_snn/models/`, `src/c5_snn/training/`, `configs/`, `tests/`, `data/raw/`, `data/processed/`, `scripts/`
5. `README.md` with project name, one-line description, and setup instructions (`pip install -e .`)
6. Initial commit pushed to `https://github.com/rogerfiske/c5_SNN`

#### Story 1.2: Logging, Seed Management & Configuration Infrastructure

> As a **developer**,
> I want centralized logging and deterministic seed management available from the start,
> so that every subsequent story has reproducibility and observability built in from day one.

**Acceptance Criteria:**

1. `src/c5_snn/utils.py` (or `utils/` module) provides a `set_global_seed(seed: int)` function that sets `random.seed`, `numpy.random.seed`, `torch.manual_seed`, and `torch.cuda.manual_seed_all`
2. `src/c5_snn/utils.py` provides a `setup_logging(level: str)` function configuring Python `logging` with timestamped console output and optional file handler
3. A base YAML config schema is established in `configs/default.yaml` with fields for `seed`, `log_level`, and placeholder sections for `data`, `model`, `training`
4. Unit tests verify: (a) `set_global_seed` produces identical random sequences across calls, (b) logging outputs to expected destinations
5. All utility functions are importable via `from c5_snn.utils import ...`

#### Story 1.3: Data Validation CLI Command

> As a **researcher**,
> I want to run `python -m c5_snn.cli validate-data` and get a clear pass/fail report on the raw dataset,
> so that I can confirm data integrity before any modeling work begins.

**Acceptance Criteria:**

1. CLI entry point at `src/c5_snn/cli.py` using argparse or click, with `validate-data` subcommand accepting an optional `--data-path` argument (default: `data/raw/CA5_matrix_binary.csv`)
2. Validation checks: (a) file exists and loads as CSV, (b) expected 45 columns present (`date`, `m_1..m_5`, `P_1..P_39`), (c) dates parseable and monotonically non-decreasing, (d) `sum(P_1..P_39) == 5` for every row, (e) multi-hot columns match `m_1..m_5` values for every row
3. CLI outputs a summary: total rows, date range, number of unique parts observed, and pass/fail per check
4. Non-zero exit code on any validation failure
5. Unit tests using a small fixture CSV (5–10 rows) covering: valid data passes, missing column fails, broken row-sum fails, mismatched m/P columns fail
6. Uses logging infrastructure from Story 1.2

#### Story 1.4: CI Skeleton & First Green Build

> As a **developer**,
> I want a GitHub Actions workflow that runs linting and tests on every push,
> so that code quality is enforced from the first commit.

**Acceptance Criteria:**

1. `.github/workflows/ci.yml` runs on push and PR to `main`
2. Steps: checkout → install Python 3.11 → `pip install -e ".[dev]"` → `pytest tests/ -v` → `ruff check src/` (or `flake8`)
3. Dev dependencies (`pytest`, `ruff` or `flake8`) specified in `pyproject.toml` `[project.optional-dependencies.dev]`
4. CI passes on current codebase (all tests from Stories 1.2–1.3 green)
5. Badge or status check visible on GitHub repo

---

### Epic 2: Data Pipeline & Evaluation Harness

**Goal**: Build the complete data preparation pipeline (windowed tensor construction, time-based splits) and the reusable evaluation framework (Recall@20, Hit@20, and secondary metrics), so that every model from Epic 3 onward can be trained and measured on a consistent, leak-free foundation.

#### Story 2.1: Windowed Tensor Construction

> As a **researcher**,
> I want validated raw data transformed into windowed tensors of shape `(W, 39)` with corresponding next-event targets,
> so that I have training-ready inputs for all models.

**Acceptance Criteria:**

1. `src/c5_snn/data/windowing.py` provides a function that takes the validated DataFrame and window size `W` (default 21) and returns: input tensors `X` of shape `(N_samples, W, 39)` and target tensors `y` of shape `(N_samples, 39)`
2. For sample at index `t`: `X[t]` contains P-vectors from events `[t, t+1, ..., t+W-1]` and `y[t]` is the P-vector of event `t+W`
3. Window size `W` is configurable via the YAML config established in Epic 1
4. Tensors are saved to `data/processed/` as `.pt` files with metadata (source file hash, W, number of samples)
5. Unit tests verify: (a) output shapes are correct for known input size, (b) first and last windows align correctly with source data, (c) no overlap between input window and target event, (d) edge cases — dataset smaller than W returns empty result gracefully

#### Story 2.2: Time-Based Train/Validation/Test Splits

> As a **researcher**,
> I want the windowed samples split into train/val/test sets by chronological order,
> so that evaluation is temporally valid with zero data leakage.

**Acceptance Criteria:**

1. `src/c5_snn/data/splits.py` provides a function that splits windowed samples by time order using configurable ratios (default: 70/15/15 train/val/test)
2. Split indices are persisted to `data/processed/splits.json` containing: split ratios, index ranges, date ranges per split, and the window size used
3. Splits are deterministic — re-running with the same config produces identical indices
4. No shuffling — train contains the earliest samples, test contains the most recent
5. A convenience function returns PyTorch `DataLoader` objects for each split with configurable batch size
6. Unit tests verify: (a) no index overlap between splits, (b) all train indices < all val indices < all test indices, (c) splits.json round-trips correctly, (d) changing ratios produces different but still valid splits

#### Story 2.3: Evaluation Harness & Metrics

> As a **researcher**,
> I want a reusable evaluation module that computes all defined metrics given model predictions and ground truth,
> so that every model (baseline and SNN) is measured consistently.

**Acceptance Criteria:**

1. `src/c5_snn/training/metrics.py` provides functions for: Recall@K, Hit@K (for K=5 and K=20), and MRR over the positive labels
2. Recall@20 defined as: of the 5 true parts, how many appear in the model's top-20 predictions, averaged over the test set
3. Hit@20 defined as: binary indicator (at least 1 true part in top-20), averaged over the test set
4. `src/c5_snn/training/evaluate.py` provides an `evaluate_model(model, dataloader, device)` function that runs inference, computes all metrics, and returns a results dict
5. Results exported as both JSON summary and CSV (one row per test sample with per-sample metrics) to a configurable output directory
6. Unit tests with hand-crafted predictions verify: (a) perfect predictions yield Recall@20 = 1.0, (b) completely wrong predictions yield Recall@20 = 0.0, (c) known partial overlap yields expected fractional recall, (d) MRR computation is correct for known rankings
7. Device-agnostic — works on CPU, CUDA, and ROCm via `torch.device` abstraction

#### Story 2.4: CLI `evaluate` Command Integration

> As a **researcher**,
> I want to run `python -m c5_snn.cli evaluate --checkpoint <path>` and get a printed metrics summary plus saved report files,
> so that I can assess any trained model with a single command.

**Acceptance Criteria:**

1. `evaluate` subcommand added to CLI, accepting `--checkpoint` (required) and `--output-dir` (default: `results/`)
2. Command loads checkpoint, loads test split from `data/processed/splits.json`, runs the evaluation harness from Story 2.3, and prints a formatted metrics table to stdout
3. JSON and CSV reports saved to the output directory
4. Non-zero exit code if checkpoint file not found or evaluation fails
5. Integration test: using a dummy model (random weights), verify the CLI runs end-to-end and produces output files with expected structure (correct keys in JSON, correct columns in CSV)

---

### Epic 3: Baseline Models

**Goal**: Implement a frequency/recency heuristic and an ANN baseline (GRU), producing the first real benchmark numbers on Recall@20 and Hit@20. These baselines establish the floor that all SNN models must beat, and serve as a sanity check that the pipeline is working correctly.

#### Story 3.1: Frequency/Recency Heuristic Baseline

> As a **researcher**,
> I want a non-learned baseline that ranks parts by historical frequency and recency within the input window,
> so that I have a simple, interpretable lower bound for model comparison.

**Acceptance Criteria:**

1. `src/c5_snn/models/baselines.py` provides a `FrequencyBaseline` class with a `predict(X: Tensor) -> Tensor` method that returns 39 scores per sample
2. Scoring strategy: for each sample's window `(W, 39)`, compute a weighted combination of (a) raw frequency count per part across the W events and (b) exponentially-decayed recency weighting (more recent events weighted higher)
3. Decay factor configurable via YAML config
4. Model conforms to same interface as future learned models — takes windowed input, returns 39 logits/scores — so the evaluation harness works without modification
5. CLI `train` subcommand recognizes `model_type: frequency_baseline` in config and runs the heuristic (no gradient training, just fit-and-evaluate)
6. Full evaluation run on the test split, producing JSON/CSV reports via the evaluation harness
7. Unit tests verify: (a) a window where part P_1 appears in every event ranks P_1 highest, (b) recency weighting correctly boosts more recent appearances, (c) output shape is `(batch, 39)`

#### Story 3.2: ANN Baseline — GRU Model

> As a **researcher**,
> I want a conventional GRU-based neural network baseline trained on the same windowed data,
> so that I have a strong learned-model benchmark to compare SNN architectures against.

**Acceptance Criteria:**

1. `src/c5_snn/models/baselines.py` (or `ann_baseline.py`) provides a `GRUBaseline(nn.Module)` with configurable hidden size, number of layers, and dropout
2. Architecture: GRU encoder processing the `(W, 39)` input sequence → final hidden state → linear head → 39 logits
3. Training uses `BCEWithLogitsLoss`, Adam optimizer, learning rate and epochs configurable via YAML config
4. Early stopping on validation Recall@20 with configurable patience
5. Checkpoint saved on best validation Recall@20, alongside the config snapshot (FR11)
6. Uses `set_global_seed` from Epic 1 for reproducibility; uses `torch.device` abstraction for CPU/CUDA/ROCm
7. Unit test: one forward pass on a random `(batch=4, W=21, 39)` tensor produces output shape `(4, 39)` without error

#### Story 3.3: Training Loop & `train` CLI Command

> As a **researcher**,
> I want to run `python -m c5_snn.cli train --config configs/baseline_gru.yaml` and get a trained model checkpoint with logged metrics,
> so that I can train any model with a single command and a config file.

**Acceptance Criteria:**

1. `src/c5_snn/training/trainer.py` provides a reusable `Trainer` class that accepts any model conforming to the common interface (forward takes windowed input, returns 39 logits)
2. Trainer loop: for each epoch — train step (forward, loss, backward, optimizer step) → validation metrics via evaluation harness → early stopping check → checkpoint save on best val Recall@20
3. Per-epoch metrics (train loss, val Recall@20, val Hit@20) logged to console and appended to a CSV file in the experiment output directory
4. `train` subcommand added to CLI, accepting `--config <path>` (required); config specifies model type, hyperparameters, data paths, seed, and output directory
5. 2-epoch timing probe: after epoch 2, print projected total training time; if >20 minutes, log a warning recommending RunPod B200
6. `pip freeze` output captured to the experiment output directory (NFR1)
7. Integration test: train the GRU baseline for 2 epochs on a tiny subset (first 100 rows, W=5), verify checkpoint file exists and evaluation produces valid metrics

#### Story 3.4: Baseline Results & First Comparison

> As a **researcher**,
> I want a documented comparison of frequency heuristic vs. GRU baseline on the full test set,
> so that I have quantified benchmarks before starting SNN work.

**Acceptance Criteria:**

1. Both models (frequency heuristic and GRU baseline) evaluated on the identical test split using the evaluation harness
2. A comparison summary produced in `results/baseline_comparison.json` containing: model name, Recall@20, Hit@20, Recall@5, Hit@5, MRR for each model
3. Console output prints a formatted comparison table
4. GRU baseline trained with at least 3 different seeds; report includes mean and standard deviation of metrics across seeds
5. Results logged with full config provenance (which config file, which checkpoint, which split)
6. `project-memory.md` updated with baseline results and observations

---

### Epic 4: SNN Phase A — Spiking MLP/CNN

**Goal**: Implement the first spiking neural network models using snnTorch with surrogate gradients and LIF neurons, validate the full SNN training pipeline on the existing infrastructure, and produce the first SNN-vs-baseline comparison.

#### Story 4.1: snnTorch Integration & Spike Encoding Layer

> As a **researcher**,
> I want a reusable spike encoding layer that converts windowed multi-hot input into spike trains compatible with snnTorch,
> so that all SNN models share a consistent, configurable input encoding.

**Acceptance Criteria:**

1. `src/c5_snn/models/encoding.py` provides a `SpikeEncoder` module supporting two modes: (a) **direct** — passes binary multi-hot values directly as spikes, (b) **rate-coded** — converts values through a Poisson-like rate coding layer with configurable number of simulation time steps
2. Encoding mode and number of time steps configurable via YAML config
3. Output shape documented: for direct mode `(batch, W, 39)`, for rate-coded mode `(batch, T_sim, W, 39)` or flattened per architecture needs
4. Unit tests verify: (a) direct mode preserves binary input exactly, (b) rate-coded mode produces valid spike trains (values in {0,1}), (c) output shapes match specification for both modes, (d) encoding is deterministic given the same seed

#### Story 4.2: Spiking MLP Model

> As a **researcher**,
> I want a spiking MLP model using LIF neurons and surrogate gradients that takes windowed input and produces 39 logits,
> so that I can evaluate the simplest possible SNN architecture on this task.

**Acceptance Criteria:**

1. `src/c5_snn/models/snn_phase_a.py` provides a `SpikingMLP(nn.Module)` using snnTorch `Leaky` (LIF) neurons with configurable `beta` (decay rate), hidden layer sizes, and number of layers
2. Architecture: SpikeEncoder → flatten window into single vector → one or more spiking fully-connected layers with LIF neurons → linear readout → 39 logits (membrane potential or spike count decoded)
3. Surrogate gradient method configurable (default: fast sigmoid from snnTorch)
4. Conforms to the common model interface — trainer and evaluation harness work without modification
5. Training uses `BCEWithLogitsLoss` + optional cardinality penalty (configurable weight, default 0)
6. Unit test: forward pass on `(batch=4, W=21, 39)` input produces `(4, 39)` output; backward pass completes without error

#### Story 4.3: Spiking 1D-CNN Model

> As a **researcher**,
> I want a spiking 1D-CNN model that applies temporal convolutions over the event window with LIF neurons,
> so that I can compare a spatially-aware SNN against the spiking MLP.

**Acceptance Criteria:**

1. `src/c5_snn/models/snn_phase_a.py` (or separate file) provides a `SpikingCNN1D(nn.Module)` using snnTorch `Leaky` neurons with configurable kernel sizes, channel counts, and number of conv layers
2. Architecture: SpikeEncoder → 1D convolution layers along the temporal (W) dimension with LIF neurons between layers → global pooling → linear readout → 39 logits
3. Same configurable surrogate gradient, beta, and cardinality penalty as Story 4.2
4. Conforms to the common model interface
5. Unit test: forward pass on `(batch=4, W=21, 39)` produces `(4, 39)` output; backward pass completes without error
6. Config file `configs/snn_phase_a_cnn.yaml` provided alongside `configs/snn_phase_a_mlp.yaml`

#### Story 4.4: SNN Phase A Training, Evaluation & Comparison

> As a **researcher**,
> I want both Phase A models trained and evaluated against the baselines from Epic 3,
> so that I can determine whether SNN shows promise on this task and which Phase A variant to carry forward.

**Acceptance Criteria:**

1. Both Spiking MLP and Spiking CNN trained on full training set using the `train` CLI command with respective config files
2. Each model trained with at least 3 seeds; metrics reported as mean ± std
3. If projected training time exceeds 20 minutes locally, training executed on RunPod B200 (document which environment was used)
4. Comparison report produced in `results/phase_a_comparison.json` containing: all baselines + Spiking MLP + Spiking CNN, with Recall@20, Hit@20, Recall@5, Hit@5, MRR for each
5. Console prints a formatted comparison table spanning all models evaluated so far
6. `project-memory.md` updated with Phase A results, observations, and recommendation for which encoding mode (direct vs rate-coded) performed better
7. If neither SNN Phase A model beats the GRU baseline, document hypotheses for why and adjustments to try in Phase B

---

### Epic 5: SNN Phase B — Spike-GRU

**Goal**: Implement a sequence-aware spiking recurrent model (Spike-GRU) that can capture temporal dependencies across the event window using spiking neurons, and determine whether explicit recurrence in the spiking domain improves performance over the feedforward Phase A models.

#### Story 5.1: Spike-GRU Architecture

> As a **researcher**,
> I want a spiking GRU model that processes the event window recurrently with LIF neurons,
> so that I can evaluate whether recurrent spiking dynamics capture temporal patterns better than feedforward SNN.

**Acceptance Criteria:**

1. `src/c5_snn/models/snn_phase_b.py` provides a `SpikeGRU(nn.Module)` that combines gated recurrent units with snnTorch LIF neurons
2. Architecture: SpikeEncoder → spiking recurrent layer(s) where LIF membrane dynamics are integrated with GRU gating (snnTorch's `RLeaky`/`RSynaptic` or custom hybrid) → final hidden state → linear readout → 39 logits
3. Configurable: hidden size, number of recurrent layers, beta (LIF decay), dropout, surrogate gradient method
4. Model processes the window step-by-step along the temporal dimension (event-by-event), allowing membrane state to accumulate across time steps
5. Conforms to common model interface — works with existing trainer, evaluator, and CLI
6. Unit test: forward pass on `(batch=4, W=21, 39)` produces `(4, 39)` output; backward pass through surrogate gradients completes without error; hidden state shape matches config

#### Story 5.2: Spike-GRU Training & Hyperparameter Sweep

> As a **researcher**,
> I want the Spike-GRU trained with a structured hyperparameter sweep over key parameters,
> so that I give this architecture a fair shot at its best performance before comparing.

**Acceptance Criteria:**

1. Config file `configs/snn_phase_b.yaml` provided with baseline hyperparameters informed by Phase A learnings
2. Sweep covers at minimum: (a) hidden size [64, 128, 256], (b) number of layers [1, 2], (c) beta [0.5, 0.8, 0.95], (d) encoding mode [direct, rate-coded] — unless Phase A conclusively settled this
3. Each configuration trained with the global seed; top-3 configurations re-run with 3 seeds for variance estimation
4. Training uses RunPod B200 if any single run exceeds 20 minutes projected locally
5. All sweep results logged to `results/phase_b_sweep.csv` with columns: config hash, hyperparameters, best val Recall@20, best val Hit@20, training time, environment (local/RunPod)
6. Best checkpoint saved with full config provenance

#### Story 5.3: Phase B Evaluation & Cumulative Comparison

> As a **researcher**,
> I want a comprehensive comparison of the best Spike-GRU against all prior models,
> so that I can quantify the value of recurrence in the spiking domain and inform Phase C design.

**Acceptance Criteria:**

1. Best Spike-GRU model evaluated on test split alongside all prior models: frequency heuristic, GRU baseline, Spiking MLP, Spiking CNN
2. Cumulative comparison report in `results/cumulative_comparison.json` with all models, all metrics, mean ± std
3. Console prints the full leaderboard table sorted by Recall@20
4. `project-memory.md` updated with: (a) does Spike-GRU beat Phase A? (b) does Spike-GRU beat ANN GRU? (c) what does the gap suggest? (d) recommendations for Phase C architecture choices
5. If Spike-GRU underperforms ANN GRU, document specific hypotheses to guide Phase C

---

### Epic 6: SNN Phase C — Spiking Transformer & Final Report

**Goal**: Implement the most advanced SNN architecture — a spiking transformer with positional encoding and self-attention adapted to the spike domain — expected to deliver the strongest results by leveraging longer-range temporal context. Then produce the comprehensive final comparison report and reproducible runbook.

#### Story 6.1: Spiking Transformer Architecture

> As a **researcher**,
> I want a spiking transformer model that applies spike-domain self-attention over the event window with positional encoding,
> so that I can capture long-range temporal dependencies that recurrent models may miss.

**Acceptance Criteria:**

1. `src/c5_snn/models/snn_phase_c.py` provides a `SpikingTransformer(nn.Module)` implementing a transformer encoder adapted for spiking neurons, inspired by SeqSNN / Spikformer approaches
2. Architecture: SpikeEncoder → positional encoding (learnable or sinusoidal, configurable) → N spiking transformer encoder layers (spike-domain self-attention + spiking feed-forward with LIF neurons) → temporal aggregation (mean pooling or CLS-token) → linear readout → 39 logits
3. Spike-domain self-attention: attention scores computed from membrane potentials or spike counts; LIF neurons gate the attention output
4. Configurable: number of encoder layers, attention heads, hidden dimension, feed-forward dimension, beta, dropout, positional encoding type
5. Supports window sizes across the full tuning range W=7–90 without architectural changes
6. Conforms to common model interface
7. Unit test: forward pass on `(batch=4, W=21, 39)` and `(batch=4, W=60, 39)` both produce `(4, 39)` output; backward pass completes without error; attention weights extractable for interpretability

#### Story 6.2: Window Size Tuning Experiment

> As a **researcher**,
> I want the spiking transformer evaluated across the full window range (W=7 to W=90),
> so that I can find the optimal temporal context length and understand how performance scales with history.

**Acceptance Criteria:**

1. Window sizes tested: W ∈ {7, 14, 21, 30, 45, 60, 90}
2. For each W: windowed tensors regenerated, splits recomputed, spiking transformer trained with best-known hyperparameters
3. Training on RunPod B200 for any run exceeding 20 minutes projected locally
4. Results logged to `results/window_tuning.csv` with columns: W, Recall@20, Hit@20, Recall@5, MRR, training time, number of samples
5. Analysis identifies: optimal W, plateau/degradation patterns, overfitting evidence at extreme W values
6. Best-performing W documented in `project-memory.md`

#### Story 6.3: Spiking Transformer Hyperparameter Sweep & Best Model

> As a **researcher**,
> I want a structured hyperparameter sweep for the spiking transformer at the optimal window size,
> so that the final SNN model represents the best this architecture can do on this task.

**Acceptance Criteria:**

1. Sweep conducted at the optimal W from Story 6.2
2. Sweep covers: (a) encoder layers [2, 4, 6], (b) attention heads [2, 4], (c) hidden dimension [64, 128], (d) beta [0.5, 0.8, 0.95], (e) encoding mode [direct, rate-coded]
3. Top-5 configurations re-run with 3 seeds for variance estimation
4. All results logged to `results/phase_c_sweep.csv`
5. Best checkpoint saved with full config provenance and `pip freeze` snapshot
6. Training on RunPod B200 as needed

#### Story 6.4: Final Comprehensive Comparison & Report

> As a **researcher**,
> I want a comprehensive final report comparing every model across all metrics with analysis and conclusions,
> so that I have a publishable-quality record of the entire experimental investigation.

**Acceptance Criteria:**

1. All models evaluated on the identical test split: frequency heuristic, GRU baseline (ANN), Spiking MLP, Spiking CNN, Spike-GRU, Spiking Transformer
2. Final leaderboard in `results/final_comparison.json` with all metrics, mean ± std
3. `results/final_report.md` containing: executive summary, full leaderboard, per-phase analysis, window size tuning findings, encoding analysis, training efficiency, failure analysis, recommendations for v1.1+
4. `project-memory.md` updated with final conclusions and v1.1 backlog items
5. Console prints the final leaderboard when running `python -m c5_snn.cli evaluate --final`

#### Story 6.5: Reproducible Runbook & Project Closure

> As a **researcher**,
> I want a documented runbook that allows anyone to reproduce the entire experiment from scratch,
> so that the work is scientifically valid and can be extended by others.

**Acceptance Criteria:**

1. `README.md` updated with complete instructions: environment setup, data location, how to run validation, how to train each model, how to evaluate, how to generate the final report
2. `configs/` directory contains the final config files for every model's best run
3. All checkpoints for best models stored with config snapshots and `pip freeze` outputs
4. A single script or CLI command sequence documented that reproduces the full pipeline end-to-end
5. `docs/prd.md`, `docs/architecture.md`, and `docs/project-memory.md` reflect final state
6. All code passes CI (tests green, linting clean)

---

## 7. Checklist Results Report

### Executive Summary

- **Overall PRD completeness**: ~92%
- **MVP scope appropriateness**: Just Right
- **Readiness for architecture phase**: **READY**
- **Most critical gap**: No user research (expected — single-researcher ML project, not a product with external users)

### Category Statuses

| Category                         | Status      | Critical Issues                                                                 |
| -------------------------------- | ----------- | ------------------------------------------------------------------------------- |
| 1. Problem Definition & Context  | PASS        | None — clear problem, measurable metrics, rationale documented                  |
| 2. MVP Scope Definition          | PASS        | Out-of-scope items clearly delineated (NFR6, calendar features, streaming)      |
| 3. User Experience Requirements  | PASS        | N/A — CLI-only pipeline, explicitly marked                                      |
| 4. Functional Requirements       | PASS        | 12 FRs covering full pipeline, testable and unambiguous                         |
| 5. Non-Functional Requirements   | PASS        | 6 NFRs, reproducibility and no-leakage well-specified                           |
| 6. Epic & Story Structure        | PASS        | 6 epics, 22 stories, sequential dependencies correct, sized for AI agent        |
| 7. Technical Guidance            | PARTIAL     | ROCm/CUDA risk flagged; pinned versions and CI details deferred to Architect    |
| 8. Cross-Functional Requirements | PARTIAL     | No data retention policy (acceptable for research); no external integrations    |
| 9. Clarity & Communication       | PASS        | Consistent terminology, well-structured, versioned                              |

### Top Issues by Priority

**BLOCKERS**: None

**HIGH**: None

**MEDIUM**:

- M1: Pinned dependency versions not in PRD (intentionally deferred to Architect)
- M2: ROCm compatibility for PyTorch + snnTorch not validated — Architect should investigate

**LOW**:

- L1: No formal experiment naming convention — developer can establish during Epic 1
- L2: `predict --asof` semantics could be more specific (uses existing checkpoint)

### Recommendations

1. Architect should validate PyTorch ROCm + snnTorch compatibility on RX 6600M early
2. Architect should pin all dependency versions in the architecture doc
3. Consider adding an experiment naming convention to coding standards

### Final Decision

**READY FOR ARCHITECT** — The PRD is comprehensive, properly structured, and ready for architectural design. The two PARTIAL categories contain items intentionally deferred to the Architect phase.

---

## 8. Next Steps

### 8.1 UX Expert Prompt

**N/A** — CLI-only pipeline, no UX design required.

### 8.2 Architect Prompt

> Review `docs/prd.md` (the BMAD-processed PRD) and the existing draft at `docs/architecture.md`. Use `docs/concept.md` and `docs/project-memory.md` as additional context. Also review `docs-imported/pc_specs.md` for hardware constraints and `docs-imported/Spike_Neural_Networks.pdf` for SNN domain knowledge. Process the architecture through the BMAD architecture template, paying special attention to: tech stack with pinned versions, source tree, data models (windowed tensors, model checkpoints, evaluation artifacts), component design, coding standards for ML/Python, and test strategy for an ML research pipeline.
