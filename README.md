# c5_SNN

Spiking Neural Network time-series forecasting pipeline for CA5 event prediction.

This project explores whether biologically-inspired Spiking Neural Networks (SNNs) can outperform conventional baselines for multi-label event forecasting on the CA5 dataset. Six model architectures were implemented, trained, and compared across three experimental phases.

## Results

**Final Leaderboard (sorted by Recall@20):**

| Rank | Model | Type | Phase | W | Recall@20 | Hit@20 | MRR |
|------|-------|------|-------|---|-----------|--------|-----|
| 1 | Frequency Baseline | Heuristic | Baseline | 21 | **0.5232** | 0.9840 | 0.3125 |
| 2 | Spiking Transformer | SNN | Phase C | 90 | 0.5178 | 0.9803 | 0.3128 |
| 3 | Spiking CNN-1D | SNN | Phase A | 21 | 0.5152 | 0.9795 | 0.3053 |
| 4 | Spike-GRU | SNN | Phase B | 21 | 0.5137 | 0.9812 | 0.3115 |
| 5 | Spiking MLP | SNN | Phase A | 21 | 0.5125 | 0.9759 | 0.3101 |
| 6 | GRU Baseline | ANN | Baseline | 21 | 0.5099 | 0.9789 | 0.3103 |

**Key Findings:**
- All 6 models cluster within 0.013 Recall@20 — no architecture achieves a breakthrough on this dataset
- The frequency heuristic baseline remains the top performer; CA5 patterns are dominated by frequency statistics
- The Spiking Transformer is the best learned model (R@20=0.5178), narrowing the gap to the frequency baseline to just 0.005
- Window size (W=90 vs W=21) was a bigger lever than architecture choice
- All models show high seed stability (std < 0.003)

## Project Status

| Sprint | Goal | Points | Status |
|--------|------|--------|--------|
| Sprint 1 | Foundation & Data Validation | 13/13 | Complete |
| Sprint 2 | Data Pipeline & Evaluation Harness | 16/16 | Complete |
| Sprint 3 | Baseline Models | 19/19 | Complete |
| Sprint 4 | SNN Phase A (MLP/CNN) | 18/18 | Complete |
| Sprint 5 | SNN Phase B (Spike-GRU) | 13/13 | Complete |
| Sprint 6 | SNN Phase C (Spiking Transformer) | 18/18 | Complete |
| Sprint 7 | Final Report & Closure | 6/6 | Complete |

**Overall: 103/103 points completed across 22 stories in 7 sprints.**

## Setup

### 1. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

### 2. Install PyTorch (environment-specific)

PyTorch must be installed **before** the package, using the correct wheel for your GPU:

**Local development (AMD RX 6600M — ROCm):**

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/rocm6.2
```

> **ROCm note:** The RX 6600M (RDNA2, gfx1032) requires the environment variable
> `HSA_OVERRIDE_GFX_VERSION=10.3.0` to be set before running any GPU code.

**RunPod (NVIDIA B200 — CUDA):**

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

**CI / CPU-only:**

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install package

```bash
# With dev dependencies (recommended)
pip install -e ".[dev]"

# Production only
pip install -e .
```

## CLI Commands

All commands are accessed via the `c5-snn` CLI:

```bash
c5-snn --help              # Show all available commands
```

### Core Commands

| Command | Description | Example |
|---------|-------------|---------|
| `validate-data` | Validate the raw CA5 dataset | `c5-snn validate-data` |
| `train` | Train a single model from config | `c5-snn train --config configs/baseline_gru.yaml` |
| `evaluate` | Evaluate a trained checkpoint | `c5-snn evaluate --checkpoint results/baseline_gru/best_model.pt` |
| `compare` | Multi-seed baseline comparison | `c5-snn compare --config configs/default.yaml --seeds 42,123,7` |

### Phase Commands

| Command | Description | Example |
|---------|-------------|---------|
| `phase-a` | Train Phase A SNN models vs baselines | `c5-snn phase-a --seeds 42,123,7` |
| `phase-b-sweep` | Spike-GRU HP sweep | `c5-snn phase-b-sweep --seeds 42,123,7` |
| `phase-b` | Cumulative comparison (baselines + A + B) | `c5-snn phase-b --seeds 42,123,7` |
| `window-tune` | Window size sweep for Spiking Transformer | `c5-snn window-tune --config configs/snn_phase_c.yaml` |
| `phase-c-sweep` | Spiking Transformer HP sweep (72 configs) | `c5-snn phase-c-sweep --config configs/snn_phase_c.yaml` |
| `final-report` | Generate final 6-model comparison & report | `c5-snn final-report` |

## Reproducing the Full Pipeline

The complete experiment can be reproduced with the following command sequence. Seeds `42,123,7` ensure deterministic results.

```bash
# Step 0: Validate input data
c5-snn validate-data

# Step 1: Baseline comparison (frequency heuristic + GRU, 3 seeds)
c5-snn compare --config configs/default.yaml --seeds 42,123,7 \
  --output results/comparison.json

# Step 2: Phase A — SNN MLP + CNN vs baselines (3 seeds)
c5-snn phase-a --seeds 42,123,7 \
  --output results/phase_a_comparison.json

# Step 3: Phase B — Spike-GRU HP sweep + cumulative comparison
c5-snn phase-b-sweep --seeds 42,123,7 \
  --output results/phase_b_sweep.csv
c5-snn phase-b --seeds 42,123,7 \
  --output results/cumulative_comparison.json

# Step 4: Phase C — Spiking Transformer (requires GPU, ~10 hours)
#   4a: Window size tuning
c5-snn window-tune --config configs/snn_phase_c.yaml \
  --output results/window_tune.csv
#   4b: 72-config HP sweep (top-5 re-run with 3 seeds)
c5-snn phase-c-sweep --config configs/snn_phase_c.yaml \
  --output results/phase_c_sweep.csv

# Step 5: Final report — assembles all results into leaderboard
c5-snn final-report \
  --cumulative results/cumulative_comparison.json \
  --phase-c-top results/phase_c_top5.json \
  --output results/final_comparison.json \
  --report results/final_report.md
```

**Notes:**
- Steps 1-3 run on CPU in ~10 minutes total
- Step 4 requires a GPU (RunPod B200 recommended); ~10 hours for the full 72-config sweep
- Step 5 is an assembly step (no training) — runs instantly from saved JSON files
- All results are saved to `results/` (gitignored); configs and code are version-controlled

## Configuration

All experiments are driven by YAML config files in `configs/`:

| Config | Model | Key Parameters |
|--------|-------|----------------|
| `default.yaml` | Frequency Baseline | W=21 (template for baselines) |
| `baseline_gru.yaml` | GRU Baseline | hidden=128, layers=1 |
| `snn_phase_a_mlp.yaml` | Spiking MLP | hidden=[256,128], beta=0.95, direct |
| `snn_phase_a_cnn.yaml` | Spiking CNN-1D | channels=[64,64], beta=0.95, direct |
| `snn_phase_b.yaml` | Spike-GRU | hidden=128, beta=0.95, direct |
| `snn_phase_c.yaml` | Spiking Transformer | d=64, h=2, l=6, beta=0.95, rate_coded, W=90 |

## Project Structure

```
src/c5_snn/            # Main package
  data/                # Data loading, validation, windowing, splits
  models/              # All model architectures (baselines + SNN)
  training/            # Training loop, metrics, evaluation
  utils/               # Seed, logging, config, device detection
  cli.py               # Click CLI entry point (11 commands)
configs/               # YAML experiment configs (6 files)
data/raw/              # Source CSV (gitignored)
data/processed/        # Generated tensors (gitignored)
results/               # Experiment outputs (gitignored)
tests/                 # pytest test suite (491 tests)
docs/                  # Architecture, sprint plan, stories, sessions
  architecture.md      # Full system architecture (17 sections)
  sprint-plan-*.md     # Sprint plan (7 sprints, 22 stories)
  stories/             # Individual story documents
  sessions/            # Session summaries and startup guides
  project-memory.md    # Persistent project decisions and context
.bmad/                 # BMAD Method sprint tracking
.github/workflows/     # CI pipeline (lint + tests on push/PR)
```

## Development

```bash
# Lint (must pass before commit)
ruff check src/ tests/

# Format
ruff format src/ tests/

# Test (491 tests, must pass before commit)
pytest tests/ -v
```

## CI

GitHub Actions runs automatically on every push and PR to `main`:

- Ruff lint check (zero errors required)
- Full pytest suite (491 tests)
- CPU-only PyTorch (fast, no GPU required)

## Architecture

The pipeline follows a strict sequential pattern with no data leakage:

```text
CSV → validate → window → split → train → evaluate → report
```

- **Task:** Multi-label ranking over 39 part categories (P_1..P_39), output Top-20
- **Loss:** BCEWithLogitsLoss
- **Splits:** 70/15/15 train/val/test, time-ordered (no shuffling)
- **Early stopping:** On `val_recall_at_20` with patience=10
- **Reproducibility:** `set_global_seed()` called before every stochastic operation; config snapshots + pip freeze saved with every checkpoint

See `docs/architecture.md` for the full 17-section architecture document.

## License

MIT
