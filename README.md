# c5_SNN

Spiking Neural Network time-series forecasting pipeline for CA5 event prediction.

## Project Status

| Sprint | Goal | Points | Status |
|--------|------|--------|--------|
| Sprint 1 | Foundation & Data Validation | 13/13 | Complete |
| Sprint 2 | Data Pipeline & Evaluation Harness | 0/16 | Next |
| Sprint 3 | Baseline Models | 0/19 | — |
| Sprint 4 | SNN Phase A (MLP/CNN) | 0/18 | — |
| Sprint 5 | SNN Phase B (Spike-GRU) | 0/13 | — |
| Sprint 6 | SNN Phase C (Spiking Transformer) | 0/18 | — |
| Sprint 7 | Final Report & Closure | 0/6 | — |

**Overall: 13/103 points completed**

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
> Add to your shell profile or set it inline:
>
> ```bash
> export HSA_OVERRIDE_GFX_VERSION=10.3.0
> ```

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
# With dev dependencies (recommended for development)
pip install -e ".[dev]"

# Production only
pip install -e .
```

## Usage

```bash
# Show available commands
c5-snn --help

# Validate the CA5 dataset
c5-snn validate-data
c5-snn validate-data --data-path data/raw/CA5_matrix_binary.csv
```

## Project Structure

```
src/c5_snn/        # Main package
  data/            # Data loading, validation, windowing, splits
  models/          # All model architectures (baselines + SNN)
  training/        # Training loop, metrics, evaluation
  utils/           # Seed, logging, config, device detection
  cli.py           # Click CLI entry point
configs/           # YAML experiment configs
data/raw/          # Source CSV (gitignored)
data/processed/    # Generated tensors (gitignored)
results/           # Experiment outputs (gitignored)
tests/             # pytest test suite (26 tests)
docs/              # Architecture, sprint plan, stories, sessions
.github/workflows/ # CI pipeline (lint + tests on push/PR)
```

## CI

GitHub Actions runs automatically on every push and PR to `main`:

- Ruff lint check
- Full pytest suite (26 tests)
- CPU-only PyTorch (fast, no GPU required)

## Development

```bash
# Lint
ruff check src/ tests/

# Format
ruff format src/ tests/

# Test
pytest tests/ -v
```
