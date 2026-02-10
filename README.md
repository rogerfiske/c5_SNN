# c5_SNN

Spiking Neural Network time-series forecasting pipeline for CA5 event prediction.

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
c5-snn validate-data --data-path data/raw/CA5_matrix_binary.csv

# Train a model from config
c5-snn train --config configs/baseline_gru.yaml

# Evaluate a checkpoint
c5-snn evaluate --checkpoint results/baseline_gru/best_model.pt

# Predict top-20 for a date
c5-snn predict --asof 2024-01-15
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
tests/             # pytest test suite
```

## Development

```bash
# Lint
ruff check src/ tests/

# Format
ruff format src/ tests/

# Test
pytest tests/ -v
```
