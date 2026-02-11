# STORY-3.2: ANN GRU Baseline

**Epic:** Epic 3 — Baseline Models
**Priority:** Must Have
**Story Points:** 5
**Status:** Completed
**Assigned To:** ai_dev_agent
**Created:** 2026-02-11
**Sprint:** 3

---

## User Story

As a researcher,
I want a conventional GRU-based neural network baseline,
so that I have a strong learned-model benchmark for SNN comparison.

---

## Description

### Background

With the FrequencyBaseline (STORY-3.1) providing a non-learned lower bound, the project needs a strong conventional neural network baseline to serve as the "beat this" target for all SNN models. A GRU (Gated Recurrent Unit) is the standard choice for sequence modeling — it processes the windowed time-series input and learns to predict which parts appear next.

This model is the first _trainable_ model in the pipeline. It establishes the GRU architecture in `baselines.py`, registers it in `MODEL_REGISTRY`, and creates its own experiment config YAML. The actual training loop and `train` CLI are handled by STORY-3.3, but the model must be fully ready for training: correct shapes, working backward pass, and config-driven hyperparameters.

### Scope

**In scope:**
- `GRUBaseline(BaseModel)` class in `src/c5_snn/models/baselines.py`
- GRU encoder -> final hidden state -> linear projection -> 39 logits
- Configurable hidden size, number of layers, and dropout via YAML config
- Registration as `"gru_baseline"` in `MODEL_REGISTRY`
- Experiment config file `configs/baseline_gru.yaml`
- Unit tests for forward pass shape, backward pass, and config-driven construction

**Out of scope:**
- Training loop and `train` CLI command (STORY-3.3)
- Running evaluation on real data (STORY-3.4)
- Spike encoding or SNN models (Sprint 4+)
- Hyperparameter tuning (Sprint 5)

### User Flow

1. User creates `configs/baseline_gru.yaml` specifying `model.type: "gru_baseline"` with hidden size, layers, and dropout
2. System calls `get_model(config)` which looks up `"gru_baseline"` in `MODEL_REGISTRY`
3. System instantiates `GRUBaseline(config)` with hyperparameters from config
4. Training loop (STORY-3.3) feeds windowed batches `(batch, W, 39)` through `model.forward(x)`
5. Model returns `(batch, 39)` logits; loss is computed via `BCEWithLogitsLoss`
6. Optimizer updates weights via `.backward()` and `.step()`
7. After training, evaluation harness computes metrics on test split

---

## Acceptance Criteria

- [ ] `GRUBaseline(BaseModel)` implemented in `src/c5_snn/models/baselines.py` with configurable `hidden_size`, `num_layers`, and `dropout`
- [ ] Architecture: `nn.GRU` encoder processes `(batch, W, 39)` input, extracts final hidden state, projects through `nn.Linear` to 39 logits
- [ ] Constructor reads all hyperparameters from `config["model"]` with sensible defaults (`hidden_size=128`, `num_layers=1`, `dropout=0.0`)
- [ ] Registered in `MODEL_REGISTRY` as `"gru_baseline"`
- [ ] `get_model({"model": {"type": "gru_baseline"}})` returns a `GRUBaseline` instance
- [ ] Forward pass: input `(batch, W, 39)` -> output `(batch, 39)` for any batch size and window size
- [ ] Model has learnable parameters (`list(model.parameters())` is non-empty)
- [ ] Backward pass completes without error: `loss.backward()` on `BCEWithLogitsLoss` output
- [ ] Device-agnostic: model and tensors work with `torch.device("cpu")` (and GPU when available)
- [ ] Config file `configs/baseline_gru.yaml` created with complete experiment config
- [ ] Unit tests verify:
  - Forward shape `(4, 21, 39) -> (4, 39)`
  - Forward shape with different window size `(8, 7, 39) -> (8, 39)`
  - Backward pass completes without error
  - Model is instance of `BaseModel` and `nn.Module`
  - `get_model` returns `GRUBaseline` for correct config
  - Custom hyperparameters from config are applied
  - Model has learnable parameters
- [ ] Uses logging from STORY-1.2 (no `print()` statements)

---

## Technical Notes

### Components

- **Modified file:** `src/c5_snn/models/baselines.py` — add `GRUBaseline` class
- **Modified file:** `src/c5_snn/models/__init__.py` — add `GRUBaseline` to exports
- **New file:** `configs/baseline_gru.yaml` — experiment config for GRU baseline
- **Modified file:** `tests/test_baselines.py` — add GRU baseline tests

### GRUBaseline Architecture

```
Input: (batch, W, 39)
    |
    v
nn.GRU(input_size=39, hidden_size=H, num_layers=L, dropout=D, batch_first=True)
    |
    v
Take final hidden state: h_n[-1] -> (batch, H)
    |
    v
nn.Linear(H, 39)
    |
    v
Output: (batch, 39) logits
```

**Key details:**
- `batch_first=True` so input shape is `(batch, seq_len, input_size)`
- Use `h_n[-1]` (last layer's final hidden state) as the sequence summary
- Single `nn.Linear` projection from hidden size to 39 output classes
- No activation on output — raw logits for `BCEWithLogitsLoss`

### Constructor Signature

```python
class GRUBaseline(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__()
        model_cfg = config.get("model", {})
        self.hidden_size = int(model_cfg.get("hidden_size", 128))
        self.num_layers = int(model_cfg.get("num_layers", 1))
        self.dropout = float(model_cfg.get("dropout", 0.0))

        self.gru = nn.GRU(
            input_size=39,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(self.hidden_size, 39)
```

**Note on dropout:** PyTorch's `nn.GRU` ignores the `dropout` parameter when `num_layers=1` and issues a warning. Guard by only applying dropout when `num_layers > 1`.

### Forward Method

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: (batch, W, 39)
    output, h_n = self.gru(x)          # output: (batch, W, H), h_n: (L, batch, H)
    last_hidden = h_n[-1]              # (batch, H) — last layer's final state
    logits = self.fc(last_hidden)      # (batch, 39)
    return logits
```

### Config File: `configs/baseline_gru.yaml`

```yaml
experiment:
  name: "baseline_gru"
  seed: 42

data:
  raw_path: "data/raw/CA5_matrix_binary.csv"
  window_size: 21
  split_ratios: [0.70, 0.15, 0.15]
  batch_size: 64

model:
  type: "gru_baseline"
  hidden_size: 128
  num_layers: 1
  dropout: 0.0

training:
  epochs: 100
  learning_rate: 0.001
  optimizer: "adam"
  early_stopping_patience: 10
  early_stopping_metric: "val_recall_at_20"

output:
  dir: "results/baseline_gru"

log_level: "INFO"
```

### Architecture References

- Section 2.4 (Strategy Pattern, Registry Pattern)
- Section 5.2 (Models Module — `baselines.py` houses `GRUBaseline`)
- Section 5.2 Model Registry (`"gru_baseline"` key)
- Section 5.3 (Training: `BCEWithLogitsLoss`, Adam optimizer)
- Section 13.3 Rule 5 (Common Model Interface)
- Section 13.3 Rule 2 (Device-Agnostic Tensors)
- Section 14.2 (Unit test: forward shape + backward completes)
- FR4 (baseline models)

### Edge Cases

- Single-step window (W=1): GRU still processes one time step — should work correctly
- Batch size 1: Must produce `(1, 39)` output
- `num_layers > 1` with dropout: Dropout applied between GRU layers (not after final layer)
- Very large hidden size: Memory consideration but no code guard needed
- `num_layers=1` with `dropout > 0`: Must suppress dropout to avoid PyTorch warning

---

## Dependencies

**Prerequisite Stories:**
- STORY-3.1: Frequency/Recency Heuristic (provides `BaseModel`, `MODEL_REGISTRY`, `get_model()`)
- STORY-1.2: Logging, Seed Management & Config (logging, `ConfigError`, `load_config`)

**Blocked Stories:**
- STORY-3.3: Training Loop & train CLI (needs `GRUBaseline` as the first trainable model)
- STORY-3.4: Baseline Results & Comparison (needs trained GRU baseline for evaluation)

**External Dependencies:**
- None (PyTorch `nn.GRU` is part of core PyTorch)

---

## Definition of Done

- [ ] `GRUBaseline` class implemented in `src/c5_snn/models/baselines.py`
- [ ] Registered as `"gru_baseline"` in `MODEL_REGISTRY`
- [ ] `src/c5_snn/models/__init__.py` updated with `GRUBaseline` export
- [ ] `configs/baseline_gru.yaml` created with complete experiment config
- [ ] Unit tests in `tests/test_baselines.py` written and passing:
  - [ ] Forward shape correct for multiple input sizes
  - [ ] Backward pass completes
  - [ ] Model is `BaseModel` and `nn.Module`
  - [ ] `get_model` returns correct type
  - [ ] Custom config hyperparameters applied
  - [ ] Has learnable parameters
- [ ] `ruff check src/ tests/` passes with zero errors
- [ ] `pytest tests/ -v` passes (all existing + new tests)
- [ ] CI green on GitHub Actions
- [ ] Acceptance criteria validated (all checked)
- [ ] Code committed to `main` branch and pushed

---

## Story Points Breakdown

- **GRUBaseline implementation:** 2 points (GRU encoder + linear head, config parsing)
- **Config file:** 1 point (YAML experiment config)
- **Testing:** 2 points (forward/backward, shape checks, registry, config-driven construction)
- **Total:** 5 points

**Rationale:** Moderate complexity. The GRU architecture is standard PyTorch but this is the first learnable model, so extra care is needed to ensure the forward/backward contract works correctly with the `BaseModel` interface. Testing must verify gradient flow, not just output shapes.

---

## Additional Notes

- This is the first trainable model in the project — all prior models (FrequencyBaseline) had no parameters
- The `BCEWithLogitsLoss` is not defined in this story — it belongs to the training loop (STORY-3.3). This story only ensures `loss.backward()` works in unit tests
- Default `hidden_size=128` is a reasonable starting point; hyperparameter tuning happens in STORY-3.4 or later sprints
- The architecture mentions training GRU with 3 seeds (STORY-3.4) and using RunPod if training exceeds 20 minutes locally
- `GRUBaseline` goes in the same `baselines.py` file as `FrequencyBaseline` — both are baseline models per the architecture source tree

---

## Progress Tracking

**Status History:**
- 2026-02-11: Created by Scrum Master (AI)
- 2026-02-11: Started by Developer (AI)
- 2026-02-11: Completed by Developer (AI)

**Actual Effort:** 5 points (matched estimate)

**Implementation Notes:**
- `GRUBaseline(BaseModel)` with `nn.GRU` encoder + `nn.Linear` projection
- `h_n[-1]` extracts last layer's final hidden state as sequence summary
- Dropout guard: only applied when `num_layers > 1` to avoid PyTorch warning
- Default hyperparameters: hidden_size=128, num_layers=1, dropout=0.0
- ~69k learnable parameters (default config)
- 18 new tests (178 total): shapes, backward/gradient flow, optimizer step, registry, properties
- Config file `configs/baseline_gru.yaml` with complete experiment specification

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
