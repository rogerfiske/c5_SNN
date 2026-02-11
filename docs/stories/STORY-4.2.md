# STORY-4.2: Spiking MLP Model

**Epic:** Epic 4 — SNN Phase A
**Priority:** Must Have
**Story Points:** 5
**Status:** Completed
**Assigned To:** ai_dev_agent
**Created:** 2026-02-11
**Sprint:** 4

---

## User Story

As a researcher,
I want a spiking MLP using LIF neurons and surrogate gradients,
so that I can evaluate the simplest SNN architecture on the CA5 prediction task.

---

## Description

### Background

With the SpikeEncoder (STORY-4.1) complete, the project can now build the first actual SNN model. The Spiking MLP is the simplest possible SNN architecture: it flattens the windowed spike input and passes it through fully-connected spiking layers using Leaky Integrate-and-Fire (LIF) neurons from snnTorch.

This model serves as the Phase A "simplest SNN" — establishing whether a basic spiking architecture can learn the CA5 prediction task at all. Even if it underperforms the GRU baseline, it validates the full SNN training pipeline (encoding → spiking layers → readout → BCEWithLogitsLoss → backprop with surrogate gradients).

### Scope

**In scope:**
- `SpikingMLP` class in `src/c5_snn/models/snn_models.py` subclassing `BaseModel`
- Architecture: SpikeEncoder → flatten → spiking FC layers (snnTorch `Leaky`) → linear readout → 39 logits
- Config-driven: beta, hidden_sizes, surrogate gradient, encoding mode, temporal aggregation
- Registration in `MODEL_REGISTRY` as `"spiking_mlp"`
- Config file `configs/snn_phase_a_mlp.yaml`
- Unit tests for shape, backward pass, config parsing, hidden state reset

**Out of scope:**
- Spiking 1D-CNN model (STORY-4.3)
- Training and evaluation on real data (STORY-4.4)
- Hyperparameter tuning (Sprint 5)
- Cardinality penalty implementation (future enhancement — config key reserved)

### User Flow

1. Researcher creates `configs/snn_phase_a_mlp.yaml` specifying model type, hidden sizes, beta, surrogate
2. Researcher runs `c5_snn train --config configs/snn_phase_a_mlp.yaml`
3. Trainer loads config → `get_model(config)` → creates `SpikingMLP` from registry
4. SpikingMLP composes SpikeEncoder internally for input encoding
5. Training loop: forward → BCEWithLogitsLoss → backward (surrogate gradients) → optimizer step
6. Evaluation uses same `evaluate_model()` pipeline as baselines

---

## Acceptance Criteria

- [ ] `SpikingMLP(BaseModel)` class in `src/c5_snn/models/snn_models.py`
- [ ] Architecture: SpikeEncoder → flatten `(T, batch, W*39)` → spiking FC layers → temporal aggregation → linear readout → `(batch, 39)` logits
- [ ] Uses `snntorch.Leaky` LIF neurons with configurable `beta` (default 0.95)
- [ ] Uses `snntorch.surrogate.fast_sigmoid` as default surrogate gradient
- [ ] Configurable `hidden_sizes` list (default `[256, 128]`) — supports arbitrary depth
- [ ] Membrane potentials initialized to zero at start of each forward pass (batch independence)
- [ ] Temporal aggregation: mean over time steps (default)
- [ ] Forward pass contract: `(batch, W, 39)` → `(batch, 39)` logits (matches BaseModel)
- [ ] Backward pass completes with no errors (surrogate gradients work)
- [ ] Registered as `"spiking_mlp"` in `MODEL_REGISTRY`; `get_model(config)` creates it
- [ ] Config file `configs/snn_phase_a_mlp.yaml` with all required fields
- [ ] Unit tests: forward shape, backward completes, config parsing, default values, hidden state reset
- [ ] `ruff check` passes and `pytest` all green

---

## Technical Notes

### Components

- **New file:** `src/c5_snn/models/snn_models.py` — `SpikingMLP` class (and later STORY-4.3 CNN)
- **Modified file:** `src/c5_snn/models/__init__.py` — add `SpikingMLP` export + registry
- **New file:** `configs/snn_phase_a_mlp.yaml` — experiment config
- **New test file:** `tests/test_snn_models.py` — SNN model unit tests
- **Existing deps:** `SpikeEncoder`, `BaseModel`, `MODEL_REGISTRY`, `snntorch.Leaky`, `snntorch.surrogate`

### snnTorch API Pattern (Verified)

```python
from snntorch import surrogate
import snntorch

# Use init_hidden=False with manual membrane potential management
fast_sig = surrogate.fast_sigmoid(slope=25)
lif = snntorch.Leaky(beta=0.95, spike_grad=fast_sig)

# Forward: loop over T timesteps
mem = torch.zeros(batch, hidden_size)  # Reset each forward call
spks = []
for t in range(T):
    cur = fc(x[t])           # (batch, hidden_size)
    spk, mem = lif(cur, mem)  # Returns (spike, membrane)
    spks.append(spk)
out = torch.stack(spks)  # (T, batch, hidden_size)
```

**Key behaviors verified:**
- `snntorch.Leaky` with `init_hidden=False` returns `(spk, mem)` tuple — clean, predictable
- Spikes are binary (0 or 1) with surrogate gradients for backprop
- `torch.stack(spks).sum()` backward completes correctly
- `surrogate.fast_sigmoid(slope=25)` is the default/recommended choice

### SpikingMLP Architecture

```python
class SpikingMLP(BaseModel):
    """Spiking MLP with LIF neurons for CA5 prediction.

    Architecture:
        SpikeEncoder → flatten(T, batch, W*39) → [FC + LIF]×N → mean(T) → Linear → (batch, 39)
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        model_cfg = config.get("model", {})

        # Encoding
        self.encoder = SpikeEncoder(config)

        # Architecture params
        self.hidden_sizes = model_cfg.get("hidden_sizes", [256, 128])
        beta = float(model_cfg.get("beta", 0.95))
        spike_grad = surrogate.fast_sigmoid(slope=25)

        # Input size: W * 39 (flattened spatial dims)
        window_size = config.get("data", {}).get("window_size", 21)
        n_features = 39
        self.input_size = window_size * n_features

        # Build spiking layers: pairs of (Linear, Leaky)
        self.fc_layers = nn.ModuleList()
        self.lif_layers = nn.ModuleList()
        in_size = self.input_size
        for h_size in self.hidden_sizes:
            self.fc_layers.append(nn.Linear(in_size, h_size))
            self.lif_layers.append(snntorch.Leaky(beta=beta, spike_grad=spike_grad))
            in_size = h_size

        # Readout layer (non-spiking)
        self.readout = nn.Linear(in_size, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, W, 39)
        spikes = self.encoder(x)  # (T, batch, W, 39)
        T, B = spikes.size(0), spikes.size(1)

        # Flatten spatial: (T, batch, W*39)
        spikes = spikes.view(T, B, -1)

        # Initialize membrane potentials (reset each forward call)
        mems = [torch.zeros(B, h, device=x.device) for h in self.hidden_sizes]

        # Temporal loop through spiking layers
        spike_rec = []
        for t in range(T):
            cur = spikes[t]
            for i, (fc, lif) in enumerate(zip(self.fc_layers, self.lif_layers)):
                cur = fc(cur)
                cur, mems[i] = lif(cur, mems[i])
            spike_rec.append(cur)

        # Aggregate over time: mean of output spikes
        spike_out = torch.stack(spike_rec)  # (T, B, last_hidden)
        agg = spike_out.mean(dim=0)         # (B, last_hidden)

        # Linear readout
        return self.readout(agg)            # (B, 39) logits
```

### Shape Transformation Table

| Stage | Shape (direct, T=1) | Shape (rate_coded, T=10) |
|-------|---------------------|--------------------------|
| Input | `(B, 21, 39)` | `(B, 21, 39)` |
| After SpikeEncoder | `(1, B, 21, 39)` | `(10, B, 21, 39)` |
| Flatten spatial | `(1, B, 819)` | `(10, B, 819)` |
| After FC+LIF layer 1 | `(1, B, 256)` | `(10, B, 256)` |
| After FC+LIF layer 2 | `(1, B, 128)` | `(10, B, 128)` |
| Mean over time | `(B, 128)` | `(B, 128)` |
| Readout | `(B, 39)` | `(B, 39)` |

### Config File

```yaml
# configs/snn_phase_a_mlp.yaml
experiment:
  name: "snn_phase_a_mlp"
  seed: 42

data:
  raw_path: "data/raw/CA5_matrix_binary.csv"
  window_size: 21
  split_ratios: [0.70, 0.15, 0.15]
  batch_size: 64

model:
  type: "spiking_mlp"
  encoding: "direct"
  timesteps: 10
  hidden_sizes: [256, 128]
  beta: 0.95
  surrogate: "fast_sigmoid"

training:
  epochs: 100
  learning_rate: 0.001
  optimizer: "adam"
  early_stopping_patience: 10
  early_stopping_metric: "val_recall_at_20"

output:
  dir: "results/snn_phase_a_mlp"

log_level: "INFO"
```

### Trainer Compatibility

The existing Trainer class from STORY-3.3 handles SpikingMLP without modification:
- Has learnable parameters → Adam optimizer created
- `BCEWithLogitsLoss` already configured
- Forward `(batch, W, 39)` → `(batch, 39)` matches Trainer expectations
- Early stopping on `val_recall_at_20` works unchanged
- Checkpoint saving works (model_state_dict includes all spiking layer states)

### Edge Cases

- **T=1 (direct mode):** Should work — single timestep means no temporal aggregation benefit, but shape logic holds
- **Single hidden layer:** `hidden_sizes: [256]` should work with arbitrary depth
- **Large hidden sizes:** Memory scales with `T * batch * hidden_size` — monitor for OOM
- **All-zeros input:** LIF neurons never spike → readout gets zeros → logits near bias values
- **beta=0 or beta=1:** Extreme values — beta=0 means no memory (instant leak), beta=1 means no decay (integrator)

### Architecture References

- Section 5.2: Models module — SpikingMLP listed as Phase A model
- Section 13.3 Rule #5: Common model interface — must follow BaseModel contract
- Section 13.4: snnTorch guidance — reset hidden states, init_hidden patterns
- Section 4.4: Config schema for SNN models
- SeqSNN paper: beta=0.95, fast_sigmoid surrogate as starting defaults

---

## Dependencies

**Prerequisite Stories:**
- STORY-4.1: snnTorch Integration & Spike Encoding (SpikeEncoder class)
- STORY-3.3: Training Loop & train CLI (Trainer compatibility)
- STORY-2.3: Evaluation Harness & Metrics (evaluate_model)

**Blocked Stories:**
- STORY-4.4: Phase A Training & Comparison (trains SpikingMLP on real data)

**External Dependencies:**
- snnTorch 0.9.1 installed (verified in STORY-4.1)

---

## Definition of Done

- [ ] `SpikingMLP` implemented in `src/c5_snn/models/snn_models.py`
- [ ] Registered as `"spiking_mlp"` in `MODEL_REGISTRY`
- [ ] `src/c5_snn/models/__init__.py` updated with export
- [ ] Config file `configs/snn_phase_a_mlp.yaml` created
- [ ] Unit tests in `tests/test_snn_models.py`:
  - [ ] Forward shape: `(4, 21, 39)` → `(4, 39)`
  - [ ] Backward completes (surrogate gradients)
  - [ ] Config parsing (hidden_sizes, beta, encoding)
  - [ ] Default config values
  - [ ] Different window sizes and batch sizes
  - [ ] Single hidden layer configuration
  - [ ] Model has learnable parameters
  - [ ] Registered in MODEL_REGISTRY and created via get_model
- [ ] `ruff check src/ tests/` passes with zero errors
- [ ] `pytest tests/ -v` passes (all existing + new tests)
- [ ] Acceptance criteria validated (all checked)
- [ ] Code committed to `main` branch and pushed

---

## Story Points Breakdown

- **SpikingMLP class (encoder + spiking layers + readout):** 2 points
- **Config file + model registration:** 1 point
- **Testing (shape, backward, config, edge cases):** 2 points
- **Total:** 5 points

**Rationale:** Moderate complexity. The architecture follows a known pattern (MLP with LIF neurons), but integrating snnTorch's temporal loop with our BaseModel contract requires careful shape management. Testing surrogate gradient backward pass and verifying Trainer compatibility adds effort. This is the project's first trainable SNN model.

---

## Additional Notes

- The file is named `snn_models.py` (not `spiking_mlp.py`) because STORY-4.3 will add `SpikingCNN1D` to the same file
- Temporal aggregation uses mean over time by default; other options (final spike, sum) can be explored during HP tuning in Sprint 5
- The `init_hidden=False` pattern with manual `torch.zeros` membrane init is preferred over `init_hidden=True` for clarity and predictable behavior
- SpikingMLP's `forward()` resets membrane potentials at the start of each call — no separate `reset_hidden()` method needed since state is local to forward
- The surrogate gradient (`fast_sigmoid`) enables standard PyTorch autograd to backpropagate through the non-differentiable spike function

---

## Progress Tracking

**Status History:**
- 2026-02-11: Created by Scrum Master (AI)
- 2026-02-11: Started by Developer (AI)
- 2026-02-11: Completed by Developer (AI)

**Actual Effort:** 5 points (matched estimate)

**Implementation Notes:**
- Created `src/c5_snn/models/snn_models.py` with `SpikingMLP(BaseModel)` class
- Architecture: SpikeEncoder → flatten → [FC + LIF]×N → mean(T) → Linear readout
- Uses `snntorch.Leaky` with `init_hidden=False` and manual membrane potential management
- `surrogate.fast_sigmoid(slope=25)` enables backprop through non-differentiable spikes
- Registered as `"spiking_mlp"` in MODEL_REGISTRY; `get_model(config)` creates it
- Config file `configs/snn_phase_a_mlp.yaml` created
- Key finding: with direct encoding (T=1), LIF neurons may not spike due to single timestep — rate_coded with T>1 allows membrane accumulation
- 38 new tests (311 total), all passing
- All 13 acceptance criteria validated

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
