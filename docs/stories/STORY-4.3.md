# STORY-4.3: Spiking 1D-CNN Model

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
I want a spiking 1D-CNN with temporal convolutions and LIF neurons,
so that I can compare a convolutional SNN architecture against the spiking MLP on the CA5 prediction task.

---

## Description

### Background

With the SpikingMLP (STORY-4.2) complete, the project now adds a second SNN architecture for Phase A comparison. The Spiking 1D-CNN applies temporal convolutions along the window dimension (W=21 time steps) before spiking neurons, allowing the model to learn local temporal patterns in the CA5 event sequences — something the SpikingMLP cannot do since it flattens all spatial structure.

Conv1d operates on the 39-feature channels across the window length, extracting local temporal features (e.g., "part X appeared in 3 consecutive steps") before LIF neurons produce spike outputs. This is analogous to how 1D-CNNs work in time-series classification, but with spiking neurons replacing ReLU activations.

The Spiking CNN shares the same SpikeEncoder front-end, surrogate gradient training, and temporal aggregation as the SpikingMLP. The key difference is the use of `nn.Conv1d` + `snntorch.Leaky` pairs instead of `nn.Linear` + `snntorch.Leaky` pairs.

### Scope

**In scope:**
- `SpikingCNN1D` class in `src/c5_snn/models/snn_models.py` subclassing `BaseModel`
- Architecture: SpikeEncoder → permute → Conv1d+LIF layers → global avg pool → temporal aggregation → linear readout → 39 logits
- Config-driven: channels, kernel_sizes, beta, encoding mode, temporal aggregation
- Registration in `MODEL_REGISTRY` as `"spiking_cnn1d"`
- Config file `configs/snn_phase_a_cnn.yaml`
- Unit tests added to `tests/test_snn_models.py`

**Out of scope:**
- Spiking MLP modifications (STORY-4.2, already complete)
- Training and evaluation on real data (STORY-4.4)
- Hyperparameter tuning (Sprint 5)
- MaxPool or stride-based downsampling (keep it simple — global avg pool only)
- Dilated or causal convolutions (future enhancement)

### User Flow

1. Researcher creates `configs/snn_phase_a_cnn.yaml` specifying model type, channels, kernel sizes, beta
2. Researcher runs `c5_snn train --config configs/snn_phase_a_cnn.yaml`
3. Trainer loads config → `get_model(config)` → creates `SpikingCNN1D` from registry
4. SpikingCNN1D composes SpikeEncoder internally for input encoding
5. Training loop: forward → BCEWithLogitsLoss → backward (surrogate gradients) → optimizer step
6. Evaluation uses same `evaluate_model()` pipeline as baselines and SpikingMLP

---

## Acceptance Criteria

- [ ] `SpikingCNN1D(BaseModel)` class in `src/c5_snn/models/snn_models.py`
- [ ] Architecture: SpikeEncoder → permute to `(T*B, 39, W)` → Conv1d+LIF layers → global avg pool over W → temporal aggregation (mean over T) → linear readout → `(batch, 39)` logits
- [ ] Uses `snntorch.Leaky` LIF neurons with configurable `beta` (default 0.95)
- [ ] Uses `snntorch.surrogate.fast_sigmoid` as default surrogate gradient
- [ ] Configurable `channels` list (default `[64, 64]`) — supports arbitrary depth
- [ ] Configurable `kernel_sizes` list (default `[3, 3]`) — one per conv layer
- [ ] Conv1d uses `padding = kernel_size // 2` to preserve spatial dimension (same padding)
- [ ] Membrane potentials initialized to zero at start of each forward pass (batch independence)
- [ ] Temporal aggregation: mean over time steps (default)
- [ ] Forward pass contract: `(batch, W, 39)` → `(batch, 39)` logits (matches BaseModel)
- [ ] Backward pass completes with no errors (surrogate gradients work)
- [ ] Registered as `"spiking_cnn1d"` in `MODEL_REGISTRY`; `get_model(config)` creates it
- [ ] Config file `configs/snn_phase_a_cnn.yaml` with all required fields
- [ ] Unit tests: forward shape, backward completes, config parsing, default values, different window sizes, learnable parameters, registry
- [ ] `ruff check` passes and `pytest` all green

---

## Technical Notes

### Components

- **Modified file:** `src/c5_snn/models/snn_models.py` — add `SpikingCNN1D` class (alongside `SpikingMLP`)
- **Modified file:** `src/c5_snn/models/__init__.py` — add `SpikingCNN1D` export + registry
- **New file:** `configs/snn_phase_a_cnn.yaml` — experiment config
- **Modified test file:** `tests/test_snn_models.py` — add SpikingCNN1D unit tests
- **Existing deps:** `SpikeEncoder`, `BaseModel`, `MODEL_REGISTRY`, `snntorch.Leaky`, `snntorch.surrogate`

### snnTorch Conv1d Pattern (Verified)

```python
import torch
import snntorch
from snntorch import surrogate

spike_grad = surrogate.fast_sigmoid(slope=25)

# Conv1d + LIF pair
conv = torch.nn.Conv1d(in_channels=39, out_channels=64, kernel_size=3, padding=1)
lif = snntorch.Leaky(beta=0.95, spike_grad=spike_grad)

# LIF works on 3D tensors: (batch, channels, length)
# Membrane potential matches conv output shape
mem = torch.zeros(batch, 64, W)  # (B, C_out, W)
spk, mem = lif(conv(x), mem)     # Both (B, 64, W)
```

**Key behaviors verified:**
- `snntorch.Leaky` works with 3D Conv1d output tensors — no need to flatten
- Membrane potential shape must match Conv1d output: `(batch, out_channels, W)`
- Backward pass through Conv1d + LIF with surrogate gradients completes correctly
- Global average pool over W after spiking preserves channel information

### SpikingCNN1D Architecture

```python
class SpikingCNN1D(BaseModel):
    """Spiking 1D-CNN with temporal convolutions for CA5 prediction.

    Architecture:
        SpikeEncoder → permute(T*B, 39, W) → [Conv1d + LIF]×N → pool(W) → mean(T) → Linear → (B, 39)
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        model_cfg = config.get("model", {})

        # Encoding
        self.encoder = SpikeEncoder(config)

        # Architecture params
        self.channels = list(model_cfg.get("channels", [64, 64]))
        self.kernel_sizes = list(model_cfg.get("kernel_sizes", [3, 3]))
        beta = float(model_cfg.get("beta", 0.95))
        spike_grad = surrogate.fast_sigmoid(slope=25)

        # Build conv + LIF layer pairs
        self.conv_layers = nn.ModuleList()
        self.lif_layers = nn.ModuleList()
        in_channels = 39  # N_FEATURES
        for out_ch, ks in zip(self.channels, self.kernel_sizes):
            self.conv_layers.append(
                nn.Conv1d(in_channels, out_ch, kernel_size=ks, padding=ks // 2)
            )
            self.lif_layers.append(snntorch.Leaky(beta=beta, spike_grad=spike_grad))
            in_channels = out_ch

        # Readout: pool over W → linear → 39 logits
        self.readout = nn.Linear(in_channels, 39)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, W, 39)
        spikes = self.encoder(x)  # (T, B, W, 39)
        T, B, W = spikes.size(0), spikes.size(1), spikes.size(2)

        # Initialize membrane potentials: (B, C_out, W) per layer
        mems = [
            torch.zeros(B, ch, W, device=x.device) for ch in self.channels
        ]

        # Temporal loop
        spike_rec = []
        for t in range(T):
            # Permute for Conv1d: (B, W, 39) -> (B, 39, W)
            cur = spikes[t].permute(0, 2, 1)

            for i, (conv, lif) in enumerate(zip(self.conv_layers, self.lif_layers)):
                cur = conv(cur)              # (B, C_out, W)
                cur, mems[i] = lif(cur, mems[i])

            # Global average pool over W: (B, C_out, W) -> (B, C_out)
            pooled = cur.mean(dim=-1)
            spike_rec.append(pooled)

        # Aggregate over time: mean
        spike_out = torch.stack(spike_rec)  # (T, B, last_channel)
        agg = spike_out.mean(dim=0)         # (B, last_channel)

        return self.readout(agg)            # (B, 39) logits
```

### Shape Transformation Table

| Stage | Shape (direct, T=1) | Shape (rate_coded, T=10) |
|-------|---------------------|--------------------------|
| Input | `(B, 21, 39)` | `(B, 21, 39)` |
| After SpikeEncoder | `(1, B, 21, 39)` | `(10, B, 21, 39)` |
| Permute for Conv1d | `(B, 39, 21)` | `(B, 39, 21)` per timestep |
| After Conv1d+LIF layer 1 | `(B, 64, 21)` | `(B, 64, 21)` per timestep |
| After Conv1d+LIF layer 2 | `(B, 64, 21)` | `(B, 64, 21)` per timestep |
| Global avg pool over W | `(B, 64)` | `(B, 64)` per timestep |
| Stack over T | `(1, B, 64)` | `(10, B, 64)` |
| Mean over time | `(B, 64)` | `(B, 64)` |
| Readout | `(B, 39)` | `(B, 39)` |

### Config File

```yaml
# configs/snn_phase_a_cnn.yaml
experiment:
  name: "snn_phase_a_cnn"
  seed: 42

data:
  raw_path: "data/raw/CA5_matrix_binary.csv"
  window_size: 21
  split_ratios: [0.70, 0.15, 0.15]
  batch_size: 64

model:
  type: "spiking_cnn1d"
  encoding: "direct"
  timesteps: 10
  channels: [64, 64]
  kernel_sizes: [3, 3]
  beta: 0.95
  surrogate: "fast_sigmoid"

training:
  epochs: 100
  learning_rate: 0.001
  optimizer: "adam"
  early_stopping_patience: 10
  early_stopping_metric: "val_recall_at_20"

output:
  dir: "results/snn_phase_a_cnn"

log_level: "INFO"
```

### Trainer Compatibility

The existing Trainer class from STORY-3.3 handles SpikingCNN1D without modification:
- Has learnable parameters → Adam optimizer created
- `BCEWithLogitsLoss` already configured
- Forward `(batch, W, 39)` → `(batch, 39)` matches Trainer expectations
- Early stopping on `val_recall_at_20` works unchanged
- Checkpoint saving works (model_state_dict includes all conv + spiking layer states)

### Edge Cases

- **T=1 (direct mode):** Single timestep — conv still extracts spatial patterns, just no temporal membrane accumulation
- **Single conv layer:** `channels: [64]`, `kernel_sizes: [3]` should work
- **Large kernels:** `kernel_size=7` with `padding=3` — preserve W dimension
- **Odd vs even kernel sizes:** Padding `ks // 2` works correctly for odd kernels (3, 5, 7); even kernels (2, 4) may reduce W by 1 — recommend odd kernels only
- **All-zeros input:** No spikes → pool gets zeros → readout near bias values
- **Small window (W=7):** Conv with kernel_size=3 still works — W preserved with same padding

### Architecture References

- Section 5.2: Models module — SpikingCNN1D listed as Phase A model
- Section 13.3 Rule #5: Common model interface — must follow BaseModel contract
- Section 13.4: snnTorch guidance — reset hidden states, time-first convention
- Section 4.4: Config schema for SNN models

---

## Dependencies

**Prerequisite Stories:**
- STORY-4.1: snnTorch Integration & Spike Encoding (SpikeEncoder class)
- STORY-4.2: Spiking MLP Model (established snnTorch patterns in snn_models.py)

**Blocked Stories:**
- STORY-4.4: Phase A Training & Comparison (trains SpikingCNN1D on real data)

**External Dependencies:**
- snnTorch 0.9.1 installed (verified in STORY-4.1)

---

## Definition of Done

- [ ] `SpikingCNN1D` implemented in `src/c5_snn/models/snn_models.py`
- [ ] Registered as `"spiking_cnn1d"` in `MODEL_REGISTRY`
- [ ] `src/c5_snn/models/__init__.py` updated with export
- [ ] Config file `configs/snn_phase_a_cnn.yaml` created
- [ ] Unit tests in `tests/test_snn_models.py`:
  - [ ] Forward shape: `(4, 21, 39)` → `(4, 39)`
  - [ ] Backward completes (surrogate gradients)
  - [ ] Config parsing (channels, kernel_sizes, beta, encoding)
  - [ ] Default config values
  - [ ] Different window sizes and batch sizes
  - [ ] Single conv layer configuration
  - [ ] Model has learnable parameters
  - [ ] Registered in MODEL_REGISTRY and created via get_model
  - [ ] Rate-coded encoding forward shape
- [ ] `ruff check src/ tests/` passes with zero errors
- [ ] `pytest tests/ -v` passes (all existing + new tests)
- [ ] Acceptance criteria validated (all checked)
- [ ] Code committed to `main` branch and pushed

---

## Story Points Breakdown

- **SpikingCNN1D class (encoder + conv layers + pool + readout):** 2 points
- **Config file + model registration:** 1 point
- **Testing (shape, backward, config, edge cases):** 2 points
- **Total:** 5 points

**Rationale:** Moderate complexity, similar to STORY-4.2. The Conv1d + LIF pattern is slightly more complex than Linear + LIF due to the permute step, 3D membrane potential shapes, and global pooling. However, the overall temporal loop and readout pattern is identical to SpikingMLP, and the file structure is established. Testing effort is comparable.

---

## Additional Notes

- SpikingCNN1D is added to the same `snn_models.py` file as SpikingMLP (as planned in STORY-4.2 notes)
- Conv1d uses "same" padding (`padding = kernel_size // 2`) so the window dimension W is preserved through all conv layers, simplifying the architecture
- Global average pooling over W collapses spatial information before the readout layer — this is simpler than adaptive pooling or flattening
- The `channels` and `kernel_sizes` lists must have the same length (one kernel size per conv layer)
- Recommend odd kernel sizes only (3, 5, 7) to ensure exact same-padding behavior
- LIF membrane potential is 3D `(B, C, W)` for conv layers — different from SpikingMLP's 2D `(B, H)` for FC layers
- The permute `(B, W, 39)` → `(B, 39, W)` maps features to Conv1d channels, treating the 39 parts as input channels and W as the temporal/spatial sequence length

---

## Progress Tracking

**Status History:**
- 2026-02-11: Created by Scrum Master (AI)
- 2026-02-11: Started by Developer (AI)
- 2026-02-11: Completed by Developer (AI)

**Actual Effort:** 5 points (matched estimate)

**Implementation Notes:**
- Added `SpikingCNN1D(BaseModel)` to `src/c5_snn/models/snn_models.py`
- Architecture: SpikeEncoder → permute(B,39,W) → [Conv1d + LIF]×N → global avg pool(W) → mean(T) → Linear readout
- Conv1d uses same padding (`kernel_size // 2`) to preserve window dimension W
- LIF membrane potentials are 3D `(B, C, W)` — different from SpikingMLP's 2D `(B, H)`
- Registered as `"spiking_cnn1d"` in MODEL_REGISTRY; `get_model(config)` creates it
- Config file `configs/snn_phase_a_cnn.yaml` created
- 40 new tests (351 total), all passing
- All 15 acceptance criteria validated

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
