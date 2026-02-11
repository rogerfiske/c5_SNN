# STORY-5.1: Spike-GRU Architecture

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
I want a spiking GRU that processes the window recurrently with LIF neurons,
so that I can evaluate whether recurrence helps in the spiking domain.

---

## Description

### Background

Phase A (Sprint 4) established that SpikingMLP and SpikingCNN1D — both feedforward architectures — cluster around Recall@20 ~0.51 with direct encoding (T=1), closely matching the GRU baseline (0.510) and FrequencyBaseline (0.523). The key finding was that direct encoding collapses the temporal loop to a single timestep, reducing SNNs to approximately feedforward networks with a threshold nonlinearity.

Phase B introduces a **recurrent** spiking architecture: Spike-GRU. Unlike Phase A models that flatten the window and process all W events simultaneously, Spike-GRU processes the window **event-by-event** — feeding events at positions 0, 1, ..., W-1 sequentially through recurrent spiking neurons. This mirrors how the conventional GRU baseline already processes the window, but replaces the GRU cell with snnTorch's `RLeaky` (recurrent Leaky Integrate-and-Fire) neurons.

The hypothesis is that combining recurrence (sequential processing of the W-length window) with spiking dynamics (membrane accumulation over encoding timesteps T) will capture temporal patterns that feedforward SNNs miss. The GRU baseline already shows that recurrent processing of the window is competitive; Spike-GRU tests whether spiking recurrence adds value.

### Scope

**In scope:**
- `SpikeGRU` class in `src/c5_snn/models/snn_phase_b.py` subclassing `BaseModel`
- Architecture: SpikeEncoder → recurrent spiking layers (snnTorch `RLeaky`) → linear readout → 39 logits
- Processes window dimension W recurrently (event-by-event) with accumulating membrane state
- Encoding dimension T handled via outer temporal loop (same as Phase A)
- Config-driven: hidden_size, num_layers, beta, dropout, surrogate, encoding, timesteps
- Registration in `MODEL_REGISTRY` as `"spike_gru"`
- Config file `configs/snn_phase_b.yaml`
- Unit tests for shape, backward pass, config parsing, registry, hidden state shapes
- Update `src/c5_snn/models/__init__.py` to export `SpikeGRU`

**Out of scope:**
- Hyperparameter sweep (STORY-5.2)
- Phase B evaluation and comparison (STORY-5.3)
- Training on real data (handled by STORY-5.2/5.3)
- `RSynaptic` variant (may be explored in HP sweep, but default implementation uses `RLeaky`)
- Cardinality penalty (future enhancement)
- Spiking Transformer (Sprint 6)

### User Flow

1. Researcher creates or modifies `configs/snn_phase_b.yaml` specifying `model.type: "spike_gru"`
2. Researcher runs `c5_snn train --config configs/snn_phase_b.yaml`
3. Trainer loads config → `get_model(config)` → creates `SpikeGRU` from registry
4. SpikeGRU composes SpikeEncoder internally for input encoding
5. Forward pass: encode (T, B, W, 39) → for each t in T: process W events recurrently → aggregate over T → readout
6. Training proceeds via Trainer with BCEWithLogitsLoss, surrogate gradients enable backprop through spiking layers
7. Early stopping on `val_recall_at_20`, checkpoint saved as usual

---

## Acceptance Criteria

- [ ] `SpikeGRU` class exists in `src/c5_snn/models/snn_phase_b.py`, subclasses `BaseModel`
- [ ] Uses snnTorch `RLeaky` for recurrent spiking layers (with `all_to_all=True` for dense recurrence)
- [ ] Processes window dimension W event-by-event with accumulating membrane and spike state
- [ ] SpikeEncoder composed internally (supports `direct` and `rate_coded` encoding modes)
- [ ] Configurable via config dict: `hidden_size`, `num_layers`, `beta`, `dropout`, `surrogate`, `encoding`, `timesteps`
- [ ] Forward signature: `(batch, W, 39) -> (batch, 39)` logits
- [ ] Handles multi-layer stacking: output spikes from layer i feed as input to layer i+1
- [ ] Temporal aggregation over encoding timesteps T (mean of final hidden spikes)
- [ ] Registered in `MODEL_REGISTRY` as `"spike_gru"`
- [ ] Config file `configs/snn_phase_b.yaml` provided with sensible Phase B defaults
- [ ] Exported from `src/c5_snn/models/__init__.py`
- [ ] Unit tests in `tests/test_snn_models.py`:
  - [ ] Forward shape: `(batch, W, 39) -> (batch, 39)`
  - [ ] Backward pass: gradients flow through surrogate function
  - [ ] Multi-layer variant produces correct shapes
  - [ ] Config-driven construction (hidden_size, num_layers, beta, encoding)
  - [ ] Registry lookup: `MODEL_REGISTRY["spike_gru"]` is `SpikeGRU`
  - [ ] Both encoding modes (`direct`, `rate_coded`) produce valid output
- [ ] `ruff check src/ tests/` passes with zero errors
- [ ] `pytest tests/ -v` passes (all existing + new tests)
- [ ] Logging: model logs architecture summary on construction (hidden_size, layers, beta, encoding)

---

## Technical Notes

### Components

- **New file:** `src/c5_snn/models/snn_phase_b.py` — SpikeGRU model class
- **Modified file:** `src/c5_snn/models/__init__.py` — export SpikeGRU
- **New file:** `configs/snn_phase_b.yaml` — Phase B default config
- **Modified file:** `tests/test_snn_models.py` — add SpikeGRU unit tests
- **Reused (no changes):** `src/c5_snn/models/base.py` — BaseModel, MODEL_REGISTRY
- **Reused (no changes):** `src/c5_snn/models/encoding.py` — SpikeEncoder
- **Reused (no changes):** `src/c5_snn/training/trainer.py` — model-agnostic Trainer

### Architecture Design

The Spike-GRU has two temporal dimensions to handle:

1. **Encoding timesteps T** (from SpikeEncoder): outer temporal loop, same as Phase A models
2. **Window dimension W**: inner recurrent loop, processing events sequentially

```
Input: x (batch, W, 39)
  |
  v
SpikeEncoder: (batch, W, 39) → (T, batch, W, 39)
  |
  v
For each t in T:                       ← Outer: encoding timesteps
  For each w in W:                     ← Inner: recurrent event processing
    input = encoded[t, :, w, :]        ← (batch, 39) at this timestep + event
    spk, mem = RLeaky(input, spk, mem) ← Recurrent LIF update
  Record final spk after W events      ← (batch, hidden_size)
  |
  v
Aggregate over T: mean(spk_records)    ← (batch, hidden_size)
  |
  v
Linear readout → (batch, 39) logits
```

### snnTorch `RLeaky` Usage

`RLeaky` is snnTorch's recurrent LIF neuron. Unlike `Leaky` (feedforward), `RLeaky` has a recurrent weight matrix that feeds output spikes back into the membrane potential:

```python
import snntorch

# Create recurrent LIF neuron layer
rlif = snntorch.RLeaky(
    beta=0.95,            # Membrane decay rate
    linear_features=128,  # Size of recurrent weight matrix (hidden_size)
    spike_grad=surrogate.fast_sigmoid(slope=25),
    all_to_all=True,      # Dense recurrent connections (not 1-to-1)
    init_hidden=False,     # Manual state management (consistent with Phase A)
)

# Usage in temporal loop:
spk = torch.zeros(batch, hidden_size, device=device)
mem = torch.zeros(batch, hidden_size, device=device)

for w in range(W):
    cur = fc_input(x[:, w, :])    # Project input to hidden_size
    spk, mem = rlif(cur, spk, mem)  # RLeaky takes (input, prev_spk, prev_mem)
```

**Key differences from `Leaky`:**
- `RLeaky` signature: `(input, prev_spk, prev_mem) -> (spk, mem)` — takes previous spike output as recurrent input
- `all_to_all=True`: dense recurrent matrix (hidden_size × hidden_size)
- `linear_features`: specifies the recurrent weight size

### Multi-Layer Stacking

For `num_layers > 1`, spikes from layer i become input to layer i+1:

```python
# Layer 0: input projection (39 → hidden_size) + RLeaky
# Layer 1+: identity/projection (hidden_size → hidden_size) + RLeaky

self.fc_input = nn.Linear(39, hidden_size)  # First layer input
self.rlif_layers = nn.ModuleList([
    snntorch.RLeaky(beta=beta, linear_features=hidden_size, ...)
    for _ in range(num_layers)
])

# In forward:
for w in range(W):
    cur = self.fc_input(x[:, w, :])
    for i, rlif in enumerate(self.rlif_layers):
        cur, mems[i] = rlif(cur, spks[i], mems[i])
        spks[i] = cur  # spikes become next layer's input
```

### Dropout Strategy

Apply dropout between recurrent layers (when `num_layers > 1`), consistent with GRU baseline's approach:

```python
if num_layers > 1 and dropout > 0:
    self.dropout = nn.Dropout(dropout)
    # Apply between layers in the recurrent loop
```

### Config Schema

```yaml
model:
  type: "spike_gru"
  encoding: "direct"          # "direct" | "rate_coded"
  timesteps: 10               # Encoding timesteps (for rate_coded)
  hidden_size: 128            # Recurrent hidden dimension
  num_layers: 1               # Number of stacked RLeaky layers
  beta: 0.95                  # Membrane decay factor
  dropout: 0.0                # Dropout between layers (0.0 for single layer)
  surrogate: "fast_sigmoid"   # Surrogate gradient function
```

### Phase A Learnings Applied

1. **Direct encoding (T=1) limits potential:** With T=1, the outer temporal loop runs once, so SpikeGRU still processes W events recurrently but without multi-timestep membrane dynamics from encoding. This is still valuable — it tests whether spiking recurrence over W helps.
2. **Rate-coded encoding (T>1) is the key lever:** With T>1, each of the W events gets multiple spike presentations, allowing membrane dynamics to accumulate across both T and W dimensions. This is the configuration expected to show the most improvement.
3. **Training speed:** Phase A SNNs trained ~2x faster than GRU with T=1. SpikeGRU with T=1 and W=21 will have a W-length inner loop (vs. GRU's built-in CUDA-optimized recurrence), so it may be slower. With T>1, training time scales linearly with T.

### Edge Cases

- **Single layer (num_layers=1):** No dropout applied, simplest recurrent path.
- **Direct encoding (T=1):** Outer loop runs once; model reduces to a pure recurrent spiking network over W.
- **Large T × W:** With T=10 and W=21, the nested loop has 210 iterations per sample. Monitor memory and training time.
- **No spikes produced:** If beta is too high or input too sparse, membrane may never reach threshold. Surrogate gradient ensures gradients still flow, but metric performance will be poor.
- **Vanishing gradients in deep stacks:** With `num_layers > 2`, surrogate gradients through multiple layers may attenuate. Recommend `num_layers ∈ {1, 2}` for Phase B.

### Architecture References

- Section 5.2: Models module — Phase B models, `snn_phase_b.py`
- Section 13.3 Rule #5: Common model interface — `BaseModel`, `(batch, W, 39) -> (batch, 39)`
- Section 13.4: snnTorch guidance — `RLeaky`/`RSynaptic`, state management
- Section 4.4: Experiment config schema
- Section 4.7: Comparison report schema (for STORY-5.3)

---

## Dependencies

**Prerequisite Stories:**
- STORY-4.1: snnTorch Integration & Spike Encoding (SpikeEncoder class)
- STORY-4.2: Spiking MLP Model (established SNN model patterns)
- STORY-4.3: Spiking 1D-CNN Model (established multi-layer SNN patterns)
- STORY-4.4: Phase A Training & Comparison (Phase A results inform encoding strategy)

**Blocked Stories:**
- STORY-5.2: Spike-GRU HP Sweep (needs SpikeGRU model to sweep)
- STORY-5.3: Phase B Evaluation & Comparison (needs SpikeGRU trained results)

**External Dependencies:**
- snnTorch 0.9.1 installed (provides `RLeaky`)
- PyTorch 2.5.1 (backend)

---

## Definition of Done

- [ ] `SpikeGRU` class implemented in `src/c5_snn/models/snn_phase_b.py`
- [ ] Subclasses `BaseModel`, registered as `"spike_gru"` in `MODEL_REGISTRY`
- [ ] Uses snnTorch `RLeaky` with `all_to_all=True`, `init_hidden=False`
- [ ] Forward: `(batch, W, 39) -> (batch, 39)` logits
- [ ] Config-driven: hidden_size, num_layers, beta, dropout, surrogate, encoding, timesteps
- [ ] Exported from `src/c5_snn/models/__init__.py`
- [ ] `configs/snn_phase_b.yaml` created with Phase B defaults
- [ ] Unit tests in `tests/test_snn_models.py`:
  - [ ] Forward shape test
  - [ ] Backward pass (gradient flow) test
  - [ ] Multi-layer shape test
  - [ ] Config-driven construction test
  - [ ] Registry lookup test
  - [ ] Both encoding modes test
- [ ] `ruff check src/ tests/` passes with zero errors
- [ ] `pytest tests/ -v` passes (all existing + new tests)
- [ ] Acceptance criteria validated (all checked)
- [ ] Code committed to `main` branch and pushed

---

## Story Points Breakdown

- **Model implementation (SpikeGRU class + RLeaky integration):** 2.5 points
- **Config file and registry integration:** 0.5 points
- **Unit tests (6+ tests for shapes, gradients, config, modes):** 1.5 points
- **Module exports and integration verification:** 0.5 points
- **Total:** 5 points

**Rationale:** Higher complexity than SpikingMLP (STORY-4.2, 5 pts) due to nested temporal loops (T × W), recurrent state management across both dimensions, and multi-layer stacking with `RLeaky`'s three-argument signature. However, the infrastructure (BaseModel, SpikeEncoder, MODEL_REGISTRY, Trainer) is all proven and tested, reducing integration risk. 5 points is appropriate for the model implementation + thorough testing.

---

## Additional Notes

- The architecture doc specifies `snn_phase_b.py` as a **new file** separate from `snn_models.py`. This keeps Phase A and Phase B models in distinct modules for clarity.
- `RLeaky` is preferred over `RSynaptic` as the default neuron type. `RSynaptic` adds a second-order synaptic current model which increases complexity without clear benefit for this task. HP sweep (STORY-5.2) can optionally test `RSynaptic` as a sweep dimension.
- The model should log its architecture summary on construction (hidden_size, num_layers, beta, encoding) using the `c5_snn` logger, consistent with SpikingMLP and SpikingCNN1D.
- Phase A results show all models at Recall@20 ~0.51 with direct encoding. The real test for SpikeGRU is with `rate_coded` encoding (T>1), which will be explored in STORY-5.2's HP sweep.
- The `all_to_all=True` parameter in `RLeaky` creates a dense recurrent weight matrix (hidden_size × hidden_size). This is analogous to GRU's hidden-to-hidden weight matrix.

---

## Progress Tracking

**Status History:**
- 2026-02-11: Created by Scrum Master (AI)
- 2026-02-11: Started by Developer (AI)
- 2026-02-11: Completed by Developer (AI)

**Actual Effort:** 5 points (matched estimate)

**Implementation Notes:**
- Created `src/c5_snn/models/snn_phase_b.py` with SpikeGRU class (~150 lines)
- Uses snnTorch `RLeaky` with `all_to_all=True`, `init_hidden=False` for manual state management
- Nested temporal loop: outer T (encoding) x inner W (recurrent event processing)
- `RLeaky` signature: `(input, prev_spk, prev_mem) -> (spk, mem)` — recurrent spikes feed back
- Multi-layer stacking: output spikes from layer i become input to layer i+1
- Dropout between layers (when num_layers > 1 and dropout > 0)
- 44 new tests (400 total), all passing
- All 19 acceptance criteria validated
- Config file `configs/snn_phase_b.yaml` with Phase B defaults

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
