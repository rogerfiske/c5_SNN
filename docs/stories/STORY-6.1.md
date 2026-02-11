# STORY-6.1: Spiking Transformer Architecture

**Epic:** Epic 6 — SNN Phase C & Final Report
**Priority:** Must Have
**Story Points:** 8
**Status:** Completed
**Assigned To:** ai_dev_agent
**Created:** 2026-02-11
**Sprint:** 6

---

## User Story

As a researcher,
I want a spiking transformer with spike-domain self-attention and positional encoding,
so that I can capture long-range temporal dependencies.

---

## Description

### Background

Sprints 3-5 have explored three SNN architectures — feedforward (SpikingMLP), convolutional (SpikingCNN1D), and recurrent (SpikeGRU). All learned models cluster around Recall@20 ~ 0.51, with the frequency baseline still leading at 0.5232. Phase B showed that encoding mode (direct vs rate_coded) makes no difference for Spike-GRU, and hidden_size is the dominant hyperparameter.

The Spiking Transformer represents the most expressive architecture in the project. Unlike MLP (no spatial structure), CNN (local receptive fields), and GRU (sequential recurrence), the transformer uses self-attention to directly model relationships between any pair of events in the window — regardless of their temporal distance. This "long-range dependency" capability is the primary hypothesis for Phase C.

The design follows the Spikformer paper (Zhou et al., 2023) which introduces Spiking Self-Attention (SSA): attention computed from spike-form Query, Key, and Value matrices without softmax normalization. This keeps the computation within the spike domain and avoids the multiply-accumulate operations that softmax requires.

### Scope

**In scope:**
- `SpikingTransformer(BaseModel)` class in `src/c5_snn/models/snn_phase_c.py`
- Spiking Self-Attention (SSA) module with multi-head support
- Learnable positional encoding for variable window sizes (W=7-90)
- Spiking feed-forward network (FFN) with LIF neurons
- Temporal aggregation and linear readout to 39 logits
- Config file `configs/snn_phase_c.yaml` with default hyperparameters
- Registration as `"spiking_transformer"` in MODEL_REGISTRY
- Unit tests for forward pass, backward pass, and shape correctness
- Smoke test: train 2 epochs on tiny data to verify gradient flow

**Out of scope:**
- Hyperparameter tuning (STORY-6.3)
- Window size tuning (STORY-6.2)
- Final comparison (STORY-6.4)
- Spike-driven attention variants (SDSA) — may explore in future
- Vision-specific components (Spiking Patch Splitting) — not applicable to time-series

### User Flow

1. Researcher creates/edits `configs/snn_phase_c.yaml` with model type `"spiking_transformer"`
2. Researcher runs `c5-snn train --config configs/snn_phase_c.yaml`
3. Pipeline loads data, builds windows, creates splits (unchanged)
4. `get_model(config)` looks up `"spiking_transformer"` in MODEL_REGISTRY
5. SpikingTransformer is instantiated with config parameters
6. Trainer runs training loop — forward, BCEWithLogitsLoss, backward, validation
7. Best checkpoint saved with metrics CSV, config snapshot, pip freeze
8. Researcher evaluates with `c5-snn evaluate --checkpoint results/.../best_model.pt`

---

## Acceptance Criteria

- [ ] `SpikingTransformer(BaseModel)` class exists in `src/c5_snn/models/snn_phase_c.py`
- [ ] Architecture: SpikeEncoder -> positional encoding -> N spiking transformer layers (SSA + spiking FFN with LIF) -> temporal aggregation -> linear -> 39 logits
- [ ] SSA module: multi-head attention from spike-form Q/K/V (linear projections through LIF neurons), scaled dot-product without softmax, LIF-gated output
- [ ] Configurable parameters: `n_layers`, `n_heads`, `d_model`, `d_ffn`, `beta`, `dropout`, `pe_type`
- [ ] Supports variable window sizes W=7 through W=90 without architectural changes
- [ ] Registered as `"spiking_transformer"` in MODEL_REGISTRY
- [ ] Config file at `configs/snn_phase_c.yaml` with sensible defaults
- [ ] Forward pass: input `(batch, 21, 39)` -> output `(batch, 39)` (default W=21)
- [ ] Forward pass: input `(batch, 60, 39)` -> output `(batch, 39)` (variable W=60)
- [ ] Backward pass completes without error (gradient flow through SSA and LIF)
- [ ] Smoke test: trains for 2 epochs on real data without NaN loss
- [ ] Unit tests in `tests/test_snn_models.py` (or new `tests/test_snn_phase_c.py`)
- [ ] `ruff check src/ tests/` passes with zero errors
- [ ] `pytest tests/ -v` passes (all existing + new tests)

---

## Technical Notes

### Components

- **New file:** `src/c5_snn/models/snn_phase_c.py` — SpikingTransformer, SpikingSelfAttention, SpikingTransformerBlock
- **Modified file:** `src/c5_snn/models/__init__.py` — add SpikingTransformer to exports
- **New file:** `configs/snn_phase_c.yaml` — default Phase C experiment config
- **New/modified file:** `tests/test_snn_models.py` or `tests/test_snn_phase_c.py` — unit tests
- **Reused (no changes):** `src/c5_snn/models/base.py`, `src/c5_snn/models/encoding.py`, `src/c5_snn/training/trainer.py`

### Architecture Design

```
Input: (batch, W, 39) — multi-hot binary event sequences

1. SpikeEncoder
   (batch, W, 39) -> (T, batch, W, 39)
   T=1 for direct encoding, T=timesteps for rate_coded

2. Input Projection
   (T, batch, W, 39) -> (T, batch, W, d_model)
   Linear layer: 39 -> d_model

3. Positional Encoding
   (T, batch, W, d_model) += PE(W, d_model)
   Learnable embedding: nn.Parameter(1, W_max, d_model) sliced to W

4. Spiking Transformer Blocks x N
   Each block:
     a. SSA (Spiking Self-Attention)
        - Q, K, V projections: Linear(d_model, d_model) through LIF
        - Multi-head: reshape to (T, batch, n_heads, W, d_head)
        - Attention: (Q @ K^T) * scale -> apply to V (NO softmax)
        - Output through LIF neuron
        - Residual connection + LIF

     b. Spiking FFN
        - Linear(d_model, d_ffn) -> LIF -> Linear(d_ffn, d_model) -> LIF
        - Residual connection + LIF

5. Temporal Aggregation
   Mean over T dimension: (T, batch, W, d_model) -> (batch, W, d_model)
   Mean over W dimension: (batch, W, d_model) -> (batch, d_model)
   OR: take last position [W-1] as "CLS-like" token

6. Readout
   Linear(d_model, 39) -> logits
   (batch, d_model) -> (batch, 39)
```

### Spiking Self-Attention (SSA) Detail

The key insight from Spikformer is that standard softmax attention is incompatible with spiking neural networks because spikes are binary (0/1) and softmax requires continuous values. SSA replaces softmax with a simple scaling factor:

```python
class SpikingSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, beta, dropout):
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # LIF neurons for Q, K, V (spike-form)
        self.lif_q = snntorch.Leaky(beta=beta, init_hidden=False,
                                     surrogate_fn=fast_sigmoid(slope=25))
        self.lif_k = snntorch.Leaky(beta=beta, ...)
        self.lif_v = snntorch.Leaky(beta=beta, ...)
        self.lif_out = snntorch.Leaky(beta=beta, ...)

    def forward(self, x, mem_q, mem_k, mem_v, mem_out):
        """
        x: (batch, W, d_model) — input for one encoding timestep
        Returns: (batch, W, d_model), updated membrane states
        """
        B, W, D = x.shape

        # Project and pass through LIF neurons (spike-form Q, K, V)
        q, mem_q = self.lif_q(self.q_proj(x), mem_q)  # (B, W, D) binary spikes
        k, mem_k = self.lif_k(self.k_proj(x), mem_k)
        v, mem_v = self.lif_v(self.v_proj(x), mem_v)

        # Reshape for multi-head: (B, W, D) -> (B, n_heads, W, d_head)
        q = q.view(B, W, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, W, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, W, self.n_heads, self.d_head).transpose(1, 2)

        # Spike-form attention: NO softmax, just scaled dot-product
        # (B, heads, W, d_head) @ (B, heads, d_head, W) -> (B, heads, W, W)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply attention to values
        # (B, heads, W, W) @ (B, heads, W, d_head) -> (B, heads, W, d_head)
        out = attn @ v

        # Reshape back: (B, heads, W, d_head) -> (B, W, D)
        out = out.transpose(1, 2).contiguous().view(B, W, D)

        # Output projection + LIF
        out, mem_out = self.lif_out(self.out_proj(out), mem_out)
        return out, mem_q, mem_k, mem_v, mem_out
```

### Positional Encoding

Use **learnable** positional encoding (not sinusoidal) since window sizes vary:

```python
# In __init__:
self.pos_embed = nn.Parameter(torch.zeros(1, max_window_size, d_model))
nn.init.trunc_normal_(self.pos_embed, std=0.02)

# In forward: slice to actual window size
x = x + self.pos_embed[:, :W, :]
```

Set `max_window_size=100` to support W=7 through W=90 without changes.

### Temporal Processing Pattern

Follow the same T-loop pattern as all other SNN models:

```python
def forward(self, x):
    spikes = self.encoder(x)  # (T, B, W, 39)
    T, B, W, F = spikes.shape

    # Project input features to d_model
    # Reset all membrane potentials per forward call

    t_records = []
    for t in range(T):
        cur = self.input_proj(spikes[t])  # (B, W, d_model)
        cur = cur + self.pos_embed[:, :W, :]

        # Pass through N transformer blocks
        for block in self.blocks:
            cur = block(cur, ...)  # SSA + FFN with membrane states

        t_records.append(cur)  # (B, W, d_model)

    # Aggregate over T
    agg = torch.stack(t_records).mean(dim=0)  # (B, W, d_model)

    # Aggregate over W (mean pool or last-token)
    agg = agg.mean(dim=1)  # (B, d_model)

    return self.readout(agg)  # (B, 39)
```

### Config File Design (`configs/snn_phase_c.yaml`)

```yaml
experiment:
  name: "snn_phase_c_spiking_transformer"
  seed: 42

data:
  raw_path: "data/raw/CA5_matrix_binary.csv"
  window_size: 21
  split_ratios: [0.70, 0.15, 0.15]
  batch_size: 64

model:
  type: "spiking_transformer"
  encoding: "direct"
  timesteps: 10
  d_model: 128
  n_heads: 4
  n_layers: 2
  d_ffn: 256
  beta: 0.95
  dropout: 0.1
  max_window_size: 100
  surrogate: "fast_sigmoid"

training:
  epochs: 100
  learning_rate: 0.001
  optimizer: "adam"
  early_stopping_patience: 10
  early_stopping_metric: "val_recall_at_20"

output:
  dir: "results/snn_phase_c_spiking_transformer"

log_level: "INFO"
```

### Hyperparameter Defaults Rationale

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `d_model` | 128 | Balance expressiveness vs compute; matches GRU hidden_size sweet spot |
| `n_heads` | 4 | 128/4 = 32-dim heads; standard transformer practice |
| `n_layers` | 2 | Start shallow; Phase B showed 1-2 layers sufficient |
| `d_ffn` | 256 | 2x d_model; standard FFN expansion ratio |
| `beta` | 0.95 | Matches Phase A/B best; slow membrane decay |
| `dropout` | 0.1 | Light regularization; may need tuning |
| `max_window_size` | 100 | Supports W=7-90 comfortably |
| `encoding` | "direct" | Phase B showed no benefit from rate_coded |

### Edge Cases

- **W=7 (small window):** Only 7 positions for attention — still valid, just simpler patterns
- **W=90 (large window):** Attention matrix is 90x90 per head — memory manageable at batch_size=64
- **d_model not divisible by n_heads:** Raise ConfigError at init time
- **NaN gradients from SSA:** SSA without softmax can produce large attention values; use scaling factor and consider clamping if needed during implementation
- **Zero spikes:** If all Q/K spikes are zero for a timestep, attention output is zero — this is valid behavior (no information to attend to)

### Architecture References

- Spikformer paper: Zhou et al. (2023) "Spikformer: When Spiking Neural Network Meets Transformer" (ICLR 2023)
- Section 5.2: Models module — `snn_phase_c.py`
- Section 2.4: Strategy Pattern — BaseModel interface
- Section 7.1: Output directory structure

---

## Dependencies

**Prerequisite Stories:**
- STORY-5.3: Phase B Evaluation & Comparison (cumulative leaderboard established)
- STORY-5.1: Spike-GRU Architecture (established snnTorch patterns: SpikeEncoder, surrogate gradients, T-loop)
- STORY-4.1: snnTorch Integration (SpikeEncoder, encoding modes)

**Blocked Stories:**
- STORY-6.2: Window Size Tuning (needs SpikingTransformer to test variable W)
- STORY-6.3: HP Sweep & Best Model (needs SpikingTransformer to tune)
- STORY-6.4: Final Comprehensive Comparison (needs best Phase C model)

**External Dependencies:**
- snnTorch 0.9.1 (already installed)
- PyTorch 2.5.1 (already installed)
- No new packages required

---

## Definition of Done

- [ ] `SpikingTransformer` class implemented in `src/c5_snn/models/snn_phase_c.py`
- [ ] SSA module with multi-head spike-form attention (no softmax)
- [ ] Learnable positional encoding supporting W=7-90
- [ ] Registered as `"spiking_transformer"` in MODEL_REGISTRY
- [ ] `configs/snn_phase_c.yaml` created with default hyperparameters
- [ ] `src/c5_snn/models/__init__.py` updated with SpikingTransformer export
- [ ] Unit tests:
  - [ ] Forward pass shape: `(4, 21, 39)` -> `(4, 39)`
  - [ ] Forward pass shape: `(4, 60, 39)` -> `(4, 39)` (variable W)
  - [ ] Backward pass completes (gradients flow)
  - [ ] Config validation (d_model % n_heads == 0)
  - [ ] Model registered in MODEL_REGISTRY
- [ ] Smoke test: 2 epochs on real data, loss decreasing, no NaN
- [ ] `ruff check src/ tests/` passes with zero errors
- [ ] `pytest tests/ -v` passes (all existing + new tests)
- [ ] Acceptance criteria validated (all checked)
- [ ] Code committed to `main` branch and pushed

---

## Story Points Breakdown

- **SSA module implementation:** 2.5 points
- **SpikingTransformer class + positional encoding:** 2.0 points
- **Config + registry + init:** 0.5 points
- **Testing (unit + smoke):** 2.0 points
- **Validation and debugging:** 1.0 points
- **Total:** 8 points

**Rationale:** This is the most complex model in the project. SSA requires careful implementation of spike-form attention with multi-head support, membrane state management across T and W dimensions, and proper gradient flow through surrogate functions. The 8-point estimate reflects the novelty of the architecture (no prior SSA code to reference in this codebase), the multiple interacting components (SSA + FFN + PE + aggregation), and the debugging/validation effort needed to ensure correct spike dynamics.

---

## Additional Notes

- This is the most complex single model in the project. The SSA mechanism is novel and may require iteration to get right.
- The architecture should be kept as simple as possible while still being a legitimate spiking transformer. Avoid over-engineering with features like relative positional encodings or gated attention variants unless the basic version fails to train.
- Phase B finding: encoding mode doesn't matter. Start with `direct` (T=1) for fast iteration. If needed, `rate_coded` is trivially switchable via config.
- Phase B finding: hidden_size dominates. The `d_model` parameter (analogous to hidden_size) should be the primary tuning knob in STORY-6.3.
- All learned models cluster at ~0.51 Recall@20. If the Spiking Transformer also lands here, the conclusion is that the dataset's temporal structure is inherently simple and architecture choice doesn't matter much — which is itself a valuable research finding.
- The `max_window_size=100` parameter allows the model to work with any W from 7-90 without reinstantiation. The positional embedding is sliced to the actual window size at runtime.
- Keep the T-loop pattern consistent with SpikingMLP, SpikingCNN1D, and SpikeGRU. Reset membrane potentials at the start of each forward call. Process T encoding timesteps, then aggregate.

---

## Progress Tracking

**Status History:**
- 2026-02-11: Created by Scrum Master (AI)
- 2026-02-11: Started by Developer (AI)
- 2026-02-11: Completed by Developer (AI)

**Actual Effort:** 8 points (matched estimate)

**Implementation Notes:**
- Created `src/c5_snn/models/snn_phase_c.py` with SpikingTransformer, SpikingSelfAttention, SpikingTransformerBlock (~290 lines)
- SSA: multi-head spike-form Q/K/V without softmax, scaled dot-product, LIF-gated output
- Learnable positional encoding: `nn.Parameter(1, max_window_size, d_model)` sliced to actual W at runtime
- Spiking FFN: Linear→LIF→Dropout→Linear→LIF with residual connections
- Temporal pattern: T-loop over encoding timesteps, mean(T)→mean(W)→Linear readout
- 286,887 parameters (default config: d_model=128, n_heads=4, n_layers=2, d_ffn=256)
- Supports variable W=7-90 without reinstantiation (pos_embed slicing)
- Smoke test: 2 epochs, val_recall_at_20=0.5067, no NaN, 9.7s on CPU
- 51 new tests (464 total), all passing
- All 14 acceptance criteria validated
- Config file `configs/snn_phase_c.yaml` created with sensible defaults
- Registered as `"spiking_transformer"` in MODEL_REGISTRY

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
