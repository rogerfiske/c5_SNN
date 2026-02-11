# STORY-4.1: snnTorch Integration & Spike Encoding

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
I want a reusable spike encoding layer that converts windowed multi-hot input into spike trains,
so that all SNN models share a consistent, configurable encoding front-end.

---

## Description

### Background

With baseline models (FrequencyBaseline, GRUBaseline) complete and benchmark numbers established in Sprint 3, the project now transitions to Spiking Neural Networks. Before building any SNN architecture (MLP, CNN, GRU), we need a shared `SpikeEncoder` module that all SNN models will use as their input layer.

The CA5 data is already binary multi-hot (0/1 values for 39 parts), which means the simplest encoding ("direct") passes values through unchanged as spike events. A more expressive "rate-coded" mode uses snnTorch's `spikegen.rate()` to expand the temporal dimension with Poisson-like spike trains, giving SNN models richer temporal patterns to learn from.

This story also validates that snnTorch 0.9.1 integrates correctly with PyTorch 2.5.1 on all compute targets (local CPU, RunPod CUDA, CI CPU-only).

### Scope

**In scope:**
- `SpikeEncoder` class in `src/c5_snn/models/encoding.py` with direct and rate-coded modes
- Config-driven mode selection via `config["model"]["encoding"]`
- Verification that `import snntorch` works and `spikegen.rate()` functions correctly
- Unit tests for shape correctness, determinism, and edge cases
- Export from `models/__init__.py` for use by STORY-4.2 and STORY-4.3

**Out of scope:**
- Latency-coded encoding (future investigation, not needed for Phase A)
- SNN model architectures (STORY-4.2, STORY-4.3)
- Training or evaluation of SNN models (STORY-4.4)
- Hyperparameter tuning of encoding parameters (Sprint 5)

### User Flow

1. Researcher specifies encoding mode in experiment config YAML:
   ```yaml
   model:
     type: "spiking_mlp"
     encoding: "direct"       # or "rate_coded"
     timesteps: 10            # only used for rate_coded
   ```
2. SNN model instantiates `SpikeEncoder` in its `__init__` using config
3. During forward pass, input `(batch, W, 39)` is passed through encoder
4. Encoder outputs spike trains in snnTorch time-first format `(T, batch, W, 39)`
5. SNN model processes spike trains through spiking layers

---

## Acceptance Criteria

- [ ] `SpikeEncoder` class in `src/c5_snn/models/encoding.py` is an `nn.Module`
- [ ] **Direct mode:** Input `(batch, W, 39)` → output `(T, batch, W, 39)` where `T=1` (unsqueeze dim 0); binary values passed through unchanged
- [ ] **Rate-coded mode:** Input `(batch, W, 39)` → output `(T, batch, W, 39)` where `T=timesteps`; uses `snntorch.spikegen.rate()` for stochastic encoding
- [ ] Config-driven: `config["model"]["encoding"]` selects mode (`"direct"` default, `"rate_coded"` option)
- [ ] Config-driven: `config["model"]["timesteps"]` sets T for rate-coded mode (default 10, ignored for direct)
- [ ] Rate-coded encoding is deterministic given the same seed (via `set_global_seed()`)
- [ ] Direct encoding with binary input (0/1) produces identical output regardless of seed
- [ ] Edge cases handled: all-zeros input → all-zeros output; all-ones input → appropriate spike density
- [ ] Invalid encoding mode raises `ConfigError` with clear message
- [ ] `import snntorch` verified working with PyTorch 2.5.1 (tested in CI environment)
- [ ] `SpikeEncoder` exported from `src/c5_snn/models/__init__.py`
- [ ] Unit tests cover both modes, shape correctness, determinism, and edge cases
- [ ] `ruff check` passes and `pytest` all green

---

## Technical Notes

### Components

- **New file:** `src/c5_snn/models/encoding.py` — `SpikeEncoder` class
- **Modified file:** `src/c5_snn/models/__init__.py` — add `SpikeEncoder` export
- **New test file:** `tests/test_encoding.py` — encoding unit tests
- **Existing deps:** `snntorch.spikegen`, `c5_snn.utils.exceptions.ConfigError`

### snnTorch API Discovery

Verified with snnTorch 0.9.1:

```python
from snntorch import spikegen

# spikegen.rate() prepends T dimension (time-first convention):
# Input:  (batch, W, 39)
# Output: (T, batch, W, 39)

x = torch.randn(4, 21, 39).clamp(0, 1)
spikes = spikegen.rate(x, num_steps=10)
# spikes.shape == (10, 4, 21, 39)
```

**Key behaviors:**
- For binary inputs (0.0 or 1.0), `spikegen.rate()` is **deterministic** — P=1 always fires, P=0 never fires
- For fractional values, it's stochastic (Bernoulli sampling per timestep)
- Our CA5 data is binary multi-hot, so rate encoding will be deterministic unless we add noise or fractional features later
- snnTorch convention: **time-first** `(T, batch, ...)`, not batch-first

### SpikeEncoder Design

```python
class SpikeEncoder(nn.Module):
    """Configurable spike encoding layer for SNN model front-ends.

    Modes:
        direct: Unsqueeze T=1 dim. Binary values passed through as spikes.
        rate_coded: Expand temporal dimension using snntorch.spikegen.rate().

    Input:  (batch, W, 39) — windowed multi-hot binary tensor
    Output: (T, batch, W, 39) — spike train in snnTorch time-first format
    """

    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config.get("model", {})
        self.encoding = model_cfg.get("encoding", "direct")
        self.timesteps = int(model_cfg.get("timesteps", 10))

        if self.encoding not in ("direct", "rate_coded"):
            raise ConfigError(
                f"Unknown encoding mode: '{self.encoding}'. "
                f"Expected 'direct' or 'rate_coded'."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoding == "direct":
            return x.unsqueeze(0)  # (1, batch, W, 39)
        else:  # rate_coded
            return spikegen.rate(x, num_steps=self.timesteps)
```

### Output Shape Contract

| Mode | Input | Output | T value |
|------|-------|--------|---------|
| `direct` | `(B, W, 39)` | `(1, B, W, 39)` | 1 |
| `rate_coded` | `(B, W, 39)` | `(T, B, W, 39)` | config timesteps |

**Downstream SNN models** will iterate over the T dimension:
```python
# In SNN model forward():
spikes = self.encoder(x)  # (T, batch, W, 39)
T = spikes.size(0)
for t in range(T):
    step_input = spikes[t]  # (batch, W, 39)
    # ... process through spiking layers
```

### Config Structure

```yaml
model:
  type: "spiking_mlp"        # model registry key
  encoding: "direct"          # "direct" | "rate_coded"
  timesteps: 10               # only for rate_coded; ignored for direct
  hidden_sizes: [256, 128]    # model-specific (STORY-4.2)
  beta: 0.95                  # LIF decay (STORY-4.2)
```

### CI Considerations

- snnTorch 0.9.1 must be added to CI dependencies if not already present
- Verify `pip install snntorch==0.9.1` works with CPU-only PyTorch in GitHub Actions
- snnTorch has no CUDA-specific requirements at import time

### Edge Cases

- **All-zeros input:** Both modes should produce all-zeros (no spikes fire)
- **All-ones input:** Direct mode passes through; rate-coded produces all-ones (P=1.0 always fires)
- **Non-binary values:** Rate-coded handles 0-1 range gracefully (Bernoulli sampling); direct passes through as-is
- **timesteps=1 with rate_coded:** Should behave like a single Bernoulli sample per value
- **Large timesteps (T=100+):** Should work but memory increases linearly with T
- **Invalid encoding string:** Raise ConfigError immediately at construction time

### Architecture References

- Section 4.4: Config schema for encoding settings
- Section 5.2: Models module — SpikeEncoder as shared front-end
- Section 13.3 Rule #3: Seed before stochastic encoding for reproducibility
- Section 13.4: snnTorch guidance — reset hidden states, time-first convention

---

## Dependencies

**Prerequisite Stories:**
- STORY-3.3: Training Loop & train CLI (Trainer infrastructure — needed for later integration)
- STORY-2.1: Windowed Tensor Construction (produces the `(batch, W, 39)` input format)

**Blocked Stories:**
- STORY-4.2: Spiking MLP Model (uses `SpikeEncoder` as input layer)
- STORY-4.3: Spiking 1D-CNN Model (uses `SpikeEncoder` as input layer)
- STORY-4.4: Phase A Training & Comparison (trains SNN models)

**External Dependencies:**
- snnTorch 0.9.1 installed (`pip install snntorch==0.9.1`)
- PyTorch 2.5.1 compatibility verified

---

## Definition of Done

- [ ] `SpikeEncoder` implemented in `src/c5_snn/models/encoding.py`
- [ ] `src/c5_snn/models/__init__.py` updated with `SpikeEncoder` export
- [ ] Unit tests in `tests/test_encoding.py`:
  - [ ] Direct mode shape correctness
  - [ ] Rate-coded mode shape correctness
  - [ ] Direct mode passes binary values unchanged
  - [ ] Rate-coded determinism with same seed
  - [ ] All-zeros input → all-zeros output
  - [ ] All-ones input handling
  - [ ] Invalid encoding mode raises ConfigError
  - [ ] Default config values work correctly
  - [ ] Custom timesteps respected
- [ ] snnTorch import verified in test suite
- [ ] `ruff check src/ tests/` passes with zero errors
- [ ] `pytest tests/ -v` passes (all existing + new tests)
- [ ] CI green on GitHub Actions
- [ ] Acceptance criteria validated (all checked)
- [ ] Code committed to `main` branch and pushed

---

## Story Points Breakdown

- **SpikeEncoder module:** 2 points
- **snnTorch integration verification:** 1 point
- **Testing (both modes, edge cases, determinism):** 2 points
- **Total:** 5 points

**Rationale:** Moderate complexity. The encoding logic itself is straightforward (wrapping `spikegen.rate()` and pass-through), but thorough testing of shape contracts, determinism, and snnTorch compatibility across environments adds effort. This is the foundation for all SNN models, so getting the interface right is critical.

---

## Additional Notes

- This is the first story to use snnTorch — it establishes the integration pattern for all future SNN stories
- The time-first `(T, batch, ...)` convention from snnTorch differs from PyTorch's batch-first — all SNN models must handle this
- For binary CA5 data, rate encoding with P=0 or P=1 is actually deterministic; stochastic behavior only appears with fractional values (relevant if we add confidence scores or normalized features later)
- The `SpikeEncoder` is intentionally a simple `nn.Module` (not a `BaseModel`) — it's a composable component, not a standalone model
- Future encoding modes (latency-coded, delta-coded) can be added to this same class

---

## Progress Tracking

**Status History:**
- 2026-02-11: Created by Scrum Master (AI)
- 2026-02-11: Started by Developer (AI)
- 2026-02-11: Completed by Developer (AI)

**Actual Effort:** 5 points (matched estimate)

**Implementation Notes:**
- Created `src/c5_snn/models/encoding.py` with `SpikeEncoder` class (direct + rate_coded modes)
- snnTorch 0.9.1 verified working with PyTorch 2.5.1
- Key finding: `spikegen.rate()` uses time-first convention `(T, batch, W, 39)`
- Binary CA5 data (0/1) is deterministic under rate encoding (P=1 always fires, P=0 never fires)
- 38 new tests (273 total), all passing
- All 13 acceptance criteria validated

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
