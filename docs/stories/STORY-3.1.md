# STORY-3.1: Frequency/Recency Heuristic Baseline

**Epic:** Epic 3 — Baseline Models
**Priority:** Must Have
**Story Points:** 3
**Status:** Completed
**Assigned To:** ai_dev_agent
**Created:** 2026-02-11
**Sprint:** 3

---

## User Story

As a researcher,
I want a non-learned baseline that ranks parts by frequency and recency within the input window,
so that I have a simple lower bound for comparison against all neural models.

---

## Description

### Background

With the full data pipeline (windowing, splitting, DataLoaders) and evaluation harness (metrics, CLI evaluate command) now complete from Sprint 2, the project needs its first model to evaluate. Before investing in neural architectures, we need a simple, interpretable heuristic baseline that sets the floor for performance.

The frequency/recency heuristic exploits a core assumption of the CA5 domain: parts that appeared frequently and recently in the input window are more likely to appear in the next event. This is the simplest reasonable predictor and provides the "beat this" target for all subsequent models.

This story also establishes the foundational model infrastructure: the `BaseModel` abstract base class that all models must subclass, the `MODEL_REGISTRY` dict for config-driven model lookup, and the `get_model()` factory function. Every subsequent model story (GRU, SNN variants) will build on this foundation.

### Scope

**In scope:**
- `BaseModel` abstract base class in `src/c5_snn/models/base.py`
- `MODEL_REGISTRY` dict and `get_model(config)` factory function
- `FrequencyBaseline` model in `src/c5_snn/models/baselines.py`
- Scoring: weighted sum of frequency counts + exponentially-decayed recency
- Conforms to `BaseModel` interface: `forward(x) -> logits` with shape `(batch, W, 39) -> (batch, 39)`
- Unit tests with hand-computed expected rankings
- Registration as `"frequency_baseline"` in `MODEL_REGISTRY`

**Out of scope:**
- GRU baseline (STORY-3.2)
- Training loop — this model has no learnable parameters (STORY-3.3)
- Running evaluation on real data (STORY-3.4)
- Spike encoding or SNN models (Sprint 4+)

### User Flow

1. User specifies `model.type: "frequency_baseline"` in experiment config YAML
2. System looks up `"frequency_baseline"` in `MODEL_REGISTRY`
3. System instantiates `FrequencyBaseline(config)` — no training needed
4. User passes windowed input `(batch, W, 39)` through `model.forward(x)`
5. System returns `(batch, 39)` scores ranking parts by frequency + recency
6. Evaluation harness computes metrics (Recall@K, Hit@K, MRR) on these scores

---

## Acceptance Criteria

- [ ] `src/c5_snn/models/base.py` implements `BaseModel(nn.Module)` abstract base class with `forward(x: Tensor) -> Tensor` contract: `(batch, W, 39) -> (batch, 39)`
- [ ] `src/c5_snn/models/base.py` implements `MODEL_REGISTRY` dict and `get_model(config: dict) -> BaseModel` factory function
- [ ] `get_model` raises `ConfigError` for unknown model type
- [ ] `src/c5_snn/models/baselines.py` implements `FrequencyBaseline(BaseModel)` with `forward()` method
- [ ] Scoring formula: for each sample, each part's score = `frequency_weight * count_in_window + recency_weight * exponentially_decayed_sum`
- [ ] Recency: more recent time steps weighted higher via exponential decay (step `t` in window weighted as `decay^(W-1-t)` where `t=0` is oldest, `t=W-1` is most recent)
- [ ] `FrequencyBaseline` registered in `MODEL_REGISTRY` as `"frequency_baseline"`
- [ ] Output shape is always `(batch, 39)` regardless of input window size
- [ ] Model has no learnable parameters (`list(model.parameters())` is empty)
- [ ] Unit tests verify:
  - Dominant part (appears in every window step) ranked highest
  - Recent-only part ranked higher than old-only part (recency weighting)
  - Output shape `(batch, 39)` for various batch sizes and window sizes
  - `get_model({"model": {"type": "frequency_baseline"}})` returns `FrequencyBaseline`
  - `get_model` raises `ConfigError` for unknown type
- [ ] Uses logging from STORY-1.2 (no `print()` statements)

---

## Technical Notes

### Components

- **New file:** `src/c5_snn/models/base.py` — `BaseModel` ABC, `MODEL_REGISTRY`, `get_model()`
- **New file:** `src/c5_snn/models/baselines.py` — `FrequencyBaseline`
- **Modified file:** `src/c5_snn/models/__init__.py` — export new classes and functions
- **New test file:** `tests/test_baselines.py`

### BaseModel Interface

```python
class BaseModel(nn.Module):
    """Abstract base class for all models in the c5_SNN pipeline.

    All models MUST implement forward() with the contract:
        Input:  (batch, W, 39) — windowed multi-hot event sequences
        Output: (batch, 39) — logits/scores for the 39 parts
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (batch, W, 39) -> (batch, 39)."""
        ...
```

### Model Registry

```python
MODEL_REGISTRY: dict[str, type[BaseModel]] = {}


def get_model(config: dict) -> BaseModel:
    """Instantiate a model from config.

    Args:
        config: Experiment config dict. Must have config["model"]["type"].

    Returns:
        Instantiated model.

    Raises:
        ConfigError: If model type not found in registry.
    """
    model_type = config.get("model", {}).get("type", "")
    if model_type not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ConfigError(
            f"Unknown model type '{model_type}'. "
            f"Available: {available}"
        )
    return MODEL_REGISTRY[model_type](config)
```

### FrequencyBaseline Scoring

For each sample in the batch, given input `x` of shape `(W, 39)`:

1. **Frequency component:** Count how many times each part is active (value=1) across the W time steps.
   ```
   freq = x.sum(dim=0)  # shape (39,) — count of active steps per part
   ```

2. **Recency component:** Weight each time step by exponential decay. Step `t=0` (oldest) gets weight `decay^(W-1)`, step `t=W-1` (most recent) gets weight `decay^0 = 1.0`.
   ```
   weights = decay ** torch.arange(W-1, -1, -1)  # shape (W,)
   recency = (x * weights.unsqueeze(1)).sum(dim=0)  # shape (39,)
   ```

3. **Combined score:**
   ```
   score = freq_weight * freq + recency_weight * recency
   ```

Default hyperparameters (from config or defaults):
- `freq_weight`: 1.0
- `recency_weight`: 1.0
- `decay`: 0.9

### Implementation Details

1. **No learnable parameters:** `FrequencyBaseline` should NOT register any `nn.Parameter`. The weights `freq_weight`, `recency_weight`, and `decay` are fixed hyperparameters from config, not learned.

2. **Vectorized computation:** The scoring should work on full batches without Python loops — use `torch.sum()`, broadcasting, and `torch.arange()`.

3. **Constructor signature:** `FrequencyBaseline(config: dict)` — extracts hyperparameters from `config.get("model", {})`.

4. **Architecture references:**
   - Section 5.2 (Models Module — BaseModel, MODEL_REGISTRY)
   - Section 2.4 (Strategy Pattern, Registry Pattern)
   - Section 13.3 (Common model interface rule)
   - FR4 (baseline heuristic model)

### Hand-Computed Test Cases

**Dominant part test:**
- Window (3 steps): part 5 active in all 3 steps, all other parts inactive
- freq[5] = 3, recency[5] = 0.81 + 0.9 + 1.0 = 2.71
- score[5] = 1.0 * 3 + 1.0 * 2.71 = 5.71
- All other parts: score = 0.0
- Part 5 should be ranked #1

**Recency vs frequency test:**
- Part A: active at t=0 only (oldest) — freq=1, recency=0.9^(W-1)
- Part B: active at t=W-1 only (newest) — freq=1, recency=1.0
- Same frequency, but Part B has higher recency → ranked higher

### Edge Cases

- All parts inactive in window → all scores = 0 (ties broken arbitrarily)
- All parts active in every step → scores proportional to window length
- Single time-step window (W=1) → recency = frequency
- Batch size = 1

---

## Dependencies

**Prerequisite Stories:**
- STORY-2.3: Evaluation Harness & Metrics (provides evaluation infrastructure)
- STORY-1.2: Logging, Seed Management & Config (logging, ConfigError)

**Blocked Stories:**
- STORY-3.2: ANN GRU Baseline (needs BaseModel and MODEL_REGISTRY from this story)
- STORY-3.3: Training Loop (needs get_model() to instantiate models)
- STORY-3.4: Baseline Results & Comparison (needs FrequencyBaseline to evaluate)

**External Dependencies:**
- None

---

## Definition of Done

- [ ] `src/c5_snn/models/base.py` implemented with `BaseModel`, `MODEL_REGISTRY`, `get_model()`
- [ ] `src/c5_snn/models/baselines.py` implemented with `FrequencyBaseline`
- [ ] `src/c5_snn/models/__init__.py` updated with exports
- [ ] Unit tests in `tests/test_baselines.py` written and passing:
  - [ ] `BaseModel` cannot be instantiated directly (abstract)
  - [ ] `get_model` returns correct model for known type
  - [ ] `get_model` raises `ConfigError` for unknown type
  - [ ] `FrequencyBaseline` output shape correct for various inputs
  - [ ] Dominant part ranked highest
  - [ ] Recency weighting: recent-only > old-only
  - [ ] No learnable parameters
  - [ ] Batch computation correct
- [ ] `ruff check src/ tests/` passes with zero errors
- [ ] `pytest tests/ -v` passes (all existing + new tests)
- [ ] CI green on GitHub Actions
- [ ] Acceptance criteria validated (all checked)
- [ ] Code committed to `main` branch and pushed

---

## Story Points Breakdown

- **BaseModel + Registry infrastructure:** 1 point (ABC + dict + factory)
- **FrequencyBaseline implementation:** 1 point (vectorized scoring)
- **Testing:** 1 point (shape checks, ranking verification, registry tests)
- **Total:** 3 points

**Rationale:** Low-to-moderate complexity. The model infrastructure (BaseModel, registry) is standard boilerplate. The frequency/recency scoring is straightforward vectorized math. Most effort goes into thorough testing that the heuristic produces correct rankings.

---

## Additional Notes

- This is the first story of Sprint 3 and establishes the model infrastructure for all subsequent models
- `FrequencyBaseline` has no training — it can be evaluated immediately via the `evaluate` CLI from STORY-2.4
- The `MODEL_REGISTRY` starts with one entry and grows as models are added in STORY-3.2 (GRU), STORY-4.2 (SpikingMLP), etc.
- The `get_model()` factory is used by the training loop (STORY-3.3) and the CLI evaluate command to instantiate models from config
- Default hyperparameters (`freq_weight=1.0`, `recency_weight=1.0`, `decay=0.9`) can be tuned later but are fixed for this story
- The combined score does NOT need to be normalized — the evaluation harness uses `torch.topk` which only cares about relative ordering

---

## Progress Tracking

**Status History:**
- 2026-02-11: Created by Scrum Master (AI)
- 2026-02-11: Started by Developer (AI)
- 2026-02-11: Completed by Developer (AI)

**Actual Effort:** 3 points (matched estimate)

**Implementation Notes:**
- `BaseModel(ABC, nn.Module)` — combined ABC with nn.Module for proper abstract method enforcement
- `MODEL_REGISTRY` dict + `get_model()` factory with `ConfigError` for unknown types
- `FrequencyBaseline` uses vectorized scoring: `freq_weight * freq + recency_weight * recency`
- Recency via exponential decay weights `decay^(W-1-t)`, computed with `torch.arange` broadcasting
- No `nn.Parameter` — all hyperparameters are fixed from config
- 26 new tests (160 total), all passing
- Hand-computed test values verified: dominant part = 5.71, recency old = 1.6561 vs new = 2.0

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
