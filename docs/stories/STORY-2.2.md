# STORY-2.2: Time-Based Train/Validation/Test Splits

**Epic:** Epic 2 — Data Pipeline & Evaluation Harness
**Priority:** Must Have
**Story Points:** 3
**Status:** Completed
**Assigned To:** ai_dev_agent
**Created:** 2026-02-11
**Sprint:** 2

---

## User Story

As a researcher,
I want the windowed samples split into train/val/test sets by chronological order,
so that evaluation is temporally valid with zero data leakage.

---

## Description

### Background

STORY-2.1 delivered windowed tensors `X (N, W, 39)` and `y (N, 39)` from the validated CA5 dataset. Before any model can be trained, these samples must be partitioned into training, validation, and test sets.

Because the CA5 data is a time series of events, random shuffling would leak future information into training — violating the fundamental principle of temporal forecasting. This story implements strict chronological splitting: the earliest events go to training, the middle events to validation, and the most recent events to testing. This ordering guarantee is the single most critical invariant for scientific validity of the entire project.

This story also provides the PyTorch `Dataset` wrapper and `DataLoader` factory that the training loop (STORY-3.3) and evaluation harness (STORY-2.3) will consume.

### Scope

**In scope:**
- Time-ordered splitting of windowed tensors by configurable ratios (default 70/15/15)
- Persistence of split metadata to `data/processed/splits.json`
- PyTorch `Dataset` class wrapping `(X, y)` tensors for a given split
- `DataLoader` factory returning train/val/test loaders with configurable batch size
- Deterministic behavior: identical config always produces identical splits
- Unit tests proving no overlap, strict ordering, JSON round-trip, and ratio edge cases

**Out of scope:**
- Data augmentation or resampling
- Cross-validation or k-fold splitting
- Any form of shuffling (even within the training set — preserves time order)
- Evaluation metrics computation (STORY-2.3)

### User Flow

1. User has windowed tensors from STORY-2.1 (`X_w21.pt`, `y_w21.pt`)
2. User calls `create_splits()` with sample count and ratios from YAML config
3. System computes split boundaries: train indices `[0, N_train)`, val `[N_train, N_train+N_val)`, test `[N_train+N_val, N)`
4. System persists `splits.json` with indices, ratios, counts, and date ranges
5. User calls `get_dataloaders()` with splits, tensors, and batch size
6. System returns a dict of `{"train": DataLoader, "val": DataLoader, "test": DataLoader}`
7. Training loop and evaluation harness consume these DataLoaders directly

---

## Acceptance Criteria

- [ ] `src/c5_snn/data/splits.py` implements `create_splits(n_samples, ratios, window_size, dates) -> SplitInfo` that computes time-ordered split indices
- [ ] Default ratios are 70/15/15 (train/val/test), configurable via YAML key `data.split_ratios`
- [ ] Ratios must sum to 1.0 (within floating-point tolerance); raise `ConfigError` otherwise
- [ ] Split indices are contiguous and non-overlapping: train `[0, t)`, val `[t, v)`, test `[v, N)` where `t < v < N`
- [ ] No shuffling — all train indices < all val indices < all test indices (strict chronological order)
- [ ] Split metadata persisted to `data/processed/splits.json` matching architecture Section 4.3 schema: `window_size`, `ratios`, `indices` (half-open ranges), `date_ranges`, `counts`
- [ ] Deterministic — same `n_samples` and `ratios` always produce identical split boundaries
- [ ] `src/c5_snn/data/dataset.py` implements `CA5Dataset(Dataset)` wrapping `(X, y)` slices for a single split
- [ ] `get_dataloaders(split_info, X, y, batch_size) -> dict[str, DataLoader]` returns DataLoaders for train, val, and test splits
- [ ] DataLoaders do NOT shuffle (time order preserved even within batches)
- [ ] Unit tests verify:
  - No index overlap between any two splits
  - Strict ordering: `max(train_indices) < min(val_indices) < min(test_indices)`
  - `splits.json` round-trip (save then load produces identical split)
  - Different ratio configurations produce valid splits (e.g., 80/10/10, 60/20/20)
  - Edge case: very small dataset with minimum viable splits
  - `CA5Dataset` returns correct `(X_i, y_i)` pairs with correct shapes
- [ ] Uses logging from STORY-1.2 (no `print()` statements)

---

## Technical Notes

### Components

- **New file:** `src/c5_snn/data/splits.py` — split computation and JSON persistence
- **New file:** `src/c5_snn/data/dataset.py` — `CA5Dataset` + `get_dataloaders()` factory
- **Modified file:** `src/c5_snn/data/__init__.py` — export new functions and classes
- **New test file:** `tests/test_splits.py`

### Key Types and Functions

```python
@dataclass
class SplitInfo:
    """Holds split boundaries and metadata."""
    window_size: int
    ratios: dict[str, float]        # {"train": 0.70, "val": 0.15, "test": 0.15}
    indices: dict[str, list[int]]   # {"train": [0, 8175], "val": [8175, 9927], ...}
    date_ranges: dict[str, list[str]]  # {"train": ["1992-01-15", "2015-06-12"], ...}
    counts: dict[str, int]          # {"train": 8175, "val": 1752, "test": 1752}


def create_splits(
    n_samples: int,
    ratios: tuple[float, float, float],
    window_size: int,
    dates: pd.Series | None = None,
) -> SplitInfo:
    """Compute time-ordered train/val/test split boundaries.

    Args:
        n_samples: Total number of windowed samples (N).
        ratios: (train_ratio, val_ratio, test_ratio), must sum to 1.0.
        window_size: Window size W (for metadata).
        dates: Optional date column from original DataFrame (for date_ranges).

    Returns:
        SplitInfo with indices, counts, and metadata.

    Raises:
        ConfigError: If ratios don't sum to 1.0 or any ratio is <= 0.
    """


def save_splits(split_info: SplitInfo, output_dir: str) -> Path:
    """Save split metadata to splits.json."""


def load_splits(path: str) -> SplitInfo:
    """Load split metadata from splits.json."""
```

```python
class CA5Dataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapping a slice of windowed tensors."""

    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def get_dataloaders(
    split_info: SplitInfo,
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
) -> dict[str, torch.utils.data.DataLoader]:
    """Create DataLoaders for each split.

    No shuffling — time order preserved.
    """
```

### Implementation Details

1. **Split boundary computation:** Simple integer arithmetic on `n_samples`:
   ```python
   n_train = int(n_samples * ratios[0])
   n_val = int(n_samples * ratios[1])
   n_test = n_samples - n_train - n_val  # remainder goes to test
   ```
   Using `int()` truncation + remainder-to-test ensures all samples are covered with no gaps.

2. **Date range mapping:** If `dates` Series is provided, map split indices back to the original DataFrame dates. For window index `t`, the corresponding date is `dates.iloc[t + window_size]` (the target event date). First date in split = date of first target, last date = date of last target.

3. **`splits.json` schema:** Must match architecture Section 4.3 exactly — half-open ranges `[start, end)`.

4. **DataLoader settings:** `shuffle=False` for all splits. `drop_last=False` to preserve all samples. `batch_size` from config (default 64).

5. **Architecture references:**
   - Section 4.3 (Split Index data model)
   - Section 5.1 (Data Module — `create_splits`, `get_dataloaders` interfaces)
   - Coding Standard #1 (no data leakage — no shuffling)

### Edge Cases

- Very small dataset: ensure each split has at least 1 sample (raise error if not possible)
- Ratios that produce 0 samples for a split (e.g., 99/0.5/0.5 on 10 samples) — handle gracefully
- Ratios that don't sum to exactly 1.0 due to float precision — use tolerance (e.g., `abs(sum - 1.0) < 1e-6`)
- `batch_size` larger than split size — DataLoader handles this naturally (single batch)

---

## Dependencies

**Prerequisite Stories:**
- STORY-2.1: Windowed Tensor Construction (must have `(X, y)` tensors to split)
- STORY-1.2: Logging, Seed Management & Config (uses `ConfigError`, logging)

**Blocked Stories:**
- STORY-2.3: Evaluation Harness & Metrics (needs DataLoaders for model evaluation)
- STORY-2.4: CLI evaluate Command (needs DataLoaders)
- STORY-3.3: Training Loop & train CLI (needs DataLoaders for training)

**External Dependencies:**
- None

---

## Definition of Done

- [ ] `src/c5_snn/data/splits.py` implemented with `create_splits()`, `save_splits()`, `load_splits()`
- [ ] `src/c5_snn/data/dataset.py` implemented with `CA5Dataset` and `get_dataloaders()`
- [ ] Unit tests in `tests/test_splits.py` written and passing:
  - [ ] No index overlap between splits
  - [ ] Strict chronological ordering of split boundaries
  - [ ] `splits.json` save/load round-trip
  - [ ] Different ratio configurations valid
  - [ ] Edge case: small dataset splits
  - [ ] `CA5Dataset.__getitem__` returns correct shapes
  - [ ] `get_dataloaders` returns dict with train/val/test keys
  - [ ] DataLoaders do not shuffle
- [ ] `ruff check src/ tests/` passes with zero errors
- [ ] `pytest tests/ -v` passes (all existing + new tests)
- [ ] CI green on GitHub Actions
- [ ] Acceptance criteria validated (all checked)
- [ ] Code committed to `main` branch and pushed

---

## Story Points Breakdown

- **Split logic + persistence:** 1 point (straightforward arithmetic + JSON I/O)
- **Dataset + DataLoader factory:** 1 point (thin PyTorch wrappers)
- **Testing:** 1 point (ordering proofs, round-trip, edge cases)
- **Total:** 3 points

**Rationale:** Low complexity — the split computation is simple integer arithmetic, and the Dataset/DataLoader wrappers are standard PyTorch boilerplate. The main value is in the tests that prove temporal ordering and no-leakage guarantees.

---

## Additional Notes

- This is the second story of Sprint 2 (Epic 2: Data Pipeline & Evaluation Harness)
- The `SplitInfo` dataclass and `get_dataloaders()` function become the primary interface between the data pipeline and the training/evaluation modules
- Architecture Section 4.3 specifies the exact `splits.json` schema — follow it precisely
- `shuffle=False` on all DataLoaders is critical — this is the second line of defense against data leakage (after chronological splitting itself)
- For the full dataset (N=11,679, W=21, ratios 70/15/15): train=8,175, val=1,752, test=1,752

---

## Progress Tracking

**Status History:**
- 2026-02-11: Created by Scrum Master (AI)
- 2026-02-11: Implemented and completed by Developer (AI)

**Actual Effort:** 3 points (matched estimate)

**Implementation Notes:**
- `src/c5_snn/data/splits.py`: `SplitInfo` dataclass, `create_splits()`, `save_splits()`, `load_splits()`
- `src/c5_snn/data/dataset.py`: `CA5Dataset`, `get_dataloaders()`
- `tests/test_splits.py`: 39 tests across 8 test classes
- All 12 acceptance criteria validated
- Ruff clean, all 93 tests passing

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
