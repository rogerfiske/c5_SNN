# STORY-2.1: Windowed Tensor Construction

**Epic:** Epic 2 — Data Pipeline & Evaluation Harness
**Priority:** Must Have
**Story Points:** 5
**Status:** Completed
**Assigned To:** ai_dev_agent
**Created:** 2026-02-11
**Sprint:** 2

---

## User Story

As a researcher,
I want validated raw data transformed into windowed tensors of shape `(W, 39)` with corresponding next-event targets,
so that I have training-ready inputs for all models.

---

## Description

### Background

Sprint 1 delivered a working `validate-data` CLI that confirms the integrity of the raw CA5 dataset (11,700 events x 45 columns). Before any model can be trained, the flat validated DataFrame must be transformed into overlapping sliding windows of binary P-vectors with corresponding next-event prediction targets.

This story implements the core data transformation step — the bridge between raw validated data and model-ready tensors. Every model in the project (baselines, SNN Phase A/B/C) will consume these windowed tensors, making this a foundational piece of the entire pipeline.

### Scope

**In scope:**
- Sliding window construction from validated DataFrame to `(N, W, 39)` input tensors and `(N, 39)` target tensors
- Configurable window size `W` via YAML config (default 21, valid range 7–90)
- Persistence of tensors to `.pt` files with accompanying JSON metadata
- SHA-256 hash of the source CSV for provenance tracking
- CLI integration: `window-data` subcommand (or incorporated into the data pipeline)
- Unit tests covering shapes, alignment, edge cases, and no-leakage guarantees

**Out of scope:**
- Train/val/test splitting (STORY-2.2)
- PyTorch Dataset/DataLoader wrappers (STORY-2.2)
- Any model training or evaluation
- Spike encoding (STORY-4.1)

### User Flow

1. User has already run `validate-data` and confirmed the CSV passes all checks
2. User runs windowing (either via CLI or programmatically from config)
3. System loads the validated CSV, extracts the 39 P-columns as float32
4. System constructs sliding windows: for each position `t`, input `X[t]` = rows `[t, t+1, ..., t+W-1]`, target `y[t]` = row `t+W`
5. System saves `X_w{W}.pt`, `y_w{W}.pt`, and `tensor_meta_w{W}.json` to `data/processed/`
6. System logs summary: number of samples, window size, date range, output paths

---

## Acceptance Criteria

- [ ] `src/c5_snn/data/windowing.py` implements `build_windows(df: pd.DataFrame, window_size: int) -> tuple[Tensor, Tensor]` producing `X` of shape `(N, W, 39)` and `y` of shape `(N, 39)` from a validated DataFrame
- [ ] `X[t]` contains P-vectors from events `[t, t+1, ..., t+W-1]` and `y[t]` contains the P-vector of event `t+W` — strict no-leakage: input window never includes the target row
- [ ] Window size `W` is configurable via YAML key `data.window_size` (default: 21, valid range: 7–90); invalid values raise `ConfigError`
- [ ] Output tensors are `dtype=torch.float32`
- [ ] Tensors saved as `data/processed/X_w{W}.pt` and `data/processed/y_w{W}.pt` via `torch.save()`
- [ ] Metadata file `data/processed/tensor_meta_w{W}.json` saved with schema: `source_file`, `source_hash` (SHA-256), `window_size`, `n_samples`, `n_features`, `date_range` (first and last dates), `created_at` (ISO timestamp)
- [ ] `data/processed/` directory created automatically if it doesn't exist
- [ ] Number of samples `N = len(df) - W` (e.g., 11,700 - 21 = 11,679 for default W)
- [ ] Unit tests verify:
  - Correct output shapes for various window sizes
  - First window alignment: `X[0]` = first W rows, `y[0]` = row at index W
  - Last window alignment: `X[-1]` = rows `[N-1..N+W-2]`, `y[-1]` = last row of DataFrame
  - No input-target overlap (target index is always `>= t+W`)
  - Empty/error result when DataFrame has fewer than `W+1` rows
  - Only P-columns (39 features) are included, not date or m-columns
- [ ] Uses logging from STORY-1.2 (no `print()` statements)

---

## Technical Notes

### Components

- **New file:** `src/c5_snn/data/windowing.py` — sliding window construction and persistence
- **Modified file:** `src/c5_snn/data/__init__.py` — export `build_windows` and `save_tensors`
- **Modified file:** `configs/default.yaml` — already has `data.window_size: 21` (no change needed)
- **New test file:** `tests/test_windowing.py`

### Key Functions

```python
def build_windows(
    df: pd.DataFrame, window_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build sliding window tensors from validated DataFrame.

    Extracts the 39 P-columns, constructs overlapping windows of size W,
    and pairs each window with the next-event target.

    Args:
        df: Validated DataFrame with columns date, m_1..m_5, P_1..P_39.
        window_size: Number of events per input window (W). Must be 7-90.

    Returns:
        Tuple of (X, y) where:
            X: Tensor of shape (N, W, 39) — input windows
            y: Tensor of shape (N, 39) — next-event targets
            N = len(df) - window_size

    Raises:
        ConfigError: If window_size is out of range [7, 90].
        DataValidationError: If DataFrame has fewer than window_size + 1 rows.
    """

def save_tensors(
    X: torch.Tensor,
    y: torch.Tensor,
    window_size: int,
    source_path: str,
    df: pd.DataFrame,
    output_dir: str = "data/processed",
) -> dict:
    """Save windowed tensors and metadata to disk.

    Args:
        X: Input tensor (N, W, 39).
        y: Target tensor (N, 39).
        window_size: Window size used.
        source_path: Path to source CSV (for hash + provenance).
        df: Original DataFrame (for date range extraction).
        output_dir: Output directory.

    Returns:
        Metadata dict (also saved as JSON).
    """
```

### Implementation Details

1. **P-column extraction:** Select columns `P_1` through `P_39` from the DataFrame, convert to a NumPy array of float32, then to a PyTorch tensor of shape `(total_events, 39)`.

2. **Sliding window construction:** Use a simple loop or `torch.unfold` / stride tricks:
   ```python
   p_tensor = torch.tensor(df[p_columns].values, dtype=torch.float32)
   N = len(p_tensor) - window_size
   X = torch.stack([p_tensor[t:t+window_size] for t in range(N)])
   y = p_tensor[window_size : window_size + N]
   ```
   For 11,679 windows of size 21 x 39, memory is ~34 MB — no concerns.

3. **Source hash:** Compute SHA-256 of the source CSV file for provenance tracking in the metadata JSON.

4. **Date range:** Extract `df["date"].iloc[0]` and `df["date"].iloc[-1]` for the metadata.

5. **Architecture references:**
   - Section 4.2 (Windowed Tensor data model)
   - Section 5.1 (Data Module interfaces)
   - Coding Standard #1 (no data leakage)

### Edge Cases

- DataFrame with exactly `W+1` rows → produces 1 sample (valid)
- DataFrame with exactly `W` rows → produces 0 samples → raise `DataValidationError`
- DataFrame with fewer than `W` rows → raise `DataValidationError`
- Window size at boundaries: W=7 (minimum), W=90 (maximum)
- Window size out of range (W=0, W=6, W=91, W=-1) → raise `ConfigError`

### Security Considerations

- None specific — no user input beyond config values, no network calls
- SHA-256 hash provides data provenance (detects if source CSV changes between runs)

---

## Dependencies

**Prerequisite Stories:**
- STORY-1.3: Data Validation CLI Command (must have validated DataFrame loader and P-column structure)
- STORY-1.2: Logging, Seed Management & Config (uses `load_config`, logging, `ConfigError`)

**Blocked Stories:**
- STORY-2.2: Time-Based Train/Validation/Test Splits (needs windowed tensors to split)
- STORY-2.3: Evaluation Harness & Metrics (needs windowed tensors via splits)

**External Dependencies:**
- None

---

## Definition of Done

- [ ] `src/c5_snn/data/windowing.py` implemented with `build_windows()` and `save_tensors()`
- [ ] Unit tests in `tests/test_windowing.py` written and passing:
  - [ ] Correct output shapes `(N, W, 39)` and `(N, 39)`
  - [ ] First and last window alignment verified
  - [ ] No input-target overlap confirmed
  - [ ] Edge cases: tiny data, boundary window sizes
  - [ ] Only P-columns extracted (39 features)
  - [ ] Metadata JSON contains all required fields
- [ ] `ruff check src/ tests/` passes with zero errors
- [ ] `pytest tests/ -v` passes (all existing + new tests)
- [ ] CI green on GitHub Actions
- [ ] Acceptance criteria validated (all checked)
- [ ] Code committed to `main` branch and pushed

---

## Story Points Breakdown

- **Windowing logic:** 2 points (sliding window + P-column extraction)
- **Persistence (`.pt` + JSON metadata):** 1 point
- **Testing:** 2 points (multiple edge cases, alignment, no-leakage proofs)
- **Total:** 5 points

**Rationale:** Moderate complexity — the core algorithm is straightforward (sliding window), but thorough testing for data leakage and edge cases is critical. The no-leakage guarantee is the single most important correctness property in the entire data pipeline.

---

## Additional Notes

- This is the first story of Sprint 2 and the first story in Epic 2 (Data Pipeline & Evaluation Harness)
- The windowed tensors produced here will be consumed by every subsequent story in the project
- Architecture Section 4.2 specifies the exact tensor format and metadata schema — follow it precisely
- For the full dataset (11,700 rows, W=21), expect N=11,679 samples, ~34 MB for X tensor
- Window size tuning (W=7 to W=90) will be explored in STORY-6.2 — this story just needs to support the configurable range

---

## Progress Tracking

**Status History:**
- 2026-02-11: Created by Scrum Master (AI)
- 2026-02-11: Implemented and completed by Developer (AI)

**Actual Effort:** 5 points (matched estimate)

**Implementation Notes:**
- `src/c5_snn/data/windowing.py`: `build_windows()` + `save_tensors()` + `_sha256()`
- `tests/test_windowing.py`: 28 tests across 4 test classes
- All acceptance criteria validated
- Ruff clean, all 54 tests passing

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
