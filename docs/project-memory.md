# Project Memory — c5_SNN

## Project Management

- **Method:** BMAD v6 (Level 2, sprint-based)
- **Workflow pattern:** `/bmad:create-story STORY-X.Y` then `/bmad:dev-story STORY-X.Y`
- **End-of-session command:** `ED` (triggers wrap-up protocol defined in CLAUDE.md)
- **User (Roger):** Not a programmer — all technical decisions delegated to AI agent
- **Sprint plan:** 7 one-week sprints, 22 stories, 103 total points
- **Repository:** https://github.com/rogerfiske/c5_SNN

## Sprint History

| Sprint | Points | Velocity | Status |
|--------|--------|----------|--------|
| Sprint 1 | 13/13 | 13 | Complete (2026-02-10) |
| Sprint 2 | 0/16 | — | Next |

## Decisions

- Canonical modeling representation: `P_1..P_39` multi-hot from `CA5_matrix_binary.csv`.
- Task framing: multi-label ranking over 39 labels; output Top-20.
- Build backend: `setuptools.build_meta` (not `setuptools.backends._legacy:_Backend` — incompatible with setuptools <70).
- PyTorch in CI: CPU-only via `--index-url https://download.pytorch.org/whl/cpu`.
- Date column loaded as string (not datetime) — parsing only for validation checks.
- All validation checks run regardless of earlier failures (report-all-at-once pattern).
- CLI uses Click; logging uses Python `logging` module (not print).

## Data Integrity Checks

- `CA5_date.csv` and `CA5_matrix_binary.csv` are aligned on dates and m_1..m_5.
- Binary integrity: each event has exactly 5 active parts and matches m_1..m_5.
- Real CSV: 11,702 rows x 45 columns, all 5 validation checks PASS.
- Date range: 1992-02-04 to 2026-02-09.

## Technical Implementation Notes

- **Exception hierarchy:** C5SNNError -> DataValidationError, ConfigError, TrainingError
- **Seed management:** `set_global_seed()` sets random, numpy, torch, cudnn.deterministic=True
- **Config:** YAML via `yaml.safe_load`, raises ConfigError on failure
- **Device:** `get_device()` with CUDA detection, ROCm HSA_OVERRIDE warning
- **Validation:** 5 checks (column count, column names, dates monotonic, row sums == 5, m/P cross-check)
- **Test fixtures:** `_make_valid_df()` generates valid CA5 DataFrames with numpy RandomState(42)

## Compute Policy

- Run a short timing probe locally before full training.
- If projected > ~20 minutes, train on RunPod GPU (B200/B100-class).
- Local: Windows, Python 3.11.9, AMD RX 6600M (CPU-only — ROCm is Linux-only).

## Open Questions

- Optimal history window W (start 21; tune 7-90).
- Best encoding for binary: direct spike vs rate-coded vs latency-coded.
- Whether to add calendar features (day-of-week seasonality) and how much they help.
- How to handle class imbalance and rare parts (loss weighting vs focal).

## Next Actions

- Sprint 2: Implement windowed tensor construction (STORY-2.1).
- Then: time-based splits, evaluation harness, evaluate CLI command.
- After Sprint 2: baseline models (frequency heuristic + GRU).
