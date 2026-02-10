# Project Memory — c5_SNN

## Decisions
- Canonical modeling representation: `P_1..P_39` multi-hot from `CA5_matrix_binary.csv`.
- Task framing: multi-label ranking over 39 labels; output Top-20.

## Data Integrity Checks
- `CA5_date.csv` and `CA5_matrix_binary.csv` are aligned on dates and m_1..m_5.
- Binary integrity: each event has exactly 5 active parts and matches m_1..m_5.

## Compute Policy
- Run a short timing probe locally before full training.
- If projected > ~20 minutes, train on RunPod GPU (B200/B100-class).

## Open Questions
- Optimal history window W (start 21; tune 7–90).
- Best encoding for binary: direct spike vs rate-coded vs latency-coded.
- Whether to add calendar features (day-of-week seasonality) and how much they help.
- How to handle class imbalance and rare parts (loss weighting vs focal).

## Next Actions
- Implement validation + windowing utilities.
- Build baselines and evaluation harness with time-based splits.
- Implement SNN Phase A and compare against baselines.
