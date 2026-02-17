"""Constants for the LLN-Pattern exclusion pipeline."""

from __future__ import annotations

# Number of values in the CA5 system (1 through 39)
NUM_VALUES = 39

# Number of positions per draw
NUM_POSITIONS = 5

# Default exclusion set size (predict 20 LEAST likely)
DEFAULT_K_EXCLUDE = 20

# Position-specific optimal LLN parameters (from prior EDA).
# recent_fraction: fraction of total data to use as recency window
# percentile_threshold: percentile above which values contribute to score
POSITION_PARAMS: dict[int, dict[str, float]] = {
    1: {"recent_fraction": 0.08, "percentile_threshold": 83.0},
    2: {"recent_fraction": 0.08, "percentile_threshold": 81.0},
    3: {"recent_fraction": 0.09, "percentile_threshold": 80.0},
    4: {"recent_fraction": 0.06, "percentile_threshold": 84.0},
    5: {"recent_fraction": 0.04, "percentile_threshold": 87.0},
}

# Boundary penalty configuration
BOUNDARY_LOW_RANGE = range(1, 8)  # Values 1-7
BOUNDARY_HIGH_RANGE = range(33, 40)  # Values 33-39
DEFAULT_BOUNDARY_PENALTY = 1.7

# Position valid ranges (from prior EDA)
POSITION_VALUE_RANGES: dict[int, tuple[int, int]] = {
    1: (1, 30),
    2: (2, 35),
    3: (3, 37),
    4: (4, 38),
    5: (9, 39),
}

# Known high-PMI pairs: (pos_a, val_a, pos_b, val_b, pmi_score)
HIGH_PMI_PAIRS: list[tuple[int, int, int, int, float]] = [
    (2, 35, 3, 37, 9.672),
    (1, 26, 2, 35, 9.392),
    (3, 3, 4, 4, 9.310),
    (3, 3, 5, 10, 9.310),
    (4, 7, 5, 9, 9.087),
]

# Exponential decay factor for recency weighting
DEFAULT_RECENCY_DECAY = 0.95
