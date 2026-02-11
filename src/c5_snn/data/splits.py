"""Time-based train/validation/test splitting for windowed tensors.

Splits windowed samples into chronologically ordered partitions with
zero data leakage. All train indices < all val indices < all test indices.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from c5_snn.utils.exceptions import ConfigError

logger = logging.getLogger(__name__)


@dataclass
class SplitInfo:
    """Holds split boundaries and metadata.

    Indices are half-open ranges [start, end) into the windowed tensor arrays.
    """

    window_size: int
    ratios: dict[str, float]
    indices: dict[str, list[int]]
    date_ranges: dict[str, list[str]]
    counts: dict[str, int]


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
        window_size: Window size W (for metadata and date mapping).
        dates: Optional date column from original DataFrame (for date_ranges).
            Must have length n_samples + window_size.

    Returns:
        SplitInfo with indices, counts, and metadata.

    Raises:
        ConfigError: If ratios don't sum to 1.0 or produce empty splits.
    """
    train_r, val_r, test_r = ratios

    # Validate ratios
    if any(r <= 0 for r in ratios):
        raise ConfigError(
            f"All split ratios must be > 0, got {ratios}"
        )

    ratio_sum = train_r + val_r + test_r
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ConfigError(
            f"Split ratios must sum to 1.0, got {ratio_sum:.6f} "
            f"from {ratios}"
        )

    # Compute split sizes — remainder goes to test
    n_train = int(n_samples * train_r)
    n_val = int(n_samples * val_r)
    n_test = n_samples - n_train - n_val

    # Ensure every split has at least 1 sample
    if n_train < 1 or n_val < 1 or n_test < 1:
        raise ConfigError(
            f"Ratios {ratios} on {n_samples} samples produce empty split: "
            f"train={n_train}, val={n_val}, test={n_test}. "
            f"Need at least 3 samples."
        )

    # Half-open index ranges [start, end)
    train_end = n_train
    val_end = n_train + n_val

    indices = {
        "train": [0, train_end],
        "val": [train_end, val_end],
        "test": [val_end, n_samples],
    }

    counts = {
        "train": n_train,
        "val": n_val,
        "test": n_test,
    }

    # Map split indices to date ranges if dates provided
    date_ranges: dict[str, list[str]] = {}
    if dates is not None:
        for split_name, (start, end) in indices.items():
            # For window index t, the target event date is dates.iloc[t + W]
            first_date = str(dates.iloc[start + window_size])
            last_date = str(dates.iloc[end - 1 + window_size])
            date_ranges[split_name] = [first_date, last_date]
    else:
        date_ranges = {
            "train": ["", ""],
            "val": ["", ""],
            "test": ["", ""],
        }

    ratios_dict = {
        "train": train_r,
        "val": val_r,
        "test": test_r,
    }

    split_info = SplitInfo(
        window_size=window_size,
        ratios=ratios_dict,
        indices=indices,
        date_ranges=date_ranges,
        counts=counts,
    )

    logger.info(
        "Created splits: train=%d, val=%d, test=%d (total=%d, W=%d)",
        n_train,
        n_val,
        n_test,
        n_samples,
        window_size,
    )

    return split_info


def save_splits(split_info: SplitInfo, output_dir: str) -> Path:
    """Save split metadata to splits.json.

    Args:
        split_info: SplitInfo to persist.
        output_dir: Directory for the output file.

    Returns:
        Path to the saved JSON file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    path = out / "splits.json"
    with open(path, "w") as f:
        json.dump(asdict(split_info), f, indent=2)

    logger.info("Saved splits to %s", path)
    return path


def load_splits(path: str) -> SplitInfo:
    """Load split metadata from a splits.json file.

    Args:
        path: Path to the splits.json file.

    Returns:
        Reconstructed SplitInfo.

    Raises:
        ConfigError: If the file does not exist or is malformed.
    """
    splits_path = Path(path)

    if not splits_path.exists():
        raise ConfigError(f"Splits file not found: {path}")

    try:
        with open(splits_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigError(f"Failed to parse splits JSON: {path}\n{e}") from e

    try:
        return SplitInfo(
            window_size=data["window_size"],
            ratios=data["ratios"],
            indices=data["indices"],
            date_ranges=data["date_ranges"],
            counts=data["counts"],
        )
    except KeyError as e:
        raise ConfigError(
            f"Splits JSON missing required field: {e}"
        ) from e
