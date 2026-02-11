"""Sliding window tensor construction for the CA5 dataset.

Transforms a validated DataFrame of binary event vectors into overlapping
windows of shape (N, W, 39) paired with next-event targets of shape (N, 39).
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from c5_snn.data.validation import P_COLUMNS
from c5_snn.utils.exceptions import ConfigError, DataValidationError

logger = logging.getLogger(__name__)

MIN_WINDOW_SIZE = 7
MAX_WINDOW_SIZE = 90


def build_windows(
    df: pd.DataFrame, window_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build sliding window tensors from a validated DataFrame.

    Extracts the 39 P-columns, constructs overlapping windows of size W,
    and pairs each window with the next-event target.

    Args:
        df: Validated DataFrame with columns date, m_1..m_5, P_1..P_39.
        window_size: Number of events per input window (W). Must be 7-90.

    Returns:
        Tuple of (X, y) where:
            X: Tensor of shape (N, W, 39) -- input windows, dtype float32
            y: Tensor of shape (N, 39) -- next-event targets, dtype float32
            N = len(df) - window_size

    Raises:
        ConfigError: If window_size is outside the valid range [7, 90].
        DataValidationError: If DataFrame has fewer than window_size + 1 rows.
    """
    if (
        not isinstance(window_size, int)
        or window_size < MIN_WINDOW_SIZE
        or window_size > MAX_WINDOW_SIZE
    ):
        raise ConfigError(
            f"window_size must be an integer in [{MIN_WINDOW_SIZE}, {MAX_WINDOW_SIZE}], "
            f"got {window_size}"
        )

    n_rows = len(df)
    n_samples = n_rows - window_size

    if n_samples < 1:
        raise DataValidationError(
            f"DataFrame has {n_rows} rows, need at least {window_size + 1} "
            f"for window_size={window_size}"
        )

    # Extract P-columns as a float32 tensor: shape (n_rows, 39)
    p_values = torch.tensor(df[P_COLUMNS].values, dtype=torch.float32)

    # Build sliding windows using stacking
    # X[t] = p_values[t : t + window_size],  y[t] = p_values[t + window_size]
    X = torch.stack([p_values[t : t + window_size] for t in range(n_samples)])
    y = p_values[window_size : window_size + n_samples]

    logger.info(
        "Built %d windows (W=%d, features=%d) from %d rows",
        n_samples,
        window_size,
        len(P_COLUMNS),
        n_rows,
    )

    return X, y


def save_tensors(
    X: torch.Tensor,
    y: torch.Tensor,
    window_size: int,
    source_path: str,
    df: pd.DataFrame,
    output_dir: str = "data/processed",
) -> dict:
    """Save windowed tensors and metadata to disk.

    Creates the output directory if it doesn't exist. Saves X and y as
    .pt files and writes a JSON metadata file for provenance tracking.

    Args:
        X: Input tensor of shape (N, W, 39).
        y: Target tensor of shape (N, 39).
        window_size: Window size used for construction.
        source_path: Path to the source CSV file (for hash computation).
        df: Original DataFrame (for date range extraction).
        output_dir: Directory for output files.

    Returns:
        Metadata dict (also saved as JSON).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    x_path = out / f"X_w{window_size}.pt"
    y_path = out / f"y_w{window_size}.pt"
    meta_path = out / f"tensor_meta_w{window_size}.json"

    # Save tensors
    torch.save(X, x_path)
    torch.save(y, y_path)
    logger.info("Saved X to %s (%s)", x_path, list(X.shape))
    logger.info("Saved y to %s (%s)", y_path, list(y.shape))

    # Compute source file hash
    source_hash = _sha256(source_path)

    # Extract date range
    date_first = str(df["date"].iloc[0])
    date_last = str(df["date"].iloc[-1])

    metadata = {
        "source_file": source_path,
        "source_hash": f"sha256:{source_hash}",
        "window_size": window_size,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[2]),
        "date_range": [date_first, date_last],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata to %s", meta_path)

    return metadata


def _sha256(file_path: str) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
