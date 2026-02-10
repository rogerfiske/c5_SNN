"""CSV loading for the CA5 dataset."""

import logging
from pathlib import Path

import pandas as pd

from c5_snn.utils.exceptions import DataValidationError

logger = logging.getLogger(__name__)


def load_csv(path: str) -> pd.DataFrame:
    """Load the CA5 CSV file and return the raw DataFrame.

    The date column is kept as a string — no datetime conversion.
    All 45 columns are returned as-is.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame with all columns from the CSV.

    Raises:
        DataValidationError: If the file does not exist or cannot be read.
    """
    csv_path = Path(path)

    if not csv_path.exists():
        raise DataValidationError(f"Data file not found: {path}")

    try:
        df = pd.read_csv(csv_path, dtype={"date": str})
    except Exception as e:
        raise DataValidationError(f"Failed to read CSV: {path}\n{e}") from e

    logger.info("Loaded %d rows x %d columns from %s", len(df), len(df.columns), path)
    return df
