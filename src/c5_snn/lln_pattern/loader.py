"""Loader and validator for CA5_date.csv (6-column positional format)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from c5_snn.utils.exceptions import DataValidationError

logger = logging.getLogger(__name__)

DATE_COL = "date"
M_COLUMNS = [f"m_{i}" for i in range(1, 6)]
EXPECTED_COLUMNS = [DATE_COL] + M_COLUMNS


def load_date_csv(path: str) -> pd.DataFrame:
    """Load CA5_date.csv and return validated DataFrame.

    Args:
        path: Path to the 6-column CSV.

    Returns:
        DataFrame with columns [date, m_1, m_2, m_3, m_4, m_5].

    Raises:
        DataValidationError: If file missing, wrong columns, or bad values.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise DataValidationError(f"Data file not found: {path}")

    try:
        df = pd.read_csv(csv_path, dtype={"date": str})
    except Exception as e:
        raise DataValidationError(f"Failed to read CSV: {path}\n{e}") from e

    validate_date_df(df)
    logger.info(
        "Loaded %d rows x %d columns from %s",
        len(df),
        len(df.columns),
        path,
    )
    return df


def validate_date_df(df: pd.DataFrame) -> None:
    """Validate the 6-column DataFrame.

    Checks:
    1. Exactly 6 columns with expected names.
    2. m_1 < m_2 < m_3 < m_4 < m_5 for every row.
    3. All m values in range [1, 39].
    4. Dates are monotonically non-decreasing.
    5. No NaN values.

    Raises:
        DataValidationError on any failure.
    """
    actual = list(df.columns)
    if actual != EXPECTED_COLUMNS:
        raise DataValidationError(
            f"Expected columns {EXPECTED_COLUMNS}, got {actual}"
        )

    if df.isna().any().any():
        raise DataValidationError("DataFrame contains NaN values")

    m_values = df[M_COLUMNS].values
    if m_values.min() < 1 or m_values.max() > 39:
        raise DataValidationError(
            f"m values out of range [1, 39]: "
            f"min={m_values.min()}, max={m_values.max()}"
        )

    diffs = np.diff(m_values, axis=1)
    if (diffs <= 0).any():
        bad_rows = np.where((diffs <= 0).any(axis=1))[0]
        raise DataValidationError(
            f"{len(bad_rows)} row(s) violate m_1 < m_2 < ... < m_5 "
            f"(first at index {bad_rows[0]})"
        )

    dates = pd.to_datetime(df["date"])
    date_diffs = dates.diff().iloc[1:]
    if (date_diffs < pd.Timedelta(0)).any():
        raise DataValidationError("Dates are not monotonically non-decreasing")
