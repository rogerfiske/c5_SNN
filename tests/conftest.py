"""Shared test fixtures for c5_SNN."""

import numpy as np
import pandas as pd
import pytest


def _make_valid_row(date: str, parts: list[int]) -> dict:
    """Build one valid CA5 row from a date and 5 part numbers."""
    assert len(parts) == 5
    row: dict = {"date": date}
    for i, p in enumerate(parts, 1):
        row[f"m_{i}"] = p
    for k in range(1, 40):
        row[f"P_{k}"] = 1 if k in parts else 0
    return row


def _make_valid_df(n: int = 20) -> pd.DataFrame:
    """Create a valid CA5-format DataFrame with n rows."""
    rng = np.random.RandomState(42)
    rows = []
    base_year = 2000
    for i in range(n):
        date = f"{base_year + i // 12}/{(i % 12) + 1}/15"
        parts = sorted(rng.choice(range(1, 40), size=5, replace=False).tolist())
        rows.append(_make_valid_row(date, parts))
    return pd.DataFrame(rows)


@pytest.fixture
def tiny_csv() -> pd.DataFrame:
    """20-row valid DataFrame matching the CA5 schema."""
    return _make_valid_df(20)


@pytest.fixture
def tiny_csv_file(tiny_csv, tmp_path):
    """Write tiny_csv to a CSV file and return the path."""
    path = tmp_path / "valid.csv"
    tiny_csv.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def bad_csv_missing_col(tiny_csv) -> pd.DataFrame:
    """DataFrame missing one P column (P_39 dropped)."""
    return tiny_csv.drop(columns=["P_39"])


@pytest.fixture
def bad_csv_row_sum(tiny_csv) -> pd.DataFrame:
    """DataFrame where row 5 has sum(P) != 5 (one extra P set to 1)."""
    df = tiny_csv.copy()
    # Find a P column that is 0 in row 5 and flip it to 1 (making sum = 6)
    for k in range(1, 40):
        col = f"P_{k}"
        if df.loc[5, col] == 0:
            df.loc[5, col] = 1
            break
    return df


@pytest.fixture
def bad_csv_mp_mismatch(tiny_csv) -> pd.DataFrame:
    """DataFrame where row 3 has m values that don't match P columns."""
    df = tiny_csv.copy()
    # Swap m_1 to a value that doesn't match any P=1 column
    # Find a part number not in the current set
    current_parts = set(df.loc[3, [f"m_{i}" for i in range(1, 6)]].values)
    for new_val in range(1, 40):
        if new_val not in current_parts:
            df.loc[3, "m_1"] = new_val
            break
    return df
