"""Data integrity checks for the CA5 dataset."""

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = (
    ["date"]
    + [f"m_{i}" for i in range(1, 6)]
    + [f"P_{i}" for i in range(1, 40)]
)

P_COLUMNS = [f"P_{i}" for i in range(1, 40)]
M_COLUMNS = [f"m_{i}" for i in range(1, 6)]


@dataclass
class CheckResult:
    """Result of a single validation check."""

    name: str
    passed: bool
    message: str
    failing_indices: list[int] | None = None


@dataclass
class ValidationReport:
    """Aggregated result of all validation checks."""

    passed: bool
    checks: list[CheckResult] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def _check_column_count(df: pd.DataFrame) -> CheckResult:
    """Check that exactly 45 columns are present."""
    n = len(df.columns)
    if n == 45:
        return CheckResult("Column count (45)", True, f"{n} columns found")
    return CheckResult("Column count (45)", False, f"Expected 45 columns, found {n}")


def _check_column_names(df: pd.DataFrame) -> CheckResult:
    """Check that all expected column names are present."""
    actual = list(df.columns)
    if actual == EXPECTED_COLUMNS:
        return CheckResult("Column names", True, "All expected columns present")

    missing = set(EXPECTED_COLUMNS) - set(actual)
    extra = set(actual) - set(EXPECTED_COLUMNS)
    parts = []
    if missing:
        parts.append(f"missing: {sorted(missing)}")
    if extra:
        parts.append(f"unexpected: {sorted(extra)}")
    return CheckResult("Column names", False, "; ".join(parts))


def _check_dates_monotonic(df: pd.DataFrame) -> CheckResult:
    """Check that dates are monotonically non-decreasing."""
    if "date" not in df.columns:
        return CheckResult("Dates monotonic", False, "No 'date' column found")

    dates = pd.to_datetime(df["date"])
    diffs = dates.diff().iloc[1:]
    bad_mask = diffs < pd.Timedelta(0)
    if not bad_mask.any():
        return CheckResult("Dates monotonic", True, "All dates non-decreasing")

    bad_indices = bad_mask[bad_mask].index.tolist()
    return CheckResult(
        "Dates monotonic",
        False,
        f"{len(bad_indices)} date(s) not monotonic",
        failing_indices=bad_indices,
    )


def _check_row_sums(df: pd.DataFrame) -> CheckResult:
    """Check that sum(P_1..P_39) == 5 for every row."""
    available = [c for c in P_COLUMNS if c in df.columns]
    if not available:
        return CheckResult("Row sums (== 5)", False, "No P columns found")

    row_sums = df[available].sum(axis=1)
    bad_mask = row_sums != 5
    if not bad_mask.any():
        return CheckResult(
            "Row sums (== 5)", True, f"All {len(df):,} rows have sum = 5"
        )

    bad_indices = bad_mask[bad_mask].index.tolist()
    return CheckResult(
        "Row sums (== 5)",
        False,
        f"{len(bad_indices)} row(s) have sum != 5",
        failing_indices=bad_indices,
    )


def _check_mp_consistency(df: pd.DataFrame) -> CheckResult:
    """Check that m_1..m_5 values match the P columns where P_k == 1."""
    m_avail = [c for c in M_COLUMNS if c in df.columns]
    p_avail = [c for c in P_COLUMNS if c in df.columns]
    if not m_avail or not p_avail:
        return CheckResult("m/P cross-check", False, "Missing m or P columns")

    bad_indices = []
    for idx in df.index:
        m_values = set(df.loc[idx, m_avail].values)
        p_indices = {
            int(col.split("_")[1]) for col in p_avail if df.loc[idx, col] == 1
        }
        if m_values != p_indices:
            bad_indices.append(idx)

    if not bad_indices:
        return CheckResult("m/P cross-check", True, "All rows consistent")

    return CheckResult(
        "m/P cross-check",
        False,
        f"{len(bad_indices)} row(s) have m/P mismatch",
        failing_indices=bad_indices,
    )


def validate(df: pd.DataFrame) -> ValidationReport:
    """Run all integrity checks on the CA5 DataFrame.

    All checks are executed regardless of earlier failures, so the report
    lists every issue at once.

    Args:
        df: Raw DataFrame from load_csv.

    Returns:
        ValidationReport with per-check results and summary.
    """
    checks = [
        _check_column_count(df),
        _check_column_names(df),
        _check_dates_monotonic(df),
        _check_row_sums(df),
        _check_mp_consistency(df),
    ]

    passed = all(c.passed for c in checks)

    # Build summary
    summary: dict = {"total_rows": len(df)}
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"])
        summary["date_first"] = str(dates.min().date())
        summary["date_last"] = str(dates.max().date())
    p_avail = [c for c in P_COLUMNS if c in df.columns]
    if p_avail:
        # Count unique part indices that appear as 1 anywhere
        unique_parts = sum(1 for c in p_avail if df[c].any())
        summary["unique_parts"] = unique_parts

    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        logger.info("  %-25s %s  %s", check.name, status, check.message)

    return ValidationReport(passed=passed, checks=checks, summary=summary)
