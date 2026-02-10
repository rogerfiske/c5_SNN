# STORY-1.3: Data Validation CLI Command

## Story Info

| Field | Value |
| --- | --- |
| **Story ID** | STORY-1.3 |
| **Epic** | Epic 1 — Foundation & Data Validation |
| **Sprint** | 1 |
| **Points** | 5 |
| **Priority** | Must Have |
| **Status** | Completed |
| **Dependencies** | STORY-1.2 (completed) |

---

## User Story

As a researcher,
I want to run `c5-snn validate-data` and get a clear pass/fail report on the raw dataset,
so that I can confirm data integrity before any modeling work begins.

---

## Acceptance Criteria

- [ ] **AC-1:** `src/c5_snn/data/loader.py` provides `load_csv(path: str) -> pd.DataFrame` that:
  - Loads the CSV file with `date` column parsed as string (not datetime — raw validation only)
  - Raises `DataValidationError` if the file does not exist or cannot be read
  - Returns the full DataFrame with all 45 columns
- [ ] **AC-2:** `src/c5_snn/data/validation.py` provides `validate(df: pd.DataFrame) -> ValidationReport` that runs all integrity checks:
  - **Check 1 — Column count:** Exactly 45 columns present (`date`, `m_1`..`m_5`, `P_1`..`P_39`)
  - **Check 2 — Column names:** Expected column names match exactly
  - **Check 3 — Dates monotonic:** The `date` column is monotonically non-decreasing when parsed as dates
  - **Check 4 — Row sums:** `sum(P_1..P_39) == 5` for every row
  - **Check 5 — m/P cross-check:** For each row, the set of `m_k` values matches the set of column indices `k` where `P_k == 1`
- [ ] **AC-3:** `ValidationReport` is a dataclass or similar structure containing:
  - `passed: bool` — overall pass/fail
  - `checks: list[CheckResult]` — individual check results, each with `name`, `passed`, `message`, and optionally `failing_indices` (list of row indices that failed)
  - `summary: dict` — total rows, date range (first/last), unique part count
- [ ] **AC-4:** `src/c5_snn/cli.py` has a `validate-data` subcommand (Click):
  - Option `--data-path` with default `data/raw/CA5_matrix_binary.csv`
  - Calls `load_csv()` then `validate()`
  - Prints a human-readable summary table: total rows, date range, and pass/fail per check with messages
  - Exits with code 0 on all checks passing, code 1 on any failure
- [ ] **AC-5:** The `validate-data` command uses `setup_logging` and `load_config` from STORY-1.2:
  - Logging is initialized before any work
  - All status messages use `logging`, not `print()` (Click `echo` is acceptable for CLI output formatting only)
- [ ] **AC-6:** Unit tests in `tests/test_validation.py` using fixture CSVs:
  - Valid CSV passes all checks
  - Missing column fails check 1/2
  - Wrong column name fails check 2
  - Non-monotonic dates fail check 3
  - Row with `sum(P) != 5` fails check 4 (reports which rows)
  - m/P mismatch fails check 5 (reports which rows)
- [ ] **AC-7:** Unit tests in `tests/test_loader.py`:
  - `load_csv` on a valid CSV returns a DataFrame with correct shape
  - `load_csv` on a non-existent path raises `DataValidationError`
- [ ] **AC-8:** Test fixtures in `tests/conftest.py`:
  - `tiny_csv` fixture: 20-row valid DataFrame matching the CA5 schema (date, m_1..m_5, P_1..P_39)
  - `bad_csv_missing_col` fixture: DataFrame missing one P column
  - `bad_csv_row_sum` fixture: DataFrame with one row having `sum(P) != 5`
  - `bad_csv_mp_mismatch` fixture: DataFrame where m values don't match P columns for one row
- [ ] **AC-9:** `ruff check src/ tests/` passes with zero errors
- [ ] **AC-10:** `pytest tests/ -v` passes with all tests green

---

## Technical Notes

### Architecture References

- **Section 4.1 (Raw Input):** CA5_matrix_binary.csv — 11,700 rows, 45 columns. `date` + `m_1`..`m_5` + `P_1`..`P_39`. Invariants: `sum(P) == 5`, `{m_1..m_5}` matches P-column indices.
- **Section 5.1 (Data Module):** `loader.py` for CSV loading, `validation.py` for integrity checks. These are the first two files in the data subpackage.
- **Section 7.1 (validate-data workflow):** Sequence diagram showing CLI → loader → validation → report flow.
- **Section 12.1 (Error Handling):** Use `DataValidationError` from `utils/exceptions.py` for load failures. Validation checks themselves report failures in the `ValidationReport` (they don't throw — the CLI interprets the report).
- **Section 12.3 (Error Patterns):** "Fail immediately, report all violations at once" — the validator should run ALL checks and report ALL failures, not stop at the first one.

### CSV Column Layout

```
Column 0:  date       (str, ISO-ish date format e.g. "2/4/1992")
Column 1:  m_1        (int, part number 1-39)
Column 2:  m_2        (int, part number 1-39)
Column 3:  m_3        (int, part number 1-39)
Column 4:  m_4        (int, part number 1-39)
Column 5:  m_5        (int, part number 1-39)
Column 6:  P_1        (int, 0 or 1)
Column 7:  P_2        (int, 0 or 1)
...
Column 44: P_39       (int, 0 or 1)
```

### m/P Cross-Check Logic

For each row `i`:
1. Collect `m_values = {df.iloc[i]['m_1'], ..., df.iloc[i]['m_5']}` — a set of 5 part numbers
2. Collect `p_indices = {k for k in range(1, 40) if df.iloc[i][f'P_{k}'] == 1}` — indices where P is 1
3. Check `m_values == p_indices`

This can be vectorized with pandas for efficiency.

### ValidationReport Structure

```python
@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    failing_indices: list[int] | None = None

@dataclass
class ValidationReport:
    passed: bool
    checks: list[CheckResult]
    summary: dict
```

### CLI Output Format

```
c5-snn validate-data

Data Validation Report
======================
File:        data/raw/CA5_matrix_binary.csv
Total rows:  11,700
Date range:  1992-02-04 to 2026-01-20
Unique parts: 39

Check                    Status  Message
─────────────────────────────────────────
Column count (45)        PASS    45 columns found
Column names             PASS    All expected columns present
Dates monotonic          PASS    All dates non-decreasing
Row sums (== 5)          PASS    All 11,700 rows have sum = 5
m/P cross-check          PASS    All rows consistent

Result: ALL CHECKS PASSED ✓
```

### What NOT to Implement

- No windowing (STORY-2.1)
- No splitting (STORY-2.2)
- No DataLoader or Dataset class (STORY-2.1)
- No data transformation or feature engineering
- Do not convert dates to datetime objects in the loader — just load raw strings. Date parsing is only needed for the monotonic check within validation.

---

## Definition of Done

1. All acceptance criteria (AC-1 through AC-10) are met
2. `c5-snn validate-data` works end-to-end on the real CSV
3. `c5-snn validate-data --data-path nonexistent.csv` exits with code 1 and a clear error
4. All unit tests pass: `pytest tests/ -v`
5. Lint clean: `ruff check src/ tests/`
6. Code committed to repository

---

## Dev Notes

_This section is updated during implementation._

- Implementation started: 2026-02-10
- Implementation completed: 2026-02-10
- Notes:
  - `loader.py`: load_csv with DataValidationError on missing file
  - `validation.py`: 5 checks (column count, column names, dates monotonic, row sums, m/P cross-check)
  - `cli.py`: validate-data subcommand with human-readable report table
  - Real CSV: 11,702 rows, all 5 checks PASS
  - 26 total unit tests (12 new for STORY-1.3), all green
  - Ruff check clean
