"""Tests for data validation checks."""

from c5_snn.data.validation import validate


class TestValidate:
    def test_valid_csv_passes(self, tiny_csv):
        """A correctly formed CSV passes all 5 checks."""
        report = validate(tiny_csv)
        assert report.passed is True
        assert len(report.checks) == 5
        for check in report.checks:
            assert check.passed, f"{check.name} failed: {check.message}"

    def test_summary_fields(self, tiny_csv):
        """The report summary contains expected fields."""
        report = validate(tiny_csv)
        assert report.summary["total_rows"] == 20
        assert "date_first" in report.summary
        assert "date_last" in report.summary
        assert "unique_parts" in report.summary

    def test_missing_column_fails_column_count(self, bad_csv_missing_col):
        """A DataFrame with a missing column fails column count check."""
        report = validate(bad_csv_missing_col)
        assert report.passed is False

        count_check = report.checks[0]
        assert count_check.name == "Column count (45)"
        assert count_check.passed is False
        assert "44" in count_check.message or "Expected 45" in count_check.message

    def test_missing_column_fails_column_names(self, bad_csv_missing_col):
        """A DataFrame with a missing column fails column names check."""
        report = validate(bad_csv_missing_col)

        names_check = report.checks[1]
        assert names_check.name == "Column names"
        assert names_check.passed is False
        assert "P_39" in names_check.message

    def test_wrong_column_name_fails(self, tiny_csv):
        """A DataFrame with a renamed column fails column names check."""
        df = tiny_csv.rename(columns={"P_1": "X_1"})
        report = validate(df)

        names_check = report.checks[1]
        assert names_check.passed is False
        assert "P_1" in names_check.message

    def test_non_monotonic_dates_fail(self, tiny_csv):
        """Non-monotonic dates fail check 3."""
        df = tiny_csv.copy()
        # Swap rows 0 and 10 dates to break monotonicity
        df.loc[0, "date"], df.loc[10, "date"] = df.loc[10, "date"], df.loc[0, "date"]
        report = validate(df)

        date_check = report.checks[2]
        assert date_check.passed is False
        assert date_check.failing_indices is not None
        assert len(date_check.failing_indices) > 0

    def test_bad_row_sum_fails(self, bad_csv_row_sum):
        """A row with sum(P) != 5 fails check 4 and reports the row index."""
        report = validate(bad_csv_row_sum)

        sum_check = report.checks[3]
        assert sum_check.passed is False
        assert sum_check.failing_indices is not None
        assert 5 in sum_check.failing_indices

    def test_mp_mismatch_fails(self, bad_csv_mp_mismatch):
        """A row with m/P mismatch fails check 5 and reports the row index."""
        report = validate(bad_csv_mp_mismatch)

        mp_check = report.checks[4]
        assert mp_check.passed is False
        assert mp_check.failing_indices is not None
        assert 3 in mp_check.failing_indices

    def test_all_checks_run_even_on_failure(self, bad_csv_row_sum):
        """All 5 checks run even when some fail (report-all-at-once pattern)."""
        report = validate(bad_csv_row_sum)
        assert len(report.checks) == 5
