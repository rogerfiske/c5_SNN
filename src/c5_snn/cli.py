"""CLI entry point for c5_SNN pipeline."""

import sys

import click

from c5_snn.utils.exceptions import DataValidationError
from c5_snn.utils.logging import setup_logging


@click.group()
def cli() -> None:
    """c5_SNN: Spiking Neural Network time-series forecasting pipeline."""


@cli.command("validate-data")
@click.option(
    "--data-path",
    default="data/raw/CA5_matrix_binary.csv",
    help="Path to the CA5 CSV file.",
    show_default=True,
)
def validate_data(data_path: str) -> None:
    """Validate the raw CA5 dataset and report integrity check results."""
    setup_logging("INFO")

    from c5_snn.data.loader import load_csv
    from c5_snn.data.validation import validate

    try:
        df = load_csv(data_path)
    except DataValidationError as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)

    report = validate(df)

    # Print human-readable report
    click.echo()
    click.echo("Data Validation Report")
    click.echo("======================")
    click.echo(f"File:         {data_path}")
    click.echo(f"Total rows:   {report.summary.get('total_rows', '?'):,}")
    if "date_first" in report.summary:
        click.echo(
            f"Date range:   {report.summary['date_first']} to {report.summary['date_last']}"
        )
    if "unique_parts" in report.summary:
        click.echo(f"Unique parts: {report.summary['unique_parts']}")
    click.echo()

    click.echo(f"{'Check':<25} {'Status':<8} Message")
    click.echo("-" * 60)
    for check in report.checks:
        status = "PASS" if check.passed else "FAIL"
        click.echo(f"{check.name:<25} {status:<8} {check.message}")
        if check.failing_indices and not check.passed:
            preview = check.failing_indices[:5]
            suffix = (
                f" ... and {len(check.failing_indices) - 5} more"
                if len(check.failing_indices) > 5
                else ""
            )
            click.echo(f"{'':25} {'':8} Rows: {preview}{suffix}")

    click.echo()
    if report.passed:
        click.echo("Result: ALL CHECKS PASSED")
    else:
        click.echo("Result: VALIDATION FAILED")
        sys.exit(1)
