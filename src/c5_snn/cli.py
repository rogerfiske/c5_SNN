"""CLI entry point for c5_SNN pipeline."""

import sys
from pathlib import Path

import click

from c5_snn.utils.exceptions import ConfigError, DataValidationError
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


@cli.command("evaluate")
@click.option(
    "--checkpoint",
    required=True,
    help="Path to model checkpoint (.pt file).",
)
@click.option(
    "--output-dir",
    default=None,
    help="Directory for result files (default: checkpoint parent dir).",
)
@click.option(
    "--data-dir",
    default="data/processed",
    help="Directory containing windowed tensors and splits.json.",
    show_default=True,
)
def evaluate(
    checkpoint: str, output_dir: str | None, data_dir: str
) -> None:
    """Evaluate a trained model on the test split."""
    import torch

    from c5_snn.data.dataset import get_dataloaders
    from c5_snn.data.splits import load_splits
    from c5_snn.training.evaluate import evaluate_model, export_results
    from c5_snn.utils.device import get_device

    setup_logging("INFO")

    # --- Validate checkpoint path ---
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        click.echo(f"ERROR: Checkpoint not found: {checkpoint}", err=True)
        sys.exit(1)

    # --- Load checkpoint ---
    try:
        ckpt = torch.load(
            ckpt_path, map_location="cpu", weights_only=False
        )
    except Exception as e:
        click.echo(f"ERROR: Failed to load checkpoint: {e}", err=True)
        sys.exit(1)

    model_name = ckpt.get("model_name", ckpt_path.stem)
    config = ckpt.get("config", {})

    if "model" in ckpt:
        model = ckpt["model"]
    else:
        click.echo(
            "ERROR: Checkpoint missing 'model' key. "
            "Model registry not yet available.",
            err=True,
        )
        sys.exit(1)

    # --- Load test data ---
    data_path = Path(data_dir)
    window_size = config.get("data", {}).get("window_size", 21)
    batch_size = config.get("data", {}).get("batch_size", 64)

    x_file = data_path / f"X_w{window_size}.pt"
    y_file = data_path / f"y_w{window_size}.pt"
    splits_file = data_path / "splits.json"

    for f in (x_file, y_file, splits_file):
        if not f.exists():
            click.echo(f"ERROR: Required file not found: {f}", err=True)
            sys.exit(1)

    try:
        X = torch.load(x_file, weights_only=True)
        y = torch.load(y_file, weights_only=True)
        split_info = load_splits(str(splits_file))
    except (ConfigError, Exception) as e:
        click.echo(f"ERROR: Failed to load data: {e}", err=True)
        sys.exit(1)

    loaders = get_dataloaders(split_info, X, y, batch_size)
    test_loader = loaders["test"]

    # --- Evaluate ---
    device = get_device()
    result = evaluate_model(model, test_loader, device)

    # --- Export results ---
    if output_dir is None:
        output_dir = str(ckpt_path.parent)

    export_results(
        result["metrics"],
        result["per_sample"],
        model_name,
        output_dir,
        checkpoint_path=checkpoint,
    )

    # --- Print metrics table ---
    click.echo()
    click.echo(f"Evaluation Results: {model_name}")
    click.echo("=" * 45)
    for key, value in result["metrics"].items():
        click.echo(f"  {key:<20} {value:.4f}")
    click.echo()
    click.echo(f"  Samples evaluated:  {len(result['per_sample'])}")
    click.echo(f"  Results saved to:   {output_dir}")
    click.echo()
