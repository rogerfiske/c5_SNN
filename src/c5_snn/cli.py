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


@cli.command("train")
@click.option(
    "--config",
    "config_path",
    required=True,
    help="Path to experiment config YAML.",
)
def train(config_path: str) -> None:
    """Train a model from a YAML experiment config."""
    from c5_snn.data.dataset import get_dataloaders
    from c5_snn.data.loader import load_csv
    from c5_snn.data.splits import create_splits
    from c5_snn.data.windowing import build_windows
    from c5_snn.models.base import get_model
    from c5_snn.training.trainer import Trainer
    from c5_snn.utils.config import load_config
    from c5_snn.utils.device import get_device
    from c5_snn.utils.seed import set_global_seed

    # 1. Load and validate config
    try:
        config = load_config(config_path)
    except ConfigError as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)

    log_level = config.get("log_level", "INFO")
    setup_logging(log_level)

    # 2. Set global seed
    seed = config.get("experiment", {}).get("seed", 42)
    set_global_seed(seed)

    # 3. Load data
    data_cfg = config.get("data", {})
    raw_path = data_cfg.get("raw_path", "data/raw/CA5_matrix_binary.csv")

    try:
        df = load_csv(raw_path)
    except (DataValidationError, FileNotFoundError) as e:
        click.echo(f"ERROR: Failed to load data: {e}", err=True)
        sys.exit(1)

    # 4. Build windowed tensors
    window_size = int(data_cfg.get("window_size", 21))
    try:
        X, y = build_windows(df, window_size)
    except (ConfigError, DataValidationError) as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)

    # 5. Create splits and dataloaders
    ratios = data_cfg.get("split_ratios", [0.70, 0.15, 0.15])
    split_info = create_splits(
        n_samples=X.shape[0],
        ratios=tuple(ratios),
        window_size=window_size,
        dates=df["date"] if "date" in df.columns else None,
    )

    batch_size = int(data_cfg.get("batch_size", 64))
    dataloaders = get_dataloaders(split_info, X, y, batch_size)

    # 6. Instantiate model via registry
    try:
        model = get_model(config)
    except ConfigError as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)

    # 7. Create Trainer and run
    device = get_device()
    trainer = Trainer(model, config, dataloaders, device)
    result = trainer.run()

    # 8. Print final summary
    click.echo()
    exp_name = config.get("experiment", {}).get("name", "unknown")
    click.echo(f"Training Complete: {exp_name}")
    click.echo("=" * 45)
    click.echo(f"  Best epoch:          {result['best_epoch']}")
    click.echo(f"  Total epochs:        {result['total_epochs']}")
    click.echo(f"  Best val_recall@20:  {result['best_val_recall_at_20']:.4f}")
    click.echo(f"  Output dir:          {trainer.output_dir}")
    click.echo()


@cli.command("compare")
@click.option(
    "--config",
    "config_path",
    required=True,
    help="Path to experiment config YAML (used for data + GRU model).",
)
@click.option(
    "--seeds",
    default="42,123,7",
    help="Comma-separated seeds for multi-seed GRU training.",
    show_default=True,
)
@click.option(
    "--output",
    "output_path",
    default="results/baseline_comparison.json",
    help="Path for comparison JSON output.",
    show_default=True,
)
def compare(config_path: str, seeds: str, output_path: str) -> None:
    """Train with multiple seeds, evaluate, and compare baselines."""
    import copy
    import time

    from c5_snn.data.dataset import get_dataloaders
    from c5_snn.data.loader import load_csv
    from c5_snn.data.splits import create_splits
    from c5_snn.data.windowing import build_windows
    from c5_snn.models.baselines import FrequencyBaseline, GRUBaseline
    from c5_snn.training.compare import (
        build_comparison,
        format_comparison_table,
        save_comparison,
    )
    from c5_snn.training.evaluate import evaluate_model
    from c5_snn.training.trainer import Trainer
    from c5_snn.utils.config import load_config
    from c5_snn.utils.device import get_device
    from c5_snn.utils.seed import set_global_seed

    # 1. Load config
    try:
        config = load_config(config_path)
    except ConfigError as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)

    log_level = config.get("log_level", "INFO")
    setup_logging(log_level)

    # 2. Parse seeds
    try:
        seed_list = [int(s.strip()) for s in seeds.split(",")]
    except ValueError:
        click.echo("ERROR: Seeds must be comma-separated integers", err=True)
        sys.exit(1)

    # 3. Load data and build pipeline
    data_cfg = config.get("data", {})
    raw_path = data_cfg.get("raw_path", "data/raw/CA5_matrix_binary.csv")

    try:
        df = load_csv(raw_path)
    except (DataValidationError, FileNotFoundError) as e:
        click.echo(f"ERROR: Failed to load data: {e}", err=True)
        sys.exit(1)

    window_size = int(data_cfg.get("window_size", 21))
    try:
        X, y = build_windows(df, window_size)
    except (ConfigError, DataValidationError) as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)

    ratios = data_cfg.get("split_ratios", [0.70, 0.15, 0.15])
    split_info = create_splits(
        n_samples=X.shape[0],
        ratios=tuple(ratios),
        window_size=window_size,
        dates=df["date"] if "date" in df.columns else None,
    )

    batch_size = int(data_cfg.get("batch_size", 64))
    dataloaders = get_dataloaders(split_info, X, y, batch_size)
    test_loader = dataloaders["test"]
    test_split_size = len(test_loader.dataset)
    device = get_device()

    model_results = []

    # 4. Evaluate FrequencyBaseline (deterministic, 1 run)
    click.echo("Evaluating FrequencyBaseline...")
    freq_config = {"model": {"type": "frequency_baseline"}}
    freq_model = FrequencyBaseline(freq_config)
    freq_eval = evaluate_model(freq_model, test_loader, device)
    model_results.append({
        "name": "frequency_baseline",
        "type": "heuristic",
        "phase": "baseline",
        "seed_metrics": [freq_eval["metrics"]],
        "training_time_s": 0,
        "environment": "local",
    })
    click.echo(
        f"  FrequencyBaseline recall@20: "
        f"{freq_eval['metrics']['recall_at_20']:.4f}"
    )

    # 5. Train + evaluate GRU with multiple seeds
    gru_seed_metrics = []
    total_gru_time = 0.0

    for seed in seed_list:
        click.echo(f"Training GRU baseline (seed={seed})...")
        set_global_seed(seed)

        seed_config = copy.deepcopy(config)
        seed_config["experiment"]["seed"] = seed
        seed_config["output"] = {
            "dir": f"results/baseline_gru_seed{seed}"
        }

        gru_model = GRUBaseline(seed_config)
        trainer = Trainer(gru_model, seed_config, dataloaders, device)

        t0 = time.time()
        trainer.run()
        elapsed = time.time() - t0
        total_gru_time += elapsed

        gru_eval = evaluate_model(gru_model, test_loader, device)
        gru_seed_metrics.append(gru_eval["metrics"])
        click.echo(
            f"  Seed {seed}: recall@20={gru_eval['metrics']['recall_at_20']:.4f}"
            f" ({elapsed:.1f}s)"
        )

    model_results.append({
        "name": "gru_baseline",
        "type": "learned",
        "phase": "baseline",
        "seed_metrics": gru_seed_metrics,
        "training_time_s": round(total_gru_time, 1),
        "environment": "local",
    })

    # 6. Build and save comparison report
    report = build_comparison(model_results, window_size, test_split_size)
    save_comparison(report, output_path)

    # 7. Print formatted table
    click.echo()
    click.echo(format_comparison_table(report))
    click.echo(f"Results saved to: {output_path}")
    click.echo()


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
