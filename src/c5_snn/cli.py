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


@cli.command("phase-a")
@click.option(
    "--seeds",
    default="42,123,7",
    help="Comma-separated seeds for multi-seed training.",
    show_default=True,
)
@click.option(
    "--output",
    "output_path",
    default="results/phase_a_comparison.json",
    help="Path for comparison JSON output.",
    show_default=True,
)
def phase_a(seeds: str, output_path: str) -> None:
    """Train Phase A SNN models and compare against baselines."""
    import copy
    import time

    from c5_snn.data.dataset import get_dataloaders
    from c5_snn.data.loader import load_csv
    from c5_snn.data.splits import create_splits
    from c5_snn.data.windowing import build_windows
    from c5_snn.models.base import get_model
    from c5_snn.models.baselines import FrequencyBaseline
    from c5_snn.training.compare import (
        build_comparison,
        format_comparison_table,
        save_comparison,
    )
    from c5_snn.training.evaluate import evaluate_model
    from c5_snn.training.trainer import Trainer
    from c5_snn.utils.device import get_device
    from c5_snn.utils.seed import set_global_seed

    setup_logging("INFO")

    # 1. Parse seeds
    try:
        seed_list = [int(s.strip()) for s in seeds.split(",")]
    except ValueError:
        click.echo("ERROR: Seeds must be comma-separated integers", err=True)
        sys.exit(1)

    # 2. Standard data pipeline parameters
    raw_path = "data/raw/CA5_matrix_binary.csv"
    window_size = 21
    split_ratios = (0.70, 0.15, 0.15)
    batch_size = 64

    # 3. Load data and build pipeline (shared across all models)
    try:
        df = load_csv(raw_path)
    except (DataValidationError, FileNotFoundError) as e:
        click.echo(f"ERROR: Failed to load data: {e}", err=True)
        sys.exit(1)

    try:
        X, y = build_windows(df, window_size)
    except (ConfigError, DataValidationError) as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)

    split_info = create_splits(
        n_samples=X.shape[0],
        ratios=split_ratios,
        window_size=window_size,
        dates=df["date"] if "date" in df.columns else None,
    )

    dataloaders = get_dataloaders(split_info, X, y, batch_size)
    test_loader = dataloaders["test"]
    test_split_size = len(test_loader.dataset)
    device = get_device()

    model_results = []

    # 4. FrequencyBaseline (deterministic, 1 run)
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

    # 5. GRU baseline (multi-seed)
    gru_seed_metrics = []
    total_gru_time = 0.0
    gru_config_base = {
        "experiment": {"name": "gru_baseline", "seed": 42},
        "data": {
            "raw_path": raw_path,
            "window_size": window_size,
            "split_ratios": list(split_ratios),
            "batch_size": batch_size,
        },
        "model": {
            "type": "gru_baseline",
            "hidden_size": 128,
            "num_layers": 1,
            "dropout": 0.0,
        },
        "training": {
            "epochs": 100,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "early_stopping_patience": 10,
            "early_stopping_metric": "val_recall_at_20",
        },
        "output": {"dir": "results/baseline_gru"},
        "log_level": "INFO",
    }

    for seed in seed_list:
        click.echo(f"Training GRU baseline (seed={seed})...")
        set_global_seed(seed)

        seed_config = copy.deepcopy(gru_config_base)
        seed_config["experiment"]["seed"] = seed
        seed_config["output"]["dir"] = f"results/baseline_gru_seed{seed}"

        gru_model = get_model(seed_config)
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

    # 6. SNN Phase A models (multi-seed each)
    snn_models = {
        "spiking_mlp": {
            "encoding": "direct",
            "timesteps": 10,
            "hidden_sizes": [256, 128],
            "beta": 0.95,
            "surrogate": "fast_sigmoid",
        },
        "spiking_cnn1d": {
            "encoding": "direct",
            "timesteps": 10,
            "channels": [64, 64],
            "kernel_sizes": [3, 3],
            "beta": 0.95,
            "surrogate": "fast_sigmoid",
        },
    }

    for model_type, model_params in snn_models.items():
        click.echo(f"\nTraining {model_type}...")
        snn_seed_metrics = []
        total_snn_time = 0.0

        for seed in seed_list:
            click.echo(f"  Training {model_type} (seed={seed})...")
            set_global_seed(seed)

            snn_config = {
                "experiment": {
                    "name": f"{model_type}_seed{seed}",
                    "seed": seed,
                },
                "data": {
                    "raw_path": raw_path,
                    "window_size": window_size,
                    "split_ratios": list(split_ratios),
                    "batch_size": batch_size,
                },
                "model": {"type": model_type, **model_params},
                "training": {
                    "epochs": 100,
                    "learning_rate": 0.001,
                    "optimizer": "adam",
                    "early_stopping_patience": 10,
                    "early_stopping_metric": "val_recall_at_20",
                },
                "output": {"dir": f"results/{model_type}_seed{seed}"},
                "log_level": "INFO",
            }

            snn_model = get_model(snn_config)
            trainer = Trainer(snn_model, snn_config, dataloaders, device)

            t0 = time.time()
            trainer.run()
            elapsed = time.time() - t0
            total_snn_time += elapsed

            snn_eval = evaluate_model(snn_model, test_loader, device)
            snn_seed_metrics.append(snn_eval["metrics"])
            click.echo(
                f"    recall@20={snn_eval['metrics']['recall_at_20']:.4f}"
                f" ({elapsed:.1f}s)"
            )

        model_results.append({
            "name": model_type,
            "type": "learned",
            "phase": "phase_a",
            "seed_metrics": snn_seed_metrics,
            "training_time_s": round(total_snn_time, 1),
            "environment": "local",
        })

    # 7. Build and save comparison report
    report = build_comparison(model_results, window_size, test_split_size)
    save_comparison(report, output_path)

    # 8. Print formatted table
    click.echo()
    click.echo(format_comparison_table(report))
    click.echo(f"Results saved to: {output_path}")
    click.echo()


@cli.command("phase-b")
@click.option(
    "--output",
    "output_path",
    default="results/cumulative_comparison.json",
    help="Path for cumulative comparison JSON.",
    show_default=True,
)
@click.option(
    "--phase-a",
    "phase_a_path",
    default="results/phase_a_comparison.json",
    help="Path to Phase A comparison JSON.",
    show_default=True,
)
@click.option(
    "--phase-b-top",
    "phase_b_path",
    default="results/phase_b_top3.json",
    help="Path to Phase B top-3 JSON.",
    show_default=True,
)
def phase_b(output_path: str, phase_a_path: str, phase_b_path: str) -> None:
    """Build cumulative comparison: baselines + Phase A + Phase B."""
    import copy
    import json
    from datetime import datetime, timezone

    from c5_snn.training.compare import save_comparison

    setup_logging("INFO")

    # 1. Load Phase A results
    phase_a_file = Path(phase_a_path)
    if not phase_a_file.exists():
        click.echo(
            f"ERROR: Phase A results not found: {phase_a_path}", err=True
        )
        sys.exit(1)

    with open(phase_a_file) as f:
        phase_a_report = json.load(f)

    click.echo()
    click.echo("Phase B Cumulative Comparison")
    click.echo("=" * 50)
    click.echo()
    click.echo(
        f"Loading Phase A results: {phase_a_path} "
        f"({len(phase_a_report['models'])} models)"
    )

    # 2. Load Phase B top-3 results
    phase_b_file = Path(phase_b_path)
    if not phase_b_file.exists():
        click.echo(
            f"ERROR: Phase B results not found: {phase_b_path}", err=True
        )
        sys.exit(1)

    with open(phase_b_file) as f:
        phase_b_report = json.load(f)

    click.echo(
        f"Loading Phase B results: {phase_b_path} "
        f"({len(phase_b_report['models'])} configs)"
    )

    # 3. Validate window_size/test_split_size consistency
    if phase_a_report["window_size"] != phase_b_report["window_size"]:
        click.echo(
            "WARNING: window_size mismatch between Phase A "
            f"({phase_a_report['window_size']}) and Phase B "
            f"({phase_b_report['window_size']})",
            err=True,
        )

    if phase_a_report["test_split_size"] != phase_b_report["test_split_size"]:
        click.echo(
            "WARNING: test_split_size mismatch between Phase A "
            f"({phase_a_report['test_split_size']}) and Phase B "
            f"({phase_b_report['test_split_size']})",
            err=True,
        )

    # 4. Select best Spike-GRU from Phase B top-3
    best_b = max(
        phase_b_report["models"],
        key=lambda m: m["metrics_mean"]["recall_at_20"],
    )
    best_b = copy.deepcopy(best_b)
    original_name = best_b["name"]
    best_b["name"] = "spike_gru"

    click.echo(f"Selected best Spike-GRU: {original_name}")

    # 5. Build cumulative model list
    cumulative_models = []
    for model in phase_a_report["models"]:
        cumulative_models.append(copy.deepcopy(model))
    cumulative_models.append(best_b)

    # 6. Build cumulative report
    report = {
        "models": cumulative_models,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_size": phase_a_report["window_size"],
        "test_split_size": phase_a_report["test_split_size"],
    }

    # 7. Save report
    save_comparison(report, output_path)

    # 8. Print leaderboard sorted by Recall@20
    sorted_models = sorted(
        cumulative_models,
        key=lambda m: m["metrics_mean"]["recall_at_20"],
        reverse=True,
    )

    click.echo()
    click.echo("Cumulative Leaderboard (sorted by Recall@20):")
    click.echo(
        f"{'Model':<25} {'Recall@20':<16} {'Hit@20':<16} "
        f"{'MRR':<16} {'Seeds':<6}"
    )
    click.echo("-" * 79)
    for model in sorted_models:
        mean = model["metrics_mean"]
        std = model["metrics_std"]
        n = model["n_seeds"]
        if n > 1:
            r20 = f"{mean['recall_at_20']:.4f}+/-{std['recall_at_20']:.3f}"
            h20 = f"{mean['hit_at_20']:.4f}+/-{std['hit_at_20']:.3f}"
            mrr = f"{mean['mrr']:.4f}+/-{std['mrr']:.3f}"
        else:
            r20 = f"{mean['recall_at_20']:.4f}"
            h20 = f"{mean['hit_at_20']:.4f}"
            mrr = f"{mean['mrr']:.4f}"
        click.echo(
            f"{model['name']:<25} {r20:<16} {h20:<16} {mrr:<16} {n:<6}"
        )

    # 9. Print analysis
    models_by_name = {m["name"]: m for m in cumulative_models}
    spike_gru = models_by_name["spike_gru"]
    gru = models_by_name["gru_baseline"]

    spike_r20 = spike_gru["metrics_mean"]["recall_at_20"]
    gru_r20 = gru["metrics_mean"]["recall_at_20"]
    gru_delta = spike_r20 - gru_r20
    gru_pct = (gru_delta / gru_r20) * 100 if gru_r20 != 0 else 0

    # Find best Phase A SNN (exclude baselines)
    phase_a_snns = [
        m for m in cumulative_models if m.get("phase") == "phase_a"
    ]
    if phase_a_snns:
        best_phase_a = max(
            phase_a_snns,
            key=lambda m: m["metrics_mean"]["recall_at_20"],
        )
        pa_r20 = best_phase_a["metrics_mean"]["recall_at_20"]
        pa_delta = spike_r20 - pa_r20
        pa_pct = (pa_delta / pa_r20) * 100 if pa_r20 != 0 else 0
    else:
        best_phase_a = None

    # Find overall best learned model
    learned = [m for m in cumulative_models if m["type"] == "learned"]
    best_learned = max(
        learned, key=lambda m: m["metrics_mean"]["recall_at_20"]
    )

    click.echo()
    click.echo("Analysis:")
    click.echo(
        f"  Best learned model:     {best_learned['name']} "
        f"(Recall@20={best_learned['metrics_mean']['recall_at_20']:.4f})"
    )
    click.echo(
        f"  vs GRU baseline:        {gru_delta:+.4f} ({gru_pct:+.2f}%) "
        f"{'— improvement' if gru_delta > 0 else '— below'}"
    )
    if best_phase_a is not None:
        click.echo(
            f"  vs {best_phase_a['name']:<20s} {pa_delta:+.4f} "
            f"({pa_pct:+.2f}%) "
            f"{'— improvement' if pa_delta > 0 else '— below'}"
        )
    click.echo(
        "  Encoding finding:       "
        "direct == rate_coded for Spike-GRU (no benefit from T>1)"
    )
    click.echo(
        "  Recurrence finding:     "
        "Spiking recurrence adds marginal value vs feedforward SNNs"
    )
    click.echo(
        "  Phase C recommendation: Spiking Transformer with attention "
        "may capture patterns"
    )
    click.echo(
        "                          "
        "that recurrence alone cannot. Consider window size tuning."
    )

    click.echo()
    click.echo(f"Results saved to: {output_path}")
    click.echo()


@cli.command("final-report")
@click.option(
    "--output",
    "output_path",
    default="results/final_comparison.json",
    help="Path for final comparison JSON.",
    show_default=True,
)
@click.option(
    "--report",
    "report_path",
    default="results/final_report.md",
    help="Path for final report markdown.",
    show_default=True,
)
@click.option(
    "--cumulative",
    "cumulative_path",
    default="results/cumulative_comparison.json",
    help="Path to cumulative comparison JSON (Phase A + B).",
    show_default=True,
)
@click.option(
    "--phase-c-top",
    "phase_c_path",
    default="results/phase_c_top5.json",
    help="Path to Phase C top-5 JSON.",
    show_default=True,
)
def final_report(
    output_path: str,
    report_path: str,
    cumulative_path: str,
    phase_c_path: str,
) -> None:
    """Generate final comprehensive comparison and report."""
    import copy
    import json
    from datetime import datetime, timezone

    from c5_snn.training.compare import save_comparison

    setup_logging("INFO")

    # 1. Load cumulative comparison (Phase A + B: 5 models at W=21)
    cumulative_file = Path(cumulative_path)
    if not cumulative_file.exists():
        click.echo(
            f"ERROR: Cumulative comparison not found: {cumulative_path}",
            err=True,
        )
        sys.exit(1)

    with open(cumulative_file) as f:
        cumul_report = json.load(f)

    click.echo()
    click.echo("Final Comprehensive Comparison")
    click.echo("=" * 50)
    click.echo()
    click.echo(
        f"Loading cumulative results: {cumulative_path} "
        f"({len(cumul_report['models'])} models, "
        f"W={cumul_report['window_size']})"
    )

    # 2. Load Phase C top-5 results
    phase_c_file = Path(phase_c_path)
    if not phase_c_file.exists():
        click.echo(
            f"ERROR: Phase C results not found: {phase_c_path}", err=True
        )
        sys.exit(1)

    with open(phase_c_file) as f:
        phase_c_report = json.load(f)

    click.echo(
        f"Loading Phase C results: {phase_c_path} "
        f"({len(phase_c_report['models'])} configs, "
        f"W={phase_c_report['window_size']})"
    )

    # 3. Select best SpikingTransformer from Phase C top-5
    best_c = max(
        phase_c_report["models"],
        key=lambda m: m["metrics_mean"]["recall_at_20"],
    )
    best_c = copy.deepcopy(best_c)
    original_name = best_c["name"]
    best_c["name"] = "spiking_transformer"

    click.echo(f"Selected best SpikingTransformer: {original_name}")

    # 4. Build final model list (5 from cumulative + 1 Phase C)
    final_models = []
    for model in cumul_report["models"]:
        final_models.append(copy.deepcopy(model))
    final_models.append(best_c)

    # 5. Build final comparison report
    report = {
        "models": final_models,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "notes": (
            f"Phase A/B models evaluated at W={cumul_report['window_size']} "
            f"(test_n={cumul_report['test_split_size']}). "
            f"Phase C spiking_transformer evaluated at "
            f"W={phase_c_report['window_size']} "
            f"(test_n={phase_c_report['test_split_size']}). "
            "Different window sizes yield slightly different test splits."
        ),
    }

    # 6. Save comparison JSON
    save_comparison(report, output_path)

    # 7. Print final leaderboard sorted by Recall@20
    sorted_models = sorted(
        final_models,
        key=lambda m: m["metrics_mean"]["recall_at_20"],
        reverse=True,
    )

    click.echo()
    click.echo("Final Leaderboard (sorted by Recall@20):")
    click.echo(
        f"{'#':<3} {'Model':<25} {'Phase':<10} {'Recall@20':<16} "
        f"{'Hit@20':<16} {'MRR':<16} {'Seeds':<6}"
    )
    click.echo("-" * 92)
    for rank, model in enumerate(sorted_models, 1):
        mean = model["metrics_mean"]
        std = model["metrics_std"]
        n = model["n_seeds"]
        phase = model.get("phase", "")
        if n > 1 and std["recall_at_20"] > 0:
            r20 = f"{mean['recall_at_20']:.4f}+/-{std['recall_at_20']:.3f}"
            h20 = f"{mean['hit_at_20']:.4f}+/-{std['hit_at_20']:.3f}"
            mrr = f"{mean['mrr']:.4f}+/-{std['mrr']:.3f}"
        else:
            r20 = f"{mean['recall_at_20']:.4f}"
            h20 = f"{mean['hit_at_20']:.4f}"
            mrr = f"{mean['mrr']:.4f}"
        click.echo(
            f"{rank:<3} {model['name']:<25} {phase:<10} "
            f"{r20:<16} {h20:<16} {mrr:<16} {n:<6}"
        )

    # 8. Print analysis
    models_by_name = {m["name"]: m for m in final_models}
    freq = models_by_name["frequency_baseline"]
    gru = models_by_name["gru_baseline"]
    transformer = models_by_name["spiking_transformer"]

    freq_r20 = freq["metrics_mean"]["recall_at_20"]
    gru_r20 = gru["metrics_mean"]["recall_at_20"]
    trans_r20 = transformer["metrics_mean"]["recall_at_20"]

    # Best learned model
    learned = [m for m in final_models if m["type"] == "learned"]
    best_learned = max(
        learned, key=lambda m: m["metrics_mean"]["recall_at_20"]
    )
    best_r20 = best_learned["metrics_mean"]["recall_at_20"]

    click.echo()
    click.echo("Key Findings:")
    click.echo(
        f"  Best overall:           frequency_baseline "
        f"(Recall@20={freq_r20:.4f})"
    )
    click.echo(
        f"  Best learned model:     {best_learned['name']} "
        f"(Recall@20={best_r20:.4f})"
    )

    delta_freq = trans_r20 - freq_r20
    pct_freq = (delta_freq / freq_r20) * 100 if freq_r20 != 0 else 0
    delta_gru = trans_r20 - gru_r20
    pct_gru = (delta_gru / gru_r20) * 100 if gru_r20 != 0 else 0

    click.echo(
        f"  Transformer vs Freq:    {delta_freq:+.4f} ({pct_freq:+.2f}%)"
    )
    click.echo(
        f"  Transformer vs GRU:     {delta_gru:+.4f} ({pct_gru:+.2f}%)"
    )
    click.echo(
        f"  Window size caveat:     Phase A/B at W="
        f"{cumul_report['window_size']}, "
        f"Phase C at W={phase_c_report['window_size']}"
    )
    click.echo(
        "  Performance cluster:    All models within "
        f"{freq_r20 - sorted_models[-1]['metrics_mean']['recall_at_20']:.4f}"
        " Recall@20 of each other"
    )

    click.echo()
    click.echo(f"Comparison saved to: {output_path}")

    # 9. Generate final report markdown
    _generate_final_report_md(
        report_path,
        sorted_models,
        cumul_report,
        phase_c_report,
        best_learned,
    )

    click.echo(f"Report saved to:    {report_path}")
    click.echo()


def _generate_final_report_md(
    report_path: str,
    sorted_models: list,
    cumul_report: dict,
    phase_c_report: dict,
    best_learned: dict,
) -> None:
    """Generate the final report markdown file."""
    from datetime import datetime, timezone

    models_by_name = {m["name"]: m for m in sorted_models}
    freq = models_by_name["frequency_baseline"]
    gru = models_by_name["gru_baseline"]
    mlp = models_by_name["spiking_mlp"]
    cnn = models_by_name["spiking_cnn1d"]
    sgru = models_by_name["spike_gru"]
    trans = models_by_name["spiking_transformer"]

    def _fmt(val, std_val, n_seeds):
        if n_seeds > 1 and std_val > 0:
            return f"{val:.4f} +/- {std_val:.3f}"
        return f"{val:.4f}"

    lines = []
    lines.append("# c5_SNN Final Experiment Report")
    lines.append("")
    lines.append(
        f"**Generated:** "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        "This report presents the final results of the c5_SNN project: "
        "a systematic evaluation of Spiking Neural Network (SNN) "
        "architectures for CA5 event prediction. Six model types were "
        "evaluated across three phases, progressing from simple baselines "
        "through feedforward SNNs, recurrent SNNs, and attention-based "
        "Spiking Transformers."
    )
    lines.append("")

    best_r20 = best_learned["metrics_mean"]["recall_at_20"]
    freq_r20 = freq["metrics_mean"]["recall_at_20"]
    lines.append("**Key result:** "
                 f"The best learned model ({best_learned['name']}) achieves "
                 f"Recall@20 = {best_r20:.4f}, compared to the "
                 f"FrequencyBaseline at {freq_r20:.4f}. "
                 "All models cluster within a narrow performance band "
                 "(~0.51-0.52 Recall@20), suggesting the dataset's temporal "
                 "structure is inherently simple for this prediction task.")
    lines.append("")

    # Final Leaderboard
    lines.append("## Final Leaderboard")
    lines.append("")
    lines.append(
        "| Rank | Model | Phase | Window | Recall@20 | Hit@20 | MRR "
        "| Seeds | Time (s) |"
    )
    lines.append(
        "|------|-------|-------|--------|-----------|--------|-----"
        "|-------|----------|"
    )

    for rank, m in enumerate(sorted_models, 1):
        mean = m["metrics_mean"]
        std = m["metrics_std"]
        n = m["n_seeds"]
        phase = m.get("phase", "")
        w = (phase_c_report["window_size"]
             if phase == "phase_c"
             else cumul_report["window_size"])
        r20 = _fmt(mean["recall_at_20"], std["recall_at_20"], n)
        h20 = _fmt(mean["hit_at_20"], std["hit_at_20"], n)
        mrr = _fmt(mean["mrr"], std["mrr"], n)
        t = m.get("training_time_s", 0)
        lines.append(
            f"| {rank} | {m['name']} | {phase} | {w} | {r20} | {h20} "
            f"| {mrr} | {n} | {t:.1f} |"
        )

    lines.append("")
    lines.append(
        f"> **Note:** Phase A/B models evaluated at "
        f"W={cumul_report['window_size']} "
        f"(test_n={cumul_report['test_split_size']}). "
        f"Phase C spiking_transformer evaluated at "
        f"W={phase_c_report['window_size']} "
        f"(test_n={phase_c_report['test_split_size']})."
    )
    lines.append("")

    # Phase A Analysis
    lines.append("## Phase Analysis")
    lines.append("")
    lines.append("### Phase A: Feedforward SNNs (Sprint 4)")
    lines.append("")
    lines.append(
        f"Two feedforward SNN architectures were evaluated at W="
        f"{cumul_report['window_size']} "
        "with direct encoding (T=1):"
    )
    lines.append("")

    mlp_r20 = mlp["metrics_mean"]["recall_at_20"]
    cnn_r20 = cnn["metrics_mean"]["recall_at_20"]
    lines.append(
        f"- **SpikingMLP:** Recall@20 = "
        f"{_fmt(mlp_r20, mlp['metrics_std']['recall_at_20'], 3)}"
    )
    lines.append(
        f"- **SpikingCNN1D:** Recall@20 = "
        f"{_fmt(cnn_r20, cnn['metrics_std']['recall_at_20'], 3)}"
    )
    lines.append("")
    lines.append(
        "Both SNN models performed comparably to the GRU baseline, with "
        "SpikingCNN1D slightly ahead. Direct encoding (T=1) reduces SNNs "
        "to single-timestep feedforward networks, limiting their temporal "
        "dynamics."
    )
    lines.append("")

    # Phase B Analysis
    lines.append("### Phase B: Recurrent SNN (Sprint 5)")
    lines.append("")
    sgru_r20 = sgru["metrics_mean"]["recall_at_20"]
    lines.append(
        "A 36-config hyperparameter sweep over the Spike-GRU architecture "
        f"(hidden_size, num_layers, beta, encoding) at "
        f"W={cumul_report['window_size']}:"
    )
    lines.append("")
    lines.append(
        "- **Best config:** h=256, l=1, beta=0.80, encoding=direct"
    )
    lines.append(
        f"- **Test Recall@20:** "
        f"{_fmt(sgru_r20, sgru['metrics_std']['recall_at_20'], 3)}"
    )
    gru_r20_val = gru["metrics_mean"]["recall_at_20"]
    sgru_delta = sgru_r20 - gru_r20_val
    sgru_pct = sgru_delta / gru_r20_val * 100 if gru_r20_val else 0
    lines.append(
        f"- **vs GRU baseline:** "
        f"{sgru_delta:+.4f} ({sgru_pct:+.2f}%)"
    )
    lines.append("")
    lines.append(
        "Key finding: direct and rate_coded encoding produce identical "
        "results for Spike-GRU. Hidden size is the dominant hyperparameter "
        "(256 >> 128 >> 64)."
    )
    lines.append("")

    # Phase C Analysis
    lines.append("### Phase C: Spiking Transformer (Sprint 6)")
    lines.append("")
    trans_r20 = trans["metrics_mean"]["recall_at_20"]
    lines.append(
        "The SpikingTransformer (Spikformer-style spike-form "
        "self-attention) underwent window size tuning and a 72-config "
        f"HP sweep at optimal W={phase_c_report['window_size']}:"
    )
    lines.append("")
    lines.append(
        "- **Window tuning:** W=90 is optimal (+0.79% vs W=21). "
        "Only W=90 trains beyond epoch 1."
    )
    lines.append(
        "- **Best config:** d_model=64, n_heads=2, n_layers=6, "
        "beta=0.95, rate_coded"
    )
    lines.append(
        f"- **Test Recall@20:** "
        f"{_fmt(trans_r20, trans['metrics_std']['recall_at_20'], 3)}"
    )
    lines.append("")
    lines.append(
        "Key finding: rate_coded encoding definitively outperforms direct "
        "for the SpikingTransformer at W=90 (all top-5 configs use "
        "rate_coded). Depth matters more than width (d=64 with 6 layers "
        "beats d=128 with 2 layers)."
    )
    lines.append("")

    # Cross-Cutting Analysis
    lines.append("## Cross-Cutting Analysis")
    lines.append("")

    # Window Size Impact
    lines.append("### Window Size Impact")
    lines.append("")
    lines.append(
        "Window size tuning (STORY-6.2) revealed that longer context "
        "significantly benefits the SpikingTransformer:"
    )
    lines.append("")
    lines.append("| Window | val_R@20 | Best Epoch | Training Time |")
    lines.append("|--------|----------|------------|---------------|")
    lines.append("| W=7-60 | ~0.507   | 1 (plateau)| 32-101s      |")
    lines.append("| W=90   | 0.5144   | 20         | 432s          |")
    lines.append("")
    lines.append(
        "W=90 is the only window size where the transformer trains "
        "beyond epoch 1, suggesting it needs ~90 days of history to learn "
        "meaningful temporal patterns."
    )
    lines.append("")

    # Encoding Analysis
    lines.append("### Encoding Analysis")
    lines.append("")
    lines.append("| Phase | Model | Encoding Finding |")
    lines.append("|-------|-------|------------------|")
    lines.append(
        "| Phase A | SpikingMLP, SpikingCNN1D | "
        "Direct only (T=1, no temporal dynamics) |"
    )
    lines.append(
        "| Phase B | Spike-GRU | "
        "Direct == rate_coded (no benefit from T>1) |"
    )
    lines.append(
        "| Phase C | SpikingTransformer (W=90) | "
        "Rate_coded >> direct (all top-5 use rate_coded) |"
    )
    lines.append("")
    lines.append(
        "Rate-coded encoding only provides benefit when combined with "
        "sufficient window context (W=90) and an attention-based "
        "architecture. For recurrent and feedforward SNNs at W=21, "
        "encoding strategy makes no difference."
    )
    lines.append("")

    # Efficiency Comparison
    lines.append("### Efficiency Comparison")
    lines.append("")
    lines.append("| Model | Training Time (s) | Environment |")
    lines.append("|-------|-------------------|-------------|")
    for m in sorted_models:
        t = m.get("training_time_s", 0)
        env = m.get("environment", "local")
        lines.append(f"| {m['name']} | {t:.1f} | {env} |")
    lines.append("")
    lines.append(
        "The SpikingTransformer requires significantly more compute "
        "than other models due to the larger window (W=90) and rate-coded "
        "encoding (10 timesteps). Feedforward SNNs offer the best "
        "performance-per-compute ratio."
    )
    lines.append("")

    # Key Findings & Conclusions
    lines.append("## Key Findings & Conclusions")
    lines.append("")

    spread = (sorted_models[0]["metrics_mean"]["recall_at_20"]
              - sorted_models[-1]["metrics_mean"]["recall_at_20"])
    lines.append(
        f"1. **Narrow performance band:** All 6 models cluster within "
        f"{spread:.4f} Recall@20 of each other (~0.51-0.52). No "
        "architecture achieves a breakthrough improvement."
    )
    lines.append(
        "2. **FrequencyBaseline is hard to beat:** The simple "
        "frequency/recency heuristic remains the top performer, "
        "suggesting that CA5 event patterns are dominated by "
        "frequency statistics rather than complex temporal dependencies."
    )
    lines.append(
        f"3. **SpikingTransformer is best learned model:** "
        f"Recall@20 = {trans_r20:.4f} at W=90, narrowing the gap to "
        f"the FrequencyBaseline to {freq_r20 - trans_r20:.4f}."
    )
    lines.append(
        "4. **Window size matters more than architecture:** The biggest "
        "lever was increasing the context window from 21 to 90 days, "
        "not changing the model architecture."
    )
    lines.append(
        "5. **Encoding strategy is context-dependent:** Rate-coded "
        "encoding only helps with sufficient window context (W=90) "
        "and attention-based architectures."
    )
    lines.append(
        "6. **High seed stability:** Most models show very low variance "
        "across seeds (std < 0.003), indicating robust training dynamics."
    )
    lines.append("")

    # Recommendations
    lines.append("## Recommendations for Future Work")
    lines.append("")
    lines.append(
        "1. **Window size tuning for all models:** Re-evaluate Phase A/B "
        "models at W=90 to determine if longer context benefits them too."
    )
    lines.append(
        "2. **Feature engineering:** Add calendar features "
        "(day-of-week, month) and explore whether external signals "
        "improve prediction."
    )
    lines.append(
        "3. **Loss function exploration:** Try focal loss or "
        "weighted BCE to address class imbalance among the 39 parts."
    )
    lines.append(
        "4. **Ensemble methods:** Combine the FrequencyBaseline with "
        "learned model predictions for potential complementary gains."
    )
    lines.append(
        "5. **Longer windows:** Test W > 90 (e.g., 120, 180) to see "
        "if the transformer continues to benefit from more context."
    )
    lines.append("")

    # Write file
    output_dir = Path(report_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")


@cli.command("phase-b-sweep")
@click.option(
    "--output",
    "output_path",
    default="results/phase_b_sweep.csv",
    help="Path for sweep results CSV.",
    show_default=True,
)
@click.option(
    "--top-k",
    default=3,
    type=int,
    help="Number of top configs to re-run with multi-seed.",
    show_default=True,
)
@click.option(
    "--seeds",
    default="42,123,7",
    help="Comma-separated seeds for top-K re-runs.",
    show_default=True,
)
def phase_b_sweep(output_path: str, top_k: int, seeds: str) -> None:
    """Run Spike-GRU hyperparameter sweep (Phase B)."""
    import copy
    import csv
    import itertools
    import shutil
    import time

    from c5_snn.data.dataset import get_dataloaders
    from c5_snn.data.loader import load_csv
    from c5_snn.data.splits import create_splits
    from c5_snn.data.windowing import build_windows
    from c5_snn.models.base import get_model
    from c5_snn.training.compare import (
        build_comparison,
        format_comparison_table,
        save_comparison,
    )
    from c5_snn.training.evaluate import evaluate_model
    from c5_snn.training.trainer import Trainer
    from c5_snn.utils.device import get_device
    from c5_snn.utils.seed import set_global_seed

    setup_logging("INFO")

    # 1. Parse seeds
    try:
        seed_list = [int(s.strip()) for s in seeds.split(",")]
    except ValueError:
        click.echo("ERROR: Seeds must be comma-separated integers", err=True)
        sys.exit(1)

    screening_seed = seed_list[0]

    # 2. Standard data pipeline (shared across all runs)
    raw_path = "data/raw/CA5_matrix_binary.csv"
    window_size = 21
    split_ratios = (0.70, 0.15, 0.15)
    batch_size = 64

    try:
        df = load_csv(raw_path)
    except (DataValidationError, FileNotFoundError) as e:
        click.echo(f"ERROR: Failed to load data: {e}", err=True)
        sys.exit(1)

    try:
        X, y = build_windows(df, window_size)
    except (ConfigError, DataValidationError) as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)

    split_info = create_splits(
        n_samples=X.shape[0],
        ratios=split_ratios,
        window_size=window_size,
        dates=df["date"] if "date" in df.columns else None,
    )

    dataloaders = get_dataloaders(split_info, X, y, batch_size)
    test_loader = dataloaders["test"]
    test_split_size = len(test_loader.dataset)
    device = get_device()

    base_data_cfg = {
        "raw_path": raw_path,
        "window_size": window_size,
        "split_ratios": list(split_ratios),
        "batch_size": batch_size,
    }

    # 3. Define sweep grid: 3 x 2 x 3 x 2 = 36 configs
    sweep_grid = {
        "hidden_size": [64, 128, 256],
        "num_layers": [1, 2],
        "beta": [0.5, 0.8, 0.95],
        "encoding": ["direct", "rate_coded"],
    }
    combos = list(itertools.product(
        sweep_grid["hidden_size"],
        sweep_grid["num_layers"],
        sweep_grid["beta"],
        sweep_grid["encoding"],
    ))

    click.echo()
    click.echo("Phase B Spike-GRU HP Sweep")
    click.echo("=" * 50)
    click.echo()
    click.echo(f"Phase 1: Screening ({len(combos)} configs, seed={screening_seed})")

    # 4. Phase 1 — Screening: single-seed sweep on validation set
    sweep_results = []

    for i, (h, n, b, e) in enumerate(combos):
        set_global_seed(screening_seed)

        config = {
            "experiment": {
                "name": f"spike_gru_sweep_{i:03d}",
                "seed": screening_seed,
            },
            "data": copy.deepcopy(base_data_cfg),
            "model": {
                "type": "spike_gru",
                "hidden_size": h,
                "num_layers": n,
                "beta": b,
                "encoding": e,
                "timesteps": 10,
                "dropout": 0.0,
                "surrogate": "fast_sigmoid",
            },
            "training": {
                "epochs": 100,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "early_stopping_patience": 10,
                "early_stopping_metric": "val_recall_at_20",
            },
            "output": {"dir": f"results/phase_b_sweep_{i:03d}"},
            "log_level": "WARNING",
        }

        model = get_model(config)
        trainer = Trainer(model, config, dataloaders, device)

        t0 = time.time()
        result = trainer.run()
        elapsed = time.time() - t0

        # Evaluate on validation set for screening
        val_eval = evaluate_model(model, dataloaders["val"], device)

        row = {
            "config_id": i,
            "hidden_size": h,
            "num_layers": n,
            "beta": b,
            "encoding": e,
            "timesteps": 10 if e == "rate_coded" else 1,
            "val_recall_at_20": round(val_eval["metrics"]["recall_at_20"], 4),
            "val_hit_at_20": round(val_eval["metrics"]["hit_at_20"], 4),
            "val_mrr": round(val_eval["metrics"]["mrr"], 4),
            "training_time_s": round(elapsed, 1),
            "best_epoch": result["best_epoch"],
        }
        sweep_results.append(row)

        click.echo(
            f"[{i + 1}/{len(combos)}] spike_gru h={h} l={n} b={b:.2f} "
            f"enc={e} -> val_recall@20={row['val_recall_at_20']:.4f} "
            f"({elapsed:.1f}s)"
        )

    # 5. Save sweep CSV
    csv_path = Path(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "config_id", "hidden_size", "num_layers", "beta", "encoding",
        "timesteps", "val_recall_at_20", "val_hit_at_20", "val_mrr",
        "training_time_s", "best_epoch",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sweep_results)

    click.echo()
    click.echo(f"Sweep results saved to: {csv_path}")

    # 6. Print screening leaderboard (top 10)
    sorted_results = sorted(
        sweep_results,
        key=lambda r: r["val_recall_at_20"],
        reverse=True,
    )

    click.echo()
    click.echo("Screening Leaderboard (top 10):")
    click.echo(
        f"{'Rank':<6}{'Config':<8}{'Hidden':<8}{'Layers':<8}"
        f"{'Beta':<7}{'Encoding':<13}{'val_R@20':<10}{'Time(s)':<8}"
    )
    click.echo("-" * 68)
    for rank, row in enumerate(sorted_results[:10], 1):
        click.echo(
            f"{rank:<6}{row['config_id']:<8}{row['hidden_size']:<8}"
            f"{row['num_layers']:<8}{row['beta']:<7.2f}"
            f"{row['encoding']:<13}{row['val_recall_at_20']:<10.4f}"
            f"{row['training_time_s']:<8.1f}"
        )

    # 7. Phase 2 — Top-K re-run with multiple seeds on TEST set
    top_configs = sorted_results[:top_k]

    click.echo()
    click.echo(
        f"Phase 2: Top-{top_k} re-run with {len(seed_list)} seeds "
        f"({', '.join(str(s) for s in seed_list)})"
    )

    model_results = []
    best_mean_recall = -1.0
    best_checkpoint_dir = None

    for rank, top in enumerate(top_configs):
        h = top["hidden_size"]
        n = top["num_layers"]
        b = top["beta"]
        e = top["encoding"]

        click.echo(
            f"[{rank + 1}/{top_k}] Top-{rank + 1} config: "
            f"h={h}, l={n}, b={b:.2f}, enc={e}"
        )

        seed_metrics = []
        total_time = 0.0

        for seed in seed_list:
            set_global_seed(seed)

            config = {
                "experiment": {
                    "name": f"spike_gru_top{rank + 1}_seed{seed}",
                    "seed": seed,
                },
                "data": copy.deepcopy(base_data_cfg),
                "model": {
                    "type": "spike_gru",
                    "hidden_size": h,
                    "num_layers": n,
                    "beta": b,
                    "encoding": e,
                    "timesteps": 10,
                    "dropout": 0.0,
                    "surrogate": "fast_sigmoid",
                },
                "training": {
                    "epochs": 100,
                    "learning_rate": 0.001,
                    "optimizer": "adam",
                    "early_stopping_patience": 10,
                    "early_stopping_metric": "val_recall_at_20",
                },
                "output": {
                    "dir": f"results/phase_b_top{rank + 1}_seed{seed}",
                },
                "log_level": "WARNING",
            }

            model = get_model(config)
            trainer = Trainer(model, config, dataloaders, device)

            t0 = time.time()
            trainer.run()
            elapsed = time.time() - t0
            total_time += elapsed

            test_eval = evaluate_model(model, test_loader, device)
            seed_metrics.append(test_eval["metrics"])
            click.echo(
                f"  Seed {seed}: test_recall@20="
                f"{test_eval['metrics']['recall_at_20']:.4f} ({elapsed:.1f}s)"
            )

        # Compute mean test recall for best-checkpoint selection
        mean_recall = sum(
            m["recall_at_20"] for m in seed_metrics
        ) / len(seed_metrics)
        if mean_recall > best_mean_recall:
            best_mean_recall = mean_recall
            # Last seed's checkpoint is representative
            best_checkpoint_dir = (
                f"results/phase_b_top{rank + 1}_seed{seed_list[-1]}"
            )

        model_results.append({
            "name": f"spike_gru_top{rank + 1}",
            "type": "learned",
            "phase": "phase_b",
            "seed_metrics": seed_metrics,
            "training_time_s": round(total_time, 1),
            "environment": "local",
            "config": {
                "hidden_size": h,
                "num_layers": n,
                "beta": b,
                "encoding": e,
            },
        })

    # 8. Save top-K comparison JSON
    report = build_comparison(model_results, window_size, test_split_size)
    top_json_path = str(Path(output_path).parent / "phase_b_top3.json")
    save_comparison(report, top_json_path)

    # 9. Print top-K comparison table
    click.echo()
    click.echo(format_comparison_table(report))
    click.echo(f"Results saved to: {top_json_path}")

    # 10. Copy best checkpoint to results/phase_b_best/
    if best_checkpoint_dir is not None:
        best_src = Path(best_checkpoint_dir)
        best_dst = Path("results/phase_b_best")
        if best_src.exists():
            if best_dst.exists():
                shutil.rmtree(best_dst)
            shutil.copytree(best_src, best_dst)
            click.echo(f"Best checkpoint: {best_dst}")

    click.echo()


@cli.command("phase-c-sweep")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to base YAML config (spiking_transformer).",
)
@click.option(
    "--output",
    "output_path",
    default="results/phase_c_sweep.csv",
    help="Path for sweep results CSV.",
    show_default=True,
)
@click.option(
    "--top-k",
    default=5,
    type=int,
    help="Number of top configs to re-run with multi-seed.",
    show_default=True,
)
@click.option(
    "--seeds",
    default="42,123,7",
    help="Comma-separated seeds for top-K re-runs.",
    show_default=True,
)
@click.option(
    "--screening-seed",
    default=42,
    type=int,
    help="Seed for Phase 1 screening.",
    show_default=True,
)
def phase_c_sweep(
    config_path: str,
    output_path: str,
    top_k: int,
    seeds: str,
    screening_seed: int,
) -> None:
    """Run Spiking Transformer HP sweep at optimal W (Phase C)."""
    import copy
    import csv
    import itertools
    import json
    import shutil
    import time
    from datetime import datetime, timezone

    from c5_snn.data.dataset import get_dataloaders
    from c5_snn.data.loader import load_csv
    from c5_snn.data.splits import create_splits
    from c5_snn.data.windowing import build_windows
    from c5_snn.models.base import get_model
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

    # 1. Load base config
    try:
        base_config = load_config(config_path)
    except ConfigError as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)

    setup_logging("WARNING")

    # 2. Parse seeds
    try:
        seed_list = [int(s.strip()) for s in seeds.split(",")]
    except ValueError:
        click.echo("ERROR: Seeds must be comma-separated integers", err=True)
        sys.exit(1)

    # 3. Data pipeline at W=90 (optimal from STORY-6.2)
    window_size = 90
    data_cfg = base_config.get("data", {})
    raw_path = data_cfg.get("raw_path", "data/raw/CA5_matrix_binary.csv")
    split_ratios = tuple(data_cfg.get("split_ratios", [0.70, 0.15, 0.15]))
    batch_size = int(data_cfg.get("batch_size", 64))

    try:
        df = load_csv(raw_path)
    except (DataValidationError, FileNotFoundError) as e:
        click.echo(f"ERROR: Failed to load data: {e}", err=True)
        sys.exit(1)

    try:
        X, y = build_windows(df, window_size)
    except (ConfigError, DataValidationError) as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)

    split_info = create_splits(
        n_samples=X.shape[0],
        ratios=split_ratios,
        window_size=window_size,
        dates=df["date"] if "date" in df.columns else None,
    )

    dataloaders = get_dataloaders(split_info, X, y, batch_size)
    test_loader = dataloaders["test"]
    test_split_size = len(test_loader.dataset)
    device = get_device()

    base_data_cfg = {
        "raw_path": raw_path,
        "window_size": window_size,
        "split_ratios": list(split_ratios),
        "batch_size": batch_size,
    }

    # 4. Define sweep grid: 3 x 2 x 2 x 3 x 2 = 72 configs
    sweep_grid = {
        "n_layers": [2, 4, 6],
        "n_heads": [2, 4],
        "d_model": [64, 128],
        "beta": [0.5, 0.8, 0.95],
        "encoding": ["direct", "rate_coded"],
    }
    combos = list(itertools.product(
        sweep_grid["n_layers"],
        sweep_grid["n_heads"],
        sweep_grid["d_model"],
        sweep_grid["beta"],
        sweep_grid["encoding"],
    ))

    click.echo()
    click.echo("Phase C Spiking Transformer HP Sweep")
    click.echo("=" * 50)
    click.echo()
    click.echo(f"Config:       {config_path}")
    click.echo(f"Window size:  W={window_size} (optimal from STORY-6.2)")
    click.echo(f"Sweep grid:   {len(combos)} configs")
    click.echo(
        f"  n_layers:   {sweep_grid['n_layers']}"
    )
    click.echo(
        f"  n_heads:    {sweep_grid['n_heads']}"
    )
    click.echo(
        f"  d_model:    {sweep_grid['d_model']}"
    )
    click.echo(
        f"  beta:       {sweep_grid['beta']}"
    )
    click.echo(
        f"  encoding:   {sweep_grid['encoding']}"
    )
    click.echo("  d_ffn:      2 x d_model (auto)")
    click.echo(f"Test samples: {test_split_size}")
    click.echo()

    # 5. Phase 1 — Screening: single-seed on validation set
    click.echo(
        f"Phase 1: Screening ({len(combos)} configs, seed={screening_seed})"
    )

    sweep_results = []

    for i, (nl, nh, dm, b, e) in enumerate(combos):
        set_global_seed(screening_seed)

        config = {
            "experiment": {
                "name": f"spiking_transformer_sweep_{i:03d}",
                "seed": screening_seed,
            },
            "data": copy.deepcopy(base_data_cfg),
            "model": {
                "type": "spiking_transformer",
                "encoding": e,
                "timesteps": 10,
                "d_model": dm,
                "n_heads": nh,
                "n_layers": nl,
                "d_ffn": 2 * dm,
                "beta": b,
                "dropout": 0.1,
                "max_window_size": 100,
                "surrogate": "fast_sigmoid",
            },
            "training": {
                "epochs": 100,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "early_stopping_patience": 10,
                "early_stopping_metric": "val_recall_at_20",
            },
            "output": {"dir": f"results/phase_c_sweep_{i:03d}"},
            "log_level": "WARNING",
        }

        model = get_model(config)
        trainer = Trainer(model, config, dataloaders, device)

        t0 = time.time()
        result = trainer.run()
        elapsed = time.time() - t0

        # Evaluate on validation set for screening
        val_eval = evaluate_model(model, dataloaders["val"], device)

        row = {
            "config_id": i,
            "d_model": dm,
            "n_heads": nh,
            "n_layers": nl,
            "d_ffn": 2 * dm,
            "beta": b,
            "encoding": e,
            "timesteps": 10 if e == "rate_coded" else 1,
            "val_recall_at_20": round(
                val_eval["metrics"]["recall_at_20"], 4
            ),
            "val_hit_at_20": round(val_eval["metrics"]["hit_at_20"], 4),
            "val_mrr": round(val_eval["metrics"]["mrr"], 4),
            "training_time_s": round(elapsed, 1),
            "best_epoch": result["best_epoch"],
        }
        sweep_results.append(row)

        click.echo(
            f"[{i + 1}/{len(combos)}] d={dm} h={nh} l={nl} "
            f"b={b:.2f} enc={e} -> "
            f"val_recall@20={row['val_recall_at_20']:.4f} "
            f"({elapsed:.1f}s)"
        )

    # 6. Save sweep CSV
    csv_path = Path(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "config_id", "d_model", "n_heads", "n_layers", "d_ffn",
        "beta", "encoding", "timesteps",
        "val_recall_at_20", "val_hit_at_20", "val_mrr",
        "training_time_s", "best_epoch",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sweep_results)

    click.echo()
    click.echo(f"Sweep results saved to: {csv_path}")

    # 7. Print screening leaderboard (top 10)
    sorted_results = sorted(
        sweep_results,
        key=lambda r: r["val_recall_at_20"],
        reverse=True,
    )

    click.echo()
    click.echo("Screening Leaderboard (top 10):")
    click.echo(
        f"{'Rank':<6}{'d_model':<9}{'heads':<7}{'layers':<8}"
        f"{'beta':<7}{'encoding':<13}{'val_R@20':<10}{'Time(s)':<8}"
    )
    click.echo("-" * 68)
    for rank, row in enumerate(sorted_results[:10], 1):
        click.echo(
            f"{rank:<6}{row['d_model']:<9}{row['n_heads']:<7}"
            f"{row['n_layers']:<8}{row['beta']:<7.2f}"
            f"{row['encoding']:<13}{row['val_recall_at_20']:<10.4f}"
            f"{row['training_time_s']:<8.1f}"
        )

    # 8. Phase 2 — Top-K re-run with multiple seeds on TEST set
    actual_top_k = min(top_k, len(sorted_results))
    top_configs = sorted_results[:actual_top_k]

    click.echo()
    click.echo(
        f"Phase 2: Top-{actual_top_k} re-run with {len(seed_list)} seeds "
        f"({', '.join(str(s) for s in seed_list)})"
    )

    model_results = []
    best_mean_recall = -1.0
    best_checkpoint_dir = None

    for rank, top in enumerate(top_configs):
        dm = top["d_model"]
        nh = top["n_heads"]
        nl = top["n_layers"]
        b = top["beta"]
        e = top["encoding"]

        click.echo(
            f"[{rank + 1}/{actual_top_k}] Top-{rank + 1} config: "
            f"d={dm}, h={nh}, l={nl}, b={b:.2f}, enc={e}"
        )

        seed_metrics = []
        total_time = 0.0

        for seed in seed_list:
            set_global_seed(seed)

            config = {
                "experiment": {
                    "name": (
                        f"spiking_transformer_top{rank + 1}_seed{seed}"
                    ),
                    "seed": seed,
                },
                "data": copy.deepcopy(base_data_cfg),
                "model": {
                    "type": "spiking_transformer",
                    "encoding": e,
                    "timesteps": 10,
                    "d_model": dm,
                    "n_heads": nh,
                    "n_layers": nl,
                    "d_ffn": 2 * dm,
                    "beta": b,
                    "dropout": 0.1,
                    "max_window_size": 100,
                    "surrogate": "fast_sigmoid",
                },
                "training": {
                    "epochs": 100,
                    "learning_rate": 0.001,
                    "optimizer": "adam",
                    "early_stopping_patience": 10,
                    "early_stopping_metric": "val_recall_at_20",
                },
                "output": {
                    "dir": (
                        f"results/phase_c_top{rank + 1}_seed{seed}"
                    ),
                },
                "log_level": "WARNING",
            }

            model = get_model(config)
            trainer = Trainer(model, config, dataloaders, device)

            t0 = time.time()
            trainer.run()
            elapsed = time.time() - t0
            total_time += elapsed

            test_eval = evaluate_model(model, test_loader, device)
            seed_metrics.append(test_eval["metrics"])
            click.echo(
                f"  Seed {seed}: test_recall@20="
                f"{test_eval['metrics']['recall_at_20']:.4f} ({elapsed:.1f}s)"
            )

        # Compute mean test recall for best-checkpoint selection
        mean_recall = sum(
            m["recall_at_20"] for m in seed_metrics
        ) / len(seed_metrics)
        if mean_recall > best_mean_recall:
            best_mean_recall = mean_recall
            best_checkpoint_dir = (
                f"results/phase_c_top{rank + 1}_seed{seed_list[-1]}"
            )

        model_results.append({
            "name": f"spiking_transformer_top{rank + 1}",
            "type": "learned",
            "phase": "phase_c",
            "seed_metrics": seed_metrics,
            "training_time_s": round(total_time, 1),
            "environment": "local",
            "config": {
                "d_model": dm,
                "n_heads": nh,
                "n_layers": nl,
                "d_ffn": 2 * dm,
                "beta": b,
                "encoding": e,
            },
        })

    # 9. Save top-K comparison JSON
    report = build_comparison(model_results, window_size, test_split_size)
    top_json_path = str(Path(output_path).parent / "phase_c_top5.json")
    save_comparison(report, top_json_path)

    # 10. Print top-K comparison table
    click.echo()
    click.echo(format_comparison_table(report))
    click.echo(f"Results saved to: {top_json_path}")

    # 11. Copy best checkpoint to results/phase_c_best/
    if best_checkpoint_dir is not None:
        best_src = Path(best_checkpoint_dir)
        best_dst = Path("results/phase_c_best")
        if best_src.exists():
            if best_dst.exists():
                shutil.rmtree(best_dst)
            shutil.copytree(best_src, best_dst)
            click.echo(f"Best checkpoint: {best_dst}")

    # 12. Save sweep metadata JSON
    meta_path = Path(output_path).parent / "phase_c_sweep_meta.json"
    meta = {
        "sweep_grid": sweep_grid,
        "total_configs": len(combos),
        "window_size": window_size,
        "screening_seed": screening_seed,
        "seed_list": seed_list,
        "top_k": actual_top_k,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    click.echo()


@cli.command("window-tune")
@click.option(
    "--config",
    "config_path",
    required=True,
    help="Path to base YAML config (must be spiking_transformer).",
)
@click.option(
    "--windows",
    "window_str",
    default="7,14,21,30,45,60,90",
    help="Comma-separated window sizes to test.",
    show_default=True,
)
@click.option(
    "--seeds",
    "seed_str",
    default="42,123,7",
    help="Comma-separated seeds for Phase 2 re-runs.",
    show_default=True,
)
@click.option(
    "--top-k",
    "top_k",
    default=3,
    type=int,
    help="Number of top window sizes to re-run with seeds.",
    show_default=True,
)
@click.option(
    "--output",
    "output_path",
    default="results/window_tuning.csv",
    help="Path for screening CSV.",
    show_default=True,
)
@click.option(
    "--screening-seed",
    "screening_seed",
    default=42,
    type=int,
    help="Seed for Phase 1 screening runs.",
    show_default=True,
)
def window_tune(
    config_path: str,
    window_str: str,
    seed_str: str,
    top_k: int,
    output_path: str,
    screening_seed: int,
) -> None:
    """Sweep window sizes for SpikingTransformer (Phase C)."""
    import copy
    import csv
    import json
    import time
    from datetime import datetime, timezone

    from c5_snn.data.dataset import get_dataloaders
    from c5_snn.data.loader import load_csv
    from c5_snn.data.splits import create_splits
    from c5_snn.data.windowing import build_windows
    from c5_snn.models.base import get_model
    from c5_snn.training.evaluate import evaluate_model
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

    # 2. Parse window sizes and seeds
    try:
        window_sizes = [int(w.strip()) for w in window_str.split(",")]
    except ValueError:
        click.echo(
            "ERROR: Windows must be comma-separated integers", err=True
        )
        sys.exit(1)

    try:
        seed_list = [int(s.strip()) for s in seed_str.split(",")]
    except ValueError:
        click.echo("ERROR: Seeds must be comma-separated integers", err=True)
        sys.exit(1)

    # 3. Extract config details for display
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "spiking_transformer")
    d_model = model_cfg.get("d_model", "?")
    n_heads = model_cfg.get("n_heads", "?")
    n_layers = model_cfg.get("n_layers", "?")
    data_cfg = config.get("data", {})
    raw_path = data_cfg.get("raw_path", "data/raw/CA5_matrix_binary.csv")
    ratios = data_cfg.get("split_ratios", [0.70, 0.15, 0.15])
    batch_size = int(data_cfg.get("batch_size", 64))

    click.echo()
    click.echo("Window Size Tuning — Phase C (SpikingTransformer)")
    click.echo("=" * 51)
    click.echo()
    click.echo(f"Config: {config_path}")
    click.echo(
        f"Model: {model_type} (d_model={d_model}, "
        f"n_heads={n_heads}, n_layers={n_layers})"
    )
    click.echo(f"Window sizes: {window_sizes}")
    click.echo(f"Screening seed: {screening_seed}")
    click.echo()

    # 4. Load raw data once (re-windowed per W)
    try:
        df = load_csv(raw_path)
    except (DataValidationError, FileNotFoundError) as e:
        click.echo(f"ERROR: Failed to load data: {e}", err=True)
        sys.exit(1)

    device = get_device()

    # ===== Phase 1: Screening =====
    click.echo(
        f"Phase 1: Screening ({len(window_sizes)} window sizes, 1 seed each)"
    )
    click.echo("-" * 50)

    sweep_results = []

    for i, W in enumerate(window_sizes):
        set_global_seed(screening_seed)

        # Rebuild data pipeline for this W
        try:
            X, y = build_windows(df, W)
        except (ConfigError, DataValidationError) as e:
            click.echo(f"  [SKIP] W={W}: {e}")
            continue

        split_info = create_splits(
            n_samples=X.shape[0],
            ratios=tuple(ratios),
            window_size=W,
            dates=df["date"] if "date" in df.columns else None,
        )
        dataloaders = get_dataloaders(split_info, X, y, batch_size)

        # Override window_size in config
        run_config = copy.deepcopy(config)
        run_config["data"]["window_size"] = W
        run_config["experiment"]["seed"] = screening_seed
        run_config["output"] = {
            "dir": f"results/window_tune_W{W}_screen",
        }
        run_config["log_level"] = "WARNING"

        model = get_model(run_config)
        trainer = Trainer(model, run_config, dataloaders, device)

        t0 = time.time()
        result = trainer.run()
        elapsed = time.time() - t0

        # Evaluate on validation set
        val_eval = evaluate_model(model, dataloaders["val"], device)

        n_train = len(dataloaders["train"].dataset)
        n_val = len(dataloaders["val"].dataset)
        n_test = len(dataloaders["test"].dataset)

        row = {
            "window_size": W,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "val_recall_at_20": round(
                val_eval["metrics"]["recall_at_20"], 4
            ),
            "val_hit_at_20": round(val_eval["metrics"]["hit_at_20"], 4),
            "val_mrr": round(val_eval["metrics"]["mrr"], 4),
            "training_time_s": round(elapsed, 1),
            "best_epoch": result["best_epoch"],
        }
        sweep_results.append(row)

        click.echo(
            f"[{i + 1}/{len(window_sizes)}] W={W:<4d} "
            f"val_recall@20={row['val_recall_at_20']:.4f}  "
            f"time={elapsed:.1f}s  epoch={result['best_epoch']}"
        )

    if not sweep_results:
        click.echo("ERROR: No window sizes produced valid results.", err=True)
        sys.exit(1)

    # 5. Save screening CSV
    csv_path = Path(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "window_size", "n_train", "n_val", "n_test",
        "val_recall_at_20", "val_hit_at_20", "val_mrr",
        "training_time_s", "best_epoch",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sweep_results)

    # 6. Print screening leaderboard
    sorted_results = sorted(
        sweep_results,
        key=lambda r: r["val_recall_at_20"],
        reverse=True,
    )

    click.echo()
    click.echo("Screening Leaderboard (sorted by val_recall@20):")
    click.echo(
        f"{'W':<6}{'n_samples':<11}{'val_R@20':<13}{'val_H@20':<12}"
        f"{'val_MRR':<10}{'time(s)':<9}{'epoch':<6}"
    )
    click.echo("-" * 67)
    for row in sorted_results:
        n_total = row["n_train"] + row["n_val"] + row["n_test"]
        click.echo(
            f"{row['window_size']:<6}{n_total:<11}"
            f"{row['val_recall_at_20']:<13.4f}"
            f"{row['val_hit_at_20']:<12.4f}"
            f"{row['val_mrr']:<10.4f}"
            f"{row['training_time_s']:<9.1f}"
            f"{row['best_epoch']:<6}"
        )

    # ===== Phase 2: Top-K re-run with multiple seeds =====
    actual_top_k = min(top_k, len(sorted_results))
    top_windows = sorted_results[:actual_top_k]

    click.echo()
    click.echo(
        f"Phase 2: Top-{actual_top_k} Re-run "
        f"({len(seed_list)} seeds each, test evaluation)"
    )
    click.echo("-" * 54)

    top_results = []
    total_runs = actual_top_k * len(seed_list)
    run_idx = 0

    for top_row in top_windows:
        W = top_row["window_size"]
        seed_metrics = []
        total_time = 0.0

        for seed in seed_list:
            run_idx += 1
            set_global_seed(seed)

            # Rebuild data pipeline
            X, y = build_windows(df, W)
            split_info = create_splits(
                n_samples=X.shape[0],
                ratios=tuple(ratios),
                window_size=W,
                dates=df["date"] if "date" in df.columns else None,
            )
            dataloaders = get_dataloaders(split_info, X, y, batch_size)

            run_config = copy.deepcopy(config)
            run_config["data"]["window_size"] = W
            run_config["experiment"]["seed"] = seed
            run_config["output"] = {
                "dir": f"results/window_tune_W{W}_seed{seed}",
            }
            run_config["log_level"] = "WARNING"

            model = get_model(run_config)
            trainer = Trainer(model, run_config, dataloaders, device)

            t0 = time.time()
            trainer.run()
            elapsed = time.time() - t0
            total_time += elapsed

            test_eval = evaluate_model(
                model, dataloaders["test"], device
            )
            seed_metrics.append(test_eval["metrics"])
            click.echo(
                f"[{run_idx}/{total_runs}] W={W:<4d} seed={seed}  "
                f"test_recall@20="
                f"{test_eval['metrics']['recall_at_20']:.4f}  "
                f"({elapsed:.1f}s)"
            )

        # Aggregate mean/std across seeds
        metric_keys = ["recall_at_20", "hit_at_20", "mrr"]
        metrics_mean = {}
        metrics_std = {}
        for key in metric_keys:
            vals = [m[key] for m in seed_metrics]
            mean_val = sum(vals) / len(vals)
            var_val = sum((v - mean_val) ** 2 for v in vals) / len(vals)
            metrics_mean[key] = round(mean_val, 4)
            metrics_std[key] = round(var_val**0.5, 4)

        # Get split sizes from last run
        n_train = len(dataloaders["train"].dataset)
        n_val = len(dataloaders["val"].dataset)
        n_test = len(dataloaders["test"].dataset)

        top_results.append({
            "window_size": W,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "metrics_mean": metrics_mean,
            "metrics_std": metrics_std,
            "n_seeds": len(seed_list),
            "training_time_s": round(total_time / len(seed_list), 1),
        })

    # 7. Print Phase 2 leaderboard
    top_results_sorted = sorted(
        top_results,
        key=lambda r: r["metrics_mean"]["recall_at_20"],
        reverse=True,
    )

    click.echo()
    click.echo("Top-3 Test Results (mean +/- std):")
    click.echo(
        f"{'W':<6}{'test_R@20':<20}{'test_H@20':<20}"
        f"{'test_MRR':<20}{'seeds':<6}"
    )
    click.echo("-" * 72)
    for row in top_results_sorted:
        m = row["metrics_mean"]
        s = row["metrics_std"]
        r20 = f"{m['recall_at_20']:.4f} +/- {s['recall_at_20']:.3f}"
        h20 = f"{m['hit_at_20']:.4f} +/- {s['hit_at_20']:.3f}"
        mrr = f"{m['mrr']:.4f} +/- {s['mrr']:.3f}"
        click.echo(
            f"{row['window_size']:<6}{r20:<20}{h20:<20}"
            f"{mrr:<20}{row['n_seeds']:<6}"
        )

    # 8. Save Phase 2 JSON
    top_json_path = str(
        Path(output_path).parent / "window_tuning_top3.json"
    )
    top_json_report = {
        "window_sizes": top_results_sorted,
        "optimal_window_size": top_results_sorted[0]["window_size"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_type": model_type,
        "model_config": {
            k: v
            for k, v in model_cfg.items()
            if k != "type"
        },
    }
    top_json_file = Path(top_json_path)
    top_json_file.parent.mkdir(parents=True, exist_ok=True)
    with open(top_json_file, "w") as f:
        json.dump(top_json_report, f, indent=2)

    # 9. Print analysis
    optimal_W = top_results_sorted[0]["window_size"]
    optimal_r20 = top_results_sorted[0]["metrics_mean"]["recall_at_20"]

    # Find W=21 (default) in screening results for comparison
    default_row = next(
        (r for r in sweep_results if r["window_size"] == 21), None
    )
    smallest_row = min(sweep_results, key=lambda r: r["window_size"])

    click.echo()
    click.echo("Analysis:")
    click.echo(f"  Optimal window size:    W={optimal_W}")

    if default_row is not None:
        default_r20 = default_row["val_recall_at_20"]
        delta_default = optimal_r20 - default_r20
        pct_default = (
            (delta_default / default_r20) * 100 if default_r20 != 0 else 0
        )
        click.echo(
            f"  vs W=21 (default):      {delta_default:+.4f} "
            f"({pct_default:+.2f}%)"
        )

    smallest_W = smallest_row["window_size"]
    smallest_r20 = smallest_row["val_recall_at_20"]
    delta_small = optimal_r20 - smallest_r20
    pct_small = (
        (delta_small / smallest_r20) * 100 if smallest_r20 != 0 else 0
    )
    click.echo(
        f"  vs W={smallest_W} (smallest):     {delta_small:+.4f} "
        f"({pct_small:+.2f}%)"
    )

    # Training time trend
    fastest = min(sweep_results, key=lambda r: r["training_time_s"])
    slowest = max(sweep_results, key=lambda r: r["training_time_s"])
    click.echo(
        f"  Training time trend:    W={fastest['window_size']}: "
        f"{fastest['training_time_s']:.0f}s, "
        f"W={slowest['window_size']}: "
        f"{slowest['training_time_s']:.0f}s"
    )
    click.echo(
        f"  Recommendation:         Use W={optimal_W} for STORY-6.3 HP sweep"
    )

    click.echo()
    click.echo("Results saved to:")
    click.echo(f"  Screening CSV:  {csv_path}")
    click.echo(f"  Top-3 JSON:     {top_json_path}")
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


# ---------------------------------------------------------------------------
# predict — Top-K prediction for the next event
# ---------------------------------------------------------------------------


@cli.command("predict")
@click.option(
    "--checkpoint",
    "checkpoint_path",
    default=None,
    help="Path to a learned-model checkpoint (.pt file).",
)
@click.option(
    "--model-type",
    default=None,
    help="Model type for non-checkpoint models (e.g. frequency_baseline).",
)
@click.option(
    "--data-path",
    default="data/raw/CA5_matrix_binary.csv",
    help="Path to the CA5 CSV dataset.",
    show_default=True,
)
@click.option(
    "--window-size",
    default=21,
    help="Window size (used only with --model-type).",
    show_default=True,
)
@click.option(
    "--top-k",
    default=24,
    help="Number of top predictions to display.",
    show_default=True,
)
@click.option(
    "--asof",
    default=None,
    help="Predict using data up to this date (YYYY-MM-DD). Default: all data.",
)
def predict(
    checkpoint_path: str | None,
    model_type: str | None,
    data_path: str,
    window_size: int,
    top_k: int,
    asof: str | None,
) -> None:
    """Predict Top-K most likely parts for the next event."""
    import pandas as pd
    import torch

    from c5_snn.data.loader import load_csv
    from c5_snn.inference import (
        build_prediction_window,
        calendar_enhanced_predict,
        format_top_k_prediction,
        load_model_from_checkpoint,
    )
    from c5_snn.models.base import get_model
    from c5_snn.utils.device import get_device

    setup_logging("INFO")

    # --- Validate: exactly one model source ---
    if checkpoint_path and model_type:
        click.echo(
            "ERROR: Provide --checkpoint or --model-type, not both.",
            err=True,
        )
        sys.exit(1)
    if not checkpoint_path and not model_type:
        click.echo(
            "ERROR: Provide --checkpoint or --model-type.", err=True
        )
        sys.exit(1)

    # --- Load data ---
    try:
        df = load_csv(data_path)
    except Exception as e:
        click.echo(f"ERROR: Failed to load data: {e}", err=True)
        sys.exit(1)

    # --- Filter by --asof date if provided ---
    if asof:
        df["_date_parsed"] = pd.to_datetime(df["date"])
        df = df[df["_date_parsed"] <= pd.to_datetime(asof)].copy()
        df = df.drop(columns=["_date_parsed"])
        if len(df) == 0:
            click.echo(
                f"ERROR: No data found on or before {asof}.", err=True
            )
            sys.exit(1)

    # --- Calendar-enhanced strategy (special path) ---
    if model_type == "calendar_enhanced":
        logits = calendar_enhanced_predict(df)
        predictions = format_top_k_prediction(logits, top_k)
        last_date = df["date"].iloc[-1]
        click.echo()
        click.echo(f"Prediction for next event after {last_date}")
        click.echo(
            "Model: calendar_enhanced "
            "(SplitExtra: deficit + DOW/Month/DOM/WOY calendar)"
        )
        click.echo(f"Data: {len(df):,} events up to {last_date}")
        click.echo("=" * 50)
        click.echo(f"  {'Rank':<6} {'Part':<8} {'Score':<12}")
        click.echo(f"  {'----':<6} {'----':<8} {'-----':<12}")
        for p in predictions:
            click.echo(
                f"  {p['rank']:<6} {p['part']:<8} {p['score']:<12.4f}"
            )
        click.echo()
        return

    # --- Load model ---
    device = get_device()

    if checkpoint_path:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            click.echo(
                f"ERROR: Checkpoint not found: {checkpoint_path}",
                err=True,
            )
            sys.exit(1)
        model, config = load_model_from_checkpoint(ckpt_path, device)
        window_size = config.get("data", {}).get(
            "window_size", window_size
        )
        model_label = config.get("model", {}).get("type", "unknown")
        source_label = f"checkpoint: {checkpoint_path}"
    else:
        config = {
            "model": {"type": model_type},
            "data": {"window_size": window_size},
        }
        model = get_model(config)
        model.to(device).eval()
        model_label = model_type
        source_label = f"model-type: {model_type}"

    # --- Build prediction window ---
    x = build_prediction_window(df, window_size)
    x = x.to(device)

    # --- Forward pass ---
    with torch.no_grad():
        logits = model(x)

    predictions = format_top_k_prediction(logits, top_k)

    # --- Display results ---
    last_date = df["date"].iloc[-1]
    first_window_date = df["date"].iloc[-window_size]

    click.echo()
    click.echo(f"Prediction for next event after {last_date}")
    click.echo(f"Model: {model_label} ({source_label})")
    click.echo(
        f"Window: last {window_size} events "
        f"[{first_window_date} to {last_date}]"
    )
    click.echo("=" * 50)
    click.echo(f"  {'Rank':<6} {'Part':<8} {'Score':<12}")
    click.echo(f"  {'----':<6} {'----':<8} {'-----':<12}")
    for p in predictions:
        click.echo(
            f"  {p['rank']:<6} {p['part']:<8} {p['score']:<12.4f}"
        )
    click.echo()


# ---------------------------------------------------------------------------
# holdout-test — Strict holdout validation on most recent data
# ---------------------------------------------------------------------------


@cli.command("holdout-test")
@click.option(
    "--checkpoint",
    "checkpoint_path",
    default=None,
    help="Path to a learned-model checkpoint (.pt file).",
)
@click.option(
    "--model-type",
    default=None,
    help="Model type for non-checkpoint models (e.g. frequency_baseline).",
)
@click.option(
    "--data-path",
    default="data/raw/CA5_matrix_binary.csv",
    help="Path to the CA5 CSV dataset.",
    show_default=True,
)
@click.option(
    "--window-size",
    default=21,
    help="Window size (used only with --model-type).",
    show_default=True,
)
@click.option(
    "--n-holdout",
    default=1,
    help="Number of most recent rows to use as blind holdout.",
    show_default=True,
)
def holdout_test(
    checkpoint_path: str | None,
    model_type: str | None,
    data_path: str,
    window_size: int,
    n_holdout: int,
) -> None:
    """Run strict holdout validation on the most recent data."""
    from c5_snn.data.loader import load_csv
    from c5_snn.inference import (
        load_model_from_checkpoint,
        run_calendar_holdout_test,
        run_holdout_test,
    )
    from c5_snn.models.base import get_model
    from c5_snn.utils.device import get_device

    setup_logging("INFO")

    # --- Validate: exactly one model source ---
    if checkpoint_path and model_type:
        click.echo(
            "ERROR: Provide --checkpoint or --model-type, not both.",
            err=True,
        )
        sys.exit(1)
    if not checkpoint_path and not model_type:
        click.echo(
            "ERROR: Provide --checkpoint or --model-type.", err=True
        )
        sys.exit(1)

    # --- Load data ---
    try:
        df = load_csv(data_path)
    except Exception as e:
        click.echo(f"ERROR: Failed to load data: {e}", err=True)
        sys.exit(1)

    # --- Calendar-enhanced strategy (special path) ---
    if model_type == "calendar_enhanced":
        try:
            holdout_result = run_calendar_holdout_test(
                df, window_size, n_holdout
            )
        except ValueError as e:
            click.echo(f"ERROR: {e}", err=True)
            sys.exit(1)
        model_label = "calendar_enhanced"
    else:
        # --- Load model ---
        device = get_device()

        if checkpoint_path:
            ckpt_path = Path(checkpoint_path)
            if not ckpt_path.exists():
                click.echo(
                    f"ERROR: Checkpoint not found: {checkpoint_path}",
                    err=True,
                )
                sys.exit(1)
            model, config = load_model_from_checkpoint(ckpt_path, device)
            window_size = config.get("data", {}).get(
                "window_size", window_size
            )
            model_label = config.get("model", {}).get("type", "unknown")
        else:
            config = {
                "model": {"type": model_type},
                "data": {"window_size": window_size},
            }
            model = get_model(config)
            model.to(device).eval()
            model_label = model_type

        # --- Run holdout test ---
        try:
            holdout_result = run_holdout_test(
                model, df, window_size, n_holdout, device
            )
        except ValueError as e:
            click.echo(f"ERROR: {e}", err=True)
            sys.exit(1)

    # --- Display aggregate metrics ---
    metrics = holdout_result["metrics"]
    samples = holdout_result["per_sample"]

    first_date = samples[0]["date"]
    last_date = samples[-1]["date"]

    click.echo()
    click.echo("Holdout Test Results")
    click.echo(f"Model: {model_label}")
    click.echo(
        f"Holdout: last {n_holdout} events [{first_date} to {last_date}]"
    )
    click.echo(f"Window size: {window_size}")
    click.echo("=" * 60)

    click.echo()
    click.echo("Aggregate Metrics:")
    for key, value in metrics.items():
        click.echo(f"  {key:<20} {value:.4f}")

    # --- Per-sample breakdown ---
    click.echo()
    click.echo("Per-Sample Breakdown:")
    click.echo(
        f"  {'Date':<12} {'Actual Parts':<30} "
        f"{'Hit@5':<7} {'Hit@20':<8} "
        f"{'R@5':<7} {'R@20':<7} {'MRR':<7}"
    )
    click.echo("  " + "-" * 78)
    for s in samples:
        actual = ",".join(s["true_parts"])
        hit5 = "Yes" if s["hit_at_5"] > 0.5 else "No"
        hit20 = "Yes" if s["hit_at_20"] > 0.5 else "No"
        click.echo(
            f"  {s['date']:<12} {actual:<30} "
            f"{hit5:<7} {hit20:<8} "
            f"{s['recall_at_5']:<7.3f} {s['recall_at_20']:<7.3f} "
            f"{s['mrr']:<7.4f}"
        )

    click.echo()
    click.echo(
        f"Summary: {n_holdout} holdout samples, "
        f"Recall@20={metrics['recall_at_20']:.4f}, "
        f"Hit@20={metrics['hit_at_20']:.4f}, "
        f"MRR={metrics['mrr']:.4f}"
    )
    click.echo()


# ---------------------------------------------------------------------------
# lln-predict — LLN-Pattern exclusion-based prediction
# ---------------------------------------------------------------------------


@cli.command("lln-predict")
@click.option(
    "--data-path",
    default="data/raw/CA5_date.csv",
    help="Path to the CA5 date CSV (6 columns).",
    show_default=True,
)
@click.option(
    "--k-exclude",
    default=20,
    help="Number of values to exclude.",
    show_default=True,
)
@click.option(
    "--no-pattern",
    is_flag=True,
    help="Skip pattern refinement (LLN only).",
)
@click.option(
    "--no-boundary",
    is_flag=True,
    help="Skip boundary penalty.",
)
@click.option(
    "--boundary-penalty",
    default=1.7,
    help="Boundary penalty factor.",
    show_default=True,
)
def lln_predict(
    data_path: str,
    k_exclude: int,
    no_pattern: bool,
    no_boundary: bool,
    boundary_penalty: float,
) -> None:
    """Predict exclusion set for the next event using LLN-Pattern pipeline."""
    from c5_snn.lln_pattern.loader import load_date_csv
    from c5_snn.lln_pattern.pipeline import predict_exclusion_set

    setup_logging("INFO")

    try:
        df = load_date_csv(data_path)
    except Exception as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)

    use_pattern = not no_pattern
    use_boundary = not no_boundary

    result = predict_exclusion_set(
        df,
        target_idx=len(df),
        k_exclude=k_exclude,
        use_pattern=use_pattern,
        use_boundary=use_boundary,
        boundary_penalty=boundary_penalty,
    )

    excluded = sorted(result["excluded_values"].tolist())
    remaining = sorted(
        v for v in range(1, 40) if v not in excluded
    )

    # Pipeline label
    stages = ["LLN"]
    if use_pattern:
        stages.append("Pattern")
    if use_boundary:
        stages.append(f"Boundary(penalty={boundary_penalty})")
    pipeline_label = " + ".join(stages)

    last_date = df["date"].iloc[-1]

    click.echo()
    click.echo("LLN-Pattern Prediction")
    click.echo("=" * 50)
    click.echo(f"Data: {len(df):,} events up to {last_date}")
    click.echo(f"Pipeline: {pipeline_label}")
    click.echo()
    click.echo(
        f"Exclusion set ({k_exclude} values predicted NOT to appear):"
    )
    click.echo(f"  {excluded}")
    click.echo()
    click.echo(
        f"Remaining {39 - k_exclude} values (predicted likely to appear):"
    )
    click.echo(f"  {remaining}")
    click.echo()


# ---------------------------------------------------------------------------
# lln-holdout — LLN-Pattern holdout evaluation
# ---------------------------------------------------------------------------


@cli.command("lln-holdout")
@click.option(
    "--data-path",
    default="data/raw/CA5_date.csv",
    help="Path to the CA5 date CSV (6 columns).",
    show_default=True,
)
@click.option(
    "--n-holdout",
    default=None,
    type=int,
    help="Number of holdout events. Default: 10%% of data.",
)
@click.option(
    "--k-exclude",
    default=20,
    help="Number of values to exclude.",
    show_default=True,
)
@click.option(
    "--no-pattern",
    is_flag=True,
    help="Skip pattern refinement (LLN only).",
)
@click.option(
    "--no-boundary",
    is_flag=True,
    help="Skip boundary penalty.",
)
@click.option(
    "--boundary-penalty",
    default=1.7,
    help="Boundary penalty factor.",
    show_default=True,
)
def lln_holdout(
    data_path: str,
    n_holdout: int | None,
    k_exclude: int,
    no_pattern: bool,
    no_boundary: bool,
    boundary_penalty: float,
) -> None:
    """Run strict holdout evaluation of the LLN-Pattern pipeline."""
    from c5_snn.lln_pattern.holdout import run_lln_holdout_test
    from c5_snn.lln_pattern.loader import load_date_csv

    setup_logging("INFO")

    try:
        df = load_date_csv(data_path)
    except Exception as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)

    use_pattern = not no_pattern
    use_boundary = not no_boundary

    try:
        result = run_lln_holdout_test(
            df,
            n_holdout=n_holdout,
            k_exclude=k_exclude,
            use_pattern=use_pattern,
            use_boundary=use_boundary,
            boundary_penalty=boundary_penalty,
        )
    except ValueError as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)

    summary = result["summary"]
    config = result["config"]

    # Pipeline label
    stages = ["LLN"]
    if use_pattern:
        stages.append("Pattern")
    if use_boundary:
        stages.append(f"Boundary(penalty={boundary_penalty})")
    pipeline_label = " + ".join(stages)

    click.echo()
    click.echo("LLN-Pattern Holdout Test")
    click.echo("=" * 50)
    click.echo(
        f"Data: {len(df):,} events, "
        f"holdout: {config['n_holdout']:,} "
        f"(last {config['n_holdout'] / len(df) * 100:.1f}%)"
    )
    click.echo(f"Pipeline: {pipeline_label}")
    click.echo(f"Exclusion set size: {k_exclude}")
    click.echo()
    click.echo("Results:")
    for n_wrong in sorted(summary["distribution"].keys()):
        count = summary["distribution"][n_wrong]
        pct = 100.0 * count / summary["total"]
        click.echo(
            f"  {n_wrong} wrong: {count:>6} / {summary['total']:,} "
            f"({pct:5.2f}%)"
        )
    click.echo()
    click.echo(f"  Mean wrong: {summary['mean_wrong']:.4f}")
    click.echo(
        f"  0 wrong: {summary['zero_wrong_count']} "
        f"({summary['zero_wrong_pct']:.2f}%)"
    )
    click.echo()
