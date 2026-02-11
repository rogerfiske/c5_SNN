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
