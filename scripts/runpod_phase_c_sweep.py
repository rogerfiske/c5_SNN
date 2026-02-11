#!/usr/bin/env python3
"""
RunPod Phase C Sweep — Spiking Transformer HP Sweep at W=90

Usage on RunPod B200:
    1. git clone https://github.com/rogerfiske/c5_SNN.git && cd c5_SNN
    2. pip install -e ".[dev]" --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple/
    3. python scripts/runpod_phase_c_sweep.py

Outputs (download these when done):
    - results/phase_c_sweep.csv          (72 rows, Phase 1 screening)
    - results/phase_c_top5.json          (top-5 multi-seed comparison)
    - results/phase_c_best/              (best checkpoint directory)
    - results/phase_c_sweep_meta.json    (sweep metadata)
"""

import copy
import csv
import itertools
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


def main():
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
    from c5_snn.utils.logging import setup_logging
    from c5_snn.utils.seed import set_global_seed

    setup_logging("WARNING")

    # --- Configuration ---
    WINDOW_SIZE = 90
    SCREENING_SEED = 42
    SEED_LIST = [42, 123, 7]
    TOP_K = 5
    OUTPUT_CSV = "results/phase_c_sweep.csv"
    OUTPUT_JSON = "results/phase_c_top5.json"
    OUTPUT_META = "results/phase_c_sweep_meta.json"
    RAW_PATH = "data/raw/CA5_matrix_binary.csv"
    SPLIT_RATIOS = (0.70, 0.15, 0.15)
    BATCH_SIZE = 64

    device = get_device()
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- Data pipeline at W=90 ---
    print(f"\nLoading data and building pipeline at W={WINDOW_SIZE}...")
    df = load_csv(RAW_PATH)
    X, y = build_windows(df, WINDOW_SIZE)

    split_info = create_splits(
        n_samples=X.shape[0],
        ratios=SPLIT_RATIOS,
        window_size=WINDOW_SIZE,
        dates=df["date"] if "date" in df.columns else None,
    )

    dataloaders = get_dataloaders(split_info, X, y, BATCH_SIZE)
    test_loader = dataloaders["test"]
    test_split_size = len(test_loader.dataset)

    base_data_cfg = {
        "raw_path": RAW_PATH,
        "window_size": WINDOW_SIZE,
        "split_ratios": list(SPLIT_RATIOS),
        "batch_size": BATCH_SIZE,
    }

    print(f"Train: {len(dataloaders['train'].dataset)}, "
          f"Val: {len(dataloaders['val'].dataset)}, "
          f"Test: {test_split_size}")

    # --- Sweep grid: 3 x 2 x 2 x 3 x 2 = 72 configs ---
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

    print("\nPhase C Spiking Transformer HP Sweep")
    print("=" * 50)
    print(f"Window size:  W={WINDOW_SIZE}")
    print(f"Sweep grid:   {len(combos)} configs")
    print(f"  n_layers:   {sweep_grid['n_layers']}")
    print(f"  n_heads:    {sweep_grid['n_heads']}")
    print(f"  d_model:    {sweep_grid['d_model']}")
    print(f"  beta:       {sweep_grid['beta']}")
    print(f"  encoding:   {sweep_grid['encoding']}")
    print("  d_ffn:      2 x d_model (auto)")

    # ==============================
    # Phase 1: Screening (single seed)
    # ==============================
    print(f"\nPhase 1: Screening ({len(combos)} configs, seed={SCREENING_SEED})")
    print("-" * 50)

    sweep_results = []
    phase1_start = time.time()

    for i, (nl, nh, dm, b, e) in enumerate(combos):
        set_global_seed(SCREENING_SEED)

        config = {
            "experiment": {
                "name": f"spiking_transformer_sweep_{i:03d}",
                "seed": SCREENING_SEED,
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
            "val_recall_at_20": round(val_eval["metrics"]["recall_at_20"], 4),
            "val_hit_at_20": round(val_eval["metrics"]["hit_at_20"], 4),
            "val_mrr": round(val_eval["metrics"]["mrr"], 4),
            "training_time_s": round(elapsed, 1),
            "best_epoch": result["best_epoch"],
        }
        sweep_results.append(row)

        print(
            f"[{i + 1}/{len(combos)}] d={dm} h={nh} l={nl} "
            f"b={b:.2f} enc={e} -> "
            f"val_recall@20={row['val_recall_at_20']:.4f} "
            f"({elapsed:.1f}s)"
        )

    phase1_elapsed = time.time() - phase1_start
    print(f"\nPhase 1 complete in {phase1_elapsed:.0f}s "
          f"({phase1_elapsed / 60:.1f} min)")

    # Save sweep CSV
    csv_path = Path(OUTPUT_CSV)
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

    print(f"Sweep results saved to: {csv_path}")

    # Print screening leaderboard (top 10)
    sorted_results = sorted(
        sweep_results,
        key=lambda r: r["val_recall_at_20"],
        reverse=True,
    )

    print("\nScreening Leaderboard (top 10):")
    print(
        f"{'Rank':<6}{'d_model':<9}{'heads':<7}{'layers':<8}"
        f"{'beta':<7}{'encoding':<13}{'val_R@20':<10}{'Time(s)':<8}"
    )
    print("-" * 68)
    for rank, row in enumerate(sorted_results[:10], 1):
        print(
            f"{rank:<6}{row['d_model']:<9}{row['n_heads']:<7}"
            f"{row['n_layers']:<8}{row['beta']:<7.2f}"
            f"{row['encoding']:<13}{row['val_recall_at_20']:<10.4f}"
            f"{row['training_time_s']:<8.1f}"
        )

    # ==============================
    # Phase 2: Top-K multi-seed on TEST set
    # ==============================
    actual_top_k = min(TOP_K, len(sorted_results))
    top_configs = sorted_results[:actual_top_k]

    print(f"\nPhase 2: Top-{actual_top_k} re-run with {len(SEED_LIST)} seeds "
          f"({', '.join(str(s) for s in SEED_LIST)})")
    print("-" * 50)

    model_results = []
    best_mean_recall = -1.0
    best_checkpoint_dir = None
    phase2_start = time.time()

    for rank, top in enumerate(top_configs):
        dm = top["d_model"]
        nh = top["n_heads"]
        nl = top["n_layers"]
        b = top["beta"]
        e = top["encoding"]

        print(
            f"[{rank + 1}/{actual_top_k}] Top-{rank + 1} config: "
            f"d={dm}, h={nh}, l={nl}, b={b:.2f}, enc={e}"
        )

        seed_metrics = []
        total_time = 0.0

        for seed in SEED_LIST:
            set_global_seed(seed)

            config = {
                "experiment": {
                    "name": f"spiking_transformer_top{rank + 1}_seed{seed}",
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
                    "dir": f"results/phase_c_top{rank + 1}_seed{seed}",
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
            print(
                f"  Seed {seed}: test_recall@20="
                f"{test_eval['metrics']['recall_at_20']:.4f} ({elapsed:.1f}s)"
            )

        mean_recall = sum(
            m["recall_at_20"] for m in seed_metrics
        ) / len(seed_metrics)
        if mean_recall > best_mean_recall:
            best_mean_recall = mean_recall
            best_checkpoint_dir = (
                f"results/phase_c_top{rank + 1}_seed{SEED_LIST[-1]}"
            )

        model_results.append({
            "name": f"spiking_transformer_top{rank + 1}",
            "type": "learned",
            "phase": "phase_c",
            "seed_metrics": seed_metrics,
            "training_time_s": round(total_time, 1),
            "environment": "runpod_b200",
            "config": {
                "d_model": dm,
                "n_heads": nh,
                "n_layers": nl,
                "d_ffn": 2 * dm,
                "beta": b,
                "encoding": e,
            },
        })

    phase2_elapsed = time.time() - phase2_start
    print(f"\nPhase 2 complete in {phase2_elapsed:.0f}s "
          f"({phase2_elapsed / 60:.1f} min)")

    # Save top-K comparison JSON
    report = build_comparison(model_results, WINDOW_SIZE, test_split_size)
    save_comparison(report, OUTPUT_JSON)

    # Print comparison table
    print()
    print(format_comparison_table(report))
    print(f"Results saved to: {OUTPUT_JSON}")

    # Copy best checkpoint
    if best_checkpoint_dir is not None:
        best_src = Path(best_checkpoint_dir)
        best_dst = Path("results/phase_c_best")
        if best_src.exists():
            if best_dst.exists():
                shutil.rmtree(best_dst)
            shutil.copytree(best_src, best_dst)
            print(f"Best checkpoint: {best_dst}")

    # Save sweep metadata
    meta = {
        "sweep_grid": sweep_grid,
        "total_configs": len(combos),
        "window_size": WINDOW_SIZE,
        "screening_seed": SCREENING_SEED,
        "seed_list": SEED_LIST,
        "top_k": actual_top_k,
        "phase1_time_s": round(phase1_elapsed, 1),
        "phase2_time_s": round(phase2_elapsed, 1),
        "total_time_s": round(phase1_elapsed + phase2_elapsed, 1),
        "device": str(device),
        "gpu": (torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else "cpu"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    Path(OUTPUT_META).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_META, "w") as f:
        json.dump(meta, f, indent=2)

    total_elapsed = phase1_elapsed + phase2_elapsed
    print(f"\n{'=' * 50}")
    print(f"Total sweep time: {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min)")
    print("\nFiles to download:")
    print(f"  {OUTPUT_CSV}")
    print(f"  {OUTPUT_JSON}")
    print(f"  {OUTPUT_META}")
    print("  results/phase_c_best/")


if __name__ == "__main__":
    main()
