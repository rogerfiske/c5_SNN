# Start Here — 2026-02-12

## Project Status: COMPLETE

The c5_SNN project is finished. All 103/103 story points across 7 sprints have been completed. No further stories or sprints are planned.

## What Was Done

Six model architectures were implemented, trained, and compared for CA5 event prediction:

| Rank | Model | Recall@20 |
|------|-------|-----------|
| 1 | Frequency Baseline | 0.5232 |
| 2 | Spiking Transformer | 0.5178 |
| 3 | Spiking CNN-1D | 0.5152 |
| 4 | Spike-GRU | 0.5137 |
| 5 | Spiking MLP | 0.5125 |
| 6 | GRU Baseline | 0.5099 |

The frequency heuristic remains the top performer. The Spiking Transformer is the best learned model.

## Key Files

- `results/final_report.md` — Full publishable analysis
- `results/final_comparison.json` — Machine-readable leaderboard
- `README.md` — Updated with reproduction guide
- `docs/project-memory.md` — All decisions and findings

## If You Want to Extend This Project

### Add a new model
1. Implement model class in `src/c5_snn/models/`
2. Register in `src/c5_snn/models/__init__.py` model registry
3. Create config in `configs/`
4. Train: `c5-snn train --config configs/your_model.yaml`
5. Evaluate: `c5-snn evaluate --checkpoint results/your_model/best_model.pt`

### Re-run the full pipeline
See the "Reproducing the Full Pipeline" section in `README.md` for the complete command sequence.

### Quick verification
```bash
c5-snn validate-data          # Verify dataset
ruff check src/ tests/        # Lint
pytest tests/ -v              # Run 491 tests
```
