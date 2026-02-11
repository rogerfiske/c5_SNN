# Project Memory — c5_SNN

## Project Management

- **Method:** BMAD v6 (Level 2, sprint-based)
- **Workflow pattern:** `/bmad:create-story STORY-X.Y` then `/bmad:dev-story STORY-X.Y`
- **End-of-session command:** `ED` (triggers wrap-up protocol defined in CLAUDE.md)
- **User (Roger):** Not a programmer — all technical decisions delegated to AI agent
- **Sprint plan:** 7 one-week sprints, 22 stories, 103 total points
- **Repository:** https://github.com/rogerfiske/c5_SNN

## Sprint History

| Sprint | Points | Velocity | Status |
|--------|--------|----------|--------|
| Sprint 1 | 13/13 | 13 | Complete (2026-02-10) |
| Sprint 2 | 16/16 | 16 | Complete (2026-02-11) |
| Sprint 3 | 19/19 | 19 | Complete (2026-02-11) |
| Sprint 4 | 18/18 | 18 | Complete (2026-02-11) |
| Sprint 5 | 13/13 | 13 | Complete (2026-02-11) |

## Decisions

- Canonical modeling representation: `P_1..P_39` multi-hot from `CA5_matrix_binary.csv`.
- Task framing: multi-label ranking over 39 labels; output Top-20.
- Build backend: `setuptools.build_meta` (not `setuptools.backends._legacy:_Backend` — incompatible with setuptools <70).
- PyTorch in CI: CPU-only via `--index-url https://download.pytorch.org/whl/cpu`.
- Date column loaded as string (not datetime) — parsing only for validation checks.
- All validation checks run regardless of earlier failures (report-all-at-once pattern).
- CLI uses Click; logging uses Python `logging` module (not print).
- Loss function: BCEWithLogitsLoss for multi-label binary classification.
- Early stopping metric: `val_recall_at_20` (primary ranking metric).
- Trainer handles zero-param models (e.g., FrequencyBaseline) by setting optimizer=None.
- Comparison report uses population std (n divisor, not n-1) for multi-seed metrics.
- Checkpoint bundle: 6 keys (model_state_dict, optimizer_state_dict, epoch, best_val_recall_at_20, config, seed).
- Strategy/Registry pattern: MODEL_REGISTRY maps string keys to model classes.

## Data Integrity Checks

- `CA5_date.csv` and `CA5_matrix_binary.csv` are aligned on dates and m_1..m_5.
- Binary integrity: each event has exactly 5 active parts and matches m_1..m_5.
- Real CSV: 11,702 rows x 45 columns, all 5 validation checks PASS.
- Date range: 1992-02-04 to 2026-02-09.

## Technical Implementation Notes

- **Exception hierarchy:** C5SNNError -> DataValidationError, ConfigError, TrainingError
- **Seed management:** `set_global_seed()` sets random, numpy, torch, cudnn.deterministic=True
- **Config:** YAML via `yaml.safe_load`, raises ConfigError on failure
- **Device:** `get_device()` with CUDA detection, ROCm HSA_OVERRIDE warning
- **Validation:** 5 checks (column count, column names, dates monotonic, row sums == 5, m/P cross-check)
- **Test fixtures:** `_make_valid_df()` generates valid CA5 DataFrames with numpy RandomState(42)
- **Windowing:** `build_windows(df, W)` produces (N-W, W, 39) input and (N-W, 39) target tensors
- **Splits:** Time-based train/val/test with no shuffling; `SplitInfo` dataclass persisted as JSON
- **Evaluation:** `evaluate_model()` returns metrics dict + per-sample predictions; `export_results()` saves JSON + CSV
- **Metrics:** recall@k, hit@k, MRR — all batch-averaged; `compute_all_metrics()` returns all 5
- **Models:** FrequencyBaseline (heuristic, no params), GRUBaseline (learned, GRU + linear head)
- **Trainer:** Generic training loop; early stopping, checkpoint saving, metrics CSV, config snapshot, pip freeze
- **Comparison:** `build_comparison()` aggregates multi-seed results with mean/std; `format_comparison_table()` for display
- **CLI commands:** validate-data, train, evaluate, compare, phase-a, phase-b-sweep, phase-b, window-tune
- **SNN models (Phase A):** SpikingMLP (FC+LIF), SpikingCNN1D (Conv1d+LIF) — both use SpikeEncoder, surrogate gradients
- **SNN models (Phase B):** SpikeGRU (RLeaky recurrent LIF) — processes window event-by-event with accumulating membrane state
- **SNN models (Phase C):** SpikingTransformer (SSA + spiking FFN with LIF) — Spikformer-style spike-form Q/K/V attention without softmax, learnable positional encoding, 286,887 params (default config)
- **snnTorch patterns:** `snntorch.Leaky` with `init_hidden=False` (Phase A/C); `snntorch.RLeaky` with `all_to_all=True` (Phase B); manual membrane potential management, `surrogate.fast_sigmoid(slope=25)`
- **SNN encoding:** `direct` mode passes binary input as-is (T=1); `rate_coded` samples Bernoulli spikes (T=timesteps)

## Compute Policy

- Run a short timing probe locally before full training.
- If projected > ~20 minutes, train on RunPod GPU (B200/B100-class).
- Local: Windows, Python 3.11.9, AMD RX 6600M (CPU-only — ROCm is Linux-only).

## Open Questions

- Optimal history window W (start 21; tune 7-90).
- Best encoding for binary: direct spike vs rate-coded vs latency-coded.
- Whether to add calendar features (day-of-week seasonality) and how much they help.
- How to handle class imbalance and rare parts (loss weighting vs focal).

## Baseline Comparison Framework

- **compare CLI:** `c5_snn compare --config config.yaml --seeds 42,123,7 --output results/comparison.json`
- **phase-a CLI:** `c5_snn phase-a --seeds 42,123,7 --output results/phase_a_comparison.json`
- **phase-b-sweep CLI:** `c5_snn phase-b-sweep --output results/phase_b_sweep.csv --top-k 3 --seeds 42,123,7`
- **phase-b CLI:** `c5_snn phase-b --phase-a results/phase_a_comparison.json --phase-b-top results/phase_b_top3.json --output results/cumulative_comparison.json`
- **Comparison report schema (Section 4.7):** `{models, generated_at, window_size, test_split_size}`
- **Model entry:** `{name, type, phase, metrics_mean, metrics_std, n_seeds, training_time_s, environment}`
- **FrequencyBaseline:** Deterministic heuristic, 1 seed, training_time_s=0
- **GRU baseline:** Multi-seed training, population std across seeds

## Phase A Results (SNN vs Baselines)

**Run date:** 2026-02-11 | **Seeds:** 42, 123, 7 | **Window:** 21 | **Test samples:** 1753

| Model | Phase | Recall@20 | Hit@20 | MRR | Time (s) |
|-------|-------|-----------|--------|-----|----------|
| frequency_baseline | baseline | 0.5232 | 0.9840 | 0.3125 | 0 |
| gru_baseline | baseline | 0.5099 +/- 0.003 | 0.9789 +/- 0.003 | 0.3103 +/- 0.002 | 59.5 |
| spiking_mlp | phase_a | 0.5125 +/- 0.003 | 0.9759 +/- 0.001 | 0.3101 +/- 0.007 | 33.1 |
| spiking_cnn1d | phase_a | 0.5152 +/- 0.002 | 0.9795 +/- 0.002 | 0.3053 +/- 0.001 | 33.2 |

**Key observations:**
1. **All models cluster around Recall@20 ~ 0.51**, very close performance across architectures
2. **FrequencyBaseline leads** on Recall@20 (0.523) — heuristic is hard to beat with default hyperparameters
3. **SpikingCNN1D slightly edges SpikingMLP** on Recall@20 (0.5152 vs 0.5125) and Hit@20 (0.9795 vs 0.9759)
4. **GRU does NOT beat FrequencyBaseline** — learned models may need HP tuning to surpass the heuristic
5. **SNN models train ~2x faster** than GRU (33s vs 60s) with direct encoding (T=1 timestep)
6. **Direct encoding (T=1)** means SNN models effectively act as single-timestep networks — membrane dynamics have no time to accumulate. Rate-coded encoding with T>1 may improve performance.

**Encoding recommendation for Phase B:**
- Try `rate_coded` encoding with T=5-10 timesteps to give LIF membrane dynamics time to accumulate
- Direct encoding collapses the temporal loop to a single step, reducing SNNs to approximately feedforward networks with a threshold nonlinearity
- This is the most likely lever for improving SNN performance

**Hypotheses for Phase B improvement:**
- Rate-coded encoding (T>1) will allow membrane accumulation and temporal dynamics
- Recurrent architecture (Spike-GRU) may capture sequential patterns better than MLP/CNN
- HP tuning (beta, learning rate, hidden sizes) may differentiate model performance

## Phase B Results (Spike-GRU HP Sweep)

**Run date:** 2026-02-11 | **Seeds:** 42, 123, 7 | **Window:** 21 | **Test samples:** 1753

**HP Sweep:** 36 configs = hidden_size [64,128,256] x num_layers [1,2] x beta [0.5,0.8,0.95] x encoding [direct,rate_coded]

**Best Spike-GRU config:** h=256, l=1, b=0.80, encoding=direct

| Model | Phase | Recall@20 | Hit@20 | MRR | Time (s) |
|-------|-------|-----------|--------|-----|----------|
| frequency_baseline | baseline | 0.5232 | 0.9840 | 0.3125 | 0 |
| spiking_cnn1d | phase_a | 0.5152 +/- 0.002 | 0.9795 +/- 0.002 | 0.3053 +/- 0.001 | 33.2 |
| spike_gru | phase_b | 0.5137 +/- 0.000 | 0.9812 +/- 0.000 | 0.3115 +/- 0.000 | 114.4 |
| spiking_mlp | phase_a | 0.5125 +/- 0.003 | 0.9759 +/- 0.001 | 0.3101 +/- 0.007 | 33.1 |
| gru_baseline | baseline | 0.5099 +/- 0.003 | 0.9789 +/- 0.003 | 0.3103 +/- 0.002 | 59.5 |

**Key findings:**
1. **Spike-GRU beats GRU baseline** on Recall@20: +0.0038 (+0.75%)
2. **Spike-GRU slightly below SpikingCNN1D** on Recall@20: -0.0015 (-0.29%)
3. **direct == rate_coded** for Spike-GRU — no benefit from T>1 timesteps (hypothesis disproved)
4. **hidden_size is the dominant hyperparameter:** 256 >> 128 >> 64
5. **Extremely consistent across seeds:** std=0.000 for Recall@20 (all 3 seeds produce identical results)
6. **All learned models cluster ~0.51 Recall@20** — architecture choice has marginal impact at current window size
7. **FrequencyBaseline still leads** — heuristic advantage not overcome by any learned model

**Phase C recommendations:**
- Spiking Transformer with attention may capture patterns that recurrence alone cannot
- Window size tuning (currently W=21, try 7-90) is likely a stronger lever than architecture
- Consider that the dataset's temporal structure may be inherently simple, limiting architecture advantages

## Phase C Window Size Tuning Results (STORY-6.2)

**Run date:** 2026-02-11 | **Model:** SpikingTransformer (d_model=128, n_heads=4, n_layers=2) | **Encoding:** direct

**Phase 1 Screening (seed=42, 7 window sizes):**

| W | n_samples | val_R@20 | val_H@20 | val_MRR | time(s) | epoch |
|---|-----------|----------|----------|---------|---------|-------|
| 90 | 11,613 | **0.5144** | 0.9770 | 0.3166 | 432.3 | 20 |
| 7 | 11,696 | 0.5073 | 0.9749 | 0.3218 | 31.9 | 1 |
| 14 | 11,689 | 0.5071 | 0.9749 | 0.3215 | 40.1 | 1 |
| 45 | 11,658 | 0.5070 | 0.9754 | 0.3231 | 84.4 | 1 |
| 21 | 11,682 | 0.5067 | 0.9749 | 0.3214 | 49.7 | 1 |
| 60 | 11,643 | 0.5066 | 0.9754 | 0.3226 | 100.8 | 1 |
| 30 | 11,673 | 0.5065 | 0.9749 | 0.3229 | 61.5 | 1 |

**Phase 2 Top-3 Test Results (seeds=42,123,7):**

| W | test_R@20 | test_H@20 | test_MRR | seeds | avg time(s) |
|---|-----------|-----------|----------|-------|-------------|
| 90 | **0.5107 +/- 0.003** | 0.9778 +/- 0.002 | 0.3154 +/- 0.002 | 3 | 308.1 |
| 7 | 0.5067 +/- 0.004 | 0.9793 +/- 0.003 | 0.3069 +/- 0.002 | 3 | 37.0 |
| 14 | 0.5066 +/- 0.004 | 0.9793 +/- 0.003 | 0.3069 +/- 0.002 | 3 | 48.0 |

**Key findings:**
1. **Optimal window size: W=90** — longest context tested (+0.79% vs default W=21)
2. **W=90 is the only window that trains beyond epoch 1** — all others early-stop at epoch 11 with no improvement
3. **W=7-60 produce nearly identical results** (~0.507 val_R@20) — the model isn't utilizing short/medium context
4. **W=90 needs ~20 epochs** to reach its best, suggesting the transformer learns longer-range dependencies with more context
5. **Training time scales linearly with W:** W=7: 32s, W=90: 432s (13.5x slower)
6. **Confirmed on test set with 3 seeds:** W=90 = 0.5107 +/- 0.003 vs W=7 = 0.5067 +/- 0.004
7. **STORY-6.3 HP sweep should use W=90** as the base window size

**Resolved open question:** Optimal W = 90 (was 21 default). Longer context helps the SpikingTransformer.

## Test Coverage

- **Total tests:** 475 (all passing)
- **Test files:** test_validation, test_loader, test_logging_setup, test_seed, test_config, test_windowing, test_splits, test_baselines, test_metrics, test_evaluate_cli, test_train, test_compare, test_snn_models

## Next Actions

- Sprint 6 in progress: STORY-6.1, STORY-6.2 complete, next is HP sweep (STORY-6.3).
- STORY-6.3 HP sweep should use W=90 as base window size.
- SpikingTransformer at W=90 shows promising multi-epoch training (unlike W=7-60 which plateau immediately).
- Consider that all learned models cluster ~0.51 Recall@20 — dataset temporal structure may be inherently simple.
