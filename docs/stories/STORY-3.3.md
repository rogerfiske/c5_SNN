# STORY-3.3: Training Loop & train CLI

**Epic:** Epic 3 — Baseline Models
**Priority:** Must Have
**Story Points:** 8
**Status:** Completed
**Assigned To:** ai_dev_agent
**Created:** 2026-02-11
**Sprint:** 3

---

## User Story

As a researcher,
I want to run `python -m c5_snn.cli train --config configs/baseline_gru.yaml` and get a trained model,
so that I can train any model with a single command.

---

## Description

### Background

With both baseline models implemented (FrequencyBaseline from STORY-3.1, GRUBaseline from STORY-3.2) and the evaluation harness ready (STORY-2.3/2.4), the project needs its core training infrastructure. This is the highest-value infrastructure story in the project — it unlocks all model training for Sprints 3 through 6.

The training loop must be generic: it accepts any `BaseModel` via the registry, trains it with `BCEWithLogitsLoss`, performs validation after each epoch, implements early stopping on `val_recall_at_20`, and saves the best checkpoint with a complete experiment snapshot (config, pip freeze, per-epoch metrics CSV).

This story also adds the `train` CLI subcommand, which loads a YAML config, sets up the full pipeline (seed, data, model, trainer), and produces all output artifacts in a single command.

### Scope

**In scope:**
- `Trainer` class in `src/c5_snn/training/trainer.py`
- Training loop: forward -> loss -> backward -> optimizer step
- Validation after each epoch using existing metrics (Recall@20, Hit@20)
- Early stopping on `val_recall_at_20` with configurable patience
- Best-model checkpoint saving (Section 4.5 format)
- Per-epoch metrics CSV logging
- Config snapshot saved alongside checkpoint
- `pip freeze` captured to output dir
- 2-epoch timing probe with RunPod warning
- `train` CLI subcommand accepting `--config`
- Integration test: train GRU 2 epochs on tiny data

**Out of scope:**
- Distributed training / multi-GPU (not needed for this project scale)
- Learning rate schedulers (can be added later if needed)
- Cardinality penalty (optional FR8 — stub but don't implement complex logic)
- TensorBoard or W&B integration (file-based logging only)
- Hyperparameter sweep (STORY-5.2)

### User Flow

1. User creates or edits a YAML config file (e.g., `configs/baseline_gru.yaml`)
2. User runs `python -m c5_snn.cli train --config configs/baseline_gru.yaml`
3. System loads config, sets global seed, loads data, creates windowed tensors and splits
4. System instantiates model via `get_model(config)` from registry
5. System creates `Trainer(model, config, dataloaders, device)` and calls `trainer.run()`
6. Each epoch: train on train split -> compute val metrics -> log to CSV + console
7. After epoch 2: timing probe logs RunPod warning if projected >20 min
8. Early stopping halts training if `val_recall_at_20` doesn't improve for `patience` epochs
9. Best checkpoint saved when `val_recall_at_20` improves
10. On completion: output dir contains `best_model.pt`, `config_snapshot.yaml`, `metrics.csv`, `pip_freeze.txt`
11. User sees final metrics summary in console

---

## Acceptance Criteria

- [ ] `Trainer` class in `src/c5_snn/training/trainer.py` accepts any `BaseModel`, config dict, dataloaders dict, and device
- [ ] Training loop performs: forward pass -> `BCEWithLogitsLoss` -> backward -> optimizer step (Adam)
- [ ] Validation runs after each epoch computing `recall_at_20` and `hit_at_20` on val split
- [ ] Early stopping on `val_recall_at_20` with configurable patience (default 10); training halts when patience exhausted
- [ ] Best-model checkpoint saved as `best_model.pt` when `val_recall_at_20` improves, containing keys: `model_state_dict`, `optimizer_state_dict`, `epoch`, `best_val_recall_at_20`, `config`, `seed`
- [ ] Per-epoch metrics logged to `metrics.csv` with columns: `epoch`, `train_loss`, `val_recall_at_20`, `val_hit_at_20`
- [ ] Config snapshot saved as `config_snapshot.yaml` at start of training (immutable — never mutated)
- [ ] `pip freeze` output saved as `pip_freeze.txt` in experiment output dir
- [ ] 2-epoch timing probe: after epoch 2, if projected total time > 20 min, log a WARNING recommending RunPod
- [ ] `train` CLI subcommand added: `python -m c5_snn.cli train --config <path>`
- [ ] `train` CLI loads config, sets seed, loads/processes data, instantiates model, runs training, and saves all artifacts
- [ ] `train` CLI prints final metrics summary on completion
- [ ] `train` CLI exits non-zero with clear error message for: missing config, invalid config, missing data files
- [ ] Integration test: train GRU baseline 2 epochs on ~100 rows, verify `best_model.pt`, `config_snapshot.yaml`, `metrics.csv`, and `pip_freeze.txt` all exist with correct structure
- [ ] All training uses `set_global_seed(config.seed)` before any stochastic operation
- [ ] Uses logging from STORY-1.2 (no `print()` statements in trainer; CLI uses `click.echo`)

---

## Technical Notes

### Components

- **New file:** `src/c5_snn/training/trainer.py` — `Trainer` class
- **Modified file:** `src/c5_snn/training/__init__.py` — add `Trainer` export
- **Modified file:** `src/c5_snn/cli.py` — add `train` subcommand
- **New test file:** `tests/test_train.py` — trainer unit tests + integration test
- **Existing deps:** `training/metrics.py`, `training/evaluate.py`, `models/base.py`, `data/splits.py`

### Trainer Class Design

```python
class Trainer:
    """Generic training loop for any BaseModel."""

    def __init__(
        self,
        model: BaseModel,
        config: dict,
        dataloaders: dict[str, DataLoader],
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.config = config

        train_cfg = config.get("training", {})
        self.epochs = int(train_cfg.get("epochs", 100))
        self.lr = float(train_cfg.get("learning_rate", 0.001))
        self.patience = int(train_cfg.get("early_stopping_patience", 10))

        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.BCEWithLogitsLoss()

        self.train_loader = dataloaders["train"]
        self.val_loader = dataloaders["val"]

        self.output_dir = config.get("output", {}).get("dir", "results/default")

    def run(self) -> dict:
        """Execute full training loop. Returns best metrics."""
        ...
```

### Training Loop Pseudocode

```
save config_snapshot.yaml
save pip_freeze.txt

best_val_recall = -1
patience_counter = 0
epoch_metrics = []

for epoch in range(epochs):
    # --- Train step ---
    model.train()
    epoch_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_x.size(0)
    avg_loss = epoch_loss / len(train_loader.dataset)

    # --- Validation step ---
    model.eval()
    val_logits, val_targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            val_logits.append(logits.cpu())
            val_targets.append(batch_y)
    val_logits = torch.cat(val_logits)
    val_targets = torch.cat(val_targets)
    val_metrics = compute_all_metrics(val_logits, val_targets)

    # --- Log ---
    epoch_metrics.append({
        "epoch": epoch + 1,
        "train_loss": avg_loss,
        "val_recall_at_20": val_metrics["recall_at_20"],
        "val_hit_at_20": val_metrics["hit_at_20"],
    })
    logger.info("Epoch %d/%d — loss=%.4f val_recall@20=%.4f",
                epoch+1, epochs, avg_loss, val_metrics["recall_at_20"])

    # --- 2-epoch timing probe ---
    if epoch == 1:
        elapsed = time.time() - start_time
        projected = elapsed / 2 * epochs
        if projected > 20 * 60:
            logger.warning("Projected training time: %.0f min. Consider RunPod.", projected / 60)

    # --- Early stopping + checkpoint ---
    if val_metrics["recall_at_20"] > best_val_recall:
        best_val_recall = val_metrics["recall_at_20"]
        patience_counter = 0
        save_checkpoint(...)
    else:
        patience_counter += 1
        if patience_counter >= patience:
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

save metrics.csv from epoch_metrics
return best metrics
```

### Checkpoint Bundle (Section 4.5)

```python
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": best_epoch,
    "best_val_recall_at_20": best_val_recall,
    "config": config,
    "seed": config.get("experiment", {}).get("seed", 42),
}
torch.save(checkpoint, output_dir / "best_model.pt")
```

### Output Directory Structure

```
results/{experiment_name}/
├── best_model.pt           # torch.save checkpoint dict
├── config_snapshot.yaml    # Frozen config (saved at start)
├── metrics.csv             # Per-epoch: epoch,train_loss,val_recall_at_20,val_hit_at_20
├── pip_freeze.txt          # pip freeze output
└── evaluation/             # Populated later by evaluate command
```

### train CLI Command

```python
@cli.command("train")
@click.option("--config", "config_path", required=True,
              help="Path to experiment config YAML.")
def train(config_path: str) -> None:
    """Train a model from a YAML experiment config."""
    # 1. Load and validate config
    # 2. Set global seed
    # 3. Load data (X, y tensors + splits)
    # 4. Create dataloaders
    # 5. Instantiate model via get_model(config)
    # 6. Create Trainer and run
    # 7. Print final summary
```

**Note:** The `train` CLI must handle the case where data hasn't been preprocessed yet. For Sprint 3, assume data is already windowed and split (tensors + splits.json exist in data dir). Full pipeline orchestration is a future enhancement.

### Architecture References

- Section 4.4 (Experiment Config YAML — training fields)
- Section 4.5 (Checkpoint Bundle — file structure and dict format)
- Section 5.3 (Training Module — Trainer responsibilities)
- Section 7.2 (train --config workflow sequence)
- Section 12.2 (Logging levels — epoch summaries at INFO, timing at WARNING)
- Section 13.3 Rule 3 (Seed before everything)
- Section 13.3 Rule 7 (Checkpoint completeness — 6 required keys)
- Section 13.3 Rule 8 (Config immutable at runtime)
- FR8 (BCEWithLogitsLoss), FR11 (checkpoint + config), NFR1 (pip freeze), NFR3 (timing probe), NFR4 (per-epoch CSV)

### Edge Cases

- **Zero-epoch training (epochs=0):** Should return immediately with no checkpoint
- **Patience=0:** Effectively no early stopping (never triggers)
- **All val metrics are 0:** Still saves first checkpoint (improves from -inf)
- **NaN loss:** Log error and stop training gracefully
- **Empty train/val loader:** Should error clearly before training starts
- **FrequencyBaseline (no params):** Trainer should handle models with no parameters (optimizer has nothing to update) — forward pass still works, val metrics still computed
- **Output dir doesn't exist:** Create it automatically

---

## Dependencies

**Prerequisite Stories:**
- STORY-3.1: Frequency/Recency Heuristic (provides `BaseModel`, `MODEL_REGISTRY`, `get_model()`)
- STORY-3.2: ANN GRU Baseline (provides first trainable model + `configs/baseline_gru.yaml`)
- STORY-2.3: Evaluation Harness & Metrics (provides `compute_all_metrics`, `evaluate_model`)
- STORY-2.4: CLI evaluate Command (establishes CLI patterns)
- STORY-2.1: Windowed Tensor Construction (data pipeline)
- STORY-2.2: Time-Based Splits (splits + DataLoaders)
- STORY-1.2: Logging, Seed Management & Config (`setup_logging`, `set_global_seed`, `load_config`)

**Blocked Stories:**
- STORY-3.4: Baseline Results & Comparison (needs trained models)
- All Sprint 4-6 stories (all model training depends on this)

**External Dependencies:**
- None (all deps are internal)

---

## Definition of Done

- [ ] `Trainer` class implemented in `src/c5_snn/training/trainer.py`
- [ ] `train` CLI subcommand implemented in `src/c5_snn/cli.py`
- [ ] `src/c5_snn/training/__init__.py` updated with `Trainer` export
- [ ] Unit tests in `tests/test_train.py`:
  - [ ] Trainer constructs with valid inputs
  - [ ] Training loop runs for specified epochs
  - [ ] Early stopping halts at correct epoch
  - [ ] Checkpoint saved with all 6 required keys
  - [ ] Metrics CSV written with correct columns
  - [ ] Config snapshot saved at start
  - [ ] pip freeze captured
  - [ ] Timing probe logs warning when projected >20 min
- [ ] Integration test:
  - [ ] Train GRU 2 epochs on ~100 rows via Trainer
  - [ ] All output files exist with correct structure
  - [ ] CLI `train --config` produces expected output
- [ ] `ruff check src/ tests/` passes with zero errors
- [ ] `pytest tests/ -v` passes (all existing + new tests)
- [ ] CI green on GitHub Actions
- [ ] Acceptance criteria validated (all checked)
- [ ] Code committed to `main` branch and pushed

---

## Story Points Breakdown

- **Trainer class (training loop + early stopping + checkpoint):** 3 points
- **Per-epoch logging + config snapshot + pip freeze:** 1 point
- **train CLI command:** 2 points
- **Testing (unit + integration):** 2 points
- **Total:** 8 points

**Rationale:** This is the most complex infrastructure story in the project. The training loop itself is standard PyTorch boilerplate, but the combination of early stopping, checkpoint management, timing probes, config snapshots, and CLI integration requires careful orchestration. The integration test is particularly important since it validates the full pipeline end-to-end.

---

## Additional Notes

- This is the critical path story for the entire project — every subsequent model training depends on it
- The `Trainer` must be model-agnostic: it works with any `BaseModel` subclass (FrequencyBaseline, GRUBaseline, and all future SNN models)
- For `FrequencyBaseline` (no learnable parameters): the Trainer still runs but loss won't decrease and parameters won't update. This is expected behavior — the model is evaluated via metrics
- The checkpoint format uses `model_state_dict` (not the full model object) per architecture Section 4.5. This differs from the existing evaluate CLI which stores the full model — the `train` CLI stores state dict for proper serialization
- The `evaluate` CLI will need to be updated later (STORY-3.4 or separate task) to load state_dict checkpoints from the trainer. For now, the integration test can verify checkpoint structure independently
- RunPod timing threshold is 20 minutes per architecture decision (Section 3.1)
- Config is immutable: snapshot at start, never modify during training (Section 13.3 Rule 8)

---

## Progress Tracking

**Status History:**
- 2026-02-11: Created by Scrum Master (AI)
- 2026-02-11: Started by Developer (AI)
- 2026-02-11: Completed by Developer (AI)

**Actual Effort:** 8 points (matched estimate)

**Implementation Notes:**
- `Trainer` class in `training/trainer.py`: generic training loop for any BaseModel
- Handles models with no learnable parameters (FrequencyBaseline) by skipping optimizer
- Training loop: forward -> BCEWithLogitsLoss -> backward -> optimizer.step()
- Early stopping on `val_recall_at_20` with configurable patience
- Checkpoint bundle with 6 required keys (Section 4.5 format)
- Per-epoch metrics CSV, config snapshot (immutable), pip freeze
- 2-epoch timing probe with RunPod warning when projected >20 min
- `train` CLI subcommand: loads config, sets seed, builds data pipeline, trains, prints summary
- 32 new tests (210 total): construction, training loop, early stopping, checkpoint, metrics CSV, config snapshot, pip freeze, timing probe, FrequencyBaseline handling, integration test, CLI tests

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
