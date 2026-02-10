# STORY-1.1: Repository & Package Scaffold

## Story Info

| Field | Value |
| --- | --- |
| **Story ID** | STORY-1.1 |
| **Epic** | Epic 1 — Foundation & Data Validation |
| **Sprint** | 1 |
| **Points** | 3 |
| **Priority** | Must Have |
| **Status** | Defined |
| **Dependencies** | None (first story) |

---

## User Story

As a developer,
I want a properly initialized git repository with a pip-installable Python package structure,
so that all subsequent work has a clean, standard foundation to build on.

---

## Acceptance Criteria

- [ ] **AC-1:** Git repository initialized with `.gitignore` covering Python (`__pycache__`, `.mypy_cache`, `*.pyc`), IDE files (`.vscode/`, `.idea/`), data artifacts (`data/raw/`, `data/processed/`), results (`results/`), and OS files (`.DS_Store`, `Thumbs.db`)
- [ ] **AC-2:** `pyproject.toml` with:
  - Project metadata: name `c5-snn`, version `0.1.0`
  - `requires-python = ">=3.11"`
  - `[project.dependencies]` with pinned versions from architecture Section 3.2:
    - `torch==2.5.1`
    - `snntorch==0.9.1`
    - `pandas==2.2.3`
    - `numpy==1.26.4`
    - `pyyaml==6.0.2`
    - `click==8.1.8`
    - `tqdm==4.67.1`
  - `[project.optional-dependencies]` dev extras:
    - `pytest==8.3.4`
    - `ruff==0.8.6`
    - `pyright==1.1.391`
  - `[project.scripts]` entry point: `c5-snn = "c5_snn.cli:cli"`
- [ ] **AC-3:** Package installable via `pip install -e ".[dev]"` with `src/c5_snn/` layout
- [ ] **AC-4:** `src/c5_snn/__init__.py` with `__version__ = "0.1.0"`
- [ ] **AC-5:** Full directory structure created per architecture Section 10 (Source Tree):
  - `src/c5_snn/` — package root with `__init__.py`
  - `src/c5_snn/data/` — with `__init__.py` (empty placeholder)
  - `src/c5_snn/models/` — with `__init__.py` (empty placeholder)
  - `src/c5_snn/training/` — with `__init__.py` (empty placeholder)
  - `src/c5_snn/utils/` — with `__init__.py` (empty placeholder)
  - `src/c5_snn/cli.py` — placeholder with Click group stub
  - `configs/` — empty directory with `.gitkeep`
  - `data/raw/` — gitignored, `.gitkeep` in `data/`
  - `data/processed/` — gitignored
  - `results/` — gitignored
  - `tests/` — with `conftest.py` placeholder
  - `scripts/` — empty directory with `.gitkeep`
- [ ] **AC-6:** `README.md` with:
  - Project name and one-line description
  - Setup instructions (venv creation, pip install)
  - ROCm install note with `HSA_OVERRIDE_GFX_VERSION=10.3.0` for RX 6600M
  - Basic usage placeholder (CLI commands)
- [ ] **AC-7:** ROCm smoke test passes — `import torch; print(torch.cuda.is_available())` runs without error on local machine (result may be `True` or `False` depending on ROCm setup; both are acceptable at this stage)
- [ ] **AC-8:** `ruff check src/ tests/` passes with zero errors on the scaffold
- [ ] **AC-9:** `pytest tests/ -v` runs successfully (0 tests collected is acceptable at scaffold stage)
- [ ] **AC-10:** Initial commit pushed to `https://github.com/rogerfiske/c5_SNN`

---

## Technical Notes

### Architecture References

- **Section 3.2 (Tech Stack Table):** All dependency versions are pinned here. Use these exact versions in `pyproject.toml`.
- **Section 10 (Source Tree):** The full directory structure to scaffold. Create all directories and placeholder `__init__.py` files, but do NOT implement any business logic — subsequent stories fill in the modules.
- **Section 11.1 (Environments):** Three install modes — ROCm (local), CUDA (RunPod), CPU (CI). The `pyproject.toml` should list `torch==2.5.1` as a dependency. Install instructions in README cover the correct PyTorch wheel URL per environment.
- **Section 13 (Coding Standards):** Ruff must pass from the start. Configure `[tool.ruff]` in `pyproject.toml` with `line-length = 100` and `target-version = "py311"`.

### PyTorch Install Strategy

PyTorch must be installed from the correct index URL per environment:

- **Local (ROCm):** `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/rocm6.2`
- **RunPod (CUDA):** `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124`
- **CI (CPU):** `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu`

Document these in `README.md`. The `pyproject.toml` lists `torch==2.5.1` without index URL; the correct wheel is selected by the install command.

### CLI Placeholder

Create a minimal Click group in `src/c5_snn/cli.py`:

```python
import click

@click.group()
def cli():
    """c5_SNN: Spiking Neural Network time-series forecasting pipeline."""
    pass
```

This allows `c5-snn --help` to work immediately after install. Actual subcommands (`validate-data`, `train`, `evaluate`, `predict`) are added in later stories.

### What NOT to Implement

This story is scaffold only. Do NOT implement:

- Any data loading, validation, or windowing logic (STORY-1.3, STORY-2.1)
- Any model classes (STORY-3.2, STORY-4.1+)
- Any training logic (STORY-3.3)
- Seed management, logging, or config utilities (STORY-1.2)

---

## Definition of Done

1. All acceptance criteria (AC-1 through AC-10) are met
2. `pip install -e ".[dev]"` succeeds in a clean venv
3. `c5-snn --help` prints the CLI help text
4. `ruff check src/ tests/` returns zero errors
5. `pytest tests/ -v` runs without error
6. Code is committed and pushed to the GitHub repository

---

## Dev Notes

_This section is updated during implementation._

- Implementation started: —
- Implementation completed: —
- Notes: —
