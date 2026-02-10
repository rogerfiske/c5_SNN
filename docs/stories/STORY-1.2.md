# STORY-1.2: Logging, Seed Management & Configuration Infrastructure

## Story Info

| Field | Value |
| --- | --- |
| **Story ID** | STORY-1.2 |
| **Epic** | Epic 1 — Foundation & Data Validation |
| **Sprint** | 1 |
| **Points** | 3 |
| **Priority** | Must Have |
| **Status** | Defined |
| **Dependencies** | STORY-1.1 (completed) |

---

## User Story

As a developer,
I want centralized logging and deterministic seed management available from the start,
so that every subsequent story has reproducibility and observability built in from day one.

---

## Acceptance Criteria

- [ ] **AC-1:** `src/c5_snn/utils/seed.py` provides `set_global_seed(seed: int) -> None` that sets:
  - `random.seed(seed)`
  - `numpy.random.seed(seed)`
  - `torch.manual_seed(seed)`
  - `torch.cuda.manual_seed_all(seed)`
  - `torch.backends.cudnn.deterministic = True`
  - `torch.backends.cudnn.benchmark = False`
- [ ] **AC-2:** `src/c5_snn/utils/logging.py` provides `setup_logging(level: str = "INFO", log_file: str | None = None) -> None` that:
  - Configures the root `c5_snn` logger
  - Uses format: `%(asctime)s [%(levelname)s] %(name)s: %(message)s`
  - Outputs to stderr (StreamHandler)
  - Optionally outputs to a file if `log_file` is provided (FileHandler)
  - Default level: `INFO`
- [ ] **AC-3:** `src/c5_snn/utils/config.py` provides `load_config(path: str) -> dict` that:
  - Loads a YAML file and returns the parsed dict
  - Raises `ConfigError` if the file does not exist
  - Raises `ConfigError` if YAML parsing fails
  - Raises `ConfigError` if the result is not a dict (e.g., empty file returns `None`)
- [ ] **AC-4:** `src/c5_snn/utils/device.py` provides `get_device() -> torch.device` that:
  - Returns `torch.device("cuda")` if `torch.cuda.is_available()`
  - Returns `torch.device("cpu")` otherwise
  - Logs the selected device at INFO level
  - Logs a WARNING if ROCm environment detected but CUDA not available (suggesting `HSA_OVERRIDE_GFX_VERSION`)
- [ ] **AC-5:** Custom exception classes defined in `src/c5_snn/utils/exceptions.py`:
  - `C5SNNError(Exception)` — base exception
  - `DataValidationError(C5SNNError)` — data integrity failures
  - `ConfigError(C5SNNError)` — config loading/parsing failures
  - `TrainingError(C5SNNError)` — unrecoverable training errors
- [ ] **AC-6:** Base config file `configs/default.yaml` created with structure:
  ```yaml
  experiment:
    name: "default"
    seed: 42

  data:
    raw_path: "data/raw/CA5_matrix_binary.csv"
    window_size: 21
    split_ratios: [0.70, 0.15, 0.15]
    batch_size: 64

  model:
    type: "frequency_baseline"

  training:
    epochs: 100
    learning_rate: 0.001
    optimizer: "adam"
    early_stopping_patience: 10
    early_stopping_metric: "val_recall_at_20"

  output:
    dir: "results/default"

  log_level: "INFO"
  ```
- [ ] **AC-7:** `src/c5_snn/utils/__init__.py` exports all public interfaces so that the following works:
  ```python
  from c5_snn.utils import set_global_seed, setup_logging, load_config, get_device
  from c5_snn.utils.exceptions import C5SNNError, ConfigError, DataValidationError, TrainingError
  ```
- [ ] **AC-8:** Unit tests in `tests/test_seed.py`:
  - Two calls with same seed produce identical `random.random()`, `numpy.random.rand()`, and `torch.rand(5)` sequences
  - Two calls with different seeds produce different sequences
- [ ] **AC-9:** Unit tests in `tests/test_config.py`:
  - `load_config` on a valid YAML file returns a dict
  - `load_config` on a non-existent path raises `ConfigError`
  - `load_config` on an invalid YAML file raises `ConfigError`
  - `load_config` on an empty file raises `ConfigError`
- [ ] **AC-10:** Unit tests in `tests/test_logging_setup.py`:
  - `setup_logging("DEBUG")` sets the `c5_snn` logger to DEBUG level
  - Logger output matches the expected format pattern
  - File handler is created when `log_file` is provided
- [ ] **AC-11:** `ruff check src/ tests/` passes with zero errors
- [ ] **AC-12:** `pytest tests/ -v` passes with all tests green

---

## Technical Notes

### Architecture References

- **Section 5.5 (Utils Module):** Defines the four utility files and their interfaces. Follow these exactly.
- **Section 12.1 (Error Handling):** Exception hierarchy — `C5SNNError` base with three subclasses. Place in a separate `exceptions.py` file so all modules can import without circular dependencies.
- **Section 12.2 (Logging Standards):** Format string, level meanings, default level. Use `logging.getLogger(__name__)` pattern in each module.
- **Section 13.3 Rule #3 (Seed Before Everything):** `set_global_seed` is called as the very first action in training/evaluation. It must be complete and deterministic.
- **Section 13.3 Rule #6 (No print()):** All utils must use `logging`, never `print()`.
- **Section 4.4 (Experiment Config):** The YAML config schema that `load_config` will parse. The `default.yaml` should match this structure.

### Implementation Details

**seed.py:**
- Import `random`, `numpy`, `torch` at module level
- Single function, no class needed
- The `torch.backends.cudnn` settings ensure deterministic behavior on GPU

**logging.py:**
- Configure the `c5_snn` logger (not the root logger) to avoid conflicting with other libraries
- Use `logging.getLogger("c5_snn")` as the base
- StreamHandler to stderr, optional FileHandler
- Call `logger.setLevel()` and handler `setLevel()` consistently

**config.py:**
- Use `yaml.safe_load()` (never `yaml.load()` — security)
- Validate that the result is a dict (empty YAML files return `None`)
- Wrap `FileNotFoundError` and `yaml.YAMLError` into `ConfigError` for consistent error handling
- Do NOT validate config schema here — just load and return. Schema validation happens in consumers.

**device.py:**
- Check `torch.cuda.is_available()` as the primary test
- ROCm detection hint: check if `os.environ.get("HSA_OVERRIDE_GFX_VERSION")` is set or if the platform suggests AMD GPU
- Log the device choice at INFO level
- This is a pure function — no side effects beyond logging

**exceptions.py:**
- Simple exception classes, no custom logic needed
- Each gets a one-line docstring

### What NOT to Implement

- No config schema validation (consumers validate what they need)
- No config merging or defaults cascade (just load raw YAML)
- No logging rotation or advanced handlers (too early)
- No GPU memory management (later stories handle this)

---

## Definition of Done

1. All acceptance criteria (AC-1 through AC-12) are met
2. `from c5_snn.utils import set_global_seed, setup_logging, load_config, get_device` works
3. `configs/default.yaml` exists and is loadable
4. All unit tests pass: `pytest tests/ -v`
5. Lint clean: `ruff check src/ tests/`
6. Code committed to repository

---

## Dev Notes

_This section is updated during implementation._

- Implementation started: —
- Implementation completed: —
- Notes: —
