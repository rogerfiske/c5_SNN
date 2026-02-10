# STORY-1.4: CI Skeleton & First Green Build

## Story Info

| Field | Value |
| --- | --- |
| **Story ID** | STORY-1.4 |
| **Epic** | Epic 1 — Foundation & Data Validation |
| **Sprint** | 1 |
| **Points** | 2 |
| **Priority** | Must Have |
| **Status** | Defined |
| **Dependencies** | STORY-1.3 (completed) |

---

## User Story

As a developer,
I want a GitHub Actions workflow that runs linting and tests on every push,
so that code quality is enforced from the first commit.

---

## Acceptance Criteria

- [ ] **AC-1:** `.github/workflows/ci.yml` exists and triggers on:
  - Push to `main` branch
  - Pull requests targeting `main`
- [ ] **AC-2:** The workflow uses Python 3.11 and installs the project with `pip install -e ".[dev]"` using CPU-only PyTorch (no CUDA/ROCm wheels in CI)
- [ ] **AC-3:** The workflow runs `ruff check src/ tests/` as a lint step
- [ ] **AC-4:** The workflow runs `pytest tests/ -v` as a test step
- [ ] **AC-5:** The workflow steps are ordered: checkout → Python setup → install → lint → test (lint runs before tests so formatting issues are caught first)
- [ ] **AC-6:** CI passes on the current codebase (all 26 tests green, ruff clean)
- [ ] **AC-7:** The CI status check is visible on the GitHub repository (green badge after push)
- [ ] **AC-8:** `ruff check src/ tests/` passes locally before push (sanity check)
- [ ] **AC-9:** `pytest tests/ -v` passes locally before push (sanity check)

---

## Technical Notes

### Architecture References

- **Section 3.1 (CI/CD Infrastructure):** GitHub Actions (free tier) — CPU-only for linting + unit tests. No cloud storage, no managed services, no containers.
- **Section 3.2 (Tech Stack):** pytest 8.3.4, Ruff 0.8.6, pyright 1.1.391 as dev dependencies. GitHub Actions for automated test + lint on push/PR.
- **Section 10 (Source Tree):** `.github/workflows/ci.yml` — GitHub Actions: pytest + ruff on push/PR.
- **Section 11.2 (CI Pipeline):** Push/PR to main → checkout → Python 3.11 setup → `pip install -e ".[dev]"` (CPU torch) → `ruff check src/` → `pytest tests/ -v`.

### Workflow Structure

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - Checkout code
      - Set up Python 3.11
      - Install CPU-only PyTorch (via --index-url for CPU wheel)
      - pip install -e ".[dev]"
      - ruff check src/ tests/
      - pytest tests/ -v
```

### CPU-Only PyTorch in CI

The standard `pip install torch==2.5.1` from PyPI installs the CPU wheel by default on Linux runners without CUDA. If this pulls a GPU wheel, use the explicit CPU index:
```
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```
Then install the rest of the project normally. This keeps CI fast and avoids downloading ~2GB GPU libraries.

### Existing Project State

- **Dev dependencies** already declared in `pyproject.toml` under `[project.optional-dependencies.dev]`: pytest==8.3.4, ruff==0.8.6, pyright==1.1.391
- **`.github/workflows/` directory** already exists (created in STORY-1.1 with `.gitkeep`)
- **26 tests** currently passing across 6 test files
- **Ruff** clean on current codebase

### What NOT to Implement

- No deployment pipeline (no staging, no production)
- No Docker containers
- No code coverage reporting (can be added later)
- No pyright step in CI yet (can be added in a future story if needed)
- No matrix testing across multiple Python versions (3.11 only)
- No caching of pip dependencies (keep it simple for now)

---

## Definition of Done

1. `.github/workflows/ci.yml` committed and pushed to `main`
2. GitHub Actions run triggers automatically on push
3. CI run completes with all steps green (lint + tests)
4. Status check visible on GitHub repository page
5. `ruff check src/ tests/` passes locally
6. `pytest tests/ -v` passes locally
7. Code committed to repository

---

## Dev Notes

_This section is updated during implementation._

- Implementation started: —
- Implementation completed: —
- Notes: —
