# Start Here Tomorrow — 2026-02-10

## Quick Context

Sprint 1 is complete. All 4 foundation stories delivered. CI is green. Next up is Sprint 2: Data Pipeline & Evaluation Harness.

## What To Do Next

### Step 1: Start Sprint 2

Run these commands in sequence to define and implement the first Sprint 2 story:

```
/bmad:create-story STORY-2.1
```

Then:

```
/bmad:dev-story STORY-2.1
```

### Sprint 2 Story Order

| Order | Story | Title | Points |
|-------|-------|-------|--------|
| 1 | STORY-2.1 | Windowed Tensor Construction | 5 |
| 2 | STORY-2.2 | Time-Based Train/Val/Test Splits | 3 |
| 3 | STORY-2.3 | Evaluation Harness & Metrics | 5 |
| 4 | STORY-2.4 | CLI evaluate Command | 3 |

**Sprint 2 goal:** Complete data pipeline and evaluation harness ready for any model.

### Step 2: End of Session

When done for the day, type `ED` to trigger the end-of-session wrap-up protocol.

## Key References

| What | Where |
|------|-------|
| Architecture | `docs/architecture.md` |
| Sprint plan | `docs/sprint-plan-c5-snn-2026-02-10.md` |
| Sprint status | `.bmad/sprint-status.yaml` |
| Project memory | `docs/project-memory.md` |
| Session history | `docs/sessions/` |
| Story documents | `docs/stories/STORY-*.md` |

## Environment

- **Working directory:** `c:\Users\Minis\CascadeProjects\c5_SNN`
- **Python:** 3.11.9 (global install, no venv currently)
- **Package:** Installed as `pip install -e ".[dev]"`
- **CLI:** `c5-snn --help` works
- **CI:** GitHub Actions green on every push to main

## Verification Commands

```bash
# Confirm everything still works
ruff check src/ tests/
pytest tests/ -v
c5-snn validate-data
```
