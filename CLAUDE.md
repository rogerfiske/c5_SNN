# c5_SNN — Claude Code Project Instructions

## Project Overview

**c5_SNN** is a Spiking Neural Network time-series forecasting pipeline for CA5 event prediction. It uses the BMAD Method v6 for project management with AI-driven development.

- **Repository:** https://github.com/rogerfiske/c5_SNN
- **Working directory:** `c:\Users\Minis\CascadeProjects\c5_SNN`
- **Tech stack:** Python 3.11, PyTorch 2.5.1, snnTorch 0.9.1, pandas, Click CLI, pytest, Ruff
- **Architecture:** Pipeline pattern (validate -> window -> split -> train -> evaluate -> report)
- **Project level:** BMAD Level 2 (sprint-based, multi-epic)

## User Profile

- The user (Roger) is **NOT a programmer** — delegate all technical decisions to the AI agent
- Follow BMAD workflows strictly when invoked
- Always commit and push after completing a story
- Run `ruff check src/ tests/` and `pytest tests/ -v` before every commit

## Key Files

| File | Purpose |
|------|---------|
| `docs/architecture.md` | Full architecture (~1040 lines, 17 sections) |
| `docs/sprint-plan-c5-snn-2026-02-10.md` | Sprint plan (7 sprints, 22 stories, 103 pts) |
| `.bmad/sprint-status.yaml` | Sprint tracking (current status, velocity) |
| `docs/project-memory.md` | Persistent project decisions and context |
| `configs/default.yaml` | Base experiment config template |
| `docs/stories/STORY-*.md` | Individual story documents |
| `docs/sessions/` | Session summaries and startup guides |

## Standard Workflow

The user follows this pattern for each story:
1. `/bmad:create-story STORY-X.Y` — Scrum Master defines the story
2. `/bmad:dev-story STORY-X.Y` — Developer implements it

## Custom Commands

### `ED` — End of Day / End of Session Wrap-Up

When the user types `ED` (case-insensitive), execute the following end-of-session protocol:

1. **Session Summary** — Create `docs/sessions/Session_summary_YYYY-MM-DD.md`:
   - What was accomplished (stories completed, key deliverables)
   - Current project status (sprint progress, points, velocity)
   - Files created/modified
   - Test results (count, pass/fail)
   - CI status
   - Any issues or blockers encountered

2. **Startup Guide** — Create `docs/sessions/Start_here_tomorrow_YYYY-MM-DD.md`:
   - Which agent/command to run next
   - Current sprint and next story to implement
   - Any context the next session needs
   - Quick-start command sequence

3. **Update README.md** — Reflect current project state and available commands

4. **Update `docs/project-memory.md`** — Add new decisions, learnings, completed items

5. **Update any stale documents** — Sprint status, story docs, etc.

6. **Git commit and push** — Stage all session wrap-up changes and push to GitHub

7. **Display summary** — Show the user a concise end-of-session report

If the date already has session files, append a suffix (e.g., `_v2`) to avoid overwriting.

## Development Standards

- **Lint:** `ruff check src/ tests/` must pass with zero errors before commit
- **Tests:** `pytest tests/ -v` must pass before commit
- **CI:** GitHub Actions runs on push/PR to main (lint + tests)
- **Commits:** Use conventional commit format (`feat:`, `fix:`, `ci:`, `chore:`, `docs:`)
- **PyTorch in CI:** CPU-only via `--index-url https://download.pytorch.org/whl/cpu`
- **No data leakage:** Train < val < test chronologically, no shuffling
- **Seeds:** Always call `set_global_seed()` before any stochastic operation
- **Config-driven:** All experiments fully specified by YAML config files

## Compute Environment

- **Local:** Windows, Python 3.11.9, AMD RX 6600M (ROCm — CPU-only wheel on Windows)
- **Remote:** RunPod B200 (CUDA 12.4) — use when training >20 min locally
- **CI:** Ubuntu, Python 3.11, CPU-only PyTorch
