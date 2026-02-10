# PM Handoff Document — c5_SNN

**Date:** 2026-02-09
**From:** BMad Master (Project Audit & Bootstrap)
**To:** Product Manager (PM Agent)

---

## 1. Project Summary

**c5_SNN** is a Spiking Neural Network (SNN) time-series forecasting project. The goal is to predict the **Top-20 most likely parts** (from a pool of 39) for the next event, given a historical sequence of events where each event contains exactly 5 unique parts.

This is a **greenfield ML research pipeline** — no production code exists yet. The project is in early bootstrap phase with planning documents and raw data in place.

---

## 2. What Exists Today

### Documents (now in `docs/`)
| File | Location | Description |
|------|----------|-------------|
| `prd.md` | `docs/prd.md` | Draft PRD from ChatGPT — **needs BMAD restructuring** |
| `architecture.md` | `docs/architecture.md` | Draft Architecture from ChatGPT — **needs BMAD restructuring** |
| `concept.md` | `docs/concept.md` | Project concept brief (problem, approach, SNN rationale) |
| `project-memory.md` | `docs/project-memory.md` | Key decisions, open questions, next actions |

### Data
| File | Location | Size | Description |
|------|----------|------|-------------|
| `CA5_matrix_binary.csv` | `data/raw/` | 1.2 MB | Primary training data: 11,700 events, 45 columns (date, m_1..m_5, P_1..P_39) |
| `CA5_date.csv` | `data/raw/` | 290 KB | Human-readable audit view (date, m_1..m_5) |

### Research Materials
| File | Location | Description |
|------|----------|-------------|
| `Spike_Neural_Networks.pdf` | `docs-imported/` | SNN research paper |
| `c5_SNN_Codex_GPT 5.3_interaction.md` | `docs-imported/` | Prior ChatGPT interaction log with decisions |
| `CA5_date_csv_CA5_matrix_binary.csv_description.md` | `docs-imported/` | Dataset column descriptions |
| `pc_specs.md` | `docs-imported/` | Developer PC specs (Ryzen 9, 64GB RAM, RX 6600M) |

### Bootstrap Code (stubs only)
| File | Location | Status |
|------|----------|--------|
| `cli.py` | `c5_SNN_bootstrap_docs/src/c5_snn/` | Skeleton CLI with validate-data stub |
| `validate_data.py` | `c5_SNN_bootstrap_docs/scripts/` | Placeholder — raises NotImplementedError |
| `__init__.py` | `c5_SNN_bootstrap_docs/src/c5_snn/` | Package init, version 0.1.0 |

### Infrastructure
- **BMAD Core**: Fully installed (`.bmad-core/` with agents, tasks, templates, workflows)
- **Git**: NOT initialized — no repository, no commits
- **Build System**: None — no pyproject.toml, requirements.txt, or setup.py at root
- **IDE**: .claude, .gemini, .windsurf integration folders present

---

## 3. Key Decisions Already Made

From prior ChatGPT sessions (documented in `docs/project-memory.md` and `docs-imported/`):

1. **Dataset**: Use `CA5_matrix_binary.csv` as canonical training representation (P_1..P_39 multi-hot). Keep `CA5_date.csv` as audit view.
2. **Task framing**: Multi-label ranking over 39 labels; output Top-20.
3. **Primary metrics**: Recall@20 and Hit@20 (averaged over test set).
4. **Loss function**: BCEWithLogitsLoss + optional cardinality penalty.
5. **Window size**: Start W=21, tune in range 7-90.
6. **Compute policy**: Local timing probe (2 epochs); if >20 min projected, use RunPod B200/B100 GPU.
7. **SNN framework**: PyTorch + snnTorch with surrogate gradients.
8. **Progressive model complexity**: Phase A (spiking MLP/CNN) -> Phase B (Spike-GRU) -> Phase C (spiking transformer).
9. **Data integrity**: Both CSVs validated as consistent; 11,700 events from 1992-2026.

---

## 4. What the PM Needs To Do

### 4.1 Process the PRD Through BMAD Workflow

The current `docs/prd.md` is a **narrative draft** from ChatGPT. It needs to be transformed into a proper BMAD PRD using the `prd-tmpl.yaml` template. Specifically, the PM must:

1. **Skip Project Brief** — the `docs/concept.md` serves this purpose adequately. Reference it as the foundation.

2. **Goals & Background Context** — Extract from current PRD sections 1-3 and concept.md:
   - Goals: Top-20 prediction, measurable Recall@20/Hit@20 lift, reproducible workflow
   - Background: SNN forecasting on CA5 parts dataset, 39 labels, multi-hot encoding
   - Create Change Log table

3. **Requirements** — The current PRD has requirements scattered across sections 5-10. The PM must formalize these into:
   - **FR (Functional Requirements)**: Data validation, windowed sampling, baseline training, SNN training (Phases A/B/C), evaluation harness, CLI commands (validate-data, train, evaluate, predict), checkpoint management, Top-20 export
   - **NFR (Non-Functional Requirements)**: Reproducibility (fixed seeds), <20 min local training probe, no data leakage, time-based splits only, ONNX/TorchScript export optional

4. **UI Goals** — Mark as N/A. This is a CLI-only ML pipeline, no UI.

5. **Technical Assumptions** — Extract and formalize:
   - Repository: Monorepo
   - Architecture: ML research pipeline (single-service)
   - Language: Python 3.11+
   - Frameworks: PyTorch, snnTorch, pandas, numpy
   - Testing: Unit + Integration (pytest)
   - CLI: argparse or click
   - Compute: Local + RunPod fallback

6. **Epic List** — The current PRD has 6 milestones that map roughly to epics:
   - Epic 1: Project Foundation (repo setup, git, data validation, CI skeleton)
   - Epic 2: Data Pipeline & Evaluation Harness (windowing, splits, metrics)
   - Epic 3: Baseline Models (frequency heuristic, ANN GRU/TCN)
   - Epic 4: SNN Phase A (spiking MLP/CNN)
   - Epic 5: SNN Phase B (Spike-GRU)
   - Epic 6: SNN Phase C (spiking transformer) + Final Report

   **Note**: The PM should validate these epics against BMAD principles — each must deliver deployable, testable value. Consider merging Phase B + C if scope is small.

7. **Epic Details with Stories** — Break each epic into vertical-slice stories with acceptance criteria. Size for AI agent execution (2-4 hours each).

8. **Run PM Checklist** — Execute `pm-checklist.md` and populate results.

9. **Next Steps** — Generate Architect prompt for BMAD architecture processing.

### 4.2 Important Context for Story Creation

- **No existing codebase** — Epic 1 stories must set up the entire project scaffold (pyproject.toml, git init, directory structure, logging, etc.)
- **Data is ready** — Raw CSVs are in `data/raw/`, validation logic is the first real implementation work
- **Cross-cutting concerns**: Logging, reproducibility (seed management), and experiment tracking should be woven into early stories, not deferred
- **Developer PC**: Ryzen 9 6900HX, 64GB RAM, RX 6600M 8GB — adequate for local dev, may need to use available RunPod w/ B200 GPU for full training
- **GitHub repo exists**: https://github.com/rogerfiske/c5_SNN (mentioned in interaction log, needs git init and push)

### 4.3 After PRD Approval

Hand off to the **Architect** (`/architect`) with this prompt:

> Review `docs/prd.md` (the BMAD-processed PRD) and the existing draft at `docs/architecture.md`. Use `docs/concept.md` and `docs/project-memory.md` as additional context. Also review `docs-imported/pc_specs.md` for hardware constraints and `docs-imported/Spike_Neural_Networks.pdf` for SNN domain knowledge. Process the architecture through the BMAD architecture template, paying special attention to: tech stack with pinned versions, source tree, data models (windowed tensors, model checkpoints, evaluation artifacts), component design, coding standards for ML/Python, and test strategy for an ML research pipeline.

---

## 5. Open Questions for PM Resolution

These should be addressed during PRD elicitation:

1. **Window size tuning scope**: Should hyperparameter tuning (W=7-90) be a formal story or handled implicitly during model training stories?
2. **Encoding strategy**: Direct spike vs rate-coded vs latency-coded — should this be an explicit FR or left to the developer during SNN phases?
3. **Calendar features**: The concept mentions optional calendar features (day-of-week, holidays). Should this be in-scope for v1 or deferred?
4. **Class imbalance handling**: Focal loss vs per-label weighting — FR or implementation detail?
5. **RunPod integration**: Does this need formal stories (provisioning, upload, download) or is it ad-hoc developer tooling?
6. **GitHub repo**: Should git init + first push be the very first story in Epic 1?
7. **Experiment tracking**: Should we use MLflow, Weights & Biases, or simple CSV logging? This affects architecture decisions.

---

## 6. File Map for Quick Reference

```
c5_SNN/
├── .bmad-core/              # BMAD framework (complete)
├── c5_SNN_bootstrap_docs/   # Original ChatGPT bootstrap (reference only)
│   ├── docs/                # Original drafts (PRD, ARCH, CONCEPT, MEMORY)
│   ├── scripts/             # Skeleton validate_data.py
│   └── src/c5_snn/          # Skeleton CLI
├── data/
│   └── raw/                 # CA5_matrix_binary.csv, CA5_date.csv
├── docs/                    # Active project documentation
│   ├── architecture/        # (empty — for sharded architecture)
│   ├── architecture.md      # Draft architecture (needs BMAD processing)
│   ├── concept.md           # Project concept brief
│   ├── prd/                 # (empty — for sharded PRD/epics)
│   ├── prd.md               # Draft PRD (needs BMAD processing)
│   ├── project-memory.md    # Decisions & open questions
│   ├── qa/                  # (empty — for QA gate docs)
│   ├── stories/             # (empty — for user stories)
│   └── PM-HANDOFF.md        # THIS DOCUMENT
├── docs-imported/           # Research materials & prior interactions
│   ├── Spike_Neural_Networks.pdf
│   ├── c5_SNN_Codex_GPT 5.3_interaction.md
│   ├── CA5_date_csv_CA5_matrix_binary.csv_description.md
│   └── pc_specs.md
└── web-bundles/             # BMAD agent configs for multi-IDE
```

---

## 7. Recommended PM Workflow

1. **Activate PM**: `/pm`
2. **Start PRD workflow**: `*create-doc prd` (or use the BMAD PRD workflow)
3. **Reference inputs**: Point to `docs/concept.md` as the project brief, `docs/prd.md` as the draft to restructure
4. **Work through each BMAD PRD section** interactively with the user
5. **Output final PRD**: `*doc-out` to `docs/prd.md`
6. **Shard if needed**: The core-config has `prdSharded: true` with epic pattern `epic-{n}*.md` in `docs/prd/`
7. **Hand off to Architect** with the prompt from section 4.3 above
