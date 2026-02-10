# c5_SNN — Project Concept Brief

## Problem
Given a historical sequence of “events”, each containing **5 unique parts** drawn from **1–39**, predict the **Top-20 most likely parts** to appear in the **next event**.

This is a **set prediction / ranking** problem (multi-label over 39 labels) rather than a positional sequence prediction.

## Data
Primary training representation:
- `data/raw/CA5_matrix_binary.csv`
  - `date, m_1..m_5` + `P_1..P_39` (multi-hot; exactly 5 ones per event)

Secondary/audit view:
- `data/raw/CA5_date.csv`
  - `date, m_1..m_5` (human readable)

## Why SNN here
The dataset already represents **events** (multi-hot vectors), which maps naturally to **spike/rate coded inputs**.
Recent time-series SNN work emphasizes:
- temporal alignment between data time steps and SNN simulation steps,
- explicit encoding layers to map continuous inputs into spike trains,
- surrogate-gradient training and SNN counterparts of CNN/RNN/Transformer backbones.

## Core approach
1. **Windowed forecasting**:
   - Input: last `W` events of `P_1..P_39` (default `W=21`, tuned).
   - Output: `39` logits for the next event.

2. **Training target**:
   - Multi-label vector `y ∈ {0,1}^{39}` with exactly 5 ones.

3. **Loss**
   - Primary: `BCEWithLogitsLoss` (multi-label)
   - Optional regularizers:
     - **cardinality penalty**: encourage ~5 positives
     - **label smoothing** (mild) for stability

4. **Decode / inference**
   - Rank 39 logits → return **Top-20** indices as the predicted pool.

## Evaluation metrics
Primary:
- **Recall@20** (of the 5 true parts, how many are in Top-20)
- **Hit@20** (binary: did we hit at least 1 true part)
Secondary:
- Recall@5 / Hit@5
- Mean Reciprocal Rank (MRR) on the 5 positives
- Exact set match (strict; expected low)

## Baselines (must-have)
Before SNN optimization, lock in strong non-SNN baselines:
- Frequency / recency heuristics
- Logistic regression (windowed)
- Temporal ConvNet / GRU (ANN)
These establish a lower bound and detect leakage/bugs.

## SNN candidate models (incremental)
Phase A (fastest):
- **Spiking-MLP** / **Spiking-1D CNN** over windowed multi-hot

Phase B (sequence-capable):
- **Spike-GRU / Spike-RNN** with LIF neurons (surrogate gradients)

Phase C (longer context):
- **Spiking Transformer / SeqSNN-style** spiking temporal model with positional encoding

## Compute policy
- Run a short timing probe locally (2 epochs).
- If projected full training exceeds ~20 minutes locally, move training to RunPod (B200/B100-class GPU).

## Deliverables
- Reproducible training + evaluation pipeline
- Experiment logs + saved checkpoints
- A CLI that outputs Top-20 predictions for a given “as-of” date
- PRD + architecture + project memory
