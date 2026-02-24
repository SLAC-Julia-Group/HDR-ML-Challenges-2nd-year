# Neural Activity Prediction -- NU/NeuroBench Challenge

## Overview

Cross-domain transfer learning: NEXUS foundation model (pretrained on particle track reconstruction) is adapted to predict future neural activity from monkey electrophysiology recordings.

## Model Description

**Architecture**: Two-stage chained inference with shared ATLAS backbone

**Backbone** (shared with coastal flooding submission):
- Asymmetric autoencoder encoder: `[2048, 1024, 512] -> latent_dim=128` with GELU activations
- Pretrained on ATLAS Open Data (particle track reconstruction)

**Stage 1 — Feature Prediction** (`feat_pred`):
- Input: waveform features (30 channels × 6 features) from steps 1–10
- Head: `Dense([1024, 1024], gelu, dropout=0.05) -> Dense(180)` → reshape to (30, 6)
- Predicts features for the prediction window

**Stage 2 — Step Generation** (`step_gen`):
- Input: predicted features from Stage 1
- Head: timestep embedding (10 → 32) + shared `Dense([256, 128], gelu, dropout=0.1) -> Dense(30)`
- Predicts 10 future steps for each channel

**Training variant**: `ft` (pretrained backbone + fine-tuned encoder, per-monkey weights)

**Subjects**: 2 monkeys with separate weights
| Monkey | Channels | Weight files |
|--------|----------|-------------|
| affi | 239 | `feat_pred_affi.weights.pkl`, `step_gen_affi.weights.pkl` |
| beignet | 89 | `feat_pred_bei.weights.pkl`, `step_gen_bei.weights.pkl` |

## Feature Engineering

10-step waveform per channel → 6 features (mirrored to ATLAS track features):

| Feature | ATLAS Analog | Description |
|---------|-------------|-------------|
| `weighted_trough_loc` | eta | Soft-min weighted time position |
| `std` | pT | Waveform variability |
| `channel_id_norm` | phi | Normalized channel index [0, 1] |
| `mean` | d0 | Mean amplitude |
| `slope` | z0 | Linear trend |
| `pulse_depth` | chi2 | Mean minus trough depth |

30-channel sliding windows (stride=1) → sort by descending std → encode → chain predict → unsort → denormalize → average across windows.

## Usage (called by ingestion program)

```python
from model import Model

m = Model(monkey_name="affi")   # or "beignet"
m.load()
predictions = m.predict(X)     # X: (N, 20, C, 9), returns (N, 20, C)
```

**Output**: `(N, 20, C)` — steps 1–10 are raw passthrough; steps 11–20 are predicted.

## Files

| File | Description |
|------|-------------|
| `model.py` | Self-contained pipeline: architecture, weight loading, feature extraction, inference |
| `feat_pred_affi/bei.weights.pkl` | Feature prediction weights (per monkey, includes fine-tuned encoder) |
| `step_gen_affi/bei.weights.pkl` | Step generation weights (per monkey, includes fine-tuned encoder) |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |
