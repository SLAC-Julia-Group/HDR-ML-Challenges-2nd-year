# Coastal Flooding Prediction -- iHARP ML Challenge 2

## Overview

Cross-domain transfer learning: an ATLAS particle physics foundation model (3.3M parameters, pretrained on charged particle track reconstruction) is adapted to predict coastal flooding events from NDBC buoy sea level time series.

## Model Description

**Architecture**: Asymmetric autoencoder encoder as backbone

- Encoder: `[2048, 1024, 512] -> latent_dim=128` with GELU activations
- Regression head: `Dense(15)` -- 15-bin softmax, one bin per flood day count (0–14)
- Total parameters: ~3.06M (encoder) + 1,935 (head)

**Training variant**: `ft` (pretrained backbone + fine-tuned encoder)
- Pretrained on ATLAS Open Data (particle track reconstruction, run2 files)
- Fine-tuned on 9 NDBC coastal stations (1950-2020)
- Loss: Focal loss over 15 flood-count bins

**Output**: `y_prob = 1 − P(bin=0) = P(≥1 flood day)` -- probability of flooding

## Feature Engineering

Sea level time series (7 days x 24 hours = 168 hours) -> 30 temporal segments x 6 features:

| Feature | ATLAS Analog | Description |
|---------|-------------|-------------|
| `energy_centroid` | eta | Energy-weighted time position [-1, 1] |
| `std` | pT | Sea level variability |
| `time_centroid` | phi | Time-weighted center of mass [-pi, pi] |
| `tail_asymmetry` | d0 | Distribution asymmetry |
| `dt` | z0 | Normalized station longitude [-1, 1] |
| `kurtosis_shifted` | chi2 | Excess kurtosis + 3 |

## Usage (called by ingestion program)

```bash
python model.py \
    --train_hourly /path/to/train_hourly.csv \
    --test_hourly  /path/to/test_hourly.csv \
    --test_index   /path/to/test_index.csv \
    --predictions_out /path/to/predictions.csv
```

## Output Format

`predictions.csv` with columns: `id, y_prob`

## Files

| File | Description |
|------|-------------|
| `model.py` | Self-contained pipeline: data loading, feature extraction, model inference |
| `model.pkl` | Trained model weights (numpy arrays) + architecture metadata |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |
