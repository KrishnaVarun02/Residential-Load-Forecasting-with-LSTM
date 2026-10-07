# Residential Load Forecasting

An undergraduate research-oriented study of **short-term residential electricity-load forecasting** using LSTM and GRU recurrent neural networks, together with conventional regression and empirical baselines.

The repository contains project reports and notebooks alongside a maintained PyTorch/scikit-learn workflow for chronological evaluation, leakage-aware preprocessing, saved artifacts, and next-interval inference.

## Research question

How do recurrent neural networks for short-term residential load forecasting compare with simpler forecasting baselines when temporal ordering, preprocessing, and model selection are controlled?

## Methodology

### Recurrent models

- Two-layer LSTM with hidden size 200 and dropout.
- Three-layer GRU with hidden sizes 75, 30, and 30.

### Comparison methods

- Linear regression
- Ridge regression
- Lasso regression
- Polynomial regression
- Persistence
- Empirical time-of-day/weekday forecasts
- Empirical MAPE-minimizing forecasts

### Temporal evaluation

- Chronological train/validation/test boundaries.
- Rolling one-step forecasting rather than recursive multi-step prediction.
- Validation loss selects the neural model checkpoint before test evaluation.
- Test data never determines preprocessing statistics or model-selection decisions.

### Leakage and preprocessing controls

- Missing features are forward-filled; initial missing values use training-only medians.
- Normalization ranges are fitted on training rows only and stored with model artifacts.
- Missing target labels are excluded from training/scoring rather than imputed.
- Duplicate or invalid timestamps are rejected.
- Windows crossing cadence gaps are excluded.
- Test values outside the training range are retained rather than clipped.

## Experiments / Evaluation

The committed daily series contains **1,442 daily observations**. The workflow predicts the next-row Global_active_power value from preceding observations.

The evaluation reports RMSE, MSE, MAE, R², explained variance, WAPE, and MAPE.

Saved run metadata includes the data fingerprint, split boundaries, fitted preprocessing, package versions, model artifacts, training history, and timestamped predictions.

## Results / Findings

On the maintained daily evaluation, the test **R² is 0.250 for LSTM and 0.396 for Ridge**. This provides a direct comparison between a recurrent model and a conventional baseline within the same maintained dataset and protocol.

The preserved project report also contains a GRU experiment reporting **R² = 0.938 and RMSE = 0.263**. That experiment uses a different feature/sampling configuration and is therefore not treated as directly comparable with the maintained daily evaluation.

The key research observation is methodological as well as numerical: the evaluation pipeline makes temporal splitting, preprocessing, model selection, and metric definitions explicit enough to distinguish results obtained under different experimental protocols.

## Technical implementation

- load_forecasting/data.py — ingestion, chronological splits, causal preprocessing, gap-aware windows
- load_forecasting/models.py — LSTM, GRU, regression and empirical models
- load_forecasting/workflow.py — training, validation selection, evaluation, artifact reload, inference
- load_forecasting/metrics.py — evaluation metrics and edge-case handling
- tests/ — preprocessing, training, persistence, and inference tests
- LSTM/ — preserved project reports and notebooks

## Reproduce the maintained workflow

Use Python 3.10–3.12:

~~~
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt

python -m load_forecasting prepare --input LSTM/household_daily.csv --output data/daily.csv --target-unit 'sum of minute kW samples'
python -m load_forecasting train --data data/daily.csv --config configs/daily.json --output runs/daily
python -m load_forecasting evaluate --run runs/daily
python -m load_forecasting predict --run runs/daily --history data/daily.csv --output runs/daily/next.json

python -m pytest -q
python -m ruff check .
~~~

## Limitations

The maintained evaluation is based on a single-household residential load series. It should therefore be interpreted as an experimental study of the specified dataset and forecasting protocol rather than as evidence of general forecasting performance across households or domains.

## Academic context

The project is part of undergraduate academic work in Computer Science and Engineering at **IIT (BHU) Varanasi**, under the guidance of **Prof. S. K. Singh and Dr. Jayashankara M**.
