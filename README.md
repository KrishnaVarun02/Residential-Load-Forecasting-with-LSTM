# Residential Load Forecasting

A runnable PyTorch and scikit-learn implementation of the LSTM, GRU and regression models in the project's **May 3, 2023 report**, with chronological evaluation, saved models and next-interval inference. Original reports, data, spreadsheets and exploratory notebooks are preserved.

The committed daily series supports a complete local run without downloading data. These runs are **new measurements**, not reproductions of the report's historical scores.

## Install and run

Use Python 3.10–3.12 (validated with Python 3.12.14 on CPU). From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt

# Normalize schema; preserve missing labels and original numeric units.
python -m load_forecasting prepare --input LSTM/household_daily.csv \
  --output data/daily.csv --target-unit 'sum of minute kW samples'

# Ten epochs maximum for both real neural architectures, plus seven baselines.
python -m load_forecasting train --data data/daily.csv \
  --config configs/daily.json --output runs/daily

# New process: reload saved weights, models and preprocessing; score held-out test.
python -m load_forecasting evaluate --run runs/daily
python -m load_forecasting evaluate --run runs/daily --split validation

# Forecast one unseen interval after the last observed row.
python -m load_forecasting predict --run runs/daily --history data/daily.csv \
  --output runs/daily/next.json

python -m ruff check .
python -m pytest -q
```

Training refuses to overwrite a nonempty run directory. Choose a new directory for each experiment. `configs/smoke.json` runs the same full neural architectures for two epochs. Dependencies have bounded ranges; `requirements-tested.txt` records the exact core and test versions used for the recorded run, rather than claiming identical numerical results across hardware or library versions.

Each run contains:

- `metadata.json`: resolved configuration, source SHA-256, split dates/counts, source units, package versions and fitted preprocessing.
- `lstm.pt`, `gru.pt`: neural weights; `*.joblib`: fitted regression/empirical models.
- `history.json`: training/validation losses, selected epoch and parameter counts.
- `metrics.json`, `validation_predictions.csv`, `test_predictions.csv`: original-unit scores and timestamped predictions for all models.

Only load artifacts you trust: joblib model loading uses Python pickle. Moving a run is supported; use `evaluate --run <directory> --data <original-data-copy>` if the original absolute data path changes. Evaluation verifies the saved data fingerprint so a changed file cannot silently change the held-out split. Inference can use new history without fitting anything.

## Data, target and leakage controls

`LSTM/household_daily.csv` has 1,442 daily rows, December 16, 2006–November 26, 2010. Its active-power values are **sums of minute kW samples**, not daily mean kW. Its voltage and current columns are also aggregates. We retain the recorded values and explicitly label metrics in those source units. If all minute observations and intervals were verified, multiplying the target by 1/60 would yield kWh; the source reconstruction and missingness details are absent, so the workflow does not make that conversion. Partial boundary days are retained.

All models predict **next-row `Global_active_power`** from earlier observations. The default uses the seven measured variables, including lagged active power. Chronological 70/15/15 boundaries are assigned by the target timestamp. Validation and test use observed earlier history, including earlier held-out rows: this is rolling one-step forecasting, not recursive multi-step forecasting.

Feature missingness is filled forward, never backward; initial missing values use training-only medians. Minima and ranges are fitted on training rows only and are saved. Missing target labels are excluded from training/scoring rather than evaluated as imputed observations. Duplicate/invalid timestamps are rejected. The cadence is inferred from training timestamps, and windows crossing a gap are excluded. Test values outside the training range are allowed and are not clipped. Inputs are loaded once; neural windows are generated lazily in minibatches. Regression/polynomial matrix size guards reject excessive expansions with an actionable error.

Neural models use seeded CPU execution, deterministic PyTorch algorithms, Adam, chronological minibatches and gradient norm clipping at 5. LSTM loss is MSE; GRU loss is MAE. The best validation-loss epoch is restored, with configurable patience. Test data never selects an epoch or preprocessing parameters. Regression hyperparameters are fixed in JSON; editing them after reviewing test scores would invalidate a clean holdout claim. Inference needs at least `max(2, lookback)` ordered history rows at the trained cadence, with the same columns/units. It emits one forecast per model for the next timestamp.

Metrics are RMSE, MSE, MAE, R², explained variance, WAPE and MAPE. MAPE excludes zero actual values and reports its included count; undefined constant/zero-target statistics are JSON `null`. MAPE can still be unstable near zero. Persistence, empirical time-of-day/weekday-type mean and empirical MAPE-minimizing forecasts provide reference baselines. The empirical MAPE estimator uses the exact inverse-absolute-load weighted median of nonzero training observations; unseen time groups fall back to the training distribution.

## Raw-data and report-specific workflows

The original minute-level `household_power_consumption.txt` / `.csv` is not committed. Obtain the original dataset independently; the reader accepts semicolon-separated `Date`, `Time` and seven measurement columns (`?` missing values), or a CSV with `datetime`. Raw active-power units are kW. No network download is part of training.

The original notebooks used **different experiments**: the 200/200 LSTM used hourly means and dropped Voltage (six inputs), while the 75/30/30 GRU used seven minute-level inputs. These runnable configurations preserve those feature/model choices while correcting the original split/preprocessing protocol:

```bash
python -m load_forecasting prepare --input LSTM/household_power_consumption.txt \
  --output data/hourly.csv --frequency h --target-unit kW
python -m load_forecasting train --data data/hourly.csv \
  --config configs/lstm_hourly.json --output runs/lstm-hourly

python -m load_forecasting prepare --input LSTM/household_power_consumption.txt \
  --output data/minute.csv --target-unit kW
python -m load_forecasting train --data data/minute.csv \
  --config configs/gru_minute.json --output runs/gru-minute
```

Hourly preparation averages each measurement; it does not sum power or invent energy labels. Config `lstm_hourly.json` sets 100 maximum epochs/batch 70; `gru_minute.json` sets 10/batch 64. Default `daily.json` uses a common seven-feature daily target for a fair, runnable local comparison. These protocols cannot be compared directly to the historical tables.

For the report's conceptual load/time/day/holiday framework, set `feature_mode` to `report_calendar` in a copied config and supply `datetime`, `Global_active_power` and a complete, genuine binary `holiday` column (other sensors are unnecessary). For schema normalization also pass `prepare --feature-mode report_calendar`. It uses lagged active power, half-hour slot 1–48, weekday 0–6, and holiday 0/1. No holiday labels or calendar jurisdiction are invented. `measured` uses seven measurements; `lstm_six` excludes Voltage. All input transforms use training-only statistics. Appropriate half-hour energy data and holiday labels are needed to evaluate that original conceptual framework.

## Code, notebooks and report coverage

- `load_forecasting/data.py`: ingestion, optional mean resampling, causal preprocessing, lazy windows and temporal boundaries.
- `load_forecasting/models.py`: real 200/200 LSTM with 0.2/0.3 dropout, 75/30/30 GRU, linear/ridge/lasso/degree-2 polynomial models, empirical baselines.
- `load_forecasting/workflow.py`: training, validation selection, checkpoint loading, evaluation and inference.
- `LSTM/report_workflow.ipynb`: portable executable notebook using the maintained workflow. Install `jupyter` to open it from the root or `LSTM` directory.
- `LSTM/forecast.py`: compatibility entry point; `python LSTM/forecast.py LSTM/household_daily.csv --epochs 2 --output runs/compat.json` now also saves reloadable artifacts.
- The six original notebooks retain their code/output as historical records, with a notice linking to the maintained notebook. Their Colab/Kaggle paths, random splits, full-data scaling and some obsolete APIs are not the supported execution path.
- `LSTM/lstm.py`: preserved, separate 48-sheet `Fri.xlsx` experiment with 19 features, 48%/52% split and a four-unit TensorFlow LSTM. Install `requirements-legacy.txt`, then run `cd LSTM && python lstm.py`. This writes `FriRNN1_generated.xlsx`; it is not the report neural comparison and was not rerun for the current validation.

See [REPORT_COVERAGE.md](REPORT_COVERAGE.md) for report authority, exact method-to-file mapping, original notebook differences and remaining evidence gaps. The original published LSTM/GRU/regression scores, multi-household aggregation and clustering claims remain unverified. The original minute dataset, preprocessing outputs, experiment splits/seeds/checkpoints and household/holiday artifacts are absent. The public reference paper and presentation are context, not substitutes for missing experimental evidence.

## Validation

`tests/test_pipeline.py` exercises train-only preprocessing and outlier invariance, missing labels, cadence gaps, raw preparation, all report architectures, gradient flow, neural loss reduction, all-model training/save/load/inference equality and the actual CLI in fresh processes. GitHub Actions runs lint, tests and a real two-epoch daily training/evaluation/inference path on Python 3.12 CPU. See [VALIDATION.md](VALIDATION.md) for the completed local run and measured scores.
