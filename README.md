# Residential Load Forecasting with LSTM

This project explores short-term residential electricity-load forecasting with recurrent neural networks. It uses household power-consumption time series to train and compare LSTM, GRU, and RNN-based forecasting workflows.

## Contents

- `LSTM/lstm_model.ipynb` and `LSTM/lstm_load_forecast-1.ipynb` — LSTM data preparation and load-forecasting experiments.
- `LSTM/lstm_gru_model.ipynb` and `LSTM/lstm_load_forecast_2.ipynb` — LSTM/GRU forecasting experiments.
- `LSTM/rolling-multi-step-forecasts-with-lstm-rnn-gru.ipynb` — multivariate, rolling multi-step forecasts with RNN, LSTM, and GRU models.
- `LSTM/ai_model_3_regressions.ipynb` — exploratory preprocessing and regression work.
- `LSTM/lstm.py` — batch script that trains a small LSTM for each worksheet in `Fri.xlsx`, evaluates RMSE/MAE/MSE/MAPE, and writes generated predictions to `FriRNN1_generated.xlsx`.
- `LSTM/forecast.py` — a repeatable one-step comparison of the report's LSTM, GRU, and four regression model families.
- `LSTM/household_daily.csv` and the Excel workbooks — prepared data and experiment outputs.
- `LSTM/*.pdf` and `LSTM/*.pptx` — project reports and presentation material.

## Dataset

The original household-power-consumption files are intentionally not committed because they exceed GitHub's 100 MB file-size limit. Obtain the raw dataset independently, then update each notebook's input path (some examples use Google Drive or Kaggle paths) to point to your local copy.

Expected raw-file names used by the notebooks include:

```text
household_power_consumption.csv
household_power_consumption.txt
```

These files are ignored by Git and can safely remain in the project directory.

## Setup

Use Python 3.10+ and install the packages for the scripts:

```bash
python -m pip install -r requirements.txt
```

The notebooks also need Jupyter and plotting packages (`jupyter`, `matplotlib`, and, for some notebooks, `seaborn`). They are exploratory records with environment-specific paths. To inspect them:

```bash
python -m pip install jupyter matplotlib seaborn
jupyter notebook
```

To run the report comparison on the committed daily sample, from the repository root:

```bash
python LSTM/forecast.py LSTM/household_daily.csv --output forecast_metrics.json
```

The original minute dataset can be passed in place of the daily sample. It is needed for a closer comparison with the reported experiments. The script uses the seven measured numeric variables, predicts the next global active power value, and writes test metrics as JSON. Use `--models linear ridge lasso polynomial` to run without TensorFlow, or `--lookback` and `--epochs` to adjust the experiment.

For the separate worksheet experiment, run from the `LSTM` directory:

```bash
cd LSTM
python lstm.py
```

## Report coverage and limitations

- The latest project report is [`LSTM/AI_report.pdf`](LSTM/AI_report.pdf). Git history contains three report files from the same initial commit; PDF creation times and text show that `AI_report.pdf` is the May 3 revision of the May 2 drafts. The 2019 IEEE paper is a cited reference.
- The report's 200/200-unit LSTM with 0.2/0.3 dropout, 75/30/30-unit GRU, and linear, ridge, lasso, and polynomial regressions are represented in `forecast.py`. It uses chronological 70/15/15 splits and scales from training rows only. Results include RMSE, MAE, R², and MAPE on nonzero actual values. This corrects leakage from random splitting and full-data scaling in the exploratory notebooks; those notebooks remain historical artifacts.
- The report describes a time/day/holiday feature framework, but the included household dataset has no holiday labels. The runnable comparison uses the seven measured numeric variables in the report's experimental section. The report's regression notebook instead predicts a derived unmetered-power target; `forecast.py` uses active power for every model so its metrics share one target. Its scores are fresh measurements, not a reproduction of the historical tables. The report's LSTM parameter count also implies six inputs, while its GRU count implies seven, so exact parameter-count reproduction is ambiguous.
- `lstm.py` retains the report-independent 48-sheet, 19-feature experiment and 48%/52% split. It now fits scaling on training rows, calculates MAE and MAPE separately, and writes `FriRNN1_generated.xlsx` so the committed workbook is preserved.
- The original raw dataset, preprocessing details for the historical reported scores, and holiday labels are absent. Published metrics cannot be verified from the committed daily sample alone.

## References

The repository includes a copy of *Short-Term Residential Load Forecasting Based on LSTM Recurrent Neural Network* in the `LSTM` directory for project context.
