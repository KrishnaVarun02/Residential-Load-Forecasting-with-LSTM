# Residential Load Forecasting with LSTM

This project explores short-term residential electricity-load forecasting with recurrent neural networks. It uses household power-consumption time series to train and compare LSTM, GRU, and RNN-based forecasting workflows.

## Contents

- `LSTM/lstm_model.ipynb` and `LSTM/lstm_load_forecast-1.ipynb` — LSTM data preparation and load-forecasting experiments.
- `LSTM/lstm_gru_model.ipynb` and `LSTM/lstm_load_forecast_2.ipynb` — LSTM/GRU forecasting experiments.
- `LSTM/rolling-multi-step-forecasts-with-lstm-rnn-gru.ipynb` — multivariate, rolling multi-step forecasts with RNN, LSTM, and GRU models.
- `LSTM/ai_model_3_regressions.ipynb` — exploratory preprocessing and regression work.
- `LSTM/lstm.py` — batch script that trains an LSTM for each worksheet in `Fri.xlsx`, evaluates RMSE/MAE/MSE, and writes predictions to `FriRNN1.xlsx`.
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

Use Python 3.10+ and install the packages required by the notebooks and script:

```bash
python -m pip install numpy pandas matplotlib scikit-learn tensorflow openpyxl jupyter
```

Then launch the notebooks from the repository root:

```bash
jupyter notebook
```

Update the dataset path in the selected notebook before running its cells. For the worksheet-based script, run it from the `LSTM` directory so its relative Excel-file paths resolve correctly:

```bash
cd LSTM
python lstm.py
```

## Notes

- The script uses a 48%/52% train/test split, a one-step look-back window, 19 input features, and a small LSTM with four units.
- Forecast quality is reported with RMSE, MAE (labelled as MAPE in the script), and MSE.
- The notebooks are experimental and may contain environment-specific paths or saved cell output; adapting paths and training parameters is expected.

## References

The repository includes a copy of *Short-Term Residential Load Forecasting Based on LSTM Recurrent Neural Network* in the `LSTM` directory for project context.
