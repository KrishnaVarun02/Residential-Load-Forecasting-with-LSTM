# Completed local validation

Recorded October 5, 2026 on Python 3.12.14/macOS arm64 CPU, with one PyTorch thread and seed 7. Exact core/test versions are in `requirements-tested.txt`. Machine-readable config, dataset hash, split dates, metrics, histories and next forecast are in [`validation/daily_cpu.json`](validation/daily_cpu.json). Generated weights and prediction CSVs remain in ignored run directories and are recreated by the commands below.

```bash
python -m ruff check .
python -m pytest -q
python -m load_forecasting train --data LSTM/household_daily.csv --config configs/daily.json --output runs/daily
python -m load_forecasting evaluate --run runs/daily --output runs/daily/reloaded.json
python -m load_forecasting predict --run runs/daily --history LSTM/household_daily.csv --output runs/daily/next.json
python LSTM/forecast.py LSTM/household_daily.csv --models linear ridge lasso polynomial --epochs 1 --output runs/compat.json
```

- Ruff: **passed**.
- Pytest: **23 passed**, no warnings (about 20 seconds). This includes real recurrent parameter gradients, loss reduction, all nine models trained and reloaded, CLI subprocesses, missing-label/gap invariants, and six-/four-feature report configurations using only their required input columns.
- Default daily training, separate-process evaluation and inference: **passed**. Reloaded test metrics exactly match the training run's saved test metrics. An independent review reran preparation and full daily training with the same resulting scores.
- Compatibility `LSTM/forecast.py` regression command: **passed**, including artifact generation.
- `LSTM/report_workflow.ipynb`: **all seven cells executed successfully** using nbclient and an isolated Python 3.12 kernel (four code cells). The kernel needed loopback-socket access; no raw-data download was performed. The notebook performs actual neural/regression training, evaluates loaded models, checks exact metrics equality and predicts the next timestamp.
- Original PDF/report files and original notebook cells/outputs were preserved; historical notebooks have one introductory notice each.

The daily run contains 1,008 training targets, 216 validation targets and 217 test targets, with lookback 1. LSTM ran 10 epochs and selected epoch 8; normalized training MSE decreased from 0.062113 to 0.013546. GRU stopped after 7 epochs (patience 5) and restored epoch 2; normalized training MAE decreased from 0.203751 to 0.096990. Epoch selection uses validation loss, not test metrics.

All following error magnitudes are in **sum-of-minute-kW-sample units** of the committed daily file (not kW); MAPE is percent. These are measurements of this implementation and protocol, not historical report results.

| Model | Test RMSE | Test MAE | Test R² | Test MAPE % |
|---|---:|---:|---:|---:|
| LSTM | 368.107 | 287.593 | 0.250 | 26.746 |
| GRU | 518.366 | 433.851 | -0.487 | 44.189 |
| Linear | 330.458 | 252.553 | 0.396 | 22.124 |
| Ridge | 330.447 | 252.527 | 0.396 | 22.120 |
| Lasso | 465.779 | 367.765 | -0.201 | 37.807 |
| Polynomial | 331.991 | 252.159 | 0.390 | 20.648 |
| Persistence | 380.156 | 261.164 | 0.200 | 18.624 |
| Empirical mean | 469.564 | 359.822 | -0.220 | 37.320 |
| Empirical MAPE | 448.266 | 358.641 | -0.112 | 29.736 |

Neural models are not uniformly better here: ridge has lower RMSE than either neural model, while persistence has lower MAPE. The retained lasso alpha 1 strongly regularizes normalized data and gives a constant forecast in this run. No model or setting was changed to improve held-out test scores. The next unseen timestamp forecast is November 27, 2010; its actual outcome is absent, so no inference accuracy claim is made.

The original minute/hourly report scores, half-hour holiday framework on real labeled data, multi-household/clustering experiments and separate 48-sheet legacy TensorFlow run remain unverified for the artifact reasons in `REPORT_COVERAGE.md`. GitHub Actions provides an additional Linux CPU lint/test/train/evaluate/infer check; the local record does not substitute for its final remote status.
