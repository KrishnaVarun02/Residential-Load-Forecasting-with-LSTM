# Report-to-code coverage

## Authoritative source and inspected history

`LSTM/AI_report.pdf` is the authoritative project report: cover date **May 3, 2023**, 20 pages, PDF creation time May 3 07:43:01 IST. `lstm_final_report.pdf` (20 pages, May 2 22:39:42) and `AI_final_report.pdf` (18 pages, May 2 21:00:06) are earlier drafts. All three entered the same initial Git commit `eb8dbbe`, so Git commit time alone does not order them. `LSTM/AI_project_Gaurav&Varun.zip` includes a byte-identical `AI_report.pdf` (SHA-256 `0ab32579aeab5e4dcf5a6d74ef5c6f5cf6a53dfd1531e88817c5cc049fc56347`) and existing notebook/presentation copies, with no extra raw data or training configurations. The 2019 IEEE paper is a cited reference, not the student's implementation report.

Before edits, repository `main` was clean at `7c5378b` (merged `imp1`), with only `main` locally and remotely; no project `AGENTS.md` was present. README, all six notebook code inventories, scripts, reports, requirements, three existing tests, data schema and Git history were reviewed. `imp2` preserves the original files and implements the missing maintained execution path. The prior `forecast.py` already implemented a chronological comparison, but lacked configuration persistence, model loading, separate evaluation/inference, trained-epoch selection and runnable notebook reconciliation.

## Implemented coverage

| Report method / evidence | Current implementation | Status and deliberate differences |
|---|---|---|
| 200/200 LSTM, sequence output then final output, dropout 0.2/0.3, scalar dense head | `models.RecurrentForecaster`; `configs/lstm_hourly.json` | Real recurrent autograd/Adam training, validation checkpoint selection, save/load and inference. Original `lstm_model.ipynb` drops Voltage, hourly means, first 365×24 rows train, 100 epochs / batch 70, MSE. Six-feature configuration retains width/features but corrects to chronological train/validation/test and training-only scaling. |
| 75/30/30 GRU, scalar head, MAE/Adam | `models.RecurrentForecaster`; `configs/gru_minute.json` | Real three-layer GRU. Last GRU uses its last output as in the report summary and notebook; prose incorrectly suggests all layers return sequences. Seven-feature config retains 10 epochs / batch 64 but replaces random 70/30 split and test-as-validation. |
| Linear, ridge, lasso, polynomial regressions | `models.regression_model`; flattened lagged windows from `data.Windows` | All four fit, serialize, evaluate and infer on the common next-interval active-power target. Ridge alpha 0.0001, lasso alpha 1, polynomial degree 2 follow notebook choices; deprecated sklearn normalization is replaced by common training-only min/max scaling. |
| Empirical means by customer/time/day type | `models.EmpiricalPredictor` | Time-of-day and weekday/weekend grouping, trained only on training targets; this dataset contains one household, so customer ID is implicit. Unseen groups fall back to training mean. |
| Empirical MAPE minimization | `models.mape_minimizer` | Exact empirical inverse-absolute-value weighted median for nonzero targets; no invented discretization grid or historical score. |
| Load + time-slot + weekday + holiday framework | `data.feature_frame`, `feature_mode=report_calendar` | Real feature preparation supported and validated; complete holiday 0/1 labels required. No committed labeled half-hour multi-household energy dataset to evaluate the conceptual framework. |
| Evaluation tables: RMSE, MAE, R², explained variance; MAPE discussion | `metrics.py`, timestamped prediction CSVs, `workflow.evaluate` | All reported metrics implemented, plus MSE/WAPE/counts; undefined values explicit. New scores are in source units and do not reproduce historical tables. |
| Usable notebook/script interface | `LSTM/report_workflow.ipynb`, `LSTM/forecast.py`, `__main__.py` | One maintained prep→train→validate→evaluate→infer implementation; old code/output preserved and clearly identified as historical. |
| Density clustering and multi-household aggregated forecasting discussion | No claim of replication | Described mainly in reference-paper narrative. Household IDs, per-household energy series, cluster choices and experiment artifacts are absent; cannot verify. |

The supported neural code is a PyTorch reimplementation. PyTorch's LSTM uses two bias vectors per gate while Keras uses one; default seven-input LSTM has **489,001** parameters, versus the report's six-input Keras LSTM **486,601**. The new six-input PyTorch LSTM has **488,201** parameters. These are architecture-family matches, not interchangeable historical checkpoints. The seven-input GRU has **34,141** parameters, matching the report. Framework initialization/training conventions and new split/preprocessing choices also differ.

## Historical notebook reconciliation

- `lstm_model.ipynb` and `lstm_load_forecast-1.ipynb`: hourly aggregation, Voltage exclusion, 200/200 LSTM; paths and scaling are not a complete leakage-safe reproduction artifact.
- `lstm_gru_model.ipynb` and `lstm_load_forecast_2.ipynb`: seven measurements; random train/test split and full-data min/max scaling; test also used as validation. The alternative LSTM here is 75/30/30, while the authoritative report's LSTM section specifies 200/200.
- `ai_model_3_regressions.ipynb`: creates derived `power_consumption` (unmetered energy), then **drops it from features** and targets contemporaneous `Global_active_power` from six other same-time measurements. These are explanatory regressions with random splits and full-data imputation, not forward forecasts. The maintained comparison explicitly uses lagged observations for every family. The prior README's assertion that this notebook predicted the derived target was incorrect.
- `rolling-multi-step-forecasts-with-lstm-rnn-gru.ipynb`: separate exploratory daily/rolling RNN/LSTM/GRU experiment (64-unit layers), repeated transformations and incompatible paths. No exact report experiment mapping or leakage-free archived run is established. The maintained workflow evaluates rolling observed-history one-step predictions; it does not claim recursive multistep results.
- `lstm.py` and Excel sheets: distinct four-unit/19-feature experiment, preserved with its original scope; not evidence for the report's 200/200 or 75/30/30 architectures.

Each historical notebook now starts with a link to `report_workflow.ipynb`; original cells and outputs remain unchanged.

## Exact evidence gaps

The minute files `household_power_consumption.csv`, `household_power_consumption.txt` and modified raw-data archive are absent (ignored because of repository size constraints). The daily CSV has no committed provenance script linking its exact missingness handling and sums to those files. No original serialized scalers, random-split row assignments, RNG state, training checkpoints, complete experiment configs or original prediction vectors exist. The report tables (LSTM RMSE 0.598 / MAE 0.428 / R² 0.506; GRU RMSE 0.263 / MAE 0.086 / R² 0.938; regression table) therefore remain unverified. The archive adds no missing artifacts.

No household IDs/multi-customer dataset, holiday calendar labels/jurisdiction, weather input series, clustering settings or household aggregation experiment configuration is available. The new workflow supports independently testable preprocessing and model methods without inventing those observations or claimed outcomes.
