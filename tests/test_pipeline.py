"""Focused causal-data, neural training and actual CLI artifact integration tests."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from load_forecasting.config import Config, MODELS
from load_forecasting.data import FEATURES, Preprocessor, build_splits, feature_frame, load_data, prepare_csv
from load_forecasting.metrics import metrics
from load_forecasting.models import EmpiricalPredictor, RecurrentForecaster, mape_minimizer
from load_forecasting.workflow import evaluate, predict, train


def sample(n=160):
    steps = np.arange(n)
    frame = pd.DataFrame({name: 2 + steps * 0.005 + np.sin(steps / 5 + i) * 0.2
                          for i, name in enumerate(FEATURES)})
    frame.insert(0, "datetime", pd.date_range("2023-01-01", periods=n, freq="h"))
    return frame


def test_causal_scaling_missing_labels_and_roundtrip():
    frame = sample(100)
    frame.loc[0, "Voltage"] = np.nan
    frame.loc[84, FEATURES[0]] = np.nan
    config = Config(lookback=3)
    splits, scaler, boundaries = build_splits(frame, config)
    changed = frame.copy()
    changed.loc[85:, list(FEATURES)] = 1e9
    other, other_scaler, _ = build_splits(changed, config)
    assert scaler.to_dict() == other_scaler.to_dict()
    np.testing.assert_array_equal(splits["train"].arrays()[0], other["train"].arrays()[0])
    np.testing.assert_array_equal(splits["validation"].arrays()[0], other["validation"].arrays()[0])
    restored = Preprocessor.from_dict(json.loads(json.dumps(scaler.to_dict())))
    np.testing.assert_array_equal(restored.transform(frame), scaler.transform(frame))
    assert boundaries == (70, 85)
    assert 84 not in splits["validation"].indices
    assert splits["train"].indices.max() < splits["validation"].indices.min() < splits["test"].indices.min()
    assert splits["validation"][0][0][-1, 0] == scaler.transform(frame)[69, 0]
    assert np.isfinite(splits["train"].arrays()[0]).all()


def test_preprocessing_is_forward_only():
    frame = sample(100)
    frame.loc[60:62, "Voltage"] = np.nan
    _, scaler, _ = build_splits(frame, Config())
    values = scaler.transform(frame)
    np.testing.assert_array_equal(values[59:63, 2], np.repeat(values[59, 2], 4))
    frame.loc[63, "Voltage"] = 1e8
    np.testing.assert_array_equal(scaler.transform(frame)[60:63, 2], values[60:63, 2])


def test_gaps_are_not_treated_as_adjacent():
    frame = sample(100)
    frame["datetime"] = frame.datetime + pd.to_timedelta(np.where(np.arange(len(frame)) >= 40, 2, 0), unit="h")
    splits, _, _ = build_splits(frame, Config(lookback=3))
    assert not set([40, 41, 42]) & set(splits["train"].indices)
    assert 43 in splits["train"].indices


def test_raw_preparation_validation_and_resampling(tmp_path):
    frame = sample(100)
    raw = frame.drop(columns="datetime")
    raw.insert(0, "Time", frame.datetime.dt.strftime("%H:%M:%S"))
    raw.insert(0, "Date", frame.datetime.dt.strftime("%d/%m/%Y"))
    raw.loc[2, "Voltage"] = np.nan
    source = tmp_path / "raw.txt"
    raw.to_csv(source, index=False, sep=";", na_rep="?")
    loaded = load_data(source)
    assert pd.isna(loaded.loc[2, "Voltage"])
    prepared = prepare_csv(source, tmp_path / "prepared.csv", frequency="2h")
    assert len(prepared) == 50
    assert prepared.iloc[0][FEATURES[0]] == pytest.approx(frame.iloc[:2][FEATURES[0]].mean())
    raw.loc[1, ["Date", "Time"]] = raw.loc[0, ["Date", "Time"]]
    raw.to_csv(source, index=False, sep=";")
    with pytest.raises(ValueError, match="Duplicate"):
        load_data(source)
    raw.loc[0, "Date"] = "garbage"
    raw.to_csv(source, index=False, sep=";")
    with pytest.raises(ValueError, match="timestamps"):
        load_data(source)


def test_report_features_require_real_holiday_labels():
    frame = sample()
    assert list(feature_frame(frame, "lstm_six")) == [name for name in FEATURES if name != "Voltage"]
    with pytest.raises(ValueError, match="holiday"):
        feature_frame(frame, "report_calendar")
    frame["holiday"] = 0
    features = feature_frame(frame, "report_calendar")
    assert list(features) == [FEATURES[0], "time_slot", "day_of_week", "holiday"]
    assert features.time_slot.max() == 47


@pytest.mark.parametrize("changes", [{"lookback": 0}, {"epochs": 0}, {"train_fraction": .9, "validation_fraction": .2},
                                    {"learning_rate": float("inf")}, {"lasso_alpha": float("nan")},
                                    {"models": ("lstm", "lstm")}, {"seed": -1}])
def test_invalid_config(changes):
    with pytest.raises(ValueError):
        replace(Config(), **changes).validate()


def test_metrics_finite_zero_and_constant_cases():
    result = metrics([0, 2, 4], [1, 1, 5])
    assert result["mape_n"] == 2
    assert result["mape_percent"] == 37.5
    assert result["wape_percent"] == 50
    assert metrics([1, 1], [1, 1])["r2"] is None
    with pytest.raises(ValueError, match="finite"):
        metrics([float("nan")], [0])


def test_empirical_baselines_and_mape_optimum():
    values = np.array([1, 2, 8, 9])
    optimum = mape_minimizer(values)
    losses = [np.mean(np.abs((values - candidate) / values)) for candidate in values]
    assert np.mean(np.abs((values - optimum) / values)) == min(losses)
    times = pd.to_datetime(["2023-01-02 10:00", "2023-01-03 10:00", "2023-01-07 10:00"])
    model = EmpiricalPredictor("empirical_mean").fit(times, [1, 3, 20])
    np.testing.assert_array_equal(model.predict(pd.to_datetime(["2023-01-04 10:00", "2023-01-08 10:00"])), [2, 20])
    assert model.predict(pd.to_datetime(["2023-01-08 11:00"]))[0] == 8


@pytest.mark.parametrize("name,widths", [("lstm", [200, 200]), ("gru", [75, 30, 30])])
def test_report_neural_architecture_and_gradient(name, widths):
    torch.set_num_threads(1)
    model = RecurrentForecaster(name, 7)
    assert [layer.hidden_size for layer in model.layers] == widths
    output = model(torch.randn(4, 3, 7))
    assert output.shape == (4,)
    output.square().mean().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_all_models_train_reload_and_infer(tmp_path):
    source = tmp_path / "data.csv"
    sample(180).to_csv(source, index=False)
    config = Config(models=MODELS, lookback=2, epochs=3, batch_size=16, seed=7)
    output = tmp_path / "run"
    scores = train(source, output, config)
    reloaded = evaluate(output, source)
    assert scores["test"] == reloaded
    assert set(reloaded) == set(MODELS)
    history = json.loads((output / "history.json").read_text())
    for name in ("lstm", "gru"):
        losses = history[name]["epochs"]
        assert losses[-1]["train_loss"] < losses[0]["train_loss"]
        assert 1 <= history[name]["best_epoch"] <= config.epochs
        assert reloaded[name]["n"] > 0
    forecast = predict(output, source)
    assert forecast["datetime"] == str(sample(181).datetime.iloc[-1])
    assert set(forecast["predictions"]) == set(MODELS)
    assert all(np.isfinite(value) for value in forecast["predictions"].values())
    # Fresh process reload uses identical persisted transforms/models.
    result = subprocess.run([sys.executable, "-m", "load_forecasting", "evaluate", "--run", str(output)],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == reloaded
    changed = sample(180)
    changed.loc[179, FEATURES[0]] += 100
    changed.to_csv(source, index=False)
    with pytest.raises(ValueError, match="differs"):
        evaluate(output, source)
    with pytest.raises(ValueError, match="not empty"):
        train(source, output, config)


def test_cli_prepare_train_evaluate_predict(tmp_path):
    source = tmp_path / "source.csv"
    sample(80).to_csv(source, index=False)
    prepared = tmp_path / "prepared.csv"
    config = tmp_path / "config.json"
    config.write_text(json.dumps(Config(models=("lstm", "gru", "ridge"), epochs=2).to_dict()))
    run = tmp_path / "run"
    commands = [
        ["prepare", "--input", str(source), "--output", str(prepared), "--target-unit", "source units"],
        ["train", "--data", str(prepared), "--config", str(config), "--output", str(run)],
        ["evaluate", "--run", str(run)],
        ["predict", "--run", str(run), "--history", str(prepared)],
    ]
    for args in commands:
        result = subprocess.run([sys.executable, "-m", "load_forecasting", *args], text=True,
                                capture_output=True, check=True)
        assert json.loads(result.stdout)
    assert Path(run / "lstm.pt").exists()


@pytest.mark.parametrize("mode,size", [("lstm_six", 6), ("report_calendar", 4)])
def test_report_feature_modes_train_save_load_infer(tmp_path, mode, size):
    source = tmp_path / "data.csv"
    frame = sample(80)
    if mode == "report_calendar":
        frame["holiday"] = (frame.datetime.dt.day == 2).astype(int)
        frame = frame[["datetime", FEATURES[0], "holiday"]]
    else:
        frame = frame.drop(columns="Voltage")
    frame.to_csv(source, index=False)
    output = tmp_path / "run"
    scores = train(source, output, Config(models=("lstm", "gru", "ridge"), feature_mode=mode, epochs=2))
    metadata = json.loads((output / "metadata.json").read_text())
    assert len(metadata["preprocessor"]["names"]) == size
    assert evaluate(output, source) == scores["test"]
    assert len(predict(output, source)["predictions"]) == 3
