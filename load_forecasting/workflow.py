"""Training, validation selection, saved-artifact evaluation and inference."""

from copy import deepcopy
import importlib.metadata
import json
from math import comb
from pathlib import Path
import random

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import Config
from .data import Preprocessor, build_splits, fingerprint, load_data
from .metrics import metrics
from .models import EmpiricalPredictor, RecurrentForecaster, regression_model


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def seed_everything(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)


def neural_predictions(model, windows, batch_size):
    model.eval()
    result = []
    with torch.no_grad():
        for inputs, _ in DataLoader(windows, batch_size=batch_size, shuffle=False):
            result.append(model(inputs).numpy())
    return np.concatenate(result)


def fit_neural(name, splits, config):
    model = RecurrentForecaster(name, splits["train"].values.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss() if name == "lstm" else nn.L1Loss()
    history, best, stale, best_state = [], float("inf"), 0, None
    for epoch in range(config.epochs):
        model.train()
        loss_sum = 0.0
        for inputs, labels in DataLoader(splits["train"], batch_size=config.batch_size, shuffle=False):
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss; inspect data and learning rate")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += loss.item() * len(inputs)
        prediction = neural_predictions(model, splits["validation"], config.batch_size)
        labels = splits["validation"].labels[splits["validation"].indices]
        validation_loss = float(np.mean((prediction - labels) ** 2 if name == "lstm" else np.abs(prediction - labels)))
        history.append({"epoch": epoch + 1, "train_loss": loss_sum / len(splits["train"]),
                        "validation_loss": validation_loss})
        if not np.isfinite(validation_loss):
            raise ValueError("Nonfinite validation loss")
        if validation_loss < best:
            best, stale, best_state = validation_loss, 0, deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"loss": "mse" if name == "lstm" else "mae", "epochs": history,
                   "best_epoch": min(history, key=lambda item: item["validation_loss"])["epoch"],
                   "parameter_count": sum(parameter.numel() for parameter in model.parameters())}


def predict_windows(name, model, windows, frame, preprocessor, config):
    if name in {"lstm", "gru"}:
        normalized = neural_predictions(model, windows, config.batch_size)
        return preprocessor.inverse_target(normalized)
    if name == "persistence":
        return preprocessor.inverse_target(windows.values[windows.indices - 1, 0])
    if name.startswith("empirical_"):
        return model.predict(frame.datetime.iloc[windows.indices])
    inputs, _ = windows.arrays()
    return preprocessor.inverse_target(model.predict(inputs))


def train(data_path: Path, output: Path, config: Config):
    config.validate()
    frame = load_data(data_path, config.feature_mode)
    splits, preprocessor, boundaries = build_splits(frame, config)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}; choose a new run directory")
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format_version": 1, "data_path": str(data_path.resolve()), "data_sha256": fingerprint(data_path),
        "config": config.to_dict(), "preprocessor": preprocessor.to_dict(), "boundaries": list(boundaries),
        "rows": len(frame), "target": "Global_active_power", "target_unit": config.target_unit,
        "protocol": "rolling observed-history one-step forecasts; chronological split by target time",
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "pandas", "scikit-learn", "joblib")},
        "splits": {name: {"n": len(part), "first_target": str(frame.datetime.iloc[part.indices[0]]),
                          "last_target": str(frame.datetime.iloc[part.indices[-1]])} for name, part in splits.items()},
    }
    scores = {"validation": {}, "test": {}}
    histories = {}
    predictions = {name: pd.DataFrame({"datetime": frame.datetime.iloc[part.indices].to_numpy(),
                                      "actual": preprocessor.inverse_target(part.labels[part.indices])})
                   for name, part in splits.items() if name != "train"}
    for name in config.models:
        seed_everything(config)
        if name in {"lstm", "gru"}:
            model, histories[name] = fit_neural(name, splits, config)
            torch.save(model.state_dict(), output / f"{name}.pt")
        elif name == "persistence":
            model = None
        elif name.startswith("empirical_"):
            windows = splits["train"]
            model = EmpiricalPredictor(name).fit(frame.datetime.iloc[windows.indices],
                                                preprocessor.inverse_target(windows.labels[windows.indices]))
            joblib.dump(model, output / f"{name}.joblib")
        else:
            inputs, labels = splits["train"].arrays()
            if name == "polynomial" and len(inputs) * comb(inputs.shape[1] + config.polynomial_degree, config.polynomial_degree) > 50_000_000:
                raise ValueError("Polynomial expansion exceeds 50M values; reduce lookback, degree or input rows")
            model = regression_model(name, config).fit(inputs, labels)
            joblib.dump(model, output / f"{name}.joblib")
        for split in scores:
            predicted = predict_windows(name, model, splits[split], frame, preprocessor, config)
            predictions[split][name] = predicted
            scores[split][name] = metrics(predictions[split].actual, predicted)
    write_json(output / "metadata.json", metadata)
    write_json(output / "metrics.json", scores)
    write_json(output / "history.json", histories)
    for name, prediction in predictions.items():
        prediction.to_csv(output / f"{name}_predictions.csv", index=False)
    return scores


def read_run(run_path):
    metadata = json.loads((run_path / "metadata.json").read_text())
    if metadata["format_version"] != 1:
        raise ValueError("Unsupported artifact format")
    config = Config.from_dict(metadata["config"])
    seed_everything(config)
    return metadata, config, Preprocessor.from_dict(metadata["preprocessor"])


def load_model(name, run_path, preprocessor):
    if name in {"lstm", "gru"}:
        model = RecurrentForecaster(name, len(preprocessor.names))
        model.load_state_dict(torch.load(run_path / f"{name}.pt", map_location="cpu", weights_only=True))
        return model.eval()
    if name == "persistence":
        return None
    return joblib.load(run_path / f"{name}.joblib")


def evaluate(run_path: Path, data_path: Path | None = None, split="test"):
    if split not in {"test", "validation"}:
        raise ValueError("Evaluation split must be test or validation")
    metadata, config, preprocessor = read_run(run_path)
    data_path = data_path or Path(metadata["data_path"])
    if fingerprint(data_path) != metadata["data_sha256"]:
        raise ValueError("Evaluation data differs from the saved chronological split; use the original file")
    frame = load_data(data_path, config.feature_mode)
    splits, _, _ = build_splits(frame, config, preprocessor, metadata["boundaries"])
    windows = splits[split]
    actual = preprocessor.inverse_target(windows.labels[windows.indices])
    return {name: metrics(actual, predict_windows(name, load_model(name, run_path, preprocessor),
                                                windows, frame, preprocessor, config)) for name in config.models}


def predict(run_path: Path, history_path: Path):
    """Predict one unseen next interval using only available history and saved state."""
    _, config, preprocessor = read_run(run_path)
    frame = load_data(history_path, config.feature_mode)
    if len(frame) < config.lookback:
        raise ValueError("History is shorter than the configured lookback")
    recent_times = frame.datetime.iloc[-config.lookback:].astype("int64").to_numpy()
    if len(recent_times) > 1 and (np.diff(recent_times) != preprocessor.cadence).any():
        raise ValueError("Inference history has a gap or different frequency")
    # Also validate the last interval for lookback=1 and reject obvious cadence mistakes.
    if frame.datetime.iloc[-1].value - frame.datetime.iloc[-2].value != preprocessor.cadence:
        raise ValueError("Inference history frequency differs from training")
    inputs = preprocessor.transform(frame)[-config.lookback:][None]
    timestamp = frame.datetime.iloc[-1] + pd.Timedelta(preprocessor.cadence, unit="ns")
    result = {"datetime": str(timestamp), "target_unit": config.target_unit, "predictions": {}}
    for name in config.models:
        model = load_model(name, run_path, preprocessor)
        if name in {"lstm", "gru"}:
            with torch.no_grad():
                value = preprocessor.inverse_target(model(torch.from_numpy(inputs)).numpy())[0]
        elif name == "persistence":
            value = preprocessor.inverse_target(inputs[0, -1, 0])
        elif name.startswith("empirical_"):
            value = model.predict([timestamp])[0]
        else:
            value = preprocessor.inverse_target(model.predict(inputs.reshape(1, -1)))[0]
        if not np.isfinite(value):
            raise ValueError(f"Nonfinite prediction from {name}")
        result["predictions"][name] = float(value)
    return result
