"""Chronological one-step household load forecasts for the project report.

The report's LSTM, GRU and regression architectures are implemented here. The
published scores are historical results, not expected outputs of this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = (
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
)


def load_data(path: Path) -> pd.DataFrame:
    """Read either the original minute data or the committed daily sample."""
    with path.open() as stream:
        separator = ";" if ";" in stream.readline() else ","
    raw = pd.read_csv(path, sep=separator, na_values="?")
    if {"Date", "Time"}.issubset(raw):
        timestamps = pd.to_datetime(
            raw["Date"] + " " + raw["Time"], dayfirst=True, errors="coerce"
        )
    elif "datetime" in raw:
        timestamps = pd.to_datetime(raw["datetime"], errors="coerce")
    else:
        raise ValueError("Expected Date/Time or datetime columns")
    missing = set(FEATURES) - set(raw)
    if missing:
        raise ValueError(f"Missing report features: {sorted(missing)}")
    frame = raw.loc[:, list(FEATURES)].apply(pd.to_numeric, errors="coerce")
    frame.insert(0, "datetime", timestamps)
    frame = frame.dropna(subset=["datetime"]).sort_values("datetime")
    if frame.datetime.duplicated().any():
        raise ValueError("Duplicate timestamps are not supported")
    # Forward filling uses only earlier observations and cannot leak future data.
    frame.loc[:, list(FEATURES)] = frame.loc[:, list(FEATURES)].ffill()
    return frame.dropna(subset=FEATURES).reset_index(drop=True)


def prepare(frame: pd.DataFrame, lookback: int = 1):
    """Fit scaling on training rows; assign windows by their target time."""
    if lookback < 1:
        raise ValueError("lookback must be positive")
    values = frame.loc[:, list(FEATURES)].to_numpy(dtype=np.float64)
    n = len(values)
    train_end, val_end = int(n * 0.70), int(n * 0.85)
    if train_end <= lookback or val_end <= train_end or val_end >= n:
        raise ValueError("Need enough chronological rows for train, validation and test")
    lo = values[:train_end].min(axis=0)
    span = values[:train_end].max(axis=0) - lo
    span[span == 0] = 1.0
    scaled = (values - lo) / span
    x = np.stack([scaled[t - lookback : t] for t in range(lookback, n)])
    y = scaled[lookback:, 0]
    target_indices = np.arange(lookback, n)
    masks = (
        target_indices < train_end,
        (target_indices >= train_end) & (target_indices < val_end),
        target_indices >= val_end,
    )
    splits = [(x[m], y[m]) for m in masks]
    if any(len(part[0]) == 0 for part in splits):
        raise ValueError("Each split needs at least one forecast window")
    return splits, lo[0], span[0]


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    if len(actual) != len(predicted) or not len(actual):
        raise ValueError("Metrics require equally sized nonempty arrays")
    error = actual - predicted
    denominator = np.sum((actual - actual.mean()) ** 2)
    nonzero = actual != 0
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "mape_percent": float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100)
        if nonzero.any() else None,
        "r2": float(1 - np.sum(error**2) / denominator) if denominator else None,
    }


def neural_model(name: str, shape: tuple[int, int]):
    import tensorflow as tf

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=shape))
    if name == "lstm":
        model.add(tf.keras.layers.LSTM(200, return_sequences=True))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.LSTM(200))
        model.add(tf.keras.layers.Dropout(0.3))
        loss = "mse"
    elif name == "gru":
        model.add(tf.keras.layers.GRU(75, return_sequences=True))
        model.add(tf.keras.layers.GRU(30, return_sequences=True))
        model.add(tf.keras.layers.GRU(30))
        loss = "mae"
    else:
        raise ValueError(f"Unknown neural model: {name}")
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer="adam", loss=loss)
    return model


def run(path: Path, models: list[str], lookback: int, epochs: int, output: Path):
    splits, target_min, target_span = prepare(load_data(path), lookback)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = splits
    results = {}
    for name in models:
        if name in {"lstm", "gru"}:
            import tensorflow as tf

            tf.keras.utils.set_random_seed(7)
            model = neural_model(name, x_train.shape[1:])
            model.fit(x_train, y_train, validation_data=(x_val, y_val),
                      epochs=epochs, batch_size=64, shuffle=False, verbose=0)
            prediction = model.predict(x_test, verbose=0).reshape(-1)
        else:
            from sklearn.linear_model import Lasso, LinearRegression, Ridge
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import PolynomialFeatures

            estimators = {
                "linear": LinearRegression,
                "ridge": Ridge,
                "lasso": Lasso,
                "polynomial": lambda: make_pipeline(
                    PolynomialFeatures(degree=2, include_bias=False), LinearRegression()
                ),
            }
            model = estimators[name]()
            model.fit(x_train.reshape(len(x_train), -1), y_train)
            prediction = model.predict(x_test.reshape(len(x_test), -1))
        # The target occupies feature zero, so only its training scale is needed.
        results[name] = metrics(y_test * target_span + target_min,
                                prediction * target_span + target_min)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"input": str(path), "lookback": lookback,
                                  "split": "chronological 70/15/15",
                                  "test_metrics": results}, indent=2) + "\n")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="raw household file or household_daily.csv")
    parser.add_argument("--models", nargs="+", choices=("lstm", "gru", "linear", "ridge", "lasso", "polynomial"), default=["lstm", "gru", "linear", "ridge", "lasso", "polynomial"])
    parser.add_argument("--lookback", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("forecast_metrics.json"))
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.models, args.lookback, args.epochs, args.output), indent=2))


if __name__ == "__main__":
    main()
