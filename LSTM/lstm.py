"""Legacy worksheet LSTM experiment with chronological, leak-free evaluation.

This experiment uses the 48 sheets in Fri.xlsx and is separate from the
seven-variable report comparison in forecast.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler


LOOKBACK = 1
TRAIN_FRACTION = 0.48
TARGET_COLUMN = 18


def windows(data, lookback, target_column):
    x = np.stack([data[i - lookback:i] for i in range(lookback, len(data))])
    y = data[lookback:, target_column]
    return x, y, np.arange(lookback, len(data))


def scores(actual, predicted):
    nonzero = actual != 0
    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "mse": float(mean_squared_error(actual, predicted)),
        "mape_percent": float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) /
                                              actual[nonzero])) * 100) if nonzero.any() else None,
    }


def run(input_path=Path("Fri.xlsx"), output_path=Path("FriRNN1_generated.xlsx")):
    np.random.seed(7)
    tf.keras.utils.set_random_seed(7)
    with pd.ExcelWriter(output_path) as writer:
        for sheet_number in range(1, 49):
            sheet = f"Sheet{sheet_number}"
            values = pd.read_excel(input_path, sheet_name=sheet).to_numpy(dtype=float)
            if values.shape[1] <= TARGET_COLUMN:
                raise ValueError(f"{sheet} has fewer than 19 features")
            split = int(len(values) * TRAIN_FRACTION)
            if split <= LOOKBACK or split >= len(values):
                raise ValueError(f"{sheet} has too few rows")
            scaler = MinMaxScaler().fit(values[:split])
            scaled = scaler.transform(values)
            x, y, indices = windows(scaled, LOOKBACK, TARGET_COLUMN)
            train, test = indices < split, indices >= split
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(LOOKBACK, values.shape[1])),
                tf.keras.layers.LSTM(4),
                tf.keras.layers.Dense(1),
            ])
            model.compile(loss="mse", optimizer="adam")
            model.fit(x[train], y[train], epochs=200, batch_size=32, verbose=0)
            predictions = model.predict(x[test], verbose=0).reshape(-1)
            target_min = scaler.data_min_[TARGET_COLUMN]
            target_span = scaler.data_range_[TARGET_COLUMN]
            actual = y[test] * target_span + target_min
            predicted = predictions * target_span + target_min
            pd.DataFrame({"actual": actual, "prediction": predicted}).to_excel(
                writer, sheet_name=sheet, index=False
            )
            print(f"{sheet}: {scores(actual, predicted)}")


if __name__ == "__main__":
    run()
