"""Causal preprocessing, split boundaries and lazy forecast windows."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = (
    "Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity",
    "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
)
TARGET = FEATURES[0]


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_data(path: Path, feature_mode="measured") -> pd.DataFrame:
    """Read raw semicolon or prepared CSV without filling any target labels."""
    with path.open() as stream:
        separator = ";" if ";" in stream.readline() else ","
    raw = pd.read_csv(path, sep=separator, na_values=["?", "nan", ""])
    if {"Date", "Time"}.issubset(raw):
        timestamps = pd.to_datetime(raw.Date + " " + raw.Time, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    elif "datetime" in raw:
        timestamps = pd.to_datetime(raw.datetime, errors="coerce")
    else:
        raise ValueError("Expected Date/Time or datetime columns")
    if timestamps.isna().any():
        raise ValueError("Invalid or missing timestamps")
    required = ([TARGET] if feature_mode == "report_calendar" else
                [name for name in FEATURES if feature_mode != "lstm_six" or name != "Voltage"])
    missing = set(required) - set(raw)
    if missing:
        raise ValueError(f"Missing report features: {sorted(missing)}")
    names = required + (["holiday"] if "holiday" in raw else [])
    frame = raw[names].apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame.insert(0, "datetime", timestamps)
    frame = frame.sort_values("datetime", kind="stable").reset_index(drop=True)
    if frame.datetime.duplicated().any():
        raise ValueError("Duplicate timestamps are not supported")
    if len(frame) < 2:
        raise ValueError("At least two chronological rows are required")
    return frame


def feature_frame(frame, mode):
    if mode in {"measured", "lstm_six"}:
        names = [name for name in FEATURES if mode != "lstm_six" or name != "Voltage"]
        return frame.loc[:, names]
    if "holiday" not in frame or not frame.holiday.isin([0, 1]).all():
        raise ValueError("report_calendar requires an explicit 0/1 holiday column for every row")
    dates = frame.datetime.dt
    return pd.DataFrame({TARGET: frame[TARGET], "time_slot": dates.hour * 2 + dates.minute // 30 + 1,
                         "day_of_week": dates.dayofweek, "holiday": frame.holiday})


def cadence_ns(frame):
    differences = np.diff(frame.datetime.astype("int64").to_numpy())
    if not len(differences) or (differences <= 0).any():
        raise ValueError("Timestamps must increase strictly")
    return int(pd.Series(differences).mode().iloc[0])


@dataclass
class Preprocessor:
    names: list[str]
    fill: np.ndarray
    minimum: np.ndarray
    span: np.ndarray
    feature_mode: str
    cadence: int

    @classmethod
    def fit(cls, frame, train_end, mode):
        features = feature_frame(frame, mode)
        training = features.iloc[:train_end]
        fill = training.median().to_numpy(dtype=np.float64)
        if not np.isfinite(fill).all():
            raise ValueError("Each feature needs at least one finite training observation")
        values = training.ffill().fillna(dict(zip(features.columns, fill))).to_numpy(dtype=np.float64)
        minimum = values.min(axis=0)
        span = values.max(axis=0) - minimum
        span[span == 0] = 1.0
        return cls(list(features.columns), fill, minimum, span, mode, cadence_ns(frame.iloc[:train_end]))

    def transform(self, frame):
        features = feature_frame(frame, self.feature_mode)
        values = features.ffill().fillna(dict(zip(self.names, self.fill))).to_numpy(dtype=np.float64)
        scaled = ((values - self.minimum) / self.span).astype(np.float32)
        if not np.isfinite(scaled).all():
            raise ValueError("Nonfinite normalized features; inspect input magnitudes")
        return scaled

    def inverse_target(self, y):
        return np.asarray(y, dtype=np.float64) * self.span[0] + self.minimum[0]

    def to_dict(self):
        return {"names": self.names, "fill": self.fill.tolist(), "minimum": self.minimum.tolist(),
                "span": self.span.tolist(), "feature_mode": self.feature_mode, "cadence": self.cadence}

    @classmethod
    def from_dict(cls, data):
        return cls(data["names"], np.array(data["fill"]), np.array(data["minimum"]),
                   np.array(data["span"]), data["feature_mode"], data["cadence"])


class Windows:
    """Store O(rows*features) data; materialize only a minibatch for neural models."""

    def __init__(self, values, labels, indices, lookback):
        self.values, self.labels, self.indices, self.lookback = values, labels, indices, lookback

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        t = self.indices[index]
        return self.values[t - self.lookback:t], np.float32(self.labels[t])

    def arrays(self):
        if len(self) * self.lookback * self.values.shape[1] > 50_000_000:
            raise ValueError("Regression windows exceed 50M values; reduce lookback or input rows")
        return (np.stack([self[i][0].reshape(-1) for i in range(len(self))]), self.labels[self.indices])


def build_splits(frame, config, preprocessor=None, boundaries=None):
    n = len(frame)
    train_end, val_end = boundaries or (
        int(n * config.train_fraction), int(n * (config.train_fraction + config.validation_fraction)))
    if not config.lookback < train_end < val_end < n:
        raise ValueError("Need enough chronological rows for train, validation and test")
    preprocessor = preprocessor or Preprocessor.fit(frame, train_end, config.feature_mode)
    values = preprocessor.transform(frame)
    labels = (frame[TARGET].to_numpy(dtype=np.float64) - preprocessor.minimum[0]) / preprocessor.span[0]
    indices = np.arange(config.lookback, n)
    times = frame.datetime.astype("int64").to_numpy()
    # Reject any window spanning a timestamp gap, including its forecast interval.
    gaps = np.r_[0, np.cumsum(np.diff(times) != preprocessor.cadence)]
    valid = np.isfinite(labels[indices]) & (gaps[indices] == gaps[indices - config.lookback])
    indices = indices[valid]
    masks = {"train": indices < train_end, "validation": (indices >= train_end) & (indices < val_end),
             "test": indices >= val_end}
    splits = {name: Windows(values, labels, indices[mask], config.lookback) for name, mask in masks.items()}
    if any(not len(part) for part in splits.values()):
        raise ValueError("Each split needs a finite target and a window without timestamp gaps")
    return splits, preprocessor, (train_end, val_end)


def prepare_csv(source, output, frequency=None, feature_mode="measured"):
    """Normalize schema; optional mean aggregation, never automatic resampling."""
    frame = load_data(source, feature_mode)
    if frequency:
        if "holiday" in frame:
            raise ValueError("Aggregate before supplying holiday labels")
        frame = frame.set_index("datetime").resample(frequency).mean().reset_index()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame
