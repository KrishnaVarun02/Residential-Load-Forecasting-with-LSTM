"""Compatibility entry point for the maintained load_forecasting package.

Prefer ``python -m load_forecasting`` for configuration, saved evaluation and
inference. This retains the previous positional-input comparison command/API.
"""

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from load_forecasting.config import Config, MODELS
from load_forecasting.data import FEATURES, build_splits, load_data as read_data
from load_forecasting.metrics import metrics  # noqa: F401 - compatibility re-export
from load_forecasting.models import RecurrentForecaster
from load_forecasting.workflow import train, write_json


def load_data(path):
    """Legacy helper retains causal forward-fill; maintained API keeps labels missing."""
    frame = read_data(path)
    frame[list(FEATURES)] = frame[list(FEATURES)].ffill()
    return frame.dropna(subset=list(FEATURES)).reset_index(drop=True)


def prepare(frame, lookback=1):
    frame = frame.copy()
    if "datetime" not in frame:
        frame["datetime"] = pd.date_range("2000-01-01", periods=len(frame), freq="D")
    splits, preprocessor, _ = build_splits(frame, Config(lookback=lookback).validate())
    arrays = [(np.stack([part[i][0] for i in range(len(part))]), part.labels[part.indices])
              for part in splits.values()]
    return arrays, preprocessor.minimum[0], preprocessor.span[0]


def neural_model(name, shape):
    """Return the report architecture, now implemented with torch.nn.Module."""
    return RecurrentForecaster(name, shape[1])


def run(path, models, lookback, epochs, output):
    artifact_dir = output.with_suffix("").with_name(output.stem + "_models")
    config = Config(models=tuple(models), lookback=lookback, epochs=epochs,
                    target_unit="source units (daily sample: sum of minute kW samples)")
    scores = train(path, artifact_dir, config)
    write_json(output, {"artifact_directory": str(artifact_dir), "test_metrics": scores["test"]})
    return scores["test"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS[:6]))
    parser.add_argument("--lookback", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("forecast_metrics.json"))
    args = parser.parse_args()
    print(run(args.input, args.models, args.lookback, args.epochs, args.output))


if __name__ == "__main__":
    main()
