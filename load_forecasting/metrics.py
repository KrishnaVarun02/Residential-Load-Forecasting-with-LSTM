"""Metrics in original target units, with defined zero/constant-target behavior."""

import numpy as np


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    if len(actual) != len(predicted) or not len(actual):
        raise ValueError("Metrics require equally sized nonempty arrays")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require finite values")
    error = actual - predicted
    denominator = np.sum((actual - actual.mean()) ** 2)
    nonzero = actual != 0
    absolute_sum = np.abs(actual).sum()
    return {
        "n": len(actual), "mse": float(np.mean(error**2)), "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "mape_percent": float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100) if nonzero.any() else None,
        "mape_n": int(nonzero.sum()),
        "wape_percent": float(np.abs(error).sum() / absolute_sum * 100) if absolute_sum else None,
        "r2": float(1 - np.sum(error**2) / denominator) if denominator else None,
        "explained_variance": float(1 - np.var(error) / np.var(actual)) if denominator else None,
    }
