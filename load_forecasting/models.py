"""Report neural architectures and comparable lagged regression/baseline models."""

import numpy as np
from torch import nn
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


class RecurrentForecaster(nn.Module):
    def __init__(self, name, input_size):
        super().__init__()
        if name == "lstm":
            self.layers = nn.ModuleList([nn.LSTM(input_size, 200, batch_first=True),
                                         nn.LSTM(200, 200, batch_first=True)])
            self.dropouts = nn.ModuleList([nn.Dropout(0.2), nn.Dropout(0.3)])
            self.output = nn.Linear(200, 1)
        elif name == "gru":
            self.layers = nn.ModuleList([nn.GRU(input_size, 75, batch_first=True),
                                         nn.GRU(75, 30, batch_first=True),
                                         nn.GRU(30, 30, batch_first=True)])
            self.dropouts = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
            self.output = nn.Linear(30, 1)
        else:
            raise ValueError(f"Unknown recurrent architecture: {name}")

    def forward(self, inputs):
        value = inputs
        for layer, dropout in zip(self.layers, self.dropouts):
            value, _ = layer(value)
            value = dropout(value)
        return self.output(value[:, -1]).squeeze(-1)


def regression_model(name, config):
    constructors = {
        "linear": LinearRegression,
        "ridge": lambda: Ridge(alpha=config.ridge_alpha),
        "lasso": lambda: Lasso(alpha=config.lasso_alpha, max_iter=10000),
        "polynomial": lambda: make_pipeline(PolynomialFeatures(config.polynomial_degree, include_bias=False),
                                            LinearRegression()),
    }
    return constructors[name]()


def group_keys(timestamps):
    return [(stamp.hour * 60 + stamp.minute, int(stamp.dayofweek >= 5)) for stamp in timestamps]


def mape_minimizer(values):
    """Weighted median minimizes sample MAPE for nonzero observed loads."""
    values = np.asarray(values)
    values = np.sort(values[values != 0])
    if not len(values):
        return 0.0
    weights = 1 / np.abs(values)
    index = np.searchsorted(np.cumsum(weights), weights.sum() / 2)
    return float(values[index])


class EmpiricalPredictor:
    """Single-household time-of-day/weekday-type distribution, training rows only."""
    def __init__(self, name):
        self.name = name

    def fit(self, timestamps, targets):
        reducer = np.mean if self.name == "empirical_mean" else mape_minimizer
        groups = {}
        for key, target in zip(group_keys(timestamps), targets):
            groups.setdefault(key, []).append(target)
        self.groups = {key: float(reducer(values)) for key, values in groups.items()}
        self.fallback = float(reducer(targets))
        return self

    def predict(self, timestamps):
        return np.array([self.groups.get(key, self.fallback) for key in group_keys(timestamps)])
