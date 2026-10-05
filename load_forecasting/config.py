"""Validated, serializable experiment settings."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

MODELS = ("lstm", "gru", "linear", "ridge", "lasso", "polynomial", "persistence", "empirical_mean", "empirical_mape")


@dataclass(frozen=True)
class Config:
    models: tuple[str, ...] = MODELS
    lookback: int = 1
    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.001
    patience: int = 5
    seed: int = 7
    threads: int = 1
    ridge_alpha: float = 0.0001
    lasso_alpha: float = 1.0
    polynomial_degree: int = 2
    feature_mode: str = "measured"
    target_unit: str = "source units"

    def validate(self):
        if not self.models or len(set(self.models)) != len(self.models) or set(self.models) - set(MODELS):
            raise ValueError(f"models must be unique choices from {MODELS}")
        for name in ("lookback", "epochs", "batch_size", "patience", "threads", "polynomial_degree"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not (0 < self.train_fraction < 1 and 0 < self.validation_fraction < 1
                and self.train_fraction + self.validation_fraction < 1):
            raise ValueError("train/validation fractions must leave a nonempty test fraction")
        if not all(math.isfinite(value) for value in (self.learning_rate, self.ridge_alpha, self.lasso_alpha)) or not self.learning_rate > 0 or not self.ridge_alpha >= 0 or not self.lasso_alpha >= 0:
            raise ValueError("learning rate must be positive; regularization must be nonnegative")
        if self.feature_mode not in {"measured", "lstm_six", "report_calendar"}:
            raise ValueError("feature_mode must be measured, lstm_six or report_calendar")
        if not isinstance(self.target_unit, str) or not self.target_unit.strip():
            raise ValueError("target_unit must describe the source units")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        if "models" in values:
            values["models"] = tuple(values["models"])
        return cls(**values).validate()

    @classmethod
    def read(cls, path: Path):
        return cls.from_dict(json.loads(path.read_text()))
