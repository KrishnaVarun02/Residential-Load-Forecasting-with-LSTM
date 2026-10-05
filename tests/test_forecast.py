import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from LSTM.forecast import FEATURES, load_data, metrics, prepare


class ForecastTests(unittest.TestCase):
    def test_chronological_split_and_train_only_scaling(self):
        frame = pd.DataFrame({name: np.arange(100, dtype=float) for name in FEATURES})
        splits, target_min, target_span = prepare(frame, lookback=3)
        train, validation, test = splits
        self.assertEqual([len(part[0]) for part in splits], [67, 15, 15])
        self.assertEqual((target_min, target_span), (0, 69))
        self.assertAlmostEqual(train[0][-1, -1, 0], 68 / 69)
        self.assertAlmostEqual(validation[0][0, -1, 0], 69 / 69)
        self.assertGreater(test[0][0, -1, 0], 1)

    def test_metrics_and_zero_actuals(self):
        result = metrics(np.array([0, 2, 4]), np.array([1, 1, 5]))
        self.assertAlmostEqual(result["mae"], 1)
        self.assertAlmostEqual(result["rmse"], 1)
        self.assertAlmostEqual(result["mape_percent"], 37.5)
        self.assertIsNone(metrics(np.zeros(2), np.zeros(2))["mape_percent"])

    def test_semicolon_input_and_forward_fill(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "household_power_consumption.csv"
            rows = [dict(Date="16/12/2006", Time="17:24:00", **{name: 1 for name in FEATURES}),
                    dict(Date="16/12/2006", Time="17:25:00", **{name: 2 for name in FEATURES})]
            rows[1]["Voltage"] = "?"
            pd.DataFrame(rows).to_csv(path, sep=";", index=False)
            frame = load_data(path)
            self.assertEqual(frame.loc[1, "Voltage"], 1)
            self.assertEqual(frame.loc[1, "Global_active_power"], 2)


if __name__ == "__main__":
    unittest.main()
