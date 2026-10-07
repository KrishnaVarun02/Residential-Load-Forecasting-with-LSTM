"""Command-line interface; run python -m load_forecasting --help."""

import argparse
import json
from pathlib import Path

from .config import Config
from .data import fingerprint, prepare_csv
from .workflow import evaluate, predict, train, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare", help="Normalize raw/prepared data, optionally resampling means")
    preparation.add_argument("--input", type=Path, required=True)
    preparation.add_argument("--output", type=Path, required=True)
    preparation.add_argument("--frequency", help="Optional pandas frequency, e.g. h; averages each measurement")
    preparation.add_argument("--feature-mode", choices=["measured", "lstm_six", "report_calendar"], default="measured")
    preparation.add_argument("--target-unit", required=True, help="Source target units, e.g. kW; not inferred")
    training = commands.add_parser("train", help="Train all configured models and select neural epochs on validation")
    training.add_argument("--data", type=Path, required=True)
    training.add_argument("--config", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    evaluation = commands.add_parser("evaluate", help="Reload artifacts and evaluate original held-out split")
    evaluation.add_argument("--run", type=Path, required=True)
    evaluation.add_argument("--data", type=Path)
    evaluation.add_argument("--split", choices=["validation", "test"], default="test")
    evaluation.add_argument("--output", type=Path)
    inference = commands.add_parser("predict", help="Forecast the next timestamp from observed history")
    inference.add_argument("--run", type=Path, required=True)
    inference.add_argument("--history", type=Path, required=True)
    inference.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            if args.input.resolve() == args.output.resolve() or args.output.exists():
                raise ValueError("Choose a new preparation output path to preserve existing data")
            frame = prepare_csv(args.input, args.output, args.frequency, args.feature_mode)
            result = {"source": str(args.input.resolve()), "source_sha256": fingerprint(args.input),
                      "rows": len(frame), "target_unit": args.target_unit,
                      "aggregation": "mean" if args.frequency else "none", "frequency": args.frequency}
            write_json(args.output.with_suffix(".metadata.json"), result)
        elif args.command == "train":
            config = Config.read(args.config)
            sidecar = args.data.with_suffix(".metadata.json")
            if sidecar.exists() and json.loads(sidecar.read_text())["target_unit"] != config.target_unit:
                raise ValueError("Config target_unit differs from prepared-data metadata")
            result = train(args.data, args.output, config)
        elif args.command == "evaluate":
            result = evaluate(args.run, args.data, args.split)
        else:
            result = predict(args.run, args.history)
        if args.command in {"evaluate", "predict"} and args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, result)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, FileNotFoundError, TypeError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
