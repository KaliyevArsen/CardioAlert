import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description="CardioAlert educational research toolkit")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download-data", help="Download original public UCI data")
    download.add_argument("--data-dir", type=Path, default=Path("data"))
    training = commands.add_parser("train", help="Train and evaluate a fixed protocol")
    training.add_argument("--data-dir", type=Path, default=Path("data"))
    training.add_argument("--protocol", choices=["cleveland", "multisite300"], default="cleveland")
    training.add_argument("--search", choices=["full", "smoke"], default="full")
    training.add_argument("--jobs", type=int, default=2)
    training.add_argument("--output", type=Path)
    prediction = commands.add_parser("predict", help="Evaluate a fictional JSON example")
    prediction.add_argument("--model", type=Path, default=Path("artifacts/cleveland/model.joblib"))
    prediction.add_argument("--input", type=Path, required=True)
    commands.add_parser("bot", help="Start optional Telegram demo using environment variables")
    args = parser.parse_args(argv)
    try:
        if args.command == "download-data":
            from .data import download_data
            print(json.dumps(download_data(args.data_dir), indent=2))
        elif args.command == "train":
            from .training import train
            output = args.output or Path("artifacts") / args.protocol
            train(args.data_dir, output, args.protocol, args.search, args.jobs)
        elif args.command == "predict":
            from .predict import load_model, predict_record
            print(json.dumps(predict_record(load_model(args.model), json.loads(args.input.read_text())), indent=2))
        elif args.command == "bot":
            from .bot import run
            run()
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
