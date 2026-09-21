"""Command-line interface for configured BoostDropout experiments."""

import argparse
import json
from pathlib import Path

import yaml

from .configuration import load_experiment_config
from .experiments import evaluate_run, run_experiment, run_grid_search


def _write_history_svg(run_dir: Path) -> Path:
    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    values = history["val_loss"]
    width, height, padding = 720, 360, 40
    maximum, minimum = max(values), min(values)
    span = maximum - minimum or 1.0
    points = " ".join(
        f"{padding + index * (width - 2 * padding) / max(len(values) - 1, 1):.1f},"
        f"{height - padding - (value - minimum) / span * (height - 2 * padding):.1f}"
        for index, value in enumerate(values)
    )
    output = run_dir / "validation-loss.svg"
    output.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<rect width="100%" height="100%" fill="white"/>'
        f'<text x="{padding}" y="24" font-family="sans-serif">Validation loss</text>'
        f'<polyline fill="none" stroke="#2563eb" stroke-width="3" points="{points}"/>'
        "</svg>\n",
        encoding="utf-8",
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="boostdropout",
        description="Reproducible BoostDropout runs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("train", "Run one YAML experiment configuration."),
        ("grid-search", "Run model parameter combinations declared in YAML."),
        ("evaluate", "Evaluate a saved run with its YAML configuration."),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--config", required=True, type=Path)
        if name == "evaluate":
            command.add_argument("--run-dir", required=True, type=Path)
    figures = subparsers.add_parser(
        "reproduce-figures",
        help="Recreate the portable validation-loss figure.",
    )
    figures.add_argument("--run-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "reproduce-figures":
        print(_write_history_svg(args.run_dir))
        return 0
    config = load_experiment_config(args.config)
    if args.command == "train":
        print(run_experiment(config))
    elif args.command == "evaluate":
        print(json.dumps(evaluate_run(args.run_dir, config), sort_keys=True))
    else:
        payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        print("\n".join(map(str, run_grid_search(config, payload.get("grid", {})))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
