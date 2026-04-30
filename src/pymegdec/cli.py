"""Command-line entry points and shared CLI helpers for PyMEGDec."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from typing import Sequence

import numpy as np

from .alpha_metrics import (
    DEFAULT_FREQUENCY_RANGE,
    DEFAULT_OCCIPITAL_PATTERN,
    DEFAULT_TIME_WINDOW,
    AlphaMetricConfig,
)
from .alpha_movement_analysis import (
    DEFAULT_POST_WINDOW,
    DEFAULT_PRE_WINDOW,
    AlphaMovementAnalysisConfig,
    export_alpha_movement_analysis,
)
from .cross_validation import cross_validate_single_dataset
from .model_transfer import evaluate_model_transfer


def _float_or_inf(value: str) -> float:
    normalized = value.lower()
    if normalized in {"inf", "+inf", "infinity", "+infinity"}:
        return float("inf")
    if normalized in {"nan", "+nan", "-nan"}:
        return float("nan")
    return float(value)


def _int_or_inf(value: str) -> int | float:
    parsed = _float_or_inf(value)
    if np.isinf(parsed):
        return parsed
    return int(parsed)


def _parse_classifier_param(value: str | None):
    if value is None:
        return np.nan

    normalized = value.strip()
    if normalized.lower() == "nan":
        return np.nan

    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(normalized)
        except (SyntaxError, ValueError, TypeError, json.JSONDecodeError):
            pass

    try:
        return float(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "classifier parameters must be numeric, JSON, or a Python literal"
        ) from exc


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-dir",
        dest="data_folder",
        default=None,
        help="Directory containing Part*Data.mat files.",
    )
    parser.add_argument(
        "--participant", type=int, default=2, help="Participant id to evaluate."
    )
    parser.add_argument(
        "--window-size", type=float, default=0.1, help="Window size in seconds."
    )
    parser.add_argument(
        "--train-window-center",
        type=float,
        default=0.2,
        help="Center of the stimulus training window.",
    )
    parser.add_argument(
        "--null-window-center",
        type=_float_or_inf,
        default=-0.2,
        help="Center of the null window, or nan.",
    )
    parser.add_argument(
        "--new-framerate",
        type=_float_or_inf,
        default=float("inf"),
        help="Target frame rate, or inf.",
    )
    parser.add_argument(
        "--classifier", default="multiclass-svm", help="Classifier name."
    )
    parser.add_argument(
        "--classifier-param",
        default=None,
        help="Classifier parameter value, JSON, or Python literal.",
    )
    parser.add_argument(
        "--components-pca",
        type=_int_or_inf,
        default=100,
        help="Number of PCA components, or inf.",
    )
    parser.add_argument(
        "--frequency-range",
        type=_float_or_inf,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=(0, float("inf")),
        help="Frequency range in Hz.",
    )


def _common_kwargs(args: argparse.Namespace) -> dict:
    return {
        "window_size": args.window_size,
        "train_window_center": args.train_window_center,
        "null_window_center": args.null_window_center,
        "new_framerate": args.new_framerate,
        "classifier": args.classifier,
        "classifier_param": _parse_classifier_param(args.classifier_param),
        "components_pca": args.components_pca,
        "frequency_range": tuple(args.frequency_range),
    }


def _build_cross_validate_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog, description="Cross-validate one participant dataset."
    )
    _add_common_args(parser)
    parser.add_argument(
        "--folds", type=int, default=10, help="Number of cross-validation folds."
    )
    return parser


def _build_transfer_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog, description="Evaluate model transfer for one participant."
    )
    _add_common_args(parser)
    return parser


def cross_validate(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = _build_cross_validate_parser(prog=prog)
    args = parser.parse_args(argv)
    accuracy = cross_validate_single_dataset(
        args.data_folder,
        args.participant,
        n_folds=args.folds,
        **_common_kwargs(args),
    )
    print(accuracy)
    return 0


def transfer(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = _build_transfer_parser(prog=prog)
    args = parser.parse_args(argv)
    accuracy = evaluate_model_transfer(
        args.data_folder,
        args.participant,
        **_common_kwargs(args),
    )
    print(accuracy)
    return 0


def _build_alpha_movement_results_parser(
    prog: str | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog, description="Analyze exported alpha movement summaries."
    )
    parser.add_argument(
        "--movement-summary",
        required=True,
        help="Input CSV from analyze_alpha_movement.py --summary-output.",
    )
    parser.add_argument(
        "--effect-output",
        required=True,
        help="Output CSV for participant/condition pre-post effects.",
    )
    parser.add_argument(
        "--condition-summary-output",
        required=True,
        help="Output CSV for condition-level effect summaries.",
    )
    parser.add_argument(
        "--plots-dir",
        default=None,
        help="Optional output directory for condition-level PNG plots.",
    )
    parser.add_argument(
        "--pre-window",
        type=parse_range,
        default=DEFAULT_PRE_WINDOW,
        help="Pre-stimulus window as start,stop in seconds.",
    )
    parser.add_argument(
        "--post-window",
        type=parse_range,
        default=DEFAULT_POST_WINDOW,
        help="Post-stimulus window as start,stop in seconds.",
    )
    parser.add_argument(
        "--plot-labels",
        nargs="*",
        default=None,
        help="Optional condition labels to include in plots.",
    )
    return parser


def alpha_movement_results(
    argv: Sequence[str] | None = None, prog: str | None = None
) -> int:
    parser = _build_alpha_movement_results_parser(prog=prog)
    args = parser.parse_args(argv)
    config = AlphaMovementAnalysisConfig(
        pre_window=args.pre_window,
        post_window=args.post_window,
        plot_labels=(
            None
            if args.plot_labels is None
            else tuple(str(label) for label in args.plot_labels)
        ),
    )
    effect_rows, summary_rows = export_alpha_movement_analysis(
        args.movement_summary,
        args.effect_output,
        args.condition_summary_output,
        plots_dir=args.plots_dir,
        config=config,
    )
    print(
        f"Wrote {len(effect_rows)} participant-condition rows to {args.effect_output}"
    )
    print(
        f"Wrote {len(summary_rows)} condition summary rows to "
        f"{args.condition_summary_output}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="PyMEGDec command-line interface.")
    parser.add_argument(
        "command",
        choices=["cross-validate", "transfer", "alpha-movement-results"],
        help="Workflow to run.",
    )

    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return 0

    command, *remaining = argv
    if command == "cross-validate":
        return cross_validate(remaining, prog="pymegdec cross-validate")
    if command == "transfer":
        return transfer(remaining, prog="pymegdec transfer")
    if command == "alpha-movement-results":
        return alpha_movement_results(remaining, prog="pymegdec alpha-movement-results")
    parser.error(f"Unsupported command: {command}")
    return 2


def parse_range(value: str) -> tuple[float, float]:
    """Parse a comma-separated numeric range."""

    lower, upper = value.split(",", maxsplit=1)
    return float(lower), float(upper)


def add_alpha_metric_arguments(parser: argparse.ArgumentParser) -> None:
    """Add alpha metric extraction options to an argument parser."""

    parser.add_argument(
        "--location-pattern",
        default=DEFAULT_OCCIPITAL_PATTERN,
        help="Regex for selecting channels by label.",
    )
    parser.add_argument(
        "--time-window",
        type=parse_range,
        default=DEFAULT_TIME_WINDOW,
        help="Time window as start,stop in seconds.",
    )
    parser.add_argument(
        "--frequency-range",
        type=parse_range,
        default=DEFAULT_FREQUENCY_RANGE,
        help="Frequency range as low,high in Hz.",
    )


def alpha_metric_config_from_args(args: argparse.Namespace) -> AlphaMetricConfig:
    """Build alpha metric config from parsed command-line arguments."""

    return AlphaMetricConfig(
        location_pattern=args.location_pattern,
        time_window=args.time_window,
        frequency_range=args.frequency_range,
    )


if __name__ == "__main__":
    raise SystemExit(main())
