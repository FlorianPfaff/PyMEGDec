"""Command-line entry points for PyMEGDec."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from typing import Sequence

import numpy as np

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
    parser.add_argument("--data-dir", dest="data_folder", default=None, help="Directory containing Part*Data.mat files.")
    parser.add_argument("--participant", type=int, default=2, help="Participant id to evaluate.")
    parser.add_argument("--window-size", type=float, default=0.1, help="Window size in seconds.")
    parser.add_argument("--train-window-center", type=float, default=0.2, help="Center of the stimulus training window.")
    parser.add_argument("--null-window-center", type=_float_or_inf, default=-0.2, help="Center of the null window, or nan.")
    parser.add_argument("--new-framerate", type=_float_or_inf, default=float("inf"), help="Target frame rate, or inf.")
    parser.add_argument("--classifier", default="multiclass-svm", help="Classifier name.")
    parser.add_argument("--classifier-param", default=None, help="Classifier parameter value, JSON, or Python literal.")
    parser.add_argument("--components-pca", type=_int_or_inf, default=100, help="Number of PCA components, or inf.")
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
    parser = argparse.ArgumentParser(prog=prog, description="Cross-validate one participant dataset.")
    _add_common_args(parser)
    parser.add_argument("--folds", type=int, default=10, help="Number of cross-validation folds.")
    return parser


def _build_transfer_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Evaluate model transfer for one participant.")
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


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="PyMEGDec command-line interface.")
    parser.add_argument("command", choices=["cross-validate", "transfer"], help="Workflow to run.")

    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return 0

    command, *remaining = argv
    if command == "cross-validate":
        return cross_validate(remaining, prog="pymegdec cross-validate")
    if command == "transfer":
        return transfer(remaining, prog="pymegdec transfer")
    parser.error(f"Unsupported command: {command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
