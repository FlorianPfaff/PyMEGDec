"""Export trial-level train-main/validate-cue stimulus predictions."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from typing import Sequence

import numpy as np

from pymegdec.alpha_metrics import write_alpha_metrics_csv
from pymegdec.data_config import resolve_data_folder
from pymegdec.reaction_time_analysis import available_participants, parse_participant_spec
from pymegdec.stimulus_decoding import (
    StimulusDecodingConfig,
    evaluate_participant_stimulus_decoding_diagnostics,
    summarize_stimulus_decoding,
    summarize_stimulus_prediction_diagnostics,
)

DEFAULT_WINDOW_CENTERS = (-0.175, 0.175)


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


def _parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(token.strip()) for token in value.split(",") if token.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one value is required.")
    return values


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
        raise argparse.ArgumentTypeError("classifier parameters must be numeric, JSON, or a Python literal") from exc


def _transfer_participants(participant_spec: str | None, data_folder) -> list[int]:
    if participant_spec:
        return parse_participant_spec(participant_spec)
    main_participants = set(available_participants(data_folder, cue=False))
    cue_participants = set(available_participants(data_folder, cue=True))
    return sorted(main_participants & cue_participants)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export trial-level stimulus predictions for selected windows.")
    parser.add_argument("--data-dir", dest="data_folder", default=None, help="Directory containing Part*Data.mat and Part*CueData.mat files.")
    parser.add_argument("--participants", default=None, help="Participant ids such as 1-4,6,8. Defaults to all participants with main and cue files.")
    parser.add_argument("--window-centers", type=_parse_float_list, default=DEFAULT_WINDOW_CENTERS, help="Comma-separated window centers in seconds.")
    parser.add_argument("--window-size", type=float, default=0.1, help="Window size in seconds.")
    parser.add_argument("--null-window-center", type=_float_or_inf, default=float("nan"), help="Center of an optional pre-stimulus null window, or nan.")
    parser.add_argument("--new-framerate", type=_float_or_inf, default=float("inf"), help="Target frame rate, or inf.")
    parser.add_argument("--classifier", default="multiclass-svm", help="Classifier name.")
    parser.add_argument("--classifier-param", default=None, help="Classifier parameter value, JSON, Python literal, numeric value, or nan.")
    parser.add_argument("--components-pca", type=_int_or_inf, default=100, help="Number of PCA components, or inf.")
    parser.add_argument(
        "--frequency-range",
        type=_float_or_inf,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=(0.0, float("inf")),
        help="Frequency range in Hz.",
    )
    parser.add_argument("--chance-classes", type=int, default=16, help="Number of stimulus classes used for the chance line.")
    parser.add_argument("--output", default="outputs/stimulus_predictions.csv", help="Output CSV with one row per validation trial and window.")
    parser.add_argument("--summary-output", default="outputs/stimulus_prediction_summary.csv", help="Optional participant/window accuracy summary CSV.")
    parser.add_argument("--accuracy-output", default=None, help="Optional participant/window accuracy CSV.")
    parser.add_argument("--confusion-output", default=None, help="Optional confusion-count CSV.")
    parser.add_argument("--per-stimulus-output", default=None, help="Optional per-stimulus recall CSV.")
    return parser


def _normalize_argv(argv: Sequence[str] | None) -> list[str] | None:
    if argv is None:
        argv = sys.argv[1:]
    normalized: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--window-centers" and index + 1 < len(argv):
            normalized.append(f"--window-centers={argv[index + 1]}")
            index += 2
            continue
        normalized.append(token)
        index += 1
    return normalized


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(_normalize_argv(argv))
    data_folder = resolve_data_folder(args.data_folder)
    participants = _transfer_participants(args.participants, data_folder)
    if not participants:
        parser.error("No participants found. Pass --participants or configure a data directory with matching main and cue MAT files.")

    config = StimulusDecodingConfig(
        window_centers=args.window_centers,
        window_size=args.window_size,
        null_window_center=args.null_window_center,
        new_framerate=args.new_framerate,
        classifier=args.classifier,
        classifier_param=_parse_classifier_param(args.classifier_param),
        components_pca=args.components_pca,
        frequency_range=tuple(args.frequency_range),
        chance_classes=args.chance_classes,
        permutations=0,
    )

    accuracy_rows = []
    prediction_rows = []
    for participant in participants:
        print(f"START participant={participant}", flush=True)
        participant_accuracy, participant_predictions = evaluate_participant_stimulus_decoding_diagnostics(
            data_folder,
            participant,
            config=config,
            diagnostic_window_centers=args.window_centers,
        )
        accuracy_rows.extend(participant_accuracy)
        prediction_rows.extend(participant_predictions)
        print(f"DONE participant={participant}", flush=True)

    write_alpha_metrics_csv(prediction_rows, args.output)
    print(f"Wrote {len(prediction_rows)} trial prediction rows to {args.output}")

    summary_rows = summarize_stimulus_decoding(accuracy_rows)
    if args.summary_output:
        write_alpha_metrics_csv(summary_rows, args.summary_output)
        print(f"Wrote {len(summary_rows)} summary rows to {args.summary_output}")
    if args.accuracy_output:
        write_alpha_metrics_csv(accuracy_rows, args.accuracy_output)
        print(f"Wrote {len(accuracy_rows)} participant/window rows to {args.accuracy_output}")
    if args.confusion_output or args.per_stimulus_output:
        confusion_rows, per_stimulus_rows = summarize_stimulus_prediction_diagnostics(prediction_rows)
        if args.confusion_output:
            write_alpha_metrics_csv(confusion_rows, args.confusion_output)
            print(f"Wrote {len(confusion_rows)} confusion rows to {args.confusion_output}")
        if args.per_stimulus_output:
            write_alpha_metrics_csv(per_stimulus_rows, args.per_stimulus_output)
            print(f"Wrote {len(per_stimulus_rows)} per-stimulus rows to {args.per_stimulus_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
