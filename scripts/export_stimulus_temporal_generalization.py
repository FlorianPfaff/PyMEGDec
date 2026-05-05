"""Export train-time/test-time stimulus temporal generalization."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from export_stimulus_predictions import (
    _float_or_inf,
    _int_or_inf,
    _normalize_argv,
    _parse_classifier_param,
    _transfer_participants,
)
from pymegdec.data_config import resolve_data_folder
from pymegdec.stimulus_decoding import (
    StimulusDecodingConfig,
    TRANSFER_DIRECTIONS,
    export_stimulus_temporal_generalization,
    window_centers_from_range,
)


def _parse_time_window(value: str) -> tuple[float, float]:
    parts = tuple(float(token.strip()) for token in value.split(",", 1))
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Time window must have the form start,stop.")
    if parts[0] > parts[1]:
        raise argparse.ArgumentTypeError("Time window start must be before stop.")
    return parts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export stimulus temporal generalization across train/test windows.")
    parser.add_argument("--data-dir", dest="data_folder", default=None, help="Directory containing Part*Data.mat and Part*CueData.mat files.")
    parser.add_argument("--participants", default=None, help="Participant ids such as 1-4,6,8. Defaults to all participants with main and cue files.")
    parser.add_argument("--time-window", type=_parse_time_window, default=(-0.4, 0.8), help="Window-center range as start,stop in seconds.")
    parser.add_argument("--window-step-s", type=float, default=0.025, help="Step between train/test window centers in seconds.")
    parser.add_argument("--window-size", type=float, default=0.1, help="Window size in seconds.")
    parser.add_argument("--null-window-center", type=_float_or_inf, default=float("nan"), help="Center of an optional pre-stimulus null window, or nan.")
    parser.add_argument("--transfer-direction", choices=TRANSFER_DIRECTIONS, default="main-to-cue", help="Train/validation dataset direction.")
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
    parser.add_argument("--chance-classes", type=int, default=16, help="Number of stimulus classes used for summary chance level.")
    parser.add_argument("--output", default="outputs/stimulus_temporal_generalization.csv", help="Output CSV with one row per participant/train-window/test-window.")
    parser.add_argument("--summary-output", default="outputs/stimulus_temporal_generalization_summary.csv", help="Output CSV summarized across participants by train/test window.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(_normalize_argv(argv))
    data_folder = resolve_data_folder(args.data_folder)
    participants = _transfer_participants(args.participants, data_folder)
    if not participants:
        parser.error("No participants found. Pass --participants or configure a data directory with matching main and cue MAT files.")

    config = StimulusDecodingConfig(
        window_centers=window_centers_from_range(args.time_window, args.window_step_s),
        window_size=args.window_size,
        null_window_center=args.null_window_center,
        new_framerate=args.new_framerate,
        classifier=args.classifier,
        classifier_param=_parse_classifier_param(args.classifier_param),
        components_pca=args.components_pca,
        frequency_range=tuple(args.frequency_range),
        chance_classes=args.chance_classes,
        permutations=0,
        transfer_direction=args.transfer_direction,
    )

    rows, summary_rows = export_stimulus_temporal_generalization(
        data_folder,
        participants,
        args.output,
        summary_output_path=args.summary_output,
        config=config,
        progress=lambda message: print(message, flush=True),
    )
    print(f"Wrote {len(rows)} participant/train/test rows to {args.output}")
    print(f"Wrote {len(summary_rows)} train/test summary rows to {args.summary_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
