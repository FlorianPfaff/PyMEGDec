"""Export exploratory alpha metrics for one participant."""

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

from pymegdec.alpha_metrics import (  # noqa: E402
    AlphaMetricConfig,
    DEFAULT_FREQUENCY_RANGE,
    DEFAULT_OCCIPITAL_PATTERN,
    DEFAULT_TIME_WINDOW,
    export_participant_alpha_metrics,
)


def _parse_range(value):
    lower, upper = value.split(",", maxsplit=1)
    return float(lower), float(upper)


def main():
    parser = argparse.ArgumentParser(
        description="Export exploratory prestimulus alpha metrics to CSV."
    )
    parser.add_argument("--data-dir", default=None, help="Directory containing Part*Data.mat files.")
    parser.add_argument("--participant", type=int, required=True, help="Participant id to export.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--cue", action="store_true", help="Use Part*CueData.mat instead of Part*Data.mat.")
    parser.add_argument(
        "--location-pattern",
        default=DEFAULT_OCCIPITAL_PATTERN,
        help="Regex for selecting channels by label.",
    )
    parser.add_argument(
        "--time-window",
        type=_parse_range,
        default=DEFAULT_TIME_WINDOW,
        help="Time window as start,stop in seconds.",
    )
    parser.add_argument(
        "--frequency-range",
        type=_parse_range,
        default=DEFAULT_FREQUENCY_RANGE,
        help="Frequency range as low,high in Hz.",
    )
    args = parser.parse_args()

    config = AlphaMetricConfig(
        location_pattern=args.location_pattern,
        time_window=args.time_window,
        frequency_range=args.frequency_range,
    )
    rows = export_participant_alpha_metrics(
        args.data_dir,
        args.participant,
        args.output,
        cue=args.cue,
        config=config,
    )
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
