"""Export sensor-level alpha movement trajectories."""

import argparse

from script_bootstrap import add_src_to_path

add_src_to_path(__file__)

from pymegdec.alpha_movement import (  # noqa: E402
    DEFAULT_MOVEMENT_TIME_WINDOW,
    DEFAULT_SENSOR_PATTERN,
    DEFAULT_TRAJECTORY_STEP_S,
    AlphaMovementConfig,
    export_alpha_movement,
)
from pymegdec.cli import parse_range  # noqa: E402
from pymegdec.reaction_time_analysis import (  # noqa: E402
    available_participants,
    parse_participant_spec,
)


def _participants(value, data_dir, cue):
    if value:
        return parse_participant_spec(value)
    return available_participants(data_dir, cue=cue)


def main():
    parser = argparse.ArgumentParser(
        description=("Export sensor-level alpha movement trajectories. The trajectory is " "a MEG sensor-array proxy, not source-localized brain movement.")
    )
    parser.add_argument("--data-dir", default=None, help="Directory containing Part*Data.mat files.")
    parser.add_argument(
        "--participants",
        default=None,
        help="Participant ids such as 1-4,6,8. Defaults to all available MAT files.",
    )
    parser.add_argument(
        "--trajectory-output",
        required=True,
        help="Output CSV for trial/timepoint sensor-level trajectories.",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional output CSV averaged by participant, condition, and time.",
    )
    parser.add_argument(
        "--cue",
        action="store_true",
        help="Use Part*CueData.mat instead of Part*Data.mat.",
    )
    parser.add_argument(
        "--location-pattern",
        default=DEFAULT_SENSOR_PATTERN,
        help="Regex for selecting channels by label. Defaults to all MEG channels.",
    )
    parser.add_argument(
        "--time-window",
        type=parse_range,
        default=DEFAULT_MOVEMENT_TIME_WINDOW,
        help="Time window as start,stop in seconds.",
    )
    parser.add_argument(
        "--frequency-range",
        type=parse_range,
        default=(8.0, 12.0),
        help="Frequency range as low,high in Hz.",
    )
    parser.add_argument(
        "--trajectory-step-s",
        type=float,
        default=DEFAULT_TRAJECTORY_STEP_S,
        help="Trajectory sampling step in seconds.",
    )
    args = parser.parse_args()

    participants = _participants(args.participants, args.data_dir, args.cue)
    if not participants:
        parser.error("No participants found. Pass --participants or configure a data " "directory with matching MAT files.")

    config = AlphaMovementConfig(
        location_pattern=args.location_pattern,
        time_window=args.time_window,
        frequency_range=args.frequency_range,
        trajectory_step_s=args.trajectory_step_s,
    )
    rows, summary_rows = export_alpha_movement(
        args.data_dir,
        participants,
        args.trajectory_output,
        summary_output_path=args.summary_output,
        cue=args.cue,
        config=config,
    )
    print(f"Wrote {len(rows)} trajectory rows to {args.trajectory_output}")
    if args.summary_output:
        print(f"Wrote {len(summary_rows)} summary rows to {args.summary_output}")


if __name__ == "__main__":
    main()
