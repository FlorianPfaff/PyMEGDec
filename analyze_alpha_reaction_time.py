"""Analyze exploratory alpha metrics against reaction time."""

import argparse

from script_bootstrap import add_src_to_path

add_src_to_path(__file__)

from pymegdec.alpha_metrics import (  # noqa: E402
    DEFAULT_FREQUENCY_RANGE,
    DEFAULT_OCCIPITAL_PATTERN,
    DEFAULT_TIME_WINDOW,
    AlphaMetricConfig,
)
from pymegdec.reaction_time_analysis import (  # noqa: E402
    DEFAULT_ALPHA_RT_METRICS,
    AlphaReactionTimeExportConfig,
    ReactionTimeCsvConfig,
    available_participants,
    export_alpha_reaction_time_analysis,
    parse_participant_spec,
    write_alpha_reaction_time_plots,
)


def _parse_range(value):
    lower, upper = value.split(",", maxsplit=1)
    return float(lower), float(upper)


def _participants(value, data_dir, cue):
    if value:
        return parse_participant_spec(value)
    return available_participants(data_dir, cue=cue)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze prestimulus alpha metrics against reaction time."
    )
    parser.add_argument(
        "--data-dir", default=None, help="Directory containing Part*Data.mat files."
    )
    parser.add_argument(
        "--participants",
        default=None,
        help="Participant ids such as 1-4,6,8. Defaults to all available MAT files.",
    )
    parser.add_argument(
        "--reaction-times",
        default=None,
        help="CSV containing participant, trial, and reaction_time columns.",
    )
    parser.add_argument(
        "--alpha-metrics", default=None, help="Optional precomputed alpha metrics CSV."
    )
    parser.add_argument(
        "--joined-output",
        required=True,
        help="Output CSV for matched trial-level alpha/RT rows.",
    )
    parser.add_argument(
        "--summary-output", required=True, help="Output CSV for association summaries."
    )
    parser.add_argument(
        "--plots-dir",
        default=None,
        help="Optional directory for simple alpha/RT scatter plots.",
    )
    parser.add_argument(
        "--cue",
        action="store_true",
        help="Use Part*CueData.mat instead of Part*Data.mat.",
    )
    parser.add_argument(
        "--trialinfo-rt-column",
        type=int,
        default=None,
        help="Zero-based trialinfo column containing RT when no CSV is supplied.",
    )
    parser.add_argument(
        "--reaction-time-scale",
        type=float,
        default=1.0,
        help="Scale applied to RT values, for example 0.001 for milliseconds.",
    )
    parser.add_argument(
        "--participant-column",
        default=None,
        help="Reaction-time CSV participant column override.",
    )
    parser.add_argument(
        "--trial-column", default=None, help="Reaction-time CSV trial column override."
    )
    parser.add_argument(
        "--reaction-time-column",
        default=None,
        help="Reaction-time CSV RT column override.",
    )
    parser.add_argument(
        "--dataset-column",
        default=None,
        help="Reaction-time CSV dataset column override.",
    )
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
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=DEFAULT_ALPHA_RT_METRICS,
        help="Alpha metrics to summarize.",
    )
    args = parser.parse_args()

    participants = _participants(args.participants, args.data_dir, args.cue)
    if not participants and (not args.alpha_metrics or not args.reaction_times):
        parser.error(
            "No participants found. Pass --participants or configure a data "
            "directory with matching MAT files."
        )
    default_participant = participants[0] if len(participants) == 1 else None
    alpha_config = AlphaMetricConfig(
        location_pattern=args.location_pattern,
        time_window=args.time_window,
        frequency_range=args.frequency_range,
    )
    csv_config = ReactionTimeCsvConfig(
        participant_column=args.participant_column,
        trial_column=args.trial_column,
        reaction_time_column=args.reaction_time_column,
        dataset_column=args.dataset_column,
        default_participant=default_participant,
        default_dataset="cue" if args.cue else "main",
        reaction_time_scale=args.reaction_time_scale,
    )
    export_config = AlphaReactionTimeExportConfig(
        reaction_times_path=args.reaction_times,
        alpha_metrics_path=args.alpha_metrics,
        joined_output_path=args.joined_output,
        summary_output_path=args.summary_output,
        cue=args.cue,
        alpha_config=alpha_config,
        csv_config=csv_config,
        trialinfo_rt_column=args.trialinfo_rt_column,
        metrics=tuple(args.metrics),
    )

    joined_rows, summary_rows = export_alpha_reaction_time_analysis(
        args.data_dir,
        participants,
        config=export_config,
    )
    if args.plots_dir:
        write_alpha_reaction_time_plots(
            joined_rows, args.plots_dir, metrics=args.metrics
        )
    print(f"Wrote {len(joined_rows)} matched trial rows to {args.joined_output}")
    print(f"Wrote {len(summary_rows)} summary rows to {args.summary_output}")


if __name__ == "__main__":
    main()
