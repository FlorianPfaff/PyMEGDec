"""Export exploratory alpha metrics for one participant."""

import argparse

from script_bootstrap import add_src_to_path

add_src_to_path(__file__)

from pymegdec.alpha_metrics import export_participant_alpha_metrics  # noqa: E402
from pymegdec.cli import (  # noqa: E402
    add_alpha_metric_arguments,
    alpha_metric_config_from_args,
)


def main():
    parser = argparse.ArgumentParser(
        description="Export exploratory prestimulus alpha metrics to CSV."
    )
    parser.add_argument(
        "--data-dir", default=None, help="Directory containing Part*Data.mat files."
    )
    parser.add_argument(
        "--participant", type=int, required=True, help="Participant id to export."
    )
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument(
        "--cue",
        action="store_true",
        help="Use Part*CueData.mat instead of Part*Data.mat.",
    )
    add_alpha_metric_arguments(parser)
    args = parser.parse_args()

    config = alpha_metric_config_from_args(args)
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
