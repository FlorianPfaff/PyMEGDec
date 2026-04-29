"""Shared command-line helpers for PyMEGDec scripts."""

from pymegdec.alpha_metrics import (
    DEFAULT_FREQUENCY_RANGE,
    DEFAULT_OCCIPITAL_PATTERN,
    DEFAULT_TIME_WINDOW,
    AlphaMetricConfig,
)


def parse_range(value):
    """Parse a comma-separated numeric range."""

    lower, upper = value.split(",", maxsplit=1)
    return float(lower), float(upper)


def add_alpha_metric_arguments(parser):
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


def alpha_metric_config_from_args(args):
    """Build alpha metric config from parsed command-line arguments."""

    return AlphaMetricConfig(
        location_pattern=args.location_pattern,
        time_window=args.time_window,
        frequency_range=args.frequency_range,
    )
