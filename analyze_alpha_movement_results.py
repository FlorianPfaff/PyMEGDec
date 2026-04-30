"""Analyze exported sensor-level alpha movement summaries."""

from script_bootstrap import add_src_to_path

add_src_to_path(__file__)

from pymegdec.cli import alpha_movement_results  # noqa: E402


def main():
    return alpha_movement_results()


if __name__ == "__main__":
    raise SystemExit(main())
