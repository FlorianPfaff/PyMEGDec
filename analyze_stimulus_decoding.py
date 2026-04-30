"""Run time-resolved stimulus decoding."""

from script_bootstrap import add_src_to_path

add_src_to_path(__file__)

from pymegdec.cli import stimulus_decoding  # noqa: E402


def main():
    return stimulus_decoding()


if __name__ == "__main__":
    raise SystemExit(main())
