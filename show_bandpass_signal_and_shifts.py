"""Backward-compatible wrapper for :mod:`pymegdec.alpha_visualization`."""

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

from pymegdec.alpha_visualization import (  # noqa: E402
    calculate_phase_differences,
    extract_channels_by_location,
    extract_phases_and_channels,
    plot_all_alpha_signals,
    plot_phase_differences,
    show_bandpass_filtered_signals,
    visualize_phase_shifts,
)

__all__ = [
    "calculate_phase_differences",
    "extract_channels_by_location",
    "extract_phases_and_channels",
    "plot_all_alpha_signals",
    "plot_phase_differences",
    "show_bandpass_filtered_signals",
    "visualize_phase_shifts",
]


if __name__ == "__main__":
    import scipy.io as sio

    trial_idx = 0
    time_window = (0, 1)
    data_folder = r"."
    part = 2
    location_pattern = r"^M.O..$"

    data = sio.loadmat(f"{data_folder}/Part{part}Data.mat")["data"][0]
    show_bandpass_filtered_signals(data, trial_idx, time_window, location_pattern)
    visualize_phase_shifts(data, trial_idx, location_pattern)
