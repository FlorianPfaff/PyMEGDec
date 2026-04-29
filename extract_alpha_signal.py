"""Backward-compatible wrapper for :mod:`pymegdec.alpha_signal`."""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

from pymegdec.alpha_signal import (  # noqa: E402
    average_phases,
    bandpass_filter_signal,
    extract_alpha_signal_and_phase,
    extract_phase,
    extract_time_basis,
    get_data_field,
    get_time_vector,
    get_trial_signal,
)

__all__ = [
    "average_phases",
    "bandpass_filter_signal",
    "extract_alpha_signal_and_phase",
    "extract_phase",
    "extract_time_basis",
    "get_data_field",
    "get_time_vector",
    "get_trial_signal",
]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import scipy.io as sio

    data_folder = r"."
    part = 2
    data = sio.loadmat(f"{data_folder}/Part{part}Data.mat")["data"][0]

    time_basis = extract_time_basis(data, trial_idx=0, channel_range=(187, 198))
    print("Robust time basis (average phase):", time_basis)

    time_vector = get_time_vector(data)
    plt.plot(time_vector, time_basis, label="Average Phase")
    plt.title("Average Alpha Phase Across Channels 187-198")
    plt.xlabel("Time (s)")
    plt.ylabel("Phase (radians)")
    plt.legend()
    plt.grid(True)
    plt.show()
