import re

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from pymegdec.alpha_signal import (
    extract_alpha_signal_and_phase,
    get_data_field,
    get_time_vector,
    get_trial_signal,
)


def _label_to_string(label):
    value = label
    while isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        value = value.flat[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _get_channel_names(data):
    labels = np.asarray(get_data_field(data, "label"), dtype=object)
    if labels.ndim == 0:
        labels = np.asarray([labels.item()], dtype=object)
    return [_label_to_string(label) for label in labels.ravel()]


def extract_channels_by_location(data, location_pattern):
    """
    Extracts channel indices by location from the data.

    Parameters:
        data (dict): The filtered MEG data containing only the alpha signal.
        location_pattern (str): The regex pattern for the desired location.

    Returns:
        list: List of indices of the channels matching the location pattern.
    """
    pattern = re.compile(location_pattern)
    channel_indices = [index for index, channel_name in enumerate(_get_channel_names(data)) if pattern.match(channel_name)]

    return channel_indices


def plot_all_alpha_signals(time_vector, signals, channel_indices, time_window):
    """
    Plots the alpha signals for all specified channels in a single plot.

    Parameters:
        time_vector (np.ndarray): The time vector.
        signals (list of np.ndarray): The alpha signals for all channels.
        channel_indices (list): List of indices of the specified channels.
        time_window (tuple): The time window (start, end) to visualize in seconds.
    """
    plt.figure()
    for idx, signal in zip(channel_indices, signals):
        plt.plot(time_vector, signal, label=f"Channel {idx}")

    plt.title("Alpha Signals (8-12 Hz) - Selected Channels")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.xlim(time_window)
    plt.legend()
    plt.tight_layout()
    plt.show()


def calculate_phase_differences(phases):
    """
    Calculates phase differences between channels.

    Parameters:
        phases (list): List of phases for each channel.

    Returns:
        np.ndarray: Matrix of phase differences between channels.
    """
    num_channels = len(phases)
    phase_diffs = np.zeros((num_channels, num_channels))

    for i in range(num_channels):
        for j in range(num_channels):
            phase_delta = np.angle(np.exp(1j * (phases[i] - phases[j])))
            phase_diffs[i, j] = np.mean(np.abs(phase_delta))

    return phase_diffs


def plot_phase_differences(phase_diffs, channel_indices):
    """
    Plots a confusion matrix-like plot for phase differences between channels.

    Parameters:
        phase_diffs (np.ndarray): Matrix of phase differences between channels.
        channel_indices (list): List of indices of the specified channels.
    """
    num_channels = len(channel_indices)
    plt.figure(figsize=(8, 6))
    plt.imshow(phase_diffs, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Phase Difference (radians)")
    plt.title("Phase Differences Between Selected Channels")
    plt.xlabel("Channel Index")
    plt.ylabel("Channel Index")
    plt.xticks(ticks=np.arange(num_channels), labels=channel_indices, rotation=90)
    plt.yticks(ticks=np.arange(num_channels), labels=channel_indices)
    plt.show()


def extract_phases_and_channels(data, trial_idx, location_pattern):
    """
    Extracts the phases and channel indices for a given trial and location pattern.

    Parameters:
        data (dict): The filtered MEG data containing only the alpha signal.
        trial_idx (int): The index of the trial to visualize.
        location_pattern (str): The regex pattern for the desired location channels.

    Returns:
        tuple: Phases and channel indices.
    """
    time_vector = get_time_vector(data, trial_idx)
    signal = get_trial_signal(data, trial_idx)

    sampling_rate = 1 / np.diff(time_vector[:2])[0]
    channel_indices = extract_channels_by_location(data, location_pattern)
    if not channel_indices:
        raise ValueError(f"No channels matched pattern: {location_pattern}")

    phases = []
    signals = []
    for channel_idx in channel_indices:
        signal_curr_chan = signal[channel_idx, :]
        filtered_signal, phase = extract_alpha_signal_and_phase(signal_curr_chan, sampling_rate)
        phases.append(phase)
        signals.append(filtered_signal)

    return phases, channel_indices, time_vector, signals


def show_bandpass_filtered_signals(data, trial_idx=0, time_window=(0, 1), location_pattern=r"^M.O..$"):
    """
    Visualizes the alpha signal for a specific trial and extracts the phase.

    Parameters:
        data (dict): The filtered MEG data containing only the alpha signal.
        trial_idx (int): The index of the trial to visualize.
        time_window (tuple): The time window (start, end) to visualize in seconds.
        location_pattern (str): The regex pattern for the desired location channels.
    """
    phases, channel_indices, time_vector, signals = extract_phases_and_channels(data, trial_idx, location_pattern)
    plot_all_alpha_signals(time_vector, signals, channel_indices, time_window)
    return phases, channel_indices


def visualize_phase_shifts(data, trial_idx=0, location_pattern=r"^M.O..$"):
    """
    Visualizes phase shifts between specified channels.

    Parameters:
        data (dict): The filtered MEG data containing only the alpha signal.
        trial_idx (int): The index of the trial to visualize.
        location_pattern (str): The regex pattern for the desired location channels.
    """
    phases, channel_indices, _, _ = extract_phases_and_channels(data, trial_idx, location_pattern)
    phase_diffs = calculate_phase_differences(phases)
    plot_phase_differences(phase_diffs, channel_indices)


if __name__ == "__main__":
    demo_trial_idx = 0
    demo_time_window = (0, 1)
    demo_data_folder = r"."
    demo_part = 2
    demo_location_pattern = r"^M.O..$"

    demo_data = sio.loadmat(f"{demo_data_folder}/Part{demo_part}Data.mat")["data"][0]
    show_bandpass_filtered_signals(
        demo_data,
        demo_trial_idx,
        demo_time_window,
        demo_location_pattern,
    )
    visualize_phase_shifts(demo_data, demo_trial_idx, demo_location_pattern)
