import numpy as np
import scipy.signal


def get_data_field(data, field_name):
    if isinstance(data, dict):
        return data[field_name]

    field = data[field_name]
    if isinstance(field, np.ndarray) and field.size == 1:
        return field.item()
    return field


def _cell_item(cell_array, index):
    values = np.asarray(cell_array, dtype=object)
    if values.ndim == 0:
        return values.item()
    if values.ndim == 2 and values.shape[0] == 1:
        return values[0, index]
    if values.ndim == 2 and values.shape[1] == 1:
        return values[index, 0]
    return values[index]


def get_time_vector(data, trial_idx=0):
    time_vector = _cell_item(get_data_field(data, "time"), trial_idx)
    return np.asarray(time_vector, dtype=float).ravel()


def get_trial_signal(data, trial_idx=0):
    trial_signal = _cell_item(get_data_field(data, "trial"), trial_idx)
    return np.asarray(trial_signal, dtype=float)


def bandpass_filter_signal(
    signal_values, sampling_rate, lowcut=8.0, highcut=12.0, order=5
):
    nyquist = 0.5 * sampling_rate
    if lowcut <= 0 or highcut <= 0:
        raise ValueError("Cutoff frequencies must be positive.")
    if lowcut >= highcut:
        raise ValueError("lowcut must be lower than highcut.")
    if highcut >= nyquist:
        raise ValueError("highcut must be lower than the Nyquist frequency.")

    sos = scipy.signal.butter(
        order,
        [lowcut, highcut],
        btype="bandpass",
        fs=sampling_rate,
        output="sos",
    )
    return scipy.signal.sosfiltfilt(sos, signal_values)


def extract_alpha_signal_and_phase(
    signal_values, sampling_rate, lowcut=8.0, highcut=12.0
):
    filtered_signal = bandpass_filter_signal(
        signal_values, sampling_rate, lowcut, highcut
    )
    analytic_signal = scipy.signal.hilbert(filtered_signal)
    return filtered_signal, np.angle(analytic_signal)


def extract_phase(signal_values, sampling_rate, lowcut=8.0, highcut=12.0):
    """
    Extracts the phase of the given signal using bandpass filtering and
    Hilbert transform.

    Parameters:
        signal_values (numpy array): The signal to extract the phase from.
        sampling_rate (float): The sampling rate of the signal.
        lowcut (float): The low cutoff frequency for the bandpass filter.
        highcut (float): The high cutoff frequency for the bandpass filter.

    Returns:
        numpy array: The phase of the filtered signal.
    """
    _, phase = extract_alpha_signal_and_phase(
        signal_values, sampling_rate, lowcut, highcut
    )
    return phase


def average_phases(phases):
    """
    Averages the phases across multiple channels.

    Parameters:
        phases (list of numpy arrays): List of phase arrays from different channels.

    Returns:
        numpy array: The average phase.
    """
    if not phases:
        raise ValueError("At least one phase array is required.")

    phase_matrix = np.vstack(phases)
    mean_phase = np.angle(np.mean(np.exp(1j * phase_matrix), axis=0))
    return mean_phase


def extract_time_basis(data, trial_idx=0, channel_range=(187, 198)):
    """
    Extracts a robust time basis based on the alpha phases across multiple
    channels for a given trial.

    Parameters:
        data (dict): The filtered MEG data containing only the alpha signal.
        trial_idx (int): The index of the trial to extract the time basis from.
        channel_range (tuple): The range of channels (start, end) to extract phases for.

    Returns:
        numpy array: The robust time basis based on the average phase.
    """
    time_vector = get_time_vector(data, trial_idx)
    signal = get_trial_signal(data, trial_idx)

    sampling_rate = 1 / np.diff(time_vector[:2])[0]

    phases = []
    for channel_idx in range(channel_range[0], channel_range[1] + 1):
        signal_curr_chan = signal[channel_idx, :]
        phase = extract_phase(signal_curr_chan, sampling_rate)
        phases.append(phase)

    mean_phase = average_phases(phases)
    return mean_phase


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import scipy.io as sio

    demo_data_folder = r"."
    demo_part = 2
    demo_data = sio.loadmat(f"{demo_data_folder}/Part{demo_part}Data.mat")["data"][0]

    # Extract the robust time basis for channels 187 to 198 for a specific trial
    demo_time_basis = extract_time_basis(
        demo_data, trial_idx=0, channel_range=(187, 198)
    )

    # Display the time basis
    print("Robust time basis (average phase):", demo_time_basis)

    # Plot the average phase to visualize
    demo_time_vector = get_time_vector(demo_data)
    plt.plot(demo_time_vector, demo_time_basis, label="Average Phase")
    plt.title("Average Alpha Phase Across Channels 187-198")
    plt.xlabel("Time (s)")
    plt.ylabel("Phase (radians)")
    plt.legend()
    plt.grid(True)
    plt.show()
