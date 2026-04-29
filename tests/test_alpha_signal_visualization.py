import unittest
from unittest.mock import patch

import numpy as np

from pymegdec.alpha_signal import extract_phase, extract_time_basis, get_time_vector
from pymegdec.alpha_visualization import (
    calculate_phase_differences,
    extract_channels_by_location,
    show_bandpass_filtered_signals,
    visualize_phase_shifts,
)


def create_test_data():
    sampling_rate = 200
    time_vector = np.arange(0, 2, 1 / sampling_rate)
    n_trials = 2
    n_occipital_channels = 12
    n_channels = n_occipital_channels + 3

    labels = np.empty((n_channels, 1), dtype=object)
    for channel_idx in range(n_occipital_channels):
        labels[channel_idx, 0] = np.array([f"M.O{channel_idx:02d}"], dtype=object)
    for channel_idx in range(n_occipital_channels, n_channels):
        labels[channel_idx, 0] = np.array([f"M.T{channel_idx:02d}"], dtype=object)

    times = np.empty((1, n_trials), dtype=object)
    trials = np.empty((1, n_trials), dtype=object)
    for trial_idx in range(n_trials):
        times[0, trial_idx] = time_vector[None, :]
        trial = np.zeros((n_channels, time_vector.size))
        for channel_idx in range(n_channels):
            phase_shift = channel_idx * 0.05 + trial_idx * 0.1
            alpha_signal = np.sin(2 * np.pi * 10 * time_vector + phase_shift)
            slow_drift = 0.2 * np.sin(2 * np.pi * 2 * time_vector)
            trial[channel_idx, :] = alpha_signal + slow_drift
        trials[0, trial_idx] = trial

    data = np.empty(
        (1,),
        dtype=[
            ("label", "O"),
            ("trial", "O"),
            ("time", "O"),
            ("trialinfo", "O"),
        ],
    )
    data["label"][0] = labels
    data["trial"][0] = trials
    data["time"][0] = times
    data["trialinfo"][0] = np.array([[1, 2]])
    return data


class TestAlphaChannelMethods(unittest.TestCase):

    def setUp(self):
        self.data = create_test_data()

    def test_extract_channels_by_location(self):
        occipital_pattern = r"^M.O..$"
        occipital_indices = extract_channels_by_location(self.data, occipital_pattern)
        self.assertEqual(occipital_indices, list(range(12)))

    def test_extract_phase_keeps_signal_length(self):
        sampling_rate = 200
        time_vector = np.arange(0, 2, 1 / sampling_rate)
        signal = np.sin(2 * np.pi * 10 * time_vector)

        phase = extract_phase(signal, sampling_rate)

        self.assertEqual(phase.shape, signal.shape)
        self.assertTrue(np.all(np.isfinite(phase)))

    @patch("pymegdec.alpha_visualization.plt.show")
    def test_show_bandpass_filtered_signals(self, mock_show):
        trial_idx = 0
        time_window = (0, 1)
        location_pattern = r"^M.O..$"

        phases, _ = show_bandpass_filtered_signals(
            self.data, trial_idx, time_window, location_pattern
        )
        self.assertEqual(len(phases), 12)
        self.assertEqual(phases[0].shape, get_time_vector(self.data, trial_idx).shape)
        mock_show.assert_called_once()

    @patch("pymegdec.alpha_visualization.plt.show")
    def test_visualize_phase_shifts(self, mock_show):
        trial_idx = 0
        location_pattern = r"^M.O..$"

        self.assertIsNone(
            visualize_phase_shifts(self.data, trial_idx, location_pattern)
        )
        mock_show.assert_called_once()

    def test_calculate_phase_differences_wraps_angles(self):
        phases = [
            np.array([np.pi - 0.1, -np.pi + 0.1]),
            np.array([-np.pi + 0.1, np.pi - 0.1]),
        ]

        phase_diffs = calculate_phase_differences(phases)

        self.assertAlmostEqual(phase_diffs[0, 1], 0.2, places=6)

    def test_extract_time_basis(self):
        time_basis = extract_time_basis(self.data, trial_idx=0, channel_range=(0, 11))

        self.assertEqual(time_basis.shape, get_time_vector(self.data, 0).shape)
        self.assertTrue(np.all(np.isfinite(time_basis)))


if __name__ == "__main__":
    unittest.main()
