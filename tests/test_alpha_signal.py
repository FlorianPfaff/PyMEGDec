import unittest

import numpy as np
from pymegdec.alpha_signal import (
    average_phases,
    bandpass_filter_signal,
    extract_time_basis,
    get_time_vector,
    sampling_rate_from_time_vector,
)
from tests.matlab_fixtures import cell_array


def _alpha_signal_fixture(time, signal):
    return {
        "time": cell_array([np.asarray(time, dtype=float)]),
        "trial": cell_array([np.asarray(signal, dtype=float)]),
    }


class TestAlphaSignalValidation(unittest.TestCase):
    def test_get_time_vector_unwraps_nested_matlab_cell(self):
        data = _alpha_signal_fixture([0.0, 0.1, 0.2], np.zeros((2, 3)))

        np.testing.assert_allclose(get_time_vector(data), [0.0, 0.1, 0.2])

    def test_sampling_rate_from_time_vector_validates_uniform_axis(self):
        self.assertAlmostEqual(sampling_rate_from_time_vector([0.0, 0.01, 0.02]), 100.0)

        with self.assertRaisesRegex(ValueError, "uniformly sampled"):
            sampling_rate_from_time_vector([0.0, 0.01, 0.03])

    def test_sampling_rate_from_time_vector_rejects_non_increasing_axis(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            sampling_rate_from_time_vector([0.0, 0.0, 0.01])

    def test_sampling_rate_from_time_vector_rejects_nonfinite_axis(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            sampling_rate_from_time_vector([0.0, np.nan, 0.02])

    def test_bandpass_filter_signal_rejects_invalid_sampling_rate(self):
        with self.assertRaisesRegex(ValueError, "positive finite"):
            bandpass_filter_signal(np.ones(100), np.nan)

    def test_extract_time_basis_rejects_signal_time_length_mismatch(self):
        time = np.arange(100, dtype=float) / 100.0
        signal = np.zeros((3, 99), dtype=float)
        data = _alpha_signal_fixture(time, signal)

        with self.assertRaisesRegex(ValueError, "samples but its time vector"):
            extract_time_basis(data, channel_range=(0, 1))

    def test_extract_time_basis_rejects_out_of_range_channel_range(self):
        time = np.arange(100, dtype=float) / 100.0
        signal = np.zeros((3, time.size), dtype=float)
        data = _alpha_signal_fixture(time, signal)

        with self.assertRaisesRegex(ValueError, "outside the available channels"):
            extract_time_basis(data, channel_range=(0, 3))

    def test_extract_time_basis_returns_mean_phase_for_valid_fixture(self):
        sampling_rate = 100.0
        time = np.arange(200, dtype=float) / sampling_rate
        signal = np.vstack(
            [
                np.sin(2 * np.pi * 10.0 * time),
                np.sin(2 * np.pi * 10.0 * time + 0.1),
                np.sin(2 * np.pi * 10.0 * time + 0.2),
            ]
        )
        data = _alpha_signal_fixture(time, signal)

        phase = extract_time_basis(data, channel_range=(0, 2))

        self.assertEqual(phase.shape, time.shape)
        self.assertTrue(np.all(np.isfinite(phase)))

    def test_average_phases_rejects_empty_input(self):
        with self.assertRaisesRegex(ValueError, "At least one"):
            average_phases([])


if __name__ == "__main__":
    unittest.main()
