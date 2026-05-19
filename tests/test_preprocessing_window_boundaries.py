"""Regression tests for preprocessing window boundary conventions."""

import unittest

import numpy as np

from pymegdec.preprocessing import _nearest_window_slice, extract_windows


def _single_trial_data(trial, time):
    trials = np.empty((1,), dtype=object)
    trials[0] = np.asarray(trial, dtype=float)

    times = np.empty((1,), dtype=object)
    times[0] = np.asarray(time, dtype=float)[None]

    return {
        "trial": np.array([[trials]], dtype=object),
        "time": np.array([[times]], dtype=object),
    }


class WindowBoundaryTests(unittest.TestCase):
    def test_nearest_window_slice_excludes_exact_stop_sample(self):
        time = np.arange(10, dtype=float) / 10.0

        window_slice = _nearest_window_slice(time, (0.2, 0.5), 0, "train")

        self.assertEqual(window_slice.start, 2)
        self.assertEqual(window_slice.stop, 5)
        self.assertEqual(window_slice.stop - window_slice.start, 3)

    def test_extract_windows_uses_half_open_stop_boundary(self):
        time = np.arange(10, dtype=float) / 10.0
        trial = np.arange(10, dtype=float)[None]
        data = _single_trial_data(trial, time)

        features, null_features = extract_windows(data, (0.2, 0.5), (np.nan, np.nan))

        self.assertEqual(null_features, [])
        self.assertEqual(features[0].shape, (3, 1))
        np.testing.assert_array_equal(features[0].ravel(), np.array([2.0, 3.0, 4.0]))

    def test_null_window_matches_half_open_train_sample_count(self):
        time = np.arange(-5, 6, dtype=float) / 10.0
        trial = np.arange(time.size, dtype=float)[None]
        data = _single_trial_data(trial, time)

        train_features, null_features = extract_windows(data, (0.1, 0.4), (-0.4, -0.1))

        self.assertEqual(train_features[0].shape, null_features[0].shape)
        self.assertEqual(train_features[0].shape, (3, 1))


if __name__ == "__main__":
    unittest.main()
