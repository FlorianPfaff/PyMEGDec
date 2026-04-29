import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pymegdec.alpha_metrics import (
    compute_alpha_metrics,
    get_channel_names,
    get_channel_positions,
    select_channels,
    write_alpha_metrics_csv,
)


def _cell_array(values):
    inner = np.empty((1, len(values)), dtype=object)
    for index, value in enumerate(values):
        inner[0, index] = value
    return inner


def _synthetic_data():
    sampling_rate = 200
    time = np.arange(-0.5, 1.0, 1 / sampling_rate)
    channel_names = ["MLO11", "MRO11", "MZO01", "MLF11"]
    positions = np.array(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    trials = []
    for trial_idx in range(2):
        trial = np.zeros((4, time.size))
        for channel_idx in range(4):
            phase = channel_idx * 0.25 + trial_idx * 0.1
            trial[channel_idx, :] = np.sin(2 * np.pi * 10 * time + phase)
        trials.append(trial)

    return {
        "label": np.array(channel_names, dtype=object)[:, None],
        "trial": _cell_array(trials),
        "time": _cell_array([time[None, :], time[None, :]]),
        "trialinfo": np.array([[1, 2]]),
        "grad": {"chanpos": positions},
    }


class TestAlphaMetrics(unittest.TestCase):
    def setUp(self):
        self.data = _synthetic_data()

    def test_select_channels_defaults_to_occipital_ctf_labels(self):
        self.assertEqual(
            get_channel_names(self.data), ["MLO11", "MRO11", "MZO01", "MLF11"]
        )
        self.assertEqual(select_channels(self.data), [0, 1, 2])
        np.testing.assert_allclose(
            get_channel_positions(self.data), self.data["grad"]["chanpos"]
        )

    def test_get_channel_positions_from_matlab_struct_array(self):
        positions = self.data["grad"]["chanpos"]
        grad = np.empty((1, 1), dtype=[("chanpos", "O")])
        grad["chanpos"][0, 0] = positions
        data = {**self.data, "grad": grad}

        np.testing.assert_allclose(get_channel_positions(data), positions)

    def test_compute_alpha_metrics_returns_one_row_per_trial(self):
        rows = compute_alpha_metrics(self.data)

        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["n_channels"], 3)
            self.assertGreater(row["alpha_power"], 0)
            self.assertTrue(np.isfinite(row["phase_plane_fit"]))
            self.assertTrue(np.isfinite(row["spatial_freq_rad_per_mm"]))
            self.assertIn("direction_rad", row)

    def test_write_alpha_metrics_csv(self):
        rows = compute_alpha_metrics(self.data)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "alpha_metrics.csv"
            write_alpha_metrics_csv(rows, output_path)

            with output_path.open(newline="", encoding="utf-8") as handle:
                loaded_rows = list(csv.DictReader(handle))

        self.assertEqual(len(loaded_rows), 2)
        self.assertEqual(loaded_rows[0]["dataset"], "main")


if __name__ == "__main__":
    unittest.main()
