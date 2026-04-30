import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pymegdec.alpha_movement import (
    AlphaMovementConfig,
    compute_alpha_movement,
    sample_time_indices,
    summarize_alpha_movement,
    write_alpha_movement_csv,
)


def _cell_array(values):
    inner = np.empty((1, len(values)), dtype=object)
    for index, value in enumerate(values):
        inner[0, index] = value
    return inner


def _moving_alpha_data():
    sampling_rate = 200
    time = np.arange(-0.5, 1.0, 1 / sampling_rate)
    carrier = np.sin(2 * np.pi * 10 * time)
    progress = (time - time[0]) / (time[-1] - time[0])
    left_envelope = 1.6 - progress
    right_envelope = 0.2 + 1.4 * progress

    positions = np.array(
        [
            [-20.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [0.0, 20.0, 0.0],
            [0.0, -20.0, 0.0],
        ]
    )
    trial = np.vstack(
        [
            left_envelope * carrier,
            right_envelope * carrier,
            0.25 * carrier,
            0.25 * carrier,
        ]
    )

    return {
        "label": np.array(["MLO11", "MRO11", "MZO01", "MLF11"], dtype=object)[:, None],
        "trial": _cell_array([trial, trial]),
        "time": _cell_array([time[None, :], time[None, :]]),
        "trialinfo": np.array([[1, 2]]),
        "grad": {"chanpos": positions},
    }


class TestAlphaMovement(unittest.TestCase):
    def setUp(self):
        self.data = _moving_alpha_data()
        self.config = AlphaMovementConfig(
            time_window=(-0.2, 0.6),
            trajectory_step_s=0.1,
        )

    def test_sample_time_indices_returns_windowed_stride(self):
        time = np.arange(-0.5, 0.6, 0.1)

        indices = sample_time_indices(time, (-0.2, 0.2), 0.2)

        np.testing.assert_allclose(time[indices], [-0.2, 0.0, 0.2], atol=1e-12)

    def test_compute_alpha_movement_tracks_weighted_centroid(self):
        rows = compute_alpha_movement(self.data, participant_id=2, config=self.config)

        self.assertGreater(len(rows), 10)
        first_trial_rows = [row for row in rows if row["trial"] == 0]
        self.assertLess(
            first_trial_rows[0]["centroid_x_mm"],
            first_trial_rows[-1]["centroid_x_mm"],
        )
        self.assertEqual(first_trial_rows[0]["participant"], 2)
        self.assertEqual(first_trial_rows[0]["trial_label"], 1)
        self.assertTrue(np.isnan(first_trial_rows[0]["speed_mm_per_s"]))
        self.assertTrue(np.isfinite(first_trial_rows[-1]["speed_mm_per_s"]))
        self.assertIn(first_trial_rows[-1]["peak_channel_name"], {"MLO11", "MRO11"})

    def test_summarize_alpha_movement_groups_by_condition_and_time(self):
        rows = compute_alpha_movement(self.data, participant_id=2, config=self.config)

        summary_rows = summarize_alpha_movement(rows)

        labels = {row["trial_label"] for row in summary_rows}
        self.assertEqual(labels, {"1", "2"})
        self.assertTrue(all(row["n_trials"] == 1 for row in summary_rows))

    def test_write_alpha_movement_csv(self):
        rows = compute_alpha_movement(self.data, participant_id=2, config=self.config)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "movement.csv"
            write_alpha_movement_csv(rows, output_path)

            with output_path.open(newline="", encoding="utf-8") as handle:
                loaded_rows = list(csv.DictReader(handle))

        self.assertEqual(len(loaded_rows), len(rows))
        self.assertEqual(loaded_rows[0]["dataset"], "main")


if __name__ == "__main__":
    unittest.main()
