import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.io as sio

from pymegdec.cross_validation import cross_validate_single_dataset
from pymegdec.model_transfer import evaluate_model_transfer


def _synthetic_part_data(labels):
    labels = np.asarray(labels, dtype=int)
    time_vector = np.linspace(-0.5, 0.5, 101)
    stim_mask = (time_vector >= 0.15) & (time_vector <= 0.25)
    null_mask = (time_vector >= -0.25) & (time_vector <= -0.15)

    trials = np.empty((1, labels.size), dtype=object)
    times = np.empty((1, labels.size), dtype=object)
    for trial_index, label in enumerate(labels):
        signal = 6.0 if label == 1 else -6.0
        trial = np.zeros((2, time_vector.size), dtype=float)
        trial[:, stim_mask] = signal
        trial[:, null_mask] = 0.0
        trial += 0.01 * (trial_index + 1)
        trials[0, trial_index] = trial
        times[0, trial_index] = time_vector[None, :]

    return {
        "trial": trials,
        "time": times,
        "trialinfo": labels[None, :],
    }


def _write_synthetic_part_data(data_dir, file_name, labels):
    sio.savemat(
        data_dir / file_name,
        {"data": _synthetic_part_data(labels)},
    )


class TestSyntheticData(unittest.TestCase):
    def test_cross_validate_single_dataset_runs_without_private_data(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            _write_synthetic_part_data(
                data_dir, "Part2Data.mat", [1, 2, 1, 2, 1, 2, 1, 2]
            )

            accuracy = cross_validate_single_dataset(
                data_dir,
                participant_id=2,
                n_folds=4,
                classifier="multiclass-svm",
                components_pca=float("inf"),
            )

        self.assertGreaterEqual(accuracy, 0.75)

    def test_evaluate_model_transfer_runs_without_private_data(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            _write_synthetic_part_data(
                data_dir, "Part2Data.mat", [1, 2, 1, 2, 1, 2, 1, 2]
            )
            _write_synthetic_part_data(data_dir, "Part2CueData.mat", [1, 2, 1, 2])

            accuracy = evaluate_model_transfer(
                data_dir,
                parts=2,
                null_window_center=np.nan,
                classifier="multiclass-svm",
                components_pca=float("inf"),
            )

        self.assertEqual(accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
