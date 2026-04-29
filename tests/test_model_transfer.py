import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pymegdec.model_transfer import evaluate_model_transfer


def _cell_array(values):
    inner = np.empty((1, len(values)), dtype=object)
    for index, value in enumerate(values):
        inner[0, index] = value

    outer = np.empty((1,), dtype=object)
    outer[0] = inner
    return outer


def _mat_data_with_time(time):
    trialinfo = np.empty((1, 1), dtype=object)
    trialinfo[0, 0] = np.array([1, 2])
    return {
        "time": _cell_array([np.array([time])]),
        "trialinfo": trialinfo,
    }


class TestEvaluateModelTransfer(unittest.TestCase):
    def setUp(self) -> None:
        self.data_folder = r"."
        self.parts = 2
        required_files = [
            Path(self.data_folder) / f"Part{self.parts}Data.mat",
            Path(self.data_folder) / f"Part{self.parts}CueData.mat",
        ]
        missing_files = [path for path in required_files if not path.exists()]
        if missing_files:
            message = "Missing required test data file(s): " + ", ".join(
                str(path) for path in missing_files
            )
            if os.getenv("CI"):
                self.fail(message)
            self.skipTest(message)

        self.null_window_center = np.nan

    def test_evaluate_model_transfer_accuracy_svm(self):
        classifier = "multiclass-svm"
        accuracy = evaluate_model_transfer(
            self.data_folder,
            self.parts,
            null_window_center=self.null_window_center,
            classifier=classifier,
            components_pca=200,
        )

        self.assertGreaterEqual(accuracy, 0.25, "Accuracy should be at least 0.25")

    def test_evaluate_model_transfer_accuracy_scikit_mlp(self):
        classifier = "scikit-mlp"
        accuracy = evaluate_model_transfer(
            self.data_folder,
            self.parts,
            null_window_center=self.null_window_center,
            classifier=classifier,
            components_pca=200,
        )

        self.assertGreaterEqual(accuracy, 0.09, "Accuracy should be at least 0.09")

    def test_evaluate_model_transfer_accuracy_pytorch_mlp(self):
        classifier = "pytorch-mlp"
        accuracy = evaluate_model_transfer(
            self.data_folder,
            self.parts,
            null_window_center=self.null_window_center,
            classifier=classifier,
            components_pca=200,
        )

        self.assertGreaterEqual(accuracy, 0.15, "Accuracy should be at least 0.15")


class TestEvaluateModelTransferSynthetic(unittest.TestCase):
    def test_evaluate_model_transfer_rejects_sampling_rate_mismatch(self):
        train_data = _mat_data_with_time([0.0, 0.1])
        val_data = _mat_data_with_time([0.0, 0.2])

        with patch(
            "pymegdec.model_transfer.sio.loadmat",
            side_effect=[
                {"data": np.array([train_data], dtype=object)},
                {"data": np.array([val_data], dtype=object)},
            ],
        ):
            with self.assertRaisesRegex(ValueError, "Sampling rate"):
                evaluate_model_transfer("unused", 1)


if __name__ == "__main__":
    unittest.main()
