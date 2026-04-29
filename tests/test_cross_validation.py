import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pymegdec.cross_validation import cross_validate_single_dataset


def _cell_array(values):
    inner = np.empty((1, len(values)), dtype=object)
    for index, value in enumerate(values):
        inner[0, index] = value

    outer = np.empty((1,), dtype=object)
    outer[0] = inner
    return outer


def _mat_data(labels):
    trialinfo = np.empty((1, 1), dtype=object)
    trialinfo[0, 0] = labels
    return {
        "trial": _cell_array([np.zeros((1, 2)) for _ in labels]),
        "trialinfo": trialinfo,
    }


class _ConstantClassifier:
    def __init__(self, label):
        self.label = label

    def predict(self, features):
        return np.full(features.shape[0], self.label)


class TestCrossValidateSingleDataset(unittest.TestCase):
    def setUp(self) -> None:
        data_folder = r"."
        participant_id = 2
        data_file = Path(data_folder) / f"Part{participant_id}Data.mat"
        if not data_file.exists():
            if os.getenv("CI"):
                self.fail(f"Missing required test data file: {data_file}")
            self.skipTest(f"Missing required test data file: {data_file}")

        self.params = {
            "data_folder": data_folder,
            "participant_id": participant_id,
            "n_folds": 10,
            "window_size": 0.1,
            "train_window_center": 0.2,
            "null_window_center": -0.2,
            "new_framerate": float("inf"),
            "classifier": "multiclass-svm",
            "classifier_param": np.nan,
            "components_pca": 200,
            "frequency_range": (0, float("inf")),
        }

    def _accuracy(self, classifier):
        return cross_validate_single_dataset(
            **{
                **self.params,
                "classifier": classifier,
            }
        )

    def test_cross_validate_single_dataset_accuracy_svm(self):
        accuracy = self._accuracy("multiclass-svm")

        self.assertGreaterEqual(accuracy, 0.25, "Accuracy should be at least 0.25")

    def test_cross_validate_single_dataset_accuracy_scikit_mlp(self):
        accuracy = self._accuracy("scikit-mlp")

        self.assertGreaterEqual(accuracy, 0.15, "Accuracy should be at least 0.15")


class TestCrossValidateSingleDatasetSynthetic(unittest.TestCase):
    def test_cross_validate_single_dataset_without_null_window(self):
        labels = np.array([1, 2, 1, 2])
        stimuli_features = [
            np.array([[index], [index + 1]], dtype=float)
            for index in range(len(labels))
        ]

        with (
            patch(
                "pymegdec.cross_validation.sio.loadmat",
                return_value={"data": np.array([_mat_data(labels)], dtype=object)},
            ),
            patch(
                "pymegdec.cross_validation.preprocess_features",
                return_value=(stimuli_features, []),
            ),
            patch(
                "pymegdec.cross_validation.train_multiclass_classifier",
                return_value=_ConstantClassifier(1),
            ),
        ):
            accuracy = cross_validate_single_dataset(
                "unused",
                1,
                n_folds=2,
                null_window_center=np.nan,
                components_pca=float("inf"),
            )

        self.assertEqual(accuracy, 0.5)


if __name__ == "__main__":
    unittest.main()
