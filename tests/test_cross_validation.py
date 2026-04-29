import os
import unittest
from pathlib import Path

import numpy as np

from pymegdec.cross_validation import cross_validate_single_dataset


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


if __name__ == "__main__":
    unittest.main()
