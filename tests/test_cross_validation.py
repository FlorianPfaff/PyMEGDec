import os
import unittest

import numpy as np

from pymegdec.cross_validation import cross_validate_single_dataset
from pymegdec.data_config import resolve_data_folder


class TestCrossValidateSingleDataset(unittest.TestCase):
    def setUp(self) -> None:
        participant_id = 2
        try:
            data_folder = resolve_data_folder(
                required=True,
                required_files=[f"Part{participant_id}Data.mat"],
            )
        except FileNotFoundError as exc:
            if os.getenv("CI"):
                self.fail(str(exc))
            self.skipTest(str(exc))

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
