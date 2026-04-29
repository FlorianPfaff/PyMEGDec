import unittest

import numpy as np

from pymegdec.data_config import resolve_data_folder
from pymegdec.model_transfer import evaluate_model_transfer


class TestEvaluateModelTransfer(unittest.TestCase):
    def setUp(self) -> None:
        self.parts = 2
        try:
            self.data_folder = resolve_data_folder(
                required=True,
                required_files=[
                    f"Part{self.parts}Data.mat",
                    f"Part{self.parts}CueData.mat",
                ],
            )
        except FileNotFoundError as exc:
            self.skipTest(str(exc))

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


if __name__ == "__main__":
    unittest.main()
