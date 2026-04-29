import os
import unittest
from pathlib import Path

import numpy as np

from pymegdec.classifiers import train_multiclass_classifier
from pymegdec.model_transfer import evaluate_model_transfer, get_original_feature_importance


class TestLinearSvmFeatures(unittest.TestCase):
    def test_multiclass_svm_uses_linear_kernel(self):
        features = np.array([[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        labels = np.array([0, 0, 1, 1])

        model = train_multiclass_classifier(features, labels, "multiclass-svm", 1.0)

        self.assertEqual(model[-1].kernel, "linear")
        self.assertTrue(hasattr(model[-1], "coef_"))

    def test_original_feature_importance_without_pca(self):
        class Model:
            coef_ = np.array([[2.0, 3.0]])

        np.testing.assert_allclose(get_original_feature_importance(Model()), [[2.0, 3.0]])

    def test_original_feature_importance_maps_pca_space(self):
        class Model:
            coef_ = np.array([[2.0, 3.0]])

        pca_components = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

        np.testing.assert_allclose(get_original_feature_importance(Model(), pca_components), [[2.0, 3.0, 0.0]])

    def test_original_feature_importance_uses_pipeline_scale(self):
        class Scaler:
            scale_ = np.array([2.0, 4.0])

        class Classifier:
            coef_ = np.array([[2.0, 8.0]])

        class Model:
            steps = [("standardscaler", Scaler()), ("svc", Classifier())]

        np.testing.assert_allclose(get_original_feature_importance(Model()), [[1.0, 2.0]])

    def test_original_feature_importance_requires_coefficients(self):
        with self.assertRaises(ValueError):
            get_original_feature_importance(object())


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


if __name__ == "__main__":
    unittest.main()
