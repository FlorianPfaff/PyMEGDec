import os
import subprocess
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from pymegdec.evaluate_model_transfer import (
    CLASSIFIER_REGISTRY,
    should_use_default_classifier_param,
    train_multiclass_classifier,
)


class TestClassifierRegistry(unittest.TestCase):
    def setUp(self) -> None:
        self.features = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.2],
                [1.0, 1.0],
                [1.0, 1.2],
                [2.0, 2.0],
                [2.0, 2.2],
            ]
        )
        self.labels = np.array([0, 0, 1, 1, 2, 2])

    def test_registry_contains_supported_classifiers(self):
        self.assertEqual(
            {
                'always1Dummy',
                'gradient-boosting',
                'knn',
                'mostFrequentDummy',
                'multiclass-svm',
                'pytorch-mlp',
                'random-forest',
                'scikit-mlp',
                'xgboost',
            },
            set(CLASSIFIER_REGISTRY),
        )

    def test_registry_trains_fast_sklearn_classifiers(self):
        classifier_params = {
            'always1Dummy': None,
            'gradient-boosting': 5,
            'knn': 1,
            'mostFrequentDummy': None,
            'multiclass-svm': 1.0,
            'random-forest': 5,
            'scikit-mlp': (5, 50),
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for classifier, classifier_param in classifier_params.items():
                with self.subTest(classifier=classifier):
                    model = train_multiclass_classifier(
                        self.features,
                        self.labels,
                        classifier,
                        classifier_param,
                        random_state=13,
                    )
                    predictions = model.predict(self.features)
                    self.assertEqual(len(predictions), len(self.labels))

    def test_random_state_reproduces_stochastic_classifier_predictions(self):
        model_a = train_multiclass_classifier(
            self.features, self.labels, 'random-forest', 5, random_state=7)
        model_b = train_multiclass_classifier(
            self.features, self.labels, 'random-forest', 5, random_state=7)

        np.testing.assert_array_equal(
            model_a.predict(self.features),
            model_b.predict(self.features),
        )

    def test_default_classifier_param_detection_handles_non_numeric_values(self):
        self.assertTrue(should_use_default_classifier_param(np.nan))
        self.assertFalse(should_use_default_classifier_param(None))
        self.assertFalse(should_use_default_classifier_param({'hidden_dim': 10}))

    def test_optional_ml_dependencies_are_lazy_imported(self):
        env = os.environ.copy()
        src_path = str(Path(__file__).resolve().parent / "src")
        env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "import pymegdec.evaluate_model_transfer; "
                    "print('xgboost' in sys.modules, "
                    "'torch' in sys.modules, "
                    "'pytorch_lightning' in sys.modules)"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )

        self.assertEqual("False False False", result.stdout.strip())

    def test_unsupported_classifier_error_lists_supported_names(self):
        with self.assertRaisesRegex(ValueError, "Supported classifiers"):
            train_multiclass_classifier(self.features, self.labels, 'unknown', None)


if __name__ == '__main__':
    unittest.main()
