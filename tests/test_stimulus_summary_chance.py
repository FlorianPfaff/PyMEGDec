import unittest

from pymegdec.stimulus_decoding import (
    summarize_stimulus_decoding,
    summarize_stimulus_temporal_generalization,
)


class TestStimulusSummaryChance(unittest.TestCase):
    def test_stimulus_summary_aggregates_mixed_chance_levels(self):
        rows = [
            {
                "participant": 1,
                "variant": "without_null",
                "transfer_direction": "main-to-cue",
                "window_center_s": 0.0,
                "classifier": "multiclass-svm",
                "components_pca": 100,
                "frequency_low_hz": 0.0,
                "frequency_high_hz": float("inf"),
                "accuracy": 0.40,
                "chance_accuracy": 0.50,
                "n_validation_classes": 2,
            },
            {
                "participant": 2,
                "variant": "without_null",
                "transfer_direction": "main-to-cue",
                "window_center_s": 0.0,
                "classifier": "multiclass-svm",
                "components_pca": 100,
                "frequency_low_hz": 0.0,
                "frequency_high_hz": float("inf"),
                "accuracy": 0.30,
                "chance_accuracy": 0.25,
                "n_validation_classes": 4,
            },
        ]

        summary = summarize_stimulus_decoding(rows)

        self.assertEqual(len(summary), 1)
        self.assertAlmostEqual(summary[0]["chance_accuracy"], 0.375)
        self.assertAlmostEqual(summary[0]["chance_percent"], 37.5)
        self.assertAlmostEqual(summary[0]["chance_accuracy_min"], 0.25)
        self.assertAlmostEqual(summary[0]["chance_accuracy_max"], 0.50)
        self.assertAlmostEqual(summary[0]["chance_classes_mean"], 3.0)
        self.assertEqual(summary[0]["chance_classes_counts"], "2:1;4:1")
        self.assertEqual(summary[0]["above_chance_count"], 1)

    def test_temporal_generalization_summary_aggregates_mixed_chance_levels(self):
        rows = [
            {
                "participant": 1,
                "variant": "without_null",
                "transfer_direction": "main-to-cue",
                "train_window_center_s": 0.0,
                "test_window_center_s": 0.0,
                "classifier": "multiclass-svm",
                "components_pca": 100,
                "frequency_low_hz": 0.0,
                "frequency_high_hz": float("inf"),
                "accuracy": 0.40,
                "chance_accuracy": 0.50,
                "n_validation_classes": 2,
            },
            {
                "participant": 2,
                "variant": "without_null",
                "transfer_direction": "main-to-cue",
                "train_window_center_s": 0.0,
                "test_window_center_s": 0.0,
                "classifier": "multiclass-svm",
                "components_pca": 100,
                "frequency_low_hz": 0.0,
                "frequency_high_hz": float("inf"),
                "accuracy": 0.30,
                "chance_accuracy": 0.25,
                "n_validation_classes": 4,
            },
        ]

        summary = summarize_stimulus_temporal_generalization(rows)

        self.assertEqual(len(summary), 1)
        self.assertAlmostEqual(summary[0]["chance_accuracy"], 0.375)
        self.assertAlmostEqual(summary[0]["chance_accuracy_min"], 0.25)
        self.assertAlmostEqual(summary[0]["chance_accuracy_max"], 0.50)
        self.assertAlmostEqual(summary[0]["chance_classes_mean"], 3.0)
        self.assertEqual(summary[0]["chance_classes_counts"], "2:1;4:1")
        self.assertEqual(summary[0]["above_chance_count"], 1)


if __name__ == "__main__":
    unittest.main()
