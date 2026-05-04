import unittest
from unittest.mock import patch

import numpy as np

from pymegdec.stimulus_decoding import (
    StimulusDecodingConfig,
    evaluate_participant_stimulus_decoding_diagnostics,
    evaluate_participant_time_resolved_stimulus_transfer,
    summarize_stimulus_decoding,
    summarize_stimulus_decoding_peaks,
    summarize_stimulus_prediction_diagnostics,
    window_centers_from_range,
)
from tests.matlab_fixtures import cell_array


def _mat_data(labels, trial_values, time):
    trialinfo = np.empty((1, 1), dtype=object)
    trialinfo[0, 0] = np.asarray(labels, dtype=int)
    return {
        "trial": cell_array([np.asarray([[0.0, value]], dtype=float) for value in trial_values]),
        "time": cell_array([np.asarray([time], dtype=float) for _ in trial_values]),
        "trialinfo": trialinfo,
    }


class TestStimulusDecoding(unittest.TestCase):
    def test_window_centers_from_range_includes_stop(self):
        self.assertEqual(
            window_centers_from_range((-0.1, 0.1), 0.1),
            (-0.1, 0.0, 0.1),
        )

    def test_evaluate_participant_time_resolved_stimulus_transfer(self):
        labels = [1, 2, 1, 2]
        train_data = _mat_data(labels, [-2.0, 2.0, -1.0, 1.0], [-0.1, 0.0])
        validation_data = _mat_data(labels, [-1.5, 1.5, -0.5, 0.5], [-0.1, 0.0])
        config = StimulusDecodingConfig(
            window_centers=(0.0,),
            window_size=0.0,
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch(
            "pymegdec.stimulus_decoding.sio.loadmat",
            side_effect=[
                {"data": np.array([train_data], dtype=object)},
                {"data": np.array([validation_data], dtype=object)},
            ],
        ):
            rows = evaluate_participant_time_resolved_stimulus_transfer("unused", 1, config=config)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["variant"], "without_null")
        self.assertEqual(rows[0]["accuracy"], 1.0)
        self.assertEqual(rows[0]["chance_accuracy"], 0.5)

    def test_evaluate_participant_stimulus_decoding_diagnostics(self):
        labels = [1, 2, 1, 2]
        train_data = _mat_data(labels, [-2.0, 2.0, -1.0, 1.0], [-0.1, 0.0])
        validation_data = _mat_data(labels, [-1.5, 1.5, -0.5, 0.5], [-0.1, 0.0])
        config = StimulusDecodingConfig(
            window_centers=(0.0, 0.1),
            window_size=0.0,
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch(
            "pymegdec.stimulus_decoding.sio.loadmat",
            side_effect=[
                {"data": np.array([train_data], dtype=object)},
                {"data": np.array([validation_data], dtype=object)},
            ],
        ):
            rows, prediction_rows = evaluate_participant_stimulus_decoding_diagnostics(
                "unused",
                1,
                config=config,
                diagnostic_window_centers=(0.0,),
            )

        self.assertEqual(len(rows), 2)
        self.assertEqual(len(prediction_rows), 4)
        self.assertEqual({row["window_center_s"] for row in prediction_rows}, {0.0})
        self.assertEqual([row["true_stimulus"] for row in prediction_rows], labels)
        self.assertEqual([row["true_stimulus_id"] for row in prediction_rows], labels)
        self.assertEqual([row["validation_trial_index"] for row in prediction_rows], [0, 1, 2, 3])
        self.assertEqual([row["validation_trial_number"] for row in prediction_rows], [1, 2, 3, 4])
        self.assertEqual({row["actual_components_pca"] for row in prediction_rows}, {1})
        self.assertTrue(all(row["correct"] for row in prediction_rows))

    def test_evaluate_participant_time_resolved_stimulus_transfer_with_permutations(
        self,
    ):
        labels = [1, 2, 1, 2]
        train_data = _mat_data(labels, [-2.0, 2.0, -1.0, 1.0], [-0.1, 0.0])
        validation_data = _mat_data(labels, [-1.5, 1.5, -0.5, 0.5], [-0.1, 0.0])
        config = StimulusDecodingConfig(
            window_centers=(0.0,),
            window_size=0.0,
            components_pca=float("inf"),
            chance_classes=2,
            permutations=3,
            permutation_seed=0,
        )

        with (
            patch(
                "pymegdec.stimulus_decoding.sio.loadmat",
                side_effect=[
                    {"data": np.array([train_data], dtype=object)},
                    {"data": np.array([validation_data], dtype=object)},
                ],
            ),
            patch(
                "pymegdec.stimulus_decoding._permutation_accuracy_curve",
                return_value=np.array([0.0, 0.25, 0.5]),
            ),
        ):
            rows = evaluate_participant_time_resolved_stimulus_transfer("unused", 1, config=config)

        self.assertEqual(rows[0]["n_permutations"], 3)
        self.assertAlmostEqual(rows[0]["permutation_p_value"], 0.25)
        self.assertAlmostEqual(rows[0]["permutation_accuracy_mean"], 0.25)

    def test_summarize_stimulus_decoding(self):
        rows = [
            {
                "variant": "without_null",
                "window_center_s": 0.0,
                "accuracy": 0.25,
                "chance_accuracy": 0.0625,
                "permutation_p_value": 0.04,
            },
            {
                "variant": "without_null",
                "window_center_s": 0.0,
                "accuracy": 0.5,
                "chance_accuracy": 0.0625,
                "permutation_p_value": 0.006,
            },
            {
                "variant": "without_null",
                "window_center_s": 0.1,
                "accuracy": 0.5,
                "chance_accuracy": 0.0625,
                "permutation_p_value": np.nan,
            },
        ]

        summary = summarize_stimulus_decoding(rows)

        self.assertEqual(len(summary), 2)
        self.assertEqual(summary[0]["n_participants"], 2)
        self.assertAlmostEqual(summary[0]["accuracy_mean"], 0.375)
        self.assertEqual(summary[0]["above_chance_count"], 2)
        self.assertEqual(summary[0]["n_with_permutation"], 2)
        self.assertEqual(summary[0]["n_significant_p_0.05"], 2)
        self.assertEqual(summary[0]["n_significant_p_0.01"], 1)

    def test_summarize_stimulus_decoding_peaks(self):
        rows = [
            {
                "participant": 1,
                "variant": "without_null",
                "window_center_s": 0.1,
                "window_start_s": 0.05,
                "window_stop_s": 0.15,
                "accuracy": 0.2,
                "percent": 20.0,
                "chance_accuracy": 0.0625,
                "chance_percent": 6.25,
            },
            {
                "participant": 1,
                "variant": "without_null",
                "window_center_s": 0.2,
                "window_start_s": 0.15,
                "window_stop_s": 0.25,
                "accuracy": 0.3,
                "percent": 30.0,
                "chance_accuracy": 0.0625,
                "chance_percent": 6.25,
            },
        ]

        peaks = summarize_stimulus_decoding_peaks(rows)

        self.assertEqual(len(peaks), 1)
        self.assertEqual(peaks[0]["peak_window_center_s"], 0.2)
        self.assertEqual(peaks[0]["peak_accuracy"], 0.3)

    def test_summarize_stimulus_prediction_diagnostics(self):
        prediction_rows = [
            {
                "participant": 1,
                "variant": "without_null",
                "window_center_s": 0.2,
                "true_stimulus": 1,
                "predicted_stimulus": 1,
                "correct": True,
            },
            {
                "participant": 1,
                "variant": "without_null",
                "window_center_s": 0.2,
                "true_stimulus": 1,
                "predicted_stimulus": 2,
                "correct": False,
            },
        ]

        confusion, per_stimulus = summarize_stimulus_prediction_diagnostics(prediction_rows)

        self.assertEqual(len(confusion), 2)
        self.assertEqual(per_stimulus[0]["true_stimulus"], 1)
        self.assertEqual(per_stimulus[0]["n_trials"], 2)
        self.assertEqual(per_stimulus[0]["n_correct"], 1)
        self.assertEqual(per_stimulus[0]["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
