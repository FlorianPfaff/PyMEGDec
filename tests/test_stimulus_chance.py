import argparse
import unittest
from unittest.mock import patch

import numpy as np
from pymegdec import cli as legacy_cli
from pymegdec import _stimulus_decoding_core as stimulus_core
from pymegdec import stimulus_cli
from pymegdec.stimulus_decoding import (
    DEFAULT_CHANCE_CLASSES,
    StimulusDecodingConfig,
    evaluate_participant_time_resolved_stimulus_transfer,
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


class TestStimulusChanceLevel(unittest.TestCase):
    def _evaluate(
        self,
        config,
        *,
        module_path="pymegdec.stimulus_decoding",
        evaluate=evaluate_participant_time_resolved_stimulus_transfer,
    ):
        labels = [1, 2, 1, 2]
        train_data = _mat_data(labels, [-2.0, 2.0, -1.0, 1.0], [-0.1, 0.0])
        validation_data = _mat_data(labels, [-1.5, 1.5, -0.5, 0.5], [-0.1, 0.0])
        with patch(
            f"{module_path}.sio.loadmat",
            side_effect=[
                {"data": np.array([train_data], dtype=object)},
                {"data": np.array([validation_data], dtype=object)},
            ],
        ):
            rows = evaluate("unused", 1, config=config)
        return rows[0]

    def test_auto_chance_uses_validation_class_count(self):
        config = StimulusDecodingConfig(
            window_centers=(0.0,),
            window_size=0.0,
            components_pca=float("inf"),
        )

        row = self._evaluate(config)

        self.assertEqual(row["n_validation_classes"], 2)
        self.assertEqual(row["chance_accuracy"], 0.5)
        self.assertEqual(row["chance_percent"], 50.0)

    def test_private_core_default_chance_uses_validation_class_count(self):
        config = stimulus_core.StimulusDecodingConfig(
            window_centers=(0.0,),
            window_size=0.0,
            components_pca=float("inf"),
        )

        row = self._evaluate(
            config,
            module_path="pymegdec._stimulus_decoding_core",
            evaluate=stimulus_core.evaluate_participant_time_resolved_stimulus_transfer,
        )

        self.assertEqual(row["n_validation_classes"], 2)
        self.assertEqual(row["chance_accuracy"], 0.5)
        self.assertEqual(row["chance_percent"], 50.0)

    def test_legacy_default_chance_classes_is_treated_as_auto_for_cli_defaults(self):
        config = StimulusDecodingConfig(
            window_centers=(0.0,),
            window_size=0.0,
            components_pca=float("inf"),
            chance_classes=DEFAULT_CHANCE_CLASSES,
        )

        row = self._evaluate(config)

        self.assertEqual(row["n_validation_classes"], 2)
        self.assertEqual(row["chance_accuracy"], 0.5)

    def test_non_default_chance_classes_remain_explicit_override(self):
        config = StimulusDecodingConfig(
            window_centers=(0.0,),
            window_size=0.0,
            components_pca=float("inf"),
            chance_classes=4,
        )

        row = self._evaluate(config)

        self.assertEqual(row["n_validation_classes"], 2)
        self.assertEqual(row["chance_accuracy"], 0.25)
        self.assertEqual(row["chance_percent"], 25.0)

    def test_inference_can_be_disabled_to_force_sixteen_way_chance(self):
        config = StimulusDecodingConfig(
            window_centers=(0.0,),
            window_size=0.0,
            components_pca=float("inf"),
            chance_classes=DEFAULT_CHANCE_CLASSES,
            infer_chance_classes=False,
        )

        row = self._evaluate(config)

        self.assertEqual(row["n_validation_classes"], 2)
        self.assertEqual(row["chance_accuracy"], 1.0 / DEFAULT_CHANCE_CLASSES)
        self.assertEqual(row["chance_percent"], 100.0 / DEFAULT_CHANCE_CLASSES)


class TestStimulusChanceCli(unittest.TestCase):
    def test_parse_chance_classes_accepts_auto_and_fixed_values(self):
        self.assertIsNone(legacy_cli.parse_chance_classes("auto"))
        self.assertIsNone(legacy_cli.parse_chance_classes("inferred"))
        self.assertEqual(legacy_cli.parse_chance_classes("16"), 16)

    def test_parse_chance_classes_rejects_invalid_values(self):
        for token in ("0", "-1", "not-a-number"):
            with self.subTest(token=token):
                with self.assertRaises(argparse.ArgumentTypeError):
                    legacy_cli.parse_chance_classes(token)

    def test_legacy_stimulus_cli_default_keeps_auto_chance(self):
        config = self._legacy_config_from_cli(["--output", "out.csv"])

        self.assertIsNone(config.chance_classes)
        self.assertTrue(config.infer_chance_classes)

    def test_legacy_stimulus_cli_explicit_sixteen_forces_fixed_chance(self):
        config = self._legacy_config_from_cli(["--output", "out.csv", "--chance-classes", "16"])

        self.assertEqual(config.chance_classes, 16)
        self.assertFalse(config.infer_chance_classes)

    def test_grouped_stimulus_base_config_default_keeps_auto_chance(self):
        config = stimulus_cli._base_config(
            self._grouped_args(chance_classes=None),
            window_centers=(0.0,),
            transfer_direction="main-to-cue",
        )

        self.assertIsNone(config.chance_classes)
        self.assertTrue(config.infer_chance_classes)

    def test_grouped_stimulus_base_config_explicit_sixteen_forces_fixed_chance(self):
        config = stimulus_cli._base_config(
            self._grouped_args(chance_classes=16),
            window_centers=(0.0,),
            transfer_direction="main-to-cue",
        )

        self.assertEqual(config.chance_classes, 16)
        self.assertFalse(config.infer_chance_classes)

    def _legacy_config_from_cli(self, argv):
        captured = {}

        def fake_export(*_args, **kwargs):
            captured["config"] = kwargs["config"]
            return [], []

        with patch("pymegdec.cli._transfer_participants", return_value=[1]), patch(
            "pymegdec.cli.export_time_resolved_stimulus_decoding",
            side_effect=fake_export,
        ):
            legacy_cli.stimulus_decoding(argv)
        return captured["config"]

    def _grouped_args(self, *, chance_classes):
        return argparse.Namespace(
            window_size=0.1,
            null_window_center=float("nan"),
            new_framerate=float("inf"),
            classifier="multiclass-svm",
            classifier_param=None,
            components_pca=100,
            frequency_range=(0.0, float("inf")),
            chance_classes=chance_classes,
            transfer_direction="main-to-cue",
        )


if __name__ == "__main__":
    unittest.main()
