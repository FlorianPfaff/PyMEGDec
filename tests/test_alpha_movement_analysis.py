import csv
import tempfile
import unittest
from pathlib import Path

from pymegdec.alpha_movement_analysis import (
    AlphaMovementAnalysisConfig,
    analyze_alpha_movement_windows,
    export_alpha_movement_analysis,
    summarize_alpha_movement_effects,
    write_alpha_movement_plots,
    write_csv_rows,
)


def _summary_rows():
    rows = []
    for participant in (1, 2):
        for trial_label, offset in (("1", 0.0), ("2", 4.0)):
            for time_s, post in (
                (-0.2, False),
                (-0.1, False),
                (0.1, True),
                (0.2, True),
            ):
                x_value = offset + (10.0 if post else 0.0) + participant
                rows.append(
                    {
                        "participant": participant,
                        "dataset": "main",
                        "trial_label": trial_label,
                        "time_s": time_s,
                        "n_trials": 3,
                        "mean_alpha_power": 100.0 + x_value,
                        "spatial_concentration": 0.2 + (0.1 if post else 0.0),
                        "centroid_x_mm": x_value,
                        "centroid_y_mm": 0.0,
                        "centroid_z_mm": 0.0,
                        "projected_x_mm": x_value,
                        "projected_y_mm": 0.0,
                        "displacement_mm": x_value,
                        "speed_mm_per_s": 20.0 + x_value,
                        "projected_speed_mm_per_s": 30.0 + x_value,
                    }
                )
    return rows


class TestAlphaMovementAnalysis(unittest.TestCase):
    def setUp(self):
        self.rows = _summary_rows()
        self.config = AlphaMovementAnalysisConfig(
            pre_window=(-0.25, 0.0),
            post_window=(0.0, 0.25),
        )

    def test_analyze_alpha_movement_windows_computes_pre_post_shifts(self):
        effect_rows = analyze_alpha_movement_windows(self.rows, self.config)

        self.assertEqual(len(effect_rows), 4)
        first = effect_rows[0]
        self.assertEqual(first["participant"], "1")
        self.assertEqual(first["trial_label"], "1")
        self.assertEqual(first["n_pre_points"], 2)
        self.assertEqual(first["n_post_points"], 2)
        self.assertAlmostEqual(first["centroid_shift_mm"], 10.0)
        self.assertAlmostEqual(first["projected_shift_mm"], 10.0)
        self.assertAlmostEqual(first["post_minus_pre_speed_mm_per_s"], 10.0)
        self.assertAlmostEqual(first["post_minus_pre_spatial_concentration"], 0.1)

    def test_summarize_alpha_movement_effects_groups_by_condition(self):
        effect_rows = analyze_alpha_movement_windows(self.rows, self.config)

        summary_rows = summarize_alpha_movement_effects(effect_rows, self.config)

        self.assertEqual(len(summary_rows), 2)
        self.assertEqual(summary_rows[0]["trial_label"], "1")
        self.assertEqual(summary_rows[0]["n_participants"], 2)
        self.assertAlmostEqual(summary_rows[0]["projected_shift_mm_mean"], 10.0)
        self.assertAlmostEqual(summary_rows[0]["post_minus_pre_alpha_power_mean"], 10.0)

    def test_export_alpha_movement_analysis_writes_csvs_and_plots(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "summary.csv"
            effect_path = temp_path / "effects.csv"
            condition_path = temp_path / "conditions.csv"
            plots_dir = temp_path / "plots"
            write_csv_rows(self.rows, input_path)

            effect_rows, summary_rows = export_alpha_movement_analysis(
                input_path,
                effect_path,
                condition_path,
                plots_dir=plots_dir,
                config=self.config,
            )

            with effect_path.open(newline="", encoding="utf-8") as handle:
                loaded_effects = list(csv.DictReader(handle))
            with condition_path.open(newline="", encoding="utf-8") as handle:
                loaded_summary = list(csv.DictReader(handle))

            self.assertEqual(len(effect_rows), len(loaded_effects))
            self.assertEqual(len(summary_rows), len(loaded_summary))
            self.assertTrue((plots_dir / "alpha_movement_projected_trajectories.png").exists())
            self.assertTrue((plots_dir / "alpha_movement_projected_speed.png").exists())
            self.assertTrue((plots_dir / "alpha_movement_displacement.png").exists())

    def test_write_alpha_movement_plots_can_filter_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            write_alpha_movement_plots(self.rows, output_dir, plot_labels=("2",))

            self.assertTrue((output_dir / "alpha_movement_projected_trajectories.png").exists())


if __name__ == "__main__":
    unittest.main()
