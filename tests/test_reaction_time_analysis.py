import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pymegdec.reaction_time_analysis import (
    ReactionTimeCsvConfig,
    ReactionTimeUnavailableError,
    analyze_alpha_reaction_times,
    extract_reaction_times_from_data,
    join_alpha_reaction_times,
    load_reaction_time_csv,
    parse_participant_spec,
    write_csv_rows,
)


def _synthetic_data(trialinfo):
    trials = np.empty(3, dtype=object)
    for index in range(3):
        trials[index] = np.zeros((1, 2))
    return {
        "trial": trials,
        "trialinfo": np.asarray(trialinfo),
    }


class TestReactionTimeAnalysis(unittest.TestCase):
    def test_parse_participant_spec(self):
        self.assertEqual(parse_participant_spec("1-3,2,8"), [1, 2, 3, 8])

    def test_load_reaction_time_csv_uses_single_participant_default(self):
        rows = [
            {"trial": 0, "rt": 0.45},
            {"trial": 1, "rt": 0.50},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rt.csv"
            write_csv_rows(rows, path)

            loaded = load_reaction_time_csv(
                path, ReactionTimeCsvConfig(default_participant=2)
            )

        self.assertEqual(loaded[0]["participant"], "2")
        self.assertEqual(loaded[0]["dataset"], "main")
        self.assertEqual(loaded[1]["reaction_time"], 0.50)

    def test_extract_reaction_times_from_trialinfo_column(self):
        data = _synthetic_data(
            [
                [1, 0.41],
                [2, 0.52],
                [3, 0.63],
            ]
        )

        rows = extract_reaction_times_from_data(
            data, participant_id=5, trialinfo_rt_column=1
        )

        self.assertEqual([row["trial"] for row in rows], [0, 1, 2])
        self.assertEqual([row["participant"] for row in rows], ["5", "5", "5"])
        np.testing.assert_allclose(
            [row["reaction_time"] for row in rows], [0.41, 0.52, 0.63]
        )

    def test_extract_reaction_times_raises_when_absent(self):
        data = _synthetic_data([[1, 2, 3]])

        with self.assertRaises(ReactionTimeUnavailableError):
            extract_reaction_times_from_data(data)

    def test_join_adds_direction_components(self):
        alpha_rows = [
            {
                "participant": 2,
                "dataset": "main",
                "trial": 0,
                "log_alpha_power": 1.0,
                "direction_rad": 0.0,
            },
            {
                "participant": 2,
                "dataset": "main",
                "trial": 1,
                "log_alpha_power": 2.0,
                "direction_rad": np.pi / 2,
            },
        ]
        reaction_rows = [
            {"participant": 2, "dataset": "main", "trial": 0, "reaction_time": 0.4},
            {"participant": 2, "dataset": "main", "trial": 1, "reaction_time": 0.6},
        ]

        joined = join_alpha_reaction_times(alpha_rows, reaction_rows)

        self.assertEqual(len(joined), 2)
        self.assertAlmostEqual(joined[0]["direction_cos"], 1.0)
        self.assertAlmostEqual(joined[1]["direction_sin"], 1.0)

    def test_analyze_alpha_reaction_times_reports_participant_and_pooled_rows(self):
        rows = []
        for participant in (1, 2):
            for trial_idx in range(4):
                value = float(trial_idx)
                rows.append(
                    {
                        "participant": participant,
                        "dataset": "main",
                        "trial": trial_idx,
                        "log_alpha_power": value,
                        "reaction_time": 0.3 + value * 0.1 + participant,
                    }
                )

        summary = analyze_alpha_reaction_times(rows, metrics=("log_alpha_power",))

        self.assertEqual(len(summary), 3)
        for row in summary:
            self.assertEqual(row["metric"], "log_alpha_power")
            self.assertAlmostEqual(row["pearson_r"], 1.0)

    def test_write_csv_rows(self):
        rows = [{"a": 1, "b": "x"}]

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rows.csv"
            write_csv_rows(rows, path)
            with path.open(newline="", encoding="utf-8") as handle:
                loaded = list(csv.DictReader(handle))

        self.assertEqual(loaded, [{"a": "1", "b": "x"}])


if __name__ == "__main__":
    unittest.main()
