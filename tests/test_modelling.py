from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from road_safety_dissertation_coding import (
    TARGET,
    clean_feature_frame,
    fit_main_models,
    local_authority_rate_table,
    mkdir,
)


class ModellingWorkflowTests(unittest.TestCase):
    def test_replaces_documented_unknown_values_but_preserves_valid_age_99(self) -> None:
        frame = pd.DataFrame(
            {
                "mean_driver_age": [99, -1, 45],
                "speed_limit": [30, -1, 70],
            }
        )

        cleaned = clean_feature_frame(
            frame,
            numeric=["mean_driver_age", "speed_limit"],
            categorical=[],
        )

        self.assertEqual(cleaned.loc[0, "mean_driver_age"], 99)
        self.assertTrue(pd.isna(cleaned.loc[1, "mean_driver_age"]))
        self.assertTrue(pd.isna(cleaned.loc[1, "speed_limit"]))

    def test_normalizes_missing_categorical_values(self) -> None:
        frame = pd.DataFrame({"road_type": ["single carriageway", None]})

        cleaned = clean_feature_frame(
            frame,
            numeric=[],
            categorical=["road_type"],
        )

        self.assertEqual(cleaned["road_type"].tolist(), ["single carriageway", "missing"])

    def test_main_models_run_on_a_small_temporal_dataset(self) -> None:
        rows = []
        for year in range(2020, 2025):
            for index in range(12):
                rows.append(
                    {
                        "collision_year": year,
                        TARGET: index % 2,
                        "month": index % 12 + 1,
                        "speed_limit": 30 if index % 2 else 60,
                        "road_type": "single" if index % 3 else "dual",
                    }
                )
        frame = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            tables = output_dir / "tables"
            figures = output_dir / "figures"
            mkdir(tables)
            mkdir(figures)

            _, _, test, y_test, probabilities = fit_main_models(
                frame,
                tables,
                figures,
                sample_size=32,
                seed=42,
            )

            self.assertEqual(len(test), 12)
            self.assertEqual(len(y_test), 12)
            self.assertEqual(
                set(probabilities),
                {
                    "Dummy majority baseline",
                    "Balanced logistic regression",
                    "Random Forest",
                },
            )
            self.assertTrue(
                (tables / "table_4_3_model_performance_2024_test.csv").exists()
            )
            self.assertTrue((figures / "figure_4_6_roc_curves.png").exists())

    def test_local_authority_table_applies_the_documented_minimum_count(self) -> None:
        frame = pd.DataFrame(
            {
                "collision_index": [f"collision-{index}" for index in range(600)],
                "local_authority_highway": ["included"] * 550 + ["excluded"] * 50,
                TARGET: [index % 4 == 0 for index in range(600)],
            }
        )

        table = local_authority_rate_table(frame, min_collisions=500)

        self.assertEqual(table["local_authority_highway"].tolist(), ["included"])
        self.assertEqual(table["collisions"].tolist(), [550])


if __name__ == "__main__":
    unittest.main()
