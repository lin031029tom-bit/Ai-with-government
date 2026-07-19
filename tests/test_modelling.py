from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from road_safety_dissertation_coding import (
    TARGET,
    clean_feature_frame,
    evaluate,
    feature_lists,
    fit_main_models,
    local_authority_rate_table,
    mkdir,
    validated_binary_target,
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
                require_all_features=False,
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
                "traffic_local_authority_name": (
                    ["Included Authority"] * 550 + ["Excluded Authority"] * 50
                ),
                TARGET: [index % 4 == 0 for index in range(600)],
            }
        )

        table = local_authority_rate_table(frame, min_collisions=500)

        self.assertEqual(table["local_authority_highway"].tolist(), ["included"])
        self.assertEqual(
            table["local_authority_name"].tolist(), ["Included Authority"]
        )
        self.assertEqual(table["collisions"].tolist(), [550])

    def test_rejects_fractional_target_in_direct_model_calls(self) -> None:
        frame = pd.DataFrame({TARGET: [0, 0.5, 1]})

        with self.assertRaisesRegex(ValueError, "binary 0/1"):
            validated_binary_target(frame)

    def test_strict_feature_selection_rejects_partial_model_schema(self) -> None:
        frame = pd.DataFrame({"month": [1], "road_type": ["single"]})

        with self.assertRaisesRegex(
            ValueError, "Missing required dissertation model features"
        ):
            feature_lists(frame, require_all=True)

        numeric, categorical = feature_lists(frame, require_all=False)
        self.assertEqual(numeric, ["month"])
        self.assertEqual(categorical, ["road_type"])

    def test_constant_baseline_score_has_chance_roc_auc(self) -> None:
        result = evaluate(
            "constant baseline",
            pd.Series([0, 0, 1, 1]),
            np.zeros(4),
        )

        self.assertEqual(result["roc_auc"], 0.5)

    def test_main_models_reject_single_class_test_split(self) -> None:
        rows = []
        for year in range(2020, 2025):
            for index in range(12):
                rows.append(
                    {
                        "collision_year": year,
                        TARGET: 0 if year == 2024 else index % 2,
                        "month": index % 12 + 1,
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

            with self.assertRaisesRegex(ValueError, "both target classes"):
                fit_main_models(
                    frame,
                    tables,
                    figures,
                    sample_size=32,
                    seed=42,
                    require_all_features=False,
                )


if __name__ == "__main__":
    unittest.main()
