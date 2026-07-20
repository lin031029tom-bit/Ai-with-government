from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class CommandLineIntegrationTests(unittest.TestCase):
    def test_alternative_dataset_command_records_relaxed_validation(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        rows = []
        for year in range(2020, 2025):
            for index in range(20):
                target = index % 2
                rows.append(
                    {
                        "collision_index": f"collision-{year}-{index}",
                        "collision_year": year,
                        "collision_severity": 2 if target else 3,
                        "serious_or_fatal": target,
                        "month": index % 12 + 1,
                        "speed_limit": 30 if target else 60,
                        "road_type": 1 if index % 3 else 6,
                    }
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            data_path = temp_path / "analysis_ready.csv"
            output_dir = temp_path / "outputs"
            pd.DataFrame(rows).to_csv(data_path, index=False)
            environment = os.environ.copy()
            environment["MPLBACKEND"] = "Agg"
            environment["MPLCONFIGDIR"] = str(temp_path / "matplotlib")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(repository_root / "road_safety_dissertation_coding.py"),
                    "--analysis-ready",
                    str(data_path),
                    "--output-dir",
                    str(output_dir),
                    "--allow-row-count-difference",
                    "--allow-feature-set-difference",
                    "--sample-size",
                    "60",
                ],
                cwd=temp_path,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(
                (output_dir / "tables" / "table_4_3_model_performance_2024_test.csv").exists()
            )
            run_information = json.loads(
                (output_dir / "tables" / "run_information.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(run_information["strict_expected_row_count"])
            self.assertFalse(run_information["strict_dissertation_feature_schema"])
            self.assertFalse(run_information["strict_expected_dataset_hash"])
            self.assertEqual(run_information["model_feature_count"], 3)
            self.assertIsNotNone(run_information["git_commit"])
            self.assertIn("git_worktree_dirty", run_information)


if __name__ == "__main__":
    unittest.main()
