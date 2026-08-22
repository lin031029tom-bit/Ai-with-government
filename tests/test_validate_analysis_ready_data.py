from __future__ import annotations

import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from validate_analysis_ready_data import CORE_REQUIRED, validate


def valid_rows() -> list[dict[str, object]]:
    rows = []
    for year in range(2020, 2025):
        for target in (0, 1):
            row = {column: 1 for column in CORE_REQUIRED}
            row.update(
                {
                    "collision_index": f"collision-{year}-{target}",
                    "collision_year": year,
                    "collision_severity": 2 if target else 3,
                    "serious_or_fatal": target,
                }
            )
            rows.append(row)
    return rows


class ValidateAnalysisReadyDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = Path(self.temp_dir.name) / "analysis_ready.csv"

    def write_rows(self, rows: list[dict[str, object]]) -> None:
        fieldnames = sorted({key for row in rows for key in row})
        with self.path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def run_validation(self, rows: list[dict[str, object]]) -> None:
        self.write_rows(rows)
        with redirect_stdout(io.StringIO()):
            validate(
                self.path,
                enforce_expected_rows=False,
                enforce_expected_features=False,
            )

    def test_accepts_valid_alternative_row_count(self) -> None:
        self.run_validation(valid_rows())

    def test_rejects_fractional_target_instead_of_truncating_it(self) -> None:
        rows = valid_rows()
        rows[0]["serious_or_fatal"] = 0.5

        with self.assertRaisesRegex(ValueError, "binary 0/1"):
            self.run_validation(rows)

    def test_rejects_missing_target(self) -> None:
        rows = valid_rows()
        rows[0]["serious_or_fatal"] = ""

        with self.assertRaisesRegex(ValueError, "missing or non-numeric"):
            self.run_validation(rows)

    def test_rejects_non_numeric_year(self) -> None:
        rows = valid_rows()
        rows[0]["collision_year"] = "unknown"

        with self.assertRaisesRegex(ValueError, "collision_year contains"):
            self.run_validation(rows)

    def test_rejects_study_year_with_only_one_target_class(self) -> None:
        rows = [
            row
            for row in valid_rows()
            if not (row["collision_year"] == 2024 and row["serious_or_fatal"] == 1)
        ]

        with self.assertRaisesRegex(ValueError, "both target classes"):
            self.run_validation(rows)

    def test_rejects_fractional_year(self) -> None:
        rows = valid_rows()
        rows[0]["collision_year"] = 2020.5

        with self.assertRaisesRegex(ValueError, "whole-number values"):
            self.run_validation(rows)

    def test_rejects_missing_collision_identifier_column(self) -> None:
        rows = valid_rows()
        for row in rows:
            del row["collision_index"]

        with self.assertRaisesRegex(ValueError, "collision_index"):
            self.run_validation(rows)

    def test_rejects_missing_collision_severity_column(self) -> None:
        rows = valid_rows()
        for row in rows:
            del row["collision_severity"]

        with self.assertRaisesRegex(ValueError, "collision_severity"):
            self.run_validation(rows)

    def test_rejects_duplicate_collision_identifier(self) -> None:
        rows = valid_rows()
        rows[1]["collision_index"] = rows[0]["collision_index"]

        with self.assertRaisesRegex(ValueError, "duplicates"):
            self.run_validation(rows)

    def test_rejects_missing_collision_identifier(self) -> None:
        rows = valid_rows()
        rows[0]["collision_index"] = ""

        with self.assertRaisesRegex(ValueError, "missing or blank"):
            self.run_validation(rows)

    def test_rejects_target_that_disagrees_with_severity(self) -> None:
        rows = valid_rows()
        rows[0]["serious_or_fatal"] = 1
        rows[0]["collision_severity"] = 3

        with self.assertRaisesRegex(ValueError, "does not match"):
            self.run_validation(rows)

    def test_rejects_infinite_numeric_feature(self) -> None:
        rows = valid_rows()
        rows[0]["speed_limit"] = float("inf")

        with self.assertRaisesRegex(ValueError, "infinite values"):
            self.run_validation(rows)

    def test_strict_hash_rejects_a_different_prepared_dataset(self) -> None:
        self.write_rows(valid_rows())

        with self.assertRaisesRegex(ValueError, "Unexpected dataset SHA-256"):
            with redirect_stdout(io.StringIO()):
                validate(
                    self.path,
                    enforce_expected_rows=False,
                    enforce_expected_features=False,
                    enforce_expected_hash=True,
                )

    def test_strict_schema_rejects_missing_dissertation_features(self) -> None:
        self.write_rows(valid_rows())

        with self.assertRaisesRegex(
            ValueError, "Missing dissertation reproduction columns"
        ):
            with redirect_stdout(io.StringIO()):
                validate(
                    self.path,
                    enforce_expected_rows=False,
                    enforce_expected_features=True,
                )


if __name__ == "__main__":
    unittest.main()
