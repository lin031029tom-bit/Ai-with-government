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
    for offset, year in enumerate(range(2020, 2025)):
        row = {column: 1 for column in CORE_REQUIRED}
        row.update(
            {
                "collision_index": f"collision-{year}",
                "collision_year": year,
                "serious_or_fatal": offset % 2,
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
            validate(self.path, enforce_expected_rows=False)

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

    def test_rejects_fractional_year(self) -> None:
        rows = valid_rows()
        rows[0]["collision_year"] = 2020.5

        with self.assertRaisesRegex(ValueError, "whole-number years"):
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


if __name__ == "__main__":
    unittest.main()
