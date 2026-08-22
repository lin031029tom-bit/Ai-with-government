from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from verify_dissertation_results import CORE_TABLES, verify_results


class DissertationResultVerificationTests(unittest.TestCase):
    def _write_core_tables(self, directory: Path, value: float = 0.5) -> None:
        tables = directory / "tables"
        tables.mkdir(parents=True)
        for filename in CORE_TABLES:
            pd.DataFrame({"metric": ["value"], "score": [value]}).to_csv(
                tables / filename,
                index=False,
            )

    def test_accepts_matching_core_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            generated = root / "generated"
            reference = root / "reference"
            self._write_core_tables(generated)
            self._write_core_tables(reference)

            verified = verify_results(
                generated,
                reference,
                require_robustness=False,
                require_permutation=False,
            )

            self.assertEqual(verified, list(CORE_TABLES))

    def test_rejects_a_changed_value(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            generated = root / "generated"
            reference = root / "reference"
            self._write_core_tables(generated)
            self._write_core_tables(reference)
            changed_path = generated / "tables" / CORE_TABLES[0]
            pd.DataFrame({"metric": ["value"], "score": [0.7]}).to_csv(
                changed_path,
                index=False,
            )

            with self.assertRaisesRegex(AssertionError, CORE_TABLES[0]):
                verify_results(
                    generated,
                    reference,
                    require_robustness=False,
                    require_permutation=False,
                )


if __name__ == "__main__":
    unittest.main()
