import ast
import unittest
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
CLEAN_NOTEBOOK = ROOT / "road_safety_dissertation_coding_clean.ipynb"
HISTORICAL_NOTEBOOK = (
    ROOT / "archive" / "road_safety_dissertation_coding_historical.ipynb"
)
VALIDATED_PUBLICATION_COMMIT = (
    "2148d0950fe026f66dd5ea1810807af32ce91af1"
)


class NotebookQualityTests(unittest.TestCase):
    def test_clean_notebook_is_valid_output_free_python(self):
        notebook = nbformat.read(CLEAN_NOTEBOOK, as_version=4)
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]

        self.assertTrue(code_cells)
        for cell in code_cells:
            self.assertIsNone(cell.execution_count)
            self.assertEqual(cell.outputs, [])
            ast.parse(cell.source)

    def test_clean_notebook_uses_the_published_one_command_snapshot(self):
        notebook = nbformat.read(CLEAN_NOTEBOOK, as_version=4)
        source = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "code"
        )

        self.assertIn(VALIDATED_PUBLICATION_COMMIT, source)
        self.assertIn("published_data/analysis_ready_road_safety.csv.gz", source)
        self.assertIn("reproduce_dissertation.py", source)
        self.assertNotIn("%run ", source)

    def test_failed_execution_log_is_archived_not_top_level(self):
        self.assertFalse((ROOT / "road_safety_dissertation_coding.ipynb").exists())
        self.assertTrue(HISTORICAL_NOTEBOOK.exists())


if __name__ == "__main__":
    unittest.main()
