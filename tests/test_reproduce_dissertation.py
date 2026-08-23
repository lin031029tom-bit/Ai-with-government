from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import reproduce_dissertation


class ReproduceDissertationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.analysis_ready = self.root / "analysis_ready.csv"
        self.output_dir = self.root / "outputs"
        self.reference_dir = self.root / "reference"
        self.analysis_ready.touch()
        (self.reference_dir / "tables").mkdir(parents=True)
        self.arguments = argparse.Namespace(
            analysis_ready=self.analysis_ready,
            output_dir=self.output_dir,
            reference_dir=self.reference_dir,
        )

    @patch("reproduce_dissertation.subprocess.run")
    @patch("reproduce_dissertation.parse_args")
    def test_runs_analysis_before_result_verification(
        self,
        mock_parse_args,
        mock_run,
    ) -> None:
        mock_parse_args.return_value = self.arguments

        with patch.dict(os.environ, {}, clear=True):
            reproduce_dissertation.main()

        self.assertEqual(mock_run.call_count, 2)
        repository_root = Path(reproduce_dissertation.__file__).resolve().parent
        resolved_analysis_ready = self.analysis_ready.resolve()
        resolved_output_dir = self.output_dir.resolve()
        resolved_reference_dir = self.reference_dir.resolve()

        analysis_call = mock_run.call_args_list[0]
        analysis_command = analysis_call.args[0]
        self.assertEqual(
            analysis_command,
            [
                reproduce_dissertation.sys.executable,
                str(repository_root / "road_safety_dissertation_coding.py"),
                "--analysis-ready",
                str(resolved_analysis_ready),
                "--output-dir",
                str(resolved_output_dir),
                "--full-training",
                "--bootstrap-iterations",
                "1000",
                "--run-temporal-validation",
                "--run-permutation",
                "--run-robustness",
            ],
        )
        self.assertEqual(analysis_call.kwargs["cwd"], repository_root)
        self.assertTrue(analysis_call.kwargs["check"])
        self.assertEqual(analysis_call.kwargs["env"]["MPLBACKEND"], "Agg")
        self.assertEqual(
            analysis_call.kwargs["env"]["MPLCONFIGDIR"],
            str(resolved_output_dir / ".matplotlib"),
        )

        verification_call = mock_run.call_args_list[1]
        self.assertEqual(
            verification_call.args[0],
            [
                reproduce_dissertation.sys.executable,
                str(repository_root / "verify_dissertation_results.py"),
                "--generated-dir",
                str(resolved_output_dir),
                "--reference-dir",
                str(resolved_reference_dir),
            ],
        )
        self.assertEqual(verification_call.kwargs["cwd"], repository_root)
        self.assertTrue(verification_call.kwargs["check"])

    @patch("reproduce_dissertation.subprocess.run")
    @patch("reproduce_dissertation.parse_args")
    def test_stops_before_verification_when_analysis_fails(
        self,
        mock_parse_args,
        mock_run,
    ) -> None:
        mock_parse_args.return_value = self.arguments
        mock_run.side_effect = subprocess.CalledProcessError(1, ["analysis"])

        with self.assertRaises(subprocess.CalledProcessError):
            reproduce_dissertation.main()

        self.assertEqual(mock_run.call_count, 1)

    @patch("reproduce_dissertation.subprocess.run")
    @patch("reproduce_dissertation.parse_args")
    def test_missing_dataset_fails_before_starting_analysis(
        self,
        mock_parse_args,
        mock_run,
    ) -> None:
        self.arguments.analysis_ready = self.root / "missing.csv"
        mock_parse_args.return_value = self.arguments

        with self.assertRaisesRegex(FileNotFoundError, "Analysis-ready dataset not found"):
            reproduce_dissertation.main()

        mock_run.assert_not_called()

    @patch("reproduce_dissertation.subprocess.run")
    @patch("reproduce_dissertation.parse_args")
    def test_rejects_overwriting_verified_reference_results(
        self,
        mock_parse_args,
        mock_run,
    ) -> None:
        self.arguments.output_dir = self.reference_dir
        mock_parse_args.return_value = self.arguments

        with self.assertRaisesRegex(ValueError, "must not overwrite"):
            reproduce_dissertation.main()

        mock_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
