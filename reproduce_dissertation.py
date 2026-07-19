#!/usr/bin/env python3
"""Run and verify the complete dissertation modelling workflow."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-ready",
        type=Path,
        required=True,
        help="Path to the validated analysis-ready CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("road_safety_coding_outputs"),
        help="Directory for generated tables and figures.",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("example_results"),
        help="Directory containing the verified reference outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parent
    analysis_script = repository_root / "road_safety_dissertation_coding.py"
    verification_script = repository_root / "verify_dissertation_results.py"

    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")
    environment.setdefault(
        "MPLCONFIGDIR", str(args.output_dir.resolve() / ".matplotlib")
    )

    analysis_command = [
        sys.executable,
        str(analysis_script),
        "--analysis-ready",
        str(args.analysis_ready),
        "--output-dir",
        str(args.output_dir),
        "--run-permutation",
        "--run-robustness",
    ]
    subprocess.run(
        analysis_command,
        cwd=repository_root,
        env=environment,
        check=True,
    )

    verification_command = [
        sys.executable,
        str(verification_script),
        "--generated-dir",
        str(args.output_dir),
        "--reference-dir",
        str(args.reference_dir),
    ]
    subprocess.run(
        verification_command,
        cwd=repository_root,
        env=environment,
        check=True,
    )
    print("Complete dissertation reproduction passed.")


if __name__ == "__main__":
    main()
