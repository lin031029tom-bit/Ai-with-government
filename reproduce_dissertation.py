#!/usr/bin/env python3
"""Run and verify the complete dissertation modelling workflow."""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_DATASET = Path("road_safety_analysis/analysis_ready_road_safety.csv")
DEFAULT_DATA_ARCHIVE = Path("published_data/analysis_ready_road_safety.csv.gz")
REQUIRED_MODULES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "scikit-learn": "sklearn",
    "openpyxl": "openpyxl",
    "nbformat": "nbformat",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-ready",
        type=Path,
        default=None,
        help=(
            "Path to the validated analysis-ready CSV. Defaults to "
            "road_safety_analysis/analysis_ready_road_safety.csv in the repository."
        ),
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
        default=None,
        help=(
            "Directory containing the verified reference outputs. Defaults to "
            "example_results in the repository."
        ),
    )
    return parser.parse_args()


def resolve_from_invocation(path: Path, invocation_dir: Path) -> Path:
    """Resolve a user-supplied relative path before changing subprocess cwd."""
    path = path.expanduser()
    if not path.is_absolute():
        path = invocation_dir / path
    return path.resolve()


def validate_runtime() -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            "Validated reproduction requires Python 3.12; current interpreter is "
            f"{sys.version_info.major}.{sys.version_info.minor}. Create the environment "
            "with `python3.12 -m venv .venv`."
        )

    missing = [
        package
        for package, module in REQUIRED_MODULES.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        raise RuntimeError(
            "Missing required packages: "
            + ", ".join(missing)
            + ". Activate the Python 3.12 environment and run "
            "`python -m pip install -r requirements.txt`."
        )


def materialise_default_dataset(repository_root: Path) -> Path:
    """Extract the published dataset once when the default CSV is not yet present."""
    destination = repository_root / DEFAULT_DATASET
    if destination.is_file():
        return destination

    archive = repository_root / DEFAULT_DATA_ARCHIVE
    if not archive.is_file():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with gzip.open(archive, "rb") as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(f"Extracted published dataset: {archive} -> {destination}")
    return destination


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parent
    invocation_dir = Path.cwd()
    analysis_script = repository_root / "road_safety_dissertation_coding.py"
    verification_script = repository_root / "verify_dissertation_results.py"

    validate_runtime()
    analysis_ready = (
        materialise_default_dataset(repository_root)
        if args.analysis_ready is None
        else resolve_from_invocation(args.analysis_ready, invocation_dir)
    )
    output_dir = resolve_from_invocation(args.output_dir, invocation_dir)
    reference_dir = (
        repository_root / "example_results"
        if args.reference_dir is None
        else resolve_from_invocation(args.reference_dir, invocation_dir)
    )

    if not analysis_ready.is_file():
        raise FileNotFoundError(
            f"Analysis-ready dataset not found: {analysis_ready}\n"
            "Place the validated CSV at that location or pass --analysis-ready PATH."
        )
    if not (reference_dir / "tables").is_dir():
        raise FileNotFoundError(
            f"Verified reference tables not found: {reference_dir / 'tables'}"
        )
    if output_dir == reference_dir:
        raise ValueError("--output-dir must not overwrite --reference-dir")

    output_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_dir = output_dir / ".matplotlib"
    matplotlib_dir.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")
    environment.setdefault("MPLCONFIGDIR", str(matplotlib_dir))

    print(f"Python: {sys.version.split()[0]}")
    print(f"Dataset: {analysis_ready}")
    print(f"Output: {output_dir}")
    print(f"Reference results: {reference_dir}")

    analysis_command = [
        sys.executable,
        str(analysis_script),
        "--analysis-ready",
        str(analysis_ready),
        "--output-dir",
        str(output_dir),
        "--full-training",
        "--bootstrap-iterations",
        "1000",
        "--run-temporal-validation",
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
        str(output_dir),
        "--reference-dir",
        str(reference_dir),
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
