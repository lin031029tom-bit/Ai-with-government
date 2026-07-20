#!/usr/bin/env python3
"""Compare generated dissertation tables with the verified reference outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CORE_TABLES = (
    "table_4_1_severity_distribution.csv",
    "table_4_2_yearly_serious_fatal_rate.csv",
    "table_4_3_model_performance_2024_test.csv",
    "table_4_4_confusion_matrix_counts.csv",
)

ROBUSTNESS_TABLES = (
    "table_4_5_random_forest_robustness_checks.csv",
    "random_forest_threshold_sensitivity.csv",
    "table_4_5_metric_uncertainty_2024.csv",
    "table_4_5_paired_model_differences_2024.csv",
    "table_4_5_rolling_origin_validation.csv",
)

INTERPRETATION_TABLES = (
    "random_forest_permutation_importance.csv",
)


def _tables_directory(path: Path) -> Path:
    nested = path / "tables"
    return nested if nested.is_dir() else path


def compare_csv(
    generated_path: Path,
    reference_path: Path,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> None:
    generated = pd.read_csv(generated_path)
    reference = pd.read_csv(reference_path)
    pd.testing.assert_frame_equal(
        generated,
        reference,
        check_dtype=False,
        check_exact=False,
        check_like=False,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )


def verify_results(
    generated_dir: Path,
    reference_dir: Path,
    *,
    require_robustness: bool = True,
    require_permutation: bool = True,
    relative_tolerance: float = 1e-9,
    absolute_tolerance: float = 1e-9,
) -> list[str]:
    generated_tables = _tables_directory(generated_dir)
    reference_tables = _tables_directory(reference_dir)

    required = list(CORE_TABLES)
    if require_robustness:
        required.extend(ROBUSTNESS_TABLES)
    if require_permutation:
        required.extend(INTERPRETATION_TABLES)

    verified: list[str] = []
    for filename in required:
        generated_path = generated_tables / filename
        reference_path = reference_tables / filename
        if not generated_path.exists():
            raise FileNotFoundError(f"Generated result is missing: {generated_path}")
        if not reference_path.exists():
            raise FileNotFoundError(f"Reference result is missing: {reference_path}")
        try:
            compare_csv(
                generated_path,
                reference_path,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
        except AssertionError as exc:
            raise AssertionError(f"{filename} differs from the verified result: {exc}") from exc
        verified.append(filename)

    return verified


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generated-dir",
        type=Path,
        required=True,
        help="Generated output directory, or its tables subdirectory.",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("example_results"),
        help="Verified output directory, or its tables subdirectory.",
    )
    parser.add_argument(
        "--skip-robustness",
        action="store_true",
        help="Do not require the robustness and threshold tables.",
    )
    parser.add_argument(
        "--skip-permutation",
        action="store_true",
        help="Do not require the permutation-importance table.",
    )
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verified = verify_results(
        args.generated_dir,
        args.reference_dir,
        require_robustness=not args.skip_robustness,
        require_permutation=not args.skip_permutation,
        relative_tolerance=args.rtol,
        absolute_tolerance=args.atol,
    )
    for filename in verified:
        print(f"Verified: {filename}")
    print(f"Result verification passed ({len(verified)} tables).")


if __name__ == "__main__":
    main()
