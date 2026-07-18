#!/usr/bin/env python3
"""Validate the prepared road-safety analysis dataset before modelling."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

EXPECTED_ROWS = 503_475
EXPECTED_YEARS = {2020, 2021, 2022, 2023, 2024}
TARGET = "serious_or_fatal"
YEAR = "collision_year"
CORE_REQUIRED = {
    TARGET,
    YEAR,
    "month",
    "hour",
    "is_weekend",
    "is_night",
    "number_of_vehicles",
    "speed_limit",
    "urban_or_rural_area",
    "road_type",
    "light_conditions",
    "weather_conditions",
    "road_surface_conditions",
}


def _format_items(items: Iterable[object]) -> str:
    return ", ".join(str(item) for item in sorted(items, key=str))


def validate(path: Path, enforce_expected_rows: bool = True) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Analysis-ready dataset not found: {path.resolve()}")

    df = pd.read_csv(path, low_memory=False)
    print(f"File: {path.resolve()}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    missing_columns = CORE_REQUIRED.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {_format_items(missing_columns)}")

    if enforce_expected_rows and len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"Unexpected row count: {len(df):,}; expected {EXPECTED_ROWS:,}. "
            "Use --allow-row-count-difference only for a documented alternative dataset."
        )

    target = pd.to_numeric(df[TARGET], errors="coerce")
    invalid_target_count = int(target.isna().sum())
    if invalid_target_count:
        raise ValueError(
            f"Target contains {invalid_target_count:,} missing or non-numeric values"
        )

    target_values = set(target.unique())
    if not target_values.issubset({0, 1}):
        raise ValueError(f"Target must be binary 0/1; found: {_format_items(target_values)}")

    year_values = pd.to_numeric(df[YEAR], errors="coerce")
    invalid_year_count = int(year_values.isna().sum())
    if invalid_year_count:
        raise ValueError(
            f"collision_year contains {invalid_year_count:,} missing or non-numeric values"
        )
    if year_values.mod(1).ne(0).any():
        raise ValueError("collision_year must contain whole-number years")

    years = set(year_values.astype(int).unique())
    if years != EXPECTED_YEARS:
        raise ValueError(
            f"Unexpected study years: {_format_items(years)}; "
            f"expected {_format_items(EXPECTED_YEARS)}"
        )

    if "collision_index" in df.columns:
        collision_index = df["collision_index"]
        missing_identifier_count = int(
            (
                collision_index.isna()
                | collision_index.astype(str).str.strip().eq("")
            ).sum()
        )
        if missing_identifier_count:
            raise ValueError(
                "collision_index contains "
                f"{missing_identifier_count:,} missing or blank values"
            )
        duplicate_count = int(df["collision_index"].duplicated().sum())
        if duplicate_count:
            raise ValueError(f"collision_index contains {duplicate_count:,} duplicates")
        print("collision_index uniqueness: passed")

    positive_rate = target.mean()
    print(f"Positive-class rate: {positive_rate:.4f}")
    print(f"Years: {_format_items(years)}")
    print("Validation passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-ready",
        type=Path,
        default=Path("road_safety_analysis/analysis_ready_road_safety.csv"),
        help="Path to the prepared analysis-ready CSV.",
    )
    parser.add_argument(
        "--allow-row-count-difference",
        action="store_true",
        help="Allow a documented alternative dataset with a different row count.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate(args.analysis_ready, enforce_expected_rows=not args.allow_row_count_difference)


if __name__ == "__main__":
    main()
