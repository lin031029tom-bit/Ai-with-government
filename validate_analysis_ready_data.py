#!/usr/bin/env python3
"""Validate the prepared road-safety analysis dataset before modelling."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from analysis_schema import (
    CATEGORICAL_FEATURES,
    CORE_REQUIRED,
    DISSERTATION_REQUIRED,
    EXPECTED_DATASET_SHA256,
    EXPECTED_ROWS,
    EXPECTED_YEARS,
    MODEL_FEATURES,
    TARGET,
    YEAR,
)


def _format_items(items: Iterable[object]) -> str:
    return ", ".join(str(item) for item in sorted(items, key=str))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_binary(series: pd.Series, name: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    invalid_count = int(values.isna().sum())
    if invalid_count:
        raise ValueError(
            f"{name} contains {invalid_count:,} missing or non-numeric values"
        )
    found = set(values.unique())
    if not found.issubset({0, 1}):
        raise ValueError(
            f"{name} must be binary 0/1; found: {_format_items(found)}"
        )
    return values.astype("int8")


def _validated_whole_numbers(series: pd.Series, name: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    invalid_count = int(values.isna().sum())
    if invalid_count:
        raise ValueError(
            f"{name} contains {invalid_count:,} missing or non-numeric values"
        )
    if values.mod(1).ne(0).any():
        raise ValueError(f"{name} must contain whole-number values")
    return values.astype(int)


def _validate_traffic_merge(df: pd.DataFrame) -> None:
    if "traffic_merge_matched" not in df.columns:
        return

    matched = _validated_binary(df["traffic_merge_matched"], "traffic_merge_matched")
    matched_count = int(matched.sum())
    match_rate = float(matched.mean())

    if "traffic_local_authority_name" in df.columns:
        missing_name = df["traffic_local_authority_name"].isna() | (
            df["traffic_local_authority_name"].astype(str).str.strip() == ""
        )
        matched_without_name = int((matched.eq(1) & missing_name).sum())
        if matched_without_name:
            raise ValueError(
                "traffic_local_authority_name is missing for "
                f"{matched_without_name:,} traffic-matched records"
            )

        mappings_per_code = (
            df.loc[matched.eq(1)]
            .groupby("local_authority_highway")["traffic_local_authority_name"]
            .nunique(dropna=True)
        )
        conflicting_codes = mappings_per_code[mappings_per_code > 1]
        if not conflicting_codes.empty:
            raise ValueError(
                "traffic_local_authority_name has conflicting names for highway "
                f"codes: {_format_items(conflicting_codes.index)}"
            )

    print(
        "Traffic-context merge: "
        f"{matched_count:,}/{len(df):,} matched ({match_rate:.4%})"
    )


def validate(
    path: Path,
    enforce_expected_rows: bool = True,
    enforce_expected_features: bool = True,
    enforce_expected_hash: bool | None = None,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Analysis-ready dataset not found: {path.resolve()}")

    if enforce_expected_hash is None:
        enforce_expected_hash = enforce_expected_rows and enforce_expected_features
    dataset_sha256 = sha256_file(path)
    if enforce_expected_hash and dataset_sha256 != EXPECTED_DATASET_SHA256:
        raise ValueError(
            "Unexpected dataset SHA-256: "
            f"{dataset_sha256}; expected {EXPECTED_DATASET_SHA256}. "
            "Use the documented alternative-data flags only when intentionally "
            "running a different prepared dataset."
        )

    df = pd.read_csv(path, low_memory=False)
    df.attrs["dataset_sha256"] = dataset_sha256
    print(f"File: {path.resolve()}")
    print(f"SHA-256: {dataset_sha256}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    missing_columns = CORE_REQUIRED.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing core required columns: {_format_items(missing_columns)}"
        )

    if enforce_expected_features:
        missing_dissertation_columns = DISSERTATION_REQUIRED.difference(df.columns)
        if missing_dissertation_columns:
            raise ValueError(
                "Missing dissertation reproduction columns: "
                f"{_format_items(missing_dissertation_columns)}"
            )

    if enforce_expected_rows and len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"Unexpected row count: {len(df):,}; expected {EXPECTED_ROWS:,}. "
            "Use --allow-row-count-difference only for a documented alternative dataset."
        )

    target = _validated_binary(df[TARGET], "Target")

    severity = _validated_whole_numbers(
        df["collision_severity"], "collision_severity"
    )
    severity_values = set(severity.unique())
    if not severity_values.issubset({1, 2, 3}):
        raise ValueError(
            "collision_severity must use official codes 1/2/3; "
            f"found: {_format_items(severity_values)}"
        )
    expected_target = severity.isin({1, 2}).astype("int8")
    target_mismatch_count = int(target.ne(expected_target).sum())
    if target_mismatch_count:
        raise ValueError(
            "Target does not match collision_severity for "
            f"{target_mismatch_count:,} records"
        )

    year_values = _validated_whole_numbers(df[YEAR], "collision_year")
    years = set(year_values.unique())
    if years != EXPECTED_YEARS:
        raise ValueError(
            f"Unexpected study years: {_format_items(years)}; "
            f"expected {_format_items(EXPECTED_YEARS)}"
        )

    target_by_year = pd.DataFrame({YEAR: year_values, TARGET: target})
    single_class_years = {
        year: sorted(group[TARGET].unique().tolist())
        for year, group in target_by_year.groupby(YEAR)
        if set(group[TARGET].unique()) != {0, 1}
    }
    if single_class_years:
        details = "; ".join(
            f"{year}: {_format_items(classes)}"
            for year, classes in single_class_years.items()
        )
        raise ValueError(
            "Each study year must contain both target classes 0 and 1; "
            f"found {details}"
        )

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
    duplicate_count = int(collision_index.duplicated().sum())
    if duplicate_count:
        raise ValueError(f"collision_index contains {duplicate_count:,} duplicates")
    print("collision_index uniqueness: passed")

    for column in MODEL_FEATURES:
        if column not in df.columns:
            continue
        if column in CATEGORICAL_FEATURES:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        non_numeric_count = int((df[column].notna() & numeric.isna()).sum())
        if non_numeric_count:
            raise ValueError(
                f"{column} contains {non_numeric_count:,} non-numeric values"
            )
        non_finite_count = int(
            np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum()
        )
        if non_finite_count:
            raise ValueError(
                f"{column} contains {non_finite_count:,} infinite values"
            )

    _validate_traffic_merge(df)

    positive_rate = target.mean()
    print(f"Positive-class rate: {positive_rate:.4f}")
    print(f"Years: {_format_items(years)}")
    print(
        "Model features: "
        f"{sum(column in df.columns for column in MODEL_FEATURES)}/"
        f"{len(MODEL_FEATURES)} present"
    )
    print("Validation passed.")
    return df


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
    parser.add_argument(
        "--allow-feature-set-difference",
        action="store_true",
        help=(
            "Allow a documented alternative dataset without the complete "
            "dissertation feature and audit schema."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate(
        args.analysis_ready,
        enforce_expected_rows=not args.allow_row_count_difference,
        enforce_expected_features=not args.allow_feature_set_difference,
    )


if __name__ == "__main__":
    main()
