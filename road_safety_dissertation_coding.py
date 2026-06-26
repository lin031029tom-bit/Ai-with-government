#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Road Safety Dissertation Coding Script
======================================

Project:
Can Machine Learning Support Evidence-Based Road Safety Policy?
Predicting Serious Road Traffic Collisions in Great Britain Using UK Open Government Data

Central research question:
To what extent can machine learning predict serious or fatal outcomes in reported road traffic
collisions in Great Britain?

What this script does:
1. Loads the analysis-ready dataset if available.
2. If the analysis-ready dataset is not available, it can build it from raw DfT files.
3. Produces descriptive statistics and figures.
4. Trains baseline, logistic regression and random forest models.
5. Evaluates models on a time-based split: 2020-2023 training period, 2024 test period.
6. Runs robustness checks:
   - dummy majority baseline
   - time-based holdout
   - leakage-sensitive variable exclusion
   - threshold sensitivity
   - urban/rural subgroup evaluation
   - optional alternative training sample size
7. Exports tables and figures for dissertation use.

Author: Yi Lin
Programme: MSc AI with Government
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------

DEFAULT_DATA_DIR = Path("road_safety_data")
DEFAULT_ANALYSIS_READY = Path("road_safety_analysis") / "analysis_ready_road_safety.csv"
DEFAULT_OUTPUT_DIR = Path("road_safety_coding_outputs")

TARGET = "serious_or_fatal"


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------
# 2. Data loading and optional analysis-ready construction
# ---------------------------------------------------------------------

def load_data_dictionary(data_dir: Path) -> pd.DataFrame | None:
    """Load the DfT data guide if available."""
    guide = data_dir / "road_safety_open_dataset_data_guide_2024.xlsx"
    if guide.exists():
        try:
            return pd.read_excel(guide, sheet_name="2024_code_list")
        except Exception:
            return None
    return None


def build_analysis_ready_from_raw(data_dir: Path, output_dir: Path) -> pd.DataFrame:
    """
    Build a simplified analysis-ready dataset from raw DfT files.

    Expected raw files:
    - collision_last_5_years.csv
    - vehicle_last_5_years.csv
    - local_authority_traffic.csv

    Casualty data is intentionally not used as an ordinary predictor because it can create
    leakage when predicting collision severity.
    """
    collision_path = data_dir / "collision_last_5_years.csv"
    vehicle_path = data_dir / "vehicle_last_5_years.csv"
    traffic_path = data_dir / "local_authority_traffic.csv"

    if not collision_path.exists():
        raise FileNotFoundError(f"Missing file: {collision_path}")

    collisions = pd.read_csv(collision_path, low_memory=False)

    # Target variable: fatal or serious = 1; slight = 0.
    collisions[TARGET] = collisions["collision_severity"].isin([1, 2]).astype(int)

    # Date/time features.
    collisions["date_parsed"] = pd.to_datetime(collisions["date"], errors="coerce", dayfirst=True)
    collisions["month"] = collisions["date_parsed"].dt.month
    collisions["quarter"] = collisions["date_parsed"].dt.quarter

    # Parse hour from HH:MM.
    collisions["hour"] = pd.to_numeric(collisions["time"].astype(str).str.slice(0, 2), errors="coerce")
    collisions["is_weekend"] = collisions["day_of_week"].isin([1, 7]).astype(int)
    collisions["is_night"] = collisions["hour"].isin(list(range(0, 6)) + list(range(20, 24))).astype(int)

    # Vehicle aggregation.
    if vehicle_path.exists():
        vehicles = pd.read_csv(vehicle_path, low_memory=False)

        vehicle_agg = vehicles.groupby("collision_index").agg(
            vehicle_record_count=("vehicle_reference", "count"),
            vehicle_type_nunique=("vehicle_type", "nunique"),
            mean_driver_age=("age_of_driver", "mean"),
            min_driver_age=("age_of_driver", "min"),
            max_driver_age=("age_of_driver", "max"),
            mean_vehicle_age=("age_of_vehicle", "mean"),
            max_vehicle_age=("age_of_vehicle", "max"),
            n_pedal_cycles=("vehicle_type", lambda x: int((x == 1).sum())),
            n_motorcycles=("vehicle_type", lambda x: int(x.isin([2, 3, 4, 5]).sum())),
            n_cars_taxis=("vehicle_type", lambda x: int(x.isin([9, 10]).sum())),
            n_buses_minibuses=("vehicle_type", lambda x: int(x.isin([11, 19]).sum())),
            n_goods_vehicles=("vehicle_type", lambda x: int(x.isin([20, 21]).sum())),
        ).reset_index()

        vehicles["young_driver_17_24"] = vehicles["age_of_driver"].between(17, 24, inclusive="both")
        vehicles["older_driver_65_plus"] = vehicles["age_of_driver"] >= 65
        driver_agg = vehicles.groupby("collision_index").agg(
            any_young_driver_17_24=("young_driver_17_24", "max"),
            any_older_driver_65_plus=("older_driver_65_plus", "max"),
        ).reset_index()
        driver_agg["any_young_driver_17_24"] = driver_agg["any_young_driver_17_24"].astype(int)
        driver_agg["any_older_driver_65_plus"] = driver_agg["any_older_driver_65_plus"].astype(int)

        vehicle_agg = vehicle_agg.merge(driver_agg, on="collision_index", how="left")
        collisions = collisions.merge(vehicle_agg, on="collision_index", how="left")

    # Traffic exposure merge.
    if traffic_path.exists():
        traffic = pd.read_csv(traffic_path, low_memory=False)

        # Flexible column mapping for DfT local authority traffic data.
        rename_map = {}
        for col in traffic.columns:
            lower = col.lower()
            if lower in ["local_authority_code", "local_authority_ons_code", "ons_code"]:
                rename_map[col] = "local_authority_code"
            elif lower in ["year", "count_date_year"]:
                rename_map[col] = "year"
            elif lower in ["link_length_km", "link_length"]:
                rename_map[col] = "traffic_link_length_km"
            elif lower in ["all_motor_vehicles", "all_mv"]:
                rename_map[col] = "traffic_all_motor_vehicles"
            elif lower in ["cars_and_taxis", "cars_taxis"]:
                rename_map[col] = "traffic_cars_taxis"
        traffic = traffic.rename(columns=rename_map)

        required = {"local_authority_code", "year"}
        if required.issubset(set(traffic.columns)):
            traffic["year"] = pd.to_numeric(traffic["year"], errors="coerce")
            collisions = collisions.merge(
                traffic,
                left_on=["local_authority_highway", "collision_year"],
                right_on=["local_authority_code", "year"],
                how="left",
                suffixes=("", "_traffic")
            )

            if "traffic_all_motor_vehicles" in collisions.columns and "traffic_link_length_km" in collisions.columns:
                collisions["traffic_all_motor_vehicles_per_km"] = (
                    collisions["traffic_all_motor_vehicles"] /
                    collisions["traffic_link_length_km"].replace({0: np.nan})
                )
            if "traffic_cars_taxis" in collisions.columns and "traffic_all_motor_vehicles" in collisions.columns:
                collisions["traffic_cars_taxis_share"] = (
                    collisions["traffic_cars_taxis"] /
                    collisions["traffic_all_motor_vehicles"].replace({0: np.nan})
                )
            collisions["traffic_merge_matched"] = collisions["local_authority_code"].notna().astype(int)

    safe_mkdir(output_dir)
    out_path = output_dir / "analysis_ready_road_safety.csv"
    collisions.to_csv(out_path, index=False)
    return collisions


def load_analysis_ready(analysis_ready_path: Path, data_dir: Path, output_dir: Path) -> pd.DataFrame:
    if analysis_ready_path.exists():
        print(f"Loading analysis-ready dataset: {analysis_ready_path}")
        return pd.read_csv(analysis_ready_path, low_memory=False)
    print("Analysis-ready dataset not found. Building from raw files...")
    return build_analysis_ready_from_raw(data_dir, output_dir)


# ---------------------------------------------------------------------
# 3. Descriptive analysis
# ---------------------------------------------------------------------

def make_rate_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    table = df.groupby(group_col, dropna=False).agg(
        collisions=("collision_index", "count"),
        serious_or_fatal=(TARGET, "sum"),
    ).reset_index()
    table["slight"] = table["collisions"] - table["serious_or_fatal"]
    table["serious_fatal_rate_pct"] = (table["serious_or_fatal"] / table["collisions"] * 100).round(2)
    return table


def run_descriptive_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    safe_mkdir(tables_dir)
    safe_mkdir(figures_dir)

    summary = {
        "n_records": int(len(df)),
        "year_min": int(df["collision_year"].min()),
        "year_max": int(df["collision_year"].max()),
        "serious_or_fatal_count": int(df[TARGET].sum()),
        "slight_count": int((df[TARGET] == 0).sum()),
        "serious_or_fatal_rate_pct": round(float(df[TARGET].mean() * 100), 2),
    }
    save_json(summary, tables_dir / "dataset_summary.json")

    severity = df.groupby("collision_severity").agg(
        collisions=("collision_index", "count")
    ).reset_index()
    severity["percentage"] = (severity["collisions"] / severity["collisions"].sum() * 100).round(2)
    severity.to_csv(tables_dir / "severity_distribution.csv", index=False)

    yearly = make_rate_table(df, "collision_year").sort_values("collision_year")
    yearly.to_csv(tables_dir / "yearly_serious_fatal_rate.csv", index=False)

    for col in ["road_type", "speed_limit", "light_conditions", "weather_conditions", "urban_or_rural_area"]:
        if col in df.columns:
            make_rate_table(df, col).to_csv(tables_dir / f"{col}_serious_fatal_rate.csv", index=False)

    # Figure 1: Severity distribution.
    fig = plt.figure(figsize=(8, 5))
    plt.bar(severity["collision_severity"].astype(str), severity["collisions"])
    plt.xlabel("Collision severity code")
    plt.ylabel("Number of collisions")
    plt.title("Distribution of Collision Severity, 2020-2024")
    plt.tight_layout()
    plt.savefig(figures_dir / "figure_1_severity_distribution.png", dpi=300)
    plt.close(fig)

    # Figure 2: Yearly serious/fatal rate.
    fig = plt.figure(figsize=(8, 5))
    plt.plot(yearly["collision_year"], yearly["serious_fatal_rate_pct"], marker="o")
    plt.xlabel("Year")
    plt.ylabel("Serious/fatal rate (%)")
    plt.title("Serious/Fatal Collision Rate by Year")
    plt.tight_layout()
    plt.savefig(figures_dir / "figure_2_yearly_rate.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------
# 4. Modelling
# ---------------------------------------------------------------------

def modelling_features(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_features = [
        "month", "hour", "is_weekend", "is_night",
        "longitude", "latitude",
        "number_of_vehicles",
        "speed_limit",
        "traffic_link_length_km",
        "traffic_all_motor_vehicles",
        "traffic_all_motor_vehicles_per_km",
        "traffic_cars_taxis_share",
        "vehicle_record_count",
        "n_pedal_cycles", "n_motorcycles", "n_cars_taxis",
        "n_buses_minibuses", "n_goods_vehicles",
        "vehicle_type_nunique",
        "mean_driver_age", "min_driver_age", "max_driver_age",
        "any_young_driver_17_24", "any_older_driver_65_plus",
        "mean_vehicle_age", "max_vehicle_age",
    ]

    categorical_features = [
        "day_of_week",
        "police_force",
        "local_authority_highway",
        "urban_or_rural_area",
        "first_road_class",
        "road_type",
        "junction_detail",
        "junction_control",
        "pedestrian_crossing",
        "light_conditions",
        "weather_conditions",
        "road_surface_conditions",
        "special_conditions_at_site",
        "carriageway_hazards",
        "trunk_road_flag",
    ]

    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]
    return numeric_features, categorical_features


def build_preprocessor(numeric_features: List[str], categorical_features: List[str], scale_numeric: bool) -> ColumnTransformer:
    if scale_numeric:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
    else:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=True)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore")

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric_features),
        ("categorical", categorical_pipe, categorical_features),
    ])


def evaluate_model(name: str, y_true: pd.Series, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    out = {
        "model": name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_prob)) > 1 else np.nan,
        "average_precision": average_precision_score(y_true, y_prob) if len(np.unique(y_prob)) > 1 else np.nan,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
    return out


def prepare_train_test(df: pd.DataFrame, sample_size: int, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str], List[str]]:
    train_df = df[df["collision_year"].between(2020, 2023)].copy()
    test_df = df[df["collision_year"] == 2024].copy()

    # Stratified sample for dissertation feasibility.
    if sample_size and sample_size < len(train_df):
        train_df = train_df.groupby(TARGET, group_keys=False).apply(
            lambda x: x.sample(
                n=max(1, int(sample_size * len(x) / len(train_df))),
                random_state=random_state
            )
        ).sample(frac=1, random_state=random_state)

    numeric_features, categorical_features = modelling_features(df)
    features = numeric_features + categorical_features

    X_train = train_df[features].copy()
    y_train = train_df[TARGET].astype(int)
    X_test = test_df[features].copy()
    y_test = test_df[TARGET].astype(int)

    # Treat common missing/unknown numerical codes as missing.
    for col in ["speed_limit", "mean_driver_age", "min_driver_age", "max_driver_age", "mean_vehicle_age", "max_vehicle_age"]:
        if col in X_train.columns:
            X_train[col] = X_train[col].replace([-1, 99], np.nan)
            X_test[col] = X_test[col].replace([-1, 99], np.nan)

    for col in categorical_features:
        X_train[col] = X_train[col].astype("object").where(~X_train[col].isna(), "missing").astype(str)
        X_test[col] = X_test[col].astype("object").where(~X_test[col].isna(), "missing").astype(str)

    return X_train, y_train, X_test, y_test, numeric_features, categorical_features


def run_models(df: pd.DataFrame, output_dir: Path, sample_size: int = 15000, random_state: int = 42) -> None:
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    safe_mkdir(tables_dir)
    safe_mkdir(figures_dir)

    X_train, y_train, X_test, y_test, numeric_features, categorical_features = prepare_train_test(
        df, sample_size=sample_size, random_state=random_state
    )

    preprocess_scaled = build_preprocessor(numeric_features, categorical_features, scale_numeric=True)
    preprocess_unscaled = build_preprocessor(numeric_features, categorical_features, scale_numeric=False)

    models = {
        "Dummy majority baseline": Pipeline([
            ("preprocess", preprocess_unscaled),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]),
        "Logistic Regression (SGD)": Pipeline([
            ("preprocess", preprocess_scaled),
            ("model", SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=0.0001,
                class_weight="balanced",
                max_iter=1000,
                random_state=random_state,
            )),
        ]),
        "Random Forest": Pipeline([
            ("preprocess", preprocess_unscaled),
            ("model", RandomForestClassifier(
                n_estimators=100,
                max_depth=16,
                min_samples_leaf=50,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=random_state,
            )),
        ]),
    }

    results = []
    probs = {}

    for name, pipeline in models.items():
        print(f"Training: {name}")
        pipeline.fit(X_train, y_train)

        if hasattr(pipeline.named_steps["model"], "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_prob = pipeline.decision_function(X_test)

        # Dummy classifier probability can be constant.
        probs[name] = y_prob
        results.append(evaluate_model(name, y_test, y_prob, threshold=0.5))

        # Feature importance for Random Forest.
        if name == "Random Forest":
            try:
                pre = pipeline.named_steps["preprocess"]
                feature_names = list(numeric_features)
                ohe = pre.named_transformers_["categorical"].named_steps["onehot"]
                feature_names += list(ohe.get_feature_names_out(categorical_features))
                importances = pipeline.named_steps["model"].feature_importances_
                imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
                imp_df = imp_df.sort_values("importance", ascending=False)
                imp_df.to_csv(tables_dir / "random_forest_feature_importance.csv", index=False)

                top = imp_df.head(15)
                fig = plt.figure(figsize=(9, 6))
                plt.barh(top["feature"][::-1], top["importance"][::-1])
                plt.xlabel("Importance")
                plt.title("Top Random Forest Feature Importances")
                plt.tight_layout()
                plt.savefig(figures_dir / "figure_random_forest_feature_importance.png", dpi=300)
                plt.close(fig)
            except Exception as e:
                print(f"Could not export feature importance: {e}")

    perf = pd.DataFrame(results)
    for col in ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]:
        perf[col] = perf[col].round(4)
    perf.to_csv(tables_dir / "model_performance_2024_test.csv", index=False)

    # ROC curves.
    fig = plt.figure(figsize=(8, 6))
    for name, y_prob in probs.items():
        if len(np.unique(y_prob)) <= 1:
            continue
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves on 2024 Test Set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "figure_roc_curves.png", dpi=300)
    plt.close(fig)

    # Threshold sensitivity for Random Forest.
    if "Random Forest" in probs:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        threshold_rows = [evaluate_model("Random Forest", y_test, probs["Random Forest"], threshold=t) for t in thresholds]
        threshold_df = pd.DataFrame(threshold_rows)
        for col in ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]:
            threshold_df[col] = threshold_df[col].round(4)
        threshold_df.to_csv(tables_dir / "threshold_sensitivity_random_forest.csv", index=False)

    # Subgroup evaluation: urban/rural.
    if "urban_or_rural_area" in X_test.columns and "Random Forest" in probs:
        subgroup_rows = []
        for subgroup in X_test["urban_or_rural_area"].unique():
            mask = X_test["urban_or_rural_area"] == subgroup
            if mask.sum() > 100:
                row = evaluate_model(f"Random Forest - urban_rural_{subgroup}", y_test[mask], probs["Random Forest"][mask], threshold=0.5)
                row["subgroup"] = subgroup
                row["n_records"] = int(mask.sum())
                subgroup_rows.append(row)
        subgroup_df = pd.DataFrame(subgroup_rows)
        subgroup_df.to_csv(tables_dir / "subgroup_evaluation_urban_rural.csv", index=False)


# ---------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Road safety dissertation coding script")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="Folder containing raw DfT files")
    parser.add_argument("--analysis-ready", type=str, default=str(DEFAULT_ANALYSIS_READY), help="Path to analysis-ready CSV")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output folder")
    parser.add_argument("--sample-size", type=int, default=15000, help="Stratified training sample size")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    analysis_ready_path = Path(args.analysis_ready)
    output_dir = Path(args.output_dir)

    safe_mkdir(output_dir)

    df = load_analysis_ready(analysis_ready_path, data_dir, output_dir)
    run_descriptive_analysis(df, output_dir)
    run_models(df, output_dir, sample_size=args.sample_size)

    print("Coding pipeline completed.")
    print(f"Outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
