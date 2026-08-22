#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproducible coding workflow for the road-safety dissertation.

Research question
-----------------
To what extent can machine learning predict serious or fatal outcomes in reported
road traffic collisions in Great Britain?

The task is retrospective severity classification conditional on a collision having
occurred and been reported. It is not a model of future collision occurrence.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analysis_schema import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    UNKNOWN_VALUE_MAP,
)
from validate_analysis_ready_data import sha256_file, validate

RANDOM_STATE = 42
DEFAULT_BOOTSTRAP_ITERATIONS = 1000
DEFAULT_ANALYSIS_READY = Path("road_safety_analysis/analysis_ready_road_safety.csv")
DEFAULT_OUTPUT_DIR = Path("road_safety_coding_outputs")

ROAD_TYPE_LABELS = {
    -1: "Unknown",
    1: "Roundabout",
    2: "One way street",
    3: "Dual carriageway",
    6: "Single carriageway",
    7: "Slip road",
    9: "Unknown",
}
LIGHT_CONDITION_LABELS = {
    -1: "Unknown",
    1: "Daylight",
    4: "Darkness - lights lit",
    5: "Darkness - lights unlit",
    6: "Darkness - no lighting",
    7: "Darkness - lighting unknown",
}
WEATHER_CONDITION_LABELS = {
    -1: "Unknown",
    1: "Fine no high winds",
    2: "Raining no high winds",
    3: "Snowing no high winds",
    4: "Fine with high winds",
    5: "Raining with high winds",
    6: "Snowing with high winds",
    7: "Fog or mist",
    8: "Other",
    9: "Unknown",
}


def mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_data(
    path: Path,
    enforce_expected_rows: bool = True,
    enforce_expected_features: bool = True,
) -> pd.DataFrame:
    return validate(
        path,
        enforce_expected_rows=enforce_expected_rows,
        enforce_expected_features=enforce_expected_features,
    )


def feature_lists(
    df: pd.DataFrame,
    require_all: bool = True,
) -> Tuple[List[str], List[str]]:
    missing = [column for column in MODEL_FEATURES if column not in df.columns]
    if require_all and missing:
        raise ValueError(
            "Missing required dissertation model features: "
            f"{', '.join(missing)}"
        )

    numeric = [c for c in NUMERIC_FEATURES if c in df.columns]
    categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    if not numeric and not categorical:
        raise ValueError("No modelling features were found in the analysis-ready data.")
    return numeric, categorical


def validated_binary_target(frame: pd.DataFrame) -> pd.Series:
    target = pd.to_numeric(frame[TARGET], errors="coerce")
    invalid_count = int(target.isna().sum())
    if invalid_count:
        raise ValueError(
            f"Target contains {invalid_count:,} missing or non-numeric values"
        )
    found = set(target.unique())
    if not found.issubset({0, 1}):
        raise ValueError(f"Target must be binary 0/1; found: {sorted(found)}")
    return target.astype("int8")


def require_both_target_classes(target: pd.Series, split_name: str) -> None:
    found = set(pd.to_numeric(target, errors="coerce").dropna().unique())
    if found != {0, 1}:
        raise ValueError(
            f"{split_name} must contain both target classes 0 and 1; "
            f"found: {sorted(found)}"
        )


def clean_feature_frame(
    frame: pd.DataFrame,
    numeric: Sequence[str],
    categorical: Sequence[str],
) -> pd.DataFrame:
    out = frame[list(numeric) + list(categorical)].copy()

    for col, unknown_values in UNKNOWN_VALUE_MAP.items():
        if col in out.columns:
            out[col] = out[col].replace(unknown_values, np.nan)

    for col in categorical:
        out[col] = (
            out[col]
            .astype("object")
            .where(out[col].notna(), "missing")
            .astype(str)
        )

    return out


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(
            handle_unknown="ignore", min_frequency=50, sparse_output=True
        )
    except TypeError:  # scikit-learn < 1.2 compatibility
        return OneHotEncoder(handle_unknown="ignore")


def preprocessor(
    numeric: Sequence[str], categorical: Sequence[str], scale_numeric: bool
) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, list(numeric)),
            ("categorical", categorical_pipeline, list(categorical)),
        ]
    )


def stratified_sample(frame: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    if n is None or n >= len(frame):
        return frame.copy()
    if n < 2:
        raise ValueError("Sample size must be at least 2")
    sample, _ = train_test_split(
        frame, train_size=n, stratify=frame[TARGET], random_state=seed
    )
    return sample.copy()


def make_models(numeric: Sequence[str], categorical: Sequence[str], seed: int) -> Dict[str, Pipeline]:
    return {
        "Dummy prevalence baseline": Pipeline(
            [
                ("preprocessor", preprocessor(numeric, categorical, False)),
                ("model", DummyClassifier(strategy="prior")),
            ]
        ),
        "Balanced logistic regression": Pipeline(
            [
                ("preprocessor", preprocessor(numeric, categorical, True)),
                (
                    "model",
                    LogisticRegression(
                        solver="liblinear",
                        class_weight="balanced",
                        C=1.0,
                        max_iter=1000,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocessor", preprocessor(numeric, categorical, False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=16,
                        min_samples_leaf=50,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def evaluate(
    name: str,
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> Dict:
    prediction = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    has_both_classes = set(pd.Series(y_true).unique()) == {0, 1}
    return {
        "model": name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, prediction),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall": recall_score(y_true, prediction, zero_division=0),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "roc_auc": (
            roc_auc_score(y_true, probabilities) if has_both_classes else np.nan
        ),
        "average_precision": (
            average_precision_score(y_true, probabilities)
            if has_both_classes
            else np.nan
        ),
        "brier_score": brier_score_loss(y_true, probabilities),
        "log_loss": (
            log_loss(y_true, probabilities, labels=[0, 1])
            if has_both_classes
            else np.nan
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


UNCERTAINTY_METRICS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
    "brier_score",
)


def stratified_bootstrap_indices(
    y_true: pd.Series,
    iterations: int,
    seed: int,
) -> Iterable[np.ndarray]:
    """Yield class-stratified bootstrap samples with the observed class counts."""
    if iterations < 1:
        raise ValueError("Bootstrap iterations must be at least 1")
    y_array = np.asarray(y_true, dtype=int)
    negative = np.flatnonzero(y_array == 0)
    positive = np.flatnonzero(y_array == 1)
    if not len(negative) or not len(positive):
        raise ValueError("Bootstrap confidence intervals require both target classes")
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        sampled = np.concatenate(
            [
                rng.choice(negative, size=len(negative), replace=True),
                rng.choice(positive, size=len(positive), replace=True),
            ]
        )
        rng.shuffle(sampled)
        yield sampled


def metric_uncertainty_outputs(
    y_true: pd.Series,
    probabilities: Dict[str, np.ndarray],
    tables: Path,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Write paired stratified-bootstrap CIs for metrics and model differences."""
    model_names = [
        name
        for name in ("Balanced logistic regression", "Random Forest")
        if name in probabilities
    ]
    if len(model_names) != 2:
        raise ValueError(
            "Uncertainty analysis requires logistic-regression and Random-Forest "
            "probabilities"
        )

    y_array = np.asarray(y_true, dtype=int)
    probability_arrays = {
        name: np.asarray(probabilities[name], dtype=float) for name in model_names
    }
    if any(len(values) != len(y_array) for values in probability_arrays.values()):
        raise ValueError("Prediction probabilities must align with the test target")

    point_estimates = {
        name: evaluate(name, y_array, probability_arrays[name])
        for name in model_names
    }
    bootstrap_values = {
        name: {metric: [] for metric in UNCERTAINTY_METRICS}
        for name in model_names
    }
    difference_values = {
        metric: [] for metric in UNCERTAINTY_METRICS
    }

    for sampled in stratified_bootstrap_indices(y_true, iterations, seed):
        sampled_y = y_array[sampled]
        sampled_rows = {}
        for name in model_names:
            sampled_rows[name] = evaluate(
                name,
                sampled_y,
                probability_arrays[name][sampled],
            )
            for metric in UNCERTAINTY_METRICS:
                bootstrap_values[name][metric].append(sampled_rows[name][metric])
        for metric in UNCERTAINTY_METRICS:
            difference_values[metric].append(
                sampled_rows["Random Forest"][metric]
                - sampled_rows["Balanced logistic regression"][metric]
            )

    interval_rows = []
    for name in model_names:
        for metric in UNCERTAINTY_METRICS:
            values = np.asarray(bootstrap_values[name][metric], dtype=float)
            interval_rows.append(
                {
                    "model": name,
                    "metric": metric,
                    "estimate": point_estimates[name][metric],
                    "ci_95_lower": np.quantile(values, 0.025),
                    "ci_95_upper": np.quantile(values, 0.975),
                    "bootstrap_iterations": iterations,
                    "bootstrap_design": "class-stratified paired resampling",
                }
            )
    intervals = pd.DataFrame(interval_rows)
    intervals.to_csv(
        tables / "table_4_5_metric_uncertainty_2024.csv",
        index=False,
    )

    difference_rows = []
    for metric in UNCERTAINTY_METRICS:
        values = np.asarray(difference_values[metric], dtype=float)
        estimate = (
            point_estimates["Random Forest"][metric]
            - point_estimates["Balanced logistic regression"][metric]
        )
        difference_rows.append(
            {
                "metric": metric,
                "difference_random_forest_minus_logistic": estimate,
                "ci_95_lower": np.quantile(values, 0.025),
                "ci_95_upper": np.quantile(values, 0.975),
                "bootstrap_iterations": iterations,
                "interpretation": (
                    "positive favours Random Forest"
                    if metric != "brier_score"
                    else "negative favours Random Forest"
                ),
            }
        )
    differences = pd.DataFrame(difference_rows)
    differences.to_csv(
        tables / "table_4_5_paired_model_differences_2024.csv",
        index=False,
    )
    return intervals, differences


def rate_table(df: pd.DataFrame, group: str) -> pd.DataFrame:
    table = df.groupby(group, dropna=False).agg(
        collisions=("collision_index", "count"), serious_or_fatal=(TARGET, "sum")
    ).reset_index()
    table["slight"] = table["collisions"] - table["serious_or_fatal"]
    table["serious_fatal_rate_pct"] = (
        table["serious_or_fatal"] / table["collisions"] * 100
    ).round(2)
    return table


def local_authority_rate_table(
    df: pd.DataFrame, min_collisions: int = 500
) -> pd.DataFrame:
    if min_collisions < 1:
        raise ValueError("min_collisions must be at least 1")
    table = rate_table(df, "local_authority_highway")

    if "traffic_local_authority_name" in df.columns:
        authority_names = (
            df.loc[
                df["traffic_local_authority_name"].notna(),
                ["local_authority_highway", "traffic_local_authority_name"],
            ]
            .drop_duplicates()
        )
        conflicting_codes = (
            authority_names.groupby("local_authority_highway")[
                "traffic_local_authority_name"
            ]
            .nunique()
            .loc[lambda counts: counts > 1]
        )
        if not conflicting_codes.empty:
            raise ValueError(
                "Conflicting traffic local-authority names for highway codes: "
                f"{', '.join(map(str, conflicting_codes.index))}"
            )
        name_map = authority_names.set_index("local_authority_highway")[
            "traffic_local_authority_name"
        ]
        table.insert(
            1,
            "local_authority_name",
            table["local_authority_highway"].map(name_map),
        )

    return (
        table[table["collisions"] >= min_collisions]
        .sort_values(
            ["serious_fatal_rate_pct", "collisions"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )


def descriptive_outputs(df: pd.DataFrame, tables: Path, figures: Path) -> None:
    severity = df.groupby("collision_severity").agg(
        collisions=("collision_index", "count")
    ).reset_index()
    severity["severity"] = severity["collision_severity"].map(
        {1: "Fatal", 2: "Serious", 3: "Slight"}
    )
    severity["percentage"] = (severity["collisions"] / len(df) * 100).round(2)
    severity.to_csv(tables / "table_4_1_severity_distribution.csv", index=False)

    yearly = rate_table(df, "collision_year").sort_values("collision_year")
    yearly.to_csv(tables / "table_4_2_yearly_serious_fatal_rate.csv", index=False)

    if "road_type" in df.columns:
        road_type = rate_table(df, "road_type").sort_values("road_type")
        road_type.insert(
            1, "road_type_label", road_type["road_type"].map(ROAD_TYPE_LABELS)
        )
        road_type.to_csv(
            tables / "road_type_serious_fatal_rate.csv", index=False
        )

    if "light_conditions" in df.columns:
        light = rate_table(df, "light_conditions").sort_values(
            "light_conditions"
        )
        light.insert(
            1,
            "light_conditions_label",
            light["light_conditions"].map(LIGHT_CONDITION_LABELS),
        )
        light.to_csv(
            tables / "light_conditions_serious_fatal_rate.csv", index=False
        )

    if "weather_conditions" in df.columns:
        weather = rate_table(df, "weather_conditions").sort_values(
            "weather_conditions"
        )
        weather.insert(
            1,
            "weather_conditions_label",
            weather["weather_conditions"].map(WEATHER_CONDITION_LABELS),
        )
        weather.to_csv(
            tables / "weather_conditions_serious_fatal_rate.csv", index=False
        )

    speed = None
    if "speed_limit" in df.columns:
        speed = rate_table(df, "speed_limit").sort_values("speed_limit")
        speed.to_csv(
            tables / "speed_limit_serious_fatal_rate.csv", index=False
        )

    urban = None
    if "urban_or_rural_area" in df.columns:
        urban = rate_table(df, "urban_or_rural_area")
        urban["area"] = urban["urban_or_rural_area"].map(
            {1: "Urban", 2: "Rural", 3: "Unallocated"}
        )
        urban.to_csv(
            tables / "urban_rural_serious_fatal_rate.csv", index=False
        )
    if "local_authority_highway" in df.columns:
        local_authority_rate_table(df).to_csv(
            tables / "local_authority_serious_fatal_rate_min_500.csv",
            index=False,
        )

    plt.figure(figsize=(8, 5))
    plt.bar(severity["severity"], severity["collisions"])
    plt.xlabel("Collision severity"); plt.ylabel("Number of collisions")
    plt.title("Distribution of Collision Severity, 2020–2024")
    plt.tight_layout(); plt.savefig(figures / "figure_4_1_severity_distribution.png", dpi=300); plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(yearly["collision_year"], yearly["serious_fatal_rate_pct"], marker="o")
    plt.xlabel("Year"); plt.ylabel("Serious/fatal rate (%)")
    plt.title("Serious/Fatal Collision Rate by Year")
    plt.tight_layout(); plt.savefig(figures / "figure_4_2_yearly_rate.png", dpi=300); plt.close()

    if speed is not None:
        speed_plot = speed[
            speed["speed_limit"].isin([20, 30, 40, 50, 60, 70])
        ]
        plt.figure(figsize=(8, 5))
        plt.bar(
            speed_plot["speed_limit"].astype(str),
            speed_plot["serious_fatal_rate_pct"],
        )
        plt.xlabel("Speed limit"); plt.ylabel("Serious/fatal rate (%)")
        plt.title("Serious/Fatal Rate by Speed Limit")
        plt.tight_layout()
        plt.savefig(
            figures / "figure_4_3_speed_limit_rate.png", dpi=300
        )
        plt.close()

    if urban is not None:
        urban_plot = urban[urban["area"].isin(["Urban", "Rural"])]
        plt.figure(figsize=(8, 5))
        plt.barh(urban_plot["area"], urban_plot["serious_fatal_rate_pct"])
        plt.xlabel("Serious/fatal rate (%)"); plt.ylabel("Area type")
        plt.title("Serious/Fatal Rate by Urban/Rural Area")
        plt.tight_layout()
        plt.savefig(
            figures / "figure_4_4_urban_rural_rate.png", dpi=300
        )
        plt.close()


def fit_main_models(
    df: pd.DataFrame,
    tables: Path,
    figures: Path,
    sample_size: int | None,
    seed: int,
    require_all_features: bool = True,
) -> Tuple[Dict[str, Pipeline], pd.DataFrame, pd.DataFrame, pd.Series, Dict[str, np.ndarray]]:
    numeric, categorical = feature_lists(df, require_all=require_all_features)
    train_all = df[df["collision_year"].between(2020, 2023)].copy()
    test = df[df["collision_year"] == 2024].copy()
    if train_all.empty or test.empty:
        raise ValueError(
            "The primary temporal split requires training records from 2020-2023 "
            "and test records from 2024"
        )
    train = stratified_sample(train_all, sample_size, seed)

    X_train = clean_feature_frame(train, numeric, categorical)
    X_test = clean_feature_frame(test, numeric, categorical)
    y_train = validated_binary_target(train)
    y_test = validated_binary_target(test)
    require_both_target_classes(y_train, "Primary training split")
    require_both_target_classes(y_test, "Primary 2024 test split")

    models = make_models(numeric, categorical, seed)
    rows: List[Dict] = []
    probabilities: Dict[str, np.ndarray] = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_test)[:, 1]
        probabilities[name] = probability
        rows.append(evaluate(name, y_test, probability))

    performance = pd.DataFrame(rows)
    performance.to_csv(tables / "table_4_3_model_performance_2024_test.csv", index=False)
    performance[["model", "true_negative", "false_positive", "false_negative", "true_positive"]].to_csv(
        tables / "table_4_4_confusion_matrix_counts.csv", index=False
    )

    fitted = performance[performance["model"] != "Dummy prevalence baseline"]
    x = np.arange(len(fitted)); width = 0.15
    plt.figure(figsize=(9, 5.5))
    for i, metric in enumerate(["precision", "recall", "f1", "roc_auc", "average_precision"]):
        plt.bar(x + (i - 2) * width, fitted[metric], width, label=metric)
    plt.xticks(x, fitted["model"], rotation=12, ha="right")
    plt.ylabel("Score"); plt.ylim(0, 0.75); plt.title("Model Performance on the 2024 Test Set")
    plt.legend(ncol=2); plt.tight_layout()
    plt.savefig(figures / "figure_4_5_model_performance.png", dpi=300); plt.close()

    plt.figure(figsize=(7, 5.5))
    for name in ["Balanced logistic regression", "Random Forest"]:
        fpr, tpr, _ = roc_curve(y_test, probabilities[name])
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, probabilities[name]):.3f})")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves on the 2024 Test Set"); plt.legend(); plt.tight_layout()
    plt.savefig(figures / "figure_4_6_roc_curves.png", dpi=300); plt.close()

    plt.figure(figsize=(7, 5.5))
    for name in ["Balanced logistic regression", "Random Forest"]:
        precision, recall, _ = precision_recall_curve(y_test, probabilities[name])
        plt.plot(recall, precision, label=f"{name} (AP={average_precision_score(y_test, probabilities[name]):.3f})")
    plt.axhline(y_test.mean(), linestyle="--", label=f"Positive prevalence ({y_test.mean():.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curves")
    plt.legend(); plt.tight_layout()
    plt.savefig(figures / "supplementary_precision_recall_curves.png", dpi=300); plt.close()

    plt.figure(figsize=(7, 5.5))
    for name in ["Balanced logistic regression", "Random Forest"]:
        observed, predicted = calibration_curve(
            y_test,
            probabilities[name],
            n_bins=10,
            strategy="quantile",
        )
        plt.plot(
            predicted,
            observed,
            marker="o",
            label=(
                f"{name} "
                f"(Brier={brier_score_loss(y_test, probabilities[name]):.3f})"
            ),
        )
    plt.plot([0, 1], [0, 1], "--", color="black", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed serious/fatal proportion")
    plt.title("Probability Calibration on the 2024 Test Set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "figure_4_7_calibration.png", dpi=300)
    plt.close()

    rf_probability = probabilities["Random Forest"]
    threshold_rows = [evaluate("Random Forest", y_test, rf_probability, t) for t in [0.30, 0.40, 0.50, 0.60, 0.70]]
    pd.DataFrame(threshold_rows).to_csv(tables / "random_forest_threshold_sensitivity.csv", index=False)

    rf_prediction = (rf_probability >= 0.50).astype(int)
    cm = confusion_matrix(y_test, rf_prediction)
    plt.figure(figsize=(6.5, 5.2)); plt.imshow(cm, cmap="viridis")
    plt.title("Confusion Matrix: Random Forest")
    plt.xticks([0, 1], ["Predicted slight", "Predicted serious/fatal"], rotation=20, ha="right")
    plt.yticks([0, 1], ["Actual slight", "Actual serious/fatal"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] < cm.max()/2 else "black")
    plt.tight_layout()
    plt.savefig(figures / "supplementary_confusion_matrix.png", dpi=300)
    plt.close()

    return models, X_test, test, y_test, probabilities


def permutation_output(
    rf: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, tables: Path, figures: Path,
    sample_size: int = 3000, seed: int = RANDOM_STATE
) -> None:
    joined = X_test.copy(); joined[TARGET] = y_test.values
    sample = stratified_sample(joined, min(sample_size, len(joined)), seed)
    y_sample = sample.pop(TARGET).astype(int)
    result = permutation_importance(
        rf, sample, y_sample, n_repeats=3, random_state=seed, n_jobs=1, scoring="roc_auc"
    )
    importance = pd.DataFrame(
        {"feature": sample.columns, "importance_mean": result.importances_mean, "importance_sd": result.importances_std}
    ).sort_values("importance_mean", ascending=False)
    importance.to_csv(tables / "random_forest_permutation_importance.csv", index=False)
    top = importance.head(15).sort_values("importance_mean")
    plt.figure(figsize=(8.5, 6)); plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_sd"])
    plt.xlabel("Mean decrease in ROC-AUC after permutation")
    plt.title("Random Forest Permutation Importance")
    plt.tight_layout()
    plt.savefig(figures / "supplementary_permutation_importance.png", dpi=300)
    plt.close()


def robustness_run(
    df: pd.DataFrame, train_years: Iterable[int], test_year: int, n: int,
    seed: int, label: str, require_all_features: bool = True
) -> Dict:
    train_years = tuple(train_years)
    numeric, categorical = feature_lists(df, require_all=require_all_features)
    train_all = df[df["collision_year"].isin(train_years)].copy()
    test = df[df["collision_year"] == test_year].copy()
    if train_all.empty or test.empty:
        raise ValueError(
            f"{label} requires non-empty training years {train_years} "
            f"and test year {test_year}"
        )
    train = stratified_sample(train_all, n, seed)
    X_train = clean_feature_frame(train, numeric, categorical)
    X_test = clean_feature_frame(test, numeric, categorical)
    y_train = validated_binary_target(train)
    y_test = validated_binary_target(test)
    require_both_target_classes(y_train, f"{label} training split")
    require_both_target_classes(y_test, f"{label} test split")
    rf = make_models(numeric, categorical, seed)["Random Forest"]
    rf.fit(X_train, y_train)
    probability = rf.predict_proba(X_test)[:, 1]
    row = evaluate(label, y_test, probability)
    row.update({
        "check": label, "train_years": "–".join(map(str, train_years)),
        "test_year": test_year, "training_records": len(train), "seed": seed,
    })
    return row


def robustness_outputs(
    df: pd.DataFrame,
    tables: Path,
    require_all_features: bool = True,
) -> pd.DataFrame:
    specifications = [
        ((2020, 2021, 2022, 2023), 2024, 15000, 42, "Primary 15,000 sample"),
        ((2020, 2021, 2022, 2023), 2024, 30000, 42, "30,000 training records"),
        ((2020, 2021, 2022, 2023), 2024, 60000, 42, "60,000 training records"),
        ((2020, 2021, 2022, 2023), 2024, 15000, 123, "Alternative seed 123"),
        ((2020, 2021, 2022, 2023), 2024, 15000, 2026, "Alternative seed 2026"),
        ((2020, 2021, 2022), 2023, 15000, 42, "Rolling test: 2023"),
        ((2021, 2022, 2023), 2024, 15000, 42, "Exclude 2020"),
    ]
    rows = []
    for spec in specifications:
        print(f"Robustness check: {spec[-1]}...")
        rows.append(
            robustness_run(
                df,
                *spec,
                require_all_features=require_all_features,
            )
        )
    result = pd.DataFrame(rows)
    result.to_csv(tables / "table_4_5_random_forest_robustness_checks.csv", index=False)
    return result


def rolling_origin_outputs(
    df: pd.DataFrame,
    tables: Path,
    figures: Path,
    seed: int = RANDOM_STATE,
    require_all_features: bool = True,
) -> pd.DataFrame:
    """Evaluate both fitted models on successive future-year hold-outs."""
    numeric, categorical = feature_lists(df, require_all=require_all_features)
    rows: List[Dict] = []
    for test_year in (2021, 2022, 2023, 2024):
        train_years = tuple(range(2020, test_year))
        train = df[df["collision_year"].isin(train_years)].copy()
        test = df[df["collision_year"] == test_year].copy()
        if train.empty or test.empty:
            raise ValueError(
                f"Rolling-origin fold requires training years {train_years} "
                f"and test year {test_year}"
            )
        X_train = clean_feature_frame(train, numeric, categorical)
        X_test = clean_feature_frame(test, numeric, categorical)
        y_train = validated_binary_target(train)
        y_test = validated_binary_target(test)
        require_both_target_classes(y_train, f"Rolling fold {test_year} training")
        require_both_target_classes(y_test, f"Rolling fold {test_year} test")
        models = make_models(numeric, categorical, seed)
        for name in ("Balanced logistic regression", "Random Forest"):
            print(
                f"Rolling-origin validation: {name}, "
                f"train {train_years[0]}-{train_years[-1]}, test {test_year}..."
            )
            model = models[name]
            model.fit(X_train, y_train)
            probability = model.predict_proba(X_test)[:, 1]
            row = evaluate(name, y_test, probability)
            row.update(
                {
                    "train_years": "–".join(map(str, train_years)),
                    "test_year": test_year,
                    "training_records": len(train),
                    "test_records": len(test),
                    "seed": seed,
                }
            )
            rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(
        tables / "table_4_5_rolling_origin_validation.csv",
        index=False,
    )

    plt.figure(figsize=(8, 5.2))
    for name, subset in result.groupby("model", sort=False):
        subset = subset.sort_values("test_year")
        plt.plot(
            subset["test_year"],
            subset["roc_auc"],
            marker="o",
            label=name,
        )
    plt.xticks([2021, 2022, 2023, 2024])
    plt.ylim(0.60, 0.72)
    plt.xlabel("Held-out test year")
    plt.ylabel("ROC-AUC")
    plt.title("Rolling-Origin Temporal Validation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "figure_4_8_rolling_origin_auc.png", dpi=300)
    plt.close()
    return result


def current_git_state() -> Tuple[str | None, bool | None]:
    repository_root = Path(__file__).resolve().parent
    completed = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        return None, None
    commit = completed.stdout.strip() or None
    status = subprocess.run(
        ["git", "-C", str(repository_root), "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    dirty = None if status.returncode else bool(status.stdout.strip())
    return commit, dirty


def current_git_commit() -> str | None:
    commit, _ = current_git_state()
    return commit


def dependency_versions() -> Dict[str, str]:
    packages = [
        "pandas",
        "numpy",
        "matplotlib",
        "scikit-learn",
        "openpyxl",
        "nbformat",
    ]
    return {
        package: importlib.metadata.version(package)
        for package in packages
    }


def run_information(
    df: pd.DataFrame,
    data_path: Path,
    sample_size: int | None,
    seed: int,
    run_permutation: bool,
    run_robustness: bool,
    bootstrap_iterations: int,
    run_temporal_validation: bool,
    enforce_expected_rows: bool,
    enforce_expected_features: bool,
    enforce_expected_hash: bool,
) -> Dict:
    numeric, categorical = feature_lists(
        df, require_all=enforce_expected_features
    )
    matched = (
        pd.to_numeric(df["traffic_merge_matched"], errors="coerce")
        if "traffic_merge_matched" in df.columns
        else None
    )
    git_commit, git_worktree_dirty = current_git_state()
    dataset_sha256 = df.attrs.get("dataset_sha256") or sha256_file(data_path)
    return {
        "research_task": (
            "retrospective severity classification conditional on a "
            "reported collision"
        ),
        "git_commit": git_commit,
        "git_worktree_dirty": git_worktree_dirty,
        "dataset_sha256": dataset_sha256,
        "dataset_rows": int(len(df)),
        "dataset_columns": int(len(df.columns)),
        "train_years": "2020–2023",
        "test_year": 2024,
        "training_sample": (
            "all available 2020–2023 records"
            if sample_size is None
            else sample_size
        ),
        "training_records": (
            int(df["collision_year"].between(2020, 2023).sum())
            if sample_size is None
            else min(
                sample_size,
                int(df["collision_year"].between(2020, 2023).sum()),
            )
        ),
        "random_state": seed,
        "positive_class_test_prevalence": float(
            df.loc[df["collision_year"] == 2024, TARGET].mean()
        ),
        "model_feature_count": len(numeric) + len(categorical),
        "model_features": numeric + categorical,
        "strict_expected_row_count": enforce_expected_rows,
        "strict_dissertation_feature_schema": enforce_expected_features,
        "strict_expected_dataset_hash": enforce_expected_hash,
        "traffic_merge_matched_records": (
            int(matched.sum()) if matched is not None else None
        ),
        "traffic_merge_match_rate": (
            float(matched.mean()) if matched is not None else None
        ),
        "permutation_executed": run_permutation,
        "robustness_executed": run_robustness,
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_design": (
            "class-stratified paired resampling"
            if bootstrap_iterations
            else None
        ),
        "rolling_origin_validation_executed": run_temporal_validation,
        "python_version": platform.python_version(),
        "dependency_versions": dependency_versions(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the road-safety dissertation analysis.")
    parser.add_argument("--analysis-ready", type=Path, default=DEFAULT_ANALYSIS_READY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=15000)
    parser.add_argument(
        "--full-training",
        action="store_true",
        help="Use every available 2020–2023 record for the primary fitted models.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
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
    parser.add_argument("--run-robustness", action="store_true", help="Run the seven additional RF checks (slower).")
    parser.add_argument("--run-permutation", action="store_true", help="Run permutation importance (slower).")
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=0,
        help=(
            "Number of class-stratified paired bootstrap iterations for 95% "
            "metric and model-difference intervals."
        ),
    )
    parser.add_argument(
        "--run-temporal-validation",
        action="store_true",
        help=(
            "Run full-data rolling-origin validation with 2021–2024 held-out "
            "test years."
        ),
    )
    args = parser.parse_args()

    tables = args.output_dir / "tables"; figures = args.output_dir / "figures"
    mkdir(tables); mkdir(figures)
    enforce_expected_rows = not args.allow_row_count_difference
    enforce_expected_features = not args.allow_feature_set_difference
    enforce_expected_hash = enforce_expected_rows and enforce_expected_features
    df = load_data(
        args.analysis_ready,
        enforce_expected_rows=enforce_expected_rows,
        enforce_expected_features=enforce_expected_features,
    )
    descriptive_outputs(df, tables, figures)
    primary_sample_size = None if args.full_training else args.sample_size
    models, X_test, test, y_test, probabilities = fit_main_models(
        df,
        tables,
        figures,
        primary_sample_size,
        args.seed,
        require_all_features=enforce_expected_features,
    )
    if args.bootstrap_iterations:
        metric_uncertainty_outputs(
            y_test,
            probabilities,
            tables,
            iterations=args.bootstrap_iterations,
            seed=args.seed,
        )
    if args.run_permutation:
        permutation_output(models["Random Forest"], X_test, y_test, tables, figures, seed=args.seed)
    if args.run_robustness:
        robustness_outputs(
            df,
            tables,
            require_all_features=enforce_expected_features,
        )
    if args.run_temporal_validation:
        rolling_origin_outputs(
            df,
            tables,
            figures,
            seed=args.seed,
            require_all_features=enforce_expected_features,
        )

    run_info = run_information(
        df,
        args.analysis_ready,
        primary_sample_size,
        args.seed,
        args.run_permutation,
        args.run_robustness,
        args.bootstrap_iterations,
        args.run_temporal_validation,
        enforce_expected_rows,
        enforce_expected_features,
        enforce_expected_hash,
    )
    (tables / "run_information.json").write_text(
        json.dumps(run_info, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Completed. Outputs are in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
