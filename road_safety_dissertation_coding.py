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
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

TARGET = "serious_or_fatal"
RANDOM_STATE = 42
DEFAULT_ANALYSIS_READY = Path("road_safety_analysis/analysis_ready_road_safety.csv")
DEFAULT_OUTPUT_DIR = Path("road_safety_coding_outputs")

NUMERIC_CANDIDATES = [
    "month", "hour", "is_weekend", "is_night", "longitude", "latitude",
    "number_of_vehicles", "speed_limit", "traffic_link_length_km",
    "traffic_all_motor_vehicles", "traffic_all_motor_vehicles_per_km",
    "traffic_cars_taxis_share", "vehicle_record_count", "n_pedal_cycles",
    "n_motorcycles", "n_cars_taxis", "n_buses_minibuses", "n_goods_vehicles",
    "vehicle_type_nunique", "mean_driver_age", "min_driver_age", "max_driver_age",
    "any_young_driver_17_24", "any_older_driver_65_plus", "mean_vehicle_age",
    "max_vehicle_age",
]
CATEGORICAL_CANDIDATES = [
    "day_of_week", "police_force", "local_authority_highway",
    "urban_or_rural_area", "first_road_class", "road_type", "junction_detail",
    "junction_control", "pedestrian_crossing", "light_conditions",
    "weather_conditions", "road_surface_conditions", "special_conditions_at_site",
    "carriageway_hazards", "trunk_road_flag",
]
UNKNOWN_NUMERIC_COLUMNS = [
    "speed_limit", "mean_driver_age", "min_driver_age", "max_driver_age",
    "mean_vehicle_age", "max_vehicle_age",
]


def mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
       raise FileNotFoundError(
    f"Analysis-ready data not found at {path}. "
    "Prepare the dataset using the process documented in the dissertation "
    "and DATA_PREPARATION_NOTES.md, or place "
    "analysis_ready_road_safety.csv at the required path."
)
    df = pd.read_csv(path, low_memory=False)
    required = {"collision_index", "collision_year", "collision_severity", TARGET}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"The analysis-ready file is missing required columns: {sorted(missing)}")
    return df


def feature_lists(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric = [c for c in NUMERIC_CANDIDATES if c in df.columns]
    categorical = [c for c in CATEGORICAL_CANDIDATES if c in df.columns]
    if not numeric and not categorical:
        raise ValueError("No modelling features were found in the analysis-ready data.")
    return numeric, categorical


def clean_feature_frame(
    frame: pd.DataFrame, numeric: Sequence[str], categorical: Sequence[str]
) -> pd.DataFrame:
    out = frame[list(numeric) + list(categorical)].copy()
    for col in UNKNOWN_NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = out[col].replace([-1, 99], np.nan)
    for col in categorical:
        out[col] = out[col].astype("object").where(out[col].notna(), "missing").astype(str)
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
    sample, _ = train_test_split(
        frame, train_size=n, stratify=frame[TARGET], random_state=seed
    )
    return sample.copy()


def make_models(numeric: Sequence[str], categorical: Sequence[str], seed: int) -> Dict[str, Pipeline]:
    return {
        "Dummy majority baseline": Pipeline(
            [
                ("preprocessor", preprocessor(numeric, categorical, False)),
                ("model", DummyClassifier(strategy="most_frequent")),
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


def evaluate(name: str, y_true: pd.Series, probabilities: np.ndarray, threshold: float = 0.5) -> Dict:
    prediction = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    variable_probability = len(np.unique(probabilities)) > 1
    return {
        "model": name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, prediction),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall": recall_score(y_true, prediction, zero_division=0),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities) if variable_probability else np.nan,
        "average_precision": average_precision_score(y_true, probabilities),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def rate_table(df: pd.DataFrame, group: str) -> pd.DataFrame:
    table = df.groupby(group, dropna=False).agg(
        collisions=("collision_index", "count"), serious_or_fatal=(TARGET, "sum")
    ).reset_index()
    table["slight"] = table["collisions"] - table["serious_or_fatal"]
    table["serious_fatal_rate_pct"] = (
        table["serious_or_fatal"] / table["collisions"] * 100
    ).round(2)
    return table


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

    speed = rate_table(df, "speed_limit").sort_values("speed_limit")
    speed.to_csv(tables / "speed_limit_serious_fatal_rate.csv", index=False)
    urban = rate_table(df, "urban_or_rural_area")
    urban["area"] = urban["urban_or_rural_area"].map({1: "Urban", 2: "Rural", 3: "Unallocated"})
    urban.to_csv(tables / "urban_rural_serious_fatal_rate.csv", index=False)

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

    speed_plot = speed[speed["speed_limit"].isin([20, 30, 40, 50, 60, 70])]
    plt.figure(figsize=(8, 5))
    plt.bar(speed_plot["speed_limit"].astype(str), speed_plot["serious_fatal_rate_pct"])
    plt.xlabel("Speed limit"); plt.ylabel("Serious/fatal rate (%)")
    plt.title("Serious/Fatal Rate by Speed Limit")
    plt.tight_layout(); plt.savefig(figures / "figure_4_3_speed_limit_rate.png", dpi=300); plt.close()

    urban_plot = urban[urban["area"].isin(["Urban", "Rural"])]
    plt.figure(figsize=(8, 5))
    plt.barh(urban_plot["area"], urban_plot["serious_fatal_rate_pct"])
    plt.xlabel("Serious/fatal rate (%)"); plt.ylabel("Area type")
    plt.title("Serious/Fatal Rate by Urban/Rural Area")
    plt.tight_layout(); plt.savefig(figures / "figure_4_4_urban_rural_rate.png", dpi=300); plt.close()


def fit_main_models(
    df: pd.DataFrame, tables: Path, figures: Path, sample_size: int, seed: int
) -> Tuple[Dict[str, Pipeline], pd.DataFrame, pd.DataFrame, pd.Series, Dict[str, np.ndarray]]:
    numeric, categorical = feature_lists(df)
    train_all = df[df["collision_year"].between(2020, 2023)].copy()
    test = df[df["collision_year"] == 2024].copy()
    train = stratified_sample(train_all, sample_size, seed)

    X_train = clean_feature_frame(train, numeric, categorical)
    X_test = clean_feature_frame(test, numeric, categorical)
    y_train = train[TARGET].astype(int)
    y_test = test[TARGET].astype(int)

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

    fitted = performance[performance["model"] != "Dummy majority baseline"]
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
    plt.tight_layout(); plt.savefig(figures / "figure_4_8_confusion_matrix.png", dpi=300); plt.close()

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
    plt.tight_layout(); plt.savefig(figures / "figure_4_7_permutation_importance.png", dpi=300); plt.close()


def robustness_run(
    df: pd.DataFrame, train_years: Iterable[int], test_year: int, n: int,
    seed: int, label: str
) -> Dict:
    numeric, categorical = feature_lists(df)
    train_all = df[df["collision_year"].isin(list(train_years))].copy()
    test = df[df["collision_year"] == test_year].copy()
    train = stratified_sample(train_all, n, seed)
    X_train = clean_feature_frame(train, numeric, categorical)
    X_test = clean_feature_frame(test, numeric, categorical)
    y_train = train[TARGET].astype(int); y_test = test[TARGET].astype(int)
    rf = make_models(numeric, categorical, seed)["Random Forest"]
    rf.fit(X_train, y_train)
    probability = rf.predict_proba(X_test)[:, 1]
    row = evaluate(label, y_test, probability)
    row.update({
        "check": label, "train_years": "–".join(map(str, train_years)),
        "test_year": test_year, "training_records": len(train), "seed": seed,
    })
    return row


def robustness_outputs(df: pd.DataFrame, tables: Path) -> pd.DataFrame:
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
        rows.append(robustness_run(df, *spec))
    result = pd.DataFrame(rows)
    result.to_csv(tables / "table_4_5_random_forest_robustness_checks.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the road-safety dissertation analysis.")
    parser.add_argument("--analysis-ready", type=Path, default=DEFAULT_ANALYSIS_READY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--run-robustness", action="store_true", help="Run the seven additional RF checks (slower).")
    parser.add_argument("--run-permutation", action="store_true", help="Run permutation importance (slower).")
    args = parser.parse_args()

    tables = args.output_dir / "tables"; figures = args.output_dir / "figures"
    mkdir(tables); mkdir(figures)
    df = load_data(args.analysis_ready)
    descriptive_outputs(df, tables, figures)
    models, X_test, test, y_test, probabilities = fit_main_models(
        df, tables, figures, args.sample_size, args.seed
    )
    if args.run_permutation:
        permutation_output(models["Random Forest"], X_test, y_test, tables, figures, seed=args.seed)
    if args.run_robustness:
        robustness_outputs(df, tables)

    run_info = {
        "research_task": "retrospective severity classification conditional on a reported collision",
        "train_years": "2020–2023", "test_year": 2024,
        "training_sample": args.sample_size, "random_state": args.seed,
        "positive_class_test_prevalence": float((test[TARGET] == 1).mean()),
        "robustness_executed": bool(args.run_robustness),
    }
    (tables / "run_information.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    print(f"Completed. Outputs are in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
