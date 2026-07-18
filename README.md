# Road Safety Dissertation: Reproducible Modelling and Evaluation

## Research question

**To what extent can machine learning predict serious or fatal outcomes in reported road traffic collisions in Great Britain?**

This repository implements a **retrospective collision-severity classification** task. It predicts severity conditional on a personal-injury collision having occurred and been reported. It does **not** forecast the occurrence, time or location of future collisions.

## Reproducibility scope

The public repository reproduces the **modelling, evaluation and robustness stages** once the prepared analysis-ready dataset is supplied. It does not independently reconstruct the complete analytical dataset from the raw Department for Transport files.

The raw files and the large analysis-ready dataset are not redistributed. Place the prepared file at:

```text
road_safety_analysis/analysis_ready_road_safety.csv
```

The dissertation documents the data sources, unit of analysis, feature groups, leakage exclusions, merge design and temporal validation strategy.

## Repository files

| File or folder | Purpose |
|---|---|
| `road_safety_dissertation_coding.py` | Main modelling, evaluation and robustness script |
| `road_safety_dissertation_coding.ipynb` | Retained Colab execution log, including setup and data-upload troubleshooting |
| `road_safety_dissertation_coding_clean.ipynb` | Clean Colab wrapper for repeat runs and output review |
| `validate_analysis_ready_data.py` | Automated checks for the prepared dataset |
| `requirements.txt` | Python dependencies |
| `CODING_VALIDATION_REPORT.md` | Summary of execution and consistency checks |
| `DATA_PREPARATION_NOTES.md` | Scope and expected structure of the prepared dataset |
| `example_results/tables/` | Model performance, robustness and importance outputs |
| `example_results/figures/` | Selected figures used in the dissertation |

## Installation

The pinned dependency set was validated with Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Automated checks

The repository includes lightweight tests for the dataset validator and numerical
preprocessing. They do not require the restricted analysis-ready dataset:

```bash
python -m py_compile \
  road_safety_dissertation_coding.py \
  validate_analysis_ready_data.py
python -m unittest discover -s tests -v
```

GitHub Actions runs the same checks on pushes and pull requests.

## Validate the prepared dataset

```bash
python validate_analysis_ready_data.py \
  --analysis-ready road_safety_analysis/analysis_ready_road_safety.csv
```

The validator checks the file path, required fields, binary target, study years, expected row count and collision identifier uniqueness where available.

## Validation status

`CODING_VALIDATION_REPORT.md` identifies the exact commit used for the documented
full-data run. Later commits should not be described as end-to-end validated until
the analysis is rerun with the prepared dataset and the report is updated.

The retained executed notebook contains attempts made before the dataset was
available and therefore includes `FileNotFoundError` output. Use the clean notebook
for a fresh end-to-end Colab run; do not treat the retained notebook alone as proof
of a successful current-commit run.

## Main analysis

```bash
python road_safety_dissertation_coding.py \
  --analysis-ready road_safety_analysis/analysis_ready_road_safety.csv \
  --output-dir road_safety_coding_outputs
```

The primary run uses a stratified 15,000-record sample from 2020-2023 and the full 2024 test set. It compares:

- dummy majority baseline;
- balanced logistic regression (`liblinear`, `class_weight='balanced'`);
- Random Forest (100 trees, maximum depth 16, minimum leaf size 50 and balanced subsampling).

The descriptive outputs also include local-authority serious/fatal rates for
authorities with at least 500 collision records, matching the reporting rule used
in the dissertation.

## Full robustness and interpretation run

```bash
python road_safety_dissertation_coding.py \
  --analysis-ready road_safety_analysis/analysis_ready_road_safety.csv \
  --output-dir road_safety_coding_outputs \
  --run-permutation \
  --run-robustness
```

The robustness analysis includes:

- 30,000 and 60,000 training records;
- alternative random seeds 123 and 2026;
- training on 2020-2022 and testing on 2023;
- excluding 2020 and training on 2021-2023 for the 2024 test.

## Interpretation

The code supports an exploratory MSc benchmark, not an operational public-sector system. The current model is a retrospective analytical tool. Any future operational use would require a clearly specified use case, full-sample retraining, probability calibration, prospective and local validation, subgroup/equity evaluation, monitoring and human oversight.
