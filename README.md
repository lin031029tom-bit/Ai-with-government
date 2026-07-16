[README.md](https://github.com/user-attachments/files/30095954/README.md)
# Road Safety Dissertation: Reproducible Coding

## Research question

**To what extent can machine learning predict serious or fatal outcomes in reported road traffic collisions in Great Britain?**

This repository implements a **retrospective collision-severity classification** task. It predicts severity conditional on a collision having occurred and been reported. It does **not** forecast the occurrence, time or location of future collisions.

## Files

| File or folder | Purpose |
|---|---|
| `road_safety_dissertation_coding.py` | Main reproducible modelling, evaluation and robustness script. It requires the prepared analysis-ready dataset described below. |
| `road_safety_dissertation_coding.ipynb` | Notebook interface for running and reviewing the analysis |
| `requirements.txt` | Python package dependencies |
| `CODING_VALIDATION_REPORT.md` | Validation and execution checks |
| `example_results/tables/` | Model performance, robustness and importance outputs |
| `example_results/figures/` | Selected figures reproduced in the dissertation |

## Data

The raw Department for Transport files are not redistributed because of their size. Download the official Road Safety Open Data and local-authority traffic data, then prepare or place:

```text
road_safety_analysis/analysis_ready_road_safety.csv
```

The script checks for the required fields and fails with a clear error if the file is missing or incompatible.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Main analysis

```bash
python road_safety_dissertation_coding.py \
  --analysis-ready road_safety_analysis/analysis_ready_road_safety.csv \
  --output-dir road_safety_coding_outputs
```

The main run uses a stratified 15,000-record sample from 2020–2023 and the full 2024 test set. It compares:

- dummy majority baseline;
- balanced logistic regression (`liblinear`, `class_weight='balanced'`);
- Random Forest (100 trees, maximum depth 16, minimum leaf size 50 and balanced subsampling).

Numerical missing values are median-imputed; selected official unknown codes are converted to missing; categorical fields are mode-imputed and one-hot encoded; numerical fields are standardised for logistic regression. Leakage-sensitive outcome and post-outcome fields are excluded.

## Robustness analysis

```bash
python road_safety_dissertation_coding.py \
  --analysis-ready road_safety_analysis/analysis_ready_road_safety.csv \
  --output-dir road_safety_coding_outputs \
  --run-robustness
```

This additionally tests:

- 30,000 and 60,000 training records;
- random seeds 123 and 2026;
- training on 2020–2022 and testing on 2023;
- excluding 2020 and training on 2021–2023 for the 2024 test.

## Outputs

The default run produces the core dissertation tables and figures, including model performance, average precision, threshold sensitivity, ROC curves and the Random Forest confusion matrix. Permutation importance is intentionally optional because it is slower. Run it with:

```bash
python road_safety_dissertation_coding.py --analysis-ready road_safety_analysis/analysis_ready_road_safety.csv --output-dir road_safety_coding_outputs --run-permutation
```

The `example_results` folder contains the numerical results used in the revised dissertation.

## Interpretation

The code supports an exploratory MSc benchmark, not an operational public-sector system. Any deployment would require full-sample retraining, calibration, local validation, subgroup/equity evaluation, monitoring and human oversight.
