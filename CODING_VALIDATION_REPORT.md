# Coding validation report

## Validation metadata

- Validation date: 17 July 2026
- Dataset records: 503,475
- Python version: 3.12
- Validation command:
  `python validate_analysis_ready_data.py --analysis-ready road_safety_analysis/analysis_ready_road_safety.csv`
- Modelling command:
  `python road_safety_dissertation_coding.py --analysis-ready road_safety_analysis/analysis_ready_road_safety.csv --output-dir road_safety_coding_outputs --run-permutation --run-robustness`
- Validated Git commit: Pending final rerun

## Scope

The coding workflow was checked against the dissertation's reported design and results. The public repository reproduces the modelling, evaluation and robustness stages conditional on the prepared analysis-ready dataset.

## Dataset checks

- Expected analysis-ready records: **503,475**
- Study years: **2020-2024**
- Training years: **2020-2023**
- Held-out test year: **2024**
- Test records: **100,927**
- Positive test prevalence: **0.2484**

## Primary 2024 test results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|---:|---:|
| Dummy majority baseline | 0.7516 | 0.0000 | 0.0000 | 0.0000 | N/A | 0.2484 |
| Balanced logistic regression | 0.5971 | 0.3368 | 0.6419 | 0.4418 | 0.6613 | 0.3928 |
| Random Forest | 0.6303 | 0.3540 | 0.5920 | 0.4430 | 0.6663 | 0.3933 |

## Robustness checks

Random Forest ROC-AUC remained between **0.6704 and 0.6838** across alternative seeds and temporal designs. Increasing the training sample to 30,000 and 60,000 records produced ROC-AUC values of **0.6763** and **0.6796** respectively.

## Interpretation check

- Random Forest has slightly higher precision, F1-score, ROC-AUC and average precision than balanced logistic regression.
- Balanced logistic regression has higher recall.
- Random Forest produces fewer false positives but more false negatives than balanced logistic regression at the 0.50 threshold.
- The task is retrospective severity classification, not future collision occurrence forecasting.

## Remaining reproducibility limitation

The raw-to-analysis-ready data preparation pipeline is not included as a complete tested script. The README and dissertation therefore describe the repository accurately as reproducing modelling, evaluation and robustness from the prepared dataset.
