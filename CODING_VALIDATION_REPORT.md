# Coding validation report

## Validation metadata

- Validation date: 19 July 2026
- Dataset records: 503,475
- Python version: 3.12.13
- Validation command:
  `python validate_analysis_ready_data.py --analysis-ready road_safety_analysis/analysis_ready_road_safety.csv`
- Modelling command:
  `python road_safety_dissertation_coding.py --analysis-ready road_safety_analysis/analysis_ready_road_safety.csv --output-dir road_safety_coding_outputs --run-permutation --run-robustness`
- Validated Git commit: `4fda2dd07c02311fff3ece62774ec2cf31b17aa5`

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
| Balanced logistic regression | 0.5976 | 0.3370 | 0.6409 | 0.4417 | 0.6612 | 0.3927 |
| Random Forest | 0.6296 | 0.3531 | 0.5901 | 0.4418 | 0.6662 | 0.3930 |

## Robustness checks

Random Forest ROC-AUC remained between **0.6702 and 0.6837** across alternative seeds and temporal designs. Increasing the training sample to 30,000 and 60,000 records produced ROC-AUC values of **0.6763** and **0.6802** respectively.

## Interpretation check

- Random Forest has slightly higher precision, F1-score, ROC-AUC and average precision than balanced logistic regression.
- Balanced logistic regression has higher recall.
- Random Forest produces fewer false positives but more false negatives than balanced logistic regression at the 0.50 threshold.
- The task is retrospective severity classification, not future collision occurrence forecasting.

## Remaining reproducibility limitation

The raw-to-analysis-ready data preparation pipeline is not included as a complete tested script. The README and dissertation therefore describe the repository accurately as reproducing modelling, evaluation and robustness from the prepared dataset.

## Validation boundary

The metrics in this report were generated from the validated code commit identified
above. Subsequent commits record this report, synchronize `example_results`, and
update the CI action runtimes; they do not change the validated modelling code or
tests. Later modelling-code changes require a fresh full-data run before they can
be described as end-to-end validated. Automated tests can check syntax,
preprocessing and small synthetic model runs, but they do not replace validation
against the 503,475-record analysis-ready dataset.
