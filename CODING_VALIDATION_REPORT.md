# Coding validation report

## Validation metadata

- Validation date: 19 July 2026
- Dataset records: 503,475
- Dataset columns: 69
- Dataset SHA-256: `5e629f2d931948429580ed778b636b31aa0775630b3e4475727e39df8ee630e1`
- Python version: 3.12.13
- Full reproduction and verification command:
  `python reproduce_dissertation.py --analysis-ready road_safety_analysis/analysis_ready_road_safety.csv --output-dir road_safety_coding_outputs`
- Validated Git commit: `4eccd08b3a97cc1ea3c1f8c45e0c38c327a656f1`
- Git worktree dirty during validated run: **no**

## Scope

The coding workflow was checked against the dissertation's reported design and
results. The public repository reproduces the descriptive, modelling, evaluation
and robustness stages conditional on the prepared analysis-ready dataset.

## Dataset checks

- Expected analysis-ready records: **503,475**
- Complete dissertation model schema: **41/41 features present**
- Exact validated dataset SHA-256: **passed**
- Collision identifier uniqueness: **passed**
- Finite numerical model features: **passed**
- Both target classes present in every study year: **passed**
- Target agreement with official collision severity: **passed**
- Traffic-context merge: **503,369/503,475 (99.9789%)**
- Study years: **2020-2024**
- Training years: **2020-2023**
- Held-out test year: **2024**
- Test records: **100,927**
- Positive test prevalence: **0.2484**

## Automated checks

- Python syntax compilation: **passed**
- Unit, command-line and reproduction-orchestration tests: **27/27 passed**
- Main modelling command automatically repeated strict dataset validation: **passed**
- One-command reproduction stopped on analysis failure and invoked result
  verification only after successful analysis: **passed**
- Generated tables matched the seven verified core, robustness, threshold and
  interpretation outputs at `1e-9` numerical tolerance: **7/7 passed**

## Primary 2024 test results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|---:|---:|
| Dummy majority baseline | 0.7516 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2484 |
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

The metrics and generated artifacts in this report were reproduced from the
clean validated code commit identified above. The subsequent artifact commit
records this report and synchronizes `example_results/run_information.json`; it
does not change the validated modelling or verification code. Later modelling
code changes require a fresh full-data run before they can be described as
full-data validated. The two orchestration tests added after the full-data run
exercise only the one-command control flow and do not change the modelling code
or verified numerical results. Automated tests check syntax, validation,
preprocessing, small synthetic model runs, orchestration and result-comparison
failures, but they do not replace validation against the 503,475-record
analysis-ready dataset.
