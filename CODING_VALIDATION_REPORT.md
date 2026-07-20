# Coding validation report

## Validation metadata

- Validation date: 20 July 2026
- Dataset records: 503,475
- Dataset columns: 69
- Dataset SHA-256: `5e629f2d931948429580ed778b636b31aa0775630b3e4475727e39df8ee630e1`
- Python version: 3.12.13
- Full reproduction and verification command:
  `python reproduce_dissertation.py --analysis-ready road_safety_analysis/analysis_ready_road_safety.csv --output-dir road_safety_coding_outputs`
- Validated Git commit: `edaa21442d1d21d8457914a72303a089e2899da1`
- Git worktree dirty during validated run: **no**

## Scope

The coding workflow was checked against the dissertation's reported design and
results. The public repository reproduces the descriptive, full-data modelling,
uncertainty, temporal-validation, calibration and robustness stages conditional
on the prepared analysis-ready dataset.

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
- Training records used by the primary models: **402,548**
- Held-out test year: **2024**
- Test records: **100,927**
- Positive test prevalence: **0.2484**

## Automated checks

- Python syntax compilation: **passed**
- Unit, command-line and reproduction-orchestration tests: **29/29 passed**
- Main modelling command automatically repeated strict dataset validation: **passed**
- Primary models fitted on every available 2020-2023 record: **passed**
- Paired class-stratified bootstrap resamples: **1,000**
- Rolling-origin hold-outs for 2021, 2022, 2023 and 2024: **passed**
- Brier score, log loss and probability-calibration figure: **generated**
- One-command reproduction stopped on analysis failure and invoked result
  verification only after successful analysis: **passed**
- Generated tables matched all core, uncertainty, temporal, robustness,
  threshold and interpretation outputs at `1e-9` numerical tolerance:
  **10/10 passed**

## Primary 2024 test results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Average precision | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dummy prevalence baseline | 0.7516 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2484 | 0.1870 |
| Balanced logistic regression | 0.6977 | 0.3885 | 0.3782 | 0.3833 | 0.6519 | 0.3829 | 0.2012 |
| Random Forest | 0.6273 | 0.3588 | 0.6355 | 0.4586 | 0.6877 | 0.4227 | 0.2211 |

## Prediction-error evidence

- Random Forest 2024 ROC-AUC was **0.6877** with a paired-bootstrap 95%
  confidence interval of **0.6841-0.6915**.
- Its average precision was **0.4227** (95% CI **0.4175-0.4282**), recall was
  **0.6355** (95% CI **0.6297-0.6417**) and F1 was **0.4586** (95% CI
  **0.4551-0.4624**).
- The Random-Forest minus logistic-regression ROC-AUC difference was **0.0359**
  (95% CI **0.0328-0.0390**).
- Rolling-origin Random Forest ROC-AUC was **0.6937**, **0.6958**, **0.7012**
  and **0.6877** for held-out years 2021-2024.
- Increasing training size from 15,000 to 30,000, 60,000 and all 402,548 prior
  records produced ROC-AUC values of **0.6662**, **0.6763**, **0.6802** and
  **0.6877**.
- Random Forest's Brier score (**0.2211**) was worse than logistic regression's
  (**0.2012**), and the calibration figure shows that neither class-weighted
  model should be interpreted as a calibrated operational risk probability.

## Interpretation check

- Random Forest has higher recall, F1, ROC-AUC and average precision than balanced logistic regression.
- Balanced logistic regression has higher accuracy and precision and a lower Brier score.
- Random Forest produces more false positives but fewer false negatives than balanced logistic regression at the 0.50 threshold.
- Threshold sensitivity is reported as an error trade-off, not as a recommended deployment threshold.
- The task is retrospective severity classification, not future collision occurrence forecasting.

## Remaining reproducibility limitation

The raw-to-analysis-ready data preparation pipeline is not included as a complete tested script. The README and dissertation therefore describe the repository accurately as reproducing modelling, evaluation and robustness from the prepared dataset.

## Validation boundary

The metrics and generated artifacts in this report were reproduced from the
clean validated code commit identified above. The subsequent artifact commit
records this report and synchronizes `example_results/`; it does not change the
validated modelling or verification code. Later modelling-code changes require
a fresh full-data run before they can be described as validated. Automated tests
check syntax, validation, preprocessing, small synthetic model runs,
orchestration and result-comparison failures, but they do not replace validation
against the 503,475-record analysis-ready dataset.
