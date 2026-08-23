# Road Safety Dissertation: Reproducible Modelling and Evaluation

## Research question

**To what extent can machine learning predict serious or fatal outcomes in reported road traffic collisions in Great Britain?**

This repository implements a **retrospective collision-severity classification** task. It predicts severity conditional on a personal-injury collision having occurred and been reported. It does **not** forecast the occurrence, time or location of future collisions.

## Reproducibility scope

The public repository reproduces the **modelling, evaluation and robustness stages**
from a published, prepared analysis-ready dataset. It does not independently
reconstruct that analytical dataset from the raw Department for Transport files.

The compressed prepared dataset is versioned at
`published_data/analysis_ready_road_safety.csv.gz`. The zero-argument reproduction
command extracts it automatically to the ignored working path:

```text
road_safety_analysis/analysis_ready_road_safety.csv
```

The dissertation documents the data sources, unit of analysis, feature groups, leakage exclusions, merge design and temporal validation strategy.

## Repository files

| File or folder | Purpose |
|---|---|
| `analysis_schema.py` | Shared strict schema for the validated dissertation dataset |
| `road_safety_dissertation_coding.py` | Main modelling, evaluation and robustness script |
| `reproduce_dissertation.py` | One-command strict full-data run followed by result verification |
| `verify_dissertation_results.py` | Compares generated dissertation tables with the verified reference outputs |
| `road_safety_dissertation_coding_clean.ipynb` | Clean Colab wrapper pinned to the validated data-publication snapshot; verifies the archive and runs the one-command reproduction |
| `archive/` | Historical pre-publication Colab log retained for provenance and clearly separated from the runnable workflow |
| `validate_analysis_ready_data.py` | Automated checks for the prepared dataset |
| `.python-version` | Supported Python interpreter version used for validation |
| `requirements.txt` | Python dependencies |
| `CODING_VALIDATION_REPORT.md` | Summary of execution and consistency checks |
| `DATA_PREPARATION_NOTES.md` | Scope and expected structure of the prepared dataset |
| `published_data/` | Compressed analysis-ready dataset, checksums, provenance and licence information |
| `example_results/tables/` | Verified descriptive, model, robustness, interpretation and provenance outputs |
| `example_results/figures/` | Verified descriptive, discrimination, calibration and rolling-validation figures |
| `published_results/` | Full-data tables, figures and provenance regenerated from the merged `main` workflow |

## Published full-data results

The browsable result package is available in [`published_results/`](published_results/README.md).
It contains the 17 machine-readable tables and 11 figures produced by the strict
full-data run on 23 August 2026. The accompanying `run_information.json` records
the source commit, dataset SHA-256, row count, feature schema, training size,
bootstrap design and dependency versions. The matching compressed analysis-ready
input is available in `published_data/`.

## Installation

Use Python 3.12; `.python-version` records the supported interpreter.

```bash
python3.12 -m venv .venv         # Windows: py -3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Automated checks

The repository includes lightweight tests for the dataset validator and numerical
preprocessing. They do not require the restricted analysis-ready dataset:

```bash
python -m py_compile \
  analysis_schema.py \
  reproduce_dissertation.py \
  road_safety_dissertation_coding.py \
  validate_analysis_ready_data.py \
  verify_dissertation_results.py
python -m unittest discover -s tests -v
```

GitHub Actions runs the same checks on pull requests and on pushes to `main`,
avoiding duplicate branch and pull-request runs.

## Validate the prepared dataset

```bash
python validate_analysis_ready_data.py \
  --analysis-ready road_safety_analysis/analysis_ready_road_safety.csv
```

The default validator is the strict dissertation-reproduction check. It requires
the complete 41-feature model schema, collision identifiers and severity fields;
checks the exact validated dataset SHA-256, row count, study years, identifier
uniqueness, finite numerical values and the binary target; verifies that every
study year contains both target classes and that the target agrees with official
collision severity; and reports traffic-context merge coverage.

For a clearly documented alternative dataset, the row-count and complete-feature
checks can be relaxed independently. Either relaxation also disables the exact
dissertation-dataset hash check while retaining the target, year, identifier and
available-feature validity checks:

```bash
python validate_analysis_ready_data.py \
  --analysis-ready path/to/alternative.csv \
  --allow-row-count-difference \
  --allow-feature-set-difference
```

## Validation status

`CODING_VALIDATION_REPORT.md` identifies the exact clean commit used for the
documented full-data run. Later modelling-code changes should not be described as
end-to-end validated until the analysis is rerun with the published dataset and the
report is updated.

Use `road_safety_dissertation_coding_clean.ipynb` for a fresh end-to-end Colab run.
It pins the validated data-publication commit, checks the published gzip archive and
runs `reproduce_dissertation.py`; it does not require a manual data upload. The
earlier failed execution log has been moved to `archive/` and is retained only for
provenance.

## Main analysis

```bash
python road_safety_dissertation_coding.py \
  --analysis-ready road_safety_analysis/analysis_ready_road_safety.csv \
  --output-dir road_safety_coding_outputs
```

The modelling command automatically runs the same strict validation before
training. Invalid or fractional targets are rejected rather than converted to
integers. The two `--allow-...-difference` flags shown above are also available
on the modelling command for explicitly documented alternative data.

The default quick run uses a stratified 15,000-record sample from 2020-2023 and
the full 2024 test set. It compares:

- dummy prevalence baseline (ROC-AUC = 0.5000 and calibrated to the training prevalence);
- balanced logistic regression (`liblinear`, `class_weight='balanced'`);
- Random Forest (100 trees, maximum depth 16, minimum leaf size 50 and balanced subsampling).

For the dissertation results, use `--full-training`. This fits both substantive
models on all available 402,548 records from 2020-2023 while preserving 2024 as
the untouched final test year:

```bash
python road_safety_dissertation_coding.py \
  --analysis-ready road_safety_analysis/analysis_ready_road_safety.csv \
  --output-dir road_safety_coding_outputs \
  --full-training
```

The descriptive outputs also include local-authority serious/fatal rates for
authorities with at least 500 collision records, matching the reporting rule used
in the dissertation. The table contains both official authority codes and readable
authority names. Separate machine-readable rate tables reproduce the dissertation's
road-type, lighting-condition and weather-condition percentages.

## Scientific validation, uncertainty and interpretation run

```bash
python road_safety_dissertation_coding.py \
  --analysis-ready road_safety_analysis/analysis_ready_road_safety.csv \
  --output-dir road_safety_coding_outputs \
  --full-training \
  --bootstrap-iterations 1000 \
  --run-temporal-validation \
  --run-permutation \
  --run-robustness
```

The complete validation design includes:

- a final 2024 hold-out after full-data training on 2020-2023;
- rolling-origin folds that test 2021, 2022, 2023 and 2024 using only prior years;
- 1,000 class-stratified paired bootstrap resamples for 95% metric intervals;
- paired confidence intervals for Random-Forest minus logistic-regression differences;
- Brier score, log loss and a probability-calibration plot;
- 30,000 and 60,000 training records;
- alternative random seeds 123 and 2026;
- training on 2020-2022 and testing on 2023;
- excluding 2020 and training on 2021-2023 for the 2024 test.

## One-command dissertation reproduction

Run the complete modelling, interpretation and robustness workflow and compare
every key generated table with the verified dissertation outputs:

```bash
python reproduce_dissertation.py
```

The zero-argument command extracts `published_data/analysis_ready_road_safety.csv.gz`
when the working CSV is absent, uses the extracted file, writes to
`road_safety_coding_outputs`, checks that Python 3.12 and all pinned dependencies
are available, and refuses to start if the dataset archive or verified reference
tables are missing. Paths supplied explicitly are resolved relative to the
directory from which the command is invoked:

```bash
python reproduce_dissertation.py \
  --analysis-ready /path/to/analysis_ready_road_safety.csv \
  --output-dir /path/to/dissertation_results
```

The command fails if strict dataset validation fails, model execution fails, a
required result is missing, or a generated core, uncertainty, temporal,
robustness, threshold or permutation-importance table differs from the verified
reference output beyond a numerical tolerance of `1e-9`.

`run_information.json` records the exact Git commit, whether the worktree was
dirty, dataset SHA-256, row and column counts, selected features, actual training
records, bootstrap design, Python and dependency versions, traffic merge
coverage and which optional analyses were executed.

## Interpretation

The code supports an exploratory MSc benchmark, not an operational public-sector
system. The current model is a retrospective analytical tool. Full-data training,
temporal hold-outs and bootstrap intervals strengthen the estimate of prediction
error, but the calibration results do not support treating the scores as reliable
operational probabilities. Any future use would require a clearly specified use
case, probability calibration, prospective and local validation, subgroup/equity
evaluation, monitoring and human oversight.
