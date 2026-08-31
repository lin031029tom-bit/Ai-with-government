# Published full-data dissertation results

This directory contains the complete shareable output from the strict dissertation
reproduction run refreshed on 31 August 2026 after the Windows reproducibility
bundle and expanded environment recording were added.

## Provenance

- Source commit: `e5d0ccef9c3b7a638b5d9cc4dae3c8fe89383ce3`
- Git worktree during execution: clean
- Python: 3.12.13
- Validated dataset rows: 503,475
- Validated dataset columns: 69
- Dataset SHA-256: `5e629f2d931948429580ed778b636b31aa0775630b3e4475727e39df8ee630e1`
- Training records (2020–2023): 402,548
- Final test year: 2024
- Bootstrap iterations: 1,000
- Result comparison tolerance: `1e-9`

The execution command was:

```bash
python reproduce_dissertation.py
```

The command completed full-data model training, paired class-stratified bootstrap
uncertainty estimation, rolling-origin temporal validation, permutation importance
and robustness checks. All 10 required dissertation result tables matched the
verified reference outputs.

The refreshed `tables/run_information.json` records the fuller reproducibility
metadata now captured by the workflow, including SciPy, joblib, threadpoolctl and
operating-system/platform fields. The rounded dissertation estimates are
unchanged.

## Contents

- `tables/`: 17 CSV/JSON files covering descriptive rates, model performance,
  confusion matrices, uncertainty, paired model differences, temporal validation,
  robustness, threshold sensitivity, permutation importance and run provenance.
- `figures/`: 11 PNG files covering descriptive distributions, model performance,
  ROC curves, calibration, rolling-origin performance and supplementary diagnostics.

The matching 156 MB analysis-ready input CSV is distributed separately as the
33 MB archive `published_data/analysis_ready_road_safety.csv.gz`. The reproduction
command extracts and validates it automatically. Its uncompressed SHA-256 must
match the value above.
