# Coding validation report

- The Python script was checked with `python -m py_compile`.
- The main workflow was executed against the analysis-ready dissertation dataset.
- The reported model configuration matches the revised Methodology chapter.
- The main test is time-based: 2020–2023 training period and full 2024 holdout.
- Model outputs include accuracy, precision, recall, F1, ROC-AUC, average precision and confusion counts.
- The robustness specifications include larger samples, alternative seeds, a rolling 2023 test and exclusion of 2020.
- The code explicitly describes the task as retrospective severity classification rather than future collision forecasting.

## Final execution result

The default full main workflow completed successfully on the 503,475-record analysis-ready dataset. Its output reproduced the dissertation metrics:

- balanced logistic regression: ROC-AUC 0.6613, recall 0.6419 and average precision 0.3928;
- Random Forest: ROC-AUC 0.6663, recall 0.5920 and average precision 0.3933.

Permutation importance is exposed as an optional flag because it is computationally slower; the validated result used in the dissertation is included in `example_results/`.
