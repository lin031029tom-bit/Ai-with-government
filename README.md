[README_coding.md](https://github.com/user-attachments/files/29362749/README_coding.md)
# Road Safety Dissertation Coding Package

## Project title

**Can Machine Learning Support Evidence-Based Road Safety Policy? Predicting Serious Road Traffic Collisions in Great Britain Using UK Open Government Data**

## Central research question

**To what extent can machine learning predict serious or fatal outcomes in reported road traffic collisions in Great Britain?**

## Files in this coding package

| File | Purpose |
|---|---|
| `road_safety_dissertation_coding.py` | Full standalone Python script for data preparation, descriptive analysis, modelling and robustness checks |
| `road_safety_dissertation_coding.ipynb` | Notebook version of the same workflow |
| `requirements.txt` | Python packages needed to run the code |
| `README_coding.md` | This instruction file |

## Expected input data

The script first tries to load:

```text
road_safety_analysis/analysis_ready_road_safety.csv
```

If that file is not available, it tries to build the analysis-ready dataset from raw files in:

```text
road_safety_data/
```

Expected raw files:

```text
collision_last_5_years.csv
vehicle_last_5_years.csv
local_authority_traffic.csv
road_safety_open_dataset_data_guide_2024.xlsx
```

The casualty file is not used as an ordinary modelling predictor because casualty-level outcomes may create data leakage when predicting collision severity.

## How to run

From the folder containing the script:

```bash
python road_safety_dissertation_coding.py
```

To specify folders manually:

```bash
python road_safety_dissertation_coding.py \
  --data-dir road_safety_data \
  --analysis-ready road_safety_analysis/analysis_ready_road_safety.csv \
  --output-dir road_safety_coding_outputs \
  --sample-size 15000
```

## Main outputs

The script creates:

```text
road_safety_coding_outputs/
  tables/
    dataset_summary.json
    severity_distribution.csv
    yearly_serious_fatal_rate.csv
    model_performance_2024_test.csv
    threshold_sensitivity_random_forest.csv
    subgroup_evaluation_urban_rural.csv
    random_forest_feature_importance.csv
  figures/
    figure_1_severity_distribution.png
    figure_2_yearly_rate.png
    figure_roc_curves.png
    figure_random_forest_feature_importance.png
```

## Why these modelling choices are used

1. **Time-based split**: 2020-2023 is used for training and 2024 for testing. This is closer to a real decision-support setting than a random split.
2. **Dummy baseline**: shows why accuracy alone is misleading under class imbalance.
3. **Logistic Regression (SGD)**: provides a transparent and interpretable baseline.
4. **Random Forest**: captures non-linear relationships and interactions in structured collision data.
5. **Leakage control**: variables such as `collision_severity`, casualty-level outcomes, `number_of_casualties` and police attendance are excluded.
6. **Threshold sensitivity**: shows how recall and precision change depending on the cut-off used for classifying serious/fatal outcomes.
7. **Urban/rural subgroup check**: tests whether performance differs across an important policy-relevant spatial context.

## Notes for dissertation writing

The coding supports the methodology and findings chapters. The model should be described as **exploratory decision support**, not an operational automated decision-making system.
