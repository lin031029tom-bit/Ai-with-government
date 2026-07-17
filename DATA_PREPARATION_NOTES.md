# Data preparation scope

The public repository does not redistribute the raw Department for Transport files or the large analysis-ready dataset.

The modelling script expects:

```text
road_safety_analysis/analysis_ready_road_safety.csv
```

The dissertation documents the preparation logic used to create the file:

1. combine the 2020-2024 Department for Transport collision files;
2. construct the binary `serious_or_fatal` target from official collision severity;
3. parse date and time fields and create month, hour, weekend and night indicators;
4. aggregate vehicle records to collision level;
5. merge local-authority traffic context by local authority and collision year;
6. convert selected official unknown codes to missing values;
7. exclude target-defining and post-outcome leakage fields;
8. retain the analysis fields required by the modelling script.

This repository therefore supports **model-level reproducibility conditional on the prepared dataset**. The empty `prepare_analysis_ready_data.py` file from the earlier repository version should be deleted unless a complete, tested raw-data preparation implementation is added.
