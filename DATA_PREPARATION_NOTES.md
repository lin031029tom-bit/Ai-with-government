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

The exact strict input schema is defined once in `analysis_schema.py` and is used
by both the validator and the modelling entry point. The validator also confirms
that the binary target agrees with official collision severity and reports the
traffic-context merge rate.

This repository therefore supports **model-level reproducibility conditional on
the prepared dataset**. A complete, tested raw-file-to-analysis-ready preparation
implementation remains outside the public repository and should not be implied
until that pipeline is added and validated.
