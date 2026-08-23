# Published analysis-ready road-safety dataset

`analysis_ready_road_safety.csv.gz` is the gzip-compressed, collision-level input
used for the dissertation's verified modelling run. Decompression is lossless.

## Integrity

- Compressed size: approximately 33 MB
- Uncompressed size: approximately 156 MB
- Rows: 503,475
- Columns: 69
- Years: 2020–2024
- Compressed SHA-256: `8efbdd94ad028113facbec818dc3e95c9a98621e6739450a56a21d7b995b00c7`
- Uncompressed SHA-256: `5e629f2d931948429580ed778b636b31aa0775630b3e4475727e39df8ee630e1`

`python reproduce_dissertation.py` extracts the archive automatically. Manual
extraction is also possible:

```bash
mkdir -p road_safety_analysis
gzip -dc published_data/analysis_ready_road_safety.csv.gz \
  > road_safety_analysis/analysis_ready_road_safety.csv
```

The strict validator checks the uncompressed hash, row count, study years,
collision-identifier uniqueness, target consistency, finite numeric values and
the complete 41-feature modelling schema before any model is trained.

## Source and licence

This is a derived research dataset prepared from publicly released, non-sensitive
Department for Transport road-safety open data for Great Britain and public
traffic-context data. It contains no names or direct personal identifiers. It does
retain public collision identifiers, dates and location fields needed for validation
and modelling, so it should be used only for legitimate analytical and research
purposes and should not be used to attempt to identify individuals.

Source: [Department for Transport, Road safety open data](https://www.gov.uk/government/statistical-data-sets/road-safety-open-data).

Contains public sector information licensed under the
[Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
The derived feature engineering and documentation are provided by Yi Lin for
transparent academic reproduction. No official endorsement is implied.

The repository documents the transformation scope in `DATA_PREPARATION_NOTES.md`.
The complete raw-file-to-analysis-ready preparation implementation is not included;
therefore the archive supports exact model-level reproduction, not independent
reconstruction from every original source file.
