"""Shared schema for the dissertation's analysis-ready dataset."""

from __future__ import annotations

TARGET = "serious_or_fatal"
YEAR = "collision_year"
EXPECTED_ROWS = 503_475
EXPECTED_YEARS = {2020, 2021, 2022, 2023, 2024}
EXPECTED_DATASET_SHA256 = (
    "5e629f2d931948429580ed778b636b31aa0775630b3e4475727e39df8ee630e1"
)

NUMERIC_FEATURES = [
    "month",
    "hour",
    "is_weekend",
    "is_night",
    "longitude",
    "latitude",
    "number_of_vehicles",
    "speed_limit",
    "traffic_link_length_km",
    "traffic_all_motor_vehicles",
    "traffic_all_motor_vehicles_per_km",
    "traffic_cars_taxis_share",
    "vehicle_record_count",
    "n_pedal_cycles",
    "n_motorcycles",
    "n_cars_taxis",
    "n_buses_minibuses",
    "n_goods_vehicles",
    "vehicle_type_nunique",
    "mean_driver_age",
    "min_driver_age",
    "max_driver_age",
    "any_young_driver_17_24",
    "any_older_driver_65_plus",
    "mean_vehicle_age",
    "max_vehicle_age",
]

CATEGORICAL_FEATURES = [
    "day_of_week",
    "police_force",
    "local_authority_highway",
    "urban_or_rural_area",
    "first_road_class",
    "road_type",
    "junction_detail",
    "junction_control",
    "pedestrian_crossing",
    "light_conditions",
    "weather_conditions",
    "road_surface_conditions",
    "special_conditions_at_site",
    "carriageway_hazards",
    "trunk_road_flag",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CORE_REQUIRED = {
    "collision_index",
    YEAR,
    "collision_severity",
    TARGET,
}

AUDIT_FIELDS = {
    "traffic_merge_matched",
    "traffic_local_authority_name",
}

DISSERTATION_REQUIRED = CORE_REQUIRED.union(MODEL_FEATURES, AUDIT_FIELDS)

UNKNOWN_VALUE_MAP = {
    "speed_limit": [-1],
    "mean_driver_age": [-1],
    "min_driver_age": [-1],
    "max_driver_age": [-1],
    "mean_vehicle_age": [-1],
    "max_vehicle_age": [-1],
}
