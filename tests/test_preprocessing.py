from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


TIMESTAMP_COLUMN = "timestamp"
TARGET_COLUMN = "target_load_mw_next_hour"

REQUIRED_COLUMNS = [
    "timestamp",
    "load_mw",
    "country",
    "market",
    "zone",
    "series_id",
    "hour",
    "day_of_week",
    "month",
    "day_of_year",
    "week_of_year",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "lag_1h",
    "lag_2h",
    "lag_3h",
    "lag_6h",
    "lag_12h",
    "lag_24h",
    "lag_48h",
    "lag_72h",
    "lag_168h",
    "rolling_mean_3h",
    "rolling_std_3h",
    "rolling_min_3h",
    "rolling_max_3h",
    "rolling_mean_6h",
    "rolling_std_6h",
    "rolling_min_6h",
    "rolling_max_6h",
    "rolling_mean_12h",
    "rolling_std_12h",
    "rolling_min_12h",
    "rolling_max_12h",
    "rolling_mean_24h",
    "rolling_std_24h",
    "rolling_min_24h",
    "rolling_max_24h",
    "rolling_mean_48h",
    "rolling_std_48h",
    "rolling_min_48h",
    "rolling_max_48h",
    "rolling_mean_168h",
    "rolling_std_168h",
    "rolling_min_168h",
    "rolling_max_168h",
    "load_change_1h",
    "load_change_24h",
    "load_change_168h",
    "load_ratio_to_24h_mean",
    "load_ratio_to_168h_mean",
    "target_load_mw_next_hour",
]

EXPECTED_MARKETS = {"AEMO", "NYISO", "ENTSO-E"}

MAX_NEGATIVE_LOAD_SHARE = 0.01
MAX_NEGATIVE_TARGET_SHARE = 0.01
MIN_POSITIVE_RATIO_SHARE = 0.99


def require_dataset(path: Path) -> Path:
    if not path.exists():
        pytest.skip(
            "Processed dataset not found. Run `python scripts/make_dataset.py` first."
        )

    return path


@pytest.mark.contract
def test_processed_dataset_file_exists(processed_dataset_path: Path) -> None:
    path = require_dataset(processed_dataset_path)

    assert path.exists()
    assert path.stat().st_size > 0


@pytest.mark.contract
def test_processed_dataset_schema_contains_required_columns(
    processed_dataset_path: Path,
) -> None:
    path = require_dataset(processed_dataset_path)

    df_sample = pd.read_parquet(path).head(10)
    columns = set(df_sample.columns)

    missing_columns = sorted(set(REQUIRED_COLUMNS) - columns)

    assert not missing_columns, f"Missing required columns: {missing_columns}"


@pytest.mark.contract
def test_processed_dataset_has_valid_targets_and_markets(
    processed_dataset_path: Path,
) -> None:
    path = require_dataset(processed_dataset_path)

    df = pd.read_parquet(
        path,
        columns=[
            TIMESTAMP_COLUMN,
            "load_mw",
            TARGET_COLUMN,
            "market",
            "series_id",
        ],
    )

    assert len(df) > 0
    assert df[TIMESTAMP_COLUMN].notna().all()
    assert df["load_mw"].notna().all()
    assert df[TARGET_COLUMN].notna().all()

    assert np.isfinite(df["load_mw"]).all()
    assert np.isfinite(df[TARGET_COLUMN]).all()

    negative_load_share = float((df["load_mw"] < 0).mean())
    negative_target_share = float((df[TARGET_COLUMN] < 0).mean())

    assert negative_load_share <= MAX_NEGATIVE_LOAD_SHARE, (
        f"Negative load share too high: {negative_load_share:.4%}. "
        "Investigate raw series with net-export or invalid load values."
    )

    assert negative_target_share <= MAX_NEGATIVE_TARGET_SHARE, (
        f"Negative target share too high: {negative_target_share:.4%}. "
        "Investigate target-generation step."
    )

    assert df["market"].nunique() >= 3
    assert df["series_id"].nunique() >= 10


@pytest.mark.contract
def test_processed_dataset_contains_expected_markets(
    processed_dataset_path: Path,
) -> None:
    path = require_dataset(processed_dataset_path)

    df = pd.read_parquet(path, columns=["market"])
    observed_markets = set(df["market"].astype(str).unique())

    missing_markets = sorted(EXPECTED_MARKETS - observed_markets)

    assert not missing_markets, f"Missing expected markets: {missing_markets}"


@pytest.mark.contract
def test_processed_dataset_supports_chronological_split(
    processed_dataset_path: Path,
) -> None:
    path = require_dataset(processed_dataset_path)

    df = pd.read_parquet(path, columns=[TIMESTAMP_COLUMN])
    timestamps = pd.to_datetime(df[TIMESTAMP_COLUMN], errors="coerce")

    train_rows = int((timestamps < "2024-01-01").sum())

    validation_rows = int(
        ((timestamps >= "2024-01-01") & (timestamps < "2025-01-01")).sum()
    )

    test_rows = int((timestamps >= "2025-01-01").sum())

    assert train_rows > 0
    assert validation_rows > 0
    assert test_rows > 0


@pytest.mark.contract
def test_engineered_features_have_reasonable_ranges(
    processed_dataset_path: Path,
) -> None:
    path = require_dataset(processed_dataset_path)

    selected_columns = [
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",
        "load_ratio_to_24h_mean",
        "load_ratio_to_168h_mean",
    ]

    df = pd.read_parquet(path, columns=selected_columns)

    assert df["hour"].between(0, 23).all()
    assert df["day_of_week"].between(0, 6).all()
    assert df["month"].between(1, 12).all()
    assert df["is_weekend"].isin([0, 1]).all()

    cyclic_columns = [
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",
    ]

    for column in cyclic_columns:
        assert df[column].between(-1.0001, 1.0001).all()

    ratio_columns = [
        "load_ratio_to_24h_mean",
        "load_ratio_to_168h_mean",
    ]

    for column in ratio_columns:
        assert df[column].notna().all()
        assert np.isfinite(df[column]).all()

        positive_share = float((df[column] > 0).mean())

        assert positive_share >= MIN_POSITIVE_RATIO_SHARE, (
            f"{column} positive share too low: {positive_share:.4%}. "
            "Investigate negative/zero rolling-mean periods."
        )
