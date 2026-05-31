from __future__ import annotations

import os
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLUMN = "target_load_mw_next_hour"


def _normalise_name(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^A-Za-z0-9_\-/\.]", "", value)
    return value


def _parse_file_metadata(file_path: str, load_column: str) -> dict[str, str]:
    parts = file_path.split("/")
    filename = parts[-1].replace(".csv", "")

    country = parts[0] if len(parts) >= 1 else "unknown"
    market = "unknown"

    if len(parts) >= 3:
        market = parts[1]
    else:
        match = re.match(r"(.+?)_(.+?)_\d{4}$", filename)
        if match:
            market = match.group(2)

    zone = load_column
    series_id = f"{country}/{market}/{zone}"

    return {
        "country": _normalise_name(country),
        "market": _normalise_name(market),
        "zone": _normalise_name(zone),
        "series_id": _normalise_name(series_id),
    }


def _read_date_window_from_env() -> tuple[pd.Timestamp, pd.Timestamp]:
    start_date = os.getenv("GRIDGUARD_START_DATE", "2021-01-01")
    end_date = os.getenv("GRIDGUARD_END_DATE", "2026-01-01")

    return pd.Timestamp(start_date), pd.Timestamp(end_date)


def _read_series_limits_from_env() -> tuple[int, int]:
    max_series = int(os.getenv("GRIDGUARD_MAX_SERIES", "30"))
    min_rows_per_series = int(os.getenv("GRIDGUARD_MIN_ROWS_PER_SERIES", "10000"))

    return max_series, min_rows_per_series


def _csv_files_from_zip(zip_file: zipfile.ZipFile) -> list[str]:
    return [
        name
        for name in zip_file.namelist()
        if name.endswith(".csv") and not name.endswith("Additional_Information.csv")
    ]


def _count_series_rows(
    zip_file: zipfile.ZipFile,
    csv_files: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict[str, int]:
    series_counts: dict[str, int] = defaultdict(int)

    print("Pass 1/2: scanning available load series...")

    for index, file_path in enumerate(csv_files, start=1):
        try:
            raw = pd.read_csv(zip_file.open(file_path), low_memory=False)
        except Exception as exc:
            print(f"Skipping unreadable file: {file_path} | {exc}")
            continue

        if "Timestamp" not in raw.columns:
            continue

        raw["timestamp"] = pd.to_datetime(raw["Timestamp"], errors="coerce")
        raw = raw.dropna(subset=["timestamp"])

        raw = raw[
            (raw["timestamp"] >= start_date)
            & (raw["timestamp"] < end_date)
        ]

        if raw.empty:
            continue

        load_columns = [column for column in raw.columns if column not in {"Timestamp", "timestamp"}]

        for column in load_columns:
            metadata = _parse_file_metadata(file_path, column)
            numeric_col = pd.to_numeric(raw[column], errors="coerce")
            series_counts[metadata["series_id"]] += int(numeric_col.notna().sum())

        if index % 50 == 0:
            print(f"Scanned {index}/{len(csv_files)} files...")

    return dict(series_counts)


def _select_series(
    series_counts: dict[str, int],
    max_series: int,
    min_rows_per_series: int,
) -> set[str]:
    eligible = {
        series_id: count
        for series_id, count in series_counts.items()
        if count >= min_rows_per_series
    }

    if not eligible:
        raise RuntimeError(
            "No eligible series found. Try lowering GRIDGUARD_MIN_ROWS_PER_SERIES."
        )

    sorted_series = sorted(
        eligible.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    if max_series > 0:
        sorted_series = sorted_series[:max_series]

    selected = {series_id for series_id, _ in sorted_series}

    print(f"Eligible series: {len(eligible)}")
    print(f"Selected series for local training: {len(selected)}")

    print("Top selected series:")
    for series_id, count in sorted_series[:10]:
        print(f"  {series_id}: {count} raw records")

    return selected


def _load_selected_series(
    zip_file: zipfile.ZipFile,
    csv_files: list[str],
    selected_series: set[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict[str, list[pd.DataFrame]]:
    series_frames: dict[str, list[pd.DataFrame]] = defaultdict(list)

    print("Pass 2/2: loading selected series and harmonising to hourly data...")

    for index, file_path in enumerate(csv_files, start=1):
        try:
            raw = pd.read_csv(zip_file.open(file_path), low_memory=False)
        except Exception as exc:
            print(f"Skipping unreadable file: {file_path} | {exc}")
            continue

        if "Timestamp" not in raw.columns:
            continue

        raw["timestamp"] = pd.to_datetime(raw["Timestamp"], errors="coerce")
        raw = raw.dropna(subset=["timestamp"])

        raw = raw[
            (raw["timestamp"] >= start_date)
            & (raw["timestamp"] < end_date)
        ]

        if raw.empty:
            continue

        load_columns = [column for column in raw.columns if column not in {"Timestamp", "timestamp"}]

        for column in load_columns:
            metadata = _parse_file_metadata(file_path, column)
            series_id = metadata["series_id"]

            if series_id not in selected_series:
                continue

            temp = raw[["timestamp", column]].copy()
            temp[column] = pd.to_numeric(temp[column], errors="coerce")
            temp = temp.dropna(subset=[column])

            if temp.empty:
                continue

            temp = temp.rename(columns={column: "load_mw"})
            temp = temp.set_index("timestamp").sort_index()

            hourly = temp.resample("1h").mean()
            hourly = hourly.dropna(subset=["load_mw"])

            if hourly.empty:
                continue

            hourly["load_mw"] = hourly["load_mw"].astype("float32")
            hourly["country"] = metadata["country"]
            hourly["market"] = metadata["market"]
            hourly["zone"] = metadata["zone"]
            hourly["series_id"] = metadata["series_id"]

            hourly = hourly.reset_index()

            series_frames[series_id].append(hourly)

        if index % 50 == 0:
            print(f"Loaded {index}/{len(csv_files)} files...")

    return series_frames


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["hour"] = out["timestamp"].dt.hour.astype("int16")
    out["day_of_week"] = out["timestamp"].dt.dayofweek.astype("int16")
    out["month"] = out["timestamp"].dt.month.astype("int16")
    out["day_of_year"] = out["timestamp"].dt.dayofyear.astype("int16")
    out["week_of_year"] = out["timestamp"].dt.isocalendar().week.astype("int16")
    out["is_weekend"] = (out["day_of_week"] >= 5).astype("int8")

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24).astype("float32")
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24).astype("float32")

    out["day_of_week_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7).astype("float32")
    out["day_of_week_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7).astype("float32")

    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12).astype("float32")
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12).astype("float32")

    return out


def _add_series_features(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("timestamp").copy()

    lag_hours = [1, 2, 3, 6, 12, 24, 48, 72, 168]

    for lag in lag_hours:
        group[f"lag_{lag}h"] = group["load_mw"].shift(lag).astype("float32")

    rolling_windows = [3, 6, 12, 24, 48, 168]

    for window in rolling_windows:
        rolling = group["load_mw"].rolling(
            window=window,
            min_periods=max(2, window // 3),
        )

        group[f"rolling_mean_{window}h"] = rolling.mean().astype("float32")
        group[f"rolling_std_{window}h"] = rolling.std().astype("float32")
        group[f"rolling_min_{window}h"] = rolling.min().astype("float32")
        group[f"rolling_max_{window}h"] = rolling.max().astype("float32")

    group["load_change_1h"] = (
        group["load_mw"] - group["lag_1h"]
    ).astype("float32")

    group["load_change_24h"] = (
        group["load_mw"] - group["lag_24h"]
    ).astype("float32")

    group["load_change_168h"] = (
        group["load_mw"] - group["lag_168h"]
    ).astype("float32")

    group["load_ratio_to_24h_mean"] = (
        group["load_mw"] / group["rolling_mean_24h"].replace(0, np.nan)
    ).astype("float32")

    group["load_ratio_to_168h_mean"] = (
        group["load_mw"] / group["rolling_mean_168h"].replace(0, np.nan)
    ).astype("float32")

    group[TARGET_COLUMN] = group["load_mw"].shift(-1).astype("float32")

    return group


def _build_feature_panel(series_frames: dict[str, list[pd.DataFrame]]) -> pd.DataFrame:
    feature_frames: list[pd.DataFrame] = []

    print("Creating lag, rolling, and target features series-by-series...")

    for index, (series_id, frames) in enumerate(series_frames.items(), start=1):
        if not frames:
            continue

        series_df = pd.concat(frames, ignore_index=True)
        series_df = series_df.drop_duplicates(subset=["series_id", "timestamp"])
        series_df = series_df.sort_values("timestamp")

        if len(series_df) < 1000:
            continue

        series_df = _add_time_features(series_df)
        series_df = _add_series_features(series_df)

        series_df = series_df.replace([np.inf, -np.inf], np.nan)
        series_df = series_df.dropna()

        if series_df.empty:
            continue

        feature_frames.append(series_df)

        if index % 5 == 0:
            print(f"Feature engineered {index}/{len(series_frames)} series...")

    if not feature_frames:
        raise RuntimeError("No feature frames were generated.")

    panel = pd.concat(feature_frames, ignore_index=True)

    for column in ["country", "market", "zone", "series_id"]:
        panel[column] = panel[column].astype("category")

    float_columns = panel.select_dtypes(include=["float64"]).columns

    for column in float_columns:
        panel[column] = panel[column].astype("float32")

    return panel


def build_feature_dataset(raw_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    raw_path = Path(raw_path)
    output_path = Path(output_path)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}")

    if raw_path.suffix.lower() != ".zip":
        raise ValueError(
            "This updated pipeline expects GloElecLoad.zip as input. "
            "Set DATA_RAW_PATH=data/raw/GloElecLoad.zip in .env."
        )

    start_date, end_date = _read_date_window_from_env()
    max_series, min_rows_per_series = _read_series_limits_from_env()

    print(f"Using date window: {start_date.date()} to {end_date.date()}")
    print(f"Maximum selected series: {max_series if max_series > 0 else 'all eligible'}")
    print(f"Minimum raw records per series: {min_rows_per_series}")

    with zipfile.ZipFile(raw_path, "r") as zip_file:
        csv_files = _csv_files_from_zip(zip_file)

        print(f"Found {len(csv_files)} electricity-load CSV files.")

        series_counts = _count_series_rows(
            zip_file=zip_file,
            csv_files=csv_files,
            start_date=start_date,
            end_date=end_date,
        )

        selected_series = _select_series(
            series_counts=series_counts,
            max_series=max_series,
            min_rows_per_series=min_rows_per_series,
        )

        series_frames = _load_selected_series(
            zip_file=zip_file,
            csv_files=csv_files,
            selected_series=selected_series,
            start_date=start_date,
            end_date=end_date,
        )

    panel = _build_feature_panel(series_frames)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output_path, index=False)

    print(f"Saved processed global electricity-load features: {output_path}")
    print(f"Final shape: {panel.shape}")
    print(f"Date range: {panel['timestamp'].min()} to {panel['timestamp'].max()}")
    print(f"Series count: {panel['series_id'].nunique()}")

    return panel
