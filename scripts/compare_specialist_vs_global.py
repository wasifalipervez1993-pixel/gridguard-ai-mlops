from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "gloelecload_features.parquet"
GLOBAL_ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
SPECIALIST_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "specialists"
SPECIALIST_REPORT_ROOT = PROJECT_ROOT / "reports" / "specialists"

TIMESTAMP_COLUMN = "timestamp"
TARGET_COLUMN = "target_load_mw_next_hour"


def safe_name(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^A-Za-z0-9_\-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def load_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def dump_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))

    wmape = (
        np.sum(np.abs(y_true - y_pred))
        / max(np.sum(np.abs(y_true)), 1e-9)
        * 100
    )

    smape = (
        np.mean(
            2
            * np.abs(y_pred - y_true)
            / np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-9)
        )
        * 100
    )

    r2 = r2_score(y_true, y_pred)

    return {
        "mae_mw": float(mae),
        "rmse_mw": float(rmse),
        "wmape_percent": float(wmape),
        "smape_percent": float(smape),
        "r2": float(r2),
    }


def load_model_bundle(artifact_dir: Path) -> dict[str, Any]:
    model_path = artifact_dir / "model.joblib"
    feature_columns_path = artifact_dir / "feature_columns.json"
    category_levels_path = artifact_dir / "category_levels.json"
    metadata_path = artifact_dir / "model_metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    return {
        "model": joblib.load(model_path),
        "feature_columns": load_json(feature_columns_path),
        "category_levels": load_json(category_levels_path, default={}),
        "metadata": load_json(metadata_path, default={}),
    }


def prepare_x(df: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    feature_columns = bundle["feature_columns"]
    category_levels = bundle["category_levels"]

    x = df[feature_columns].copy()

    for column in ["country", "market", "zone", "series_id"]:
        if column in x.columns:
            levels = category_levels.get(column)

            if levels:
                x[column] = pd.Categorical(x[column].astype(str), categories=levels)
            else:
                x[column] = x[column].astype("category")

    return x


def load_region_test_data(
    processed_path: Path,
    market: str | None,
    series_id: str | None,
) -> tuple[pd.DataFrame, str]:
    if not processed_path.exists():
        raise FileNotFoundError(processed_path)

    df = pd.read_parquet(processed_path)
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN], errors="coerce")

    df = df.dropna(subset=[TIMESTAMP_COLUMN, TARGET_COLUMN])
    df = df[df[TIMESTAMP_COLUMN] >= "2025-01-01"].copy()

    if market:
        region_df = df[df["market"].astype(str) == market].copy()
        specialist_name = f"market_{market}"

    elif series_id:
        region_df = df[df["series_id"].astype(str) == series_id].copy()
        specialist_name = f"series_{series_id}"

    else:
        raise ValueError("Provide either --market or --series-id.")

    if region_df.empty:
        raise ValueError(
            f"No 2025 test rows found for market={market}, series_id={series_id}"
        )

    region_df = region_df.sort_values(TIMESTAMP_COLUMN)

    return region_df, safe_name(specialist_name)


def baseline_metrics(region_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    y_true = region_df[TARGET_COLUMN].to_numpy()

    baselines = {
        "last_hour_baseline": region_df["load_mw"].to_numpy(),
        "same_hour_yesterday_baseline": region_df["lag_24h"].to_numpy(),
        "same_hour_last_week_baseline": region_df["lag_168h"].to_numpy(),
        "rolling_24h_average_baseline": region_df["rolling_mean_24h"].to_numpy(),
        "rolling_168h_average_baseline": region_df["rolling_mean_168h"].to_numpy(),
    }

    return {
        name: metrics(y_true, prediction)
        for name, prediction in baselines.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare regional specialist model against global champion on the same 2025 test subset."
    )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--market",
        type=str,
        help="Market/operator to compare, e.g., AEMO, NYISO, ENTSO-E.",
    )

    group.add_argument(
        "--series-id",
        type=str,
        help="Single series_id to compare, e.g., Australia/AEMO/SA1.",
    )

    parser.add_argument(
        "--processed-path",
        type=Path,
        default=DEFAULT_PROCESSED_PATH,
    )

    parser.add_argument(
        "--global-artifact-dir",
        type=Path,
        default=GLOBAL_ARTIFACT_DIR,
    )

    parser.add_argument(
        "--min-relative-improvement-percent",
        type=float,
        default=5.0,
        help="Minimum WMAPE relative improvement required to recommend specialist deployment.",
    )

    args = parser.parse_args()

    region_df, specialist_name = load_region_test_data(
        processed_path=args.processed_path,
        market=args.market,
        series_id=args.series_id,
    )

    specialist_artifact_dir = SPECIALIST_ARTIFACT_ROOT / specialist_name
    output_dir = SPECIALIST_REPORT_ROOT / specialist_name
    output_path = output_dir / "specialist_vs_global_comparison.json"

    global_bundle = load_model_bundle(args.global_artifact_dir)
    specialist_bundle = load_model_bundle(specialist_artifact_dir)

    y_true = region_df[TARGET_COLUMN].to_numpy()

    x_global = prepare_x(region_df, global_bundle)
    x_specialist = prepare_x(region_df, specialist_bundle)

    global_pred = global_bundle["model"].predict(x_global)
    specialist_pred = specialist_bundle["model"].predict(x_specialist)

    global_metrics = metrics(y_true, global_pred)
    specialist_metrics = metrics(y_true, specialist_pred)
    baselines = baseline_metrics(region_df)

    best_baseline_name = min(
        baselines,
        key=lambda name: baselines[name]["wmape_percent"],
    )

    best_baseline_metrics = baselines[best_baseline_name]

    wmape_relative_improvement = (
        (global_metrics["wmape_percent"] - specialist_metrics["wmape_percent"])
        / max(global_metrics["wmape_percent"], 1e-9)
        * 100
    )

    mae_relative_improvement = (
        (global_metrics["mae_mw"] - specialist_metrics["mae_mw"])
        / max(global_metrics["mae_mw"], 1e-9)
        * 100
    )

    specialist_vs_baseline_wmape_reduction = (
        (best_baseline_metrics["wmape_percent"] - specialist_metrics["wmape_percent"])
        / max(best_baseline_metrics["wmape_percent"], 1e-9)
        * 100
    )

    deploy_specialist = (
        wmape_relative_improvement >= args.min_relative_improvement_percent
    )

    comparison = {
        "specialist_name": specialist_name,
        "selection_scope": {
            "market": args.market,
            "series_id": args.series_id,
            "test_rows": int(len(region_df)),
            "series_count": int(region_df["series_id"].nunique()),
            "test_start": str(region_df[TIMESTAMP_COLUMN].min()),
            "test_end": str(region_df[TIMESTAMP_COLUMN].max()),
        },
        "global_model_on_region": global_metrics,
        "specialist_model_on_region": specialist_metrics,
        "baseline_metrics_on_region": baselines,
        "best_baseline_name": best_baseline_name,
        "improvement_specialist_vs_global": {
            "wmape_relative_improvement_percent": float(wmape_relative_improvement),
            "mae_relative_improvement_percent": float(mae_relative_improvement),
        },
        "improvement_specialist_vs_best_baseline": {
            "best_baseline_name": best_baseline_name,
            "wmape_reduction_percent": float(specialist_vs_baseline_wmape_reduction),
        },
        "deployment_decision": {
            "deploy_specialist": bool(deploy_specialist),
            "rule": (
                "Deploy only if specialist improves WMAPE over global model "
                f"by at least {args.min_relative_improvement_percent:.1f}%."
            ),
        },
    }

    dump_json(comparison, output_path)

    print("\n==============================================")
    print("SPECIALIST VS GLOBAL COMPARISON")
    print("==============================================")
    print(f"Specialist: {specialist_name}")
    print(f"Test rows:  {len(region_df):,}")
    print(f"Series:     {region_df['series_id'].nunique()}")
    print("\nGlobal model on region:")
    print(f"  WMAPE: {global_metrics['wmape_percent']:.4f}%")
    print(f"  MAE:   {global_metrics['mae_mw']:.4f} MW")
    print(f"  RMSE:  {global_metrics['rmse_mw']:.4f} MW")
    print("\nSpecialist model on same region:")
    print(f"  WMAPE: {specialist_metrics['wmape_percent']:.4f}%")
    print(f"  MAE:   {specialist_metrics['mae_mw']:.4f} MW")
    print(f"  RMSE:  {specialist_metrics['rmse_mw']:.4f} MW")
    print("\nImprovement specialist vs global:")
    print(f"  WMAPE relative improvement: {wmape_relative_improvement:.2f}%")
    print(f"  MAE relative improvement:   {mae_relative_improvement:.2f}%")
    print("\nDeployment decision:")
    print(f"  Deploy specialist: {deploy_specialist}")
    print(f"\nSaved comparison to: {output_path}")


if __name__ == "__main__":
    main()
