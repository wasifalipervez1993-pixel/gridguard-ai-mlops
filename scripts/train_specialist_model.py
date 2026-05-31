from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from gridguard.train import train_models  # noqa: E402


DEFAULT_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "gloelecload_features.parquet"
SPECIALIST_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "specialists"
SPECIALIST_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "specialists"
SPECIALIST_REPORT_DIR = PROJECT_ROOT / "reports" / "specialists"

TIMESTAMP_COLUMN = "timestamp"
TARGET_COLUMN = "target_load_mw_next_hour"


def safe_name(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^A-Za-z0-9_\-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def remove_unused_categories(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for column in ["country", "market", "zone", "series_id"]:
        if column in out.columns and hasattr(out[column], "cat"):
            out[column] = out[column].cat.remove_unused_categories()

    return out


def build_specialist_dataset(
    processed_path: Path,
    market: str | None,
    series_id: str | None,
) -> tuple[pd.DataFrame, str]:
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {processed_path}. "
            "Run python scripts/make_dataset.py first."
        )

    df = pd.read_parquet(processed_path)

    if market:
        specialist_df = df[df["market"].astype(str) == market].copy()
        specialist_name = f"market_{market}"

    elif series_id:
        specialist_df = df[df["series_id"].astype(str) == series_id].copy()
        specialist_name = f"series_{series_id}"

    else:
        raise ValueError("You must provide either --market or --series-id.")

    if specialist_df.empty:
        raise ValueError(
            "No rows found for the requested specialist. "
            f"market={market}, series_id={series_id}"
        )

    specialist_df[TIMESTAMP_COLUMN] = pd.to_datetime(
        specialist_df[TIMESTAMP_COLUMN],
        errors="coerce",
    )

    specialist_df = specialist_df.dropna(subset=[TIMESTAMP_COLUMN, TARGET_COLUMN])
    specialist_df = specialist_df.sort_values(TIMESTAMP_COLUMN)
    specialist_df = remove_unused_categories(specialist_df)

    train_rows = int((specialist_df[TIMESTAMP_COLUMN] < "2024-01-01").sum())

    val_rows = int(
        (
            (specialist_df[TIMESTAMP_COLUMN] >= "2024-01-01")
            & (specialist_df[TIMESTAMP_COLUMN] < "2025-01-01")
        ).sum()
    )

    test_rows = int((specialist_df[TIMESTAMP_COLUMN] >= "2025-01-01").sum())

    if train_rows == 0 or val_rows == 0 or test_rows == 0:
        raise ValueError(
            "Specialist split is not valid. Need train rows before 2024, "
            "validation rows in 2024, and test rows in 2025. "
            f"Got train={train_rows}, val={val_rows}, test={test_rows}."
        )

    print("\n==============================")
    print("SPECIALIST DATASET SUMMARY")
    print("==============================")
    print(f"Specialist: {specialist_name}")
    print(f"Rows:       {len(specialist_df):,}")
    print(f"Train:      {train_rows:,}")
    print(f"Validation: {val_rows:,}")
    print(f"Test:       {test_rows:,}")
    print(f"Series:     {specialist_df['series_id'].nunique()}")
    print(f"Date range: {specialist_df[TIMESTAMP_COLUMN].min()} to {specialist_df[TIMESTAMP_COLUMN].max()}")

    return specialist_df, safe_name(specialist_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a regional or zone-specific GridGuard AI specialist model."
    )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--market",
        type=str,
        help="Market/operator to train specialist for, e.g., AEMO, NYISO, ENTSO-E.",
    )

    group.add_argument(
        "--series-id",
        type=str,
        help="Single series_id to train specialist for, e.g., Australia/AEMO/SA1.",
    )

    parser.add_argument(
        "--processed-path",
        type=Path,
        default=DEFAULT_PROCESSED_PATH,
        help="Path to global processed parquet dataset.",
    )

    args = parser.parse_args()

    specialist_df, specialist_name = build_specialist_dataset(
        processed_path=args.processed_path,
        market=args.market,
        series_id=args.series_id,
    )

    SPECIALIST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    specialist_processed_path = SPECIALIST_DATA_DIR / f"{specialist_name}_features.parquet"

    specialist_df.to_parquet(specialist_processed_path, index=False)

    artifact_dir = SPECIALIST_ARTIFACT_DIR / specialist_name
    reports_dir = SPECIALIST_REPORT_DIR / specialist_name

    print("\n==============================")
    print("TRAINING SPECIALIST MODEL")
    print("==============================")
    print(f"Processed specialist data: {specialist_processed_path}")
    print(f"Artifact directory:        {artifact_dir}")
    print(f"Report directory:          {reports_dir}")

    report = train_models(
        processed_path=specialist_processed_path,
        artifact_dir=artifact_dir,
        reports_dir=reports_dir,
    )

    print("\n==============================")
    print("SPECIALIST TRAINING COMPLETE")
    print("==============================")
    print(f"Specialist name: {specialist_name}")
    print(f"WMAPE: {report['model_metrics']['wmape_percent']:.4f}%")
    print(f"MAE:   {report['model_metrics']['mae_mw']:.4f} MW")
    print(f"RMSE:  {report['model_metrics']['rmse_mw']:.4f} MW")
    print(f"Reports saved to: {reports_dir}")
    print(f"Artifacts saved to: {artifact_dir}")


if __name__ == "__main__":
    main()
