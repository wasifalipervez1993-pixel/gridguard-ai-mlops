from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "reports" / "training_report.json"
SERIES_PATH = PROJECT_ROOT / "reports" / "per_series_metrics.json"
FEATURE_PATH = PROJECT_ROOT / "artifacts" / "feature_importance.json"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    report = load_json(REPORT_PATH)

    if "per_series_metrics" in report:
        df_series = pd.DataFrame(report["per_series_metrics"])
    else:
        df_series = pd.DataFrame(load_json(SERIES_PATH))

    df_feat = pd.DataFrame(load_json(FEATURE_PATH))

    total_rows = int(df_series["rows"].sum())
    rows_by_market = (
        df_series.groupby("market", observed=True)["rows"]
        .sum()
        .sort_values(ascending=False)
    )
    share_by_market = rows_by_market / total_rows * 100

    print("\n==============================")
    print("GRIDGUARD AI MODEL RISK AUDIT")
    print("==============================")

    print(f"\nTotal 2025 test rows: {total_rows:,}")

    print("\nRows by market:")
    for market, rows in rows_by_market.items():
        print(f"  {market:10s}: {int(rows):8,d} rows | {share_by_market[market]:6.2f}%")

    nyiso_rows = int(rows_by_market.get("NYISO", 0))
    print(f"\nNYISO row share: {nyiso_rows / total_rows * 100:.2f}%")

    print("\nPer-market average WMAPE:")
    market_wmape = (
        df_series.groupby("market", observed=True)["wmape_percent"]
        .mean()
        .sort_values(ascending=False)
    )
    for market, wmape in market_wmape.items():
        print(f"  {market:10s}: {wmape:.3f}%")

    print("\nWorst 10 series by WMAPE:")
    worst = df_series.sort_values("wmape_percent", ascending=False).head(10)
    print(worst[["series_id", "rows", "wmape_percent", "r2"]].to_string(index=False))

    print("\nFeature concentration:")
    top_6_gain = float(df_feat["gain_share_percent"].head(6).sum())
    top_10_gain = float(df_feat["gain_share_percent"].head(10).sum())
    print(f"  Top 6 gain share:  {top_6_gain:.2f}%")
    print(f"  Top 10 gain share: {top_10_gain:.2f}%")

    print("\nTop 15 features:")
    print(
        df_feat[["feature", "gain_share_percent", "split_importance"]]
        .head(15)
        .to_string(index=False)
    )

    print("\nRecommended interpretation:")
    print("- Use micro WMAPE as operational volume-weighted accuracy.")
    print("- Use macro market/series WMAPE as fairness/generalization accuracy.")
    print("- Use worst-series WMAPE to identify zones needing weather or local calibration.")
    print("- Use feature concentration to justify a lean production-model experiment.")


if __name__ == "__main__":
    main()
