from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "gloelecload_features.parquet"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"

TARGET_COLUMN = "target_load_mw_next_hour"
TIMESTAMP_COLUMN = "timestamp"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    model = joblib.load(ARTIFACT_DIR / "model.joblib")
    feature_columns = load_json(ARTIFACT_DIR / "feature_columns.json")
    category_levels = load_json(ARTIFACT_DIR / "category_levels.json")

    df = pd.read_parquet(PROCESSED_PATH)

    sample = df[df[TIMESTAMP_COLUMN] >= "2025-01-01"].tail(1000)

    x = sample[feature_columns].copy()

    for column in ["country", "market", "zone", "series_id"]:
        if column in x.columns:
            levels = category_levels.get(column)
            if levels:
                x[column] = pd.Categorical(x[column].astype(str), categories=levels)
            else:
                x[column] = x[column].astype("category")

    start = time.perf_counter()
    _ = model.predict(x)
    total_time = time.perf_counter() - start

    per_prediction_ms = total_time / len(x) * 1000

    result = {
        "sample_size": int(len(x)),
        "total_prediction_seconds": float(total_time),
        "average_latency_ms_per_prediction": float(per_prediction_ms),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with (REPORTS_DIR / "inference_benchmark.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(result, file, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
