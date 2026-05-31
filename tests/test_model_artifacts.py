from __future__ import annotations

import json
from pathlib import Path

import joblib
import pytest


MODEL_DIRS = [
    "artifacts",
    "artifacts/specialists/market_AEMO",
    "artifacts/specialists/market_NYISO",
]


@pytest.mark.contract
@pytest.mark.parametrize("relative_dir", MODEL_DIRS)
def test_model_artifact_files_are_loadable(project_root: Path, relative_dir: str) -> None:
    model_dir = project_root / relative_dir

    if not model_dir.exists():
        pytest.fail(f"Model directory does not exist: {model_dir}")

    model = joblib.load(model_dir / "model.joblib")
    model_p10 = joblib.load(model_dir / "model_p10.joblib")
    model_p90 = joblib.load(model_dir / "model_p90.joblib")
    anomaly_model = joblib.load(model_dir / "anomaly_model.joblib")

    assert model is not None
    assert model_p10 is not None
    assert model_p90 is not None
    assert anomaly_model is not None


@pytest.mark.contract
@pytest.mark.parametrize("relative_dir", MODEL_DIRS)
def test_model_metadata_contract(project_root: Path, relative_dir: str) -> None:
    model_dir = project_root / relative_dir
    metadata_path = model_dir / "model_metadata.json"
    feature_columns_path = model_dir / "feature_columns.json"
    category_levels_path = model_dir / "category_levels.json"

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    with feature_columns_path.open("r", encoding="utf-8") as file:
        feature_columns = json.load(file)

    with category_levels_path.open("r", encoding="utf-8") as file:
        category_levels = json.load(file)

    assert metadata["project"] == "GridGuard AI"
    assert metadata["target"] == "target_load_mw_next_hour"
    assert metadata["feature_count"] == len(feature_columns)
    assert metadata["quantile_forecasting_enabled"] is True
    assert metadata["prediction_interval"] == "P10-P90"

    assert "country" in category_levels
    assert "market" in category_levels
    assert "zone" in category_levels
    assert "series_id" in category_levels
