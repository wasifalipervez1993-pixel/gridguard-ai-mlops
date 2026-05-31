from __future__ import annotations

import json
from pathlib import Path

import pytest


REQUIRED_ARTIFACT_FILES = [
    "model.joblib",
    "model_p10.joblib",
    "model_p90.joblib",
    "anomaly_model.joblib",
    "feature_columns.json",
    "category_levels.json",
    "model_metadata.json",
    "deployment_manifest.json",
]


def load_registry(artifacts_dir: Path) -> dict:
    registry_path = artifacts_dir / "model_registry.json"

    if not registry_path.exists():
        pytest.fail(
            "Model registry not found. Run `python scripts/build_model_registry.py`."
        )

    with registry_path.open("r", encoding="utf-8") as file:
        return json.load(file)


@pytest.mark.governance
def test_model_registry_exists_and_has_expected_structure(artifacts_dir: Path) -> None:
    registry = load_registry(artifacts_dir)

    assert registry["registry_version"]
    assert registry["routing_policy"]["deployment_mode"] == "champion_specialist"
    assert registry["routing_policy"]["fallback_model_key"] == "global"
    assert registry["routing_policy"]["routing_priority"] == ["series_id", "market"]

    assert "models" in registry
    assert "global" in registry["models"]


@pytest.mark.governance
def test_registry_has_approved_specialists(artifacts_dir: Path) -> None:
    registry = load_registry(artifacts_dir)

    approved = registry["summary"]["approved_specialists"]

    assert "market_AEMO" in approved
    assert "market_NYISO" in approved


@pytest.mark.governance
def test_all_approved_models_have_required_artifacts(
    project_root: Path,
    artifacts_dir: Path,
) -> None:
    registry = load_registry(artifacts_dir)

    approved_status = registry["routing_policy"]["approved_status"]

    for model_key, model_spec in registry["models"].items():
        if model_spec["status"] != approved_status:
            continue

        model_dir = project_root / model_spec["model_dir"]

        assert model_dir.exists(), f"Model directory missing for {model_key}"

        missing_files = [
            filename
            for filename in REQUIRED_ARTIFACT_FILES
            if not (model_dir / filename).exists()
        ]

        assert not missing_files, f"{model_key} missing files: {missing_files}"


@pytest.mark.governance
def test_specialists_are_approved_by_threshold_and_artifacts(
    artifacts_dir: Path,
) -> None:
    registry = load_registry(artifacts_dir)

    for model_key in registry["summary"]["approved_specialists"]:
        model_spec = registry["models"][model_key]
        decision = model_spec["deployment_decision"]

        assert decision["approved"] is True
        assert decision["actual_wmape_relative_improvement_percent"] >= decision[
            "min_wmape_relative_improvement_percent"
        ]
        assert model_spec["artifact_validation"]["is_valid"] is True
