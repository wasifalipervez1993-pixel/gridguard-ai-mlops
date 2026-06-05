from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.mark.governance
def test_model_registry_paths_are_posix_portable(artifacts_dir: Path) -> None:
    registry_path = artifacts_dir / "model_registry.json"

    assert registry_path.exists(), "model_registry.json is missing."

    registry = json.loads(registry_path.read_text(encoding="utf-8"))

    for model_key, model_spec in registry.get("models", {}).items():
        model_dir = model_spec.get("model_dir", "")

        assert "\\" not in model_dir, (
            f"{model_key} has non-portable Windows path in model_dir: {model_dir}"
        )

        artifact_validation = model_spec.get("artifact_validation", {})
        artifact_model_dir = artifact_validation.get("model_dir", "")

        assert "\\" not in artifact_model_dir, (
            f"{model_key} has non-portable Windows path in artifact_validation.model_dir: "
            f"{artifact_model_dir}"
        )

        comparison_report = model_spec.get("comparison_report", "")

        assert "\\" not in comparison_report, (
            f"{model_key} has non-portable Windows path in comparison_report: "
            f"{comparison_report}"
        )
