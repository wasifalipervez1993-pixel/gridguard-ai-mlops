from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = PROJECT_ROOT / "configs" / "model_registry_policy.json"


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def dump_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)


def require_key(payload: dict[str, Any], key: str) -> Any:
    if key not in payload:
        raise KeyError(f"Missing required policy key: {key}")

    return payload[key]


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def validate_artifacts(model_dir: Path, required_files: list[str]) -> dict[str, Any]:
    missing_files = []

    for filename in required_files:
        if not (model_dir / filename).exists():
            missing_files.append(filename)

    return {
        "model_dir": str(model_dir.relative_to(PROJECT_ROOT) if model_dir.is_relative_to(PROJECT_ROOT) else model_dir),
        "required_files": required_files,
        "missing_files": missing_files,
        "is_valid": len(missing_files) == 0,
    }


def build_global_model_entry(policy: dict[str, Any]) -> dict[str, Any]:
    global_key = require_key(policy, "fallback_model_key")
    global_model_dir = resolve_project_path(require_key(policy, "global_model_dir"))
    required_files = require_key(policy, "required_artifact_files")

    artifact_validation = validate_artifacts(
        model_dir=global_model_dir,
        required_files=required_files,
    )

    metadata_path = global_model_dir / "model_metadata.json"
    metrics_path = global_model_dir / "metrics.json"
    deployment_manifest_path = global_model_dir / "deployment_manifest.json"

    metadata = load_json(metadata_path) if metadata_path.exists() else {}
    metrics = load_json(metrics_path) if metrics_path.exists() else {}
    deployment_manifest = (
        load_json(deployment_manifest_path)
        if deployment_manifest_path.exists()
        else {}
    )

    return {
        "model_key": global_key,
        "scope": "global",
        "status": require_key(policy, "approved_status"),
        "model_dir": str(global_model_dir.relative_to(PROJECT_ROOT)),
        "match": {},
        "artifact_validation": artifact_validation,
        "metadata": metadata,
        "metrics": metrics,
        "deployment_manifest": deployment_manifest,
        "routing_note": "Default fallback model used when no approved specialist matches.",
    }


def specialist_scope_and_match(selection_scope: dict[str, Any]) -> tuple[str, dict[str, str]]:
    series_id = selection_scope.get("series_id")
    market = selection_scope.get("market")

    if series_id:
        return "series_id", {"series_id": str(series_id)}

    if market:
        return "market", {"market": str(market).upper().strip()}

    raise ValueError(
        "Specialist comparison report must contain either selection_scope.series_id "
        "or selection_scope.market."
    )


def build_specialist_entry(
    comparison_path: Path,
    policy: dict[str, Any],
) -> dict[str, Any]:
    comparison = load_json(comparison_path)

    specialist_name = comparison["specialist_name"]
    selection_scope = comparison["selection_scope"]

    scope, match = specialist_scope_and_match(selection_scope)

    specialist_root = resolve_project_path(require_key(policy, "specialist_artifacts_root"))
    model_dir = specialist_root / specialist_name
    required_files = require_key(policy, "required_artifact_files")

    artifact_validation = validate_artifacts(
        model_dir=model_dir,
        required_files=required_files,
    )

    improvement = float(
        comparison["improvement_specialist_vs_global"][
            "wmape_relative_improvement_percent"
        ]
    )

    min_improvement = float(
        require_key(policy, "min_wmape_relative_improvement_percent")
    )

    approved_status = require_key(policy, "approved_status")
    rejected_status = require_key(policy, "rejected_status")

    status = (
        approved_status
        if improvement >= min_improvement and artifact_validation["is_valid"]
        else rejected_status
    )

    metadata_path = model_dir / "model_metadata.json"
    metrics_path = model_dir / "metrics.json"
    deployment_manifest_path = model_dir / "deployment_manifest.json"

    metadata = load_json(metadata_path) if metadata_path.exists() else {}
    metrics = load_json(metrics_path) if metrics_path.exists() else {}
    deployment_manifest = (
        load_json(deployment_manifest_path)
        if deployment_manifest_path.exists()
        else {}
    )

    return {
        "model_key": specialist_name,
        "scope": scope,
        "status": status,
        "model_dir": str(model_dir.relative_to(PROJECT_ROOT)),
        "match": match,
        "artifact_validation": artifact_validation,
        "comparison_report": str(comparison_path.relative_to(PROJECT_ROOT)),
        "selection_scope": selection_scope,
        "global_model_on_region": comparison["global_model_on_region"],
        "specialist_model_on_region": comparison["specialist_model_on_region"],
        "improvement_specialist_vs_global": comparison[
            "improvement_specialist_vs_global"
        ],
        "deployment_decision": {
            "status": status,
            "min_wmape_relative_improvement_percent": min_improvement,
            "actual_wmape_relative_improvement_percent": improvement,
            "approved": status == approved_status,
            "reason": (
                "Approved because specialist improvement exceeds threshold "
                "and all required artifacts exist."
                if status == approved_status
                else "Rejected because improvement threshold or artifact validation failed."
            ),
        },
        "metadata": metadata,
        "metrics": metrics,
        "deployment_manifest": deployment_manifest,
    }


def build_registry(policy_path: Path = DEFAULT_POLICY_PATH) -> dict[str, Any]:
    policy = load_json(policy_path)

    registry_version = require_key(policy, "registry_version")
    specialist_report_glob = require_key(policy, "specialist_report_glob")
    registry_output_path = resolve_project_path(
        require_key(policy, "registry_output_path")
    )

    global_entry = build_global_model_entry(policy)

    models: dict[str, Any] = {
        global_entry["model_key"]: global_entry,
    }

    comparison_paths = sorted(PROJECT_ROOT.glob(specialist_report_glob))

    for comparison_path in comparison_paths:
        specialist_entry = build_specialist_entry(
            comparison_path=comparison_path,
            policy=policy,
        )

        models[specialist_entry["model_key"]] = specialist_entry

    approved_specialists = [
        model_key
        for model_key, model in models.items()
        if model_key != policy["fallback_model_key"]
        and model["status"] == policy["approved_status"]
    ]

    rejected_specialists = [
        model_key
        for model_key, model in models.items()
        if model_key != policy["fallback_model_key"]
        and model["status"] == policy["rejected_status"]
    ]

    registry = {
        "registry_version": registry_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy_source": str(policy_path.relative_to(PROJECT_ROOT)),
        "routing_policy": {
            "deployment_mode": require_key(policy, "deployment_mode"),
            "fallback_model_key": require_key(policy, "fallback_model_key"),
            "routing_priority": require_key(policy, "routing_priority"),
            "min_wmape_relative_improvement_percent": require_key(
                policy,
                "min_wmape_relative_improvement_percent",
            ),
            "approved_status": require_key(policy, "approved_status"),
            "rejected_status": require_key(policy, "rejected_status"),
        },
        "summary": {
            "model_count": len(models),
            "approved_specialists": approved_specialists,
            "rejected_specialists": rejected_specialists,
        },
        "models": models,
    }

    dump_json(registry, registry_output_path)

    print("\n==============================")
    print("MODEL REGISTRY BUILT")
    print("==============================")
    print(f"Registry path: {registry_output_path}")
    print(f"Models: {len(models)}")
    print(f"Approved specialists: {approved_specialists}")
    print(f"Rejected specialists: {rejected_specialists}")

    return registry


def main() -> None:
    build_registry()


if __name__ == "__main__":
    main()
