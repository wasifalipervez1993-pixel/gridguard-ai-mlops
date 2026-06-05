from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gridguard.predictor import GridGuardPredictor, ModelNotReadyError


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ModelRouter:
    """
    Registry-driven champion-specialist model router.

    No market, region, or series is hardcoded here.

    The router reads artifacts/model_registry.json and routes using:
    - registry routing_priority
    - approved model status
    - match rules stored in registry
    - global fallback model from registry
    """

    def __init__(self, registry_path: str | Path) -> None:
        self.registry_path = self._resolve_path(registry_path)
        self.registry = self._load_registry(self.registry_path)

        self.routing_policy = self.registry["routing_policy"]
        self.models = self.registry["models"]

        self.fallback_model_key = self.routing_policy["fallback_model_key"]
        self.approved_status = self.routing_policy["approved_status"]
        self.routing_priority = self.routing_policy["routing_priority"]

        self.predictors: dict[str, GridGuardPredictor] = {}

        self._load_approved_predictors()

    @staticmethod
    def _resolve_path(path: str | Path) -> Path:
        # Registry paths must be portable across Windows, Linux CI, Docker, and Hugging Face.
        # Normalize any old Windows-style backslashes before creating a Path object.
        normalized = str(path).replace("\\", "/")
        value = Path(normalized)

        if value.is_absolute():
            return value

        return PROJECT_ROOT / value

    @staticmethod
    def _load_registry(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(
                f"Model registry not found: {path}. "
                "Run `python scripts/build_model_registry.py` first."
            )

        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _model_dir(self, model_spec: dict[str, Any]) -> Path:
        return self._resolve_path(model_spec["model_dir"])

    def _load_approved_predictors(self) -> None:
        for model_key, model_spec in self.models.items():
            if model_spec.get("status") != self.approved_status:
                continue

            model_dir = self._model_dir(model_spec)
            predictor = GridGuardPredictor(model_dir)

            if predictor.is_ready:
                self.predictors[model_key] = predictor

    @property
    def is_ready(self) -> bool:
        fallback_predictor = self.predictors.get(self.fallback_model_key)
        return fallback_predictor is not None and fallback_predictor.is_ready

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "router_type": "registry_driven_champion_specialist_router",
            "registry_path": str(self.registry_path),
            "registry_version": self.registry.get("registry_version"),
            "registry_created_at_utc": self.registry.get("created_at_utc"),
            "routing_policy": self.routing_policy,
            "summary": self.registry.get("summary", {}),
            "model_status": self.get_model_status(),
        }

    def get_model_status(self) -> dict[str, Any]:
        status: dict[str, Any] = {}

        for model_key, model_spec in self.models.items():
            predictor = self.predictors.get(model_key)

            status[model_key] = {
                "scope": model_spec.get("scope"),
                "status": model_spec.get("status"),
                "match": model_spec.get("match", {}),
                "model_dir": model_spec.get("model_dir"),
                "loaded": predictor is not None,
                "ready": predictor.is_ready if predictor is not None else False,
                "artifact_validation": model_spec.get("artifact_validation"),
                "deployment_decision": model_spec.get("deployment_decision"),
            }

        return status

    def get_available_series(self) -> list[str]:
        series = set()

        for predictor in self.predictors.values():
            series.update(predictor.get_available_series())

        return sorted(series)

    @staticmethod
    def _normalize_value(field: str, value: Any) -> str:
        if value is None:
            return ""

        text = str(value).strip()

        if field == "market":
            return text.upper()

        return text

    def _match_model(
        self,
        payload: dict[str, Any],
    ) -> tuple[str | None, dict[str, Any] | None, str | None]:
        for field in self.routing_priority:
            request_value = self._normalize_value(field, payload.get(field))

            if not request_value:
                continue

            for model_key, model_spec in self.models.items():
                if model_key == self.fallback_model_key:
                    continue

                if model_spec.get("status") != self.approved_status:
                    continue

                match_rules = model_spec.get("match", {})
                expected_value = self._normalize_value(field, match_rules.get(field))

                if expected_value and expected_value == request_value:
                    return model_key, model_spec, field

        return None, None, None

    def _fallback_metadata(
        self,
        requested_market: str,
        reason: str,
    ) -> dict[str, Any]:
        fallback_spec = self.models[self.fallback_model_key]

        return {
            "model_scope": "global_fallback",
            "selected_model_key": self.fallback_model_key,
            "selected_market": requested_market or "unknown",
            "selected_model_dir": fallback_spec["model_dir"],
            "matched_field": None,
            "fallback_used": True,
            "routing_reason": reason,
            "registry_version": self.registry.get("registry_version"),
        }

    def _select_predictor(
        self,
        payload: dict[str, Any],
    ) -> tuple[GridGuardPredictor, dict[str, Any]]:
        if not self.is_ready:
            raise ModelNotReadyError(
                "Global fallback model is not ready. "
                "Run training and rebuild the model registry."
            )

        requested_market = self._normalize_value("market", payload.get("market"))

        matched_key, matched_spec, matched_field = self._match_model(payload)

        if matched_key and matched_spec:
            predictor = self.predictors.get(matched_key)

            if predictor is not None and predictor.is_ready:
                return predictor, {
                    "model_scope": matched_spec.get("scope"),
                    "selected_model_key": matched_key,
                    "selected_market": requested_market or "unknown",
                    "selected_model_dir": matched_spec["model_dir"],
                    "matched_field": matched_field,
                    "match": matched_spec.get("match", {}),
                    "fallback_used": False,
                    "routing_reason": (
                        "Approved specialist selected from model registry."
                    ),
                    "registry_version": self.registry.get("registry_version"),
                }

            fallback_predictor = self.predictors[self.fallback_model_key]

            return fallback_predictor, self._fallback_metadata(
                requested_market=requested_market,
                reason=(
                    f"Matched specialist {matched_key} exists in registry "
                    "but predictor is not loaded or not ready. "
                    "Using global fallback."
                ),
            )

        fallback_predictor = self.predictors[self.fallback_model_key]

        return fallback_predictor, self._fallback_metadata(
            requested_market=requested_market,
            reason=(
                "No approved specialist matched this request in the model registry. "
                "Using global fallback."
            ),
        )

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        predictor, routing_metadata = self._select_predictor(payload)

        result = predictor.predict(payload)

        result["routing"] = routing_metadata
        result["served_by_model_scope"] = routing_metadata["model_scope"]
        result["served_by_model_key"] = routing_metadata["selected_model_key"]
        result["fallback_used"] = routing_metadata["fallback_used"]

        return result
