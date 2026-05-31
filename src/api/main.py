from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from pydantic import BaseModel, Field

from gridguard.config import settings
from gridguard.model_router import ModelRouter
from gridguard.mongo import PredictionLogger
from gridguard.predictor import ModelNotReadyError


app = FastAPI(
    title="GridGuard AI API",
    description=(
        "Risk-aware global and regional electricity-load forecasting API "
        "with registry-driven champion-specialist model routing."
    ),
    version="3.1.0",
)


REQUEST_COUNTER = Counter(
    "gridguard_prediction_requests_total",
    "Total number of prediction requests",
)

ERROR_COUNTER = Counter(
    "gridguard_prediction_errors_total",
    "Total number of prediction errors",
)

LATENCY_HISTOGRAM = Histogram(
    "gridguard_prediction_latency_seconds",
    "Prediction request latency in seconds",
)

PREDICTED_LOAD_GAUGE = Gauge(
    "gridguard_predicted_next_hour_load_mw",
    "Latest predicted next-hour electricity load in MW",
)

EXPECTED_CHANGE_GAUGE = Gauge(
    "gridguard_expected_load_change_mw",
    "Latest expected next-hour load change in MW",
)

PREDICTION_INTERVAL_WIDTH_GAUGE = Gauge(
    "gridguard_prediction_interval_width_mw",
    "Latest P10-P90 prediction interval width in MW",
)

ROUTING_COUNTER = Counter(
    "gridguard_model_routing_predictions_total",
    "Prediction count by routed model.",
    ["model_scope", "model_key", "fallback_used"],
)


metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

registry_path = Path(settings.model_dir) / "model_registry.json"
router = ModelRouter(registry_path=registry_path)

logger = PredictionLogger()


class PredictionRequest(BaseModel):
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        json_schema_extra={"example": "2025-08-15T18:00:00"},
        description="Timestamp of the current grid-load observation.",
    )

    country: str = Field(
        ...,
        json_schema_extra={"example": "Belgium"},
        description="Country or region name for the grid-load series.",
    )

    market: str = Field(
        ...,
        json_schema_extra={"example": "ENTSO-E"},
        description="Grid market/operator. The router uses this field only through the model registry.",
    )

    zone: str = Field(
        ...,
        json_schema_extra={"example": "Belgium"},
        description="Grid zone, balancing area, or regional load area.",
    )

    series_id: str = Field(
        ...,
        json_schema_extra={"example": "Belgium/ENTSO-E/Belgium"},
        description="Full series identifier in the format country/market/zone.",
    )

    load_mw: float = Field(
        ...,
        ge=0,
        json_schema_extra={"example": 5200.0},
        description="Current electricity load in MW.",
    )

    lag_1h: float = Field(5150.0, ge=0)
    lag_2h: float = Field(5050.0, ge=0)
    lag_3h: float = Field(4980.0, ge=0)
    lag_6h: float = Field(4700.0, ge=0)
    lag_12h: float = Field(4300.0, ge=0)
    lag_24h: float = Field(5100.0, ge=0)
    lag_48h: float = Field(5000.0, ge=0)
    lag_72h: float = Field(4950.0, ge=0)
    lag_168h: float = Field(5050.0, ge=0)

    rolling_mean_3h: float = Field(5060.0, ge=0)
    rolling_std_3h: float = Field(85.0, ge=0)
    rolling_min_3h: float = Field(4980.0, ge=0)
    rolling_max_3h: float = Field(5200.0, ge=0)

    rolling_mean_6h: float = Field(4900.0, ge=0)
    rolling_std_6h: float = Field(160.0, ge=0)
    rolling_min_6h: float = Field(4700.0, ge=0)
    rolling_max_6h: float = Field(5200.0, ge=0)

    rolling_mean_12h: float = Field(4650.0, ge=0)
    rolling_std_12h: float = Field(280.0, ge=0)
    rolling_min_12h: float = Field(4200.0, ge=0)
    rolling_max_12h: float = Field(5200.0, ge=0)

    rolling_mean_24h: float = Field(4850.0, ge=0)
    rolling_std_24h: float = Field(350.0, ge=0)
    rolling_min_24h: float = Field(4100.0, ge=0)
    rolling_max_24h: float = Field(5400.0, ge=0)

    rolling_mean_48h: float = Field(4800.0, ge=0)
    rolling_std_48h: float = Field(370.0, ge=0)
    rolling_min_48h: float = Field(4000.0, ge=0)
    rolling_max_48h: float = Field(5500.0, ge=0)

    rolling_mean_168h: float = Field(4750.0, ge=0)
    rolling_std_168h: float = Field(420.0, ge=0)
    rolling_min_168h: float = Field(3900.0, ge=0)
    rolling_max_168h: float = Field(5600.0, ge=0)


class FeedbackRequest(BaseModel):
    prediction_id: str | None = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    actual_next_hour_load_mw: float = Field(..., ge=0)
    predicted_next_hour_load_mw: float = Field(..., ge=0)
    user_comment: str | None = None


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Welcome to GridGuard AI",
        "use_case": "Registry-driven champion-specialist electricity-load forecasting",
        "docs": "/docs",
        "metrics": "/metrics",
        "health": "/health",
        "series_options": "/series/options",
        "router_status": "/router/status",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if router.is_ready else "model_not_ready",
        "app_name": settings.app_name,
        "environment": settings.environment,
        "model_ready": router.is_ready,
        "router_metadata": router.metadata,
        "available_series_count": len(router.get_available_series()),
    }


@app.get("/router/status")
def router_status() -> dict[str, Any]:
    metadata = router.metadata
    model_status = metadata.get("model_status", {})

    public_models = {}

    for model_key, status in model_status.items():
        deployment_decision = status.get("deployment_decision") or {}

        public_models[model_key] = {
            "scope": status.get("scope"),
            "status": status.get("status"),
            "match": status.get("match"),
            "loaded": status.get("loaded"),
            "ready": status.get("ready"),
            "artifact_valid": (
                status.get("artifact_validation", {}).get("is_valid")
                if status.get("artifact_validation")
                else None
            ),
            "approved": deployment_decision.get("approved"),
            "improvement_threshold_percent": deployment_decision.get(
                "min_wmape_relative_improvement_percent"
            ),
            "actual_improvement_percent": deployment_decision.get(
                "actual_wmape_relative_improvement_percent"
            ),
        }

    return {
        "router_type": metadata.get("router_type"),
        "registry_version": metadata.get("registry_version"),
        "registry_created_at_utc": metadata.get("registry_created_at_utc"),
        "deployment_mode": metadata.get("routing_policy", {}).get("deployment_mode"),
        "fallback_model_key": metadata.get("routing_policy", {}).get("fallback_model_key"),
        "routing_priority": metadata.get("routing_policy", {}).get("routing_priority"),
        "summary": metadata.get("summary"),
        "models": public_models,
    }


@app.get("/series/options")
def series_options() -> dict[str, Any]:
    series = router.get_available_series()

    return {
        "series": series,
        "count": len(series),
    }


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    start = time.perf_counter()
    REQUEST_COUNTER.inc()

    try:
        payload = request.model_dump(exclude_none=True)
        result = router.predict(payload)

        PREDICTED_LOAD_GAUGE.set(result["predicted_next_hour_load_mw"])
        EXPECTED_CHANGE_GAUGE.set(result["expected_change_mw"])

        p10 = result.get("predicted_load_p10_mw")
        p90 = result.get("predicted_load_p90_mw")

        if p10 is not None and p90 is not None:
            PREDICTION_INTERVAL_WIDTH_GAUGE.set(abs(p90 - p10))

        ROUTING_COUNTER.labels(
            model_scope=str(result.get("served_by_model_scope")),
            model_key=str(result.get("served_by_model_key")),
            fallback_used=str(result.get("fallback_used")),
        ).inc()

        elapsed = time.perf_counter() - start
        LATENCY_HISTOGRAM.observe(elapsed)

        log_record = {
            "request": payload,
            "response": result,
            "routing": result.get("routing"),
            "latency_seconds": elapsed,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        logger.log_prediction(log_record)

        return result

    except ModelNotReadyError as exc:
        ERROR_COUNTER.inc()
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    except Exception as exc:
        ERROR_COUNTER.inc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.post("/feedback")
def feedback(request: FeedbackRequest) -> dict[str, str]:
    logger.log_feedback(request.model_dump())
    return {"status": "feedback_logged"}


@app.get("/logs/latest")
def latest_logs(limit: int = 25) -> dict[str, Any]:
    limit = min(max(limit, 1), 100)
    return {"logs": logger.latest_predictions(limit=limit)}
