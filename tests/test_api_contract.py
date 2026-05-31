from __future__ import annotations

import pytest


@pytest.mark.integration
def test_health_endpoint(api_client) -> None:
    response = api_client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_ready"] is True
    assert data["available_series_count"] >= 1


@pytest.mark.integration
def test_router_status_endpoint(api_client) -> None:
    response = api_client.get("/router/status")

    assert response.status_code == 200

    data = response.json()

    assert data["router_type"] == "registry_driven_champion_specialist_router"
    assert data["fallback_model_key"] == "global"

    approved_specialists = data["summary"]["approved_specialists"]

    assert "market_AEMO" in approved_specialists
    assert "market_NYISO" in approved_specialists

    models = data["models"]

    assert models["global"]["ready"] is True
    assert models["global"]["artifact_valid"] is True

    assert models["market_AEMO"]["ready"] is True
    assert models["market_AEMO"]["artifact_valid"] is True
    assert models["market_AEMO"]["approved"] is True

    assert models["market_NYISO"]["ready"] is True
    assert models["market_NYISO"]["artifact_valid"] is True
    assert models["market_NYISO"]["approved"] is True


@pytest.mark.integration
def test_series_options_endpoint(api_client) -> None:
    response = api_client.get("/series/options")

    assert response.status_code == 200

    data = response.json()

    assert data["count"] >= 30
    assert "USA/NYISO/N.Y.C." in data["series"]
    assert "Australia/AEMO/SA1" in data["series"]
    assert "Belgium/ENTSO-E/Belgium" in data["series"]


@pytest.mark.integration
def test_predict_routes_to_nyiso_specialist(api_client, payload_factory) -> None:
    payload = payload_factory(
        country="USA",
        market="NYISO",
        zone="N.Y.C.",
        series_id="USA/NYISO/N.Y.C.",
        load_mw=5200,
    )

    response = api_client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["served_by_model_key"] == "market_NYISO"
    assert data["served_by_model_scope"] == "market"
    assert data["fallback_used"] is False

    assert data["prediction_interval_available"] is True
    assert "predicted_next_hour_load_mw" in data
    assert "predicted_load_p10_mw" in data
    assert "predicted_load_p90_mw" in data
    assert "operational_lower_bound_mw" in data
    assert "operational_upper_bound_mw" in data
    assert "is_anomaly" in data
    assert "routing" in data


@pytest.mark.integration
def test_predict_routes_to_aemo_specialist(api_client, payload_factory) -> None:
    payload = payload_factory(
        country="Australia",
        market="AEMO",
        zone="SA1",
        series_id="Australia/AEMO/SA1",
        load_mw=1500,
    )

    response = api_client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["served_by_model_key"] == "market_AEMO"
    assert data["served_by_model_scope"] == "market"
    assert data["fallback_used"] is False

    assert data["prediction_interval_available"] is True
    assert "predicted_next_hour_load_mw" in data
    assert "predicted_load_p10_mw" in data
    assert "predicted_load_p90_mw" in data
    assert "is_anomaly" in data
    assert "routing" in data


@pytest.mark.integration
def test_predict_routes_to_global_fallback_for_entsoe(api_client, payload_factory) -> None:
    payload = payload_factory(
        country="Belgium",
        market="ENTSO-E",
        zone="Belgium",
        series_id="Belgium/ENTSO-E/Belgium",
        load_mw=11000,
    )

    response = api_client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["served_by_model_key"] == "global"
    assert data["served_by_model_scope"] == "global_fallback"
    assert data["fallback_used"] is True

    assert data["prediction_interval_available"] is True
    assert "predicted_next_hour_load_mw" in data
    assert "predicted_load_p10_mw" in data
    assert "predicted_load_p90_mw" in data
    assert "is_anomaly" in data
    assert "routing" in data


@pytest.mark.integration
def test_metrics_endpoint_exposes_custom_metrics(api_client) -> None:
    response = api_client.get("/metrics")

    assert response.status_code == 200

    text = response.text

    assert "gridguard_prediction_requests_total" in text
    assert "gridguard_prediction_errors_total" in text
    assert "gridguard_prediction_latency_seconds" in text
    assert "gridguard_model_routing_predictions_total" in text
