from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def artifacts_dir(project_root: Path) -> Path:
    return project_root / "artifacts"


@pytest.fixture(scope="session")
def processed_dataset_path(project_root: Path) -> Path:
    return project_root / "data" / "processed" / "gloelecload_features.parquet"


@pytest.fixture(scope="session")
def api_client() -> Any:
    from fastapi.testclient import TestClient
    from api.main import app

    return TestClient(app)


@pytest.fixture()
def payload_factory() -> Any:
    def _make_payload(
        country: str,
        market: str,
        zone: str,
        series_id: str,
        load_mw: float,
    ) -> dict[str, Any]:
        return {
            "timestamp": "2025-08-15T18:00:00",
            "country": country,
            "market": market,
            "zone": zone,
            "series_id": series_id,
            "load_mw": load_mw,
            "lag_1h": load_mw * 0.99,
            "lag_2h": load_mw * 0.97,
            "lag_3h": load_mw * 0.96,
            "lag_6h": load_mw * 0.92,
            "lag_12h": load_mw * 0.86,
            "lag_24h": load_mw * 0.98,
            "lag_48h": load_mw * 0.96,
            "lag_72h": load_mw * 0.95,
            "lag_168h": load_mw * 0.97,
            "rolling_mean_3h": load_mw * 0.97,
            "rolling_std_3h": max(load_mw * 0.02, 1.0),
            "rolling_min_3h": load_mw * 0.95,
            "rolling_max_3h": load_mw * 1.01,
            "rolling_mean_6h": load_mw * 0.95,
            "rolling_std_6h": max(load_mw * 0.04, 1.0),
            "rolling_min_6h": load_mw * 0.90,
            "rolling_max_6h": load_mw * 1.01,
            "rolling_mean_12h": load_mw * 0.92,
            "rolling_std_12h": max(load_mw * 0.06, 1.0),
            "rolling_min_12h": load_mw * 0.84,
            "rolling_max_12h": load_mw * 1.02,
            "rolling_mean_24h": load_mw * 0.94,
            "rolling_std_24h": max(load_mw * 0.08, 1.0),
            "rolling_min_24h": load_mw * 0.80,
            "rolling_max_24h": load_mw * 1.04,
            "rolling_mean_48h": load_mw * 0.93,
            "rolling_std_48h": max(load_mw * 0.09, 1.0),
            "rolling_min_48h": load_mw * 0.78,
            "rolling_max_48h": load_mw * 1.05,
            "rolling_mean_168h": load_mw * 0.92,
            "rolling_std_168h": max(load_mw * 0.10, 1.0),
            "rolling_min_168h": load_mw * 0.75,
            "rolling_max_168h": load_mw * 1.06,
        }

    return _make_payload
