from __future__ import annotations

from pathlib import Path

import pytest


FORBIDDEN_ROUTER_TOKENS = [
    '"AEMO"',
    "'AEMO'",
    '"NYISO"',
    "'NYISO'",
    "market_AEMO",
    "market_NYISO",
]


@pytest.mark.governance
def test_model_router_does_not_hardcode_specialist_markets(project_root: Path) -> None:
    router_path = project_root / "src" / "gridguard" / "model_router.py"

    assert router_path.exists(), "model_router.py does not exist."

    source = router_path.read_text(encoding="utf-8")

    forbidden_found = [
        token for token in FORBIDDEN_ROUTER_TOKENS if token in source
    ]

    assert not forbidden_found, (
        "Router contains hardcoded specialist market/model tokens: "
        f"{forbidden_found}. Use model_registry.json instead."
    )


@pytest.mark.governance
def test_api_uses_model_registry_not_hardcoded_specialists(project_root: Path) -> None:
    api_path = project_root / "src" / "api" / "main.py"

    assert api_path.exists(), "api/main.py does not exist."

    source = api_path.read_text(encoding="utf-8")

    assert "model_registry.json" in source
    assert "ModelRouter" in source

    forbidden_found = [
        token for token in FORBIDDEN_ROUTER_TOKENS if token in source
    ]

    assert not forbidden_found, (
        "API contains hardcoded specialist market/model tokens: "
        f"{forbidden_found}. Use model_registry.json instead."
    )
