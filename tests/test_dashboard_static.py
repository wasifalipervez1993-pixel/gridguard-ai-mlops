from __future__ import annotations

import py_compile
from pathlib import Path

import pytest


@pytest.mark.dashboard
def test_dashboard_app_exists(project_root: Path) -> None:
    dashboard_path = project_root / "dashboard" / "app.py"

    assert dashboard_path.exists()
    assert dashboard_path.stat().st_size > 0


@pytest.mark.dashboard
def test_dashboard_app_has_no_syntax_errors(project_root: Path) -> None:
    dashboard_path = project_root / "dashboard" / "app.py"

    py_compile.compile(str(dashboard_path), doraise=True)


@pytest.mark.dashboard
def test_dashboard_contains_key_enterprise_sections(project_root: Path) -> None:
    dashboard_path = project_root / "dashboard" / "app.py"
    source = dashboard_path.read_text(encoding="utf-8")

    required_sections = [
        "Live Forecast",
        "Model Routing",
        "Business Impact",
        "Monitoring",
        "Prediction Logs",
        "Feature Inputs",
        "Raw Payloads",
    ]

    missing_sections = [
        section for section in required_sections if section not in source
    ]

    assert not missing_sections, f"Dashboard missing sections: {missing_sections}"
