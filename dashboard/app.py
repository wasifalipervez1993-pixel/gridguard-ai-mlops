from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="GridGuard AI | Enterprise Grid Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


DEFAULT_API_URL = "http://127.0.0.1:8000"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_REGISTRY_PATH = PROJECT_ROOT / "artifacts" / "model_registry.json"


# =============================================================================
# STYLING
# =============================================================================
def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --gg-bg: #0b0f19;
            --gg-card: #111827;
            --gg-card-2: #0f172a;
            --gg-border: #263244;
            --gg-text: #f8fafc;
            --gg-muted: #cbd5e1;
            --gg-soft: #94a3b8;
            --gg-blue: #3b82f6;
            --gg-red: #ff4b4b;
            --gg-green: #22c55e;
            --gg-yellow: #f59e0b;
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1480px;
        }

        [data-testid="stSidebar"] {
            min-width: 305px;
            max-width: 365px;
        }

        .hero-card {
            background: linear-gradient(135deg, #111827 0%, #1e3a8a 55%, #2563eb 100%);
            color: white;
            padding: 26px 32px;
            border-radius: 24px;
            margin-bottom: 18px;
            box-shadow: 0 12px 34px rgba(0, 0, 0, 0.35);
            border: 1px solid rgba(255, 255, 255, 0.10);
        }

        .hero-card h1 {
            color: #ffffff !important;
            margin-bottom: 10px;
            font-size: clamp(1.7rem, 2.2vw, 2.45rem);
            line-height: 1.25;
            font-weight: 800;
        }

        .hero-card p {
            color: #e5edf9 !important;
            font-size: clamp(0.95rem, 1.1vw, 1.1rem);
            line-height: 1.65;
            margin-bottom: 0;
        }

        .gg-metric-card {
            background: linear-gradient(145deg, var(--gg-card) 0%, var(--gg-card-2) 100%);
            border: 1px solid var(--gg-border);
            border-radius: 18px;
            padding: 17px 18px;
            min-height: 112px;
            box-shadow: 0 8px 22px rgba(0, 0, 0, 0.24);
            overflow-wrap: anywhere;
        }

        .gg-metric-label {
            color: var(--gg-muted);
            font-size: 0.84rem;
            font-weight: 600;
            margin-bottom: 7px;
            letter-spacing: 0.01em;
        }

        .gg-metric-value {
            color: var(--gg-text);
            font-size: clamp(1.18rem, 1.45vw, 1.62rem);
            font-weight: 800;
            line-height: 1.25;
            margin-bottom: 6px;
        }

        .gg-metric-help {
            color: var(--gg-soft);
            font-size: 0.78rem;
            line-height: 1.35;
            margin-top: 6px;
        }

        .section-note {
            background-color: #101827;
            border: 1px solid #1e3a8a;
            border-left: 5px solid var(--gg-blue);
            padding: 14px 18px;
            border-radius: 14px;
            margin-top: 8px;
            margin-bottom: 14px;
            color: var(--gg-text);
            line-height: 1.55;
        }

        .success-box {
            background-color: #052e1a;
            border: 1px solid #16a34a;
            padding: 14px 18px;
            border-radius: 14px;
            color: #bbf7d0;
            line-height: 1.55;
        }

        .warning-box {
            background-color: #422006;
            border: 1px solid #f59e0b;
            padding: 14px 18px;
            border-radius: 14px;
            color: #fde68a;
            line-height: 1.55;
        }

        .danger-box {
            background-color: #450a0a;
            border: 1px solid #ef4444;
            padding: 14px 18px;
            border-radius: 14px;
            color: #fecaca;
            line-height: 1.55;
        }

        .info-box {
            background-color: #0b2447;
            border: 1px solid #3b82f6;
            padding: 14px 18px;
            border-radius: 14px;
            color: #bfdbfe;
            line-height: 1.55;
        }

        .small-muted {
            color: var(--gg-soft);
            font-size: 0.90rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 1px solid #263244;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 12px 12px 0 0;
            padding: 10px 15px;
            font-weight: 700;
        }

        h1, h2, h3 {
            color: #f8fafc !important;
            letter-spacing: -0.02em;
        }

        p, li, span {
            line-height: 1.55;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 14px;
            overflow: hidden;
        }

        hr {
            margin-top: 1.2rem;
            margin-bottom: 1.2rem;
        }

        @media (max-width: 1200px) {
            .block-container {
                max-width: 100%;
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .hero-card {
                padding: 22px 24px;
            }

            .gg-metric-card {
                min-height: 100px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


# =============================================================================
# API HELPERS
# =============================================================================
def api_get(api_url: str, endpoint: str) -> dict[str, Any] | None:
    try:
        response = requests.get(f"{api_url}{endpoint}", timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        st.error(f"Could not fetch `{endpoint}` from API: {exc}")
        return None


def api_post(
    api_url: str,
    endpoint: str,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    try:
        response = requests.post(
            f"{api_url}{endpoint}",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        st.error(f"Prediction request failed: {exc}")
        return None


def get_text(api_url: str, endpoint: str) -> str:
    try:
        response = requests.get(f"{api_url}{endpoint}", timeout=15)
        response.raise_for_status()
        return response.text
    except Exception as exc:
        return f"Could not fetch {endpoint}: {exc}"


def load_local_registry() -> dict[str, Any] | None:
    if not LOCAL_REGISTRY_PATH.exists():
        return None

    try:
        with LOCAL_REGISTRY_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return None


def parse_prometheus_metrics(metrics_text: str) -> dict[str, float]:
    parsed: dict[str, float] = {}

    for line in metrics_text.splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        if " " not in line:
            continue

        key, value = line.rsplit(" ", 1)

        try:
            parsed[key] = float(value)
        except ValueError:
            continue

    return parsed


def parse_prometheus_labels(metric_name: str) -> dict[str, str]:
    match = re.search(r"\{(.+)\}", metric_name)

    if not match:
        return {}

    label_text = match.group(1)
    labels: dict[str, str] = {}

    for item in label_text.split(","):
        if "=" not in item:
            continue

        key, value = item.split("=", 1)
        labels[key.strip()] = value.strip().strip('"')

    return labels


# =============================================================================
# GENERAL HELPERS
# =============================================================================
def format_mw(value: Any) -> str:
    try:
        return f"{float(value):,.2f} MW"
    except Exception:
        return "N/A"


def format_percent(value: Any) -> str:
    try:
        number = float(value)

        if number == 0:
            return "0.00%"

        if abs(number) < 0.01:
            return f"{number:.4f}%"

        return f"{number:.2f}%"

    except Exception:
        return "N/A"


def infer_country_from_series(series_id: str) -> str:
    parts = series_id.split("/")
    return parts[0] if len(parts) >= 1 else ""


def infer_market_from_series(series_id: str) -> str:
    parts = series_id.split("/")
    return parts[1] if len(parts) >= 2 else ""


def infer_zone_from_series(series_id: str) -> str:
    parts = series_id.split("/")
    return parts[2] if len(parts) >= 3 else ""


DEFAULT_LOAD_BY_SERIES = {
    "USA/NYISO/N.Y.C.": 5200.0,
    "USA/NYISO/CAPITL": 1700.0,
    "USA/NYISO/CENTRL": 1700.0,
    "USA/NYISO/DUNWOD": 650.0,
    "USA/NYISO/GENESE": 1100.0,
    "USA/NYISO/HUD_VL": 1200.0,
    "USA/NYISO/LONGIL": 3100.0,
    "USA/NYISO/MHK_VL": 850.0,
    "USA/NYISO/MILLWD": 500.0,
    "USA/NYISO/NORTH": 600.0,
    "USA/NYISO/WEST": 1700.0,
    "Australia/AEMO/NSW1": 8500.0,
    "Australia/AEMO/QLD1": 7000.0,
    "Australia/AEMO/SA1": 1500.0,
    "Australia/AEMO/TAS1": 1200.0,
    "Australia/AEMO/VIC1": 5600.0,
    "Singapore/NEMS/Singapore": 7000.0,
    "USA/MISO/MISO": 80000.0,
    "Germany/ENTSO-E/Germany": 55000.0,
    "Belgium/ENTSO-E/Belgium": 11000.0,
    "Netherlands/ENTSO-E/Netherlands": 15000.0,
    "Spain/ENTSO-E/Spain": 30000.0,
    "Poland/ENTSO-E/Poland": 22000.0,
    "Finland/ENTSO-E/Finland": 10000.0,
    "Austria/ENTSO-E/Austria": 8500.0,
    "Ireland/ENTSO-E/Ireland": 5000.0,
    "Croatia/ENTSO-E/Croatia": 2500.0,
    "Hungary/ENTSO-E/Hungary": 5500.0,
    "Romania/ENTSO-E/Romania": 7000.0,
    "Luxembourg/ENTSO-E/Luxembourg": 900.0,
}


def default_load_for_series(series_id: str) -> float:
    return DEFAULT_LOAD_BY_SERIES.get(series_id, 5000.0)


def build_reasonable_defaults(load_mw: float, scenario: str) -> dict[str, float]:
    if scenario == "Rising demand":
        factors = {
            "lag_1h": 0.97,
            "lag_2h": 0.95,
            "lag_3h": 0.93,
            "lag_6h": 0.90,
            "lag_12h": 0.84,
            "lag_24h": 0.96,
            "lag_48h": 0.94,
            "lag_72h": 0.93,
            "lag_168h": 0.95,
        }
    elif scenario == "Falling demand":
        factors = {
            "lag_1h": 1.03,
            "lag_2h": 1.05,
            "lag_3h": 1.06,
            "lag_6h": 1.08,
            "lag_12h": 1.10,
            "lag_24h": 1.04,
            "lag_48h": 1.05,
            "lag_72h": 1.06,
            "lag_168h": 1.04,
        }
    else:
        factors = {
            "lag_1h": 0.99,
            "lag_2h": 0.97,
            "lag_3h": 0.96,
            "lag_6h": 0.92,
            "lag_12h": 0.86,
            "lag_24h": 0.98,
            "lag_48h": 0.96,
            "lag_72h": 0.95,
            "lag_168h": 0.97,
        }

    values = {key: load_mw * factor for key, factor in factors.items()}

    recent_values = [
        values["lag_1h"],
        values["lag_2h"],
        values["lag_3h"],
        load_mw,
    ]

    values.update(
        {
            "rolling_mean_3h": sum(recent_values[:3]) / 3,
            "rolling_std_3h": max(load_mw * 0.015, 1.0),
            "rolling_min_3h": min(recent_values[:3]),
            "rolling_max_3h": max(recent_values),
            "rolling_mean_6h": load_mw * 0.95,
            "rolling_std_6h": max(load_mw * 0.035, 1.0),
            "rolling_min_6h": load_mw * 0.90,
            "rolling_max_6h": load_mw * 1.02,
            "rolling_mean_12h": load_mw * 0.92,
            "rolling_std_12h": max(load_mw * 0.055, 1.0),
            "rolling_min_12h": load_mw * 0.84,
            "rolling_max_12h": load_mw * 1.03,
            "rolling_mean_24h": load_mw * 0.94,
            "rolling_std_24h": max(load_mw * 0.075, 1.0),
            "rolling_min_24h": load_mw * 0.80,
            "rolling_max_24h": load_mw * 1.05,
            "rolling_mean_48h": load_mw * 0.93,
            "rolling_std_48h": max(load_mw * 0.085, 1.0),
            "rolling_min_48h": load_mw * 0.78,
            "rolling_max_48h": load_mw * 1.06,
            "rolling_mean_168h": load_mw * 0.92,
            "rolling_std_168h": max(load_mw * 0.10, 1.0),
            "rolling_min_168h": load_mw * 0.75,
            "rolling_max_168h": load_mw * 1.08,
        }
    )

    return values


def risk_label(result: dict[str, Any]) -> tuple[str, str, str]:
    is_anomaly = bool(result.get("is_anomaly", False))
    expected_change_percent = abs(float(result.get("expected_change_percent", 0.0)))

    if is_anomaly:
        return (
            "Critical attention",
            "The input pattern is flagged as abnormal by the anomaly detector.",
            "danger",
        )

    if expected_change_percent >= 10:
        return (
            "Watch",
            "The next-hour load change is large. Operators should monitor this case.",
            "warning",
        )

    if expected_change_percent >= 5:
        return (
            "Moderate",
            "The next-hour change is noticeable but not abnormal.",
            "info",
        )

    return (
        "Normal",
        "The forecasted operating condition looks stable.",
        "success",
    )


def model_scope_label(result: dict[str, Any]) -> str:
    scope = result.get("served_by_model_scope", "unknown")
    model_key = result.get("served_by_model_key", "unknown")

    if scope == "market":
        return f"Regional specialist: {model_key}"

    if scope == "global_fallback":
        return "Global fallback model"

    return f"{scope}: {model_key}"


def make_forecast_chart(result: dict[str, Any]) -> go.Figure:
    current = float(result.get("current_load_mw", 0.0))
    point = float(result.get("predicted_next_hour_load_mw", 0.0))

    p10 = result.get("predicted_load_p10_mw")
    p90 = result.get("predicted_load_p90_mw")
    op_low = result.get("operational_lower_bound_mw")
    op_high = result.get("operational_upper_bound_mw")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=["Current load", "Next-hour forecast"],
            y=[current, point],
            mode="lines+markers",
            name="Point forecast",
            line=dict(width=4),
            marker=dict(size=12),
        )
    )

    if p10 is not None and p90 is not None:
        fig.add_trace(
            go.Scatter(
                x=["P10 estimate", "P90 estimate"],
                y=[float(p10), float(p90)],
                mode="lines+markers",
                name="Raw P10–P90 estimate",
                line=dict(width=3, dash="dot"),
                marker=dict(size=10),
            )
        )

    if op_low is not None and op_high is not None:
        fig.add_trace(
            go.Scatter(
                x=["Operational lower", "Operational upper"],
                y=[float(op_low), float(op_high)],
                mode="lines+markers",
                name="Display-safe operating range",
                line=dict(width=5),
                marker=dict(size=11),
            )
        )

    fig.update_layout(
        title="Forecast, Uncertainty Range, and Operating Bounds",
        xaxis_title="Forecast component",
        yaxis_title="Load (MW)",
        height=430,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=20, r=20, t=80, b=20),
    )

    return fig


def make_delta_chart(result: dict[str, Any]) -> go.Figure:
    current = float(result.get("current_load_mw", 0.0))
    point = float(result.get("predicted_next_hour_load_mw", 0.0))
    delta = point - current

    display_limit = max(abs(delta) * 1.6, max(current * 0.01, 1.0))

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=["Next-hour change"],
            y=[delta],
            text=[format_mw(delta)],
            textposition="outside",
            name="Expected change",
        )
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_width=1,
        opacity=0.6,
    )

    fig.update_layout(
        title="Expected Load Change",
        yaxis_title="MW",
        xaxis_title="",
        height=340,
        margin=dict(l=20, r=20, t=70, b=30),
        showlegend=False,
        yaxis=dict(range=[-display_limit, display_limit]),
    )

    return fig


def make_registry_table(router_status: dict[str, Any]) -> pd.DataFrame:
    models = router_status.get("models", {})
    rows = []

    for model_key, item in models.items():
        rows.append(
            {
                "Model": model_key,
                "Scope": item.get("scope"),
                "Status": item.get("status"),
                "Match rule": json.dumps(item.get("match", {})),
                "Ready": item.get("ready"),
                "Artifacts valid": item.get("artifact_valid"),
                "Approved": item.get("approved"),
                "Threshold (%)": item.get("improvement_threshold_percent"),
                "Improvement (%)": item.get("actual_improvement_percent"),
            }
        )

    return pd.DataFrame(rows)


def make_business_impact_table(registry: dict[str, Any] | None) -> pd.DataFrame:
    if not registry:
        return pd.DataFrame()

    rows = []

    for model_key, spec in registry.get("models", {}).items():
        metrics = spec.get("metrics", {})
        decision = spec.get("deployment_decision", {})
        selection = spec.get("selection_scope", {})

        rows.append(
            {
                "Model": model_key,
                "Scope": spec.get("scope"),
                "Status": spec.get("status"),
                "Test rows": selection.get("test_rows", spec.get("metadata", {}).get("test_rows")),
                "Series count": selection.get("series_count", spec.get("metadata", {}).get("series_count")),
                "WMAPE (%)": metrics.get("wmape_percent"),
                "MAE (MW)": metrics.get("mae_mw"),
                "RMSE (MW)": metrics.get("rmse_mw"),
                "Worst-series WMAPE (%)": metrics.get("worst_series_wmape_percent"),
                "Improvement vs global (%)": decision.get("actual_wmape_relative_improvement_percent"),
            }
        )

    return pd.DataFrame(rows)


def make_improvement_chart(registry: dict[str, Any] | None) -> go.Figure:
    labels: list[str] = []
    values: list[float] = []

    if registry:
        for model_key, spec in registry.get("models", {}).items():
            if model_key == "global":
                continue

            decision = spec.get("deployment_decision", {})
            improvement = decision.get("actual_wmape_relative_improvement_percent")

            if improvement is not None:
                labels.append(model_key)
                values.append(float(improvement))

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            text=[f"{value:.2f}%" for value in values],
            textposition="outside",
            name="Improvement",
        )
    )

    fig.add_hline(
        y=5.0,
        line_dash="dash",
        annotation_text="Deployment threshold: 5%",
        annotation_position="top left",
    )

    fig.update_layout(
        title="Approved Specialist Improvement vs Global Model",
        xaxis_title="Specialist model",
        yaxis_title="Relative WMAPE improvement (%)",
        height=380,
        margin=dict(l=20, r=20, t=70, b=20),
        showlegend=False,
    )

    return fig


def show_metric_card(label: str, value: Any, help_text: str | None = None) -> None:
    safe_label = html.escape(str(label))
    safe_value = html.escape(str(value))
    safe_help = html.escape(str(help_text)) if help_text else ""

    help_html = (
        f'<div class="gg-metric-help">{safe_help}</div>'
        if safe_help
        else ""
    )

    st.markdown(
        f"""
        <div class="gg-metric-card">
            <div class="gg-metric-label">{safe_label}</div>
            <div class="gg-metric-value">{safe_value}</div>
            {help_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

def show_status_box(kind: str, text: str) -> None:
    css_class = {
        "success": "success-box",
        "warning": "warning-box",
        "danger": "danger-box",
        "info": "info-box",
    }.get(kind, "info-box")

    st.markdown(f'<div class="{css_class}">{text}</div>', unsafe_allow_html=True)


# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.title("⚙️ GridGuard AI Controls")

api_url = st.sidebar.text_input("API URL", value=DEFAULT_API_URL)

health = api_get(api_url, "/health")
router_status = api_get(api_url, "/router/status")
series_options = api_get(api_url, "/series/options")
local_registry = load_local_registry()

if not health or not router_status or not series_options:
    st.stop()

available_series = series_options.get("series", [])

if not available_series:
    st.error("No grid series were returned by the API.")
    st.stop()

st.sidebar.success("API connected")

default_series = "USA/NYISO/N.Y.C."

selected_series = st.sidebar.selectbox(
    "Select grid series",
    options=available_series,
    index=available_series.index(default_series)
    if default_series in available_series
    else 0,
    help="The selected series determines country, market, zone, and routing behavior.",
)

country = infer_country_from_series(selected_series)
market = infer_market_from_series(selected_series)
zone = infer_zone_from_series(selected_series)

st.sidebar.markdown("#### Detected routing fields")
st.sidebar.write(f"**Country:** {country}")
st.sidebar.write(f"**Market:** {market}")
st.sidebar.write(f"**Zone:** {zone}")

timestamp_value = st.sidebar.text_input(
    "Prediction timestamp",
    value="2025-08-15T18:00:00",
    help="Timestamp of the current observation. The model predicts the next hour.",
)

scenario = st.sidebar.selectbox(
    "Scenario profile",
    options=["Stable demand", "Rising demand", "Falling demand"],
    index=0,
    help="Generates reasonable lag/rolling inputs for demonstration.",
)

base_load = default_load_for_series(selected_series)
series_key = re.sub(r"[^A-Za-z0-9_]+", "_", selected_series)

load_mw = st.sidebar.number_input(
    "Current load (MW)",
    min_value=0.0,
    value=float(base_load),
    step=max(float(base_load) * 0.01, 1.0),
    key=f"load_mw_{series_key}",
)

st.sidebar.caption(
    "The generated inputs are scenario values for live demonstration. "
    "For strict historical evaluation, use rows from the processed test set."
)

defaults = build_reasonable_defaults(load_mw, scenario)

with st.sidebar.expander("Advanced load-history inputs", expanded=False):
    user_inputs = {}

    for key, value in defaults.items():
        user_inputs[key] = st.number_input(
            key,
            min_value=0.0,
            value=float(round(value, 3)),
            step=max(float(load_mw) * 0.005, 1.0),
            key=f"{series_key}_{scenario}_{key}",
        )

run_prediction = st.sidebar.button("Run forecast", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### Routing expectation")
if market in ["AEMO", "NYISO"]:
    st.sidebar.info(f"Expected route: `{market}` regional specialist")
else:
    st.sidebar.info("Expected route: global fallback model")


# =============================================================================
# HEADER
# =============================================================================
st.markdown(
    """
    <div class="hero-card">
        <h1>⚡ GridGuard AI — Enterprise Grid Forecasting Dashboard</h1>
        <p>
        Registry-driven champion–specialist electricity-load forecasting with
        regional model routing, P10–P90 operating range, anomaly screening,
        prediction logging, and Prometheus monitoring.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("How to read this dashboard", expanded=False):
    st.write(
        """
        This dashboard is designed for both technical and non-technical users.

        - **Live Forecast** explains the next-hour electricity-load prediction.
        - **Model Routing** shows which model served the prediction and why.
        - **Business Impact** shows model accuracy and specialist improvements.
        - **Monitoring** shows production-style API and routing metrics.
        - **Prediction Logs** shows the latest logged predictions.
        - **Feature Inputs** explains what inputs were sent to the model.
        - **Raw Payloads** is for debugging and audit review.
        """
    )


# =============================================================================
# TOP SYSTEM STATUS
# =============================================================================
top_col1, top_col2, top_col3, top_col4, top_col5 = st.columns(5)

with top_col1:
    show_metric_card("API status", health.get("status", "unknown"))

with top_col2:
    show_metric_card("Model ready", str(health.get("model_ready", False)))

with top_col3:
    approved_count = len(router_status.get("summary", {}).get("approved_specialists", []))
    show_metric_card("Approved specialists", approved_count)

with top_col4:
    show_metric_card("Available series", series_options.get("count", 0))

with top_col5:
    show_metric_card(
        "Fallback model",
        router_status.get("fallback_model_key", "global"),
    )


# =============================================================================
# TABS
# =============================================================================
(
    tab_forecast,
    tab_routing,
    tab_business,
    tab_monitoring,
    tab_logs,
    tab_features,
    tab_raw,
) = st.tabs(
    [
        "Live Forecast",
        "Model Routing",
        "Business Impact",
        "Monitoring",
        "Prediction Logs",
        "Feature Inputs",
        "Raw Payloads",
    ]
)


# =============================================================================
# LIVE FORECAST TAB
# =============================================================================
with tab_forecast:
    st.subheader("Live Forecast and Operational Interpretation")

    payload = {
        "timestamp": timestamp_value,
        "country": country,
        "market": market,
        "zone": zone,
        "series_id": selected_series,
        "load_mw": load_mw,
        **user_inputs,
    }

    if run_prediction:
        result = api_post(api_url, "/predict", payload)

        if result:
            st.session_state["last_prediction_result"] = result
            st.session_state["last_prediction_payload"] = payload

    result = st.session_state.get("last_prediction_result")

    if not result:
        show_status_box(
            "info",
            "Select a grid series from the sidebar and click <b>Run forecast</b> to generate a prediction.",
        )
    else:
        risk_status, risk_explanation, risk_kind = risk_label(result)

        kpi_cols = st.columns(5)

        with kpi_cols[0]:
            show_metric_card("Current load", format_mw(result.get("current_load_mw")))

        with kpi_cols[1]:
            show_metric_card(
                "Next-hour forecast",
                format_mw(result.get("predicted_next_hour_load_mw")),
            )

        with kpi_cols[2]:
            show_metric_card(
                "Expected change",
                format_mw(result.get("expected_change_mw")),
                format_percent(result.get("expected_change_percent")),
            )

        with kpi_cols[3]:
            show_metric_card("Risk status", risk_status, risk_explanation)

        with kpi_cols[4]:
            show_metric_card(
                "Model used",
                model_scope_label(result),
                result.get("routing", {}).get("routing_reason"),
            )

        show_status_box(risk_kind, f"<b>{risk_status}</b> — {risk_explanation}")

        p10 = result.get("predicted_load_p10_mw")
        p90 = result.get("predicted_load_p90_mw")
        op_low = result.get("operational_lower_bound_mw")
        op_high = result.get("operational_upper_bound_mw")
        interval_contains_point = bool(result.get("prediction_interval_contains_point"))

        range_cols = st.columns(5)

        with range_cols[0]:
            show_metric_card(
                "P10 estimate",
                format_mw(p10),
                "Raw lower quantile estimate.",
            )

        with range_cols[1]:
            show_metric_card(
                "P90 estimate",
                format_mw(p90),
                "Raw upper quantile estimate.",
            )

        with range_cols[2]:
            show_metric_card(
                "Operational lower",
                format_mw(op_low),
                "Display-safe lower operating bound.",
            )

        with range_cols[3]:
            show_metric_card(
                "Operational upper",
                format_mw(op_high),
                "Display-safe upper operating bound.",
            )

        with range_cols[4]:
            show_metric_card(
                "Interval status",
                "Calibrated" if interval_contains_point else "Needs review",
                "Whether the point forecast lies inside raw P10–P90.",
            )

        if not interval_contains_point:
            show_status_box(
                "warning",
                (
                    "The point forecast is outside the raw P10–P90 quantile interval. "
                    "This can happen because the point model and quantile models are trained separately. "
                    "For presentation and operations, use the operational lower/upper bounds."
                ),
            )

        chart_col1, chart_col2 = st.columns([2, 1])

        with chart_col1:
            st.plotly_chart(make_forecast_chart(result), use_container_width=True)

        with chart_col2:
            st.plotly_chart(make_delta_chart(result), use_container_width=True)

        st.markdown("### Plain-language explanation")

        expected_change = float(result.get("expected_change_mw", 0.0))
        expected_change_percent = float(result.get("expected_change_percent", 0.0))

        direction = "increase" if expected_change > 0 else "decrease"

        st.write(
            f"The system expects electricity load to **{direction} by "
            f"{format_mw(abs(expected_change))}** "
            f"({format_percent(abs(expected_change_percent))}) in the next hour. "
            f"The request was served by **{model_scope_label(result)}**."
        )

        st.info(result.get("routing", {}).get("routing_reason", "No routing reason returned."))


# =============================================================================
# MODEL ROUTING TAB
# =============================================================================
with tab_routing:
    st.subheader("Registry-Driven Model Routing")

    st.markdown(
        """
        <div class="section-note">
        The router does not hardcode markets inside the application logic.
        It reads the active model registry and selects the best approved model
        based on routing priority, model status, artifact validation, and
        the specialist-vs-global comparison result.
        </div>
        """,
        unsafe_allow_html=True,
    )

    routing_cols = st.columns(4)

    with routing_cols[0]:
        show_metric_card("Router type", router_status.get("router_type", "unknown"))

    with routing_cols[1]:
        show_metric_card("Registry version", router_status.get("registry_version", "unknown"))

    with routing_cols[2]:
        show_metric_card("Deployment mode", router_status.get("deployment_mode", "unknown"))

    with routing_cols[3]:
        show_metric_card("Routing priority", " → ".join(router_status.get("routing_priority", [])))

    registry_df = make_registry_table(router_status)

    st.markdown("### Active model registry")
    st.dataframe(registry_df, use_container_width=True, hide_index=True)

    st.markdown("### Routing rules in simple words")
    st.write(
        """
        - If the request belongs to an **approved specialist market**, the specialist model is used.
        - If there is no approved specialist, the **global fallback model** is used.
        - A specialist is approved only when it passes the configured improvement threshold and artifact validation.
        """
    )

    if st.session_state.get("last_prediction_result"):
        last_result = st.session_state["last_prediction_result"]
        routing = last_result.get("routing", {})

        st.markdown("### Last prediction routing decision")
        routing_summary = pd.DataFrame(
            [
                {
                    "Selected model": routing.get("selected_model_key"),
                    "Model scope": routing.get("model_scope"),
                    "Matched field": routing.get("matched_field"),
                    "Fallback used": routing.get("fallback_used"),
                    "Reason": routing.get("routing_reason"),
                }
            ]
        )

        st.dataframe(routing_summary, use_container_width=True, hide_index=True)


# =============================================================================
# BUSINESS IMPACT TAB
# =============================================================================
with tab_business:
    st.subheader("Business Impact and Model Quality")

    st.write(
        """
        This section explains whether the system is useful from an operations perspective.
        The global model provides broad coverage, while approved specialists improve
        accuracy for selected markets.
        """
    )

    business_df = make_business_impact_table(local_registry)

    if not business_df.empty:
        st.dataframe(
            business_df,
            use_container_width=True,
            hide_index=True,
        )

        st.plotly_chart(
            make_improvement_chart(local_registry),
            use_container_width=True,
        )

        global_row = business_df[business_df["Model"] == "global"]

        if not global_row.empty:
            global_wmape = global_row.iloc[0].get("WMAPE (%)")
            global_mae = global_row.iloc[0].get("MAE (MW)")
            global_rmse = global_row.iloc[0].get("RMSE (MW)")

            impact_cols = st.columns(3)

            with impact_cols[0]:
                show_metric_card("Global WMAPE", format_percent(global_wmape))

            with impact_cols[1]:
                show_metric_card("Global MAE", format_mw(global_mae))

            with impact_cols[2]:
                show_metric_card("Global RMSE", format_mw(global_rmse))

        st.markdown("### Business interpretation")
        st.write(
            """
            - **WMAPE** shows volume-weighted forecasting error. Lower is better.
            - **MAE** shows the average absolute error in MW.
            - **RMSE** penalizes larger mistakes more strongly.
            - **Specialist improvement** shows how much a regional model improves over the global model on the same region.
            """
        )
    else:
        st.warning(
            "Local model registry was not found. Run `python scripts/build_model_registry.py` "
            "and make sure `artifacts/model_registry.json` exists."
        )


# =============================================================================
# MONITORING TAB
# =============================================================================
with tab_monitoring:
    st.subheader("Production Monitoring Snapshot")

    metrics_text = get_text(api_url, "/metrics")
    metrics = parse_prometheus_metrics(metrics_text)

    request_total = metrics.get("gridguard_prediction_requests_total", 0.0)
    error_total = metrics.get("gridguard_prediction_errors_total", 0.0)
    latency_count = metrics.get("gridguard_prediction_latency_seconds_count", 0.0)
    latency_sum = metrics.get("gridguard_prediction_latency_seconds_sum", 0.0)

    avg_latency = latency_sum / latency_count if latency_count else 0.0
    error_rate = error_total / request_total * 100 if request_total else 0.0

    monitor_cols = st.columns(5)

    with monitor_cols[0]:
        show_metric_card("Prediction requests", int(request_total))

    with monitor_cols[1]:
        show_metric_card("Prediction errors", int(error_total))

    with monitor_cols[2]:
        show_metric_card("Error rate", f"{error_rate:.2f}%")

    with monitor_cols[3]:
        show_metric_card("Average latency", f"{avg_latency:.3f} s")

    with monitor_cols[4]:
        show_metric_card(
            "Latest interval width",
            format_mw(metrics.get("gridguard_prediction_interval_width_mw", 0.0)),
        )

    routing_rows = []

    for metric_name, value in metrics.items():
        if metric_name.startswith("gridguard_model_routing_predictions_total"):
            labels = parse_prometheus_labels(metric_name)

            routing_rows.append(
                {
                    "model_scope": labels.get("model_scope"),
                    "model_key": labels.get("model_key"),
                    "fallback_used": labels.get("fallback_used"),
                    "count": int(value),
                }
            )

    st.markdown("### Routing counters")

    if routing_rows:
        routing_df = pd.DataFrame(routing_rows)
        st.dataframe(routing_df, use_container_width=True, hide_index=True)

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=routing_df["model_key"],
                y=routing_df["count"],
                text=routing_df["count"],
                textposition="outside",
            )
        )

        fig.update_layout(
            title="Prediction Count by Routed Model",
            xaxis_title="Model",
            yaxis_title="Count",
            height=360,
            margin=dict(l=20, r=20, t=60, b=20),
            showlegend=False,
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No routing counters found yet. Run a few predictions first.")

    with st.expander("Raw Prometheus metrics"):
        st.code(metrics_text)


# =============================================================================
# PREDICTION LOGS TAB
# =============================================================================
with tab_logs:
    st.subheader("Latest Prediction Logs")

    logs = api_get(api_url, "/logs/latest?limit=20")

    if logs and logs.get("logs"):
        log_rows = []

        for item in logs["logs"]:
            response = item.get("response", {})
            request = item.get("request", {})
            routing = response.get("routing", {})

            log_rows.append(
                {
                    "created_at_utc": item.get("created_at_utc"),
                    "series_id": request.get("series_id"),
                    "market": request.get("market"),
                    "predicted_load_mw": response.get("predicted_next_hour_load_mw"),
                    "expected_change_mw": response.get("expected_change_mw"),
                    "expected_change_%": response.get("expected_change_percent"),
                    "is_anomaly": response.get("is_anomaly"),
                    "model_key": routing.get("selected_model_key"),
                    "fallback_used": routing.get("fallback_used"),
                    "latency_seconds": item.get("latency_seconds"),
                }
            )

        st.dataframe(
            pd.DataFrame(log_rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "No logs returned. If MongoDB Atlas is not configured, this endpoint may be empty."
        )


# =============================================================================
# FEATURE INPUTS TAB
# =============================================================================
with tab_features:
    st.subheader("Feature Inputs Sent to the Model")

    last_payload = st.session_state.get("last_prediction_payload", {})

    if not last_payload:
        st.info("Run a forecast first to see feature input analysis.")
    else:
        st.write(
            """
            The model uses current load, lagged load values, rolling statistics,
            calendar features derived from timestamp, and routing metadata such as
            country, market, zone, and series ID.
            """
        )

        feature_groups = {
            "Identity and routing": [
                "country",
                "market",
                "zone",
                "series_id",
                "timestamp",
            ],
            "Current observation": [
                "load_mw",
            ],
            "Lag features": [
                "lag_1h",
                "lag_2h",
                "lag_3h",
                "lag_6h",
                "lag_12h",
                "lag_24h",
                "lag_48h",
                "lag_72h",
                "lag_168h",
            ],
            "Rolling statistics": [
                key for key in last_payload.keys() if key.startswith("rolling_")
            ],
        }

        for group_name, keys in feature_groups.items():
            rows = []

            for key in keys:
                if key in last_payload:
                    rows.append({"feature": key, "value": last_payload[key]})

            if rows:
                st.markdown(f"### {group_name}")
                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True,
                )

        st.markdown("### Why these features matter")
        st.write(
            """
            - **Current load** gives the strongest short-term signal.
            - **Lag features** tell the model how demand behaved in the recent past.
            - **Rolling statistics** summarize local volatility and operating range.
            - **Market and series fields** allow the router and model to account for regional behavior.
            """
        )


# =============================================================================
# RAW PAYLOADS TAB
# =============================================================================
with tab_raw:
    st.subheader("Raw Request, Response, and Router Status")

    st.markdown("### Last request payload")
    st.json(st.session_state.get("last_prediction_payload", {}))

    st.markdown("### Last prediction response")
    st.json(st.session_state.get("last_prediction_result", {}))

    st.markdown("### Public router status")
    st.json(router_status)

    with st.expander("Local full model registry"):
        if local_registry:
            st.json(local_registry)
        else:
            st.info("Local registry file not found.")
