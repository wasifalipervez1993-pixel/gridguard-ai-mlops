from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


class ModelNotReadyError(RuntimeError):
    pass


class GridGuardPredictor:
    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)

        self.model_path = self.model_dir / "model.joblib"
        self.p10_model_path = self.model_dir / "model_p10.joblib"
        self.p90_model_path = self.model_dir / "model_p90.joblib"
        self.anomaly_model_path = self.model_dir / "anomaly_model.joblib"
        self.feature_columns_path = self.model_dir / "feature_columns.json"
        self.metadata_path = self.model_dir / "model_metadata.json"
        self.category_levels_path = self.model_dir / "category_levels.json"

        self.model = None
        self.p10_model = None
        self.p90_model = None
        self.anomaly_model = None

        self.feature_columns: list[str] = []
        self.metadata: dict[str, Any] = {}
        self.category_levels: dict[str, list[str]] = {}

        self._load()

    @property
    def is_ready(self) -> bool:
        return (
            self.model is not None
            and self.anomaly_model is not None
            and len(self.feature_columns) > 0
        )

    def _load_json(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default

        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _load(self) -> None:
        if not self.model_path.exists():
            return

        if not self.anomaly_model_path.exists():
            return

        if not self.feature_columns_path.exists():
            return

        self.model = joblib.load(self.model_path)
        self.anomaly_model = joblib.load(self.anomaly_model_path)

        if self.p10_model_path.exists():
            self.p10_model = joblib.load(self.p10_model_path)

        if self.p90_model_path.exists():
            self.p90_model = joblib.load(self.p90_model_path)

        self.feature_columns = self._load_json(self.feature_columns_path, [])
        self.metadata = self._load_json(self.metadata_path, {})
        self.category_levels = self._load_json(self.category_levels_path, {})

    def get_available_series(self) -> list[str]:
        return self.category_levels.get("series_id", [])

    def _parse_timestamp(self, payload: dict[str, Any]) -> datetime:
        raw_timestamp = payload.get("timestamp")

        if not raw_timestamp:
            return datetime.now(timezone.utc).replace(tzinfo=None)

        try:
            return datetime.fromisoformat(
                str(raw_timestamp).replace("Z", "+00:00")
            ).replace(tzinfo=None)
        except Exception:
            return datetime.now(timezone.utc).replace(tzinfo=None)

    @staticmethod
    def _safe_divide(numerator: float, denominator: float) -> float:
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def _prepare_features(
        self,
        payload: dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        timestamp = self._parse_timestamp(payload)

        load_mw = float(payload.get("load_mw", 0.0))

        country = str(payload.get("country", "USA"))
        market = str(payload.get("market", "NYISO"))
        zone = str(payload.get("zone", "N.Y.C."))
        series_id = str(payload.get("series_id", f"{country}/{market}/{zone}"))

        data: dict[str, Any] = {
            "load_mw": load_mw,
            "country": country,
            "market": market,
            "zone": zone,
            "series_id": series_id,
            "hour": timestamp.hour,
            "day_of_week": timestamp.weekday(),
            "month": timestamp.month,
            "day_of_year": timestamp.timetuple().tm_yday,
            "week_of_year": int(timestamp.isocalendar().week),
            "is_weekend": 1 if timestamp.weekday() >= 5 else 0,
        }

        data["hour_sin"] = float(np.sin(2 * np.pi * data["hour"] / 24))
        data["hour_cos"] = float(np.cos(2 * np.pi * data["hour"] / 24))

        data["day_of_week_sin"] = float(
            np.sin(2 * np.pi * data["day_of_week"] / 7)
        )
        data["day_of_week_cos"] = float(
            np.cos(2 * np.pi * data["day_of_week"] / 7)
        )

        data["month_sin"] = float(np.sin(2 * np.pi * data["month"] / 12))
        data["month_cos"] = float(np.cos(2 * np.pi * data["month"] / 12))

        for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
            data[f"lag_{lag}h"] = float(payload.get(f"lag_{lag}h", load_mw))

        for window in [3, 6, 12, 24, 48, 168]:
            data[f"rolling_mean_{window}h"] = float(
                payload.get(f"rolling_mean_{window}h", load_mw)
            )
            data[f"rolling_std_{window}h"] = float(
                payload.get(f"rolling_std_{window}h", 0.0)
            )
            data[f"rolling_min_{window}h"] = float(
                payload.get(f"rolling_min_{window}h", load_mw)
            )
            data[f"rolling_max_{window}h"] = float(
                payload.get(f"rolling_max_{window}h", load_mw)
            )

        data["load_change_1h"] = load_mw - data["lag_1h"]
        data["load_change_24h"] = load_mw - data["lag_24h"]
        data["load_change_168h"] = load_mw - data["lag_168h"]

        data["load_ratio_to_24h_mean"] = self._safe_divide(
            load_mw,
            data["rolling_mean_24h"],
        )

        data["load_ratio_to_168h_mean"] = self._safe_divide(
            load_mw,
            data["rolling_mean_168h"],
        )

        row = {}

        for column in self.feature_columns:
            row[column] = data.get(column, 0.0)

        feature_df = pd.DataFrame([row], columns=self.feature_columns)

        for column in ["country", "market", "zone", "series_id"]:
            if column in feature_df.columns:
                levels = self.category_levels.get(column)

                if levels:
                    feature_df[column] = pd.Categorical(
                        feature_df[column].astype(str),
                        categories=levels,
                    )
                else:
                    feature_df[column] = feature_df[column].astype("category")

        numeric_anomaly_features = self.metadata.get("numeric_anomaly_features", [])
        anomaly_df = feature_df[numeric_anomaly_features].copy()

        return feature_df, anomaly_df

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.is_ready:
            raise ModelNotReadyError(
                "Model is not ready. Run `python scripts/train_model.py` first."
            )

        features, anomaly_features = self._prepare_features(payload)

        predicted_load = float(self.model.predict(features)[0])

        prediction_p10 = None
        prediction_p90 = None

        if self.p10_model is not None:
            prediction_p10 = float(self.p10_model.predict(features)[0])

        if self.p90_model is not None:
            prediction_p90 = float(self.p90_model.predict(features)[0])

        if (
            prediction_p10 is not None
            and prediction_p90 is not None
            and prediction_p10 > prediction_p90
        ):
            prediction_p10, prediction_p90 = prediction_p90, prediction_p10

        operational_lower_bound = None
        operational_upper_bound = None
        prediction_interval_contains_point = None

        if prediction_p10 is not None and prediction_p90 is not None:
            raw_lower = min(prediction_p10, prediction_p90)
            raw_upper = max(prediction_p10, prediction_p90)

            prediction_interval_contains_point = (
                raw_lower <= predicted_load <= raw_upper
            )

            operational_lower_bound = min(raw_lower, predicted_load)
            operational_upper_bound = max(raw_upper, predicted_load)

        anomaly_raw = int(self.anomaly_model.predict(anomaly_features)[0])
        anomaly_score = float(
            self.anomaly_model.decision_function(anomaly_features)[0]
        )
        is_anomaly = anomaly_raw == -1

        current_load = float(features["load_mw"].iloc[0])
        delta_mw = predicted_load - current_load
        delta_percent = (delta_mw / current_load) * 100 if current_load != 0 else 0.0

        return {
            "predicted_next_hour_load_mw": predicted_load,
            "predicted_load_p10_mw": prediction_p10,
            "predicted_load_p90_mw": prediction_p90,
            "prediction_interval_available": (
                prediction_p10 is not None and prediction_p90 is not None
            ),
            "operational_lower_bound_mw": operational_lower_bound,
            "operational_upper_bound_mw": operational_upper_bound,
            "prediction_interval_contains_point": prediction_interval_contains_point,
            "current_load_mw": current_load,
            "expected_change_mw": delta_mw,
            "expected_change_percent": delta_percent,
            "is_anomaly": is_anomaly,
            "anomaly_score": anomaly_score,
            "model_version": self.metadata.get("created_at_utc", "unknown"),
            "model_type": self.metadata.get("model_type", "unknown"),
            "dataset": self.metadata.get("dataset", "unknown"),
            "target": self.metadata.get("target", "unknown"),
            "features_used": features.iloc[0].to_dict(),
        }
