from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
except Exception:
    lgb = None
    LGBMRegressor = None


TARGET_COLUMN = "target_load_mw_next_hour"
TIMESTAMP_COLUMN = "timestamp"
CATEGORICAL_COLUMNS = ["country", "market", "zone", "series_id"]

# Context-aware anomaly features.
# These are better than feeding all raw time/load features into IsolationForest.
ANOMALY_FEATURES = [
    "load_change_1h",
    "load_change_24h",
    "load_change_168h",
    "load_ratio_to_24h_mean",
    "load_ratio_to_168h_mean",
    "rolling_std_3h",
    "rolling_std_6h",
    "rolling_std_24h",
    "rolling_std_168h",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
]


def _json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))

    wmape = (
        np.sum(np.abs(y_true - y_pred))
        / max(np.sum(np.abs(y_true)), 1e-9)
        * 100
    )

    smape = (
        np.mean(
            2
            * np.abs(y_pred - y_true)
            / np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-9)
        )
        * 100
    )

    r2 = r2_score(y_true, y_pred)

    return {
        "mae_mw": float(mae),
        "rmse_mw": float(rmse),
        "wmape_percent": float(wmape),
        "smape_percent": float(smape),
        "r2": float(r2),
    }


def _lgb_wmape_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[str, float, bool]:
    wmape = (
        np.sum(np.abs(y_true - y_pred))
        / max(np.sum(np.abs(y_true)), 1e-9)
        * 100
    )
    return "wmape_percent", float(wmape), False


def _prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    excluded = {TIMESTAMP_COLUMN, TARGET_COLUMN}

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded
    ]

    x = df[feature_columns].copy()
    y = df[TARGET_COLUMN].copy()

    for column in CATEGORICAL_COLUMNS:
        if column in x.columns:
            x[column] = x[column].astype("category")

    return x, y, feature_columns


def _get_category_levels(
    x: pd.DataFrame,
    categorical_features: list[str],
) -> dict[str, list[str]]:
    levels: dict[str, list[str]] = {}

    for column in categorical_features:
        levels[column] = sorted(
            x[column].dropna().astype(str).unique().tolist()
        )

    return levels


def _get_production_model_params() -> dict[str, Any]:
    if LGBMRegressor is None or lgb is None:
        raise ImportError(
            "LightGBM is required. Install it using: pip install lightgbm"
        )

    return {
        "boosting_type": "gbdt",
        "objective": "regression",
        "n_estimators": 6000,
        "learning_rate": 0.025,
        "num_leaves": 64,
        "max_depth": -1,
        "min_child_samples": 40,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.05,
        "reg_lambda": 0.50,
        "random_state": 42,
        "n_jobs": -1,
        "force_col_wise": True,
        "verbosity": -1,
    }


def _fit_production_candidate(
    params: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    categorical_features: list[str],
) -> tuple[Any, dict[str, float], int]:
    model_name = "lgbm_l2_balanced"
    print(f"Training production candidate model: {model_name}")

    model = LGBMRegressor(**params)

    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric=_lgb_wmape_metric,
        categorical_feature=categorical_features,
        callbacks=[
            lgb.early_stopping(stopping_rounds=250, verbose=False),
            lgb.log_evaluation(period=250),
        ],
    )

    val_prediction = model.predict(x_val)
    val_metrics = _metrics(y_val.to_numpy(), val_prediction)

    best_iteration = getattr(model, "best_iteration_", None)

    if best_iteration is None or best_iteration <= 0:
        best_iteration = getattr(model, "n_estimators_", params["n_estimators"])

    print(
        f"{model_name} | best_iteration={best_iteration} | "
        f"validation WMAPE={val_metrics['wmape_percent']:.4f}% | "
        f"validation MAE={val_metrics['mae_mw']:.4f} MW"
    )

    return model, val_metrics, int(best_iteration)


def _evaluate_baselines(
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, dict[str, float]]:
    baselines: dict[str, np.ndarray] = {
        "last_hour_baseline": x_test["load_mw"].to_numpy(),
        "same_hour_yesterday_baseline": x_test["lag_24h"].to_numpy(),
        "same_hour_last_week_baseline": x_test["lag_168h"].to_numpy(),
        "rolling_24h_average_baseline": x_test["rolling_mean_24h"].to_numpy(),
        "rolling_168h_average_baseline": x_test["rolling_mean_168h"].to_numpy(),
    }

    results: dict[str, dict[str, float]] = {}

    for name, prediction in baselines.items():
        results[name] = _metrics(y_test.to_numpy(), prediction)

    return results


def _per_series_metrics(
    test_df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> list[dict[str, Any]]:
    temp = test_df[
        [TIMESTAMP_COLUMN, "country", "market", "zone", "series_id"]
    ].copy()

    temp["actual"] = y_true.to_numpy()
    temp["prediction"] = y_pred

    rows: list[dict[str, Any]] = []

    for series_id, group in temp.groupby("series_id", observed=True):
        metrics = _metrics(
            group["actual"].to_numpy(),
            group["prediction"].to_numpy(),
        )

        rows.append(
            {
                "series_id": str(series_id),
                "country": str(group["country"].iloc[0]),
                "market": str(group["market"].iloc[0]),
                "zone": str(group["zone"].iloc[0]),
                "rows": int(len(group)),
                **metrics,
            }
        )

    return sorted(rows, key=lambda row: row["wmape_percent"], reverse=True)


def _per_market_metrics(
    test_df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> list[dict[str, Any]]:
    temp = test_df[[TIMESTAMP_COLUMN, "market"]].copy()
    temp["actual"] = y_true.to_numpy()
    temp["prediction"] = y_pred

    rows: list[dict[str, Any]] = []

    for market, group in temp.groupby("market", observed=True):
        metrics = _metrics(
            group["actual"].to_numpy(),
            group["prediction"].to_numpy(),
        )

        rows.append(
            {
                "market": str(market),
                "rows": int(len(group)),
                **metrics,
            }
        )

    return sorted(rows, key=lambda row: row["wmape_percent"], reverse=True)


def _per_month_metrics(
    test_df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> list[dict[str, Any]]:
    temp = test_df[[TIMESTAMP_COLUMN]].copy()
    temp["actual"] = y_true.to_numpy()
    temp["prediction"] = y_pred
    temp["month"] = temp[TIMESTAMP_COLUMN].dt.to_period("M").astype(str)

    rows: list[dict[str, Any]] = []

    for month, group in temp.groupby("month"):
        metrics = _metrics(
            group["actual"].to_numpy(),
            group["prediction"].to_numpy(),
        )

        rows.append(
            {
                "month": str(month),
                "rows": int(len(group)),
                **metrics,
            }
        )

    return rows


def _feature_importance_gain(
    model: Any,
    feature_columns: list[str],
) -> list[dict[str, Any]]:
    if not hasattr(model, "booster_"):
        return []

    gain_importance = model.booster_.feature_importance(importance_type="gain")
    split_importance = model.booster_.feature_importance(importance_type="split")

    total_gain = float(np.sum(gain_importance))

    rows: list[dict[str, Any]] = []

    for feature, gain, split in zip(feature_columns, gain_importance, split_importance):
        gain_share = float(gain / total_gain * 100) if total_gain > 0 else 0.0

        rows.append(
            {
                "feature": feature,
                "gain_importance": float(gain),
                "gain_share_percent": gain_share,
                "split_importance": int(split),
            }
        )

    return sorted(rows, key=lambda row: row["gain_importance"], reverse=True)


def _train_quantile_models(
    base_params: dict[str, Any],
    x_train_val: pd.DataFrame,
    y_train_val: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    categorical_features: list[str],
    artifact_dir: Path,
) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)

    quantile_results: dict[str, Any] = {}

    for label, alpha in {"p10": 0.10, "p90": 0.90}.items():
        print(f"Training quantile model: {label} alpha={alpha}")

        params = dict(base_params)
        params["objective"] = "quantile"
        params["alpha"] = alpha
        params["metric"] = "quantile"

        # Quantile models are used for operational risk intervals.
        # 3000 trees is a good cost/benefit compromise.
        params["n_estimators"] = min(int(params.get("n_estimators", 3000)), 3000)

        model = LGBMRegressor(**params)

        model.fit(
            x_train_val,
            y_train_val,
            categorical_feature=categorical_features,
        )

        prediction = model.predict(x_test)

        if label == "p10":
            coverage = float(np.mean(y_test.to_numpy() >= prediction))
        else:
            coverage = float(np.mean(y_test.to_numpy() <= prediction))

        model_path = artifact_dir / f"model_{label}.joblib"
        joblib.dump(model, model_path)

        quantile_results[label] = {
            "alpha": alpha,
            "model_path": str(model_path),
            "mean_prediction": float(np.mean(prediction)),
            "coverage_direction": coverage,
        }

    return quantile_results


def train_models(
    processed_path: str | Path,
    artifact_dir: str | Path,
    reports_dir: str | Path = "reports",
    max_rows: int | None = None,
) -> dict[str, Any]:
    processed_path = Path(processed_path)
    artifact_dir = Path(artifact_dir)
    reports_dir = Path(reports_dir)

    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {processed_path}. "
            "Run `python scripts/make_dataset.py` first."
        )

    df = pd.read_parquet(processed_path)

    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN], errors="coerce")
    df = df.dropna(subset=[TIMESTAMP_COLUMN, TARGET_COLUMN])
    df = df.sort_values(TIMESTAMP_COLUMN)

    if max_rows is not None:
        df = df.tail(max_rows)

    train_df = df[df[TIMESTAMP_COLUMN] < "2024-01-01"].copy()

    val_df = df[
        (df[TIMESTAMP_COLUMN] >= "2024-01-01")
        & (df[TIMESTAMP_COLUMN] < "2025-01-01")
    ].copy()

    test_df = df[df[TIMESTAMP_COLUMN] >= "2025-01-01"].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            "Chronological split failed. Need train data before 2024, "
            "validation data in 2024, and test data in 2025."
        )

    x_train, y_train, feature_columns = _prepare_features(train_df)
    x_val, y_val, _ = _prepare_features(val_df)
    x_test, y_test, _ = _prepare_features(test_df)

    categorical_features = [
        column for column in CATEGORICAL_COLUMNS if column in feature_columns
    ]

    production_params = _get_production_model_params()

    _, val_metrics, best_iteration = _fit_production_candidate(
        params=production_params,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        categorical_features=categorical_features,
    )

    model_name = "lgbm_l2_balanced"

    model_selection_results = {
        model_name: {
            **val_metrics,
            "best_iteration": best_iteration,
            "objective": production_params.get("objective"),
            "learning_rate": production_params.get("learning_rate"),
            "num_leaves": production_params.get("num_leaves"),
            "min_child_samples": production_params.get("min_child_samples"),
            "subsample": production_params.get("subsample"),
            "colsample_bytree": production_params.get("colsample_bytree"),
            "reg_alpha": production_params.get("reg_alpha"),
            "reg_lambda": production_params.get("reg_lambda"),
            "production_candidate_only": True,
        }
    }

    print(f"Selected production model: {model_name}")
    print(f"Validation WMAPE: {val_metrics['wmape_percent']:.4f}%")
    print(f"Best iteration: {best_iteration}")

    train_val_df = pd.concat([train_df, val_df], ignore_index=True)

    x_train_val, y_train_val, _ = _prepare_features(train_val_df)

    category_levels = _get_category_levels(
        x_train_val,
        categorical_features,
    )

    final_params = dict(production_params)
    final_params["n_estimators"] = max(best_iteration, 50)

    final_model = LGBMRegressor(**final_params)

    final_model.fit(
        x_train_val,
        y_train_val,
        categorical_feature=categorical_features,
    )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    test_prediction = final_model.predict(x_test)

    model_metrics = _metrics(
        y_test.to_numpy(),
        test_prediction,
    )

    baseline_metrics = _evaluate_baselines(x_test, y_test)

    best_baseline_name = min(
        baseline_metrics,
        key=lambda name: baseline_metrics[name]["wmape_percent"],
    )

    best_baseline = baseline_metrics[best_baseline_name]

    improvement_vs_best_baseline = {
        "best_baseline_name": best_baseline_name,
        "wmape_reduction_percent": float(
            (best_baseline["wmape_percent"] - model_metrics["wmape_percent"])
            / best_baseline["wmape_percent"]
            * 100
        ),
        "mae_reduction_percent": float(
            (best_baseline["mae_mw"] - model_metrics["mae_mw"])
            / best_baseline["mae_mw"]
            * 100
        ),
        "rmse_reduction_percent": float(
            (best_baseline["rmse_mw"] - model_metrics["rmse_mw"])
            / best_baseline["rmse_mw"]
            * 100
        ),
    }

    anomaly_features = [
        column
        for column in ANOMALY_FEATURES
        if column in feature_columns
    ]

    anomaly_model = IsolationForest(
        n_estimators=400,
        contamination=0.01,
        random_state=42,
        n_jobs=-1,
    )

    anomaly_model.fit(x_train_val[anomaly_features])

    anomaly_pred = anomaly_model.predict(x_test[anomaly_features])
    anomaly_scores = anomaly_model.decision_function(x_test[anomaly_features])

    anomaly_metrics = {
        "test_anomaly_rate": float((anomaly_pred == -1).mean()),
        "decision_function_mean": float(np.mean(anomaly_scores)),
        "decision_function_std": float(np.std(anomaly_scores)),
        "anomaly_feature_count": int(len(anomaly_features)),
        "anomaly_feature_strategy": "context_aware_load_change_and_variability_features",
    }

    per_series_metrics = _per_series_metrics(
        test_df=test_df,
        y_true=y_test,
        y_pred=test_prediction,
    )

    per_market_metrics = _per_market_metrics(
        test_df=test_df,
        y_true=y_test,
        y_pred=test_prediction,
    )

    per_month_metrics = _per_month_metrics(
        test_df=test_df,
        y_true=y_test,
        y_pred=test_prediction,
    )

    model_metrics["micro_wmape_percent"] = model_metrics["wmape_percent"]

    model_metrics["macro_market_wmape_percent"] = float(
        np.mean([row["wmape_percent"] for row in per_market_metrics])
    )

    model_metrics["macro_series_wmape_percent"] = float(
        np.mean([row["wmape_percent"] for row in per_series_metrics])
    )

    model_metrics["worst_series_wmape_percent"] = float(
        per_series_metrics[0]["wmape_percent"]
    )

    feature_importance = _feature_importance_gain(
        model=final_model,
        feature_columns=feature_columns,
    )

    quantile_results = _train_quantile_models(
        base_params=final_params,
        x_train_val=x_train_val,
        y_train_val=y_train_val,
        x_test=x_test,
        y_test=y_test,
        categorical_features=categorical_features,
        artifact_dir=artifact_dir,
    )

    created_at = datetime.now(timezone.utc).isoformat()

    metadata = {
        "project": "GridGuard AI",
        "dataset": "Worldwide Electricity Load Dataset / GloElecLoad",
        "model_type": f"{model_name} + IsolationForest",
        "forecasting_model": model_name,
        "anomaly_model": "IsolationForest",
        "target": TARGET_COLUMN,
        "created_at_utc": created_at,
        "split_strategy": "train_before_2024_validate_2024_test_2025",
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "train_validation_rows": int(len(train_val_df)),
        "test_rows": int(len(test_df)),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "categorical_features": categorical_features,
        "numeric_anomaly_features": anomaly_features,
        "series_count": int(df["series_id"].nunique()),
        "data_start": str(df[TIMESTAMP_COLUMN].min()),
        "data_end": str(df[TIMESTAMP_COLUMN].max()),
        "test_start": str(test_df[TIMESTAMP_COLUMN].min()),
        "test_end": str(test_df[TIMESTAMP_COLUMN].max()),
        "best_iteration": int(best_iteration),
        "production_training_mode": "single_best_candidate",
        "quantile_forecasting_enabled": True,
        "prediction_interval": "P10-P90",
    }

    report = {
        "metadata": metadata,
        "model_selection_results": model_selection_results,
        "baseline_metrics": baseline_metrics,
        "model_metrics": model_metrics,
        "improvement_vs_best_baseline": improvement_vs_best_baseline,
        "anomaly_metrics": anomaly_metrics,
        "quantile_forecast_models": quantile_results,
        "top_25_feature_importance": feature_importance[:25],
        "worst_10_series_by_wmape": per_series_metrics[:10],
        "per_market_metrics": per_market_metrics,
        "per_month_metrics": per_month_metrics,
    }

    deployment_manifest = {
        "point_model": "model.joblib",
        "lower_quantile_model": "model_p10.joblib",
        "upper_quantile_model": "model_p90.joblib",
        "anomaly_model": "anomaly_model.joblib",
        "forecast_type": "point_forecast_with_p10_p90_prediction_interval",
        "recommended_dashboard_label": "Expected range: P10-P90",
        "production_training_mode": "single_best_candidate",
        "selected_model": model_name,
    }

    joblib.dump(final_model, artifact_dir / "model.joblib")
    joblib.dump(anomaly_model, artifact_dir / "anomaly_model.joblib")

    _json_dump(feature_columns, artifact_dir / "feature_columns.json")
    _json_dump(model_metrics, artifact_dir / "metrics.json")
    _json_dump(metadata, artifact_dir / "model_metadata.json")
    _json_dump(feature_importance, artifact_dir / "feature_importance.json")
    _json_dump(category_levels, artifact_dir / "category_levels.json")
    _json_dump(deployment_manifest, artifact_dir / "deployment_manifest.json")

    _json_dump(report, reports_dir / "training_report.json")
    _json_dump(per_series_metrics, reports_dir / "per_series_metrics.json")
    _json_dump(per_market_metrics, reports_dir / "per_market_metrics.json")
    _json_dump(per_month_metrics, reports_dir / "per_month_metrics.json")

    pd.DataFrame(per_series_metrics).to_csv(
        reports_dir / "per_series_metrics.csv",
        index=False,
    )

    pd.DataFrame(per_market_metrics).to_csv(
        reports_dir / "per_market_metrics.csv",
        index=False,
    )

    pd.DataFrame(per_month_metrics).to_csv(
        reports_dir / "per_month_metrics.csv",
        index=False,
    )

    print(json.dumps(report, indent=2, default=str))

    return report
