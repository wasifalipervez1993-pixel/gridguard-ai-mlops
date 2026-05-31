from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "GridGuard AI")
    environment: str = os.getenv("ENVIRONMENT", "local")
    model_dir: Path = Path(os.getenv("MODEL_DIR", "artifacts"))
    data_raw_path: Path = Path(
        os.getenv("DATA_RAW_PATH", "data/raw/GloElecLoad.zip")
    )
    data_processed_path: Path = Path(
        os.getenv("DATA_PROCESSED_PATH", "data/processed/gloelecload_features.parquet")
    )
    mongodb_uri: str | None = os.getenv("MONGODB_URI") or None
    mongodb_db: str = os.getenv("MONGODB_DB", "gridguard")
    mongodb_prediction_collection: str = os.getenv(
        "MONGODB_PREDICTION_COLLECTION", "prediction_logs"
    )
    mongodb_feedback_collection: str = os.getenv(
        "MONGODB_FEEDBACK_COLLECTION", "feedback_logs"
    )


settings = Settings()
