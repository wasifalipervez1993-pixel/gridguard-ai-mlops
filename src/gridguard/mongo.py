from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

try:
    from bson import ObjectId
    from pymongo import MongoClient
    from pymongo.collection import Collection
    from pymongo.errors import PyMongoError
except Exception:  # pragma: no cover
    MongoClient = None
    Collection = Any
    ObjectId = None
    PyMongoError = Exception


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if ObjectId is not None and isinstance(value, ObjectId):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_json_safe(item) for item in value]

    return value


class PredictionLogger:
    """
    MongoDB Atlas logger for prediction and feedback records.

    If MONGODB_URI is missing, logging is disabled safely.
    This allows the app to run locally, in tests, and in Docker without MongoDB.
    """

    def __init__(self) -> None:
        self.mongodb_uri = os.getenv("MONGODB_URI", "").strip()
        self.database_name = (
            os.getenv("MONGODB_DB")
            or os.getenv("MONGODB_DB_NAME")
            or "gridguard"
        ).strip()
        self.prediction_collection_name = os.getenv(
            "MONGODB_PREDICTION_COLLECTION",
            "prediction_logs",
        ).strip()
        self.feedback_collection_name = os.getenv(
            "MONGODB_FEEDBACK_COLLECTION",
            "feedback_logs",
        ).strip()

        self.enabled = bool(self.mongodb_uri)
        self.client: Any | None = None
        self.prediction_collection: Collection | None = None
        self.feedback_collection: Collection | None = None

        if self.enabled:
            self._connect()

    def _connect(self) -> None:
        if MongoClient is None:
            self.enabled = False
            return

        try:
            self.client = MongoClient(
                self.mongodb_uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=5000,
            )

            # Force connection check early.
            self.client.admin.command("ping")

            database = self.client[self.database_name]
            self.prediction_collection = database[self.prediction_collection_name]
            self.feedback_collection = database[self.feedback_collection_name]

            self.prediction_collection.create_index("created_at_utc")
            self.prediction_collection.create_index("request.series_id")
            self.prediction_collection.create_index("request.market")
            self.prediction_collection.create_index("response.served_by_model_key")

            self.feedback_collection.create_index("created_at_utc")
            self.feedback_collection.create_index("prediction_id")

        except PyMongoError:
            self.enabled = False
            self.client = None
            self.prediction_collection = None
            self.feedback_collection = None

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "database": self.database_name if self.enabled else None,
            "prediction_collection": (
                self.prediction_collection_name if self.enabled else None
            ),
            "feedback_collection": self.feedback_collection_name if self.enabled else None,
        }

    def log_prediction(self, record: dict[str, Any]) -> str | None:
        if not self.enabled or self.prediction_collection is None:
            return None

        payload = {
            **record,
            "created_at_utc": record.get("created_at_utc", _utc_now()),
            "logger": "mongodb_atlas",
        }

        try:
            result = self.prediction_collection.insert_one(payload)
            return str(result.inserted_id)
        except PyMongoError:
            return None

    def log_feedback(self, record: dict[str, Any]) -> str | None:
        if not self.enabled or self.feedback_collection is None:
            return None

        payload = {
            **record,
            "created_at_utc": record.get("created_at_utc", _utc_now()),
            "logger": "mongodb_atlas",
        }

        try:
            result = self.feedback_collection.insert_one(payload)
            return str(result.inserted_id)
        except PyMongoError:
            return None

    def latest_predictions(self, limit: int = 25) -> list[dict[str, Any]]:
        if not self.enabled or self.prediction_collection is None:
            return []

        safe_limit = min(max(int(limit), 1), 100)

        try:
            cursor = (
                self.prediction_collection.find({})
                .sort("created_at_utc", -1)
                .limit(safe_limit)
            )

            return [_json_safe(document) for document in cursor]

        except PyMongoError:
            return []
    