from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pymongo import MongoClient
from pymongo.errors import PyMongoError

from gridguard.config import settings


class PredictionLogger:
    def __init__(self) -> None:
        self.client = None
        self.db = None
        self.local_log_dir = Path("logs")
        self.local_log_dir.mkdir(exist_ok=True)

        if settings.mongodb_uri:
            try:
                self.client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=3000)
                self.client.admin.command("ping")
                self.db = self.client[settings.mongodb_db]
                print("Connected to MongoDB Atlas")
            except PyMongoError as exc:
                print(f"MongoDB connection failed; using local JSONL logs. Reason: {exc}")
                self.client = None
                self.db = None

    def _with_timestamp(self, record: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(record)
        enriched["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        return enriched

    def _write_local(self, filename: str, record: dict[str, Any]) -> None:
        path = self.local_log_dir / filename
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def log_prediction(self, record: dict[str, Any]) -> None:
        record = self._with_timestamp(record)
        if self.db is not None:
            self.db[settings.mongodb_prediction_collection].insert_one(record)
        else:
            self._write_local("prediction_logs.jsonl", record)

    def log_feedback(self, record: dict[str, Any]) -> None:
        record = self._with_timestamp(record)
        if self.db is not None:
            self.db[settings.mongodb_feedback_collection].insert_one(record)
        else:
            self._write_local("feedback_logs.jsonl", record)

    def latest_predictions(self, limit: int = 50) -> list[dict[str, Any]]:
        if self.db is None:
            path = self.local_log_dir / "prediction_logs.jsonl"
            if not path.exists():
                return []
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            return rows[-limit:][::-1]

        cursor = (
            self.db[settings.mongodb_prediction_collection]
            .find({}, {"_id": 0})
            .sort("created_at_utc", -1)
            .limit(limit)
        )
        return list(cursor)
