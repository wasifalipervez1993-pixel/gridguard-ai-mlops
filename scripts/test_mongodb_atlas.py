from __future__ import annotations

from gridguard.mongo import PredictionLogger


def main() -> None:
    logger = PredictionLogger()

    print("MongoDB logger status:")
    print(logger.status())

    if not logger.enabled:
        raise SystemExit(
            "MongoDB logging is disabled. Check MONGODB_URI in your .env file."
        )

    inserted_id = logger.log_prediction(
        {
            "request": {
                "series_id": "mongodb_test",
                "market": "test",
            },
            "response": {
                "predicted_next_hour_load_mw": 123.45,
                "served_by_model_key": "test_model",
            },
            "latency_seconds": 0.001,
        }
    )

    print(f"Inserted test prediction id: {inserted_id}")

    logs = logger.latest_predictions(limit=3)
    print(f"Latest logs returned: {len(logs)}")

    if logs:
        print(logs[0])


if __name__ == "__main__":
    main()
