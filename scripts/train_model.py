from pathlib import Path
import sys
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

for path in [PROJECT_ROOT, SRC_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gridguard.config import settings  # noqa: E402
from gridguard.train import train_models  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GridGuard AI models.")

    parser.add_argument(
        "--processed-path",
        type=str,
        default=str(settings.data_processed_path),
        help="Path to the processed hourly feature dataset.",
    )

    parser.add_argument(
        "--artifact-dir",
        type=str,
        default=str(settings.model_dir),
        help="Directory where trained model artifacts will be saved.",
    )

    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports",
        help="Directory where training reports will be saved.",
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional number of most recent rows to use for quick training.",
    )

    args = parser.parse_args()

    train_models(
        processed_path=args.processed_path,
        artifact_dir=args.artifact_dir,
        reports_dir=args.reports_dir,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
