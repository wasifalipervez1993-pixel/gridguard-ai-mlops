from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

for path in [PROJECT_ROOT, SRC_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gridguard.config import settings  # noqa: E402
from gridguard.preprocessing import build_feature_dataset  # noqa: E402


def main() -> None:
    print(f"Raw dataset path: {settings.data_raw_path}")
    print(f"Processed output path: {settings.data_processed_path}")

    build_feature_dataset(
        raw_path=settings.data_raw_path,
        output_path=settings.data_processed_path,
    )


if __name__ == "__main__":
    main()
