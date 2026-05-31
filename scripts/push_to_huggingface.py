from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def push_artifacts(repo_id: str, artifact_dir: str = "artifacts") -> None:
    path = Path(artifact_dir)
    if not path.exists():
        raise FileNotFoundError(f"Artifact directory does not exist: {path}")

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(path),
        path_in_repo=".",
        commit_message="Upload GridGuard AI model artifacts",
    )
    print(f"Uploaded artifacts to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="Example: username/gridguard-ai-model")
    parser.add_argument("--artifact-dir", default="artifacts")
    args = parser.parse_args()
    push_artifacts(args.repo_id, args.artifact_dir)
