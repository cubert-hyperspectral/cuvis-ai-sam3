"""Download default SAM3 model weights for packaged server installs."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

DEFAULT_REPO_ID = "facebook/sam3"
DEFAULT_FILENAME = "sam3.pt"


def _default_target() -> Path:
    if getattr(__import__("sys"), "frozen", False):
        base = Path(__import__("sys").executable).resolve().parent
    else:
        base = Path(__file__).resolve().parent.parent
    return base / "models" / DEFAULT_FILENAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download SAM3 checkpoint weights.")
    parser.add_argument(
        "--path",
        type=Path,
        default=_default_target(),
        help="Target checkpoint path (default: <app>/models/sam3.pt)",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Hugging Face repository ID.",
    )
    parser.add_argument(
        "--filename",
        default=DEFAULT_FILENAME,
        help="Checkpoint filename in repository.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if already cached.",
    )
    return parser.parse_args()


def download_checkpoint(target_path: Path, repo_id: str, filename: str, force: bool) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            force_download=force,
            local_dir=str(target_path.parent),
            local_dir_use_symlinks=False,
        )
    )
    if downloaded.resolve() != target_path.resolve():
        shutil.copy2(downloaded, target_path)
    return target_path


def main() -> int:
    args = parse_args()
    try:
        path = download_checkpoint(
            target_path=args.path,
            repo_id=args.repo_id,
            filename=args.filename,
            force=args.force,
        )
    except Exception as exc:  # pragma: no cover - exercised in installer integration, not unit
        print(f"ERROR: weight download failed: {exc}")
        return 1

    print(f"Downloaded checkpoint to: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
