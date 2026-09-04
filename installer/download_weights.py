"""Download the default SAM3 model weights for packaged server installs.

Provisions through cuvis-ai-core's weight registry (the ``cubert-gmbh/sam3``
mirror: public, commit-pinned, sha256-verified) and copies the checkpoint to the
install's ``models/`` directory. ``--repo-id`` / ``--filename`` override the
registry for a custom source.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cuvis_ai_core.data.model_weights import ModelDownloadError, ModelWeights

DEFAULT_MODEL = "sam3"
DEFAULT_FILENAME = "sam3.pt"


def _default_target() -> Path:
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).resolve().parent
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
        default=None,
        help="Hugging Face repository ID (default: the cubert-gmbh/sam3 mirror).",
    )
    parser.add_argument(
        "--filename",
        default=None,
        help="Checkpoint filename in the repository (default: sam3.pt).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if already cached.",
    )
    return parser.parse_args()


def download_checkpoint(
    target_path: Path, repo_id: str | None, filename: str | None, force: bool
) -> Path:
    """Provision the checkpoint into the shared cache and copy it to ``target_path``."""
    name = DEFAULT_MODEL if repo_id is None and filename is None else None
    return ModelWeights.download_model(
        name, repo_id=repo_id, filename=filename, out=target_path, force=force
    )


def main() -> int:
    args = parse_args()
    try:
        path = download_checkpoint(
            target_path=args.path,
            repo_id=args.repo_id,
            filename=args.filename,
            force=args.force,
        )
    except ModelDownloadError as exc:
        print(f"ERROR: weight download failed: {exc}")
        return 1

    print(f"Downloaded checkpoint to: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
