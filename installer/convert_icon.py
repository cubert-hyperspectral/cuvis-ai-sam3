"""Create installer/app icon ICO for SAM3 server builds."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

SIZES = [16, 32, 48, 64, 128, 256]


def _resolve_source_png() -> Path:
    root = Path(__file__).resolve().parent.parent
    candidates = [
        root / "assets" / "sam3_icon.png",
        root / "assets" / "model_diagram.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No icon source image found. Expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def main() -> None:
    src = _resolve_source_png()
    dst = Path(__file__).resolve().parent / "app_icon.ico"

    img = Image.open(src)
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    icons = [img.resize((size, size), Image.Resampling.LANCZOS) for size in SIZES]
    icons[0].save(dst, format="ICO", sizes=[(size, size) for size in SIZES], append_images=icons[1:])
    print(f"Created {dst} ({dst.stat().st_size:,} bytes) from {src.name}")


if __name__ == "__main__":
    main()

