"""Render mask overlay video from SAM3 tracking results.

Reads a video and a ``tracking_results.json`` (produced by sub-task 2.2) and
renders coloured mask overlays with object IDs — no SAM3 inference required.

The overlay rendering logic (colour palette, alpha blending, contours, ID
labels) matches ``StreamingOverlayWriter`` in ``sam3_tracking_example.py``
exactly so that the standalone re-render looks identical to the built-in
overlay produced during tracking.

Usage
-----
Basic::

    uv run python examples/object_tracking/render_tracking_overlay.py \
        --video-path outputs/false_rgb.mp4 \
        --tracking-json outputs/tracking_test/tracking_results.json

Custom output and settings::

    uv run python examples/object_tracking/render_tracking_overlay.py \
        --video-path outputs/false_rgb.mp4 \
        --tracking-json outputs/tracking_test/tracking_results.json \
        --output-video-path outputs/overlay.mp4 \
        --mask-alpha 0.5 --no-contours
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Colour palette — identical to sam3_tracking_example.py
# ---------------------------------------------------------------------------

OBJECT_COLORS: list[tuple[int, int, int]] = [
    (255, 0, 0),      # red
    (0, 255, 0),      # green
    (0, 100, 255),    # blue
    (255, 255, 0),    # yellow
    (255, 0, 255),    # magenta
    (0, 255, 255),    # cyan
    (255, 128, 0),    # orange
    (128, 0, 255),    # purple
    (0, 255, 128),    # spring green
    (255, 128, 128),  # salmon
]


def get_object_color(obj_id: int) -> tuple[int, int, int]:
    """Return a deterministic RGB colour for *obj_id*."""
    return OBJECT_COLORS[obj_id % len(OBJECT_COLORS)]


# ---------------------------------------------------------------------------
# Overlay rendering — mirrors StreamingOverlayWriter.write_frame()
# ---------------------------------------------------------------------------

def render_overlay_frame(
    frame_bgr: np.ndarray,
    masks_by_id: dict[int, np.ndarray],
    *,
    alpha: float = 0.4,
    draw_contours: bool = True,
    draw_ids: bool = True,
) -> np.ndarray:
    """Render coloured mask overlays onto a single BGR frame.

    Produces the same visual result as ``StreamingOverlayWriter.write_frame()``
    in ``sam3_tracking_example.py``.

    Parameters
    ----------
    frame_bgr : np.ndarray
        BGR image, shape ``(H, W, 3)``, dtype ``uint8``.
    masks_by_id : dict[int, np.ndarray]
        Mapping of ``object_id`` to binary mask ``(H, W)``.
    alpha : float
        Overlay opacity (0–1).
    draw_contours : bool
        Draw contour outlines on mask edges.
    draw_ids : bool
        Render ``ID:<n>`` labels above each mask.

    Returns
    -------
    np.ndarray
        BGR frame with overlays applied.
    """
    if not masks_by_id:
        return frame_bgr

    base = frame_bgr.copy()
    overlay = frame_bgr.copy()

    for obj_id, mask in masks_by_id.items():
        color_rgb = get_object_color(obj_id)
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        mask_bool = mask.astype(bool)
        overlay[mask_bool] = color_bgr

        if draw_contours:
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(base, contours, -1, color_bgr, 2)

    out = cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0)

    if draw_ids:
        for obj_id, mask in masks_by_id.items():
            ys, xs = np.where(mask)
            if len(ys) == 0:
                continue
            cx, cy = int(xs.mean()), int(ys.min()) - 10
            color_rgb = get_object_color(obj_id)
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
            cv2.putText(
                out,
                f"ID:{obj_id}",
                (cx - 20, max(cy, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color_bgr,
                2,
            )

    return out


# ---------------------------------------------------------------------------
# RLE decoding
# ---------------------------------------------------------------------------

def decode_rle_mask(rle: dict) -> np.ndarray:
    """Decode a single COCO-style RLE dict to a binary mask ``(H, W)``."""
    if isinstance(rle["counts"], str):
        rle = {"size": rle["size"], "counts": rle["counts"].encode("utf-8")}
    return mask_utils.decode(rle).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main rendering loop
# ---------------------------------------------------------------------------

def render_overlay_video(
    video_path: str | Path,
    tracking_json_path: str | Path,
    output_video_path: str | Path,
    *,
    mask_alpha: float = 0.4,
    show_ids: bool = True,
    show_contours: bool = True,
    start_frame: int = 0,
    end_frame: int = -1,
    frame_rate: float | None = None,
) -> Path:
    """Render an overlay video from tracking results.

    Parameters
    ----------
    video_path : path-like
        Source video (MP4).
    tracking_json_path : path-like
        ``tracking_results.json`` from SAM3 tracking (sub-task 2.2).
    output_video_path : path-like
        Destination overlay MP4.
    mask_alpha : float
        Overlay opacity (0–1).
    show_ids : bool
        Draw ``ID:<n>`` labels.
    show_contours : bool
        Draw contour outlines on mask edges.
    start_frame : int
        First frame to render (inclusive).
    end_frame : int
        Last frame to render (inclusive, -1 = end of video).
    frame_rate : float or None
        Output FPS.  ``None`` copies the source video's FPS.

    Returns
    -------
    Path
        Resolved path to the written overlay video.
    """
    with open(tracking_json_path) as f:
        tracking = json.load(f)

    if "frames" not in tracking:
        raise ValueError("tracking JSON missing 'frames' key")

    # Build lookup: frame_idx -> {object_id: rle_dict}
    frame_data: dict[int, dict[int, dict]] = {}
    for entry in tracking["frames"]:
        idx = entry["frame_idx"]
        objs: dict[int, dict] = {}
        for obj in entry.get("objects", []):
            objs[obj["object_id"]] = obj["mask_rle"]
        frame_data[idx] = objs

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    fps = frame_rate if frame_rate is not None else src_fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if end_frame < 0:
        end_frame = total_frames - 1

    output_path = Path(output_video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    try:
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for fidx in tqdm(
            range(start_frame, end_frame + 1),
            desc="Rendering overlay",
            unit="frame",
        ):
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # Decode masks for this frame.
            masks_by_id: dict[int, np.ndarray] = {}
            for obj_id, rle in frame_data.get(fidx, {}).items():
                masks_by_id[obj_id] = decode_rle_mask(rle)

            out = render_overlay_frame(
                frame_bgr, masks_by_id,
                alpha=mask_alpha, draw_contours=show_contours, draw_ids=show_ids,
            )
            writer.write(out)
    finally:
        writer.release()
        cap.release()

    print(f"Overlay video written to: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render mask overlay video from SAM3 tracking results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video-path", required=True, help="Path to the source MP4 video.")
    p.add_argument("--tracking-json", required=True, help="Path to tracking_results.json.")
    p.add_argument(
        "--output-video-path", default=None,
        help="Output overlay MP4 path.  Default: <tracking_json_dir>/overlay.mp4",
    )
    p.add_argument("--mask-alpha", type=float, default=0.4, help="Overlay opacity (0-1).")
    p.add_argument("--show-ids", action=argparse.BooleanOptionalAction, default=True, help="Render ID labels.")
    p.add_argument("--show-contours", action=argparse.BooleanOptionalAction, default=True, help="Draw contour outlines.")
    p.add_argument("--start-frame", type=int, default=0, help="First frame to render (inclusive).")
    p.add_argument("--end-frame", type=int, default=-1, help="Last frame to render (-1 = end).")
    p.add_argument("--frame-rate", type=float, default=None, help="Output FPS (default: same as source).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    tracking_json = Path(args.tracking_json)
    if not tracking_json.exists():
        print(f"Error: tracking JSON not found: {tracking_json}", file=sys.stderr)
        sys.exit(1)

    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    output = args.output_video_path
    if output is None:
        output = tracking_json.parent / "overlay.mp4"

    render_overlay_video(
        video_path=video_path,
        tracking_json_path=tracking_json,
        output_video_path=output,
        mask_alpha=args.mask_alpha,
        show_ids=args.show_ids,
        show_contours=args.show_contours,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        frame_rate=args.frame_rate,
    )


if __name__ == "__main__":
    main()
