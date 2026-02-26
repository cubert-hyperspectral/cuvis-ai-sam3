"""Render mask overlay video from SAM3 tracking results.

Supports both tracker JSON formats:
- Legacy frame-based format with a top-level "frames" list
- Video COCO-like format with "videos" and "annotations"
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
    """Return a deterministic RGB color for object id."""
    return OBJECT_COLORS[obj_id % len(OBJECT_COLORS)]


def render_overlay_frame(
    frame_bgr: np.ndarray,
    masks_by_id: dict[int, np.ndarray],
    *,
    alpha: float = 0.4,
    draw_contours: bool = True,
    draw_ids: bool = True,
) -> np.ndarray:
    """Render colored mask overlays onto one BGR frame."""
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


def decode_rle_mask(rle: dict) -> np.ndarray:
    """Decode one COCO-style RLE dict to a binary mask (H, W)."""
    if isinstance(rle["counts"], str):
        rle = {"size": rle["size"], "counts": rle["counts"].encode("utf-8")}
    return mask_utils.decode(rle).astype(np.uint8)


def _frame_lookup_from_legacy(tracking: dict) -> dict[int, dict[int, dict]]:
    frame_data: dict[int, dict[int, dict]] = {}
    for entry in tracking["frames"]:
        frame_idx = int(entry["frame_idx"])
        objs: dict[int, dict] = {}
        for obj in entry.get("objects", []):
            objs[int(obj["object_id"])] = obj["mask_rle"]
        frame_data[frame_idx] = objs
    return frame_data


def _frame_lookup_from_video_coco(tracking: dict) -> dict[int, dict[int, dict]]:
    videos = tracking.get("videos", [])
    annotations = tracking.get("annotations", [])
    if not videos:
        raise ValueError("COCO tracking JSON missing 'videos' entries")

    video = videos[0]
    frame_indices = video.get("frame_indices")
    if frame_indices is None:
        start_frame = int(video.get("start_frame", 0))
        length = int(video.get("length", 0))
        if length > 0:
            frame_indices = list(range(start_frame, start_frame + length))
        else:
            max_len = max((len(ann.get("segmentations", [])) for ann in annotations), default=0)
            frame_indices = list(range(start_frame, start_frame + max_len))

    frame_data: dict[int, dict[int, dict]] = {}
    for ann in annotations:
        track_id = int(ann.get("track_id", ann.get("id", 0)))
        segmentations = ann.get("segmentations", [])
        for idx, seg in enumerate(segmentations):
            if seg is None or idx >= len(frame_indices):
                continue
            frame_idx = int(frame_indices[idx])
            frame_data.setdefault(frame_idx, {})[track_id] = seg

    return frame_data


def build_frame_lookup(tracking: dict) -> dict[int, dict[int, dict]]:
    """Build frame->object->RLE lookup from either supported tracking JSON format."""
    if "frames" in tracking:
        return _frame_lookup_from_legacy(tracking)
    if "videos" in tracking and "annotations" in tracking:
        return _frame_lookup_from_video_coco(tracking)
    raise ValueError("Unsupported tracking JSON format: expected legacy frames or video COCO")


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
    """Render an overlay video from tracker JSON outputs."""
    with open(tracking_json_path, encoding="utf-8") as f:
        tracking = json.load(f)

    frame_data = build_frame_lookup(tracking)

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

        for frame_idx in tqdm(
            range(start_frame, end_frame + 1),
            desc="Rendering overlay",
            unit="frame",
        ):
            ret, frame_bgr = cap.read()
            if not ret:
                break

            masks_by_id: dict[int, np.ndarray] = {}
            for obj_id, rle in frame_data.get(frame_idx, {}).items():
                masks_by_id[obj_id] = decode_rle_mask(rle)

            out = render_overlay_frame(
                frame_bgr,
                masks_by_id,
                alpha=mask_alpha,
                draw_contours=show_contours,
                draw_ids=show_ids,
            )
            writer.write(out)
    finally:
        writer.release()
        cap.release()

    print(f"Overlay video written to: {output_path}")
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render mask overlay video from SAM3 tracking results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video-path", required=True, help="Path to the source MP4 video.")
    p.add_argument(
        "--tracking-json",
        required=True,
        help="Path to tracking_results.json (legacy frames JSON or video COCO JSON).",
    )
    p.add_argument(
        "--output-video-path",
        default=None,
        help="Output overlay MP4 path. Default: <tracking_json_dir>/overlay.mp4",
    )
    p.add_argument("--mask-alpha", type=float, default=0.4, help="Overlay opacity (0-1).")
    p.add_argument(
        "--show-ids",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render ID labels.",
    )
    p.add_argument(
        "--show-contours",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw contour outlines.",
    )
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
