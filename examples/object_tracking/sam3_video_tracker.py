"""SAM3 video object tracking CLI tool.

Runs SAM3 on an MP4 video, detects objects via text prompt on the start frame,
tracks them through all frames, and writes:
- COCO-style video JSON with per-track per-frame segmentations
- Overlay MP4 with colored masks and IDs
- Optional per-frame per-object mask PNGs

Example (50 frames):
    uv run python examples/object_tracking/sam3_video_tracker.py \
        --video-path "D:\\data\\XMR_notarget_Busstation\\20260226\\Auto_013+01.mp4" \
        --end-frame 49
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import logging
import sys
from pathlib import Path

import click
import cv2
import numpy as np
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MASK_ALPHA = 0.2
REPO_ROOT = Path(__file__).resolve().parents[2]

# Distinct colors for object IDs (RGB, 0-255)
OBJECT_COLORS = [
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


def encode_masks_rle(binary_masks: np.ndarray) -> list[dict]:
    """Encode binary masks to COCO-style RLE dictionaries."""
    from sam3.train.masks_ops import robust_rle_encode

    mask_tensor = torch.from_numpy(binary_masks).bool()
    if mask_tensor.ndim == 2:
        mask_tensor = mask_tensor.unsqueeze(0)
    if torch.cuda.is_available():
        mask_tensor = mask_tensor.cuda()

    rle_list = robust_rle_encode(mask_tensor)
    sanitized: list[dict] = []
    for rle in rle_list:
        counts = rle.get("counts")
        if isinstance(counts, bytes):
            counts = counts.decode("utf-8")
        sanitized.append({
            "size": [int(rle["size"][0]), int(rle["size"][1])],
            "counts": counts,
        })
    return sanitized


def save_mask_png(mask: np.ndarray, path: str) -> None:
    """Save a single binary mask as an 8-bit PNG (0 or 255)."""
    img = Image.fromarray((mask.astype(np.uint8)) * 255, mode="L")
    img.save(path)


def get_object_color(obj_id: int) -> tuple[int, int, int]:
    """Get a distinct RGB color for the given object ID."""
    return OBJECT_COLORS[obj_id % len(OBJECT_COLORS)]


def _bbox_norm_to_abs(
    bbox_xywh_norm: np.ndarray | list[float],
    width: int,
    height: int,
) -> list[float]:
    x, y, w, h = [float(v) for v in bbox_xywh_norm]
    return [x * width, y * height, w * width, h * height]


def process_frame_outputs(
    frame_idx: int,
    outputs: dict,
    output_dir: Path,
    save_masks: bool,
    video_width: int,
    video_height: int,
) -> tuple[dict, dict[int, np.ndarray]]:
    """Process one frame from SAM3 into serializable object entries + overlay masks."""
    obj_ids = outputs["out_obj_ids"]
    probs = outputs["out_probs"]
    binary_masks = outputs["out_binary_masks"]
    boxes = outputs["out_boxes_xywh"]

    objects: list[dict] = []
    masks_by_id: dict[int, np.ndarray] = {}

    if len(obj_ids) > 0:
        rle_list = encode_masks_rle(binary_masks)

        if save_masks:
            frame_dir = output_dir / "masks" / f"frame_{frame_idx:06d}"
            frame_dir.mkdir(parents=True, exist_ok=True)

        for i, obj_id in enumerate(obj_ids):
            oid = int(obj_id)
            bbox_abs = _bbox_norm_to_abs(boxes[i], video_width, video_height)
            area = float(np.count_nonzero(binary_masks[i]))

            obj_data = {
                "object_id": oid,
                "mask_rle": rle_list[i],
                "detection_score": float(probs[i]),
                "bbox_xywh": bbox_abs,
                "area": area,
            }
            objects.append(obj_data)
            masks_by_id[oid] = binary_masks[i]

            if save_masks:
                mask_path = frame_dir / f"obj_{oid:03d}.png"
                save_mask_png(binary_masks[i], str(mask_path))

    return {"frame_idx": frame_idx, "objects": objects}, masks_by_id


class StreamingOverlayWriter:
    """Renders overlay frames during tracking in a single pass."""

    def __init__(
        self,
        video_path: str,
        output_path: str,
        start_frame: int,
        alpha: float = MASK_ALPHA,
    ) -> None:
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self.next_frame_idx = start_frame
        self.alpha = alpha
        self.output_path = output_path
        self.writer: cv2.VideoWriter | None = None

    def write_frame(
        self,
        frame_idx: int,
        masks_by_id: dict[int, np.ndarray],
    ) -> None:
        """Read one source frame, blend masks, and write it to output video."""
        while self.next_frame_idx <= frame_idx:
            ret, frame_bgr = self.cap.read()
            if not ret:
                return
            self.next_frame_idx += 1

        if self.writer is None:
            h, w = frame_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (w, h))

        if masks_by_id:
            overlay = frame_bgr.copy()
            for obj_id, mask in masks_by_id.items():
                color_rgb = get_object_color(obj_id)
                color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
                mask_bool = mask.astype(bool)
                overlay[mask_bool] = color_bgr
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(frame_bgr, contours, -1, color_bgr, 2)

            frame_bgr = cv2.addWeighted(overlay, self.alpha, frame_bgr, 1 - self.alpha, 0)

            for obj_id, mask in masks_by_id.items():
                ys, xs = np.where(mask)
                if len(ys) > 0:
                    cx, cy = int(xs.mean()), int(ys.min()) - 10
                    color_rgb = get_object_color(obj_id)
                    color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
                    cv2.putText(
                        frame_bgr,
                        f"ID:{obj_id}",
                        (cx - 20, max(cy, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color_bgr,
                        2,
                    )

        self.writer.write(frame_bgr)

    def close(self) -> None:
        self.cap.release()
        if self.writer is not None:
            self.writer.release()
            logger.info("Overlay video: %s", self.output_path)


def _read_video_meta(video_path: str) -> tuple[int, int, float, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for metadata: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return width, height, fps, total_frames


def _resolve_output_dir(video_path: str, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir.resolve()
    return (REPO_ROOT / "outputs" / Path(video_path).stem).resolve()


def _build_video_coco(
    video_path: str,
    prompt: str,
    start_frame: int,
    end_frame: int,
    video_width: int,
    video_height: int,
    frames_data: list[dict],
) -> dict:
    sorted_frames = sorted(frames_data, key=lambda item: int(item["frame_idx"]))
    frame_indices = [int(frame["frame_idx"]) for frame in sorted_frames]

    tracks: dict[int, dict[int, dict]] = {}
    for frame in sorted_frames:
        frame_idx = int(frame["frame_idx"])
        for obj in frame.get("objects", []):
            track_id = int(obj["object_id"])
            tracks.setdefault(track_id, {})[frame_idx] = obj

    annotations: list[dict] = []
    for ann_id, track_id in enumerate(sorted(tracks), start=1):
        per_frame = tracks[track_id]
        segmentations: list[dict | None] = []
        bboxes: list[list[float]] = []
        areas: list[float] = []
        scores: list[float] = []

        for frame_idx in frame_indices:
            obj = per_frame.get(frame_idx)
            if obj is None:
                segmentations.append(None)
                bboxes.append([0.0, 0.0, 0.0, 0.0])
                areas.append(0.0)
                continue

            segmentations.append(obj["mask_rle"])
            bboxes.append([float(v) for v in obj["bbox_xywh"]])
            areas.append(float(obj["area"]))
            scores.append(float(obj["detection_score"]))

        annotations.append({
            "id": ann_id,
            "track_id": track_id,
            "video_id": 1,
            "category_id": 1,
            "iscrowd": 0,
            "segmentations": segmentations,
            "bboxes": bboxes,
            "areas": areas,
            "score": float(np.mean(scores)) if scores else 0.0,
            "height": video_height,
            "width": video_width,
            "noun_phrase": prompt,
        })

    now_utc = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    return {
        "info": {
            "description": "SAM3 video tracking output",
            "source_video_path": str(video_path),
            "prompt": prompt,
            "source_start_frame": int(start_frame),
            "source_end_frame": int(end_frame),
            "num_frames_tracked": len(sorted_frames),
            "generated_at": now_utc,
        },
        "videos": [
            {
                "id": 1,
                "video_name": Path(video_path).name,
                "height": video_height,
                "width": video_width,
                "length": len(sorted_frames),
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "frame_indices": frame_indices,
            }
        ],
        "annotations": annotations,
        "categories": [{"id": 1, "name": prompt}],
    }


def _write_results(
    output_dir: Path,
    video_path: str,
    prompt: str,
    start_frame: int,
    end_frame: int,
    video_width: int,
    video_height: int,
    frames_data: list[dict],
) -> None:
    """Write tracking results to video-style COCO JSON."""
    results = _build_video_coco(
        video_path=video_path,
        prompt=prompt,
        start_frame=start_frame,
        end_frame=end_frame,
        video_width=video_width,
        video_height=video_height,
        frames_data=frames_data,
    )

    json_path = output_dir / "tracking_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %s", json_path)


def run_tracking(
    video_path: str,
    start_frame: int,
    end_frame: int,
    prompt: str,
    output_dir: Path | None,
    save_masks: bool,
    checkpoint_path: str | None,
    bf16: bool,
    compile_model: bool,
) -> None:
    """Run SAM3 video object tracking."""
    if not Path(video_path).exists():
        logger.error("Video path does not exist: %s", video_path)
        sys.exit(1)

    if not torch.cuda.is_available():
        logger.error("CUDA is required for this script, but no CUDA device is available.")
        sys.exit(1)

    if start_frame < 0:
        logger.error("--start-frame must be >= 0")
        sys.exit(1)

    if end_frame >= 0 and end_frame < start_frame:
        logger.error("--end-frame (%d) must be >= --start-frame (%d)", end_frame, start_frame)
        sys.exit(1)

    video_width, video_height, _, _ = _read_video_meta(video_path)

    resolved_output_dir = _resolve_output_dir(video_path, output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    if save_masks:
        (resolved_output_dir / "masks").mkdir(exist_ok=True)

    logger.info("Video: %s", video_path)
    logger.info("Prompt: '%s'", prompt)
    logger.info("Output: %s", resolved_output_dir)
    logger.info("Device: cuda")
    logger.info("Overlay alpha: %.1f", MASK_ALPHA)
    if bf16:
        logger.info("bfloat16 autocast: enabled")
    if compile_model:
        num_frames = (end_frame - start_frame + 1) if end_frame >= 0 else -1
        if 0 < num_frames < 50:
            logger.warning(
                "torch.compile warmup takes 30-60s; for %d frames consider omitting --compile",
                num_frames,
            )
        logger.info("torch.compile: enabled")

    from sam3.model_builder import build_sam3_video_predictor

    predictor = build_sam3_video_predictor(
        checkpoint_path=checkpoint_path,
        video_loader_type="cv2_lazy",
        compile=compile_model,
    )

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if bf16
        else contextlib.nullcontext()
    )

    overlay_path = str(resolved_output_dir / "tracking_overlay.mp4")
    overlay_writer = StreamingOverlayWriter(
        video_path=video_path,
        output_path=overlay_path,
        start_frame=start_frame,
        alpha=MASK_ALPHA,
    )

    session_id = None
    frames_data: list[dict] = []
    try:
        response = predictor.handle_request({
            "type": "start_session",
            "resource_path": video_path,
        })
        session_id = response["session_id"]
        logger.info("Session started: %s", session_id)

        logger.info("Adding prompt '%s' on frame %d...", prompt, start_frame)
        with amp_ctx:
            prompt_response = predictor.handle_request({
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": start_frame,
                "text": prompt,
            })

        initial_outputs = prompt_response["outputs"]
        initial_obj_ids = initial_outputs["out_obj_ids"]
        logger.info("Detected %d object(s) on frame %d", len(initial_obj_ids), start_frame)

        max_frame_num = None
        if end_frame >= 0:
            max_frame_num = end_frame - start_frame

        if compile_model:
            logger.info(
                "Propagating tracking forward (torch.compile warmup on first frames, ~30-60s)..."
            )
        else:
            logger.info("Propagating tracking forward...")

        with amp_ctx:
            for response in predictor.handle_stream_request({
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "forward",
                "start_frame_index": start_frame,
                "max_frame_num_to_track": max_frame_num,
            }):
                frame_idx = int(response["frame_index"])
                outputs = response["outputs"]
                frame_data, mask_map = process_frame_outputs(
                    frame_idx=frame_idx,
                    outputs=outputs,
                    output_dir=resolved_output_dir,
                    save_masks=save_masks,
                    video_width=video_width,
                    video_height=video_height,
                )
                frames_data.append(frame_data)
                overlay_writer.write_frame(frame_idx, mask_map)

        actual_end_frame = max((f["frame_idx"] for f in frames_data), default=start_frame)

        _write_results(
            output_dir=resolved_output_dir,
            video_path=video_path,
            prompt=prompt,
            start_frame=start_frame,
            end_frame=actual_end_frame,
            video_width=video_width,
            video_height=video_height,
            frames_data=frames_data,
        )

        all_obj_ids = sorted(
            {
                int(obj["object_id"])
                for frame in frames_data
                for obj in frame.get("objects", [])
            }
        )

        logger.info("--- Tracking Complete ---")
        logger.info("Frames tracked: %d", len(frames_data))
        logger.info("Unique objects: %d (IDs: %s)", len(all_obj_ids), all_obj_ids)
        logger.info("Results: %s", resolved_output_dir / "tracking_results.json")
        logger.info("Overlay: %s", resolved_output_dir / "tracking_overlay.mp4")
        if save_masks:
            logger.info("Mask PNGs: %s", resolved_output_dir / "masks")

    finally:
        overlay_writer.close()
        if session_id is not None:
            predictor.handle_request({
                "type": "close_session",
                "session_id": session_id,
            })
            logger.info("Session closed.")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--video-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to MP4 video file or directory of JPEG frames.",
)
@click.option("--start-frame", type=int, default=0, show_default=True, help="First frame to process.")
@click.option(
    "--end-frame",
    type=int,
    default=-1,
    show_default=True,
    help="Last frame to process (-1 = end of video).",
)
@click.option(
    "--prompt",
    type=str,
    default="person",
    show_default=True,
    help="Text prompt for object detection.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help=(
        "Directory to write results. Default: "
        "D:/code-repos/cuvis-ai-sam3/outputs/{video_name}"
    ),
)
@click.option(
    "--save-masks",
    is_flag=True,
    help="Save per-frame per-object mask PNGs (adds disk I/O per frame).",
)
@click.option(
    "--checkpoint-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Override path to SAM3 model checkpoint.",
)
@click.option(
    "--bf16",
    is_flag=True,
    help="Enable bfloat16 autocast for inference (faster on Ampere+ GPUs).",
)
@click.option(
    "--compile",
    "compile_model",
    is_flag=True,
    help=(
        "Enable torch.compile for model components (adds 30-60s warmup, "
        "but gives 2-3x speedup per frame; best for >50 frames)."
    ),
)
def main(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    prompt: str,
    output_dir: Path | None,
    save_masks: bool,
    checkpoint_path: Path | None,
    bf16: bool,
    compile_model: bool,
) -> None:
    """SAM3 video object tracking with text prompts."""
    run_tracking(
        video_path=str(video_path),
        start_frame=start_frame,
        end_frame=end_frame,
        prompt=prompt,
        output_dir=output_dir,
        save_masks=save_masks,
        checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
        bf16=bf16,
        compile_model=compile_model,
    )


if __name__ == "__main__":
    main()
