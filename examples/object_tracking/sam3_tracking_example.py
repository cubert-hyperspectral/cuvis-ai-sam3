"""SAM3 video object tracking CLI tool.

Runs SAM3 on an MP4 video, detects objects via text prompt on the start frame,
tracks them through all frames, and outputs:
- A JSON file with consistent object IDs and RLE-encoded masks
- An overlay MP4 video showing colored masks on the original frames
- Optionally, per-frame per-object mask PNGs

Uses lazy on-demand frame loading (LazyCv2VideoFrameLoader) to avoid
preloading all video frames into RAM.

Usage:
    uv run python examples/object_tracking/sam3_tracking_example.py \
        --video-path outputs/channel_selector_false_rgb/trained_false_rgb.mp4 \
        --prompt "person" --output-dir outputs/tracking_test --save-masks

Quick test (10 frames):
    uv run python examples/object_tracking/sam3_tracking_example.py \
        --video-path outputs/channel_selector_false_rgb/trained_false_rgb.mp4 \
        --prompt "person" --output-dir outputs/tracking_quick --end-frame 10

Performance flags:
    --bf16          Enable bfloat16 autocast (~1.3-1.8x faster on Ampere+ GPUs)
    --compile       Enable torch.compile (adds 30-60s warmup, 2-3x faster per frame)
    --skip-overlay  Skip overlay video rendering
    --profile       Save torch profiler trace to output directory
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

class PipelineTimer:
    """Collects wall-clock timings for pipeline stages."""

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

    @contextlib.contextmanager
    def time(self, label: str):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        yield
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - t0
        self.timings[label] = elapsed
        logger.info("%s: %.2f s", label, elapsed)

    def summary(self) -> None:
        total = sum(self.timings.values())
        logger.info("--- Timing Summary ---")
        for label, elapsed in self.timings.items():
            pct = 100 * elapsed / total if total > 0 else 0
            logger.info("  %-30s %7.2f s  (%5.1f%%)", label, elapsed, pct)
        logger.info("  %-30s %7.2f s", "TOTAL", total)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SAM3 video object tracking with text prompts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video-path",
        type=str,
        required=True,
        help="Path to MP4 video file or directory of JPEG frames.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First frame to process.",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=-1,
        help="Last frame to process (-1 = end of video).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="person",
        help="Text prompt for object detection.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./tracking_output",
        help="Directory to write results.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference.",
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Save per-frame per-object mask PNGs (adds disk I/O per frame).",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Override path to SAM3 model checkpoint.",
    )
    parser.add_argument(
        "--mask-alpha",
        type=float,
        default=0.4,
        help="Mask overlay opacity in the output video (0.0-1.0).",
    )
    # --- Performance flags ---
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Enable bfloat16 autocast for inference (faster on Ampere+ GPUs).",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help=(
            "Enable torch.compile for model components (adds 30-60s warmup, "
            "but gives 2-3x speedup per frame; best for >50 frames)."
        ),
    )
    parser.add_argument(
        "--skip-overlay",
        action="store_true",
        help="Skip rendering the overlay video (faster for quick tests).",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable PyTorch profiler and save trace to output directory.",
    )
    return parser.parse_args()


def encode_masks_rle(binary_masks: np.ndarray) -> list[dict]:
    """Encode binary masks to RLE format using SAM3's GPU-accelerated encoder.

    Parameters
    ----------
    binary_masks : np.ndarray
        Boolean mask array of shape ``(N, H, W)``.

    Returns
    -------
    list[dict]
        List of RLE dicts with ``"size"`` and ``"counts"`` keys.
    """
    from sam3.train.masks_ops import robust_rle_encode

    mask_tensor = torch.from_numpy(binary_masks).bool()
    if mask_tensor.ndim == 2:
        mask_tensor = mask_tensor.unsqueeze(0)
    if torch.cuda.is_available():
        mask_tensor = mask_tensor.cuda()
    return robust_rle_encode(mask_tensor)


def save_mask_png(mask: np.ndarray, path: str) -> None:
    """Save a single binary mask as an 8-bit PNG (0 or 255)."""
    img = Image.fromarray((mask.astype(np.uint8)) * 255, mode="L")
    img.save(path)


def get_object_color(obj_id: int) -> tuple[int, int, int]:
    """Get a distinct RGB color for the given object ID."""
    return OBJECT_COLORS[obj_id % len(OBJECT_COLORS)]


def process_frame_outputs(
    frame_idx: int,
    outputs: dict,
    output_dir: Path,
    save_masks: bool,
) -> tuple[dict, dict[int, np.ndarray]]:
    """Process a single frame's outputs into the serialization format.

    Parameters
    ----------
    frame_idx : int
        The frame index.
    outputs : dict
        Raw outputs from SAM3 predictor.
    output_dir : Path
        Root output directory.
    save_masks : bool
        Whether to save mask PNGs.

    Returns
    -------
    tuple[dict, dict[int, np.ndarray]]
        Frame data dict for JSON, and a mapping of object_id -> binary mask.
    """
    obj_ids = outputs["out_obj_ids"]
    probs = outputs["out_probs"]
    binary_masks = outputs["out_binary_masks"]
    boxes = outputs["out_boxes_xywh"]

    objects = []
    masks_by_id: dict[int, np.ndarray] = {}

    if len(obj_ids) > 0:
        rle_list = encode_masks_rle(binary_masks)

        if save_masks:
            frame_dir = output_dir / "masks" / f"frame_{frame_idx:06d}"
            frame_dir.mkdir(parents=True, exist_ok=True)

        for i, obj_id in enumerate(obj_ids):
            oid = int(obj_id)
            obj_data = {
                "object_id": oid,
                "mask_rle": rle_list[i],
                "detection_score": float(probs[i]),
                "bbox_xywh": [float(v) for v in boxes[i]],
            }
            objects.append(obj_data)
            masks_by_id[oid] = binary_masks[i]

            if save_masks:
                mask_path = frame_dir / f"obj_{oid:03d}.png"
                save_mask_png(binary_masks[i], str(mask_path))

    return {"frame_idx": frame_idx, "objects": objects}, masks_by_id


# ---------------------------------------------------------------------------
# Streaming overlay writer (single-pass, no memory accumulation)
# ---------------------------------------------------------------------------

class StreamingOverlayWriter:
    """Renders overlay frames during tracking — no mask accumulation needed."""

    def __init__(
        self,
        video_path: str,
        output_path: str,
        start_frame: int,
        alpha: float = 0.4,
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
        """Read the original frame, blend masks, write to video."""
        # Seek forward to the correct frame (handles any gaps)
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


def main() -> None:
    """Run SAM3 video object tracking."""
    args = parse_args()
    timer = PipelineTimer()

    # --- Validate inputs ---
    video_path = args.video_path
    if not os.path.exists(video_path):
        logger.error("Video path does not exist: %s", video_path)
        sys.exit(1)

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU.")
        args.device = "cpu"

    # --- Create output directory ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_masks:
        (output_dir / "masks").mkdir(exist_ok=True)

    logger.info("Video: %s", video_path)
    logger.info("Prompt: '%s'", args.prompt)
    logger.info("Output: %s", output_dir)
    logger.info("Device: %s", args.device)
    if args.bf16:
        logger.info("bfloat16 autocast: enabled")
    if args.compile:
        num_frames = (args.end_frame - args.start_frame) if args.end_frame >= 0 else -1
        if 0 < num_frames < 50:
            logger.warning(
                "torch.compile warmup takes 30-60s; for %d frames consider omitting --compile",
                num_frames,
            )
        logger.info("torch.compile: enabled")

    # --- Build predictor ---
    with timer.time("Model load"):
        from sam3.model_builder import build_sam3_video_predictor

        predictor = build_sam3_video_predictor(
            checkpoint_path=args.checkpoint_path,
            video_loader_type="cv2_lazy",
            compile=args.compile,
        )

    # --- Autocast context ---
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.bf16 and args.device == "cuda"
        else contextlib.nullcontext()
    )

    # --- Profiler context ---
    if args.profile:
        prof_ctx = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
    else:
        prof_ctx = contextlib.nullcontext()

    # --- Run tracking ---
    session_id = None
    try:
        with prof_ctx as prof:
            # Start session
            with timer.time("Session start"):
                response = predictor.handle_request({
                    "type": "start_session",
                    "resource_path": video_path,
                })
                session_id = response["session_id"]
                logger.info("Session started: %s", session_id)

            # Add text prompt on start frame
            logger.info(
                "Adding prompt '%s' on frame %d...", args.prompt, args.start_frame
            )
            with amp_ctx:
                with timer.time("Prompt detection"):
                    prompt_response = predictor.handle_request({
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": args.start_frame,
                        "text": args.prompt,
                    })

            # Check initial detection results
            frames_data = []
            initial_outputs = prompt_response["outputs"]
            initial_obj_ids = initial_outputs["out_obj_ids"]
            logger.info(
                "Detected %d object(s) on frame %d", len(initial_obj_ids), args.start_frame
            )

            if len(initial_obj_ids) == 0:
                logger.warning("No objects detected with prompt '%s'. Exiting.", args.prompt)
                _write_results(
                    output_dir, video_path, args.prompt, args.start_frame,
                    args.start_frame, frames_data,
                )
                return

            # Compute max frames to track
            max_frame_num = None
            if args.end_frame >= 0:
                max_frame_num = args.end_frame - args.start_frame
                if max_frame_num <= 0:
                    logger.error(
                        "--end-frame (%d) must be > --start-frame (%d)",
                        args.end_frame, args.start_frame,
                    )
                    sys.exit(1)

            # --- Set up streaming overlay writer (renders during tracking) ---
            overlay_writer = None
            if not args.skip_overlay:
                overlay_path = str(output_dir / "tracking_overlay.mp4")
                overlay_writer = StreamingOverlayWriter(
                    video_path=video_path,
                    output_path=overlay_path,
                    start_frame=args.start_frame,
                    alpha=args.mask_alpha,
                )

            # Propagate forward
            if args.compile:
                logger.info(
                    "Propagating tracking forward (torch.compile warmup on first frames, ~30-60s)..."
                )
            else:
                logger.info("Propagating tracking forward...")
            with amp_ctx:
                with timer.time("Tracking propagation"):
                    for response in predictor.handle_stream_request({
                        "type": "propagate_in_video",
                        "session_id": session_id,
                        "propagation_direction": "forward",
                        "start_frame_index": args.start_frame,
                        "max_frame_num_to_track": max_frame_num,
                    }):
                        frame_idx = response["frame_index"]
                        outputs = response["outputs"]
                        frame_data, mask_map = process_frame_outputs(
                            frame_idx, outputs, output_dir, args.save_masks
                        )
                        frames_data.append(frame_data)
                        # Stream overlay frame immediately — no mask accumulation
                        if overlay_writer is not None:
                            overlay_writer.write_frame(frame_idx, mask_map)
                        # mask_map is discarded here — not stored in memory

            if overlay_writer is not None:
                overlay_writer.close()

        # --- Post-tracking (outside profiler) ---

        # Determine actual end frame
        if len(frames_data) > 0:
            actual_end_frame = max(f["frame_idx"] for f in frames_data)
        else:
            actual_end_frame = args.start_frame

        # Write JSON results
        with timer.time("JSON write"):
            _write_results(
                output_dir, video_path, args.prompt, args.start_frame,
                actual_end_frame, frames_data,
            )

        if not args.skip_overlay:
            logger.info("Overlay video: %s", output_dir / "tracking_overlay.mp4")
        else:
            logger.info("Skipping overlay render (--skip-overlay).")

        # Profiler output
        if args.profile and prof is not None:
            trace_path = str(output_dir / "torch_trace.json")
            prof.export_chrome_trace(trace_path)
            logger.info("Profiler trace: %s", trace_path)
            logger.info(
                "\n%s",
                prof.key_averages().table(sort_by="cuda_time_total", row_limit=20),
            )

        # Summary
        all_obj_ids = set()
        for fd in frames_data:
            for obj in fd["objects"]:
                all_obj_ids.add(obj["object_id"])

        logger.info("--- Tracking Complete ---")
        logger.info("Frames tracked: %d", len(frames_data))
        logger.info("Unique objects: %d (IDs: %s)", len(all_obj_ids), sorted(all_obj_ids))
        logger.info("Results: %s", output_dir / "tracking_results.json")
        if not args.skip_overlay:
            logger.info("Overlay: %s", output_dir / "tracking_overlay.mp4")
        if args.save_masks:
            logger.info("Mask PNGs: %s", output_dir / "masks")

        timer.summary()

        # Frame rate metrics
        num_frames = len(frames_data)
        if num_frames > 0:
            tracking_time = timer.timings.get("Tracking propagation", 0)
            total_time = sum(timer.timings.values())
            if tracking_time > 0:
                tracking_fps = num_frames / tracking_time
                logger.info("--- Frame Rate ---")
                logger.info("  Tracking:   %.2f FPS  (%.3f s/frame)", tracking_fps, 1.0 / tracking_fps)
            if total_time > 0:
                end_to_end_fps = num_frames / total_time
                logger.info("  End-to-end: %.2f FPS  (%.3f s/frame)", end_to_end_fps, 1.0 / end_to_end_fps)

    finally:
        if session_id is not None:
            predictor.handle_request({
                "type": "close_session",
                "session_id": session_id,
            })
            logger.info("Session closed.")


def _write_results(
    output_dir: Path,
    video_path: str,
    prompt: str,
    start_frame: int,
    end_frame: int,
    frames_data: list[dict],
) -> None:
    """Write tracking results to JSON."""
    frames_data.sort(key=lambda f: f["frame_idx"])

    results = {
        "video_path": str(video_path),
        "prompt": prompt,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "num_frames_tracked": len(frames_data),
        "frames": frames_data,
    }

    json_path = output_dir / "tracking_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %s", json_path)


if __name__ == "__main__":
    main()
