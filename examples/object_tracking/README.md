# SAM3 Object Tracking Examples

Track objects in MP4 videos using SAM3 text-prompt detection and temporal propagation.

## Prerequisites

- Python 3.11
- NVIDIA GPU with CUDA support (6+ GB VRAM)
- [uv](https://docs.astral.sh/uv/) package manager

## Setup

```powershell
cd D:\code-repos\cuvis-ai-sam3

# Install all dependencies (including dev extras)
uv sync --all-extras
```

Model weights (`models/sam3.pt`, ~3.4 GB) are already downloaded in this repo.
If missing, the script auto-downloads them from HuggingFace on first run.

## sam3_tracking_example.py

Detects objects via text prompt on the start frame, tracks them through all
frames, and outputs structured JSON with RLE-encoded masks, an overlay video,
and optional per-frame mask PNGs.

Uses lazy on-demand frame loading (`LazyCv2VideoFrameLoader`) so GPU memory
stays around ~6 GB regardless of video length.

### Quick test (10 frames)

```powershell
uv run python examples/object_tracking/sam3_tracking_example.py `
    --video-path path/to/video.mp4 `
    --prompt "person" `
    --output-dir outputs/tracking_quick `
    --end-frame 10 `
    --save-masks
```

### Full video tracking

```powershell
uv run python examples/object_tracking/sam3_tracking_example.py `
    --video-path path/to/video.mp4 `
    --prompt "person" `
    --output-dir outputs/tracking_full `
    --save-masks
```

### CLI arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--video-path` | str | **required** | Path to MP4 video or JPEG frame directory |
| `--prompt` | str | `"person"` | Text prompt for object detection |
| `--output-dir` | str | `./tracking_output` | Directory to write results |
| `--start-frame` | int | `0` | First frame to process |
| `--end-frame` | int | `-1` | Last frame to process (`-1` = end of video) |
| `--device` | str | `cuda` | Device: `cuda` or `cpu` |
| `--save-masks` | flag | off | Save per-frame per-object mask PNGs |
| `--checkpoint-path` | str | `None` | Override model checkpoint path |
| `--mask-alpha` | float | `0.4` | Mask overlay opacity (0.0-1.0) |
| `--bf16` | flag | off | Enable bfloat16 autocast (faster on Ampere+ GPUs) |
| `--compile` | flag | off | Enable torch.compile (30-60s warmup, 2-3x faster per frame) |
| `--skip-overlay` | flag | off | Skip overlay video rendering |
| `--profile` | flag | off | Save torch profiler trace to output directory |

### Performance tips

For **quick tests** (< 50 frames), use `--bf16 --skip-overlay` to minimize overhead:

```powershell
uv run python examples/object_tracking/sam3_tracking_example.py `
    --video-path path/to/video.mp4 `
    --prompt "person" `
    --output-dir outputs/tracking_quick `
    --end-frame 10 `
    --bf16 --skip-overlay
```

For **long videos** (100+ frames), add `--compile` for 2-3x per-frame speedup
(the 30-60s compilation warmup is amortized over many frames):

```powershell
uv run python examples/object_tracking/sam3_tracking_example.py `
    --video-path path/to/video.mp4 `
    --prompt "person" `
    --output-dir outputs/tracking_full `
    --bf16 --compile
```

To **profile** and identify bottlenecks, use `--profile` to save a Chrome trace:

```powershell
uv run python examples/object_tracking/sam3_tracking_example.py `
    --video-path path/to/video.mp4 `
    --prompt "person" `
    --output-dir outputs/profile_test `
    --end-frame 10 `
    --bf16 --profile
```

Open `outputs/profile_test/torch_trace.json` in `chrome://tracing` to inspect
GPU kernel timings. A timing summary table is printed at the end of every run.

### Outputs

```
{output_dir}/
    tracking_results.json   # Object IDs, RLE masks, bboxes, scores per frame
    tracking_overlay.mp4    # Original video with colored mask overlays
    masks/                  # (when --save-masks)
        frame_000000/
            obj_001.png     # 8-bit single-channel mask (0 or 255)
            obj_002.png
        frame_000001/
            ...
```

### JSON schema

```json
{
    "video_path": "path/to/video.mp4",
    "prompt": "person",
    "start_frame": 0,
    "end_frame": 299,
    "num_frames_tracked": 300,
    "frames": [
        {
            "frame_idx": 0,
            "objects": [
                {
                    "object_id": 1,
                    "mask_rle": {"size": [480, 640], "counts": "..."},
                    "detection_score": 0.95,
                    "bbox_xywh": [0.12, 0.34, 0.25, 0.60]
                }
            ]
        }
    ]
}
```
