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

## sam3_video_tracker.py

`examples/object_tracking/sam3_video_tracker.py` detects objects via text prompt on
the start frame, tracks them through subsequent frames, and writes:

- `tracking_results.json` in video-COCO-style format
- `tracking_overlay.mp4` with always-on overlays (`alpha=0.2`)
- optional mask PNGs (`--save-masks`)

The script uses lazy frame loading (`cv2_lazy`) to avoid preloading full video
content into RAM.

### Defaults and fixed behavior

- Device is internal CUDA-only (no `--device` flag)
- Overlay alpha is fixed to `0.2` (no `--mask-alpha` flag)
- Overlay rendering is always enabled (no `--skip-overlay` flag)
- Profiling output was removed (no `--profile` flag)
- If `--output-dir` is omitted, outputs go to:
  - `D:\code-repos\cuvis-ai-sam3\outputs\{video_name}`

### Quick test (10 frames)

```powershell
uv run python examples/object_tracking/sam3_video_tracker.py `
    --video-path path/to/video.mp4 `
    --prompt "person" `
    --output-dir outputs/tracking_quick `
    --end-frame 10 `
    --save-masks
```

### Exact 50-frame run (requested example)

```powershell
uv run python examples/object_tracking/sam3_video_tracker.py `
    --video-path "D:\data\XMR_notarget_Busstation\20260226\Auto_013+01-trustimulus.mp4" `
    --end-frame 49
```

With default settings, this writes to:

- `D:\code-repos\cuvis-ai-sam3\outputs\Auto_013+01-trustimulus\tracking_results.json`
- `D:\code-repos\cuvis-ai-sam3\outputs\Auto_013+01-trustimulus\tracking_overlay.mp4`

### CLI arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--video-path` | path | **required** | Path to MP4 video or JPEG frame directory |
| `--prompt` | str | `"person"` | Text prompt for object detection |
| `--output-dir` | path | `None` | Output directory; defaults to `D:\code-repos\cuvis-ai-sam3\outputs\{video_name}` |
| `--start-frame` | int | `0` | First frame to process |
| `--end-frame` | int | `-1` | Last frame to process (`-1` = end of video) |
| `--save-masks` | flag | off | Save per-frame per-object mask PNGs |
| `--checkpoint-path` | path | `None` | Override model checkpoint path |
| `--bf16` | flag | off | Enable bfloat16 autocast |
| `--compile` | flag | off | Enable torch.compile (30-60s warmup) |

### Outputs

```text
{output_dir}/
    tracking_results.json   # Video-COCO style tracking results
    tracking_overlay.mp4    # Original video with colored mask overlays
    masks/                  # (when --save-masks)
        frame_000000/
            obj_001.png
            obj_002.png
        frame_000001/
            ...
```

### JSON schema (video COCO style)

```json
{
  "info": {
    "description": "SAM3 video tracking output",
    "source_video_path": "path/to/video.mp4",
    "prompt": "person",
    "source_start_frame": 0,
    "source_end_frame": 49,
    "num_frames_tracked": 50,
    "generated_at": "2026-02-26T12:00:00+00:00"
  },
  "videos": [
    {
      "id": 1,
      "video_name": "video.mp4",
      "height": 720,
      "width": 1280,
      "length": 50,
      "start_frame": 0,
      "end_frame": 49,
      "frame_indices": [0, 1, 2]
    }
  ],
  "annotations": [
    {
      "id": 1,
      "track_id": 7,
      "video_id": 1,
      "category_id": 1,
      "iscrowd": 0,
      "segmentations": [
        {"size": [720, 1280], "counts": "..."},
        null
      ],
      "bboxes": [[100.0, 120.0, 80.0, 140.0], [0.0, 0.0, 0.0, 0.0]],
      "areas": [5421.0, 0.0],
      "score": 0.93,
      "height": 720,
      "width": 1280,
      "noun_phrase": "person"
    }
  ],
  "categories": [
    {"id": 1, "name": "person"}
  ]
}
```

---

## render_tracking_overlay.py

Re-renders overlays from `tracking_results.json` without re-running SAM3
inference.

The script supports both formats:

- legacy frame-based tracking JSON (`frames` list)
- new video-COCO tracking JSON (`videos` + `annotations`)

### Example: render overlay from tracker results

```powershell
uv run python examples/object_tracking/render_tracking_overlay.py `
    --video-path "D:\data\XMR_notarget_Busstation\20260226\Auto_013+01-trustimulus.mp4" `
    --tracking-json "D:\code-repos\cuvis-ai-sam3\outputs\Auto_013+01-trustimulus\tracking_results.json"
```

### Customize rendering

```powershell
# Lower opacity, no contour outlines
uv run python examples/object_tracking/render_tracking_overlay.py `
    --video-path path/to/video.mp4 `
    --tracking-json path/to/tracking_results.json `
    --mask-alpha 0.3 --no-contours

# IDs only, no contours, first 50 frames
uv run python examples/object_tracking/render_tracking_overlay.py `
    --video-path path/to/video.mp4 `
    --tracking-json path/to/tracking_results.json `
    --no-contours --end-frame 50
```

### CLI arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--video-path` | str | **required** | Path to the source MP4 video |
| `--tracking-json` | str | **required** | Path to tracking JSON (legacy or video COCO) |
| `--output-video-path` | str | `<json_dir>/overlay.mp4` | Output overlay MP4 path |
| `--mask-alpha` | float | `0.4` | Overlay opacity (0-1) |
| `--show-ids` / `--no-show-ids` | flag | on | Render `ID:<n>` labels |
| `--show-contours` / `--no-show-contours` | flag | on | Draw contour outlines |
| `--start-frame` | int | `0` | First frame to render (inclusive) |
| `--end-frame` | int | `-1` | Last frame to render (`-1` = end) |
| `--frame-rate` | float | source FPS | Output video FPS |
