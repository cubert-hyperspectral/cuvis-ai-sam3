# Changelog

## [Unreleased]

- Added Windows installer packaging scaffold under `installer/` for SAM3 REST server, tray launcher, and weight downloader executables
- Added `configs/sam3-server.env` runtime config template for packaged installs (`SAM3_HOST`, `SAM3_PORT`, `SAM3_CHECKPOINT_PATH`, `SAM3_DEVICE`)
- Added mandatory checkpoint download step in the Inno Setup installer flow; setup aborts when weight download fails
- Fixed `SAM3TrackerInference`, `SAM3StreamingPropagation`, and `SAM3ObjectTracker` nodes producing empty outputs for propagation windows ≥ 8 frames by setting `hotstart_unmatch_thresh = inf` on the SAM3 model at load time (ALL-5389 bbox propagation bug)
- Added `disable_hotstart_retro_suppression` to REST `PropagateRequest` to let clients opt out of retroactive hotstart output suppression during interactive propagation
- Added end-to-end forwarding of `disable_hotstart_retro_suppression` from REST route/service through `Sam3VideoPredictor` into model `propagate_in_video` calls
- Fixed propagation output suppression behavior by using per-frame removed-object snapshots when the new flag is enabled, preventing earlier yielded frames from being retroactively emptied
- Fixed hotstart-removal behavior when `disable_hotstart_retro_suppression=True`: hotstart suppression/removal heuristics are now bypassed in tracker update planning so tracks are not physically dropped after unmatched-threshold windows
- Added REST tests for the new propagate request field default/override and route forwarding behavior
- Added cuvis_ai_sam3 wrapper package with plugin scaffolding and node stubs
- Added UV/CUDA-ready pyproject.toml with optional torch+cu126 dependency group
- Added uv.lock for reproducible dependency resolution
- Added LazyCv2VideoFrameLoader for on-demand frame reading (avoids preloading all frames into RAM)
- Added cv2_lazy video loader type in load_video_frames_from_video_file
- Added opencv-python-headless dependency for lazy cv2 loader
- Added sam3_tracking_example.py CLI tool with text-prompt detection and forward propagation
- Added tracking overlay MP4 output with colored masks, contours, and object ID labels
- Added JSON output with RLE-encoded masks, bboxes, and detection scores per frame
- Added --bf16, --compile, --skip-overlay, --save-masks, and --profile CLI flags
- Added StreamingOverlayWriter for single-pass overlay rendering (no mask accumulation)
- Added compile parameter passthrough in Sam3VideoPredictor for torch.compile support
- Added _trim_output_per_object method to evict stale per-object tensor slices from memory
- Added frame_filter scan depth cap (4x max_obj_ptrs_in_encoder) to prevent O(N) backward iteration
- Added post-yield eviction of cached_frame_outputs and per-frame metadata in propagate_in_video
- Added cleanup of hotstart metadata (unmatched_frame_inds, overlap_pair_to_frame_inds) for removed objects
- Changed trim_past_non_cond_mem_for_eval default from False to True in model_builder
- Changed debug print in sam3_tracker_base to logger.debug
- Changed _det_track_one_frame metadata initialization to avoid O(N²) deepcopy of append-only structures
- Changed rank0_metadata to shallow copy (nested structures are append-only or replaced, not mutated)
- Fixed progressive GPU memory growth during long video tracking (memory now plateaus after ~64 frames)
- Fixed O(N) per-frame slowdown by bounding output_dict and output_dict_per_obj to sliding window
- Removed MANIFEST.in (replaced by pyproject.toml build configuration)
