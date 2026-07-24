# Changelog

## [Unreleased]

- Added a process-wide shared SAM3 vision-language backbone: all SAM3 nodes in one runtime share a single GPU-resident ViT + text backbone through a get-or-build registry, so loading a second SAM3 pipeline (or switching between two) reuses the resident backbone and rebuilds only the lightweight heads instead of re-reading the 3.4GB `sam3.pt` and reconstructing the full 848M-parameter model. Nodes claim the entry at construction (weakref-backed) and it is freed when the last claim drops, with the head state dicts kept in CPU RAM for a cheap re-warm.
- `build_sam3_image_model` / `build_sam3_video_model` accept optional `backbone=` and `state_dict=` injection (both default to the previous behavior), and `_load_checkpoint` gained `state_dict` / `skip_prefixes` so an injected backbone's weights are reused without a second disk read; the video builder loads `strict=False` and asserts the only missing keys are `detector.backbone.*`.
- The shared backbone is pinned in eval (its `train()` is overridden to warn and no-op, because the ViT carries `drop_path_rate=0.1` and the text tower `dropout=0.1`), built eagerly during `to(device)`, and keyed by architecture flags, checkpoint identity, and device. `compile_model=True` and `CUVIS_SAM3_NO_BACKBONE_SHARING=1` bypass sharing for A/B timing and field diagnosis.
- Added a pure-tensor unit suite for the registry (claims, free-at-zero, eval pin, device re-key, bypasses, mid-build failure, concurrency, builder seam) plus a GPU integration test asserting the image and video models share one backbone from a single checkpoint read.
- Note: this shared backbone is a plugin-level mechanism for reusing the model across SAM3 pipelines within one runtime. It is an interim bridge that a future framework-level incremental pipeline reload is expected to supersede.

## 0.2.1 - 2026-07-17

- Require `cuvis-ai-schemas>=0.8.0` and `cuvis-ai-core>=0.11.0`, adopting the released framework versions.
- Raised the `click` floor to `>=8.4.2` (locked 8.4.2) for PYSEC-2026-2132, a command-injection flaw in `click.edit()`; matches the floor cuvis-ai-core now locks.
- Corrected the `setuptools` cap to `<82` (was `<83`): `pkg_resources`, which `sam3/model_builder.py` imports at runtime, was removed in setuptools 82.0.0, so `<83` wrongly admitted a `pkg_resources`-free line. Ignored PYSEC-2026-3447 (setuptools, fixed only in 83.0.0) in the pip-audit step accordingly.
- Raised the `pillow` floor to `>=12.3.0` (locked 12.3.0) for the Pillow 12.3.0 advisory batch (PYSEC-2026-2253/2254/2255/2256/2257/3451/3452/3453).

## 0.2.0 - 2026-07-07

- Added `SAM3PointExpansion`: single-frame interactive node that expands positive / negative / neutral click points into one object mask, caching the ViT image embedding per `frame_id` so re-prompts only re-run the lightweight mask decoder.
- Renamed the RGB input port `rgb_frame` -> `rgb_image` on every SAM3 node (breaking); pipelines wiring `<sam3_node>.inputs.rgb_frame` must switch to `.inputs.rgb_image`.
- Fixed `SAM3MaskPropagation` collapsing a single-component seed to a tiny blob; `_interior_points_per_component` now samples several interior points per component (adaptive grid plus distance-transform anchor) instead of one.
- Extracted the shared single-frame image-predictor logic into `_Sam3ImageNode` (`_sam3_image_base.py`); `SAM3SegmentEverything` now subclasses it, dropping ~76 lines of duplication (underscore-prefixed so `auto_register_package` skips the base).
- Added a `no-local-sources` CI workflow that fails if `pyproject.toml` declares a local `[tool.uv.sources]` path entry.
- CI: scope the detect-secrets scan to git-tracked files (drop `--all-files`), cutting the Security Scanning job runtime.

## 0.1.7 - 2026-06-23

- Require `cuvis-ai-core>=0.10.0` and `cuvis-ai-schemas>=0.7.0`, adopting the released framework versions.
- Capped `setuptools<83` (was `<81`) so the plugin co-installs with cuvis-ai, which requires `setuptools>=81`. `sam3/model_builder.py` imports the deprecated `pkg_resources` at runtime, which still ships in setuptools 81 and 82.
- Bumped `starlette>=1.3.1` for the CVE-2026-54282 / CVE-2026-54283 fixes, and ignored the torch `CVE-2025-3000` (no fixed release) and cryptography `GHSA-537c-gmf6-5ccf` (the 48.0.1 fix must land in a cuvis-ai-core release first) advisories in the pip-audit step.
- Bumped `pydantic-settings>=2.14.2` (GHSA-4xgf-cpjx-pc3j, symlink secrets-dir escape) and refreshed the locked `msgpack` to 1.2.1 (GHSA-6v7p-g79w-8964, pulled in transitively by pip-audit).

## 0.1.6 - 2026-06-10

- Require `cuvis-ai-core>=0.7.1` and `cuvis-ai-schemas>=0.5.2`, inheriting the upstream security floors (`gitpython`, `idna`, `urllib3`, `aiohttp`) transitively instead of pinning them here; kept the `starlette>=1.0.1` pin (REST API, PYSEC-2026-161).
- Synced the fork with `facebookresearch/sam3` `main`.
- Added the `cuvis_ai_compat.yml` dependency-compatibility workflow (audits the plugin's deps against the cuvis-ai-core lock).
- Removed the PyPI/TestPyPI release workflow; the plugin is distributed via git tags referenced from cuvis-ai plugin manifests.
- Excluded Meta's upstream `test/` directory from ruff; stripped `torch` / `torchvision` wheel hashes from `uv.lock`.

## 0.1.5 - 2026-04-30

- Added `reset()` to `SAM3TrackerInference` and overrides in `SAM3TextPropagation`, `SAM3BboxPropagation`, and `SAM3MaskPropagation`. `reset()` clears per-stream tracker state (`_inference_state`, `_generator`, `_frame_buffer`, `_frame_idx`, `_source_frame_ids`, `_internal_to_export_obj_id`, `_next_export_obj_id`, plus prompt-specific seed/category state in subclasses) while preserving the loaded `_model` on GPU. `Predictor._reset_nodes()` now drives this automatically at the start of each `predict()` run, so reusing the same streaming node across two runs (for example, mask propagation on RGB then on a CIR rendering of the same scene) starts the second run with a fresh stream instead of carrying state over.
- Refactored `cleanup()` on the same four classes to delegate to `reset()` and then null `_model` and reset `_evict_horizon`, removing duplicated attribute lists between the two methods. External `cleanup()` behavior is unchanged.

## 0.1.4 - 2026-04-29

- Annotated all six SAM3 node classes with `_category = NodeCategory.MODEL` and `_tags` ClassVars. The base `SAM3TrackerInference` carries the shared `{VIDEO, RGB, MASK, TRACKING, SEGMENTATION, INFERENCE, LEARNABLE, BATCHED, STATEFUL, TORCH}` set; `SAM3TextPropagation`, `SAM3BboxPropagation`, and `SAM3PointPropagation` add their prompt modality (`TEXT`, `BBOX`, `KEYPOINTS`); `SAM3MaskPropagation` reuses the base set (mask is already present); `SAM3SegmentEverything` declares the single-frame variant `{RGB, IMAGE, MASK, SEGMENTATION, INFERENCE, LEARNABLE, BATCHED, TORCH}`.
- Added `cuvis-ai-schemas>=0.4.0` to dependencies (`NodeCategory` / `NodeTag` enums live there).
- Stripped `hash` fields from `torch` / `torchvision` wheel entries in `uv.lock`.

## 0.1.3 - 2026-04-10

- Added `cleanup()` hooks to `SAM3SegmentEverything`, `SAM3TrackerInference`, `SAM3TextPropagation`, `SAM3BboxPropagation`, and `SAM3MaskPropagation` so loaded models, frame buffers, generators, and prompt-tracking runtime state are released when the hosting gRPC session tears down its pipeline.
- Fixed missing CUDA autocast context during streaming propagation by wrapping all `add_prompt`, `add_mask`, `propagate_in_video`, and generator `next()` calls in `torch.autocast`.
- Changed autocast device selection from hardcoded CUDA to device-agnostic lookup via `_AUTOCAST_DTYPE` class variable, enabling bfloat16 autocast on CPU and extensibility to MPS/XPU.
- Changed minimum `cuvis-ai-core` dependency to `>=0.3.4` for eager GPU resource cleanup support.
- Added tests for cleanup lifecycle on all node types and autocast context coverage during streaming model calls.

## 0.1.2 - 2026-04-08

- Changed the root README to a cuvis.ai-focused plugin landing page and moved the upstream SAM3 project README into `README_original.md`
- Removed public Windows installer build and executable references from the root README

## 0.1.1 - 2026-04-08

- Added optional runtime `text_prompt` support to `SAM3MaskPropagation`, applying semantic context only while a mask prompt is injected without switching to text-driven detection
- Changed the minimum `cuvis-ai-core` dependency to `>=0.3.0`

## 0.1.0 - 2026-04-07

- Added the `cuvis_ai_sam3` wrapper package and `register_all_nodes()` entry point for cuvis.ai plugin registration
- Added cuvis.ai nodes for SAM3 streaming tracking and segmentation: `SAM3TrackerInference`, `SAM3TextPropagation`, `SAM3BboxPropagation`, `SAM3PointPropagation`, `SAM3MaskPropagation`, and `SAM3SegmentEverything`
- Added open-ended streaming propagation that consumes one RGB frame per `forward()` call and maintains temporal state across a video
- Added runtime text prompting to `SAM3TextPropagation`, including lazy initialization plus `category_ids` and `category_semantics` outputs
- Added runtime bounding-box prompting to `SAM3BboxPropagation` with stable exported object IDs aligned to the selected prompt object
- Added runtime mask prompting to `SAM3MaskPropagation` with lazy initialization from label-map inputs
- Added `SAM3SegmentEverything` for prompt-free per-frame mask generation using point-grid prompting, filtering, and NMS deduplication
- Added tracker threshold controls, state eviction, and progress logging to improve long-running streaming stability
- Added `LazyCv2VideoFrameLoader` and the `cv2_lazy` video-loader path to avoid preloading whole videos into RAM
- Added object-tracking example tooling, including `examples/object_tracking/sam3_video_tracker.py`, `examples/object_tracking/render_tracking_overlay.py`, and updated example documentation
- Added a FastAPI REST service with configuration, middleware, session lifecycle, text/point/bbox prompt ingestion, propagation, object removal endpoints, test coverage, and Windows installer scaffolding for the REST server, tray launcher, and checkpoint downloader
- Added repository tooling for CI, linting, type checking, security scanning, packaging validation, code coverage, git hooks, and secret scanning
- Changed packaging to a UV-managed cuvis.ai plugin project with package-driven build metadata in `pyproject.toml`
- Changed prompt handling so `SAM3TextPropagation`, `SAM3BboxPropagation`, and `SAM3MaskPropagation` accept runtime input ports instead of constructor-seeded prompts (breaking)
- Changed streaming propagation to preserve frame-sized empty outputs, to avoid double-counting `input_frame_id_offset` when explicit `frame_id` values are supplied, and to reduce overly verbose per-frame logging
- Changed the README from upstream-only installation guidance to cuvis.ai plugin-oriented usage and installation guidance
- Changed the REST API CLI to publish the `rest-api` entry point and accept `IP:PORT` shorthand arguments
- Updated the bundled upstream SAM3 fork to the current `facebookresearch/sam3` main with the SAM 3.1 release, including upstream image-only inference support, lazy `decord` import, cv2 empty-frame handling, `torch.compile` support, and position-encoding compile fixes
- Fixed progressive GPU memory growth and frame-time slowdown in long video tracking by trimming cached model state, per-object tensors, stale metadata, and other long-lived frame caches
- Fixed torch compile runtime stability and post-rebase predictor compatibility in the streaming, segment-everything, and REST service paths
- Fixed text-propagation output IDs by remapping internal SAM object ID `0` to stable exported IDs greater than zero and keeping category outputs aligned
- Fixed retroactive hotstart suppression behavior via `disable_hotstart_retro_suppression`, including REST/API plumbing and regression coverage
- Fixed earlier propagation windows returning empty outputs by overriding hotstart unmatched-threshold handling during model setup
- Fixed CI dependency alignment and release metadata cleanup, including the move away from `MANIFEST.in` to `pyproject.toml` build configuration
