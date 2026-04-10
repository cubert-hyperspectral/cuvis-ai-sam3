# Changelog

## [Unreleased]

- Fixed SAM3 streaming and segment-everything nodes to release loaded models, frame buffers, generators, and prompt-tracking runtime state when the hosting gRPC session tears down its pipeline.

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
