# Changelog

## [Unreleased]
  * Added the `cuvis_ai_sam3` wrapper package and `register_all_nodes()` entry point for cuvis.ai plugin registration.
  * Added cuvis.ai nodes for SAM3 tracking and segmentation: `SAM3TrackerInference`, `SAM3TextPropagation`, `SAM3BboxPropagation`, `SAM3PointPropagation`, `SAM3MaskPropagation`, and `SAM3SegmentEverything`.
  * Added open-ended streaming propagation that consumes one RGB frame per `forward()` call and maintains temporal state across a video.
  * Added runtime text prompting to `SAM3TextPropagation`, including lazy initialization plus `category_ids` and `category_semantics` outputs.
  * Added runtime bounding-box prompting to `SAM3BboxPropagation` with stable exported object IDs based on prompt object IDs.
  * Added runtime mask prompting to `SAM3MaskPropagation` with lazy initialization from label-map inputs.
  * Added `SAM3SegmentEverything` for prompt-free per-frame mask generation using point-grid prompting, filtering, and NMS deduplication.
  * Added `LazyCv2VideoFrameLoader` and the `cv2_lazy` video-loader path to avoid preloading whole videos into RAM.
  * Added the `examples/object_tracking/sam3_video_tracker.py` CLI and accompanying object-tracking example documentation.
  * Added a FastAPI REST service with configuration, middleware, session lifecycle, text/point/bbox prompt ingestion, propagation, object removal endpoints, and test coverage.
  * Added Windows installer/build scaffolding for the REST server, tray launcher, and checkpoint downloader.
  * Added repository tooling for CI, linting, type checking, security scanning, packaging validation, code coverage, git hooks, and secret scanning.
  * Changed packaging to a UV-managed cuvis.ai plugin project with locked dependencies and package-driven build metadata in `pyproject.toml`.
  * Breaking: changed `SAM3TextPropagation`, `SAM3BboxPropagation`, and `SAM3MaskPropagation` from constructor-seeded prompts to runtime input ports.
  * Changed streaming propagation to open-ended operation without fixed `num_frames` or prompt-index configuration, with prompts seeded from stream frame `0`.
  * Changed empty streaming outputs to preserve the current frame size instead of collapsing to `1x1`.
  * Changed bbox output IDs to align with the selected prompt object.
  * Changed the README from upstream-focused installation guidance to cuvis.ai plugin-oriented usage and installation guidance.
  * Changed the REST API CLI to accept an `IP:PORT` shorthand argument.
  * Changed the published core dependency from a git-pinned `cuvis-ai-core` source to the released `cuvis-ai-core>=0.2.0` package.
  * Changed the package CLI entry points to keep `rest-api` and remove the duplicate `sam3-rest-api` script alias.
  * Fixed progressive GPU memory growth and frame-time slowdown in long video tracking by trimming cached model state, per-object tensors, stale metadata, and other long-lived frame caches.
  * Fixed torch-compile/runtime stability by passing compile settings through the video predictor and tightening streaming state handling.
  * Fixed explicit `frame_id` handling so `input_frame_id_offset` is not double-counted.
  * Fixed text-propagation output IDs by remapping internal SAM object ID `0` to stable exported IDs greater than zero.
  * Fixed retroactive hotstart suppression behavior via `disable_hotstart_retro_suppression`, including REST/API plumbing and regression coverage.
  * Fixed earlier propagation windows returning empty outputs by overriding hotstart unmatched-threshold handling during model setup.
  * Fixed CI dependency health by refreshing audited pins in `uv.lock` and realigning the lockfile with current package metadata.
  * Removed `MANIFEST.in` in favor of `pyproject.toml` build configuration.
