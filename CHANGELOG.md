# Changelog

## 0.6.0 - 2026-11-17

- **`SAM31TextPropagation`: text-prompt tracking on SAM 3.1 Object Multiplex.** New node with the ports of `SAM3TextPropagation`, so a pipeline keeps its wiring: it swaps the class name, sets `weight_name: sam3.1` and replaces `max_tracker_states` by `max_num_objects`. SAM 3.1 tracks the objects of a prompt jointly in buckets of sixteen, so a frame costs about the same for one object as for a dozen. On the bus-station demo take (300 frames of 1080 x 1000 pixels, one frame per call, RTX 5070 Ti Laptop, SAM 3 at its pipeline thresholds 0.5 / 0.7, the new node at its defaults) the median time per frame is 444 ms against 1160 ms for `person` (about 12 against 9 objects) and 445 ms against 561 ms for `street light` (3 objects); for `bus` and `car` (one object) the two are equal within 3 %. The node is faster once several objects are tracked and no faster for a single one. Foreground agreement with SAM 3 (IoU of the union of all masks, per frame) is 0.98 for `car`, 0.92 for `bus`, 0.83 for `person` and 0.71 for `street light`: small objects agree least. Defaults are the SAM 3.1 thresholds (`score_threshold_detection=0.4`, `new_det_thresh=0.65`, `det_nms_thresh=0.1`, `overlap_suppress_thresh=0.7`). `det_nms_thresh` is an overlap over the smaller of two masks, as the SAM 3.1 builder sets it, and not an IoU. Limits: the node needs a CUDA device and one process (it refuses a model built with `WORLD_SIZE` above 1); a different prompt restarts tracking on its frame under a new category, because SAM 3.1 cannot add a second prompt to a running stream (the same prompt again changes nothing); objects are never released, so once `max_num_objects` (default 64) have been seen the stream stops picking up new ones until the next prompt or `reset()`; the node cannot be part of an exported span. It drives the model with its own streaming code (one output per frame, a 16-frame tracker memory window that keeps GPU memory flat on long streams), builds the model on the GPU the pipeline was moved to and refuses a checkpoint that leaves model weights unset.
- `cuvis_ai_sam3/weights.py` declares a second row, `sam3.1` (`cubert-gmbh/sam3.1`: `sam3.1_multiplex.pt`, 3.5 GB, and its `config.json`; a mirror of the gated `facebook/sam3.1`). The row is selected by the new `weight_name` hparam of `SAM31TextPropagation` and its group has no default row, so the six SAM 3 nodes and their pipelines keep needing `sam3` alone. A pipeline provisions `sam3.1` by setting `weight_name: sam3.1` on the node. A host that provisions every unconditional row of a plugin, as CuvisNEXT does, asks for `sam3` as well when such a pipeline loads. `checkpoint_path` bypasses the registry as before.
- The vendored `sam3/model/decoder.py` lets the decoder attention use the efficient and math kernels where PyTorch has no flash kernel (its Windows wheels). `functional_attention` asked for flash alone and failed there with "No available kernel" as soon as the SAM 3.1 tracker attended to its memory (upstream issue 523 of `facebookresearch/sam3`). Flash stays the first choice where it exists.

## 0.5.3 - 2026-11-05

- Python 3.12 and 3.13 are supported: `requires-python` is `>=3.11,<3.14` (was `<3.12`), so pip installs the plugin on Colab, which runs 3.13. CI tests 3.11 and 3.13. The relock changes no package version.
- Security: the locked environment is refreshed for the advisories published since 0.5.2: aiohttp 3.14.5, certifi 2026.10.23, filelock 4.2.7, fsspec 2026.9.0, gitpython 3.2.1, hydra-core 1.3.8, msgpack 1.2.4, multidict 7.2.1, pillow 13.0.0 and urllib3 2.8.0. Ignored with a comment in CI next to the existing entries: setuptools GHSA-xmj3-9gf7-h52w (fixed in 84.1.0, behind the `<82` cap that keeps pkg_resources for the upstream model builder; the flaw sits in the legacy egg install of a crafted wheel on Windows).

## 0.5.2 - 2026-11-03

- The SAM3 image nodes (`SAM3SegmentEverything`, `SAM3PointExpansion`) and the streaming trackers build their model inside `shared_backbone.preserve_autocast_state()`: the sam3 library enters `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` in its constructors with a bare `__enter__()` and never leaves it, so after `pipeline.to("cuda")` every convolution or matmul in the thread ran in bfloat16 and the next float32 output port downstream failed its dtype check (`PortCompatibilityError: Dtype mismatch: source has torch.bfloat16, target expects torch.float32`). The nodes wrap their own predict calls in an autocast context already; the enabled flag, dtype, cache flag and nesting depth are now restored after the build.
- Every forward of those nodes returns `mask`, `object_ids` and `detection_scores` on the device of the incoming `rgb_image` (`shared_backbone.outputs_on_device`) instead of the cpu where they are assembled, so consumers on a cuda pipeline no longer have to move the mask themselves.
- Security: refreshed the locked `anyio` to 4.14.2 (was 4.12.1) for CVE-2026-63374 (certificate validation of IDNA-encoded host names in `connect_tcp()` / `TLSStream.wrap()`) and CVE-2026-64847 (process-pool workers wedge on an undrained stderr pipe). anyio is pulled in transitively by httpx, starlette, sse-starlette and watchfiles, so there is no floor of ours to raise; nothing else in the lock moved.
- Security: refreshed the locked `cryptography` to 50.0.1 (was 46.0.7) for GHSA-537c-gmf6-5ccf and PYSEC-2026-3552/3553/3554 and dropped their four pip-audit ignores. The note that tied this bump to a cuvis-ai-core release was wrong: the plugin compat audit compares this plugin's declared specifiers with core's lock, not the two locks with each other, and core locks 50.0.1 now as well.

## 0.5.1 - 2026-09-16

- `SAM3PointExpansion._parse_points` no longer raises `ValueError` for a prompt with no positive point (all-negative or empty). `forward()` already gates on the same condition and returns an empty mask without calling the model, matching `RTSAM2PointExpansion`; the parse step raised where `forward` returned empty, so a caller reading the parsed arrays directly saw an error the pipeline itself does not. Parsing now agrees with the empty-return contract.
- Security: floors `cuvis-ai-core>=0.17.3` and `pytorch-lightning>=2.6.6` for PYSEC-2026-3967, a remote-code-execution vulnerability in Lightning's checkpoint `_load_state` path that executes attacker-controlled module names from a crafted checkpoint's `_instantiator` hyperparameters, bypassing `weights_only=True`. cuvis-ai-core 0.17.3 raised the same floor and lock, so the fix is inherited through core; the explicit plugin floor keeps the `--strict` core-lock audit consistent. Floors equal the locked versions.

## 0.5.0 - 2026-09-07

- `cuvis_ai_sam3/weights.py` declares the plugin's weights (`WEIGHTS`: the `sam3` row with the `cubert-gmbh/sam3` pin, its `config.json` aux file, size, licence, and `checkpoint_path` as the hparam that bypasses the cache) and the package registers them with `ModelWeights.register` at import; cuvis-ai's `emit_metadata` projects the same tuple into the plugin manifest's `weights:` block, so CuvisNEXT and the installer can provision SAM3 without importing the plugin. Floors `cuvis-ai-core>=0.17.0` (the registry is populated by plugin declarations and the built-in plugin rows left core, so upgrade the plugins together with core) and `cuvis-ai-schemas>=0.12.0` (`PluginWeightEntry`).

## 0.4.0 - 2026-09-04

- **SAM3 weights come from the `cubert-gmbh/sam3` Hugging Face mirror through cuvis-ai-core's weight registry.** The shared backbone resolves its checkpoint with `ModelWeights.resolve("sam3")` (cached path, download when online, `ModelWeightsMissingError` naming `download-model download sam3` when offline) instead of `sam3.model_builder.download_ckpt_from_hf`, so no Hugging Face account, Meta gate or token is needed and the offline child finds the weights under `models--cubert-gmbh--sam3`. `installer/download_weights.py` provisions through the same registry (`--repo-id` / `--filename` still override it). Requires cuvis-ai-core 0.16.0. A cache under the old `models--facebook--sam3` folder is not reused; the weights download once more.

## 0.3.3 - 2026-08-31

- Security: refreshed the locked environment for the advisory batch published since 0.3.2 - aiohttp 3.14.3 (PYSEC-2026-3545/3546/3547), gitpython 3.1.61 (CVE-2026-73620, CVE-2026-76217 and the intermediate GHSA batch; this also retires the three 3.1.51-era pip-audit ignores), hydra-core 1.3.6 (CVE-2026-68508), pip floor raised to >=26.2 (PYSEC-2026-3721). The three cryptography 46.0.7 advisories (PYSEC-2026-3552/3553/3554) have no fixed release yet and are ignored with a comment in CI.
- Gated the `decord` dependency to the platforms it ships wheels for (linux x86_64 and windows amd64): decord 0.6.0 has no aarch64 or macOS wheels and no sdist, so composed child environments on aarch64 hosts (e.g. Jetson Thor) failed `uv sync` with "decord==0.6.0 ... doesn't have a source distribution or wheel for the current platform". decord is only imported lazily by the upstream video-file loaders; the plugin nodes and the REST API use the cv2 loader path throughout. Added a regression test that imports every shipped `cuvis_ai_sam3` and `rest_api` module (plus the upstream loader modules) in a fresh interpreter with decord import-blocked. Local x86 development environments still install decord and are unchanged.

## 0.3.2 - 2026-08-20

- Removed the dead `[tool.uv.sources]` / `[[tool.uv.index]]` torch cu128 configuration: torch is not a direct dependency of this package and the committed lock resolves it from PyPI, so the tables had no effect anywhere (uv honours them only at the resolution root). Composed child environments receive the host-mirrored torch build from cuvis-ai-core >= 0.12.1.

## 0.3.1 - 2026-07-29

- Streaming propagation nodes now survive a failed stream instead of poisoning the session: any exception inside the propagation generator tears the stream state down to an explicit needs-seed state and surfaces the ORIGINAL error exactly once (previously every later frame reported "generator exhausted early", hiding the real failure with no way to recover short of a pipeline reload). A seed-less generator start (the upstream "No points are provided" case) gets the same teardown and a clear needs-seed message, and the next non-empty prompt starts a fresh stream.
- Prompt-frame image features are now pinned against eviction until the tracker has consolidated that frame's outputs. The cache is evicted from three places (the node's pruning, the detector's per-step pop, and the tracker's miss-path rebind), and the periodic reconditioning pass still needs PAST prompt frames' features while the video tracker runs with `backbone=None` — previously this raised "Image features for frame N are not cached" mid-stream. A pin-aware cache container installed at state creation covers all eviction paths without patching vendored code; point/mask inputs are pruned in lockstep with the consolidated and pending frame sets so the tracker's propagation preflight invariant holds.
- `SAM3PointExpansion`'s per-frame embedding cache now keys on frame content in addition to `frame_id`: ids repeat across videos and counter restarts, and a stale hit decoded clicks against the previous scene's embedding (wrong-object masks).

## 0.3.0 - 2026-07-27

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
