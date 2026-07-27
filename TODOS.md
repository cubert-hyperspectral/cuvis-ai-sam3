# TODOs

## Deduplicate tracker weights across the image and video models

**What:** The image model's `inst_interactive_predictor` and the video model's
tracker both load the same `tracker.*` checkpoint keys into two separate module
instances. After the shared vision-language backbone (see
`cuvis_ai_sam3/shared_backbone.py`) the tracker weights are the next-largest
duplicated block when both an image node and a video node are resident in one
process.

**Why:** Extends the shared-backbone pattern to a second component, trimming
VRAM and head-build time further. The heads split measured at build time is
~0.18GB total, so the tracker is a modest slice of that; size the exact win
against the profiling numbers before investing.

**Constraint:** Unlike the backbone, the tracker is stateful (memory bank, per
sequence). Sharing needs a stateless-weights / stateful-state split so two
claimants do not clobber each other's tracking state. This is more invasive
than the backbone case and must not regress streaming correctness.

**Where to start:** `sam3/model_builder.py` `_load_checkpoint` (image path
remaps `tracker.*` -> `inst_interactive_predictor.model.*`) and
`build_sam3_video_model` (loads the full state dict). The
`cuvis_ai_sam3/shared_backbone.py` registry is the pattern to follow for the
weight half.

**Depends on:** the shared-backbone registry landing first.
