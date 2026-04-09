"""SAM3 streaming propagation nodes — one RGB frame per forward(), specialized by prompt type.

Hierarchy:
    SAM3TrackerInference(Node)           — abstract base with shared infrastructure
    ├── SAM3TextPropagation              — text/concept prompt
    ├── SAM3BboxPropagation              — bounding-box prompt
    ├── SAM3PointPropagation             — point prompt
    └── SAM3MaskPropagation              — binary mask prompt
"""

from __future__ import annotations

import contextlib
import json
from abc import abstractmethod
from typing import Any

import cv2
import numpy as np
import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PortSpec
from loguru import logger

from .utils import _bbox_iou_xyxy, _binary_mask_from_xyxy, _centroid_point_from_binary_mask

_STREAMING_SENTINEL = 10**7


class _FrameBuffer:
    """Pre-allocated frame buffer that implements SAM3's ``img_batch`` protocol.

    Receives RGB frames from the pipeline one at a time.  SAM3's engine reads
    ``img_batch[frame_idx]`` during propagation — this object returns the
    pre-processed tensor for the requested index.
    """

    def __init__(
        self,
        image_size: int,
        device: torch.device,
        image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> None:
        self.image_size = int(image_size)
        self.device = torch.device(device)
        self._img_mean = torch.tensor(image_mean, dtype=torch.float16).view(3, 1, 1).to(self.device)
        self._img_std = torch.tensor(image_std, dtype=torch.float16).view(3, 1, 1).to(self.device)
        self._frames: dict[int, torch.Tensor] = {}
        self._next_idx = 0

    # -- public API -----------------------------------------------------------

    def add(self, frame_float_hwc: np.ndarray) -> int:
        """Preprocess an RGB float32 [0,1] frame and store in the next slot.

        Returns the frame index assigned to this frame.
        """
        if frame_float_hwc.ndim != 3 or frame_float_hwc.shape[2] != 3:
            raise ValueError(f"Expected frame shape [H, W, 3], got {tuple(frame_float_hwc.shape)}.")
        # float [0,1] → uint8 [0,255]
        frame_uint8 = np.clip(frame_float_hwc * 255.0, 0, 255).astype(np.uint8)
        # resize to model image_size
        frame_resized = cv2.resize(
            frame_uint8,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_CUBIC,
        )
        # uint8 → float32 CHW → float16 → normalize
        frame_np = frame_resized.astype(np.float32)
        frame_tensor = torch.from_numpy(frame_np).permute(2, 0, 1)
        frame_tensor = frame_tensor.to(device=self.device, dtype=torch.float16)
        frame_tensor -= self._img_mean
        frame_tensor /= self._img_std

        idx = self._next_idx
        self._frames[idx] = frame_tensor
        self._next_idx += 1
        return idx

    # -- SAM3 img_batch protocol ----------------------------------------------

    def __getitem__(self, index: int | torch.Tensor) -> torch.Tensor:
        if isinstance(index, (int, np.integer)):
            idx = int(index)
            if idx not in self._frames:
                if not self._frames:
                    raise IndexError(
                        f"Frame {idx} not yet buffered (have {sorted(self._frames.keys())})."
                    )
                latest_idx = max(self._frames)
                if idx > latest_idx:
                    # SAM3 detector may prefetch the next chunk ahead of the current frame.
                    # While streaming, we fall back to the most recent buffered frame.
                    logger.warning(
                        "FrameBuffer: frame {} not buffered, returning latest frame {} instead",
                        idx,
                        latest_idx,
                    )
                    return self._frames[latest_idx]
                raise IndexError(
                    f"Frame {idx} not available in buffer (have {sorted(self._frames.keys())})."
                )
            return self._frames[idx]
        if isinstance(index, torch.Tensor):
            if index.numel() == 1:
                return self.__getitem__(int(index.item())).unsqueeze(0)
            return torch.stack([self.__getitem__(int(v)) for v in index.tolist()], dim=0)
        raise TypeError(f"Index must be int or Tensor, got {type(index)}.")

    def __len__(self) -> int:
        return self._next_idx

    # -- memory management ----------------------------------------------------

    def prune_before(self, frame_idx: int) -> None:
        """Drop frames with index < frame_idx to free GPU memory."""
        for k in [k for k in self._frames if k < frame_idx]:
            del self._frames[k]

    def to(self, device: torch.device | str, *args: Any, **kwargs: Any) -> _FrameBuffer:
        del args, kwargs
        self.device = torch.device(device)
        self._img_mean = self._img_mean.to(self.device)
        self._img_std = self._img_std.to(self.device)
        self._frames = {k: v.to(self.device) for k, v in self._frames.items()}
        return self


# =============================================================================
# Base class
# =============================================================================


class SAM3TrackerInference(Node):
    """Abstract base for streaming SAM3 propagation nodes.

    Consumes one RGB frame per ``forward()`` call, maintaining temporal state
    across the entire video sequence. Subclasses implement ``_apply_prompt()``
    for their specific prompt type.

    Compatible with ``TrackingOverlayNode`` and ``TrackingCocoJsonNode`` sinks.
    """

    INPUT_SPECS = {
        "rgb_frame": PortSpec(dtype=torch.float32, shape=(1, -1, -1, 3)),
        "frame_id": PortSpec(
            dtype=torch.int64,
            shape=(1,),
            description="Source frame index [1]. If omitted, local stream index is used.",
            optional=True,
        ),
    }
    OUTPUT_SPECS = {
        "mask": PortSpec(dtype=torch.int32, shape=(1, -1, -1)),
        "object_ids": PortSpec(dtype=torch.int64, shape=(1, -1)),
        "detection_scores": PortSpec(dtype=torch.float32, shape=(1, -1)),
    }

    def __init__(
        self,
        checkpoint_path: str | None = None,
        compile_model: bool = False,
        # -- SAM3 thresholds --
        score_threshold_detection: float = 0.5,
        new_det_thresh: float = 0.7,
        det_nms_thresh: float = 0.1,
        overlap_suppress_thresh: float = 0.7,
        max_tracker_states: int = 5,
        **kwargs: Any,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._compile_model = compile_model
        self._score_threshold_detection = score_threshold_detection
        self._new_det_thresh = new_det_thresh
        self._det_nms_thresh = det_nms_thresh
        self._overlap_suppress_thresh = overlap_suppress_thresh
        self._max_tracker_states = max_tracker_states
        self._log_every_n_frames = 50

        # Runtime state (initialized on first forward)
        self._model: Any = None
        self._frame_buffer: _FrameBuffer | None = None
        self._inference_state: dict[str, Any] | None = None
        self._generator: Any = None
        self._frame_idx: int = 0
        self._source_frame_ids: list[int] = []
        self._internal_to_export_obj_id: dict[int, int] = {}
        self._next_export_obj_id: int = 1
        self._evict_horizon: int = 64
        # Keep a tiny recent frame window in memory for safety.
        self._buffer_keep_recent: int = 2

        super().__init__(
            checkpoint_path=checkpoint_path,
            compile_model=compile_model,
            score_threshold_detection=score_threshold_detection,
            new_det_thresh=new_det_thresh,
            det_nms_thresh=det_nms_thresh,
            overlap_suppress_thresh=overlap_suppress_thresh,
            max_tracker_states=max_tracker_states,
            **kwargs,
        )

    # -- Model loading --------------------------------------------------------

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        from sam3.model_builder import build_sam3_video_model

        build_kwargs: dict[str, Any] = {}
        if self._checkpoint_path:
            build_kwargs["checkpoint_path"] = self._checkpoint_path
        if self._compile_model:
            build_kwargs["compile"] = True

        self._model = build_sam3_video_model(**build_kwargs)

        # Disable hotstart look-ahead for streaming (incompatible with frame-at-a-time
        # buffering) but keep recondition_every_nth_frame=16 for performance.
        self._model.hotstart_delay = 0

        # Set thresholds as model attributes (like SAM3TrackerInference)
        self._model.score_threshold_detection = self._score_threshold_detection
        self._model.new_det_thresh = self._new_det_thresh
        self._model.det_nms_thresh = self._det_nms_thresh
        self._model.suppress_overlapping_based_on_recent_occlusion_threshold = (
            self._overlap_suppress_thresh
        )

        logger.info(
            "SAM3 model loaded (image_size={}, device={}, hotstart_delay={})",
            self._model.image_size,
            self._model.device,
            getattr(self._model, "hotstart_delay", "?"),
        )
        tracker = getattr(self._model, "tracker", None)
        max_obj_ptrs = int(getattr(tracker, "max_obj_ptrs_in_encoder", 16))
        self._evict_horizon = 4 * max_obj_ptrs
        self._install_streaming_detector_guard()

    def _install_streaming_detector_guard(self) -> None:
        """Prevent detector look-ahead from requesting frames not buffered yet."""
        detector = getattr(self._model, "detector", None)
        if detector is None or getattr(detector, "_streaming_guard_installed", False):
            return

        original_forward = detector.forward_video_grounding_multigpu

        def _forward_video_grounding_multigpu_streaming(*args: Any, **kwargs: Any) -> Any:
            requested_num_frames = kwargs.get("num_frames")
            if requested_num_frames is not None and self._frame_buffer is not None:
                buffered_frames = max(
                    1, min(int(self._frame_buffer._next_idx), int(requested_num_frames))
                )
                kwargs["num_frames"] = buffered_frames
            multigpu_buffer = kwargs.get("multigpu_buffer")
            if isinstance(multigpu_buffer, dict):
                # Keep detector buffering aligned with the actually buffered frames.
                multigpu_buffer.clear()
            return original_forward(*args, **kwargs)

        detector.forward_video_grounding_multigpu = _forward_video_grounding_multigpu_streaming
        detector._streaming_guard_installed = True

    def _model_device(self) -> torch.device:
        try:
            return next(self._model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _model_eval_context(self) -> contextlib.AbstractContextManager[None]:
        """Run streaming-model inference under the expected CUDA autocast mode."""
        if self._model is None:
            return contextlib.nullcontext()
        if self._model_device().type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    # -- State initialization -------------------------------------------------

    def _build_state(self, orig_height: int, orig_width: int) -> dict[str, Any]:
        """Build SAM3 inference state using dict-based streaming pattern (like v1)."""
        from sam3.model.data_misc import BatchedDatapoint
        from sam3.model.geometry_encoders import Prompt

        device = self._model_device()

        input_batch = BatchedDatapoint(
            img_batch=self._frame_buffer,
            find_text_batch=["<text placeholder>", "visual"],
            find_inputs={},
            find_targets={},
            find_metadatas={},
        )

        state: dict[str, Any] = {
            "image_size": int(self._model.image_size),
            "num_frames": 0,
            "orig_height": int(orig_height),
            "orig_width": int(orig_width),
            "constants": {},
            "input_batch": input_batch,
            "previous_stages_out": {},
            "text_prompt": None,
            "per_frame_raw_point_input": {},
            "per_frame_raw_box_input": {},
            "per_frame_visual_prompt": {},
            "per_frame_geometric_prompt": {},
            "per_frame_cur_step": {},
            "visual_prompt_embed": None,
            "visual_prompt_mask": None,
            "tracker_inference_states": [],
            "tracker_metadata": {},
            "feature_cache": {},
            "cached_frame_outputs": {},
            "action_history": [],
            "is_image_only": False,
        }

        state["constants"]["empty_geometric_prompt"] = Prompt(
            box_embeddings=torch.zeros(0, 1, 4, device=device),
            box_mask=torch.zeros(1, 0, device=device, dtype=torch.bool),
            box_labels=torch.zeros(0, 1, device=device, dtype=torch.long),
            point_embeddings=torch.zeros(0, 1, 2, device=device),
            point_mask=torch.zeros(1, 0, device=device, dtype=torch.bool),
            point_labels=torch.zeros(0, 1, device=device, dtype=torch.long),
        )
        return state

    def _extend_state_for_frame(
        self,
        frame_idx: int,
        state: dict[str, Any] | None = None,
    ) -> None:
        """Register a new frame index in the inference state (dict-based, like v1)."""
        from sam3.model.data_misc import FindStage, convert_my_tensors
        from sam3.model.utils.misc import copy_data_to_device

        if state is None:
            state = self._inference_state
        if state is None:
            raise RuntimeError("Inference state is not initialized.")
        input_batch = state["input_batch"]

        if frame_idx in input_batch.find_inputs:
            return

        device = self._model_device()
        stage = FindStage(
            img_ids=[frame_idx],
            text_ids=[0],
            input_boxes=[torch.zeros(258)],
            input_boxes_mask=[torch.empty(0, dtype=torch.bool)],
            input_boxes_label=[torch.empty(0, dtype=torch.long)],
            input_points=[torch.empty(0, 257)],
            input_points_mask=[torch.empty(0, dtype=torch.bool)],
            object_ids=[],
        )
        stage = convert_my_tensors(stage)
        stage = copy_data_to_device(stage, device, non_blocking=True)

        input_batch.find_inputs[frame_idx] = stage
        input_batch.find_targets[frame_idx] = None
        input_batch.find_metadatas[frame_idx] = None
        state["previous_stages_out"][frame_idx] = None
        state["per_frame_raw_point_input"][frame_idx] = None
        state["per_frame_raw_box_input"][frame_idx] = None
        state["per_frame_visual_prompt"][frame_idx] = None
        state["per_frame_geometric_prompt"][frame_idx] = None
        state["per_frame_cur_step"][frame_idx] = 0
        if int(state.get("num_frames", 0)) < frame_idx + 1:
            state["num_frames"] = frame_idx + 1
        for tracker_state in state.get("tracker_inference_states", []):
            if isinstance(tracker_state, dict):
                if int(tracker_state.get("num_frames", 0)) < frame_idx + 1:
                    tracker_state["num_frames"] = frame_idx + 1

    def _next_generator_output(self, requested_frame_idx: int) -> tuple[int, dict | None]:
        """Advance the SAM3 propagation generator with a clear error on early exhaustion."""
        if self._generator is None:
            raise RuntimeError("Propagation generator is not initialized.")
        self._mark_cudagraph_step_begin()
        try:
            with self._model_eval_context():
                return next(self._generator)
        except StopIteration as exc:
            raise RuntimeError(
                f"{self.__class__.__name__} generator exhausted early at frame {requested_frame_idx}."
            ) from exc

    def _resolve_source_frame_id(
        self,
        frame_id: torch.Tensor | None,
        fallback_stream_idx: int,
    ) -> int:
        if frame_id is None or frame_id.numel() == 0:
            return int(fallback_stream_idx)
        return int(frame_id.reshape(-1)[0].item())

    def _register_source_frame_id(self, stream_idx: int, source_frame_id: int) -> None:
        if stream_idx == len(self._source_frame_ids):
            self._source_frame_ids.append(int(source_frame_id))
        elif stream_idx < len(self._source_frame_ids):
            prev = int(self._source_frame_ids[stream_idx])
            if prev != int(source_frame_id):
                logger.warning(
                    "Frame ID mismatch at local index {}: previous={}, current={}",
                    stream_idx,
                    prev,
                    source_frame_id,
                )
                self._source_frame_ids[stream_idx] = int(source_frame_id)
        else:
            while len(self._source_frame_ids) < stream_idx:
                self._source_frame_ids.append(int(len(self._source_frame_ids)))
            self._source_frame_ids.append(int(source_frame_id))

    def _mark_cudagraph_step_begin(self) -> None:
        """Mark CUDAGraph step boundaries for compiled execution (if available)."""
        if not self._compile_model:
            return
        compiler = getattr(torch, "compiler", None)
        mark_step = getattr(compiler, "cudagraph_mark_step_begin", None)
        if callable(mark_step):
            mark_step()

    def _source_frame_id_for_stream_idx(self, stream_idx: int) -> int:
        if 0 <= stream_idx < len(self._source_frame_ids):
            return int(self._source_frame_ids[stream_idx])
        return int(stream_idx)

    @staticmethod
    def _empty_output(height: int = 1, width: int = 1) -> dict[str, torch.Tensor]:
        return {
            "mask": torch.zeros(1, int(height), int(width), dtype=torch.int32),
            "object_ids": torch.zeros(1, 0, dtype=torch.int64),
            "detection_scores": torch.zeros(1, 0, dtype=torch.float32),
        }

    def _initialize_stream_state(self, frame_np: np.ndarray) -> None:
        """Initialize frame buffer and SAM3 state from the current RGB frame."""
        orig_h, orig_w = int(frame_np.shape[0]), int(frame_np.shape[1])
        device = self._model_device()

        self._frame_buffer = _FrameBuffer(
            image_size=int(self._model.image_size),
            device=device,
        )
        self._frame_buffer.add(frame_np)

        self._inference_state = self._build_state(orig_h, orig_w)
        self._extend_state_for_frame(0)

    def _start_generator(self, start_frame_idx: int = 0) -> None:
        """Start the open-ended propagation generator from the given internal frame."""
        self._inference_state["num_frames"] = _STREAMING_SENTINEL
        for tracker_state in self._inference_state.get("tracker_inference_states", []):
            if isinstance(tracker_state, dict):
                tracker_state["num_frames"] = _STREAMING_SENTINEL

        with self._model_eval_context():
            self._generator = self._model.propagate_in_video(
                self._inference_state,
                start_frame_idx=int(start_frame_idx),
                max_frame_num_to_track=None,
                reverse=False,
            )

    # -- Prompt application (subclass responsibility) -------------------------

    @abstractmethod
    def _apply_prompt(self) -> None:
        """Apply the configured prompt to stream frame 0 of the inference state."""

    # -- Output conversion ----------------------------------------------------

    @property
    def _requires_cached_frame_outputs_on_prompt_frame(self) -> bool:
        """Whether cached_frame_outputs must be pre-seeded on the prompt frame.

        Point and mask prompts require this; text and bbox do not.
        Override in subclasses that need it.
        """
        return False

    @property
    def _remap_internal_object_ids(self) -> bool:
        """Whether internal SAM object IDs should be remapped to exported IDs >= 1."""
        return False

    def _filter_objects(
        self,
        obj_ids: np.ndarray,
        binary_masks: np.ndarray,
        probs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Hook for subclasses to filter/remap detected objects. Default: pass-through."""
        return obj_ids, binary_masks, probs

    def _remap_object_ids(self, obj_ids: np.ndarray) -> np.ndarray:
        """Map internal SAM IDs to stable exported IDs when enabled.

        The exported label-map representation reserves ``0`` for background, so
        text-prompt tracking uses this mapping to guarantee object IDs are
        strictly positive and stable across frames.
        """
        if not self._remap_internal_object_ids:
            return obj_ids

        remapped = np.empty_like(obj_ids, dtype=np.int64)
        for idx, raw_obj_id in enumerate(obj_ids.tolist()):
            internal_id = int(raw_obj_id)
            export_id = self._internal_to_export_obj_id.get(internal_id)
            if export_id is None:
                export_id = self._next_export_obj_id
                self._internal_to_export_obj_id[internal_id] = export_id
                self._next_export_obj_id += 1
            remapped[idx] = int(export_id)
        return remapped

    def _pack_output(
        self,
        postprocessed: dict | None,
        frame_shape: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Convert SAM3 per-object output to per-frame label map format."""
        if frame_shape is None:
            empty_h, empty_w = 1, 1
        else:
            empty_h, empty_w = int(frame_shape[0]), int(frame_shape[1])
        if postprocessed is None or len(postprocessed.get("out_obj_ids", [])) == 0:
            return self._empty_output(empty_h, empty_w)

        obj_ids = np.asarray(postprocessed["out_obj_ids"], dtype=np.int64)
        binary_masks = np.asarray(postprocessed["out_binary_masks"], dtype=bool)
        probs = np.asarray(postprocessed["out_probs"], dtype=np.float32)

        obj_ids, binary_masks, probs = self._filter_objects(obj_ids, binary_masks, probs)
        if obj_ids.shape[0] == 0:
            return self._empty_output(empty_h, empty_w)
        obj_ids = self._remap_object_ids(obj_ids)

        h, w = binary_masks.shape[1], binary_masks.shape[2]
        label_map = np.zeros((h, w), dtype=np.int32)
        for oid, m in zip(obj_ids, binary_masks, strict=False):
            label_map[m] = int(oid)

        return {
            "mask": torch.from_numpy(label_map).unsqueeze(0),
            "object_ids": torch.from_numpy(np.array(obj_ids, dtype=np.int64)).unsqueeze(0),
            "detection_scores": torch.from_numpy(np.array(probs, dtype=np.float32)).unsqueeze(0),
        }

    @staticmethod
    def _prune_dict(container: Any, keep_from: int) -> None:
        """Remove integer keys strictly less than ``keep_from`` from a dict."""
        if not isinstance(container, dict):
            return
        stale = [k for k in container if isinstance(k, int) and k < keep_from]
        for key in stale:
            container.pop(key, None)

    def _prune_state_for_frame(self, frame_idx: int) -> None:
        """Free cached historical data to keep long runs bounded in memory."""
        if self._inference_state is None:
            return
        state = self._inference_state

        keep_from = max(0, frame_idx - self._buffer_keep_recent + 1)

        cached = state.get("cached_frame_outputs")
        if isinstance(cached, dict):
            stale = [k for k in cached if isinstance(k, int) and k < keep_from]
            for key in stale:
                cached.pop(key, None)

        tracker_md = state.get("tracker_metadata", {})
        scores_fw = tracker_md.get("obj_id_to_tracker_score_frame_wise")
        if isinstance(scores_fw, dict):
            stale = [k for k in scores_fw if isinstance(k, int) and k < keep_from]
            for key in stale:
                scores_fw.pop(key, None)

        rank0_md = tracker_md.get("rank0_metadata", {})
        suppressed_map = rank0_md.get("suppressed_obj_ids", {})
        if isinstance(suppressed_map, dict):
            stale = [k for k in suppressed_map if isinstance(k, int) and k < keep_from]
            for key in stale:
                suppressed_map.pop(key, None)

        evict_before = frame_idx - self._evict_horizon
        if evict_before > 0:
            unmatched = rank0_md.get("unmatched_frame_inds")
            if isinstance(unmatched, dict):
                for obj_id in list(unmatched.keys()):
                    lst = unmatched[obj_id]
                    if isinstance(lst, list):
                        unmatched[obj_id] = [f for f in lst if f >= evict_before]

            overlap = rank0_md.get("overlap_pair_to_frame_inds")
            if isinstance(overlap, dict):
                for pair in list(overlap.keys()):
                    lst = overlap[pair]
                    if isinstance(lst, list):
                        overlap[pair] = [f for f in lst if f >= evict_before]

            for tracker_state in state.get("tracker_inference_states", []):
                if not isinstance(tracker_state, dict):
                    continue
                output_dict = tracker_state.get("output_dict", {})
                non_cond = output_dict.get("non_cond_frame_outputs", {})
                stale = [k for k in non_cond if isinstance(k, int) and k < evict_before]
                for key in stale:
                    del non_cond[key]
                for obj_dict in tracker_state.get("output_dict_per_obj", {}).values():
                    if not isinstance(obj_dict, dict):
                        continue
                    non_cond = obj_dict.get("non_cond_frame_outputs", {})
                    stale = [k for k in non_cond if isinstance(k, int) and k < evict_before]
                    for key in stale:
                        del non_cond[key]
                tracked = tracker_state.get("frames_already_tracked", {})
                if isinstance(tracked, dict):
                    stale = [k for k in tracked if isinstance(k, int) and k < evict_before]
                    for key in stale:
                        del tracked[key]

        if self._frame_buffer is not None:
            self._frame_buffer.prune_before(keep_from)

        self._prune_dict(state.get("previous_stages_out"), keep_from)
        self._prune_dict(state.get("per_frame_raw_point_input"), keep_from)
        self._prune_dict(state.get("per_frame_raw_box_input"), keep_from)
        self._prune_dict(state.get("per_frame_visual_prompt"), keep_from)
        self._prune_dict(state.get("per_frame_geometric_prompt"), keep_from)
        self._prune_dict(state.get("per_frame_cur_step"), keep_from)

        input_batch = state.get("input_batch")
        if input_batch is not None:
            self._prune_dict(input_batch.find_inputs, keep_from)
            self._prune_dict(input_batch.find_targets, keep_from)
            self._prune_dict(input_batch.find_metadatas, keep_from)

    def _inject_mask_prompt_for_object(
        self,
        binary_mask: np.ndarray,
        *,
        frame_idx: int,
        obj_id: int,
    ) -> None:
        """Inject one object update into tracker state using mask + centroid point."""
        device = self._model_device()
        self._inference_state["cached_frame_outputs"].setdefault(int(frame_idx), {})

        mask_binary = np.asarray(binary_mask > 0, dtype=np.float32)
        mask_tensor = torch.from_numpy(mask_binary).to(device=device, dtype=torch.float32)
        with self._model_eval_context():
            self._model.add_mask(
                self._inference_state,
                frame_idx=int(frame_idx),
                obj_id=int(obj_id),
                mask=mask_tensor,
            )

        point_xy = _centroid_point_from_binary_mask(mask_binary)
        if point_xy is None:
            return

        with self._model_eval_context():
            self._model.add_prompt(
                self._inference_state,
                frame_idx=int(frame_idx),
                points=torch.tensor([list(point_xy)], dtype=torch.float32, device=device),
                point_labels=torch.tensor([1], dtype=torch.int64, device=device),
                obj_id=int(obj_id),
            )

    def _evict_excess_tracker_states(self, frame_idx: int) -> None:
        """Evict oldest non-primary tracker states when the count exceeds the cap."""
        if self._inference_state is None or self._max_tracker_states <= 0:
            return
        states = self._inference_state.get("tracker_inference_states", [])
        if len(states) <= self._max_tracker_states:
            return

        remove_fn = getattr(self._model, "_tracker_remove_objects", None)
        if not callable(remove_fn):
            return

        n_to_evict = len(states) - self._max_tracker_states
        obj_ids_to_remove: list[int] = []
        for tracker_state in states[1 : 1 + n_to_evict]:
            if not isinstance(tracker_state, dict):
                continue
            obj_ids_to_remove.extend(tracker_state.get("obj_ids", []))

        if not obj_ids_to_remove:
            return

        logger.trace(
            "Evicting {} tracker state(s) at frame {} ({} -> {}), obj_ids={}",
            n_to_evict,
            frame_idx,
            len(states),
            self._max_tracker_states,
            obj_ids_to_remove,
        )
        remove_fn(states, obj_ids_to_remove)

        removed_set = {int(v) for v in obj_ids_to_remove}
        md = self._inference_state.get("tracker_metadata", {})
        rank = int(getattr(self._model, "rank", 0))

        ids_per_gpu = md.get("obj_ids_per_gpu")
        if ids_per_gpu is not None and rank < len(ids_per_gpu):
            old_ids = ids_per_gpu[rank]
            keep = np.array([int(v) not in removed_set for v in old_ids], dtype=bool)
            ids_per_gpu[rank] = old_ids[keep]

        ids_all = md.get("obj_ids_all_gpu")
        keep_all: np.ndarray | None = None
        if ids_all is not None and hasattr(ids_all, "__len__") and len(ids_all) > 0:
            keep_all = np.array([int(v) not in removed_set for v in ids_all], dtype=bool)
            md["obj_ids_all_gpu"] = ids_all[keep_all]

        num_per_gpu = md.get("num_obj_per_gpu")
        if num_per_gpu is not None and ids_per_gpu is not None:
            for idx in range(len(num_per_gpu)):
                if idx < len(ids_per_gpu):
                    num_per_gpu[idx] = len(ids_per_gpu[idx])

        scores = md.get("obj_id_to_score")
        if isinstance(scores, dict):
            for obj_id in obj_ids_to_remove:
                scores.pop(int(obj_id), None)

        rank0_md = md.get("rank0_metadata", {})
        for key in ("obj_first_frame_idx", "trk_keep_alive"):
            container = rank0_md.get(key)
            if isinstance(container, dict):
                for obj_id in obj_ids_to_remove:
                    container.pop(int(obj_id), None)
        unmatched = rank0_md.get("unmatched_frame_inds")
        if isinstance(unmatched, dict):
            for obj_id in obj_ids_to_remove:
                unmatched.pop(int(obj_id), None)

        removed_obj_ids = rank0_md.get("removed_obj_ids")
        if isinstance(removed_obj_ids, set):
            removed_obj_ids.update(removed_set)

        overlap = rank0_md.get("overlap_pair_to_frame_inds")
        if isinstance(overlap, dict):
            stale = [k for k in overlap if k[0] in removed_set or k[1] in removed_set]
            for key in stale:
                del overlap[key]

        masklet_confirmation = rank0_md.get("masklet_confirmation")
        if isinstance(masklet_confirmation, dict) and keep_all is not None:
            for key in ("status", "consecutive_det_num"):
                arr = masklet_confirmation.get(key)
                if arr is not None and hasattr(arr, "__len__") and len(arr) == len(keep_all):
                    masklet_confirmation[key] = arr[keep_all]

    # -- Forward --------------------------------------------------------------

    def forward(
        self,
        rgb_frame: torch.Tensor,
        frame_id: torch.Tensor | None = None,
        context: Context | None = None,  # noqa: ARG002
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Process one RGB frame and return tracking outputs.

        Parameters
        ----------
        rgb_frame : torch.Tensor
            Shape ``[1, H, W, 3]``, float32, values in ``[0, 1]``.

        Returns
        -------
        dict with keys: ``mask``, ``object_ids``, ``detection_scores``.
        """
        self._ensure_model()

        frame_np = rgb_frame[0].detach().cpu().numpy()  # [H, W, 3] float32
        frame_shape = (int(frame_np.shape[0]), int(frame_np.shape[1]))
        stream_idx = self._frame_idx
        source_frame_id = self._resolve_source_frame_id(frame_id, fallback_stream_idx=stream_idx)
        self._register_source_frame_id(stream_idx, source_frame_id)

        if stream_idx == 0:
            # -- First frame: initialize state and buffer --
            internal_frame_idx = 0
            self._initialize_stream_state(frame_np)

            if self._requires_cached_frame_outputs_on_prompt_frame:
                # Interactivity prompt paths require cached outputs on the prompt frame.
                self._inference_state["cached_frame_outputs"].setdefault(0, {})
            self._mark_cudagraph_step_begin()
            self._apply_prompt()

            self._start_generator(start_frame_idx=0)
            _yield_frame_idx, postprocessed = self._next_generator_output(internal_frame_idx)
            result = self._pack_output(postprocessed, frame_shape=frame_shape)
        else:
            # -- Subsequent frames: buffer + extend state + advance generator --
            internal_frame_idx = self._frame_buffer.add(frame_np)
            self._extend_state_for_frame(internal_frame_idx)
            _yield_frame_idx, postprocessed = self._next_generator_output(internal_frame_idx)
            result = self._pack_output(postprocessed, frame_shape=frame_shape)

        self._evict_excess_tracker_states(internal_frame_idx)
        self._prune_state_for_frame(internal_frame_idx)
        if torch.cuda.is_available() and internal_frame_idx > 0 and internal_frame_idx % 50 == 0:
            torch.cuda.empty_cache()

        self._frame_idx += 1

        # if self._log_every_n_frames > 0 and stream_idx % self._log_every_n_frames == 0:
        #     n_objs = result["object_ids"].shape[1]
        #     logger.info(
        #         "{}: local frame {}, source frame {}, {} objects",
        #         self.__class__.__name__,
        #         stream_idx,
        #         source_frame_id,
        #         n_objs,
        #     )

        return result


# =============================================================================
# Concrete subclasses
# =============================================================================


class SAM3TextPropagation(SAM3TrackerInference):
    """SAM3 streaming propagation with a text/concept prompt.

    Detects and tracks all objects matching runtime ``text_prompt`` inputs
    (e.g. ``"person"`` or ``"car"``).
    """

    INPUT_SPECS = {
        **SAM3TrackerInference.INPUT_SPECS,
        "text_prompt": PortSpec(
            dtype=str,
            shape=(),
            description="Optional text prompt applied on the current frame.",
            optional=True,
        ),
    }
    OUTPUT_SPECS = {
        **SAM3TrackerInference.OUTPUT_SPECS,
        "category_ids": PortSpec(
            dtype=torch.int64,
            shape=(1, -1),
            description="Category IDs [1,N], aligned with object_ids.",
        ),
        "category_semantics": PortSpec(
            dtype=torch.uint8,
            shape=(-1,),
            description=(
                "UTF-8 JSON bytes of the cumulative category-id-to-text mapping, "
                'for example {"1":"person","2":"car"}.'
            ),
        ),
    }

    def __init__(self, **kwargs: Any) -> None:
        self._seed_source_stream_idx: int | None = None
        self._semantic_to_category_id: dict[str, int] = {}
        self._category_id_to_semantic: dict[int, str] = {}
        self._export_obj_id_to_category_id: dict[int, int] = {}
        self._next_category_id: int = 1
        self._current_prompt_category_id: int | None = None
        self._last_successful_prompt_category_id: int | None = None
        super().__init__(**kwargs)

    @property
    def _remap_internal_object_ids(self) -> bool:
        # Text prompts can return SAM internal object IDs starting at 0.
        # The pipeline label-map representation reserves 0 for background.
        return True

    def _apply_prompt(self) -> None:
        raise RuntimeError(
            "SAM3TextPropagation applies prompts from the runtime 'text_prompt' input."
        )

    @staticmethod
    def _normalize_runtime_text_prompt(text_prompt: str | None) -> str | None:
        if text_prompt is None:
            return None
        if not isinstance(text_prompt, str):
            raise ValueError(
                f"Expected runtime text_prompt to be a string, got {type(text_prompt).__name__}."
            )
        normalized = text_prompt.strip()
        return normalized or None

    def _reserve_category_id(self, prompt_text: str) -> int:
        category_id = self._semantic_to_category_id.get(prompt_text)
        if category_id is not None:
            return int(category_id)
        category_id = self._next_category_id
        self._semantic_to_category_id[prompt_text] = int(category_id)
        self._category_id_to_semantic[int(category_id)] = prompt_text
        self._next_category_id += 1
        return int(category_id)

    def _encode_category_semantics(self) -> torch.Tensor:
        payload = json.dumps(
            {
                str(category_id): semantic
                for category_id, semantic in sorted(self._category_id_to_semantic.items())
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return torch.tensor(list(payload), dtype=torch.uint8)

    def _empty_output(
        self,
        height: int = 1,
        width: int = 1,
    ) -> dict[str, torch.Tensor]:
        result = super()._empty_output(height, width)
        result["category_ids"] = torch.zeros(1, 0, dtype=torch.int64)
        result["category_semantics"] = self._encode_category_semantics()
        return result

    def _apply_runtime_text_prompt(
        self,
        *,
        prompt_text: str,
        frame_idx: int,
        reset_state: bool,
    ) -> dict | None:
        if self._inference_state is None:
            raise RuntimeError("Inference state must be initialized before applying a text prompt.")

        state = self._inference_state
        clamped_num_frames = int(frame_idx) + 1
        original_num_frames = int(state.get("num_frames", clamped_num_frames))
        original_tracker_num_frames: list[tuple[dict[str, Any], int]] = []

        state["num_frames"] = int(clamped_num_frames)
        for tracker_state in state.get("tracker_inference_states", []):
            if isinstance(tracker_state, dict):
                original_tracker_num_frames.append(
                    (tracker_state, int(tracker_state.get("num_frames", clamped_num_frames)))
                )
                tracker_state["num_frames"] = int(clamped_num_frames)

        try:
            with self._model_eval_context():
                _, postprocessed = self._model.add_prompt(
                    state,
                    frame_idx=int(frame_idx),
                    text_str=prompt_text,
                    reset_state=bool(reset_state),
                )
        finally:
            state["num_frames"] = int(original_num_frames)
            for tracker_state, original_tracker_num_frames_value in original_tracker_num_frames:
                tracker_state["num_frames"] = int(original_tracker_num_frames_value)

        state["action_history"].clear()
        return postprocessed

    def _category_id_for_new_export_track(self) -> int:
        category_id = self._current_prompt_category_id
        if category_id is None:
            category_id = self._last_successful_prompt_category_id
        if category_id is None:
            raise RuntimeError(
                "SAM3TextPropagation encountered a new exported track without any successful text "
                "prompt category in state."
            )
        return int(category_id)

    def _pack_output(
        self,
        postprocessed: dict | None,
        frame_shape: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        result = super()._pack_output(postprocessed, frame_shape=frame_shape)
        export_object_ids = [
            int(value) for value in result["object_ids"].reshape(-1).to(dtype=torch.int64).tolist()
        ]
        category_ids: list[int] = []
        for export_object_id in export_object_ids:
            category_id = self._export_obj_id_to_category_id.get(export_object_id)
            if category_id is None:
                category_id = self._category_id_for_new_export_track()
                self._export_obj_id_to_category_id[export_object_id] = int(category_id)
            category_ids.append(int(category_id))

        result["category_ids"] = torch.tensor([category_ids], dtype=torch.int64)
        result["category_semantics"] = self._encode_category_semantics()
        return result

    def forward(
        self,
        rgb_frame: torch.Tensor,
        text_prompt: str | None = None,
        frame_id: torch.Tensor | None = None,
        context: Context | None = None,  # noqa: ARG002
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Process one RGB frame with an optional runtime text prompt."""
        frame_np = rgb_frame[0].detach().cpu().numpy()
        frame_shape = (int(frame_np.shape[0]), int(frame_np.shape[1]))

        stream_idx = self._frame_idx
        source_frame_id = self._resolve_source_frame_id(frame_id, fallback_stream_idx=stream_idx)
        self._register_source_frame_id(stream_idx, source_frame_id)

        normalized_prompt = self._normalize_runtime_text_prompt(text_prompt)
        has_prompt = normalized_prompt is not None
        prompt_category_id: int | None = None
        if normalized_prompt is not None:
            prompt_category_id = self._reserve_category_id(normalized_prompt)

        if self._inference_state is not None or has_prompt:
            self._ensure_model()

        self._current_prompt_category_id = None
        internal_frame_idx: int | None = None

        if self._inference_state is None:
            if not has_prompt:
                self._frame_idx += 1
                return self._empty_output(*frame_shape)

            self._initialize_stream_state(frame_np)
            self._seed_source_stream_idx = stream_idx
            internal_frame_idx = 0
            self._mark_cudagraph_step_begin()
            self._apply_runtime_text_prompt(
                prompt_text=normalized_prompt,
                frame_idx=internal_frame_idx,
                reset_state=True,
            )
            self._current_prompt_category_id = int(prompt_category_id)
            self._last_successful_prompt_category_id = int(prompt_category_id)
            self._start_generator(start_frame_idx=internal_frame_idx)
        else:
            internal_frame_idx = self._frame_buffer.add(frame_np)
            self._extend_state_for_frame(internal_frame_idx)
            if has_prompt:
                self._mark_cudagraph_step_begin()
                self._apply_runtime_text_prompt(
                    prompt_text=normalized_prompt,
                    frame_idx=internal_frame_idx,
                    reset_state=False,
                )
                self._current_prompt_category_id = int(prompt_category_id)
                self._last_successful_prompt_category_id = int(prompt_category_id)
                self._start_generator(start_frame_idx=internal_frame_idx)

        if internal_frame_idx is None:
            raise RuntimeError("Internal frame index was not initialized for text propagation.")

        _yield_frame_idx, postprocessed = self._next_generator_output(internal_frame_idx)
        result = self._pack_output(postprocessed, frame_shape=frame_shape)
        self._current_prompt_category_id = None

        self._evict_excess_tracker_states(internal_frame_idx)
        self._prune_state_for_frame(internal_frame_idx)
        if torch.cuda.is_available() and internal_frame_idx > 0 and internal_frame_idx % 50 == 0:
            torch.cuda.empty_cache()

        self._frame_idx += 1
        return result


class SAM3BboxPropagation(SAM3TrackerInference):
    """SAM3 streaming propagation with runtime bbox prompts.

    The optional ``bboxes`` input carries a per-frame list of prompt dicts with
    ``element_id``, ``object_id``, ``x_min``, ``y_min``, ``x_max``, and ``y_max``.
    Frames before the first prompt emit empty outputs. On prompted frames, SAM3
    uses a temporary bbox-grounding pass to obtain prompt-frame masks, matches
    them back to the requested boxes by one-to-one highest-IoU assignment, and
    injects the resulting masks into the tracker under the requested
    ``object_id`` values.
    """

    INPUT_SPECS = {
        **SAM3TrackerInference.INPUT_SPECS,
        "bboxes": PortSpec(
            dtype=list,
            shape=(),
            description=(
                "Optional per-frame list of bbox prompt dicts with keys "
                "element_id, object_id, x_min, y_min, x_max, y_max."
            ),
            optional=True,
        ),
    }

    def __init__(self, **kwargs: Any) -> None:
        self._seed_source_stream_idx: int | None = None
        super().__init__(**kwargs)

    def _apply_prompt(self) -> None:
        raise RuntimeError("SAM3BboxPropagation applies prompts from the runtime 'bboxes' input.")

    @staticmethod
    def _normalize_runtime_bboxes(
        bboxes: list[dict[str, Any]] | None,
        expected_hw: tuple[int, int],
    ) -> list[dict[str, float | int]]:
        if bboxes is None:
            return []
        if not isinstance(bboxes, list):
            raise ValueError(
                f"Expected runtime bboxes to be a list of dicts, got {type(bboxes).__name__}."
            )

        height, width = int(expected_hw[0]), int(expected_hw[1])
        deduped_by_object_id: dict[int, dict[str, float | int]] = {}
        for idx, raw_box in enumerate(bboxes):
            if not isinstance(raw_box, dict):
                raise ValueError(
                    f"Expected bbox prompt at index {idx} to be a dict, got {type(raw_box).__name__}."
                )

            element_id = int(raw_box.get("element_id", 0))
            if element_id != 0:
                raise ValueError(
                    f"Runtime bbox prompt at index {idx} has element_id={element_id}; "
                    "single-frame SAM3 propagation expects element_id=0."
                )

            if "object_id" not in raw_box:
                raise ValueError(f"Runtime bbox prompt at index {idx} is missing 'object_id'.")
            object_id = int(raw_box["object_id"])
            if object_id <= 0:
                raise ValueError(
                    f"Runtime bbox prompt at index {idx} has invalid object_id={object_id}; "
                    "object_id must be > 0."
                )

            missing = [key for key in ("x_min", "y_min", "x_max", "y_max") if key not in raw_box]
            if missing:
                raise ValueError(
                    f"Runtime bbox prompt at index {idx} is missing required keys: {missing}."
                )

            x_min = float(raw_box["x_min"])
            y_min = float(raw_box["y_min"])
            x_max = float(raw_box["x_max"])
            y_max = float(raw_box["y_max"])
            if not (0.0 <= x_min < x_max <= float(width)):
                raise ValueError(
                    "Runtime bbox x-range is invalid for the current RGB frame: "
                    f"(x_min={x_min}, x_max={x_max}, width={width})."
                )
            if not (0.0 <= y_min < y_max <= float(height)):
                raise ValueError(
                    "Runtime bbox y-range is invalid for the current RGB frame: "
                    f"(y_min={y_min}, y_max={y_max}, height={height})."
                )

            deduped_by_object_id[object_id] = {
                "element_id": element_id,
                "object_id": object_id,
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
            }

        return list(deduped_by_object_id.values())

    @staticmethod
    def _prompt_box_xyxy(prompt_box: dict[str, float | int]) -> np.ndarray:
        return np.asarray(
            [
                float(prompt_box["x_min"]),
                float(prompt_box["y_min"]),
                float(prompt_box["x_max"]),
                float(prompt_box["y_max"]),
            ],
            dtype=np.float32,
        )

    @classmethod
    def _prompt_box_xywh_normalized(
        cls,
        prompt_box: dict[str, float | int],
        frame_shape: tuple[int, int],
    ) -> np.ndarray:
        height, width = int(frame_shape[0]), int(frame_shape[1])
        box_xyxy = cls._prompt_box_xyxy(prompt_box)
        x_min, y_min, x_max, y_max = [float(v) for v in box_xyxy.tolist()]
        return np.asarray(
            [
                x_min / width,
                y_min / height,
                (x_max - x_min) / width,
                (y_max - y_min) / height,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _postprocessed_boxes_xyxy(
        postprocessed: dict | None, frame_shape: tuple[int, int]
    ) -> np.ndarray:
        if postprocessed is None:
            return np.zeros((0, 4), dtype=np.float32)
        boxes_xywh = np.asarray(postprocessed.get("out_boxes_xywh", []), dtype=np.float32)
        if boxes_xywh.ndim != 2 or boxes_xywh.shape[-1] != 4:
            return np.zeros((0, 4), dtype=np.float32)
        height, width = int(frame_shape[0]), int(frame_shape[1])
        boxes_xyxy = np.array(boxes_xywh, copy=True, dtype=np.float32)
        boxes_xyxy[:, 0] *= width
        boxes_xyxy[:, 1] *= height
        boxes_xyxy[:, 2] = (boxes_xywh[:, 0] + boxes_xywh[:, 2]) * width
        boxes_xyxy[:, 3] = (boxes_xywh[:, 1] + boxes_xywh[:, 3]) * height
        return boxes_xyxy

    def _probe_prompt_frame(
        self,
        *,
        frame_np: np.ndarray,
        prompt_boxes: list[dict[str, float | int]],
        frame_shape: tuple[int, int],
    ) -> dict | None:
        device = self._model_device()
        probe_buffer = _FrameBuffer(
            image_size=int(self._model.image_size),
            device=device,
        )
        probe_buffer.add(frame_np)

        original_buffer = self._frame_buffer
        try:
            self._frame_buffer = probe_buffer
            probe_state = self._build_state(int(frame_shape[0]), int(frame_shape[1]))
            self._extend_state_for_frame(0, state=probe_state)
        finally:
            self._frame_buffer = original_buffer

        boxes_xywh = np.stack(
            [self._prompt_box_xywh_normalized(prompt, frame_shape) for prompt in prompt_boxes],
            axis=0,
        )
        labels = torch.ones(len(prompt_boxes), dtype=torch.long, device=device)
        with self._model_eval_context():
            _, postprocessed = self._model.add_prompt(
                probe_state,
                frame_idx=0,
                boxes_xywh=boxes_xywh,
                box_labels=labels,
            )
        return postprocessed

    @classmethod
    def _match_prompt_masks(
        cls,
        *,
        prompt_boxes: list[dict[str, float | int]],
        postprocessed: dict | None,
        frame_shape: tuple[int, int],
    ) -> list[np.ndarray | None]:
        matches: list[np.ndarray | None] = [None] * len(prompt_boxes)
        if postprocessed is None:
            return matches

        candidate_masks = np.asarray(postprocessed.get("out_binary_masks", []), dtype=bool)
        if candidate_masks.ndim != 3 or candidate_masks.shape[0] == 0:
            return matches

        candidate_boxes = cls._postprocessed_boxes_xyxy(postprocessed, frame_shape)
        if candidate_boxes.shape[0] != candidate_masks.shape[0]:
            return matches

        scores: list[tuple[float, int, int]] = []
        for prompt_idx, prompt_box in enumerate(prompt_boxes):
            prompt_xyxy = cls._prompt_box_xyxy(prompt_box)
            for candidate_idx, candidate_xyxy in enumerate(candidate_boxes):
                scores.append(
                    (
                        _bbox_iou_xyxy(prompt_xyxy, candidate_xyxy),
                        prompt_idx,
                        candidate_idx,
                    )
                )

        assigned_prompts: set[int] = set()
        assigned_candidates: set[int] = set()
        for _iou, prompt_idx, candidate_idx in sorted(scores, reverse=True):
            if prompt_idx in assigned_prompts or candidate_idx in assigned_candidates:
                continue
            assigned_prompts.add(prompt_idx)
            assigned_candidates.add(candidate_idx)
            matches[prompt_idx] = np.asarray(candidate_masks[candidate_idx], dtype=np.uint8)

        return matches

    def _apply_runtime_bboxes(
        self,
        *,
        frame_np: np.ndarray,
        prompt_boxes: list[dict[str, float | int]],
        frame_shape: tuple[int, int],
        frame_idx: int,
    ) -> None:
        postprocessed = self._probe_prompt_frame(
            frame_np=frame_np,
            prompt_boxes=prompt_boxes,
            frame_shape=frame_shape,
        )
        matched_masks = self._match_prompt_masks(
            prompt_boxes=prompt_boxes,
            postprocessed=postprocessed,
            frame_shape=frame_shape,
        )

        for prompt_box, matched_mask in zip(prompt_boxes, matched_masks, strict=False):
            if matched_mask is None or int(np.count_nonzero(matched_mask)) == 0:
                matched_mask = _binary_mask_from_xyxy(
                    x_min=float(prompt_box["x_min"]),
                    y_min=float(prompt_box["y_min"]),
                    x_max=float(prompt_box["x_max"]),
                    y_max=float(prompt_box["y_max"]),
                    frame_shape=frame_shape,
                )
            self._inject_mask_prompt_for_object(
                np.asarray(matched_mask, dtype=np.uint8),
                frame_idx=frame_idx,
                obj_id=int(prompt_box["object_id"]),
            )

        self._inference_state["action_history"].clear()

    def forward(
        self,
        rgb_frame: torch.Tensor,
        bboxes: list[dict[str, Any]] | None = None,
        frame_id: torch.Tensor | None = None,
        context: Context | None = None,  # noqa: ARG002
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Process one RGB frame with optional runtime bbox prompts."""
        frame_np = rgb_frame[0].detach().cpu().numpy()
        frame_shape = (int(frame_np.shape[0]), int(frame_np.shape[1]))

        stream_idx = self._frame_idx
        source_frame_id = self._resolve_source_frame_id(frame_id, fallback_stream_idx=stream_idx)
        self._register_source_frame_id(stream_idx, source_frame_id)

        prompt_boxes = self._normalize_runtime_bboxes(bboxes, expected_hw=frame_shape)
        has_prompt = len(prompt_boxes) > 0
        internal_frame_idx: int | None = None

        if self._inference_state is None:
            if not has_prompt:
                self._frame_idx += 1
                return self._empty_output(*frame_shape)

            self._ensure_model()
            self._initialize_stream_state(frame_np)
            self._seed_source_stream_idx = stream_idx
            internal_frame_idx = 0
            self._mark_cudagraph_step_begin()
            self._apply_runtime_bboxes(
                frame_np=frame_np,
                prompt_boxes=prompt_boxes,
                frame_shape=frame_shape,
                frame_idx=internal_frame_idx,
            )
            self._start_generator(start_frame_idx=internal_frame_idx)
        else:
            internal_frame_idx = self._frame_buffer.add(frame_np)
            self._extend_state_for_frame(internal_frame_idx)
            if has_prompt:
                self._mark_cudagraph_step_begin()
                self._apply_runtime_bboxes(
                    frame_np=frame_np,
                    prompt_boxes=prompt_boxes,
                    frame_shape=frame_shape,
                    frame_idx=internal_frame_idx,
                )

        if internal_frame_idx is None:
            raise RuntimeError("Internal frame index was not initialized for bbox propagation.")

        _yield_frame_idx, postprocessed = self._next_generator_output(internal_frame_idx)
        result = self._pack_output(postprocessed, frame_shape=frame_shape)

        self._evict_excess_tracker_states(internal_frame_idx)
        self._prune_state_for_frame(internal_frame_idx)
        if torch.cuda.is_available() and internal_frame_idx > 0 and internal_frame_idx % 50 == 0:
            torch.cuda.empty_cache()

        self._frame_idx += 1
        return result


class SAM3PointPropagation(SAM3TrackerInference):
    """SAM3 streaming propagation with a point prompt.

    Tracks a single object specified by click points and ``prompt_obj_id``.
    """

    def __init__(
        self,
        prompt_points: list[list[float]],
        prompt_point_labels: list[int],
        prompt_obj_id: int,
        **kwargs: Any,
    ) -> None:
        self._prompt_points = prompt_points
        self._prompt_point_labels = prompt_point_labels
        self._prompt_obj_id = prompt_obj_id
        super().__init__(
            prompt_points=prompt_points,
            prompt_point_labels=prompt_point_labels,
            prompt_obj_id=prompt_obj_id,
            **kwargs,
        )

    @property
    def _requires_cached_frame_outputs_on_prompt_frame(self) -> bool:
        return True

    def _apply_prompt(self) -> None:
        device = self._model_device()
        points = torch.tensor(self._prompt_points, dtype=torch.float32, device=device)
        point_labels = torch.tensor(self._prompt_point_labels, dtype=torch.int64, device=device)
        with self._model_eval_context():
            self._model.add_prompt(
                self._inference_state,
                frame_idx=0,
                points=points,
                point_labels=point_labels,
                obj_id=self._prompt_obj_id,
            )
        # Streaming starts from a fresh state with no cached base predictions.
        # Force a full propagation pass instead of tracker-only partial propagation.
        self._inference_state["action_history"].clear()


class SAM3MaskPropagation(SAM3TrackerInference):
    """SAM3 streaming propagation with runtime label-map prompts.

    The optional ``mask`` input is an int32 label map shaped ``[1, H, W]``:
    ``0`` is background and each positive label value is treated as an object ID.
    Propagation starts on the first frame where a non-empty prompt mask arrives.
    An optional runtime ``text_prompt`` can provide semantic context while the
    mask is injected, without switching the stream into text-driven detection.
    """

    INPUT_SPECS = {
        **SAM3TrackerInference.INPUT_SPECS,
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(1, -1, -1),
            description=(
                "Optional int32 label map [1,H,W]. 0=background, each positive "
                "label is treated as an object ID prompt on that frame."
            ),
            optional=True,
        ),
        "text_prompt": PortSpec(
            dtype=str,
            shape=(),
            description=(
                "Optional text description applied only while injecting the runtime "
                "mask prompt on the current frame."
            ),
            optional=True,
        ),
    }

    def __init__(self, **kwargs: Any) -> None:
        self._seed_source_stream_idx: int | None = None
        super().__init__(**kwargs)

    @property
    def _requires_cached_frame_outputs_on_prompt_frame(self) -> bool:
        return True

    def _apply_prompt(self) -> None:
        raise RuntimeError("SAM3MaskPropagation applies prompts from the runtime 'mask' input.")

    @staticmethod
    def _normalize_runtime_text_prompt(text_prompt: str | None) -> str | None:
        if text_prompt is None:
            return None
        if not isinstance(text_prompt, str):
            raise ValueError(
                f"Expected runtime text_prompt to be a string, got {type(text_prompt).__name__}."
            )
        normalized = text_prompt.strip()
        return normalized or None

    @staticmethod
    def _normalize_runtime_mask(
        mask: torch.Tensor | None,
        expected_hw: tuple[int, int],
    ) -> np.ndarray | None:
        if mask is None:
            return None
        if mask.ndim != 3 or int(mask.shape[0]) != 1:
            raise ValueError(
                f"Expected runtime mask shape [1,H,W], got {tuple(int(v) for v in mask.shape)}."
            )
        expected_h, expected_w = int(expected_hw[0]), int(expected_hw[1])
        actual_h, actual_w = int(mask.shape[1]), int(mask.shape[2])
        if (actual_h, actual_w) != (expected_h, expected_w):
            raise ValueError(
                "Runtime mask shape does not match current RGB frame: "
                f"mask={(actual_h, actual_w)}, rgb={(expected_h, expected_w)}."
            )
        return np.asarray(mask[0].detach().cpu().numpy(), dtype=np.int64)

    def _push_runtime_text_context(self, text_prompt: str | None) -> dict[str, Any] | None:
        normalized_prompt = self._normalize_runtime_text_prompt(text_prompt)
        if normalized_prompt is None:
            return None
        if self._inference_state is None:
            raise RuntimeError("Inference state must be initialized before applying text context.")

        input_batch = self._inference_state["input_batch"]
        text_id_raw = getattr(self._model, "TEXT_ID_FOR_TEXT", 0)
        text_id = int(text_id_raw) if isinstance(text_id_raw, (int, np.integer)) else 0
        snapshot = {
            "find_text_batch_0": input_batch.find_text_batch[0],
            "text_ids": {
                int(frame_idx): stage.text_ids.clone()
                for frame_idx, stage in input_batch.find_inputs.items()
            },
        }

        input_batch.find_text_batch[0] = normalized_prompt
        for stage in input_batch.find_inputs.values():
            stage.text_ids[...] = text_id
        return snapshot

    def _restore_runtime_text_context(self, snapshot: dict[str, Any] | None) -> None:
        if snapshot is None or self._inference_state is None:
            return

        input_batch = self._inference_state["input_batch"]
        input_batch.find_text_batch[0] = snapshot["find_text_batch_0"]
        for frame_idx, saved_text_ids in snapshot["text_ids"].items():
            stage = input_batch.find_inputs.get(int(frame_idx))
            if stage is not None:
                stage.text_ids[...] = saved_text_ids

    def _apply_runtime_mask(
        self,
        label_map: np.ndarray,
        frame_idx: int,
        text_prompt: str | None = None,
    ) -> None:
        text_snapshot = self._push_runtime_text_context(text_prompt)
        try:
            positive_labels = [
                int(label) for label in np.unique(label_map).tolist() if int(label) > 0
            ]
            for obj_id in positive_labels:
                self._inject_mask_prompt_for_object(
                    label_map == obj_id,
                    frame_idx=int(frame_idx),
                    obj_id=int(obj_id),
                )
            # Force regular propagation after the initial mask add.
            # Tracker partial propagation for mask-only prompts can require point inputs.
            self._inference_state["action_history"].clear()
        finally:
            self._restore_runtime_text_context(text_snapshot)

    def forward(
        self,
        rgb_frame: torch.Tensor,
        mask: torch.Tensor | None = None,
        text_prompt: str | None = None,
        frame_id: torch.Tensor | None = None,
        context: Context | None = None,  # noqa: ARG002
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Process one RGB frame with optional runtime mask prompts."""
        frame_np = rgb_frame[0].detach().cpu().numpy()
        frame_shape = (int(frame_np.shape[0]), int(frame_np.shape[1]))

        stream_idx = self._frame_idx
        source_frame_id = self._resolve_source_frame_id(frame_id, fallback_stream_idx=stream_idx)
        self._register_source_frame_id(stream_idx, source_frame_id)

        label_map = self._normalize_runtime_mask(mask, expected_hw=frame_shape)
        has_prompt = label_map is not None and bool(np.any(label_map > 0))
        internal_frame_idx: int | None = None

        if self._inference_state is None:
            if not has_prompt:
                self._frame_idx += 1
                return self._empty_output(*frame_shape)

            self._ensure_model()
            self._initialize_stream_state(frame_np)
            self._seed_source_stream_idx = stream_idx
            internal_frame_idx = 0
            self._mark_cudagraph_step_begin()
            self._apply_runtime_mask(
                label_map,
                frame_idx=internal_frame_idx,
                text_prompt=text_prompt,
            )
            self._start_generator(start_frame_idx=internal_frame_idx)
        else:
            internal_frame_idx = self._frame_buffer.add(frame_np)
            self._extend_state_for_frame(internal_frame_idx)
            if has_prompt:
                self._mark_cudagraph_step_begin()
                self._apply_runtime_mask(
                    label_map,
                    frame_idx=internal_frame_idx,
                    text_prompt=text_prompt,
                )

        if internal_frame_idx is None:
            raise RuntimeError("Internal frame index was not initialized for mask propagation.")

        _yield_frame_idx, postprocessed = self._next_generator_output(internal_frame_idx)
        result = self._pack_output(postprocessed, frame_shape=frame_shape)

        self._evict_excess_tracker_states(internal_frame_idx)
        self._prune_state_for_frame(internal_frame_idx)
        if torch.cuda.is_available() and internal_frame_idx > 0 and internal_frame_idx % 50 == 0:
            torch.cuda.empty_cache()

        self._frame_idx += 1
        return result


__all__ = [
    "SAM3TrackerInference",
    "SAM3TextPropagation",
    "SAM3BboxPropagation",
    "SAM3PointPropagation",
    "SAM3MaskPropagation",
]
