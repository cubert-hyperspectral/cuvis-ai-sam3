"""SAM3 streaming propagation nodes — one RGB frame per forward(), specialized by prompt type.

Hierarchy:
    SAM3StreamingPropagationBase(Node)   — abstract base with shared infrastructure
    ├── SAM3TextPropagation              — text/concept prompt
    ├── SAM3BboxPropagation              — bounding-box prompt
    ├── SAM3PointPropagation             — point prompt
    └── SAM3MaskPropagation              — binary mask prompt
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from loguru import logger

from cuvis_ai_core.node import Node
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PortSpec


class _FrameBuffer:
    """Pre-allocated frame buffer that implements SAM3's ``img_batch`` protocol.

    Receives RGB frames from the pipeline one at a time.  SAM3's engine reads
    ``img_batch[frame_idx]`` during propagation — this object returns the
    pre-processed tensor for the requested index.
    """

    def __init__(
        self,
        num_frames: int,
        image_size: int,
        device: torch.device,
        image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> None:
        self._num_frames = int(num_frames)
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
            raise ValueError(
                f"Expected frame shape [H, W, 3], got {tuple(frame_float_hwc.shape)}."
            )
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
        return self._num_frames

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


class SAM3StreamingPropagationBase(Node):
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
        num_frames: int,
        checkpoint_path: str | None = None,
        compile_model: bool = False,
        prompt_frame_idx: int = 0,
        prompt_frame_id: int | None = None,
        input_frame_id_offset: int = 0,
        # -- SAM3 thresholds --
        score_threshold_detection: float = 0.5,
        new_det_thresh: float = 0.7,
        det_nms_thresh: float = 0.1,
        overlap_suppress_thresh: float = 0.7,
        max_tracker_states: int = 5,
        progress_log_interval: int = 50,
        **kwargs: Any,
    ) -> None:
        if num_frames < 1:
            raise ValueError(f"num_frames must be >= 1, got {num_frames}.")
        if prompt_frame_id is None:
            if prompt_frame_idx < 0 or prompt_frame_idx >= num_frames:
                raise ValueError(
                    f"prompt_frame_idx must be in [0, {num_frames - 1}], got {prompt_frame_idx}."
                )
        elif prompt_frame_id < 0:
            raise ValueError(f"prompt_frame_id must be >= 0, got {prompt_frame_id}.")

        self._num_frames = int(num_frames)
        self._checkpoint_path = checkpoint_path
        self._compile_model = compile_model
        self._prompt_frame_idx = int(prompt_frame_idx)
        self._prompt_frame_id = int(prompt_frame_id) if prompt_frame_id is not None else None
        self._input_frame_id_offset = int(input_frame_id_offset)
        self._resolved_prompt_frame_idx: int | None = None
        self._score_threshold_detection = score_threshold_detection
        self._new_det_thresh = new_det_thresh
        self._det_nms_thresh = det_nms_thresh
        self._overlap_suppress_thresh = overlap_suppress_thresh
        self._max_tracker_states = max_tracker_states
        self._progress_log_interval = progress_log_interval

        # Runtime state (initialized on first forward)
        self._model: Any = None
        self._frame_buffer: _FrameBuffer | None = None
        self._inference_state: dict[str, Any] | None = None
        self._generator: Any = None
        self._frame_idx: int = 0
        self._source_frame_ids: list[int] = []
        self._frame_id_to_stream_idx: dict[int, int] = {}

        super().__init__(
            num_frames=num_frames,
            checkpoint_path=checkpoint_path,
            compile_model=compile_model,
            prompt_frame_idx=prompt_frame_idx,
            prompt_frame_id=prompt_frame_id,
            input_frame_id_offset=input_frame_id_offset,
            score_threshold_detection=score_threshold_detection,
            new_det_thresh=new_det_thresh,
            det_nms_thresh=det_nms_thresh,
            overlap_suppress_thresh=overlap_suppress_thresh,
            max_tracker_states=max_tracker_states,
            progress_log_interval=progress_log_interval,
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
                buffered_frames = max(1, min(int(self._frame_buffer._next_idx), int(requested_num_frames)))
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

    def _extend_state_for_frame(self, frame_idx: int) -> None:
        """Register a new frame index in the inference state (dict-based, like v1)."""
        from sam3.model.data_misc import FindStage, convert_my_tensors
        from sam3.model.utils.misc import copy_data_to_device

        state = self._inference_state
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
        state["num_frames"] = frame_idx + 1
        for tracker_state in state.get("tracker_inference_states", []):
            if isinstance(tracker_state, dict):
                tracker_state["num_frames"] = frame_idx + 1

    def _prepare_state_for_full_sequence(self) -> None:
        """Register all frames so propagation sees the full sequence length."""
        state = self._inference_state
        for frame_idx in range(self._num_frames):
            self._extend_state_for_frame(frame_idx)
            state["cached_frame_outputs"].setdefault(frame_idx, {})

    def _next_generator_output(self, requested_frame_idx: int) -> tuple[int, dict | None]:
        """Advance the SAM3 propagation generator with a clear error on early exhaustion."""
        if self._generator is None:
            raise RuntimeError("Propagation generator is not initialized.")
        try:
            return next(self._generator)
        except StopIteration as exc:
            raise RuntimeError(
                f"{self.__class__.__name__} generator exhausted early at frame "
                f"{requested_frame_idx} (configured num_frames={self._num_frames})."
            ) from exc

    def _resolve_source_frame_id(
        self,
        frame_id: torch.Tensor | None,
        fallback_stream_idx: int,
    ) -> int:
        if frame_id is None or frame_id.numel() == 0:
            return int(fallback_stream_idx + self._input_frame_id_offset)
        return int(frame_id.reshape(-1)[0].item())

    def _register_frame_id_mapping(self, stream_idx: int, source_frame_id: int) -> None:
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

        existing = self._frame_id_to_stream_idx.get(int(source_frame_id))
        if existing is None:
            self._frame_id_to_stream_idx[int(source_frame_id)] = int(stream_idx)
        elif existing != int(stream_idx):
            logger.warning(
                "Duplicate source frame_id {} seen at local indices {} and {}. Using first mapping.",
                source_frame_id,
                existing,
                stream_idx,
            )

    def _source_frame_id_for_stream_idx(self, stream_idx: int) -> int:
        if 0 <= stream_idx < len(self._source_frame_ids):
            return int(self._source_frame_ids[stream_idx])
        return int(stream_idx)

    def _maybe_resolve_prompt_frame_idx_from_source_id(self) -> int | None:
        if self._prompt_frame_id is None:
            if self._resolved_prompt_frame_idx is None:
                self._resolved_prompt_frame_idx = int(self._prompt_frame_idx)
            return self._resolved_prompt_frame_idx

        if self._resolved_prompt_frame_idx is not None:
            return self._resolved_prompt_frame_idx

        resolved = self._frame_id_to_stream_idx.get(int(self._prompt_frame_id))
        if resolved is None:
            return None

        self._resolved_prompt_frame_idx = int(resolved)
        logger.info(
            "{}: resolved prompt_frame_id {} -> local prompt index {}",
            self.__class__.__name__,
            self._prompt_frame_id,
            self._resolved_prompt_frame_idx,
        )
        return self._resolved_prompt_frame_idx

    def _prompt_idx_for_model(self) -> int:
        prompt_idx = self._maybe_resolve_prompt_frame_idx_from_source_id()
        if prompt_idx is None:
            raise RuntimeError(
                "Prompt frame has not been observed yet. "
                f"Expected source prompt_frame_id={self._prompt_frame_id}."
            )
        return int(prompt_idx)

    @staticmethod
    def _empty_output() -> dict[str, torch.Tensor]:
        return {
            "mask": torch.zeros(1, 1, 1, dtype=torch.int32),
            "object_ids": torch.zeros(1, 0, dtype=torch.int64),
            "detection_scores": torch.zeros(1, 0, dtype=torch.float32),
        }

    # -- Prompt application (subclass responsibility) -------------------------

    @abstractmethod
    def _apply_prompt(self) -> None:
        """Apply the configured prompt to ``prompt_frame_idx`` of the inference state."""

    # -- Output conversion ----------------------------------------------------

    @property
    def _requires_cached_frame_outputs_on_prompt_frame(self) -> bool:
        """Whether cached_frame_outputs must be pre-seeded on the prompt frame.

        Point and mask prompts require this; text and bbox do not.
        Override in subclasses that need it.
        """
        return False

    def _filter_objects(
        self,
        obj_ids: np.ndarray,
        binary_masks: np.ndarray,
        probs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Hook for subclasses to filter/remap detected objects. Default: pass-through."""
        return obj_ids, binary_masks, probs

    def _pack_output(
        self,
        postprocessed: dict | None,
    ) -> dict[str, torch.Tensor]:
        """Convert SAM3 per-object output to per-frame label map format."""
        if postprocessed is None or len(postprocessed.get("out_obj_ids", [])) == 0:
            return self._empty_output()

        obj_ids = np.asarray(postprocessed["out_obj_ids"], dtype=np.int64)
        binary_masks = np.asarray(postprocessed["out_binary_masks"], dtype=bool)
        probs = np.asarray(postprocessed["out_probs"], dtype=np.float32)

        obj_ids, binary_masks, probs = self._filter_objects(obj_ids, binary_masks, probs)
        if obj_ids.shape[0] == 0:
            return self._empty_output()

        h, w = binary_masks.shape[1], binary_masks.shape[2]
        label_map = np.zeros((h, w), dtype=np.int32)
        for oid, m in zip(obj_ids, binary_masks):
            label_map[m] = int(oid)

        return {
            "mask": torch.from_numpy(label_map).unsqueeze(0),
            "object_ids": torch.from_numpy(np.array(obj_ids, dtype=np.int64)).unsqueeze(0),
            "detection_scores": torch.from_numpy(np.array(probs, dtype=np.float32)).unsqueeze(0),
        }

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
        stream_idx = self._frame_idx
        source_frame_id = self._resolve_source_frame_id(frame_id, fallback_stream_idx=stream_idx)
        self._register_frame_id_mapping(stream_idx, source_frame_id)
        prompt_frame_idx = self._maybe_resolve_prompt_frame_idx_from_source_id()

        if stream_idx == 0:
            # -- First frame: initialize state and buffer --
            orig_h, orig_w = frame_np.shape[0], frame_np.shape[1]
            device = self._model_device()

            self._frame_buffer = _FrameBuffer(
                num_frames=self._num_frames,
                image_size=int(self._model.image_size),
                device=device,
            )
            self._frame_buffer.add(frame_np)

            self._inference_state = self._build_state(orig_h, orig_w)
            self._extend_state_for_frame(0)
        else:
            # -- Subsequent frames: buffer + extend state --
            self._frame_buffer.add(frame_np)
            self._extend_state_for_frame(stream_idx)

        if self._generator is None and prompt_frame_idx is not None and stream_idx >= prompt_frame_idx:
            self._prepare_state_for_full_sequence()
            if self._requires_cached_frame_outputs_on_prompt_frame:
                # Interactivity prompt paths require cached outputs on the prompted frame.
                self._inference_state["cached_frame_outputs"].setdefault(prompt_frame_idx, {})
            self._apply_prompt()

            # Start one generator, potentially beginning at a non-zero prompt frame.
            self._generator = self._model.propagate_in_video(
                self._inference_state,
                start_frame_idx=prompt_frame_idx,
                max_frame_num_to_track=self._num_frames - prompt_frame_idx - 1,
                reverse=False,
            )
            _yield_frame_idx, postprocessed = self._next_generator_output(stream_idx)
            result = self._pack_output(postprocessed)
        else:
            if self._generator is None:
                # Frames before the prompt frame: keep outputs empty.
                result = self._pack_output(None)
            else:
                # Frames after prompt: advance the running generator.
                _yield_frame_idx, postprocessed = self._next_generator_output(stream_idx)
                result = self._pack_output(postprocessed)

        self._frame_idx += 1

        if self._progress_log_interval > 0 and stream_idx % self._progress_log_interval == 0:
            n_objs = result["object_ids"].shape[1]
            logger.info(
                "{}: local frame {}/{}, source frame {}, {} objects",
                self.__class__.__name__,
                stream_idx,
                self._num_frames,
                source_frame_id,
                n_objs,
            )

        return result


# =============================================================================
# Concrete subclasses
# =============================================================================


class SAM3TextPropagation(SAM3StreamingPropagationBase):
    """SAM3 streaming propagation with a text/concept prompt.

    Detects and tracks all objects matching ``prompt_text`` (e.g. "person").
    """

    def __init__(
        self,
        num_frames: int,
        prompt_text: str = "person",
        **kwargs: Any,
    ) -> None:
        self._prompt_text = prompt_text
        super().__init__(num_frames=num_frames, prompt_text=prompt_text, **kwargs)

    def _apply_prompt(self) -> None:
        prompt_idx = self._prompt_idx_for_model()
        self._model.add_prompt(
            self._inference_state,
            frame_idx=prompt_idx,
            text_str=self._prompt_text,
        )


class SAM3BboxPropagation(SAM3StreamingPropagationBase):
    """SAM3 streaming propagation with a bounding-box prompt.

    Tracks a single object initialized by a bounding box. The best-matching
    SAM object is selected via IoU. If ``prompt_obj_id`` is provided, emitted
    outputs use that ID; otherwise the selected SAM internal ID is used.
    """

    def __init__(
        self,
        num_frames: int,
        prompt_bboxes_xywh: list[list[float]],
        prompt_bbox_labels: list[int] | None = None,
        prompt_obj_id: int | None = None,
        **kwargs: Any,
    ) -> None:
        if prompt_bboxes_xywh and len(prompt_bboxes_xywh) > 1:
            raise ValueError(
                "SAM3BboxPropagation currently supports exactly one initial bbox prompt. "
                "Provide a single bbox prompt only."
            )
        self._prompt_bboxes_xywh = prompt_bboxes_xywh
        self._prompt_bbox_labels = prompt_bbox_labels
        self._prompt_obj_id = prompt_obj_id
        self._selected_internal_bbox_obj_id: int | None = None
        self._effective_output_bbox_obj_id: int | None = None
        super().__init__(
            num_frames=num_frames,
            prompt_bboxes_xywh=prompt_bboxes_xywh,
            prompt_bbox_labels=prompt_bbox_labels,
            prompt_obj_id=prompt_obj_id,
            **kwargs,
        )

    @staticmethod
    def _bbox_iou_xywh(box_a: np.ndarray, box_b: np.ndarray) -> float:
        ax1, ay1, aw, ah = [float(v) for v in box_a.tolist()]
        bx1, by1, bw, bh = [float(v) for v in box_b.tolist()]
        if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
            return 0.0
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter = inter_w * inter_h

        area_a = aw * ah
        area_b = bw * bh
        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return inter / union

    def _select_bbox_object_id(
        self,
        *,
        prompt_box_xywh: np.ndarray,
        postprocessed: dict | None,
    ) -> int | None:
        if postprocessed is None:
            return None
        raw_obj_ids = postprocessed.get("out_obj_ids")
        if raw_obj_ids is None or len(raw_obj_ids) == 0:
            return None

        obj_ids = np.asarray(raw_obj_ids, dtype=np.int64)
        raw_boxes = postprocessed.get("out_boxes_xywh")
        if raw_boxes is None:
            return int(obj_ids[0])
        boxes = np.asarray(raw_boxes, dtype=np.float32)
        if boxes.ndim != 2 or boxes.shape[0] != obj_ids.shape[0] or boxes.shape[1] != 4:
            return int(obj_ids[0])

        best_idx = 0
        best_iou = -1.0
        for idx in range(obj_ids.shape[0]):
            iou = self._bbox_iou_xywh(prompt_box_xywh, boxes[idx])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        return int(obj_ids[best_idx])

    def _apply_prompt(self) -> None:
        if not self._prompt_bboxes_xywh:
            raise ValueError("prompt_bboxes_xywh required for SAM3BboxPropagation.")
        prompt_idx = self._prompt_idx_for_model()
        device = self._model_device()
        boxes = np.array(self._prompt_bboxes_xywh, dtype=np.float32)
        labels = (
            torch.tensor(self._prompt_bbox_labels, dtype=torch.long, device=device)
            if self._prompt_bbox_labels
            else torch.ones(len(boxes), dtype=torch.long, device=device)
        )
        _, postprocessed = self._model.add_prompt(
            self._inference_state,
            frame_idx=prompt_idx,
            boxes_xywh=boxes,
            box_labels=labels,
        )
        selected_internal = self._select_bbox_object_id(
            prompt_box_xywh=boxes[0],
            postprocessed=postprocessed,
        )
        self._selected_internal_bbox_obj_id = selected_internal
        if selected_internal is None:
            self._effective_output_bbox_obj_id = None
            logger.warning(
                "bbox prompt produced no selectable object on frame {}",
                prompt_idx,
            )
        else:
            self._effective_output_bbox_obj_id = (
                self._prompt_obj_id if self._prompt_obj_id is not None else selected_internal
            )
            logger.info(
                "bbox selected internal SAM id {} -> output id {}",
                selected_internal,
                self._effective_output_bbox_obj_id,
            )

    def _filter_objects(
        self,
        obj_ids: np.ndarray,
        binary_masks: np.ndarray,
        probs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._selected_internal_bbox_obj_id is None:
            return (
                np.array([], dtype=np.int64),
                np.zeros((0,) + binary_masks.shape[1:], dtype=bool),
                np.array([], dtype=np.float32),
            )
        selected_matches = np.nonzero(obj_ids == int(self._selected_internal_bbox_obj_id))[0]
        if selected_matches.size == 0:
            return (
                np.array([], dtype=np.int64),
                np.zeros((0,) + binary_masks.shape[1:], dtype=bool),
                np.array([], dtype=np.float32),
            )
        sel_idx = int(selected_matches[0])
        out_id = (
            int(self._effective_output_bbox_obj_id)
            if self._effective_output_bbox_obj_id is not None
            else int(obj_ids[sel_idx])
        )
        return (
            np.array([out_id], dtype=np.int64),
            binary_masks[sel_idx : sel_idx + 1],
            probs[sel_idx : sel_idx + 1],
        )


class SAM3PointPropagation(SAM3StreamingPropagationBase):
    """SAM3 streaming propagation with a point prompt.

    Tracks a single object specified by click points and ``prompt_obj_id``.
    """

    def __init__(
        self,
        num_frames: int,
        prompt_points: list[list[float]],
        prompt_point_labels: list[int],
        prompt_obj_id: int,
        **kwargs: Any,
    ) -> None:
        self._prompt_points = prompt_points
        self._prompt_point_labels = prompt_point_labels
        self._prompt_obj_id = prompt_obj_id
        super().__init__(
            num_frames=num_frames,
            prompt_points=prompt_points,
            prompt_point_labels=prompt_point_labels,
            prompt_obj_id=prompt_obj_id,
            **kwargs,
        )

    @property
    def _requires_cached_frame_outputs_on_prompt_frame(self) -> bool:
        return True

    def _apply_prompt(self) -> None:
        prompt_idx = self._prompt_idx_for_model()
        device = self._model_device()
        points = torch.tensor(self._prompt_points, dtype=torch.float32, device=device)
        point_labels = torch.tensor(self._prompt_point_labels, dtype=torch.int64, device=device)
        self._model.add_prompt(
            self._inference_state,
            frame_idx=prompt_idx,
            points=points,
            point_labels=point_labels,
            obj_id=self._prompt_obj_id,
        )


class SAM3MaskPropagation(SAM3StreamingPropagationBase):
    """SAM3 streaming propagation with a binary mask prompt.

    Loads a mask from ``prompt_mask_path``, adds it with a centroid point
    prompt, and tracks the object specified by ``prompt_obj_id``.
    """

    def __init__(
        self,
        num_frames: int,
        prompt_mask_path: str,
        prompt_obj_id: int,
        **kwargs: Any,
    ) -> None:
        self._prompt_mask_path = prompt_mask_path
        self._prompt_obj_id = prompt_obj_id
        super().__init__(
            num_frames=num_frames,
            prompt_mask_path=prompt_mask_path,
            prompt_obj_id=prompt_obj_id,
            **kwargs,
        )

    @property
    def _requires_cached_frame_outputs_on_prompt_frame(self) -> bool:
        return True

    def _apply_prompt(self) -> None:
        prompt_idx = self._prompt_idx_for_model()
        device = self._model_device()
        mask_gray = cv2.imread(self._prompt_mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            raise FileNotFoundError(f"Cannot load mask: {self._prompt_mask_path}")
        mask_binary = (mask_gray > 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_binary).to(device=device, dtype=torch.float32)
        self._model.add_mask(
            self._inference_state,
            frame_idx=prompt_idx,
            obj_id=self._prompt_obj_id,
            mask=mask_tensor,
        )
        ys, xs = np.where(mask_binary > 0)
        if xs.size > 0 and ys.size > 0:
            h_mask, w_mask = mask_binary.shape
            point_x = float(xs.mean() / w_mask)
            point_y = float(ys.mean() / h_mask)
            self._model.add_prompt(
                self._inference_state,
                frame_idx=prompt_idx,
                points=torch.tensor([[point_x, point_y]], dtype=torch.float32, device=device),
                point_labels=torch.tensor([1], dtype=torch.int64, device=device),
                obj_id=self._prompt_obj_id,
            )
        # Force regular propagation after the initial mask add.
        # Tracker partial propagation for mask-only prompts can require point inputs.
        self._inference_state["action_history"].clear()
