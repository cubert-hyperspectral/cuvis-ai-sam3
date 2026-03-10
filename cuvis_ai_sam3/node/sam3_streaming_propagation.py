"""SAM3 streaming propagation node — one RGB frame per forward(), all prompt types."""

from __future__ import annotations

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


class SAM3StreamingPropagation(Node):
    """Streaming SAM3 propagation node — consumes one RGB frame per ``forward()``.

    Accepts prompt configuration via constructor params. On the first frame it
    initializes state/buffers. Once ``prompt_frame_idx`` is reached, it applies the
    prompt and starts a single ``propagate_in_video`` generator. On subsequent
    frames, it advances the generator with ``next()``. The generator preserves all
    temporal memory across frames.

    Compatible with ``TrackingOverlayNode`` and ``TrackingCocoJsonNode`` sinks.
    """

    INPUT_SPECS = {
        "rgb_frame": PortSpec(dtype=torch.float32, shape=(1, -1, -1, 3)),
    }
    OUTPUT_SPECS = {
        "frame_id": PortSpec(dtype=torch.int64, shape=(1,)),
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
        # -- prompt configuration --
        prompt_type: str = "text",
        prompt_text: str = "person",
        prompt_bboxes_xywh: list[list[float]] | None = None,
        prompt_bbox_labels: list[int] | None = None,
        prompt_points: list[list[float]] | None = None,
        prompt_point_labels: list[int] | None = None,
        prompt_obj_id: int = 1,
        prompt_mask_path: str | None = None,
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
        if prompt_frame_idx < 0 or prompt_frame_idx >= num_frames:
            raise ValueError(
                f"prompt_frame_idx must be in [0, {num_frames - 1}], got {prompt_frame_idx}."
            )
        valid_types = ("text", "bbox", "point", "mask")
        if prompt_type not in valid_types:
            raise ValueError(f"prompt_type must be one of {valid_types}, got '{prompt_type}'.")
        if prompt_type == "bbox" and prompt_bboxes_xywh and len(prompt_bboxes_xywh) > 1:
            raise ValueError(
                "SAM3StreamingPropagation currently supports exactly one initial bbox prompt. "
                "Provide a single bbox prompt only."
            )

        self._num_frames = int(num_frames)
        self._checkpoint_path = checkpoint_path
        self._compile_model = compile_model
        self._prompt_frame_idx = int(prompt_frame_idx)
        self._prompt_type = prompt_type
        self._prompt_text = prompt_text
        self._prompt_bboxes_xywh = prompt_bboxes_xywh
        self._prompt_bbox_labels = prompt_bbox_labels
        self._prompt_points = prompt_points
        self._prompt_point_labels = prompt_point_labels
        self._prompt_obj_id = prompt_obj_id
        self._prompt_mask_path = prompt_mask_path
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

        super().__init__(
            num_frames=num_frames,
            checkpoint_path=checkpoint_path,
            compile_model=compile_model,
            prompt_frame_idx=prompt_frame_idx,
            prompt_type=prompt_type,
            prompt_text=prompt_text,
            prompt_bboxes_xywh=prompt_bboxes_xywh,
            prompt_bbox_labels=prompt_bbox_labels,
            prompt_points=prompt_points,
            prompt_point_labels=prompt_point_labels,
            prompt_obj_id=prompt_obj_id,
            prompt_mask_path=prompt_mask_path,
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
                "SAM3StreamingPropagation generator exhausted early at frame "
                f"{requested_frame_idx} (configured num_frames={self._num_frames})."
            ) from exc

    # -- Prompt application ---------------------------------------------------

    def _apply_prompt(self) -> None:
        """Apply the configured prompt to ``prompt_frame_idx`` of the inference state."""
        state = self._inference_state
        device = self._model_device()
        prompt_frame_idx = self._prompt_frame_idx

        if self._prompt_type == "text":
            self._model.add_prompt(state, frame_idx=prompt_frame_idx, text_str=self._prompt_text)

        elif self._prompt_type == "bbox":
            if not self._prompt_bboxes_xywh:
                raise ValueError("prompt_bboxes_xywh required for bbox prompt_type.")
            boxes = np.array(self._prompt_bboxes_xywh, dtype=np.float32)
            labels = (
                torch.tensor(self._prompt_bbox_labels, dtype=torch.long, device=device)
                if self._prompt_bbox_labels
                else torch.ones(len(boxes), dtype=torch.long, device=device)
            )
            self._model.add_prompt(
                state,
                frame_idx=prompt_frame_idx,
                boxes_xywh=boxes,
                box_labels=labels,
            )

        elif self._prompt_type == "point":
            if not self._prompt_points:
                raise ValueError("prompt_points required for point prompt_type.")
            if not self._prompt_point_labels:
                raise ValueError("prompt_point_labels required for point prompt_type.")
            points = torch.tensor(self._prompt_points, dtype=torch.float32, device=device)
            point_labels = torch.tensor(self._prompt_point_labels, dtype=torch.int64, device=device)
            self._model.add_prompt(
                state,
                frame_idx=prompt_frame_idx,
                points=points,
                point_labels=point_labels,
                obj_id=self._prompt_obj_id,
            )

        elif self._prompt_type == "mask":
            if not self._prompt_mask_path:
                raise ValueError("prompt_mask_path required for mask prompt_type.")
            mask_gray = cv2.imread(self._prompt_mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_gray is None:
                raise FileNotFoundError(f"Cannot load mask: {self._prompt_mask_path}")
            mask_binary = (mask_gray > 127).astype(np.float32)
            mask_tensor = torch.from_numpy(mask_binary).to(device=device, dtype=torch.float32)
            self._model.add_mask(
                state,
                frame_idx=prompt_frame_idx,
                obj_id=self._prompt_obj_id,
                mask=mask_tensor,
            )
            ys, xs = np.where(mask_binary > 0)
            if xs.size > 0 and ys.size > 0:
                h_mask, w_mask = mask_binary.shape
                point_x = float(xs.mean() / w_mask)
                point_y = float(ys.mean() / h_mask)
                self._model.add_prompt(
                    state,
                    frame_idx=prompt_frame_idx,
                    points=torch.tensor([[point_x, point_y]], dtype=torch.float32, device=device),
                    point_labels=torch.tensor([1], dtype=torch.int64, device=device),
                    obj_id=self._prompt_obj_id,
                )
            # Force regular propagation after the initial mask add.
            # Tracker partial propagation for mask-only prompts can require point inputs.
            state["action_history"].clear()

    # -- Output conversion ----------------------------------------------------

    @staticmethod
    def _pack_output(frame_idx: int, postprocessed: dict | None) -> dict[str, torch.Tensor]:
        """Convert SAM3 per-object output to per-frame label map format."""
        if postprocessed is None or len(postprocessed.get("out_obj_ids", [])) == 0:
            return {
                "frame_id": torch.tensor([frame_idx], dtype=torch.int64),
                "mask": torch.zeros(1, 1, 1, dtype=torch.int32),
                "object_ids": torch.zeros(1, 0, dtype=torch.int64),
                "detection_scores": torch.zeros(1, 0, dtype=torch.float32),
            }

        obj_ids = postprocessed["out_obj_ids"]
        binary_masks = postprocessed["out_binary_masks"]
        probs = postprocessed["out_probs"]

        h, w = binary_masks.shape[1], binary_masks.shape[2]
        label_map = np.zeros((h, w), dtype=np.int32)
        for oid, m in zip(obj_ids, binary_masks):
            label_map[m] = int(oid)

        return {
            "frame_id": torch.tensor([frame_idx], dtype=torch.int64),
            "mask": torch.from_numpy(label_map).unsqueeze(0),
            "object_ids": torch.from_numpy(np.array(obj_ids, dtype=np.int64)).unsqueeze(0),
            "detection_scores": torch.from_numpy(np.array(probs, dtype=np.float32)).unsqueeze(0),
        }

    # -- Forward --------------------------------------------------------------

    def forward(
        self,
        rgb_frame: torch.Tensor,
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
        dict with keys: ``frame_id``, ``mask``, ``object_ids``, ``detection_scores``.
        """
        self._ensure_model()

        frame_np = rgb_frame[0].detach().cpu().numpy()  # [H, W, 3] float32
        frame_idx = self._frame_idx

        if frame_idx == 0:
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
            if self._prompt_type in {"point", "mask"}:
                # Interactivity prompt paths require cached outputs on the prompted frame.
                self._inference_state["cached_frame_outputs"].setdefault(self._prompt_frame_idx, {})
        else:
            # -- Subsequent frames: buffer + extend state --
            self._frame_buffer.add(frame_np)
            self._extend_state_for_frame(frame_idx)

        if self._generator is None and frame_idx >= self._prompt_frame_idx:
            self._prepare_state_for_full_sequence()
            self._apply_prompt()

            # Start one generator, potentially beginning at a non-zero prompt frame.
            self._generator = self._model.propagate_in_video(
                self._inference_state,
                start_frame_idx=self._prompt_frame_idx,
                max_frame_num_to_track=self._num_frames - self._prompt_frame_idx - 1,
                reverse=False,
            )
            yield_frame_idx, postprocessed = self._next_generator_output(frame_idx)
            result = self._pack_output(yield_frame_idx, postprocessed)
        else:
            if self._generator is None:
                # Frames before the prompt frame: keep outputs empty.
                result = self._pack_output(frame_idx, None)
            else:
                # Frames after prompt: advance the running generator.
                yield_frame_idx, postprocessed = self._next_generator_output(frame_idx)
                result = self._pack_output(yield_frame_idx, postprocessed)

        self._frame_idx += 1

        if self._progress_log_interval > 0 and frame_idx % self._progress_log_interval == 0:
            n_objs = result["object_ids"].shape[1]
            logger.info(
                "SAM3StreamingPropagation: frame {}/{}, {} objects",
                frame_idx,
                self._num_frames,
                n_objs,
            )

        return result
