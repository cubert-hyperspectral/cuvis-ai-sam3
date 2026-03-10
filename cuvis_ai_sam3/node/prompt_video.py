"""SAM3 PVS (Promptable Video Segmentation) node — tracks only prompted objects."""

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


class _TrackerFrameBuffer:
    """Streaming frame buffer for the PVS tracker.

    The tracker reads frames via ``inference_state["images"][frame_idx]``
    (see ``sam3_tracking_predictor.py`` line 1067).  When ``init_state()``
    is called without ``video_path``, ``images`` is not populated — this
    buffer is injected into ``inference_state["images"]`` to supply frames
    one at a time as they arrive from the pipeline.
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

    def add(self, frame_float_hwc: np.ndarray) -> int:
        """Preprocess an RGB float32 [0,1] frame and store in the next slot.

        Returns the internal frame index assigned.
        """
        if frame_float_hwc.ndim != 3 or frame_float_hwc.shape[2] != 3:
            raise ValueError(
                f"Expected frame shape [H, W, 3], got {tuple(frame_float_hwc.shape)}."
            )
        frame_uint8 = np.clip(frame_float_hwc * 255.0, 0, 255).astype(np.uint8)
        frame_resized = cv2.resize(
            frame_uint8,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_CUBIC,
        )
        frame_np = frame_resized.astype(np.float32)
        frame_tensor = torch.from_numpy(frame_np).permute(2, 0, 1)
        frame_tensor = frame_tensor.to(device=self.device, dtype=torch.float16)
        frame_tensor -= self._img_mean
        frame_tensor /= self._img_std

        idx = self._next_idx
        self._frames[idx] = frame_tensor
        self._next_idx += 1
        return idx

    def __getitem__(self, index: int | torch.Tensor) -> torch.Tensor:
        if isinstance(index, (int, np.integer)):
            idx = int(index)
            if idx not in self._frames:
                raise IndexError(
                    f"Frame {idx} not yet buffered (have {sorted(self._frames.keys())})."
                )
            return self._frames[idx]
        if isinstance(index, torch.Tensor):
            if index.numel() == 1:
                return self.__getitem__(int(index.item())).unsqueeze(0)
            return torch.stack([self.__getitem__(int(v)) for v in index.tolist()], dim=0)
        raise TypeError(f"Index must be int or Tensor, got {type(index)}.")

    def __len__(self) -> int:
        return self._next_idx

    def prune_before(self, frame_idx: int) -> None:
        """Drop frames with index < frame_idx to free GPU memory."""
        for k in [k for k in self._frames if k < frame_idx]:
            del self._frames[k]


class SAM3ObjectTracker(Node):
    """PVS object tracker — tracks only explicitly prompted objects.

    Uses ``Sam3TrackerPredictor`` directly, supporting bbox, point, and mask
    prompts for one or multiple objects.  All prompts are applied on a single
    ``prompt_frame_idx`` (a measurement ID matched against the ``frame_id``
    input port).

    Compatible with ``TrackingOverlayNode`` and ``TrackingCocoJsonNode`` sinks.
    """

    INPUT_SPECS = {
        "rgb_frame": PortSpec(dtype=torch.float32, shape=(1, -1, -1, 3)),
        "frame_id": PortSpec(dtype=torch.int64, shape=(1,)),
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
        # -- prompt configuration (at least one must be provided) --
        prompt_bboxes: list[dict] | None = None,
        prompt_points: list[dict] | None = None,
        prompt_masks: list[dict] | None = None,
        progress_log_interval: int = 50,
        **kwargs: Any,
    ) -> None:
        if num_frames < 1:
            raise ValueError(f"num_frames must be >= 1, got {num_frames}.")
        has_prompts = any([prompt_bboxes, prompt_points, prompt_masks])
        if not has_prompts:
            raise ValueError(
                "At least one of prompt_bboxes, prompt_points, or prompt_masks must be provided."
            )

        self._num_frames = int(num_frames)
        self._checkpoint_path = checkpoint_path
        self._compile_model = compile_model
        self._prompt_frame_idx = int(prompt_frame_idx)
        self._prompt_bboxes = prompt_bboxes or []
        self._prompt_points = prompt_points or []
        self._prompt_masks = prompt_masks or []
        self._progress_log_interval = progress_log_interval

        # Runtime state (initialized when prompt frame arrives)
        self._predictor: Any = None
        self._frame_buffer: _TrackerFrameBuffer | None = None
        self._inference_state: dict[str, Any] | None = None
        self._generator: Any = None
        self._internal_idx: int = 0
        self._tracker_started: bool = False
        self._generator_exhausted: bool = False
        self._prompt_frame_seen: bool = False

        super().__init__(
            num_frames=num_frames,
            checkpoint_path=checkpoint_path,
            compile_model=compile_model,
            prompt_frame_idx=prompt_frame_idx,
            prompt_bboxes=prompt_bboxes,
            prompt_points=prompt_points,
            prompt_masks=prompt_masks,
            progress_log_interval=progress_log_interval,
            **kwargs,
        )

    # -- Model loading --------------------------------------------------------

    def _ensure_model(self) -> None:
        if self._predictor is not None:
            return

        from sam3.model_builder import build_sam3_video_model

        build_kwargs: dict[str, Any] = {}
        if self._checkpoint_path:
            build_kwargs["checkpoint_path"] = self._checkpoint_path
        if self._compile_model:
            build_kwargs["compile"] = True

        sam3_model = build_sam3_video_model(**build_kwargs)
        self._predictor = sam3_model.tracker
        self._predictor.backbone = sam3_model.detector.backbone

        logger.info(
            "SAM3ObjectTracker: model loaded (image_size={})",
            self._predictor.image_size,
        )

    def _predictor_device(self) -> torch.device:
        try:
            return next(self._predictor.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    # -- Prompt application ---------------------------------------------------

    def _apply_prompts(self) -> None:
        """Apply all configured prompts on internal frame index 0.

        Note: tensors are created on CPU intentionally — the SAM3 predictor API
        moves inputs to ``inference_state["device"]`` internally.
        """
        state = self._inference_state

        for bbox_prompt in self._prompt_bboxes:
            obj_id = bbox_prompt["obj_id"]
            xywh = bbox_prompt["bbox_xywh"]
            x, y, w, h = xywh
            xyxy = torch.tensor(
                [[x, y, x + w, y + h]], dtype=torch.float32,
            )
            self._predictor.add_new_points_or_box(
                state, frame_idx=0, obj_id=obj_id, box=xyxy,
            )
            logger.debug(
                "SAM3ObjectTracker: bbox prompt obj_id={} xyxy={}", obj_id, xyxy.tolist()
            )

        for point_prompt in self._prompt_points:
            obj_id = point_prompt["obj_id"]
            points = torch.tensor(
                point_prompt["points"], dtype=torch.float32,
            )
            labels = torch.tensor(
                point_prompt["labels"], dtype=torch.int64,
            )
            self._predictor.add_new_points_or_box(
                state, frame_idx=0, obj_id=obj_id, points=points, labels=labels,
            )
            logger.debug(
                "SAM3ObjectTracker: point prompt obj_id={} n_points={}", obj_id, len(points)
            )

        for mask_prompt in self._prompt_masks:
            obj_id = mask_prompt["obj_id"]
            mask_path = mask_prompt["mask_path"]
            mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_gray is None:
                raise FileNotFoundError(f"Cannot load mask: {mask_path}")
            mask_binary = (mask_gray > 127).astype(np.float32)
            mask_tensor = torch.from_numpy(mask_binary).to(dtype=torch.float32)
            self._predictor.add_new_mask(
                state, frame_idx=0, obj_id=obj_id, mask=mask_tensor,
            )
            logger.debug(
                "SAM3ObjectTracker: mask prompt obj_id={} shape={}", obj_id, mask_tensor.shape
            )

    # -- Output conversion ----------------------------------------------------

    @staticmethod
    def _empty_output(height: int, width: int) -> dict[str, torch.Tensor]:
        """Return correctly-typed empty outputs."""
        return {
            "mask": torch.zeros(1, height, width, dtype=torch.int32),
            "object_ids": torch.zeros(1, 0, dtype=torch.int64),
            "detection_scores": torch.zeros(1, 0, dtype=torch.float32),
        }

    @staticmethod
    def _pack_output(
        obj_ids: list[int],
        video_res_masks: torch.Tensor,
        obj_scores: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Convert tracker 5-tuple output to node output format.

        Parameters
        ----------
        obj_ids : list[int]
            Object IDs from the tracker.
        video_res_masks : torch.Tensor
            Logit masks at video resolution, shape ``[N, 1, H, W]``.
        obj_scores : torch.Tensor
            Per-object score logits, shape ``[N, 1]``.
        """
        if len(obj_ids) == 0:
            return {
                "mask": torch.zeros(1, 1, 1, dtype=torch.int32),
                "object_ids": torch.zeros(1, 0, dtype=torch.int64),
                "detection_scores": torch.zeros(1, 0, dtype=torch.float32),
            }

        # video_res_masks: [N, 1, H, W] logits → binary
        h, w = video_res_masks.shape[-2], video_res_masks.shape[-1]
        label_map = np.zeros((h, w), dtype=np.int32)
        scores = []
        for i, oid in enumerate(obj_ids):
            binary_mask = (video_res_masks[i, 0] > 0.0).cpu().numpy()
            label_map[binary_mask] = int(oid)
            score = torch.sigmoid(obj_scores[i]).item() if obj_scores.numel() > 0 else 1.0
            scores.append(score)

        return {
            "mask": torch.from_numpy(label_map).unsqueeze(0),  # [1, H, W]
            "object_ids": torch.tensor([obj_ids], dtype=torch.int64),  # [1, N]
            "detection_scores": torch.tensor([scores], dtype=torch.float32),  # [1, N]
        }

    # -- Forward --------------------------------------------------------------

    def forward(
        self,
        rgb_frame: torch.Tensor,
        frame_id: torch.Tensor,
        context: Context | None = None,  # noqa: ARG002
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Process one RGB frame and return tracking outputs.

        Parameters
        ----------
        rgb_frame : torch.Tensor
            Shape ``[1, H, W, 3]``, float32, values in ``[0, 1]``.
        frame_id : torch.Tensor
            Shape ``[1]``, int64 — measurement index from data node.
        """
        frame_np = rgb_frame[0].detach().cpu().numpy()  # [H, W, 3]
        orig_h, orig_w = frame_np.shape[0], frame_np.shape[1]
        mesu_id = int(frame_id[0].item())

        # Before prompt frame: return empty outputs, no model interaction
        if not self._tracker_started and mesu_id != self._prompt_frame_idx:
            if self._progress_log_interval > 0 and self._internal_idx % self._progress_log_interval == 0:
                logger.info(
                    "SAM3ObjectTracker: frame mesu_id={} (pre-prompt), waiting for {}",
                    mesu_id,
                    self._prompt_frame_idx,
                )
            return self._empty_output(orig_h, orig_w)

        # Generator exhausted: return empty for remaining frames
        if self._generator_exhausted:
            return self._empty_output(orig_h, orig_w)

        # Prompt frame: initialize tracker
        if not self._tracker_started and mesu_id == self._prompt_frame_idx:
            self._prompt_frame_seen = True
            self._ensure_model()
            device = self._predictor_device()

            self._frame_buffer = _TrackerFrameBuffer(
                image_size=int(self._predictor.image_size),
                device=device,
            )

            self._inference_state = self._predictor.init_state(
                video_height=orig_h,
                video_width=orig_w,
                num_frames=self._num_frames,
            )
            # Inject streaming frame buffer
            self._inference_state["images"] = self._frame_buffer

            # Buffer the prompt frame (internal index 0)
            self._frame_buffer.add(frame_np)
            self._internal_idx = 1

            # Apply all prompts on internal frame 0
            self._apply_prompts()

            # Start propagation generator
            self._generator = self._predictor.propagate_in_video(
                self._inference_state,
                start_frame_idx=0,
                max_frame_num_to_track=self._num_frames,
                reverse=False,
                propagate_preflight=True,
            )
            self._tracker_started = True

            # Get prompt frame result
            try:
                gen_frame_idx, obj_ids, _low_res, video_res_masks, obj_scores = next(self._generator)
            except StopIteration:
                self._generator_exhausted = True
                return self._empty_output(orig_h, orig_w)

            result = self._pack_output(obj_ids, video_res_masks, obj_scores)

            if self._progress_log_interval > 0:
                logger.info(
                    "SAM3ObjectTracker: prompt frame mesu_id={}, {} objects tracked",
                    mesu_id,
                    result["object_ids"].shape[1],
                )
            return result

        # After prompt frame: buffer frame and advance generator
        self._frame_buffer.add(frame_np)
        self._internal_idx += 1

        try:
            gen_frame_idx, obj_ids, _low_res, video_res_masks, obj_scores = next(self._generator)
        except StopIteration:
            self._generator_exhausted = True
            return self._empty_output(orig_h, orig_w)

        result = self._pack_output(obj_ids, video_res_masks, obj_scores)

        if (
            self._progress_log_interval > 0
            and (self._internal_idx - 1) % self._progress_log_interval == 0
        ):
            logger.info(
                "SAM3ObjectTracker: frame mesu_id={} (internal={}), {} objects",
                mesu_id,
                self._internal_idx - 1,
                result["object_ids"].shape[1],
            )

        return result

    # -- Lifecycle ------------------------------------------------------------

    def close(self) -> None:
        """Validate prompt frame was seen and clean up resources."""
        if not self._prompt_frame_seen:
            logger.warning(
                "SAM3ObjectTracker: prompt_frame_idx={} was never seen in the stream "
                "— no tracking performed",
                self._prompt_frame_idx,
            )
        self._generator = None
        self._inference_state = None
        self._frame_buffer = None
