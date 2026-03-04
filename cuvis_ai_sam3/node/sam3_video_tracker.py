"""Streaming SAM3 tracker node with tentative and confirmed outputs."""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PortSpec

from sam3.model.data_misc import BatchedDatapoint, FindStage, convert_my_tensors
from sam3.model.geometry_encoders import Prompt
from sam3.model.utils.misc import copy_data_to_device
from sam3.model_builder import build_sam3_video_model


class _FrameBuffer:
    """Dict-backed frame buffer that mirrors SAM3 lazy loader preprocessing."""

    def __init__(
        self,
        image_size: int,
        device: torch.device,
        image_mean: tuple[float, float, float],
        image_std: tuple[float, float, float],
    ) -> None:
        self.image_size = int(image_size)
        self.device = torch.device(device)
        self._img_mean = torch.tensor(image_mean, dtype=torch.float16).view(3, 1, 1)
        self._img_std = torch.tensor(image_std, dtype=torch.float16).view(3, 1, 1)
        self._img_mean = self._img_mean.to(self.device)
        self._img_std = self._img_std.to(self.device)
        self._frames: dict[int, torch.Tensor] = {}
        self._next_idx = 0

    def add(self, frame_uint8_hwc: np.ndarray) -> int:
        """Preprocess and append one RGB uint8 frame."""
        if frame_uint8_hwc.ndim != 3 or frame_uint8_hwc.shape[2] != 3:
            raise ValueError(
                f"Expected frame shape [H, W, 3], received {tuple(frame_uint8_hwc.shape)}."
            )
        if frame_uint8_hwc.dtype != np.uint8:
            frame_uint8_hwc = np.clip(frame_uint8_hwc, 0, 255).astype(np.uint8)

        frame_resized = cv2.resize(
            frame_uint8_hwc,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_CUBIC,
        )
        frame_np = frame_resized.astype(np.float32)
        frame_tensor = torch.from_numpy(frame_np).permute(2, 0, 1)
        frame_tensor = frame_tensor.to(device=self.device, dtype=torch.float16)
        frame_tensor -= self._img_mean
        frame_tensor /= self._img_std

        frame_idx = self._next_idx
        self._frames[frame_idx] = frame_tensor
        self._next_idx += 1
        return frame_idx

    def __getitem__(self, index: int | torch.Tensor) -> torch.Tensor:
        """Return preprocessed frame(s) by integer or tensor index."""
        if isinstance(index, (int, np.integer)):
            idx = int(index)
            if idx not in self._frames:
                raise IndexError(f"Frame index {idx} not found in _FrameBuffer.")
            return self._frames[idx]
        if isinstance(index, torch.Tensor):
            if index.numel() == 1:
                return self.__getitem__(int(index.item())).unsqueeze(0)
            indices = [int(v) for v in index.tolist()]
            return torch.stack([self.__getitem__(idx) for idx in indices], dim=0)
        raise TypeError(f"_FrameBuffer indices must be int or torch.Tensor, got {type(index)}.")

    def __len__(self) -> int:
        """Return the total number of frames added so far."""
        return self._next_idx

    def prune_before(self, frame_idx: int) -> None:
        """Drop cached frames with keys `< frame_idx`."""
        stale = [idx for idx in self._frames.keys() if idx < frame_idx]
        for idx in stale:
            self._frames.pop(idx, None)

    def to(self, device: torch.device | str, *args: Any, **kwargs: Any) -> _FrameBuffer:
        """Move cached tensors and normalization buffers to a new device."""
        del args, kwargs
        self.device = torch.device(device)
        self._img_mean = self._img_mean.to(self.device)
        self._img_std = self._img_std.to(self.device)
        self._frames = {idx: frame.to(self.device) for idx, frame in self._frames.items()}
        return self


class SAM3TrackerInference(Node):
    """One-frame SAM3 tracking node with tentative and confirmed outputs."""

    INPUT_SPECS = {
        "rgb_frame": PortSpec(
            dtype=torch.float32,
            shape=(1, -1, -1, 3),
            description="Single RGB frame [1, H, W, 3] in [0, 1].",
        ),
    }

    OUTPUT_SPECS = {
        "frame_id": PortSpec(
            dtype=torch.int64,
            shape=(1,),
            description="Absolute frame index [1].",
        ),
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(1, -1, -1),
            description="Tentative label map [1, H, W].",
        ),
        "object_ids": PortSpec(
            dtype=torch.int64,
            shape=(1, -1),
            description="Tentative object IDs [1, N].",
        ),
        "detection_scores": PortSpec(
            dtype=torch.float32,
            shape=(1, -1),
            description="Tentative detection scores [1, N].",
        ),
        "tracker_scores": PortSpec(
            dtype=torch.float32,
            shape=(1, -1),
            description="Tentative tracker scores [1, N].",
        ),
        "confirmed_mask": PortSpec(
            dtype=torch.int32,
            shape=(1, -1, -1),
            description="Confirmed label map [1, H, W].",
            optional=True,
        ),
        "confirmed_object_ids": PortSpec(
            dtype=torch.int64,
            shape=(1, -1),
            description="Confirmed object IDs [1, N_c].",
            optional=True,
        ),
        "confirmed_detection_scores": PortSpec(
            dtype=torch.float32,
            shape=(1, -1),
            description="Confirmed detection scores [1, N_c].",
            optional=True,
        ),
        "confirmed_tracker_scores": PortSpec(
            dtype=torch.float32,
            shape=(1, -1),
            description="Confirmed tracker scores [1, N_c].",
            optional=True,
        ),
    }

    def __init__(
        self,
        checkpoint_path: str | None = None,
        prompt_text: str = "person",
        masklet_confirmation_consecutive_det_thresh: int = 3,
        score_threshold_detection: float = 0.5,
        new_det_thresh: float = 0.7,
        det_nms_thresh: float = 0.1,
        overlap_suppress_thresh: float = 0.7,
        compile_model: bool = False,
        confirmation_warmup_frames: int = 2,
        confirmation_warmup_thresh: int = 1,
        confirmation_high_confidence_thresh: float = 0.9,
        max_tracker_states: int = 5,
        enable_state_diagnostics: bool = False,
        progress_log_interval: int = 50,
        **kwargs: Any,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.prompt_text = prompt_text
        self.masklet_confirmation_consecutive_det_thresh = int(
            masklet_confirmation_consecutive_det_thresh
        )
        self.score_threshold_detection = float(score_threshold_detection)
        self.new_det_thresh = float(new_det_thresh)
        self.det_nms_thresh = float(det_nms_thresh)
        self.overlap_suppress_thresh = float(overlap_suppress_thresh)
        self.compile_model = bool(compile_model)
        self.confirmation_warmup_frames = int(confirmation_warmup_frames)
        self.confirmation_warmup_thresh = int(confirmation_warmup_thresh)
        self.confirmation_high_confidence_thresh = float(confirmation_high_confidence_thresh)
        self.max_tracker_states = int(max_tracker_states)
        self.enable_state_diagnostics = bool(enable_state_diagnostics)
        self.progress_log_interval = max(0, int(progress_log_interval))

        super().__init__(
            checkpoint_path=checkpoint_path,
            prompt_text=prompt_text,
            masklet_confirmation_consecutive_det_thresh=masklet_confirmation_consecutive_det_thresh,
            score_threshold_detection=score_threshold_detection,
            new_det_thresh=new_det_thresh,
            det_nms_thresh=det_nms_thresh,
            overlap_suppress_thresh=overlap_suppress_thresh,
            compile_model=compile_model,
            confirmation_warmup_frames=confirmation_warmup_frames,
            confirmation_warmup_thresh=confirmation_warmup_thresh,
            confirmation_high_confidence_thresh=confirmation_high_confidence_thresh,
            max_tracker_states=max_tracker_states,
            enable_state_diagnostics=enable_state_diagnostics,
            progress_log_interval=progress_log_interval,
            **kwargs,
        )

        self._model = build_sam3_video_model(
            checkpoint_path=checkpoint_path,
            compile=self.compile_model,
        )
        self._model.score_threshold_detection = self.score_threshold_detection
        self._model.new_det_thresh = self.new_det_thresh
        self._model.det_nms_thresh = self.det_nms_thresh
        self._model.suppress_overlapping_based_on_recent_occlusion_threshold = (
            self.overlap_suppress_thresh
        )
        self._inference_state: dict[str, Any] | None = None
        self._frame_buffer: _FrameBuffer | None = None
        self._frame_id = 0
        self._det_count: dict[int, int] = {}
        self._prompted = False
        self._evict_horizon: int = 4 * getattr(
            getattr(self._model, "tracker", None), "max_obj_ptrs_in_encoder", 16
        )
        self._progress_start_time: float | None = None
        self._progress_last_log_time: float | None = None
        self._progress_last_log_frame: int = 0

    def reset(self) -> None:
        """Reset streaming state for a new sequence."""
        self._inference_state = None
        self._frame_buffer = None
        self._frame_id = 0
        self._det_count = {}
        self._prompted = False
        self._progress_start_time = None
        self._progress_last_log_time = None
        self._progress_last_log_frame = 0

    def _model_device(self) -> torch.device:
        """Return the device of the first model parameter, defaulting to CPU."""
        try:
            return next(self._model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _init_streaming_state(
        self,
        orig_height: int,
        orig_width: int,
        device: torch.device,
    ) -> dict[str, Any]:
        """Build the initial SAM3 inference state dict for streaming mode."""
        if self._frame_buffer is None:
            raise RuntimeError("_FrameBuffer must be initialized before state construction.")

        state: dict[str, Any] = {
            "image_size": int(self._model.image_size),
            "num_frames": 0,
            "orig_height": int(orig_height),
            "orig_width": int(orig_width),
            "constants": {},
            "input_batch": BatchedDatapoint(
                img_batch=self._frame_buffer,
                find_text_batch=["<text placeholder>", "visual"],
                find_inputs={},
                find_targets={},
                find_metadatas={},
            ),
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
        """Register a new frame index in the inference state and input batch."""
        if self._inference_state is None:
            raise RuntimeError("Inference state is not initialized.")
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

    def _slice_postprocessed(
        self,
        postprocessed: dict[str, Any],
        keep_idx: np.ndarray,
    ) -> dict[str, Any]:
        """Slice postprocessed arrays to keep only the given object indices."""
        obj_ids = np.asarray(postprocessed.get("out_obj_ids", np.zeros(0)), dtype=np.int64)
        probs = np.asarray(
            postprocessed.get("out_probs", np.zeros(obj_ids.shape[0])),
            dtype=np.float32,
        )
        boxes = np.asarray(
            postprocessed.get("out_boxes_xywh", np.zeros((obj_ids.shape[0], 4))),
            dtype=np.float32,
        )
        masks = np.asarray(postprocessed.get("out_binary_masks", np.zeros((0, 0, 0))), dtype=bool)

        if self._inference_state is not None and masks.ndim != 3:
            h = int(self._inference_state["orig_height"])
            w = int(self._inference_state["orig_width"])
            masks = np.zeros((obj_ids.shape[0], h, w), dtype=bool)

        idx = np.asarray(keep_idx, dtype=np.int64)
        return {
            "out_obj_ids": obj_ids[idx] if obj_ids.size else obj_ids[:0],
            "out_probs": probs[idx] if probs.size else probs[:0],
            "out_boxes_xywh": boxes[idx] if boxes.size else boxes[:0],
            "out_binary_masks": masks[idx] if masks.size else masks[:0],
            "frame_stats": postprocessed.get("frame_stats"),
        }

    def _apply_confirmation_filter(
        self,
        postprocessed: dict[str, Any],
        frame_idx: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Split detections into tentative (all) and confirmed subsets."""
        obj_ids = np.asarray(postprocessed.get("out_obj_ids", np.zeros(0)), dtype=np.int64)
        probs = np.asarray(
            postprocessed.get("out_probs", np.zeros(obj_ids.shape[0])),
            dtype=np.float32,
        )

        current_ids = {int(obj_id) for obj_id in obj_ids.tolist()}
        for obj_id in list(self._det_count.keys()):
            if obj_id not in current_ids:
                self._det_count[obj_id] = 0
        for obj_id in current_ids:
            self._det_count[obj_id] = self._det_count.get(obj_id, 0) + 1

        is_warmup = frame_idx < self.confirmation_warmup_frames
        active_thresh = (
            self.confirmation_warmup_thresh
            if is_warmup
            else self.masklet_confirmation_consecutive_det_thresh
        )

        confirmed_ids: set[int] = set()
        for i, obj_id in enumerate(obj_ids.tolist()):
            if self._det_count.get(int(obj_id), 0) >= active_thresh:
                confirmed_ids.add(int(obj_id))
            if float(probs[i]) >= self.confirmation_high_confidence_thresh:
                confirmed_ids.add(int(obj_id))

        keep_confirmed = np.array(
            [int(obj_id) in confirmed_ids for obj_id in obj_ids.tolist()],
            dtype=bool,
        )
        confirmed_idx = np.flatnonzero(keep_confirmed)
        tentative_idx = np.arange(obj_ids.shape[0], dtype=np.int64)

        tentative = self._slice_postprocessed(postprocessed, tentative_idx)
        confirmed = self._slice_postprocessed(postprocessed, confirmed_idx)
        return tentative, confirmed

    def _tracker_score_map(self, frame_idx: int) -> dict[int, float]:
        """Return per-object tracker scores for a given frame index."""
        if self._inference_state is None:
            return {}
        tracker_md = self._inference_state.get("tracker_metadata", {})
        frame_map = tracker_md.get("obj_id_to_tracker_score_frame_wise", {})
        scores = frame_map.get(frame_idx, {})
        if not isinstance(scores, dict):
            return {}
        return {int(k): float(v) for k, v in scores.items()}

    def _pack_label_mask(self, postprocessed: dict[str, Any]) -> torch.Tensor:
        """Merge per-object binary masks into a single [1, H, W] label map."""
        obj_ids = np.asarray(postprocessed.get("out_obj_ids", np.zeros(0)), dtype=np.int64)
        masks = np.asarray(postprocessed.get("out_binary_masks", np.zeros((0, 0, 0))), dtype=bool)

        if masks.ndim == 3:
            height, width = int(masks.shape[1]), int(masks.shape[2])
        elif self._inference_state is not None:
            height = int(self._inference_state["orig_height"])
            width = int(self._inference_state["orig_width"])
        else:
            height = width = 0

        label_mask = np.zeros((height, width), dtype=np.int32)
        for obj_id, obj_mask in zip(obj_ids.tolist(), masks, strict=False):
            label_mask[np.asarray(obj_mask, dtype=bool)] = int(obj_id)
        return torch.from_numpy(label_mask).unsqueeze(0)

    @staticmethod
    def _pack_ids(postprocessed: dict[str, Any]) -> torch.Tensor:
        """Pack object IDs into a [1, N] int64 tensor."""
        obj_ids = np.asarray(postprocessed.get("out_obj_ids", np.zeros(0)), dtype=np.int64)
        if obj_ids.size == 0:
            return torch.empty((1, 0), dtype=torch.int64)
        return torch.from_numpy(obj_ids.reshape(1, -1))

    @staticmethod
    def _pack_detection_scores(postprocessed: dict[str, Any]) -> torch.Tensor:
        """Pack detection probabilities into a [1, N] float32 tensor."""
        probs = np.asarray(postprocessed.get("out_probs", np.zeros(0)), dtype=np.float32)
        if probs.size == 0:
            return torch.empty((1, 0), dtype=torch.float32)
        return torch.from_numpy(probs.reshape(1, -1))

    @staticmethod
    def _pack_tracker_scores(
        postprocessed: dict[str, Any],
        tracker_scores: dict[int, float],
    ) -> torch.Tensor:
        """Pack per-object tracker scores into a [1, N] float32 tensor."""
        obj_ids = np.asarray(postprocessed.get("out_obj_ids", np.zeros(0)), dtype=np.int64)
        if obj_ids.size == 0:
            return torch.empty((1, 0), dtype=torch.float32)
        values = np.asarray([tracker_scores.get(int(obj_id), 0.0) for obj_id in obj_ids])
        return torch.from_numpy(values.astype(np.float32).reshape(1, -1))

    @staticmethod
    def _prune_dict(container: Any, keep_from: int) -> None:
        """Remove integer keys strictly less than *keep_from* from a dict."""
        if not isinstance(container, dict):
            return
        stale = [k for k in container.keys() if isinstance(k, int) and k < keep_from]
        for key in stale:
            container.pop(key, None)

    def _prune_state_for_frame(self, frame_idx: int) -> None:
        """Free cached data for the given frame to limit memory growth.

        Because we call ``_run_single_frame_inference`` directly (bypassing
        ``propagate_in_video``'s yield-loop cleanup), we must replicate its
        eviction logic here to prevent unbounded dict growth in tracker
        metadata and per-object output dicts.
        """
        if self._inference_state is None:
            return
        state = self._inference_state

        state.get("cached_frame_outputs", {}).pop(frame_idx, None)

        tracker_md = state.get("tracker_metadata", {})

        # -- obj_id_to_tracker_score_frame_wise: evict all frames before current --
        scores_fw = tracker_md.get("obj_id_to_tracker_score_frame_wise")
        if isinstance(scores_fw, dict):
            stale = [k for k in scores_fw if isinstance(k, int) and k < frame_idx]
            for k in stale:
                scores_fw.pop(k, None)

        # -- rank0_metadata structures --
        rank0_md = tracker_md.get("rank0_metadata", {})

        suppressed_map = rank0_md.get("suppressed_obj_ids", {})
        if isinstance(suppressed_map, dict):
            suppressed_map.pop(frame_idx, None)

        evict_before = frame_idx - self._evict_horizon

        # unmatched_frame_inds: obj_id -> [frame_indices]
        unmatched = rank0_md.get("unmatched_frame_inds")
        if isinstance(unmatched, dict):
            for obj_id in list(unmatched.keys()):
                lst = unmatched[obj_id]
                if isinstance(lst, list):
                    unmatched[obj_id] = [f for f in lst if f >= evict_before]

        # overlap_pair_to_frame_inds: (obj_a, obj_b) -> [frame_indices]
        overlap = rank0_md.get("overlap_pair_to_frame_inds")
        if isinstance(overlap, dict):
            for pair in list(overlap.keys()):
                lst = overlap[pair]
                if isinstance(lst, list):
                    overlap[pair] = [f for f in lst if f >= evict_before]

        # -- per-object tracker output dicts (non-cond only; cond frames must
        #    be kept because propagate_in_video_preflight asserts their presence) --
        if evict_before > 0:
            for tracker_state in state.get("tracker_inference_states", []):
                if not isinstance(tracker_state, dict):
                    continue
                output_dict = tracker_state.get("output_dict", {})
                non_cond = output_dict.get("non_cond_frame_outputs", {})
                stale = [k for k in non_cond if isinstance(k, int) and k < evict_before]
                for k in stale:
                    del non_cond[k]
                for obj_dict in tracker_state.get("output_dict_per_obj", {}).values():
                    if not isinstance(obj_dict, dict):
                        continue
                    non_cond = obj_dict.get("non_cond_frame_outputs", {})
                    stale = [k for k in non_cond if isinstance(k, int) and k < evict_before]
                    for k in stale:
                        del non_cond[k]
                tracked = tracker_state.get("frames_already_tracked", {})
                if isinstance(tracked, dict):
                    stale = [k for k in tracked if isinstance(k, int) and k < evict_before]
                    for k in stale:
                        del tracked[k]

        if self._frame_buffer is not None:
            self._frame_buffer.prune_before(frame_idx)

        self._prune_dict(state.get("previous_stages_out"), frame_idx)
        self._prune_dict(state.get("per_frame_raw_point_input"), frame_idx)
        self._prune_dict(state.get("per_frame_raw_box_input"), frame_idx)
        self._prune_dict(state.get("per_frame_visual_prompt"), frame_idx)
        self._prune_dict(state.get("per_frame_geometric_prompt"), frame_idx)
        self._prune_dict(state.get("per_frame_cur_step"), frame_idx)

        input_batch = state.get("input_batch")
        if input_batch is not None:
            self._prune_dict(input_batch.find_inputs, frame_idx)
            self._prune_dict(input_batch.find_targets, frame_idx)
            self._prune_dict(input_batch.find_metadatas, frame_idx)

    def _evict_excess_tracker_states(self, frame_idx: int) -> None:
        """Remove the oldest non-primary tracker states when the count exceeds the limit.

        Each tracker state requires a full SAM2 propagation per frame, so the
        per-frame cost grows linearly with the state count.  We keep the first
        state (ts0, which holds the initial prompt objects) and evict the oldest
        remaining states by removing all their objects through the model's own
        ``_tracker_remove_objects`` API so that metadata stays consistent.

        After removing tracker states we must also patch ``tracker_metadata`` so
        that ``obj_ids_per_gpu`` / ``obj_ids_all_gpu`` / ``num_obj_per_gpu`` stay
        in sync — otherwise ``run_tracker_propagation`` will hit a shape-mismatch
        assertion on the next frame.
        """
        if self._inference_state is None or self.max_tracker_states <= 0:
            return
        states = self._inference_state.get("tracker_inference_states", [])
        if len(states) <= self.max_tracker_states:
            return

        from loguru import logger

        # Collect object IDs from the oldest non-primary states that exceed the limit.
        # States are ordered oldest-first; index 0 is the prompt state we always keep.
        n_to_evict = len(states) - self.max_tracker_states
        obj_ids_to_remove: list[int] = []
        for ts in states[1 : 1 + n_to_evict]:
            if not isinstance(ts, dict):
                continue
            obj_ids_to_remove.extend(ts.get("obj_ids", []))

        if not obj_ids_to_remove:
            return

        logger.trace(
            "Evicting {} tracker state(s) ({} -> {}), removing obj_ids={}",
            n_to_evict,
            len(states),
            self.max_tracker_states,
            obj_ids_to_remove,
        )
        # Remove the objects from tracker states (also drops empty states).
        self._model._tracker_remove_objects(states, obj_ids_to_remove)

        # Patch tracker_metadata so it matches the surviving objects.
        removed_set = {int(i) for i in obj_ids_to_remove}
        md = self._inference_state.get("tracker_metadata", {})
        rank = getattr(self._model, "rank", 0)

        # obj_ids_per_gpu: list of np arrays, one per GPU
        ids_per_gpu = md.get("obj_ids_per_gpu")
        if ids_per_gpu is not None and rank < len(ids_per_gpu):
            old_ids = ids_per_gpu[rank]
            keep = np.array([int(i) not in removed_set for i in old_ids], dtype=bool)
            ids_per_gpu[rank] = old_ids[keep]

        # obj_ids_all_gpu: flat np array of all object IDs
        ids_all = md.get("obj_ids_all_gpu")
        keep_all: np.ndarray | None = None
        if ids_all is not None and hasattr(ids_all, "__len__") and len(ids_all) > 0:
            keep_all = np.array([int(i) not in removed_set for i in ids_all], dtype=bool)
            md["obj_ids_all_gpu"] = ids_all[keep_all]

        # num_obj_per_gpu: np array of counts per GPU
        num_per_gpu = md.get("num_obj_per_gpu")
        if num_per_gpu is not None and ids_per_gpu is not None:
            for g in range(len(num_per_gpu)):
                if g < len(ids_per_gpu):
                    num_per_gpu[g] = len(ids_per_gpu[g])

        # obj_id_to_score: remove evicted IDs
        scores = md.get("obj_id_to_score")
        if isinstance(scores, dict):
            for oid in obj_ids_to_remove:
                scores.pop(int(oid), None)

        # rank0_metadata: clean up per-object structures
        r0 = md.get("rank0_metadata", {})
        for key in ("obj_first_frame_idx", "trk_keep_alive"):
            container = r0.get(key)
            if isinstance(container, dict):
                for oid in obj_ids_to_remove:
                    container.pop(int(oid), None)
        unmatched = r0.get("unmatched_frame_inds")
        if isinstance(unmatched, dict):
            for oid in obj_ids_to_remove:
                unmatched.pop(int(oid), None)

        # removed_obj_ids: mark evicted objects as removed so _process_hotstart
        # guards (line 1438) skip them when iterating overlap_pair_to_frame_inds.
        removed_set_r0 = r0.get("removed_obj_ids")
        if isinstance(removed_set_r0, set):
            removed_set_r0.update(removed_set)

        # overlap_pair_to_frame_inds: drop any pair that references an evicted object.
        # Without this, _process_hotstart iterates the pair and calls
        # obj_first_frame_idx[evicted_id] which raises KeyError.
        overlap = r0.get("overlap_pair_to_frame_inds")
        if isinstance(overlap, dict):
            stale_keys = [k for k in overlap if k[0] in removed_set or k[1] in removed_set]
            for k in stale_keys:
                del overlap[k]

        # masklet_confirmation: filter arrays to match obj_ids_all_gpu
        mc = r0.get("masklet_confirmation")
        if isinstance(mc, dict) and keep_all is not None:
            for arr_key in ("status", "consecutive_det_num"):
                arr = mc.get(arr_key)
                if arr is not None and hasattr(arr, "__len__") and len(arr) == len(keep_all):
                    mc[arr_key] = arr[keep_all]

    @staticmethod
    def _should_log_progress(frame_idx: int, interval: int) -> bool:
        """Return True when a periodic progress update should be emitted."""
        if interval <= 0:
            return False
        # Always emit one early signal after the first processed frame,
        # then every `interval` processed frames.
        return frame_idx == 0 or (frame_idx + 1) % interval == 0

    def _log_progress(self, frame_idx: int) -> None:
        """Log lightweight periodic progress metrics for long-running jobs."""
        if not self._should_log_progress(frame_idx, self.progress_log_interval):
            return

        from loguru import logger

        now = time.perf_counter()
        if self._progress_start_time is None:
            self._progress_start_time = now
        if self._progress_last_log_time is None:
            self._progress_last_log_time = self._progress_start_time

        frames_done = frame_idx + 1
        elapsed_total = max(now - self._progress_start_time, 1e-6)
        elapsed_chunk = max(now - self._progress_last_log_time, 1e-6)
        chunk_frames = max(frame_idx - self._progress_last_log_frame, 1)
        avg_s_per_frame = elapsed_total / frames_done
        recent_s_per_frame = elapsed_chunk / chunk_frames

        state_count = 0
        object_count = 0
        if self._inference_state is not None:
            states = self._inference_state.get("tracker_inference_states", [])
            if isinstance(states, list):
                state_count = len(states)
            md = self._inference_state.get("tracker_metadata", {})
            ids_all = md.get("obj_ids_all_gpu")
            if ids_all is not None and hasattr(ids_all, "__len__"):
                object_count = len(ids_all)

        logger.info(
            "SAM3 progress: frame={} | states={} | objects={} | avg={:.2f}s/frame | recent={:.2f}s/frame",
            frame_idx,
            state_count,
            object_count,
            avg_s_per_frame,
            recent_s_per_frame,
        )

        self._progress_last_log_time = now
        self._progress_last_log_frame = frame_idx

    def forward(
        self,
        rgb_frame: torch.Tensor,
        context: Context | None = None,  # noqa: ARG002
        **_: Any,
    ) -> dict[str, Any]:
        """Process a single RGB frame and emit tentative + confirmed tracking outputs."""
        detector = getattr(self._model, "detector", None)
        world_size = getattr(detector, "world_size", 1)
        if int(world_size) != 1:
            raise RuntimeError("SAM3TrackerInference currently supports only world_size == 1.")

        if rgb_frame.ndim != 4 or rgb_frame.shape[0] != 1 or rgb_frame.shape[-1] != 3:
            raise ValueError(
                f"rgb_frame must have shape [1, H, W, 3], received {tuple(rgb_frame.shape)}."
            )

        if self._inference_state is None:
            device = self._model_device()
            orig_height = int(rgb_frame.shape[1])
            orig_width = int(rgb_frame.shape[2])
            self._frame_buffer = _FrameBuffer(
                image_size=int(self._model.image_size),
                device=device,
                image_mean=tuple(self._model.image_mean),
                image_std=tuple(self._model.image_std),
            )
            self._inference_state = self._init_streaming_state(
                orig_height=orig_height,
                orig_width=orig_width,
                device=device,
            )

        if self._frame_buffer is None or self._inference_state is None:
            raise RuntimeError("Streaming state was not initialized.")
        if self._progress_start_time is None:
            self._progress_start_time = time.perf_counter()
            self._progress_last_log_time = self._progress_start_time

        frame_np = rgb_frame[0].detach().cpu().to(torch.float32).numpy()
        frame_uint8 = np.clip(frame_np * 255.0, 0.0, 255.0).astype(np.uint8)
        frame_idx = self._frame_buffer.add(frame_uint8)
        if frame_idx != self._frame_id:
            raise RuntimeError(f"Frame index mismatch: expected {self._frame_id}, got {frame_idx}.")

        self._extend_state_for_frame(frame_idx)
        state = self._inference_state

        if frame_idx == 0 and not self._prompted:
            _, postprocessed = self._model.add_prompt(
                state,
                frame_idx=0,
                text_str=self.prompt_text,
            )
            if self.compile_model and hasattr(self._model, "_compile_model"):
                self._model._compile_model()
            self._prompted = True
        else:
            if not self._prompted:
                raise RuntimeError("Prompt was not initialized before frame propagation.")
            raw_out = self._model._run_single_frame_inference(state, frame_idx, reverse=False)
            postprocessed = self._model._postprocess_output(
                state,
                raw_out,
                removed_obj_ids=raw_out.get("removed_obj_ids"),
                suppressed_obj_ids=raw_out.get("suppressed_obj_ids"),
                unconfirmed_obj_ids=[],
            )

        tentative, confirmed = self._apply_confirmation_filter(postprocessed, frame_idx)
        tracker_scores = self._tracker_score_map(frame_idx)

        outputs = {
            "frame_id": torch.tensor([frame_idx], dtype=torch.int64),
            "mask": self._pack_label_mask(tentative),
            "object_ids": self._pack_ids(tentative),
            "detection_scores": self._pack_detection_scores(tentative),
            "tracker_scores": self._pack_tracker_scores(tentative, tracker_scores),
            "confirmed_mask": self._pack_label_mask(confirmed),
            "confirmed_object_ids": self._pack_ids(confirmed),
            "confirmed_detection_scores": self._pack_detection_scores(confirmed),
            "confirmed_tracker_scores": self._pack_tracker_scores(confirmed, tracker_scores),
        }

        self._evict_excess_tracker_states(frame_idx)
        self._prune_state_for_frame(frame_idx)
        if torch.cuda.is_available() and frame_idx > 0 and frame_idx % 50 == 0:
            torch.cuda.empty_cache()

        # Optional diagnostics for long-run state-size debugging.
        if self.enable_state_diagnostics and frame_idx % 25 == 0:
            self._log_state_sizes(frame_idx)
        self._log_progress(frame_idx)

        self._frame_id += 1
        return outputs

    def _log_state_sizes(self, frame_idx: int) -> None:
        """Log sizes of key state dicts to diagnose memory growth."""
        from loguru import logger as _lg

        state = self._inference_state
        if state is None:
            return
        parts: list[str] = [f"frame={frame_idx}"]

        md = state.get("tracker_metadata", {})
        sfw = md.get("obj_id_to_tracker_score_frame_wise", {})
        parts.append(f"score_fw={len(sfw)}")

        r0 = md.get("rank0_metadata", {})
        um = r0.get("unmatched_frame_inds", {})
        parts.append(f"unmatched_keys={len(um)}")
        um_total = sum(len(v) for v in um.values() if isinstance(v, list))
        parts.append(f"unmatched_total={um_total}")
        ol = r0.get("overlap_pair_to_frame_inds", {})
        ol_total = sum(len(v) for v in ol.values() if isinstance(v, list))
        parts.append(f"overlap_total={ol_total}")

        for i, ts in enumerate(state.get("tracker_inference_states", [])):
            if not isinstance(ts, dict):
                continue
            od = ts.get("output_dict", {})
            nc = len(od.get("non_cond_frame_outputs", {}))
            cc = len(od.get("cond_frame_outputs", {}))
            parts.append(f"ts{i}_nc={nc}_cc={cc}")
            for oi, obj_d in ts.get("output_dict_per_obj", {}).items():
                if isinstance(obj_d, dict):
                    onc = len(obj_d.get("non_cond_frame_outputs", {}))
                    parts.append(f"ts{i}_obj{oi}_nc={onc}")
            fat = len(ts.get("frames_already_tracked", {}))
            parts.append(f"ts{i}_fat={fat}")

        fb = self._frame_buffer
        parts.append(f"fbuf={len(fb._frames) if fb else 0}")

        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024**3)
            parts.append(f"gpu={alloc:.2f}GB")

        _lg.debug("[state-diag] {}", " | ".join(parts))


__all__ = ["SAM3TrackerInference", "_FrameBuffer"]
