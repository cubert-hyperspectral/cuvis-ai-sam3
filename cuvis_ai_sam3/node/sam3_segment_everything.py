"""Prompt-free SAM3 segment-everything node.

This module adapts the SAM2 automatic mask generator flow to the local SAM3
image predictor API. Each ``forward()`` call processes a single RGB frame
independently and emits a dense per-frame label map plus per-instance scores.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PortSpec
from loguru import logger
from PIL import Image
from torchvision.ops import masks_to_boxes
from torchvision.ops.boxes import batched_nms, box_area


@dataclass(slots=True)
class _MaskCandidate:
    """One mask proposal produced during AMG point sampling."""

    score: float
    mask: torch.Tensor
    box_xyxy: torch.Tensor
    point_xy: torch.Tensor
    crop_box_xyxy: torch.Tensor


class SAM3SegmentEverything(Node):
    """Segment everything on one RGB frame using SAM3 point-grid prompting.

    The node mirrors the SAM2 automatic mask generator pattern:
    sample a dense point grid, predict masks for each point in batches, filter
    by predicted IoU and mask stability, suppress duplicates with NMS, and
    finally rasterize the surviving masks into an int32 label map.

    Output IDs are per-frame only: every ``forward()`` call restarts instance
    numbering from ``1`` and keeps ``0`` for background.
    """

    INPUT_SPECS = {
        "rgb_frame": PortSpec(
            dtype=torch.float32,
            shape=(1, -1, -1, 3),
            description="RGB frame [1,H,W,3] in float32 with values in [0,1].",
        ),
        "frame_id": PortSpec(
            dtype=torch.int64,
            shape=(1,),
            description="Optional source frame index [1]. Ignored by this stateless node.",
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
        device: str | None = None,
        compile_model: bool = False,
        points_per_side: int = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.8,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        mask_threshold: float = 0.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        min_mask_region_area: int = 0,
        multimask_output: bool = True,
        **kwargs: Any,
    ) -> None:
        if points_per_side < 1:
            raise ValueError("points_per_side must be >= 1.")
        if points_per_batch < 1:
            raise ValueError("points_per_batch must be >= 1.")
        if crop_n_layers < 0:
            raise ValueError("crop_n_layers must be >= 0.")
        if crop_n_points_downscale_factor < 1:
            raise ValueError("crop_n_points_downscale_factor must be >= 1.")
        if min_mask_region_area < 0:
            raise ValueError("min_mask_region_area must be >= 0.")
        if stability_score_offset < 0:
            raise ValueError("stability_score_offset must be >= 0.")
        if crop_overlap_ratio < 0:
            raise ValueError("crop_overlap_ratio must be >= 0.")

        for name, value in (
            ("pred_iou_thresh", pred_iou_thresh),
            ("stability_score_thresh", stability_score_thresh),
            ("box_nms_thresh", box_nms_thresh),
            ("crop_nms_thresh", crop_nms_thresh),
        ):
            if not (0.0 <= float(value) <= 1.0):
                raise ValueError(f"{name} must be in [0, 1].")

        self._checkpoint_path = checkpoint_path
        self._requested_device = device
        self._resolved_device = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._compile_model = bool(compile_model)
        self._points_per_side = int(points_per_side)
        self._points_per_batch = int(points_per_batch)
        self._pred_iou_thresh = float(pred_iou_thresh)
        self._stability_score_thresh = float(stability_score_thresh)
        self._stability_score_offset = float(stability_score_offset)
        self._mask_threshold = float(mask_threshold)
        self._box_nms_thresh = float(box_nms_thresh)
        self._crop_n_layers = int(crop_n_layers)
        self._crop_nms_thresh = float(crop_nms_thresh)
        self._crop_overlap_ratio = float(crop_overlap_ratio)
        self._crop_n_points_downscale_factor = int(crop_n_points_downscale_factor)
        self._min_mask_region_area = int(min_mask_region_area)
        self._multimask_output = bool(multimask_output)

        self._model: Any | None = None
        self._processor: Any | None = None
        self._warned_small_region_skip = False
        self._point_grids = self._build_point_grids(
            n_per_side=self._points_per_side,
            n_layers=self._crop_n_layers,
            scale_per_layer=self._crop_n_points_downscale_factor,
        )

        super().__init__(
            checkpoint_path=checkpoint_path,
            device=device,
            compile_model=compile_model,
            points_per_side=points_per_side,
            points_per_batch=points_per_batch,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            stability_score_offset=stability_score_offset,
            mask_threshold=mask_threshold,
            box_nms_thresh=box_nms_thresh,
            crop_n_layers=crop_n_layers,
            crop_nms_thresh=crop_nms_thresh,
            crop_overlap_ratio=crop_overlap_ratio,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            min_mask_region_area=min_mask_region_area,
            multimask_output=multimask_output,
            **kwargs,
        )

    @staticmethod
    def _build_point_grid(n_per_side: int) -> np.ndarray:
        offset = 1.0 / (2 * n_per_side)
        points_one_side = np.linspace(offset, 1.0 - offset, n_per_side, dtype=np.float32)
        points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
        points_y = np.tile(points_one_side[:, None], (1, n_per_side))
        return np.stack([points_x, points_y], axis=-1).reshape(-1, 2)

    @classmethod
    def _build_point_grids(
        cls,
        n_per_side: int,
        n_layers: int,
        scale_per_layer: int,
    ) -> list[np.ndarray]:
        point_grids: list[np.ndarray] = []
        for layer_idx in range(n_layers + 1):
            n_points = max(1, int(n_per_side / (scale_per_layer**layer_idx)))
            point_grids.append(cls._build_point_grid(n_points))
        return point_grids

    @staticmethod
    def _generate_crop_boxes(
        frame_shape: tuple[int, int],
        n_layers: int,
        overlap_ratio: float,
    ) -> tuple[list[list[int]], list[int]]:
        crop_boxes: list[list[int]] = []
        layer_indices: list[int] = []
        frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
        short_side = min(frame_h, frame_w)

        crop_boxes.append([0, 0, frame_w, frame_h])
        layer_indices.append(0)

        def crop_len(orig_len: int, n_crops: int, overlap: int) -> int:
            return int(math.ceil((overlap * (n_crops - 1) + orig_len) / n_crops))

        for layer_idx in range(n_layers):
            n_crops_per_side = 2 ** (layer_idx + 1)
            overlap = int(overlap_ratio * short_side * (2 / n_crops_per_side))
            crop_w = crop_len(frame_w, n_crops_per_side, overlap)
            crop_h = crop_len(frame_h, n_crops_per_side, overlap)
            crop_x0s = [int((crop_w - overlap) * idx) for idx in range(n_crops_per_side)]
            crop_y0s = [int((crop_h - overlap) * idx) for idx in range(n_crops_per_side)]
            for x0, y0 in product(crop_x0s, crop_y0s):
                crop_boxes.append([x0, y0, min(x0 + crop_w, frame_w), min(y0 + crop_h, frame_h)])
                layer_indices.append(layer_idx + 1)

        return crop_boxes, layer_indices

    @staticmethod
    def _batch_iterator(batch_size: int, array: np.ndarray) -> list[np.ndarray]:
        return [array[i : i + batch_size] for i in range(0, len(array), batch_size)]

    @staticmethod
    def _uncrop_boxes_xyxy(boxes: torch.Tensor, crop_box_xyxy: list[int]) -> torch.Tensor:
        x0, y0, _, _ = crop_box_xyxy
        offset = boxes.new_tensor([x0, y0, x0, y0]).view(1, 4)
        return boxes + offset

    @staticmethod
    def _uncrop_points(points: torch.Tensor, crop_box_xyxy: list[int]) -> torch.Tensor:
        x0, y0, _, _ = crop_box_xyxy
        offset = points.new_tensor([x0, y0]).view(1, 2)
        return points + offset

    @staticmethod
    def _uncrop_masks(
        masks: torch.Tensor,
        crop_box_xyxy: list[int],
        orig_h: int,
        orig_w: int,
    ) -> torch.Tensor:
        x0, y0, x1, y1 = crop_box_xyxy
        if x0 == 0 and y0 == 0 and x1 == orig_w and y1 == orig_h:
            return masks
        pad_x = orig_w - (x1 - x0)
        pad_y = orig_h - (y1 - y0)
        pad = (x0, pad_x - x0, y0, pad_y - y0)
        return F.pad(masks, pad, value=0)

    @staticmethod
    def _is_box_near_crop_edge(
        boxes_xyxy: torch.Tensor,
        crop_box_xyxy: list[int],
        frame_shape: tuple[int, int],
        atol: float = 1.0,
    ) -> torch.Tensor:
        if boxes_xyxy.numel() == 0:
            return torch.zeros((0,), dtype=torch.bool, device=boxes_xyxy.device)
        crop_box = torch.as_tensor(crop_box_xyxy, dtype=torch.float32, device=boxes_xyxy.device)
        frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
        image_box = torch.tensor(
            [0, 0, frame_w, frame_h], dtype=torch.float32, device=boxes_xyxy.device
        )
        uncropped = SAM3SegmentEverything._uncrop_boxes_xyxy(boxes_xyxy.float(), crop_box_xyxy)
        near_crop_edge = torch.isclose(uncropped, crop_box[None, :], atol=atol, rtol=0.0)
        near_image_edge = torch.isclose(uncropped, image_box[None, :], atol=atol, rtol=0.0)
        return torch.any(torch.logical_and(near_crop_edge, ~near_image_edge), dim=1)

    @staticmethod
    def _calculate_stability_score(
        mask_logits: torch.Tensor,
        mask_threshold: float,
        threshold_offset: float,
    ) -> torch.Tensor:
        intersections = (
            (mask_logits > (mask_threshold + threshold_offset))
            .sum(-1, dtype=torch.int16)
            .sum(-1, dtype=torch.int32)
        )
        unions = (
            (mask_logits > (mask_threshold - threshold_offset))
            .sum(-1, dtype=torch.int16)
            .sum(-1, dtype=torch.int32)
        )
        return intersections.to(torch.float32) / unions.clamp_min(1).to(torch.float32)

    @staticmethod
    def _mask_to_box_xyxy(binary_masks: torch.Tensor) -> torch.Tensor:
        if binary_masks.numel() == 0:
            return torch.zeros((0, 4), dtype=torch.float32)
        return masks_to_boxes(binary_masks.to(dtype=torch.float32))

    @staticmethod
    def _box_nms(
        boxes_xyxy: torch.Tensor, scores: torch.Tensor, iou_threshold: float
    ) -> torch.Tensor:
        if boxes_xyxy.numel() == 0:
            return torch.zeros((0,), dtype=torch.int64)
        categories = torch.zeros(
            (boxes_xyxy.shape[0],), dtype=torch.int64, device=boxes_xyxy.device
        )
        return batched_nms(
            boxes_xyxy.float(), scores.float(), categories, iou_threshold=float(iou_threshold)
        )

    @staticmethod
    def _remove_small_regions(
        mask: np.ndarray, area_thresh: int, mode: str
    ) -> tuple[np.ndarray, bool]:
        import cv2  # type: ignore

        if area_thresh <= 0:
            return mask.astype(bool, copy=False), False
        if mode not in {"holes", "islands"}:
            raise ValueError(f"Unsupported region-cleanup mode: {mode}")

        correct_holes = mode == "holes"
        working_mask = (correct_holes ^ mask).astype(np.uint8, copy=False)
        n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
        sizes = stats[:, -1][1:]
        small_regions = [label + 1 for label, size in enumerate(sizes) if size < area_thresh]
        if not small_regions:
            return mask.astype(bool, copy=False), False

        fill_labels = [0] + small_regions
        if not correct_holes:
            fill_labels = [label for label in range(n_labels) if label not in fill_labels]
            if not fill_labels:
                fill_labels = [int(np.argmax(sizes)) + 1]
        return np.isin(regions, fill_labels), True

    def _ensure_model(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        build_kwargs: dict[str, Any] = {
            "device": self._resolved_device,
            "enable_inst_interactivity": True,
        }
        if self._checkpoint_path:
            build_kwargs["checkpoint_path"] = self._checkpoint_path
        if self._compile_model:
            build_kwargs["compile"] = True

        self._model = build_sam3_image_model(**build_kwargs)
        self._processor = Sam3Processor(
            self._model, device=self._resolved_device, confidence_threshold=0.0
        )
        logger.info(
            "SAM3SegmentEverything model loaded (device={}, points_per_side={}, crop_n_layers={})",
            self._resolved_device,
            self._points_per_side,
            self._crop_n_layers,
        )

    def _model_eval_context(self) -> contextlib.AbstractContextManager[None]:
        """Run image-model inference under the expected CUDA autocast mode.

        The SAM3 image stack emits bfloat16 activations in its vision path and
        expects CUDA autocast to reconcile those with float32 weights. Make the
        autocast mode explicit here instead of relying on ambient state.
        """
        if str(self._resolved_device).startswith("cuda"):
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    @staticmethod
    def _normalize_frame(rgb_frame: torch.Tensor) -> np.ndarray:
        if rgb_frame.ndim != 4 or int(rgb_frame.shape[0]) != 1:
            raise ValueError(
                f"Expected rgb_frame shape [1,H,W,3], got {tuple(int(v) for v in rgb_frame.shape)}."
            )
        frame_np = np.asarray(rgb_frame[0].detach().cpu().numpy(), dtype=np.float32)
        if frame_np.ndim != 3 or int(frame_np.shape[2]) != 3:
            raise ValueError(f"Expected RGB frame with shape [H,W,3], got {tuple(frame_np.shape)}.")
        return np.clip(frame_np, 0.0, 1.0)

    @staticmethod
    def _empty_output(height: int, width: int) -> dict[str, torch.Tensor]:
        return {
            "mask": torch.zeros((1, int(height), int(width)), dtype=torch.int32),
            "object_ids": torch.zeros((1, 0), dtype=torch.int64),
            "detection_scores": torch.zeros((1, 0), dtype=torch.float32),
        }

    def _remove_small_regions_if_enabled(
        self,
        masks: torch.Tensor,
        scores: torch.Tensor,
        points_xy: torch.Tensor,
        crop_box_xyxy: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._min_mask_region_area <= 0 or masks.numel() == 0:
            return (
                masks,
                scores,
                self._mask_to_box_xyxy(masks),
                self._uncrop_points(points_xy, crop_box_xyxy),
            )

        kept_masks: list[torch.Tensor] = []
        kept_scores: list[float] = []
        kept_points: list[torch.Tensor] = []
        try:
            for index in range(masks.shape[0]):
                mask_np = masks[index].detach().cpu().numpy().astype(bool, copy=False)
                mask_np, _ = self._remove_small_regions(
                    mask_np, self._min_mask_region_area, mode="holes"
                )
                mask_np, _ = self._remove_small_regions(
                    mask_np, self._min_mask_region_area, mode="islands"
                )
                cleaned = torch.from_numpy(mask_np.astype(bool, copy=False))
                if int(cleaned.sum().item()) < self._min_mask_region_area:
                    continue
                kept_masks.append(cleaned)
                kept_scores.append(float(scores[index].item()))
                kept_points.append(points_xy[index])
        except ModuleNotFoundError:
            if not self._warned_small_region_skip:
                logger.warning(
                    "OpenCV is unavailable; skipping min_mask_region_area postprocessing in SAM3SegmentEverything."
                )
                self._warned_small_region_skip = True
            return (
                masks,
                scores,
                self._mask_to_box_xyxy(masks),
                self._uncrop_points(points_xy, crop_box_xyxy),
            )

        if not kept_masks:
            empty_masks = torch.zeros((0,) + tuple(masks.shape[-2:]), dtype=torch.bool)
            empty_scores = torch.zeros((0,), dtype=torch.float32)
            empty_boxes = torch.zeros((0, 4), dtype=torch.float32)
            empty_points = torch.zeros((0, 2), dtype=torch.float32)
            return empty_masks, empty_scores, empty_boxes, empty_points

        cleaned_masks = torch.stack(kept_masks, dim=0)
        cleaned_scores = torch.tensor(kept_scores, dtype=torch.float32)
        cleaned_points = torch.stack(kept_points, dim=0).to(dtype=torch.float32)
        cleaned_boxes = self._mask_to_box_xyxy(cleaned_masks)
        return (
            cleaned_masks,
            cleaned_scores,
            cleaned_boxes,
            self._uncrop_points(cleaned_points, crop_box_xyxy),
        )

    def _process_point_batch(
        self,
        inference_state: dict[str, Any],
        points_xy: np.ndarray,
        crop_box_xyxy: list[int],
        frame_shape: tuple[int, int],
    ) -> list[_MaskCandidate]:
        if points_xy.size == 0:
            return []
        if self._model is None:
            raise RuntimeError("SAM3SegmentEverything model is not initialized.")

        point_coords = np.asarray(points_xy, dtype=np.float32)[:, None, :]
        point_labels = np.ones((point_coords.shape[0], 1), dtype=np.int32)
        with self._model_eval_context():
            masks_np, scores_np, _ = self._model.predict_inst(
                inference_state,
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=self._multimask_output,
                return_logits=True,
                normalize_coords=True,
            )

        mask_logits = torch.as_tensor(masks_np, dtype=torch.float32)
        if mask_logits.ndim == 3:
            mask_logits = mask_logits.unsqueeze(0)
        if mask_logits.ndim != 4:
            raise ValueError(
                f"SAM3SegmentEverything expected masks with shape [B,C,H,W], got {tuple(mask_logits.shape)}."
            )

        iou_preds = torch.as_tensor(scores_np, dtype=torch.float32)
        if iou_preds.ndim == 1:
            iou_preds = iou_preds.unsqueeze(0)
        if iou_preds.ndim != 2:
            raise ValueError(
                f"SAM3SegmentEverything expected scores with shape [B,C], got {tuple(iou_preds.shape)}."
            )

        if mask_logits.shape[:2] != iou_preds.shape:
            raise ValueError(
                "SAM3SegmentEverything received mismatched mask and score shapes: "
                f"masks={tuple(mask_logits.shape)}, scores={tuple(iou_preds.shape)}."
            )

        flat_logits = mask_logits.flatten(0, 1)
        flat_scores = iou_preds.flatten()
        flat_points = torch.from_numpy(np.asarray(points_xy, dtype=np.float32)).repeat_interleave(
            mask_logits.shape[1], dim=0
        )

        keep = torch.ones((flat_scores.shape[0],), dtype=torch.bool)
        if self._pred_iou_thresh > 0.0:
            keep &= flat_scores >= self._pred_iou_thresh

        if self._stability_score_thresh > 0.0:
            stability = self._calculate_stability_score(
                flat_logits,
                mask_threshold=self._mask_threshold,
                threshold_offset=self._stability_score_offset,
            )
            keep &= stability >= self._stability_score_thresh

        binary_masks = flat_logits > self._mask_threshold
        areas = binary_masks.flatten(1).sum(dim=1)
        keep &= areas > 0
        if self._min_mask_region_area > 0:
            keep &= areas >= self._min_mask_region_area

        if not bool(keep.any()):
            return []

        binary_masks = binary_masks[keep]
        flat_scores = flat_scores[keep]
        flat_points = flat_points[keep]

        local_boxes = self._mask_to_box_xyxy(binary_masks)
        keep = ~self._is_box_near_crop_edge(local_boxes, crop_box_xyxy, frame_shape)
        if not bool(keep.any()):
            return []

        binary_masks = binary_masks[keep]
        flat_scores = flat_scores[keep]
        flat_points = flat_points[keep]
        local_boxes = local_boxes[keep]

        uncropped_masks = self._uncrop_masks(
            binary_masks, crop_box_xyxy, frame_shape[0], frame_shape[1]
        )
        uncropped_boxes = self._uncrop_boxes_xyxy(local_boxes, crop_box_xyxy)
        (
            uncropped_masks,
            flat_scores,
            uncropped_boxes,
            uncropped_points,
        ) = self._remove_small_regions_if_enabled(
            masks=uncropped_masks.to(dtype=torch.bool),
            scores=flat_scores,
            points_xy=flat_points,
            crop_box_xyxy=crop_box_xyxy,
        )

        candidates: list[_MaskCandidate] = []
        for index in range(uncropped_masks.shape[0]):
            mask_t = uncropped_masks[index].to(dtype=torch.bool).cpu()
            if not bool(mask_t.any()):
                continue
            candidates.append(
                _MaskCandidate(
                    score=float(flat_scores[index].item()),
                    mask=mask_t,
                    box_xyxy=uncropped_boxes[index].to(dtype=torch.float32).cpu(),
                    point_xy=uncropped_points[index].to(dtype=torch.float32).cpu(),
                    crop_box_xyxy=torch.tensor(crop_box_xyxy, dtype=torch.float32),
                )
            )
        return candidates

    def _deduplicate_candidates(
        self,
        candidates: list[_MaskCandidate],
        iou_threshold: float,
        *,
        prefer_smaller_crops: bool,
    ) -> list[_MaskCandidate]:
        if len(candidates) <= 1:
            return list(candidates)

        ordered_candidates = list(candidates)
        if prefer_smaller_crops:
            ordered_candidates.sort(key=self._candidate_sort_key, reverse=True)

        boxes_xyxy = torch.stack(
            [candidate.box_xyxy for candidate in ordered_candidates], dim=0
        ).to(dtype=torch.float32)
        scores = torch.tensor(
            [candidate.score for candidate in ordered_candidates],
            dtype=torch.float32,
        )

        keep_indices = self._box_nms(boxes_xyxy, scores, iou_threshold)
        return [ordered_candidates[int(index)] for index in keep_indices.cpu().tolist()]

    def _process_crop(
        self,
        frame_np: np.ndarray,
        crop_box_xyxy: list[int],
        layer_idx: int,
        frame_shape: tuple[int, int],
    ) -> list[_MaskCandidate]:
        if self._processor is None:
            raise RuntimeError("SAM3SegmentEverything processor is not initialized.")

        x0, y0, x1, y1 = crop_box_xyxy
        cropped_frame = frame_np[y0:y1, x0:x1, :]
        if cropped_frame.size == 0:
            return []

        cropped_u8 = np.clip(cropped_frame * 255.0, 0.0, 255.0).astype(np.uint8)
        with self._model_eval_context():
            inference_state = self._processor.set_image(Image.fromarray(cropped_u8))

        crop_hw = cropped_frame.shape[:2]
        points_scale = np.array(crop_hw, dtype=np.float32)[None, ::-1]
        points_for_crop = self._point_grids[layer_idx] * points_scale

        candidates: list[_MaskCandidate] = []
        for batch_points in self._batch_iterator(self._points_per_batch, points_for_crop):
            candidates.extend(
                self._process_point_batch(
                    inference_state=inference_state,
                    points_xy=batch_points,
                    crop_box_xyxy=crop_box_xyxy,
                    frame_shape=frame_shape,
                )
            )

        return self._deduplicate_candidates(
            candidates,
            self._box_nms_thresh,
            prefer_smaller_crops=False,
        )

    @staticmethod
    def _candidate_sort_key(candidate: _MaskCandidate) -> tuple[float, float]:
        crop_area = float(box_area(candidate.crop_box_xyxy.view(1, 4)).item())
        return (float(candidate.score), -crop_area)

    def _collect_candidates(self, frame_np: np.ndarray) -> list[_MaskCandidate]:
        frame_shape = (int(frame_np.shape[0]), int(frame_np.shape[1]))
        crop_boxes, layer_indices = self._generate_crop_boxes(
            frame_shape,
            n_layers=self._crop_n_layers,
            overlap_ratio=self._crop_overlap_ratio,
        )

        all_candidates: list[_MaskCandidate] = []
        for crop_box_xyxy, layer_idx in zip(crop_boxes, layer_indices, strict=False):
            all_candidates.extend(
                self._process_crop(
                    frame_np=frame_np,
                    crop_box_xyxy=crop_box_xyxy,
                    layer_idx=layer_idx,
                    frame_shape=frame_shape,
                )
            )

        if len(crop_boxes) > 1:
            all_candidates = self._deduplicate_candidates(
                all_candidates,
                self._crop_nms_thresh,
                prefer_smaller_crops=True,
            )
        return all_candidates

    def _pack_output(
        self,
        kept_candidates: list[_MaskCandidate],
        frame_shape: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        height, width = int(frame_shape[0]), int(frame_shape[1])
        if not kept_candidates:
            return self._empty_output(height=height, width=width)

        sorted_candidates = sorted(kept_candidates, key=self._candidate_sort_key, reverse=True)
        label_map = torch.zeros((height, width), dtype=torch.int32)
        kept_scores: list[float] = []

        for candidate in sorted_candidates:
            visible_mask = candidate.mask.to(dtype=torch.bool) & (label_map == 0)
            if not bool(visible_mask.any()):
                continue
            object_id = len(kept_scores) + 1
            label_map[visible_mask] = int(object_id)
            kept_scores.append(float(candidate.score))

        if not kept_scores:
            return self._empty_output(height=height, width=width)

        object_ids = torch.arange(1, len(kept_scores) + 1, dtype=torch.int64).unsqueeze(0)
        detection_scores = torch.tensor(kept_scores, dtype=torch.float32).unsqueeze(0)
        return {
            "mask": label_map.unsqueeze(0),
            "object_ids": object_ids,
            "detection_scores": detection_scores,
        }

    @torch.inference_mode()
    def forward(
        self,
        rgb_frame: torch.Tensor,
        frame_id: torch.Tensor | None = None,  # noqa: ARG002
        context: Context | None = None,  # noqa: ARG002
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Segment all mask candidates in the current frame."""
        self._ensure_model()

        frame_np = self._normalize_frame(rgb_frame)
        frame_shape = (int(frame_np.shape[0]), int(frame_np.shape[1]))
        candidates = self._collect_candidates(frame_np)
        result = self._pack_output(candidates, frame_shape=frame_shape)

        logger.debug(
            "SAM3SegmentEverything: frame {}x{} -> {} kept masks",
            frame_shape[0],
            frame_shape[1],
            result["object_ids"].shape[1],
        )
        return result


__all__ = ["SAM3SegmentEverything"]
