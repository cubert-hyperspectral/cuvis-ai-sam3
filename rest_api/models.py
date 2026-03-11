"""Pydantic request/response schemas for the SAM3 REST API."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, model_validator

# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class StartSessionRequest(BaseModel):
    video_path: str
    session_id: str | None = None


class AddPromptRequest(BaseModel):
    frame_index: int
    # PCS (text)
    text: str | None = None
    # PVS (point/box)
    points: list[list[float]] | None = None
    point_labels: list[int] | None = None
    bounding_boxes: list[list[float]] | None = None
    bounding_box_is_positive: list[int] | None = None  # 1 = foreground/include, 0 = background/exclude
    obj_id: int | None = None

    @model_validator(mode="after")
    def _require_at_least_one_prompt(self) -> AddPromptRequest:
        if self.text is None and self.points is None and self.bounding_boxes is None:
            raise ValueError("At least one of text, points, or bounding_boxes is required")
        return self


class PropagateRequest(BaseModel):
    direction: Literal["forward", "backward", "both"] = "forward"
    start_frame_index: int | None = None
    max_frames: int = 100
    disable_hotstart_retro_suppression: bool = False


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class StartSessionResponse(BaseModel):
    session_id: str
    num_frames: int
    frame_rate: float
    width: int
    height: int


class SessionInfoResponse(BaseModel):
    session_id: str
    video_path: str
    num_frames: int
    frame_rate: float
    width: int
    height: int


class ObjectResult(BaseModel):
    obj_id: int
    bbox_xywh: list[float]
    score: float
    mask_rle: dict | None = None


class AddPromptResponse(BaseModel):
    frame_index: int
    objects: list[ObjectResult]


class PropagateFrameResult(BaseModel):
    frame_index: int
    objects: list[ObjectResult]


# ---------------------------------------------------------------------------
# Helpers — convert SAM3 numpy outputs to Pydantic models
# ---------------------------------------------------------------------------


def mask_to_rle(binary_mask: np.ndarray) -> dict:
    """Encode a 2-D bool mask to COCO-style RLE dict."""
    flat = binary_mask.ravel(order="F")  # Fortran (column-major) order per COCO
    changes = np.diff(np.concatenate([[0], flat.astype(np.int8), [0]]))
    starts = np.where(changes != 0)[0]
    counts = np.diff(starts).tolist()
    return {"counts": counts, "size": list(binary_mask.shape)}


def outputs_to_object_results(
    outputs: dict,
    include_masks: bool = False,
) -> list[ObjectResult]:
    """Convert SAM3 postprocessed output dict to a list of ``ObjectResult``.

    SAM3 outputs contain:
    - ``out_obj_ids``: numpy int array
    - ``out_probs``: numpy float array
    - ``out_boxes_xywh``: numpy float array, normalized [0,1]
    - ``out_binary_masks``: numpy bool array (N, H, W)
    """
    obj_ids = outputs.get("out_obj_ids", np.array([]))
    probs = outputs.get("out_probs", np.array([]))
    boxes = outputs.get("out_boxes_xywh", np.zeros((0, 4)))
    masks = outputs.get("out_binary_masks", None)

    results: list[ObjectResult] = []
    for i, obj_id in enumerate(obj_ids):
        rle = None
        if include_masks and masks is not None and i < len(masks):
            rle = mask_to_rle(masks[i])
        results.append(
            ObjectResult(
                obj_id=int(obj_id),
                bbox_xywh=[float(v) for v in boxes[i]],
                score=float(probs[i]),
                mask_rle=rle,
            )
        )
    return results
