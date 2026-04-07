"""Small geometry and prompt helper utilities for SAM3 node implementations."""

from __future__ import annotations

import numpy as np


def _centroid_point_from_binary_mask(mask_binary: np.ndarray) -> tuple[float, float] | None:
    """Return the normalized centroid of a binary mask or ``None`` when empty."""
    ys, xs = np.where(mask_binary > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    height, width = mask_binary.shape
    return float(xs.mean() / width), float(ys.mean() / height)


def _binary_mask_from_xyxy(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    frame_shape: tuple[int, int],
) -> np.ndarray:
    """Rasterize one xyxy box into a binary mask on the current frame size."""
    height, width = int(frame_shape[0]), int(frame_shape[1])
    mask = np.zeros((height, width), dtype=np.uint8)

    x0 = int(np.floor(max(0.0, min(float(x_min), float(width - 1)))))
    y0 = int(np.floor(max(0.0, min(float(y_min), float(height - 1)))))
    x1 = int(np.ceil(max(float(x0 + 1), min(float(x_max), float(width)))))
    y1 = int(np.ceil(max(float(y0 + 1), min(float(y_max), float(height)))))

    mask[y0:y1, x0:x1] = 1
    return mask


def _bbox_iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two xyxy boxes."""
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a.tolist()]
    bx1, by1, bx2, by2 = [float(v) for v in box_b.tolist()]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


__all__ = [
    "_bbox_iou_xyxy",
    "_binary_mask_from_xyxy",
    "_centroid_point_from_binary_mask",
]
