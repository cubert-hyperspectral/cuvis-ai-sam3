"""Small geometry and prompt helper utilities for SAM3 node implementations."""

from __future__ import annotations

import cv2
import numpy as np


def _centroid_point_from_binary_mask(mask_binary: np.ndarray) -> tuple[float, float] | None:
    """Return the normalized centroid of a binary mask or ``None`` when empty."""
    ys, xs = np.where(mask_binary > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    height, width = mask_binary.shape
    return float(xs.mean() / width), float(ys.mean() / height)


def _interior_points_per_component(
    mask_binary: np.ndarray, points_per_component: int = 16
) -> list[tuple[float, float]]:
    """Return several normalized interior points per connected component of a binary mask.

    A single point per component is too weak a prompt to hold a large object under SAM3: the
    mask-conditioned segmentation collapses to a tiny region around that lone point (a full-bus
    seed with one point shrinks to a ~1.5k px blob, while ~8+ interior points hold the whole
    ~66k px object). Sample a roughly uniform grid of interior pixels per component, with the
    spacing chosen so each component yields about ``points_per_component`` points, and always
    include the component's distance-transform interior (the pixel deepest inside that region)
    so even a tiny region keeps at least one anchor that never lands in a background gap.
    Returns normalized ``(x / width, y / height)`` points, or an empty list for an empty mask.
    """
    fg = (mask_binary > 0).astype(np.uint8)
    if int(fg.sum()) == 0:
        return []
    height, width = fg.shape
    num_labels, labels = cv2.connectedComponents(fg, connectivity=8)
    target = max(1, int(points_per_component))
    points: list[tuple[float, float]] = []
    for label in range(1, num_labels):  # 0 is the background label
        component = (labels == label).astype(np.uint8)
        dist = cv2.distanceTransform(component, cv2.DIST_L2, 5)
        y, x = np.unravel_index(int(dist.argmax()), dist.shape)
        # Insertion-ordered de-dup, deepest-inside anchor first.
        picked: dict[tuple[int, int], None] = {(int(x), int(y)): None}

        ys, xs = np.where(component > 0)
        # Spacing that yields ~target points across the component's area.
        step = max(1, int(np.sqrt(float(xs.size) / target)))
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        for gy in range(y0, y1 + 1, step):
            for gx in range(x0, x1 + 1, step):
                if component[gy, gx]:
                    picked.setdefault((int(gx), int(gy)), None)

        points.extend((float(px) / width, float(py) / height) for px, py in picked)
    return points


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
    "_interior_points_per_component",
]
