"""Tests for SAM3 node geometry/prompt helper utilities."""

from __future__ import annotations

import numpy as np

from cuvis_ai_sam3.node.utils import _interior_points_per_component


def _to_pixel(point: tuple[float, float], height: int, width: int) -> tuple[int, int]:
    """Map a normalized (x, y) point back to integer pixel coordinates."""
    x, y = point
    px = min(int(round(x * width)), width - 1)
    py = min(int(round(y * height)), height - 1)
    return px, py


def test_empty_mask_yields_no_points() -> None:
    mask = np.zeros((20, 30), dtype=np.float32)
    assert _interior_points_per_component(mask) == []


def test_single_component_yields_multiple_interior_points() -> None:
    # A single large component must get several interior points, not one: a lone point collapses
    # the mask-conditioned segmentation to a tiny blob under SAM3.
    mask = np.zeros((40, 40), dtype=np.float32)
    mask[10:30, 10:30] = 1.0
    points = _interior_points_per_component(mask)
    assert len(points) > 1
    for point in points:
        px, py = _to_pixel(point, *mask.shape)
        assert mask[py, px] > 0  # every point sits inside the component


def test_multiple_points_for_small_and_large_components() -> None:
    # Spacing is chosen so each component yields ~target points regardless of area, so both a small
    # and a large block get several interior anchors (never a single collapse-prone point).
    for y0, y1, x0, x1 in [(30, 50, 30, 50), (10, 70, 10, 70)]:  # 20x20 and 60x60
        mask = np.zeros((80, 80), dtype=np.float32)
        mask[y0:y1, x0:x1] = 1.0
        points = _interior_points_per_component(mask)
        assert len(points) > 1
        for point in points:
            px, py = _to_pixel(point, *mask.shape)
            assert mask[py, px] > 0


def test_two_disjoint_components_points_inside_each() -> None:
    # An object split into two disjoint blocks (e.g. occluded). Points must land inside both blocks,
    # never in the background gap between them, with several anchors per block.
    mask = np.zeros((40, 80), dtype=np.float32)
    mask[10:30, 5:25] = 1.0  # left block
    mask[10:30, 55:75] = 1.0  # right block (disjoint)

    points = _interior_points_per_component(mask)
    assert len(points) > 2
    xs = []
    for point in points:
        px, py = _to_pixel(point, *mask.shape)
        assert mask[py, px] > 0  # never in the background gap between blocks
        xs.append(px)

    # both blocks are represented (points on the left half and on the right half)
    assert any(x < 40 for x in xs) and any(x >= 40 for x in xs)
