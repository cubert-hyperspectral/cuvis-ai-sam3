"""Tests for Pydantic models and output conversion helpers."""

from __future__ import annotations

import numpy as np
import pytest

from rest_api.models import (
    AddPromptRequest,
    PropagateRequest,
    mask_to_rle,
    outputs_to_object_results,
)


def test_add_prompt_requires_at_least_one():
    """Validator rejects requests with no prompt type."""
    with pytest.raises(ValueError, match="At least one of"):
        AddPromptRequest(frame_index=0)


def test_add_prompt_text_only():
    req = AddPromptRequest(frame_index=5, text="person")
    assert req.text == "person"
    assert req.points is None


def test_add_prompt_bbox():
    req = AddPromptRequest(
        frame_index=0,
        bounding_boxes=[[0.1, 0.2, 0.3, 0.4]],
        bounding_box_labels=[1],
    )
    assert len(req.bounding_boxes) == 1


def test_propagate_defaults():
    req = PropagateRequest()
    assert req.direction == "forward"
    assert req.max_frames == 100
    assert req.start_frame_index is None


def test_mask_to_rle():
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    mask[2, 1] = True
    rle = mask_to_rle(mask)
    assert rle["size"] == [4, 4]
    assert isinstance(rle["counts"], list)
    assert all(isinstance(c, int) for c in rle["counts"])


def test_outputs_to_object_results_no_masks():
    outputs = {
        "out_obj_ids": np.array([0, 1]),
        "out_probs": np.array([0.9, 0.8]),
        "out_boxes_xywh": np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.5, 0.2, 0.3]]),
        "out_binary_masks": np.ones((2, 10, 10), dtype=bool),
    }
    results = outputs_to_object_results(outputs, include_masks=False)
    assert len(results) == 2
    assert results[0].obj_id == 0
    assert results[0].mask_rle is None
    assert results[1].score == pytest.approx(0.8)


def test_outputs_to_object_results_with_masks():
    outputs = {
        "out_obj_ids": np.array([5]),
        "out_probs": np.array([0.95]),
        "out_boxes_xywh": np.array([[0.1, 0.2, 0.3, 0.4]]),
        "out_binary_masks": np.ones((1, 8, 8), dtype=bool),
    }
    results = outputs_to_object_results(outputs, include_masks=True)
    assert len(results) == 1
    assert results[0].mask_rle is not None
    assert "counts" in results[0].mask_rle
    assert results[0].mask_rle["size"] == [8, 8]


def test_outputs_to_object_results_empty():
    outputs = {
        "out_obj_ids": np.array([]),
        "out_probs": np.array([]),
        "out_boxes_xywh": np.zeros((0, 4)),
        "out_binary_masks": np.zeros((0, 10, 10), dtype=bool),
    }
    results = outputs_to_object_results(outputs)
    assert results == []
