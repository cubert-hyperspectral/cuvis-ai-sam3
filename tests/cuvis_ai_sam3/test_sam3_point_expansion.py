"""Tests for the interactive single-frame SAM3 point-expansion node."""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np
import pytest
import torch

from cuvis_ai_sam3.node.sam3_point_expansion import SAM3PointExpansion

pytestmark = pytest.mark.unit


class _FakeProcessor:
    """Records set_image calls and returns a unique embedding-state per call."""

    def __init__(self) -> None:
        self.images: list[object] = []

    def set_image(self, image: object) -> dict[str, int]:
        self.images.append(image)
        return {"embed_id": len(self.images)}


class _FakeModel:
    """Returns fixed masks/scores from predict_inst and records the kwargs it was called with."""

    def __init__(self, masks: np.ndarray, scores: np.ndarray) -> None:
        self._masks = np.asarray(masks, dtype=np.float32)
        self._scores = np.asarray(scores, dtype=np.float32)
        self.calls: list[dict[str, Any]] = []

    def predict_inst(self, inference_state: dict[str, object], **kwargs: Any):
        self.calls.append({"inference_state": inference_state, **kwargs})
        low_res = np.zeros_like(self._masks, dtype=np.float32)
        return self._masks.copy(), self._scores.copy(), low_res


def _wire(node: SAM3PointExpansion, model: _FakeModel | None = None) -> _FakeProcessor:
    """Attach fakes so forward() runs without building the real SAM3 model."""
    processor = _FakeProcessor()
    node._processor = processor
    node._model = (
        model
        if model is not None
        else _FakeModel(
            masks=np.full((1, 1, 4, 4), 5.0, dtype=np.float32),
            scores=np.asarray([[0.9]], dtype=np.float32),
        )
    )
    return processor


def _frame(h: int = 4, w: int = 4) -> torch.Tensor:
    return torch.rand(1, h, w, 3, dtype=torch.float32)


def _fid(value: int) -> torch.Tensor:
    return torch.tensor([value], dtype=torch.int64)


def _pos(x: float, y: float) -> dict[str, Any]:
    return {"element_id": 0, "x": x, "y": y, "type": "positive"}


def _neg(x: float, y: float) -> dict[str, Any]:
    return {"element_id": 0, "x": x, "y": y, "type": "negative"}


class TestSAM3PointExpansion:
    def test_constructor_rejects_nonpositive_obj_id(self) -> None:
        with pytest.raises(ValueError, match="prompt_obj_id"):
            SAM3PointExpansion(prompt_obj_id=0)

    def test_empty_points_returns_empty_mask_without_calling_model(self) -> None:
        node = SAM3PointExpansion(name="test_pe_empty")
        node._model = _FakeModel(np.zeros((1, 1, 4, 4), np.float32), np.zeros((1, 1), np.float32))
        node._processor = _FakeProcessor()

        result = node.forward(_frame(), points=None)

        assert result["mask"].shape == (1, 4, 4)
        assert result["mask"].dtype == torch.int32
        assert int(torch.count_nonzero(result["mask"]).item()) == 0
        assert result["object_ids"].shape == (1, 0)
        assert result["detection_scores"].shape == (1, 0)
        assert node._model.calls == []
        assert node._processor.images == []

    def test_negative_only_returns_empty_mask(self) -> None:
        node = SAM3PointExpansion(name="test_pe_neg_only")
        _wire(node)
        result = node.forward(_frame(), points=[_neg(1.0, 1.0)])
        assert int(torch.count_nonzero(result["mask"]).item()) == 0

    def test_parse_points_requires_positive(self) -> None:
        with pytest.raises(ValueError, match="at least one positive"):
            SAM3PointExpansion._parse_points([_neg(1.0, 1.0)])

    def test_parse_points_empty_returns_empty_arrays(self) -> None:
        coords, labels = SAM3PointExpansion._parse_points([])
        assert coords.shape == (0, 2)
        assert labels.shape == (0,)
        assert coords.dtype == np.float32
        assert labels.dtype == np.int32

    def test_parse_points_rejects_non_list(self) -> None:
        with pytest.raises(ValueError, match="list of dicts"):
            SAM3PointExpansion._parse_points("not-a-list")  # type: ignore[arg-type]

    def test_parse_points_rejects_non_dict_entry(self) -> None:
        with pytest.raises(ValueError, match="index 0 to be a dict"):
            SAM3PointExpansion._parse_points([42])  # type: ignore[list-item]

    def test_parse_points_rejects_unknown_type(self) -> None:
        with pytest.raises(ValueError, match="unknown type"):
            SAM3PointExpansion._parse_points([{"x": 1.0, "y": 1.0, "type": "bogus"}])

    def test_parse_points_requires_x_and_y(self) -> None:
        with pytest.raises(ValueError, match="missing 'x' or 'y'"):
            SAM3PointExpansion._parse_points([{"type": "positive"}])

    def test_pack_output_promotes_3d_masks(self) -> None:
        node = SAM3PointExpansion(prompt_obj_id=3, name="test_pe_pack_3d")
        masks = np.full((1, 4, 4), 5.0, dtype=np.float32)
        result = node._pack_output(masks, np.asarray([0.9]), height=4, width=4)
        assert sorted(int(v) for v in torch.unique(result["mask"]).tolist()) == [3]
        assert result["object_ids"].tolist() == [[3]]

    def test_pack_output_rejects_bad_ndim(self) -> None:
        node = SAM3PointExpansion(name="test_pe_pack_baddim")
        with pytest.raises(ValueError, match=r"shape \[B,C,H,W\]"):
            node._pack_output(np.zeros((4, 4)), np.asarray([0.9]), height=4, width=4)

    def test_pack_output_no_candidates_returns_empty(self) -> None:
        node = SAM3PointExpansion(name="test_pe_pack_empty")
        result = node._pack_output(
            np.zeros((1, 0, 4, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            height=4,
            width=4,
        )
        assert int(torch.count_nonzero(result["mask"]).item()) == 0
        assert result["object_ids"].shape == (1, 0)

    def test_positive_negative_labels_mapped_and_multimask_off(self) -> None:
        node = SAM3PointExpansion(name="test_pe_labels")
        _wire(node)

        node.forward(_frame(), points=[_pos(1.0, 1.0), _neg(3.0, 3.0)], frame_id=_fid(11))

        call = node._model.calls[0]
        coords = np.asarray(call["point_coords"])
        labels = np.asarray(call["point_labels"])
        assert coords.shape == (1, 2, 2)
        assert labels.tolist() == [[1, 0]]
        assert coords[0].tolist() == [[1.0, 1.0], [3.0, 3.0]]
        # Two points -> single-mask decode.
        assert call["multimask_output"] is False
        assert call["normalize_coords"] is True

    def test_single_positive_requests_multimask(self) -> None:
        node = SAM3PointExpansion(name="test_pe_multimask")
        _wire(node)
        node.forward(_frame(), points=[_pos(2.0, 2.0)], frame_id=_fid(11))
        assert node._model.calls[0]["multimask_output"] is True

    def test_neutral_points_are_dropped(self) -> None:
        node = SAM3PointExpansion(name="test_pe_neutral")
        _wire(node)
        neutral = {"element_id": 0, "x": 0.0, "y": 0.0, "type": "neutral"}
        node.forward(_frame(), points=[_pos(2.0, 2.0), neutral], frame_id=_fid(11))
        labels = np.asarray(node._model.calls[0]["point_labels"])
        assert labels.tolist() == [[1]]

    def test_embedding_cached_per_frame_id(self) -> None:
        node = SAM3PointExpansion(name="test_pe_cache")
        processor = _wire(node)

        node.forward(_frame(), points=[_pos(1.0, 1.0)], frame_id=_fid(11))
        node.forward(_frame(), points=[_pos(1.0, 1.0), _neg(3.0, 3.0)], frame_id=_fid(11))
        assert len(processor.images) == 1  # re-prompt of frame 11 did NOT re-embed

        node.forward(_frame(), points=[_pos(1.0, 1.0)], frame_id=_fid(12))
        assert len(processor.images) == 2  # new frame re-embeds

    def test_missing_frame_id_reembeds_each_call(self) -> None:
        node = SAM3PointExpansion(name="test_pe_no_fid")
        processor = _wire(node)
        node.forward(_frame(), points=[_pos(1.0, 1.0)])
        node.forward(_frame(), points=[_pos(1.0, 1.0)])
        assert len(processor.images) == 2

    def test_best_candidate_selected_by_score(self) -> None:
        # Candidate 0 (score 0.3) lights top-left; candidate 1 (score 0.9) lights bottom-right.
        masks = np.full((1, 2, 4, 4), -5.0, dtype=np.float32)
        masks[0, 0, 0:2, 0:2] = 5.0
        masks[0, 1, 2:4, 2:4] = 5.0
        model = _FakeModel(masks=masks, scores=np.asarray([[0.3, 0.9]], dtype=np.float32))
        node = SAM3PointExpansion(prompt_obj_id=7, name="test_pe_best")
        _wire(node, model)

        result = node.forward(_frame(), points=[_pos(2.0, 2.0)], frame_id=_fid(11))

        mask = result["mask"][0]
        assert int(mask[3, 3].item()) == 7  # bottom-right (high-score candidate)
        assert int(mask[0, 0].item()) == 0  # top-left (low-score candidate) not chosen
        assert sorted(int(v) for v in torch.unique(mask).tolist()) == [0, 7]
        assert result["mask"].dtype == torch.int32
        assert result["object_ids"].tolist() == [[7]]
        assert result["detection_scores"][0].tolist() == pytest.approx([0.9])

    def test_all_background_mask_returns_empty(self) -> None:
        model = _FakeModel(
            masks=np.full((1, 1, 4, 4), -5.0, dtype=np.float32),
            scores=np.asarray([[0.9]], dtype=np.float32),
        )
        node = SAM3PointExpansion(name="test_pe_allbg")
        _wire(node, model)
        result = node.forward(_frame(), points=[_pos(1.0, 1.0)], frame_id=_fid(11))
        assert int(torch.count_nonzero(result["mask"]).item()) == 0
        assert result["object_ids"].shape == (1, 0)

    def test_cleanup_and_reset_drop_cache(self) -> None:
        node = SAM3PointExpansion(name="test_pe_cleanup")
        node._model = object()
        node._processor = object()
        node._cached_frame_id = 11
        node._cached_state = {"embed_id": 1}

        node.cleanup()
        assert node._model is None
        assert node._processor is None
        assert node._cached_frame_id is None
        assert node._cached_state is None

        node._cached_frame_id = 5
        node._cached_state = {"embed_id": 2}
        node.reset()
        assert node._cached_frame_id is None
        assert node._cached_state is None

    def test_output_specs_match_tracking_contract(self) -> None:
        specs = SAM3PointExpansion.OUTPUT_SPECS
        assert specs["mask"].dtype == torch.int32
        assert specs["object_ids"].dtype == torch.int64
        assert specs["detection_scores"].dtype == torch.float32


class TestSam3ImageBaseHelpers:
    """Cover the shared ``_Sam3ImageNode`` helpers via a concrete subclass."""

    def test_normalize_frame_rejects_wrong_batch_shape(self) -> None:
        with pytest.raises(ValueError, match=r"shape \[1,H,W,3\]"):
            SAM3PointExpansion._normalize_frame(torch.rand(2, 4, 4, 3))

    def test_normalize_frame_rejects_non_rgb_channels(self) -> None:
        with pytest.raises(ValueError, match=r"shape \[H,W,3\]"):
            SAM3PointExpansion._normalize_frame(torch.rand(1, 4, 4, 4))

    def test_normalize_frame_clips_to_unit_range(self) -> None:
        frame = torch.tensor([[[[-1.0, 0.5, 2.0]]]], dtype=torch.float32)
        out = SAM3PointExpansion._normalize_frame(frame)
        assert out.shape == (1, 1, 3)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_model_eval_context_is_noop_on_unsupported_device(self) -> None:
        node = SAM3PointExpansion(name="test_pe_ctx_mps")
        node._resolved_device = "mps"
        assert isinstance(node._model_eval_context(), contextlib.nullcontext)

    def test_model_eval_context_autocasts_on_cpu(self) -> None:
        node = SAM3PointExpansion(name="test_pe_ctx_cpu")
        node._resolved_device = "cpu"
        assert isinstance(node._model_eval_context(), torch.autocast)
