"""Tests for the SAM3 prompt-free segment-everything node."""

from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from cuvis_ai_sam3.node.sam3_segment_everything import (
    SAM3SegmentEverything,
    _MaskCandidate,
)

pytestmark = pytest.mark.unit


class _FakeProcessor:
    def __init__(self) -> None:
        self.images = []

    def set_image(self, image: object) -> dict[str, int]:
        self.images.append(image)
        return {"call_index": len(self.images)}


class _FakeModel:
    def __init__(self, masks: np.ndarray, scores: np.ndarray) -> None:
        self._masks = np.asarray(masks, dtype=np.float32)
        self._scores = np.asarray(scores, dtype=np.float32)
        self.calls: list[dict[str, object]] = []

    def predict_inst(self, inference_state: dict[str, object], **kwargs: object):  # noqa: ANN003
        self.calls.append({"inference_state": inference_state, **kwargs})
        low_res = np.zeros_like(self._masks, dtype=np.float32)
        return self._masks.copy(), self._scores.copy(), low_res


def _candidate(
    *,
    score: float,
    box_xyxy: Sequence[float],
    crop_box_xyxy: Sequence[float],
    point_xy: Sequence[float] = (0.5, 0.5),
    frame_shape: tuple[int, int] = (10, 12),
) -> _MaskCandidate:
    height, width = frame_shape
    x0, y0, x1, y1 = [int(v) for v in box_xyxy]
    mask = torch.zeros((height, width), dtype=torch.bool)
    mask[y0:y1, x0:x1] = True
    return _MaskCandidate(
        score=float(score),
        mask=mask,
        box_xyxy=torch.tensor(box_xyxy, dtype=torch.float32),
        point_xy=torch.tensor(point_xy, dtype=torch.float32),
        crop_box_xyxy=torch.tensor(crop_box_xyxy, dtype=torch.float32),
    )


class TestSAM3SegmentEverything:
    def test_constructor_validation_rejects_invalid_settings(self) -> None:
        with pytest.raises(ValueError, match="points_per_side"):
            SAM3SegmentEverything(points_per_side=0)
        with pytest.raises(ValueError, match="points_per_batch"):
            SAM3SegmentEverything(points_per_batch=0)
        with pytest.raises(ValueError, match="pred_iou_thresh"):
            SAM3SegmentEverything(pred_iou_thresh=1.1)
        with pytest.raises(ValueError, match="crop_n_layers"):
            SAM3SegmentEverything(crop_n_layers=-1)

    def test_ensure_model_is_lazy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sam3.model.sam3_image_processor as sam3_image_processor
        import sam3.model_builder as sam3_model_builder

        build_calls: list[dict[str, object]] = []

        def fake_build_sam3_image_model(**kwargs: object) -> object:
            build_calls.append(dict(kwargs))
            return object()

        class FakeSam3Processor:
            def __init__(self, model: object, **kwargs: object) -> None:
                self.model = model
                self.kwargs = kwargs

        monkeypatch.setattr(
            sam3_model_builder,
            "build_sam3_image_model",
            fake_build_sam3_image_model,
        )
        monkeypatch.setattr(sam3_image_processor, "Sam3Processor", FakeSam3Processor)

        node = SAM3SegmentEverything(
            checkpoint_path="weights.pt",
            device="cpu",
            compile_model=True,
        )
        assert node._model is None
        assert node._processor is None

        node._ensure_model()
        node._ensure_model()

        assert len(build_calls) == 1
        assert build_calls[0]["checkpoint_path"] == "weights.pt"
        assert build_calls[0]["device"] == "cpu"
        assert build_calls[0]["compile"] is True
        assert build_calls[0]["enable_inst_interactivity"] is True
        assert isinstance(node._processor, FakeSam3Processor)

    def test_empty_forward_returns_full_frame_empty_outputs(self) -> None:
        node = SAM3SegmentEverything(name="test_segment_everything_empty")
        node._ensure_model = MagicMock()
        node._collect_candidates = MagicMock(return_value=[])

        result = node.forward(torch.rand(1, 10, 12, 3, dtype=torch.float32))

        assert result["mask"].shape == (1, 10, 12)
        assert result["mask"].dtype == torch.int32
        assert torch.count_nonzero(result["mask"]).item() == 0
        assert result["object_ids"].shape == (1, 0)
        assert result["object_ids"].dtype == torch.int64
        assert result["detection_scores"].shape == (1, 0)
        assert result["detection_scores"].dtype == torch.float32

    def test_process_crop_batches_points_by_points_per_batch(self) -> None:
        node = SAM3SegmentEverything(points_per_batch=2, name="test_segment_everything_batches")
        node._processor = _FakeProcessor()
        node._point_grids = [
            np.asarray(
                [
                    [0.1, 0.1],
                    [0.2, 0.2],
                    [0.3, 0.3],
                    [0.4, 0.4],
                    [0.5, 0.5],
                ],
                dtype=np.float32,
            )
        ]

        seen_batch_sizes: list[int] = []

        def fake_process_point_batch(**kwargs: object) -> list[_MaskCandidate]:
            batch_points = np.asarray(kwargs["points_xy"], dtype=np.float32)
            seen_batch_sizes.append(int(batch_points.shape[0]))
            return []

        node._process_point_batch = MagicMock(side_effect=fake_process_point_batch)

        candidates = node._process_crop(
            frame_np=np.zeros((8, 8, 3), dtype=np.float32),
            crop_box_xyxy=[0, 0, 8, 8],
            layer_idx=0,
            frame_shape=(8, 8),
        )

        assert candidates == []
        assert seen_batch_sizes == [2, 2, 1]
        assert len(node._processor.images) == 1

    def test_process_point_batch_flattens_multimask_outputs(self) -> None:
        node = SAM3SegmentEverything(
            pred_iou_thresh=0.6,
            stability_score_thresh=0.0,
            multimask_output=True,
            name="test_segment_everything_flatten",
        )
        node._model = _FakeModel(
            masks=np.asarray(
                [
                    [
                        [
                            [2, 2, 0, 0],
                            [2, 2, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                        ],
                        [
                            [0, 0, 0, 0],
                            [0, 1, 1, 0],
                            [0, 1, 1, 0],
                            [0, 0, 0, 0],
                        ],
                    ],
                    [
                        [
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [1, 1, 0, 0],
                            [1, 1, 0, 0],
                        ],
                        [
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 1, 1],
                            [0, 0, 1, 1],
                        ],
                    ],
                ],
                dtype=np.float32,
            ),
            scores=np.asarray([[0.95, 0.25], [0.85, 0.7]], dtype=np.float32),
        )

        candidates = node._process_point_batch(
            inference_state={"mock": 1},
            points_xy=np.asarray([[0.25, 0.25], [0.75, 0.75]], dtype=np.float32),
            crop_box_xyxy=[0, 0, 4, 4],
            frame_shape=(4, 4),
        )

        assert [candidate.score for candidate in candidates] == pytest.approx([0.95, 0.85, 0.7])
        assert [tuple(candidate.point_xy.tolist()) for candidate in candidates] == [
            (0.25, 0.25),
            (0.75, 0.75),
            (0.75, 0.75),
        ]
        model_call = node._model.calls[0]
        assert np.asarray(model_call["point_coords"]).shape == (2, 1, 2)
        assert np.asarray(model_call["point_labels"]).shape == (2, 1)
        assert model_call["return_logits"] is True
        assert model_call["normalize_coords"] is True

    def test_process_point_batch_filters_by_stability_score(self) -> None:
        node = SAM3SegmentEverything(
            pred_iou_thresh=0.0,
            stability_score_thresh=0.95,
            stability_score_offset=1.0,
            mask_threshold=0.0,
            name="test_segment_everything_stability",
        )
        node._model = _FakeModel(
            masks=np.asarray(
                [
                    [
                        [
                            [2, 2, -2, -2],
                            [2, 2, -2, -2],
                            [-2, -2, -2, -2],
                            [-2, -2, -2, -2],
                        ],
                        [
                            [0.2, 0.2, -2.0, -2.0],
                            [0.2, 0.2, -2.0, -2.0],
                            [-2.0, -2.0, -2.0, -2.0],
                            [-2.0, -2.0, -2.0, -2.0],
                        ],
                    ]
                ],
                dtype=np.float32,
            ),
            scores=np.asarray([[0.9, 0.9]], dtype=np.float32),
        )

        candidates = node._process_point_batch(
            inference_state={"mock": 1},
            points_xy=np.asarray([[0.5, 0.5]], dtype=np.float32),
            crop_box_xyxy=[0, 0, 4, 4],
            frame_shape=(4, 4),
        )

        assert len(candidates) == 1
        assert candidates[0].score == pytest.approx(0.9)

    def test_process_point_batch_filters_small_masks(self) -> None:
        node = SAM3SegmentEverything(
            pred_iou_thresh=0.0,
            stability_score_thresh=0.0,
            min_mask_region_area=4,
            name="test_segment_everything_min_area",
        )
        node._model = _FakeModel(
            masks=np.asarray(
                [
                    [
                        [
                            [1, 0, 0, 0],
                            [1, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                        ],
                        [
                            [1, 1, 0, 0],
                            [1, 1, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                        ],
                    ]
                ],
                dtype=np.float32,
            ),
            scores=np.asarray([[0.9, 0.8]], dtype=np.float32),
        )

        candidates = node._process_point_batch(
            inference_state={"mock": 1},
            points_xy=np.asarray([[0.2, 0.2]], dtype=np.float32),
            crop_box_xyxy=[0, 0, 4, 4],
            frame_shape=(4, 4),
        )

        assert len(candidates) == 1
        assert candidates[0].score == pytest.approx(0.8)
        assert int(candidates[0].mask.sum().item()) == 4

    def test_process_crop_runs_within_crop_nms(self) -> None:
        node = SAM3SegmentEverything(points_per_batch=1, name="test_segment_everything_crop_nms")
        node._processor = _FakeProcessor()
        node._point_grids = [
            np.asarray(
                [
                    [0.25, 0.25],
                    [0.75, 0.75],
                ],
                dtype=np.float32,
            )
        ]

        duplicate_candidates = [
            _candidate(
                score=0.9, box_xyxy=[1, 1, 5, 5], crop_box_xyxy=[0, 0, 8, 8], frame_shape=(8, 8)
            ),
            _candidate(
                score=0.8, box_xyxy=[1, 1, 5, 5], crop_box_xyxy=[0, 0, 8, 8], frame_shape=(8, 8)
            ),
        ]

        node._process_point_batch = MagicMock(
            side_effect=[[duplicate_candidates[0]], [duplicate_candidates[1]]]
        )

        kept = node._process_crop(
            frame_np=np.zeros((8, 8, 3), dtype=np.float32),
            crop_box_xyxy=[0, 0, 8, 8],
            layer_idx=0,
            frame_shape=(8, 8),
        )

        assert len(kept) == 1
        assert kept[0].score == pytest.approx(0.9)

    def test_cross_crop_nms_prefers_smaller_crop_when_scores_tie(self) -> None:
        node = SAM3SegmentEverything(name="test_segment_everything_cross_crop_nms")
        full_crop = _candidate(
            score=0.9,
            box_xyxy=[2, 2, 6, 6],
            crop_box_xyxy=[0, 0, 12, 10],
        )
        smaller_crop = _candidate(
            score=0.9,
            box_xyxy=[2, 2, 6, 6],
            crop_box_xyxy=[2, 2, 8, 8],
        )

        kept = node._deduplicate_candidates(
            [full_crop, smaller_crop],
            iou_threshold=0.5,
            prefer_smaller_crops=True,
        )

        assert len(kept) == 1
        assert kept[0].crop_box_xyxy.tolist() == pytest.approx([2.0, 2.0, 8.0, 8.0])

    def test_pack_output_assigns_contiguous_ids_in_score_order(self) -> None:
        node = SAM3SegmentEverything(name="test_segment_everything_pack")
        lower_score = _candidate(
            score=0.6,
            box_xyxy=[6, 1, 9, 4],
            crop_box_xyxy=[0, 0, 12, 10],
        )
        higher_score = _candidate(
            score=0.95,
            box_xyxy=[1, 1, 5, 5],
            crop_box_xyxy=[0, 0, 12, 10],
        )

        packed = node._pack_output([lower_score, higher_score], frame_shape=(10, 12))

        assert packed["object_ids"].tolist() == [[1, 2]]
        assert packed["detection_scores"][0].tolist() == pytest.approx([0.95, 0.6])
        mask_values = sorted(int(v) for v in torch.unique(packed["mask"]).tolist())
        assert mask_values == [0, 1, 2]

    def test_output_specs_match_tracking_contract(self) -> None:
        specs = SAM3SegmentEverything.OUTPUT_SPECS

        assert specs["mask"].dtype == torch.int32
        assert specs["object_ids"].dtype == torch.int64
        assert specs["detection_scores"].dtype == torch.float32
