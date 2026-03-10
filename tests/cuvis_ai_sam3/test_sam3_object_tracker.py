"""Tests for SAM3ObjectTracker node (PVS path) with mocked tracker predictor."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call

import cv2
import numpy as np
import pytest
import torch

from cuvis_ai_sam3.node.prompt_video import (
    SAM3ObjectTracker,
    _TrackerFrameBuffer,
)


# ---------------------------------------------------------------------------
# _TrackerFrameBuffer tests
# ---------------------------------------------------------------------------


class TestTrackerFrameBuffer:
    def test_add_and_getitem(self) -> None:
        buf = _TrackerFrameBuffer(image_size=8, device=torch.device("cpu"))
        frame = np.random.rand(6, 5, 3).astype(np.float32)
        idx = buf.add(frame)
        assert idx == 0
        out = buf[0]
        assert out.shape == (3, 8, 8)
        assert out.dtype == torch.float16

    def test_sequential_add(self) -> None:
        buf = _TrackerFrameBuffer(image_size=4, device=torch.device("cpu"))
        for i in range(3):
            idx = buf.add(np.random.rand(4, 4, 3).astype(np.float32))
            assert idx == i
        assert len(buf) == 3

    def test_missing_frame_raises(self) -> None:
        buf = _TrackerFrameBuffer(image_size=4, device=torch.device("cpu"))
        with pytest.raises(IndexError):
            _ = buf[0]

    def test_tensor_index(self) -> None:
        buf = _TrackerFrameBuffer(image_size=4, device=torch.device("cpu"))
        buf.add(np.zeros((3, 3, 3), dtype=np.float32))
        buf.add(np.zeros((3, 3, 3), dtype=np.float32))
        stacked = buf[torch.tensor([0, 1], dtype=torch.int64)]
        assert stacked.shape == (2, 3, 4, 4)

    def test_prune_before(self) -> None:
        buf = _TrackerFrameBuffer(image_size=4, device=torch.device("cpu"))
        for _ in range(3):
            buf.add(np.zeros((3, 3, 3), dtype=np.float32))
        buf.prune_before(2)
        with pytest.raises(IndexError):
            _ = buf[0]
        _ = buf[2]  # should work


# ---------------------------------------------------------------------------
# Helpers for mocking SAM3 tracker predictor
# ---------------------------------------------------------------------------


def _make_mock_predictor(image_size: int = 8) -> MagicMock:
    """Create a mock Sam3TrackerPredictor."""
    predictor = MagicMock()
    predictor.image_size = image_size

    param = torch.nn.Parameter(torch.zeros(1))
    predictor.parameters.side_effect = lambda: iter([param])

    return predictor


def _make_mock_sam3_model(image_size: int = 8) -> MagicMock:
    """Create a mock sam3 model with tracker and detector."""
    model = MagicMock()
    model.tracker = _make_mock_predictor(image_size)
    model.detector = MagicMock()
    model.detector.backbone = MagicMock()
    return model


def _make_tracker_generator(num_frames: int, obj_ids: list[int], h: int = 10, w: int = 12):
    """Generator mimicking propagate_in_video 5-tuple output."""
    for frame_idx in range(num_frames):
        n = len(obj_ids)
        video_res_masks = torch.ones(n, 1, h, w, dtype=torch.float32)  # logits > 0 → all positive
        obj_scores = torch.ones(n, 1, dtype=torch.float32)
        yield frame_idx, obj_ids, torch.zeros(n, 1, h // 4, w // 4), video_res_masks, obj_scores


# ---------------------------------------------------------------------------
# SAM3ObjectTracker tests
# ---------------------------------------------------------------------------


class TestSAM3ObjectTracker:
    def _setup_node_with_mock(
        self,
        node: SAM3ObjectTracker,
        obj_ids: list[int] | None = None,
        h: int = 10,
        w: int = 12,
    ) -> MagicMock:
        """Inject mock predictor into node, return the mock."""
        if obj_ids is None:
            obj_ids = [1]
        mock_model = _make_mock_sam3_model()
        predictor = mock_model.tracker

        def _init_state(**kwargs):
            return {
                "video_height": kwargs.get("video_height", h),
                "video_width": kwargs.get("video_width", w),
                "num_frames": kwargs.get("num_frames", 5),
                "images": None,
                "point_inputs_per_obj": {},
                "mask_inputs_per_obj": {},
                "cached_features": {},
                "constants": {},
                "obj_id_to_idx": {},
                "obj_idx_to_id": {},
                "obj_ids": [],
                "output_dict": {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
                "first_ann_frame_idx": None,
                "output_dict_per_obj": {},
                "temp_output_dict_per_obj": {},
                "consolidated_frame_inds": {"cond_frame_outputs": set(), "non_cond_frame_outputs": set()},
                "tracking_has_started": False,
                "frames_already_tracked": {},
            }

        predictor.init_state.side_effect = _init_state

        num_frames = node._num_frames
        predictor.propagate_in_video.side_effect = lambda state, **kw: _make_tracker_generator(
            num_frames, obj_ids, h, w
        )

        # Inject: bypass _ensure_model
        node._predictor = predictor
        node._ensure_model = MagicMock()
        return predictor

    def _run_node(
        self,
        node: SAM3ObjectTracker,
        num_frames: int,
        h: int = 10,
        w: int = 12,
        start_mesu: int = 0,
    ) -> list[dict[str, torch.Tensor]]:
        """Run the node for num_frames with sequential mesu_index values."""
        results = []
        for i in range(num_frames):
            rgb = torch.rand(1, h, w, 3, dtype=torch.float32)
            frame_id = torch.tensor([start_mesu + i], dtype=torch.int64)
            result = node.forward(rgb, frame_id)
            results.append(result)
        return results

    # -- Single bbox ----------------------------------------------------------

    def test_single_bbox(self) -> None:
        node = SAM3ObjectTracker(
            num_frames=5,
            prompt_frame_idx=0,
            prompt_bboxes=[{"obj_id": 1, "bbox_xywh": [0.1, 0.2, 0.3, 0.4]}],
            name="test_single_bbox",
        )
        predictor = self._setup_node_with_mock(node, obj_ids=[1])
        results = self._run_node(node, num_frames=5)

        assert len(results) == 5
        # Verify add_new_points_or_box was called with correct args
        predictor.add_new_points_or_box.assert_called_once()
        call_kwargs = predictor.add_new_points_or_box.call_args
        assert call_kwargs.kwargs["frame_idx"] == 0
        assert call_kwargs.kwargs["obj_id"] == 1
        box = call_kwargs.kwargs["box"]
        expected_xyxy = [0.1, 0.2, 0.1 + 0.3, 0.2 + 0.4]
        np.testing.assert_allclose(box.cpu().numpy().flatten(), expected_xyxy, atol=1e-5)

    # -- Multiple bboxes ------------------------------------------------------

    def test_multiple_bboxes(self) -> None:
        node = SAM3ObjectTracker(
            num_frames=3,
            prompt_frame_idx=0,
            prompt_bboxes=[
                {"obj_id": 1, "bbox_xywh": [0.1, 0.2, 0.3, 0.4]},
                {"obj_id": 2, "bbox_xywh": [0.5, 0.1, 0.2, 0.3]},
            ],
            name="test_multi_bbox",
        )
        predictor = self._setup_node_with_mock(node, obj_ids=[1, 2])
        results = self._run_node(node, num_frames=3)

        assert len(results) == 3
        assert predictor.add_new_points_or_box.call_count == 2
        # Check each call has its own obj_id
        calls = predictor.add_new_points_or_box.call_args_list
        assert calls[0].kwargs["obj_id"] == 1
        assert calls[1].kwargs["obj_id"] == 2

    # -- Point prompt ---------------------------------------------------------

    def test_point_prompt(self) -> None:
        node = SAM3ObjectTracker(
            num_frames=3,
            prompt_frame_idx=0,
            prompt_points=[
                {"obj_id": 1, "points": [[0.4, 0.5], [0.42, 0.6]], "labels": [1, 1]},
            ],
            name="test_point",
        )
        predictor = self._setup_node_with_mock(node, obj_ids=[1])
        results = self._run_node(node, num_frames=3)

        assert len(results) == 3
        predictor.add_new_points_or_box.assert_called_once()
        call_kwargs = predictor.add_new_points_or_box.call_args.kwargs
        assert call_kwargs["obj_id"] == 1
        assert call_kwargs["points"].shape == (2, 2)
        assert call_kwargs["labels"].shape == (2,)

    # -- Mask prompt ----------------------------------------------------------

    def test_mask_prompt(self, tmp_path: Path) -> None:
        mask_img = np.zeros((10, 12), dtype=np.uint8)
        mask_img[2:8, 3:9] = 255
        mask_path = tmp_path / "mask.png"
        cv2.imwrite(str(mask_path), mask_img)

        node = SAM3ObjectTracker(
            num_frames=3,
            prompt_frame_idx=0,
            prompt_masks=[{"obj_id": 3, "mask_path": str(mask_path)}],
            name="test_mask",
        )
        predictor = self._setup_node_with_mock(node, obj_ids=[3])
        results = self._run_node(node, num_frames=3)

        assert len(results) == 3
        predictor.add_new_mask.assert_called_once()
        call_kwargs = predictor.add_new_mask.call_args.kwargs
        assert call_kwargs["obj_id"] == 3
        assert call_kwargs["mask"].dim() == 2

    # -- Mixed prompts --------------------------------------------------------

    def test_mixed_prompts(self) -> None:
        node = SAM3ObjectTracker(
            num_frames=3,
            prompt_frame_idx=0,
            prompt_bboxes=[{"obj_id": 1, "bbox_xywh": [0.1, 0.2, 0.3, 0.4]}],
            prompt_points=[{"obj_id": 1, "points": [[0.25, 0.4]], "labels": [1]}],
            name="test_mixed",
        )
        predictor = self._setup_node_with_mock(node, obj_ids=[1])
        results = self._run_node(node, num_frames=3)

        assert len(results) == 3
        # bbox + point refinement for same obj_id
        assert predictor.add_new_points_or_box.call_count == 2

    # -- Non-zero prompt_frame_idx --------------------------------------------

    def test_non_zero_prompt_frame_idx(self) -> None:
        node = SAM3ObjectTracker(
            num_frames=5,
            prompt_frame_idx=50,
            prompt_bboxes=[{"obj_id": 1, "bbox_xywh": [0.1, 0.2, 0.3, 0.4]}],
            name="test_prompt_frame_50",
        )
        predictor = self._setup_node_with_mock(node, obj_ids=[1])
        # mesu_index starts at 48: frames 48, 49, 50, 51, 52
        results = self._run_node(node, num_frames=5, start_mesu=48)

        assert len(results) == 5
        # Frames 48, 49: pre-prompt → empty
        assert results[0]["object_ids"].shape[1] == 0
        assert results[1]["object_ids"].shape[1] == 0
        # Frame 50: prompt frame → tracked objects
        assert results[2]["object_ids"].shape[1] == 1
        # Frames 51, 52: post-prompt → tracked objects
        assert results[3]["object_ids"].shape[1] == 1
        assert results[4]["object_ids"].shape[1] == 1

        # Prompts applied on internal frame 0
        call_kwargs = predictor.add_new_points_or_box.call_args.kwargs
        assert call_kwargs["frame_idx"] == 0

        # propagate_in_video called with start_frame_idx=0
        prop_kwargs = predictor.propagate_in_video.call_args.kwargs
        assert prop_kwargs["start_frame_idx"] == 0

    # -- Output shapes and dtypes ---------------------------------------------

    def test_output_shapes_single_object(self) -> None:
        node = SAM3ObjectTracker(
            num_frames=3,
            prompt_frame_idx=0,
            prompt_bboxes=[{"obj_id": 1, "bbox_xywh": [0.1, 0.2, 0.3, 0.4]}],
            name="test_shapes_single",
        )
        self._setup_node_with_mock(node, obj_ids=[1], h=10, w=12)
        results = self._run_node(node, num_frames=3, h=10, w=12)

        for r in results:
            assert r["mask"].shape == (1, 10, 12)
            assert r["mask"].dtype == torch.int32
            assert r["object_ids"].shape == (1, 1)
            assert r["object_ids"].dtype == torch.int64
            assert r["detection_scores"].shape == (1, 1)
            assert r["detection_scores"].dtype == torch.float32

    def test_output_shapes_multi_object(self) -> None:
        node = SAM3ObjectTracker(
            num_frames=3,
            prompt_frame_idx=0,
            prompt_bboxes=[
                {"obj_id": 1, "bbox_xywh": [0.1, 0.2, 0.3, 0.4]},
                {"obj_id": 2, "bbox_xywh": [0.5, 0.1, 0.2, 0.3]},
            ],
            name="test_shapes_multi",
        )
        self._setup_node_with_mock(node, obj_ids=[1, 2], h=10, w=12)
        results = self._run_node(node, num_frames=3, h=10, w=12)

        for r in results:
            assert r["mask"].shape == (1, 10, 12)
            assert r["object_ids"].shape == (1, 2)
            assert r["detection_scores"].shape == (1, 2)

    # -- Pre-prompt frames: empty outputs -------------------------------------

    def test_pre_prompt_frames_empty(self) -> None:
        node = SAM3ObjectTracker(
            num_frames=5,
            prompt_frame_idx=3,
            prompt_bboxes=[{"obj_id": 1, "bbox_xywh": [0.1, 0.2, 0.3, 0.4]}],
            name="test_pre_prompt_empty",
        )
        self._setup_node_with_mock(node, obj_ids=[1], h=8, w=10)

        # Only send 3 pre-prompt frames (mesu_index 0, 1, 2)
        for i in range(3):
            rgb = torch.rand(1, 8, 10, 3, dtype=torch.float32)
            frame_id = torch.tensor([i], dtype=torch.int64)
            result = node.forward(rgb, frame_id)

            assert result["mask"].shape == (1, 8, 10)
            assert result["mask"].dtype == torch.int32
            assert (result["mask"] == 0).all()
            assert result["object_ids"].shape == (1, 0)
            assert result["object_ids"].dtype == torch.int64
            assert result["detection_scores"].shape == (1, 0)
            assert result["detection_scores"].dtype == torch.float32

    # -- Missed prompt frame: close() warns -----------------------------------

    def test_missed_prompt_frame_close_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        node = SAM3ObjectTracker(
            num_frames=3,
            prompt_frame_idx=99,
            prompt_bboxes=[{"obj_id": 1, "bbox_xywh": [0.1, 0.2, 0.3, 0.4]}],
            name="test_missed_prompt",
        )
        self._setup_node_with_mock(node, obj_ids=[1])
        # Run with mesu_index 0, 1, 2 — prompt frame 99 never arrives
        self._run_node(node, num_frames=3, start_mesu=0)

        with caplog.at_level("WARNING"):
            node.close()

        assert node._prompt_frame_seen is False

    # -- Output specs ---------------------------------------------------------

    def test_output_specs(self) -> None:
        specs = SAM3ObjectTracker.OUTPUT_SPECS
        assert "mask" in specs
        assert "object_ids" in specs
        assert "detection_scores" in specs
        assert "frame_id" not in specs  # no frame_id output
        assert specs["mask"].dtype == torch.int32
        assert specs["object_ids"].dtype == torch.int64
        assert specs["detection_scores"].dtype == torch.float32

    def test_input_specs(self) -> None:
        specs = SAM3ObjectTracker.INPUT_SPECS
        assert "rgb_frame" in specs
        assert "frame_id" in specs
        assert specs["frame_id"].dtype == torch.int64

    # -- Validation -----------------------------------------------------------

    def test_no_prompts_raises(self) -> None:
        with pytest.raises(ValueError, match="(?i)at least one"):
            SAM3ObjectTracker(
                num_frames=5,
                name="test_no_prompts",
            )

    def test_num_frames_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="num_frames"):
            SAM3ObjectTracker(
                num_frames=0,
                prompt_bboxes=[{"obj_id": 1, "bbox_xywh": [0.1, 0.2, 0.3, 0.4]}],
                name="test_zero",
            )

    # -- Single generator called once -----------------------------------------

    def test_single_generator_called_once(self) -> None:
        node = SAM3ObjectTracker(
            num_frames=5,
            prompt_frame_idx=0,
            prompt_bboxes=[{"obj_id": 1, "bbox_xywh": [0.1, 0.2, 0.3, 0.4]}],
            name="test_single_gen",
        )
        predictor = self._setup_node_with_mock(node, obj_ids=[1])
        self._run_node(node, num_frames=5)
        assert predictor.propagate_in_video.call_count == 1

    # -- Generator exhausted returns empty ------------------------------------

    def test_generator_exhausted_returns_empty(self) -> None:
        node = SAM3ObjectTracker(
            num_frames=5,
            prompt_frame_idx=0,
            prompt_bboxes=[{"obj_id": 1, "bbox_xywh": [0.1, 0.2, 0.3, 0.4]}],
            name="test_exhausted",
        )
        predictor = self._setup_node_with_mock(node, obj_ids=[1])
        # Override generator to only yield 2 frames instead of 5
        predictor.propagate_in_video.side_effect = lambda state, **kw: _make_tracker_generator(
            2, [1], 10, 12
        )
        results = self._run_node(node, num_frames=5, h=10, w=12)

        # First 2 frames have objects, last 3 are empty
        assert results[0]["object_ids"].shape[1] == 1
        assert results[1]["object_ids"].shape[1] == 1
        assert results[2]["object_ids"].shape[1] == 0
        assert results[3]["object_ids"].shape[1] == 0
        assert results[4]["object_ids"].shape[1] == 0
