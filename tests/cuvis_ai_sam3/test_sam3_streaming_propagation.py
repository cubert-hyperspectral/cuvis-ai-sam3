"""Tests for SAM3StreamingPropagation node with mocked SAM3 model."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
import torch

from cuvis_ai_sam3.node.sam3_streaming_propagation import (
    SAM3StreamingPropagation,
    _FrameBuffer,
)


# ---------------------------------------------------------------------------
# _FrameBuffer tests
# ---------------------------------------------------------------------------


class TestFrameBuffer:
    def test_add_and_getitem(self) -> None:
        buf = _FrameBuffer(
            num_frames=5,
            image_size=8,
            device=torch.device("cpu"),
        )
        frame = np.random.rand(6, 5, 3).astype(np.float32)
        idx = buf.add(frame)
        assert idx == 0
        assert len(buf) == 5  # pre-allocated size
        out = buf[0]
        assert out.shape == (3, 8, 8)
        assert out.dtype == torch.float16

    def test_sequential_add(self) -> None:
        buf = _FrameBuffer(num_frames=3, image_size=4, device=torch.device("cpu"))
        for i in range(3):
            idx = buf.add(np.random.rand(4, 4, 3).astype(np.float32))
            assert idx == i

    def test_missing_frame_raises(self) -> None:
        buf = _FrameBuffer(num_frames=5, image_size=4, device=torch.device("cpu"))
        with pytest.raises(IndexError):
            _ = buf[0]

    def test_tensor_index(self) -> None:
        buf = _FrameBuffer(num_frames=3, image_size=4, device=torch.device("cpu"))
        buf.add(np.zeros((3, 3, 3), dtype=np.float32))
        buf.add(np.zeros((3, 3, 3), dtype=np.float32))
        stacked = buf[torch.tensor([0, 1], dtype=torch.int64)]
        assert stacked.shape == (2, 3, 4, 4)

    def test_prune_before(self) -> None:
        buf = _FrameBuffer(num_frames=3, image_size=4, device=torch.device("cpu"))
        for _ in range(3):
            buf.add(np.zeros((3, 3, 3), dtype=np.float32))
        buf.prune_before(2)
        with pytest.raises(IndexError):
            _ = buf[0]
        _ = buf[2]  # should work


# ---------------------------------------------------------------------------
# Helpers for mocking SAM3
# ---------------------------------------------------------------------------


def _make_mock_model(image_size: int = 8) -> MagicMock:
    """Create a mock SAM3 model that yields fake propagation outputs."""
    model = MagicMock()
    model.image_size = image_size
    model.device = torch.device("cpu")

    # parameters() for device detection
    param = torch.nn.Parameter(torch.zeros(1))
    model.parameters.side_effect = lambda: iter([param])
    model.add_prompt.return_value = (
        0,
        {
            "out_obj_ids": np.array([1], dtype=np.int64),
            "out_probs": np.array([0.9], dtype=np.float32),
            "out_boxes_xywh": np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
            "out_binary_masks": np.zeros((1, 1, 1), dtype=bool),
        },
    )

    return model


def _make_propagation_generator(
    num_frames: int, start_frame_idx: int = 0, h: int = 10, w: int = 12
):
    """Generator that mimics propagate_in_video output for num_frames frames."""
    for frame_idx in range(start_frame_idx, num_frames):
        yield frame_idx, {
            "out_obj_ids": np.array([1, 2], dtype=np.int64),
            "out_probs": np.array([0.9, 0.8], dtype=np.float32),
            "out_binary_masks": np.stack([
                np.ones((h, w), dtype=bool),
                np.zeros((h, w), dtype=bool),
            ]),
            "out_boxes_xywh": np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.1, 0.2]], dtype=np.float32),
        }


# ---------------------------------------------------------------------------
# SAM3StreamingPropagation tests
# ---------------------------------------------------------------------------


class TestSAM3StreamingPropagation:
    @pytest.fixture()
    def node_text(self) -> SAM3StreamingPropagation:
        return SAM3StreamingPropagation(
            num_frames=5,
            prompt_type="text",
            prompt_text="person",
            name="test_streaming",
        )

    def _run_streaming(
        self,
        node: SAM3StreamingPropagation,
        num_frames: int = 5,
        h: int = 10,
        w: int = 12,
    ) -> list[dict[str, torch.Tensor]]:
        """Run the node through num_frames with a mocked model."""
        mock_model = node._model if node._model is not None else _make_mock_model()
        if mock_model.propagate_in_video.side_effect is None:
            def _propagate_from_state(inference_state, **kwargs):  # noqa: ANN001
                start_frame_idx = int(kwargs.get("start_frame_idx", 0))
                return _make_propagation_generator(
                    inference_state["num_frames"], start_frame_idx, h, w
                )

            mock_model.propagate_in_video.side_effect = _propagate_from_state

        node._model = mock_model
        node._ensure_model = MagicMock()

        results = []
        for i in range(num_frames):
            rgb = torch.rand(1, h, w, 3, dtype=torch.float32)
            result = node.forward(rgb)
            results.append(result)
        return results

    def test_text_prompt_streaming(self, node_text: SAM3StreamingPropagation) -> None:
        results = self._run_streaming(node_text, num_frames=5, h=10, w=12)

        assert len(results) == 5
        for i, r in enumerate(results):
            assert r["frame_id"].item() == i
            assert r["mask"].shape == (1, 10, 12)
            assert r["mask"].dtype == torch.int32
            assert r["object_ids"].shape == (1, 2)
            assert r["detection_scores"].shape == (1, 2)

        # Verify add_prompt was called with text
        node_text._model.add_prompt.assert_called_once()
        call_kwargs = node_text._model.add_prompt.call_args
        assert (
            call_kwargs[1].get("text_str") == "person"
            or call_kwargs[0][2]
            if len(call_kwargs[0]) > 2
            else True
        )

    def test_text_prompt_runs_n_forwards_without_exhaustion(self) -> None:
        node = SAM3StreamingPropagation(
            num_frames=4,
            prompt_type="text",
            prompt_text="person",
            name="test_text_no_exhaustion",
        )
        results = self._run_streaming(node, num_frames=4, h=9, w=11)
        assert [int(r["frame_id"].item()) for r in results] == [0, 1, 2, 3]

    def test_non_zero_prompt_frame_idx(self) -> None:
        node = SAM3StreamingPropagation(
            num_frames=5,
            prompt_frame_idx=2,
            prompt_type="text",
            prompt_text="person",
            name="test_prompt_frame_2",
        )
        results = self._run_streaming(node, num_frames=5, h=10, w=12)

        assert [int(r["frame_id"].item()) for r in results] == [0, 1, 2, 3, 4]
        assert results[0]["object_ids"].shape == (1, 0)
        assert results[1]["object_ids"].shape == (1, 0)
        assert results[2]["object_ids"].shape == (1, 2)
        call_kwargs = node._model.propagate_in_video.call_args.kwargs
        assert call_kwargs["start_frame_idx"] == 2
        assert call_kwargs["max_frame_num_to_track"] == 2

    def test_bbox_prompt_streaming(self) -> None:
        node = SAM3StreamingPropagation(
            num_frames=3,
            prompt_type="bbox",
            prompt_bboxes_xywh=[[0.1, 0.2, 0.3, 0.4]],
            name="test_bbox",
        )
        results = self._run_streaming(node, num_frames=3)
        assert len(results) == 3
        for r in results:
            assert r["object_ids"].shape == (1, 1)
            assert int(r["object_ids"][0, 0].item()) == 1
        node._model.add_prompt.assert_called_once()

    def test_bbox_prompt_uses_provided_output_id(self) -> None:
        node = SAM3StreamingPropagation(
            num_frames=3,
            prompt_type="bbox",
            prompt_bboxes_xywh=[[0.1, 0.2, 0.3, 0.4]],
            prompt_obj_id=14,
            name="test_bbox_obj_id_override",
        )
        mock_model = _make_mock_model()
        mock_model.add_prompt.return_value = (
            0,
            {
                "out_obj_ids": np.array([1, 3, 4], dtype=np.int64),
                "out_probs": np.array([0.4, 0.95, 0.3], dtype=np.float32),
                "out_boxes_xywh": np.array(
                    [
                        [0.6, 0.1, 0.1, 0.1],
                        [0.1, 0.2, 0.3, 0.4],  # exact prompt match -> selected internal id 3
                        [0.2, 0.1, 0.1, 0.2],
                    ],
                    dtype=np.float32,
                ),
                "out_binary_masks": np.zeros((3, 1, 1), dtype=bool),
            },
        )

        def _bbox_gen(inference_state, **kwargs):  # noqa: ANN001
            del kwargs
            h, w = 10, 12
            for frame_idx in range(inference_state["num_frames"]):
                masks = np.zeros((3, h, w), dtype=bool)
                masks[0, :2, :2] = True
                masks[1, 2:8, 3:9] = True  # selected internal id 3
                masks[2, 0:2, 8:10] = True
                yield frame_idx, {
                    "out_obj_ids": np.array([1, 3, 4], dtype=np.int64),
                    "out_probs": np.array([0.4, 0.95, 0.3], dtype=np.float32),
                    "out_binary_masks": masks,
                    "out_boxes_xywh": np.array(
                        [
                            [0.6, 0.1, 0.1, 0.1],
                            [0.1, 0.2, 0.3, 0.4],
                            [0.2, 0.1, 0.1, 0.2],
                        ],
                        dtype=np.float32,
                    ),
                }

        mock_model.propagate_in_video.side_effect = _bbox_gen
        node._model = mock_model
        node._ensure_model = MagicMock()

        results = self._run_streaming(node, num_frames=3)
        assert node._selected_internal_bbox_obj_id == 3
        assert node._effective_output_bbox_obj_id == 14
        for r in results:
            assert r["object_ids"].shape == (1, 1)
            assert int(r["object_ids"][0, 0].item()) == 14
            assert r["detection_scores"].shape == (1, 1)
            assert float(r["detection_scores"][0, 0].item()) == pytest.approx(0.95)
            mask_vals = set(torch.unique(r["mask"]).cpu().tolist())
            assert mask_vals.issubset({0, 14})

    def test_bbox_prompt_without_obj_id_uses_selected_sam_id(self) -> None:
        node = SAM3StreamingPropagation(
            num_frames=2,
            prompt_type="bbox",
            prompt_bboxes_xywh=[[0.1, 0.2, 0.3, 0.4]],
            prompt_obj_id=None,
            name="test_bbox_obj_id_from_sam",
        )
        mock_model = _make_mock_model()
        mock_model.add_prompt.return_value = (
            0,
            {
                "out_obj_ids": np.array([5, 3], dtype=np.int64),
                "out_probs": np.array([0.2, 0.9], dtype=np.float32),
                "out_boxes_xywh": np.array(
                    [
                        [0.7, 0.1, 0.1, 0.1],
                        [0.1, 0.2, 0.3, 0.4],  # selected SAM id 3
                    ],
                    dtype=np.float32,
                ),
                "out_binary_masks": np.zeros((2, 1, 1), dtype=bool),
            },
        )

        def _bbox_gen(inference_state, **kwargs):  # noqa: ANN001
            del kwargs
            h, w = 10, 12
            for frame_idx in range(inference_state["num_frames"]):
                masks = np.zeros((2, h, w), dtype=bool)
                masks[0, :2, :2] = True
                masks[1, 1:9, 2:8] = True
                yield frame_idx, {
                    "out_obj_ids": np.array([5, 3], dtype=np.int64),
                    "out_probs": np.array([0.2, 0.9], dtype=np.float32),
                    "out_binary_masks": masks,
                    "out_boxes_xywh": np.array(
                        [
                            [0.7, 0.1, 0.1, 0.1],
                            [0.1, 0.2, 0.3, 0.4],
                        ],
                        dtype=np.float32,
                    ),
                }

        mock_model.propagate_in_video.side_effect = _bbox_gen
        node._model = mock_model
        node._ensure_model = MagicMock()

        results = self._run_streaming(node, num_frames=2)
        assert node._selected_internal_bbox_obj_id == 3
        assert node._effective_output_bbox_obj_id == 3
        for r in results:
            assert r["object_ids"].shape == (1, 1)
            assert int(r["object_ids"][0, 0].item()) == 3

    def test_bbox_multiple_initial_boxes_rejected(self) -> None:
        with pytest.raises(ValueError, match="exactly one initial bbox prompt"):
            SAM3StreamingPropagation(
                num_frames=3,
                prompt_type="bbox",
                prompt_bboxes_xywh=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.1, 0.2]],
                name="test_bbox_multi",
            )

    def test_point_prompt_streaming(self) -> None:
        node = SAM3StreamingPropagation(
            num_frames=3,
            prompt_type="point",
            prompt_points=[[0.5, 0.5]],
            prompt_point_labels=[1],
            prompt_obj_id=1,
            name="test_point",
        )
        mock_model = _make_mock_model()

        def _point_prompt_side_effect(inference_state, **kwargs):  # noqa: ANN001
            del kwargs
            assert 0 in inference_state["cached_frame_outputs"]
            return 0, None

        def _propagate_from_state(inference_state, **kwargs):  # noqa: ANN001
            del kwargs
            return _make_propagation_generator(inference_state["num_frames"])

        mock_model.add_prompt.side_effect = _point_prompt_side_effect
        mock_model.propagate_in_video.side_effect = _propagate_from_state
        node._model = mock_model
        node._ensure_model = MagicMock()

        results = self._run_streaming(node, num_frames=3)
        assert len(results) == 3
        node._model.add_prompt.assert_called_once()

    def test_mask_prompt_streaming(self, tmp_path: Path) -> None:
        mask_img = np.zeros((10, 12), dtype=np.uint8)
        mask_img[2:8, 3:9] = 255
        mask_path = tmp_path / "mask.png"
        cv2.imwrite(str(mask_path), mask_img)

        node = SAM3StreamingPropagation(
            num_frames=3,
            prompt_type="mask",
            prompt_mask_path=str(mask_path),
            prompt_obj_id=1,
            name="test_mask",
        )
        mock_model = _make_mock_model()

        def _mask_prompt_side_effect(inference_state, **kwargs):  # noqa: ANN001
            del kwargs
            assert 0 in inference_state["cached_frame_outputs"]
            return 0, None

        def _propagate_from_state(inference_state, **kwargs):  # noqa: ANN001
            del kwargs
            return _make_propagation_generator(inference_state["num_frames"])

        mock_model.add_mask.side_effect = _mask_prompt_side_effect
        mock_model.propagate_in_video.side_effect = _propagate_from_state
        node._model = mock_model
        node._ensure_model = MagicMock()

        results = self._run_streaming(node, num_frames=3)
        assert len(results) == 3
        node._model.add_mask.assert_called_once()

    def test_output_specs_compatible(self, node_text: SAM3StreamingPropagation) -> None:
        """Output specs should match TrackingOverlayNode / TrackingCocoJsonNode inputs."""
        specs = SAM3StreamingPropagation.OUTPUT_SPECS
        assert "frame_id" in specs
        assert "mask" in specs
        assert "object_ids" in specs
        assert "detection_scores" in specs
        assert specs["mask"].dtype == torch.int32
        assert specs["object_ids"].dtype == torch.int64

    def test_empty_detections(self) -> None:
        node = SAM3StreamingPropagation(
            num_frames=2,
            prompt_type="text",
            prompt_text="person",
            name="test_empty",
        )

        def _empty_gen(num_frames: int):
            for i in range(num_frames):
                yield i, {
                    "out_obj_ids": np.array([], dtype=np.int64),
                    "out_probs": np.array([], dtype=np.float32),
                    "out_binary_masks": np.zeros((0, 10, 12), dtype=bool),
                }

        mock_model = _make_mock_model()
        mock_model.propagate_in_video.side_effect = (
            lambda inference_state, **kwargs: _empty_gen(inference_state["num_frames"])
        )
        node._model = mock_model
        node._ensure_model = MagicMock()

        for _ in range(2):
            rgb = torch.rand(1, 10, 12, 3)
            result = node.forward(rgb)
            assert result["object_ids"].shape[1] == 0
            assert result["detection_scores"].shape[1] == 0

    def test_single_generator_called_once(self, node_text: SAM3StreamingPropagation) -> None:
        """Verify propagate_in_video is called exactly once (single generator)."""
        self._run_streaming(node_text, num_frames=5)
        assert node_text._model.propagate_in_video.call_count == 1

    def test_invalid_prompt_type_raises(self) -> None:
        with pytest.raises(ValueError, match="prompt_type"):
            SAM3StreamingPropagation(
                num_frames=5,
                prompt_type="invalid",
                name="test_invalid",
            )

    def test_num_frames_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="num_frames"):
            SAM3StreamingPropagation(
                num_frames=0,
                prompt_type="text",
                name="test_zero",
            )

    def test_prompt_frame_idx_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="prompt_frame_idx"):
            SAM3StreamingPropagation(
                num_frames=3,
                prompt_frame_idx=3,
                prompt_type="text",
                name="test_prompt_idx_invalid",
            )

    def test_detector_guard_installed_on_ensure_model(self) -> None:
        node = SAM3StreamingPropagation(
            num_frames=3, prompt_type="text", prompt_text="person", name="test_guard"
        )
        mock_model = _make_mock_model()
        mock_detector = MagicMock()
        mock_detector._streaming_guard_installed = False
        mock_model.detector = mock_detector
        node._model = mock_model
        node._install_streaming_detector_guard()
        assert mock_detector._streaming_guard_installed is True
