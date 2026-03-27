"""Tests for SAM3 streaming propagation nodes with mocked SAM3 model."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from cuvis_ai_sam3.node.sam3_streaming_propagation import (
    SAM3BboxPropagation,
    SAM3MaskPropagation,
    SAM3PointPropagation,
    SAM3TextPropagation,
    SAM3TrackerInference,
    _FrameBuffer,
)

# ---------------------------------------------------------------------------
# _FrameBuffer tests
# ---------------------------------------------------------------------------


class TestFrameBuffer:
    def test_add_and_getitem(self) -> None:
        buf = _FrameBuffer(
            image_size=8,
            device=torch.device("cpu"),
        )
        frame = np.random.rand(6, 5, 3).astype(np.float32)
        idx = buf.add(frame)
        assert idx == 0
        assert len(buf) == 1  # buffered count
        out = buf[0]
        assert out.shape == (3, 8, 8)
        assert out.dtype == torch.float16

    def test_sequential_add(self) -> None:
        buf = _FrameBuffer(image_size=4, device=torch.device("cpu"))
        for i in range(3):
            idx = buf.add(np.random.rand(4, 4, 3).astype(np.float32))
            assert idx == i
        assert len(buf) == 3

    def test_missing_frame_raises(self) -> None:
        buf = _FrameBuffer(image_size=4, device=torch.device("cpu"))
        with pytest.raises(IndexError):
            _ = buf[0]

    def test_tensor_index(self) -> None:
        buf = _FrameBuffer(image_size=4, device=torch.device("cpu"))
        buf.add(np.zeros((3, 3, 3), dtype=np.float32))
        buf.add(np.zeros((3, 3, 3), dtype=np.float32))
        stacked = buf[torch.tensor([0, 1], dtype=torch.int64)]
        assert stacked.shape == (2, 3, 4, 4)

    def test_prune_before(self) -> None:
        buf = _FrameBuffer(image_size=4, device=torch.device("cpu"))
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
    model.add_mask.return_value = (0, None)

    def _propagate_from_state(_inference_state, **kwargs):  # noqa: ANN001
        start_frame_idx = int(kwargs.get("start_frame_idx", 0))
        return _make_propagation_generator(start_frame_idx=start_frame_idx, h=10, w=12)

    model.propagate_in_video.side_effect = _propagate_from_state

    return model


def _make_propagation_generator(start_frame_idx: int = 0, h: int = 10, w: int = 12):
    """Generator that mimics propagate_in_video output for a streaming sequence."""
    frame_idx = start_frame_idx
    while True:
        yield (
            frame_idx,
            {
                "out_obj_ids": np.array([1, 2], dtype=np.int64),
                "out_probs": np.array([0.9, 0.8], dtype=np.float32),
                "out_binary_masks": np.stack(
                    [
                        np.ones((h, w), dtype=bool),
                        np.zeros((h, w), dtype=bool),
                    ]
                ),
                "out_boxes_xywh": np.array(
                    [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.1, 0.2]], dtype=np.float32
                ),
            },
        )
        frame_idx += 1


def _run_streaming(
    node: SAM3TrackerInference,
    num_frames: int = 5,
    h: int = 10,
    w: int = 12,
    frame_ids: list[int] | None = None,
    masks: list[torch.Tensor | None] | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Run the node through num_frames with a mocked model."""
    mock_model = node._model if node._model is not None else _make_mock_model()
    if mock_model.propagate_in_video.side_effect is None:

        def _propagate_from_state(_inference_state, **kwargs):  # noqa: ANN001
            start_frame_idx = int(kwargs.get("start_frame_idx", 0))
            return _make_propagation_generator(start_frame_idx=start_frame_idx, h=h, w=w)

        mock_model.propagate_in_video.side_effect = _propagate_from_state

    node._model = mock_model
    node._ensure_model = MagicMock()

    results = []
    if frame_ids is not None and len(frame_ids) != num_frames:
        raise ValueError("frame_ids length must match num_frames.")
    if masks is not None and len(masks) != num_frames:
        raise ValueError("masks length must match num_frames.")
    for i in range(num_frames):
        rgb = torch.rand(1, h, w, 3, dtype=torch.float32)
        frame_id_t = (
            torch.tensor([int(frame_ids[i])], dtype=torch.int64) if frame_ids is not None else None
        )
        mask_t = masks[i] if masks is not None else None
        result = node.forward(rgb, frame_id=frame_id_t, mask=mask_t)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# SAM3TextPropagation tests
# ---------------------------------------------------------------------------


class TestSAM3TextPropagation:
    @pytest.fixture()
    def node_text(self) -> SAM3TextPropagation:
        return SAM3TextPropagation(
            prompt_text="person",
            name="test_streaming",
        )

    def test_text_prompt_streaming(self, node_text: SAM3TextPropagation) -> None:
        results = _run_streaming(node_text, num_frames=5, h=10, w=12)

        assert len(results) == 5
        for r in results:
            assert r["mask"].shape == (1, 10, 12)
            assert r["mask"].dtype == torch.int32
            assert r["object_ids"].shape == (1, 2)
            assert r["detection_scores"].shape == (1, 2)

        # Verify add_prompt was called exactly once on first frame
        node_text._model.add_prompt.assert_called_once()
        call_kwargs = node_text._model.add_prompt.call_args.kwargs
        assert call_kwargs["frame_idx"] == 0
        assert call_kwargs["text_str"] == "person"

    def test_prompt_applied_on_first_frame(self) -> None:
        node = SAM3TextPropagation(prompt_text="person", name="test_first_frame_prompt")
        mock_model = _make_mock_model()
        node._model = mock_model
        node._ensure_model = MagicMock()

        _run_streaming(node, num_frames=1, h=10, w=12)

        mock_model.add_prompt.assert_called_once()
        assert mock_model.add_prompt.call_args.kwargs["frame_idx"] == 0

    def test_frame_id_port_sets_source_mapping(self) -> None:
        node = SAM3TextPropagation(prompt_text="person", name="test_frame_id_mapping")
        frame_ids = [100, 101, 102, 103, 104]
        _run_streaming(node, num_frames=5, h=10, w=12, frame_ids=frame_ids)
        assert node._source_frame_ids == frame_ids

    def test_fallback_stream_idx_when_no_frame_id(self) -> None:
        node = SAM3TextPropagation(prompt_text="person", name="test_fallback_stream_idx")
        _run_streaming(node, num_frames=4, h=10, w=12, frame_ids=None)
        assert node._source_frame_ids == [0, 1, 2, 3]

    def test_text_prompt_remaps_zero_internal_id(self) -> None:
        node = SAM3TextPropagation(
            prompt_text="person",
            name="test_text_id_remap",
        )
        mock_model = _make_mock_model()

        def _gen(_inference_state, **kwargs):  # noqa: ANN001
            start_idx = int(kwargs.get("start_frame_idx", 0))
            h, w = 10, 12
            frame_idx = start_idx
            while True:
                masks = np.zeros((2, h, w), dtype=bool)
                masks[0, 1:5, 1:5] = True
                masks[1, 5:9, 6:10] = True
                yield (
                    frame_idx,
                    {
                        "out_obj_ids": np.array([0, 1], dtype=np.int64),
                        "out_probs": np.array([0.91, 0.83], dtype=np.float32),
                        "out_binary_masks": masks,
                        "out_boxes_xywh": np.array(
                            [
                                [0.08, 0.10, 0.30, 0.30],
                                [0.50, 0.50, 0.30, 0.30],
                            ],
                            dtype=np.float32,
                        ),
                    },
                )
                frame_idx += 1

        mock_model.propagate_in_video.side_effect = _gen
        node._model = mock_model
        node._ensure_model = MagicMock()

        results = _run_streaming(node, num_frames=3, h=10, w=12)
        first_ids: list[int] | None = None
        for result in results:
            obj_ids = [int(v) for v in result["object_ids"][0].cpu().tolist()]
            assert len(obj_ids) == 2
            assert all(v > 0 for v in obj_ids)
            assert len(set(obj_ids)) == 2
            if first_ids is None:
                first_ids = obj_ids
            else:
                assert obj_ids == first_ids

            mask_values = {int(v) for v in torch.unique(result["mask"]).cpu().tolist()}
            assert 0 in mask_values
            for obj_id in obj_ids:
                assert obj_id in mask_values

    def test_single_generator_called_once(self, node_text: SAM3TextPropagation) -> None:
        _run_streaming(node_text, num_frames=5)
        assert node_text._model.propagate_in_video.call_count == 1
        call_kwargs = node_text._model.propagate_in_video.call_args.kwargs
        assert call_kwargs["start_frame_idx"] == 0
        assert call_kwargs["max_frame_num_to_track"] is None


# ---------------------------------------------------------------------------
# SAM3BboxPropagation tests
# ---------------------------------------------------------------------------


class TestSAM3BboxPropagation:
    def test_bbox_prompt_streaming(self) -> None:
        node = SAM3BboxPropagation(
            prompt_bboxes_xywh=[[0.1, 0.2, 0.3, 0.4]],
            name="test_bbox",
        )
        results = _run_streaming(node, num_frames=3)
        assert len(results) == 3
        for r in results:
            assert r["object_ids"].shape == (1, 1)
            assert int(r["object_ids"][0, 0].item()) == 1
        node._model.add_prompt.assert_called_once()
        assert node._model.add_prompt.call_args.kwargs["frame_idx"] == 0

    def test_bbox_prompt_uses_provided_output_id(self) -> None:
        node = SAM3BboxPropagation(
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

        def _bbox_gen(_inference_state, **kwargs):  # noqa: ANN001
            del kwargs
            h, w = 10, 12
            frame_idx = 0
            while True:
                masks = np.zeros((3, h, w), dtype=bool)
                masks[0, :2, :2] = True
                masks[1, 2:8, 3:9] = True  # selected internal id 3
                masks[2, 0:2, 8:10] = True
                yield (
                    frame_idx,
                    {
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
                    },
                )
                frame_idx += 1

        mock_model.propagate_in_video.side_effect = _bbox_gen
        node._model = mock_model
        node._ensure_model = MagicMock()

        results = _run_streaming(node, num_frames=3)
        assert node._selected_internal_bbox_obj_id == 3
        assert node._effective_output_bbox_obj_id == 14
        for r in results:
            assert r["object_ids"].shape == (1, 1)
            assert int(r["object_ids"][0, 0].item()) == 14
            assert r["detection_scores"].shape == (1, 1)
            assert float(r["detection_scores"][0, 0].item()) == pytest.approx(0.95)
            mask_vals = set(torch.unique(r["mask"]).cpu().tolist())
            assert mask_vals.issubset({0, 14})


# ---------------------------------------------------------------------------
# SAM3PointPropagation tests
# ---------------------------------------------------------------------------


class TestSAM3PointPropagation:
    def test_point_prompt_streaming(self) -> None:
        node = SAM3PointPropagation(
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

        mock_model.add_prompt.side_effect = _point_prompt_side_effect
        node._model = mock_model
        node._ensure_model = MagicMock()

        results = _run_streaming(node, num_frames=3)
        assert len(results) == 3
        node._model.add_prompt.assert_called_once()
        assert node._model.add_prompt.call_args.kwargs["frame_idx"] == 0


# ---------------------------------------------------------------------------
# SAM3MaskPropagation tests
# ---------------------------------------------------------------------------


class TestSAM3MaskPropagation:
    @staticmethod
    def _mask_from_labels(label_map: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(label_map, dtype=np.int32)).unsqueeze(0)

    def test_mask_prompt_streaming(self) -> None:
        mask = np.zeros((10, 12), dtype=np.int32)
        mask[2:8, 3:9] = 1

        node = SAM3MaskPropagation(name="test_mask")
        mock_model = _make_mock_model()

        def _mask_prompt_side_effect(inference_state, **kwargs):  # noqa: ANN001
            del kwargs
            assert 0 in inference_state["cached_frame_outputs"]
            return 0, None

        mock_model.add_mask.side_effect = _mask_prompt_side_effect
        node._model = mock_model
        node._ensure_model = MagicMock()

        results = _run_streaming(
            node,
            num_frames=3,
            masks=[self._mask_from_labels(mask), None, None],
        )
        assert len(results) == 3
        node._model.add_mask.assert_called_once()
        assert node._model.add_mask.call_args.kwargs["frame_idx"] == 0
        assert node._model.add_mask.call_args.kwargs["obj_id"] == 1
        node._model.add_prompt.assert_called_once()
        assert node._model.add_prompt.call_args.kwargs["frame_idx"] == 0
        assert node._model.add_prompt.call_args.kwargs["obj_id"] == 1

    def test_no_mask_returns_full_frame_empty_output_and_skips_model_init(self) -> None:
        node = SAM3MaskPropagation(name="test_mask_no_seed")
        node._ensure_model = MagicMock()

        result = node.forward(torch.rand(1, 10, 12, 3, dtype=torch.float32), mask=None)

        assert result["mask"].shape == (1, 10, 12)
        assert torch.count_nonzero(result["mask"]).item() == 0
        assert result["object_ids"].shape == (1, 0)
        assert result["detection_scores"].shape == (1, 0)
        node._ensure_model.assert_not_called()

    def test_first_mask_lazily_initializes_model(self) -> None:
        node = SAM3MaskPropagation(name="test_mask_lazy_init")
        mock_model = _make_mock_model()

        def _ensure_model() -> None:
            node._model = mock_model

        node._ensure_model = MagicMock(side_effect=_ensure_model)
        empty_rgb = torch.rand(1, 10, 12, 3, dtype=torch.float32)
        prompt_mask = torch.zeros(1, 10, 12, dtype=torch.int32)
        prompt_mask[:, 2:8, 3:9] = 4

        first = node.forward(empty_rgb, mask=None, frame_id=torch.tensor([65], dtype=torch.int64))
        second = node.forward(empty_rgb, mask=None, frame_id=torch.tensor([66], dtype=torch.int64))
        seeded = node.forward(
            empty_rgb,
            mask=prompt_mask,
            frame_id=torch.tensor([67], dtype=torch.int64),
        )

        assert node._ensure_model.call_count == 1
        assert first["mask"].shape == (1, 10, 12)
        assert second["mask"].shape == (1, 10, 12)
        assert seeded["mask"].shape == (1, 10, 12)
        assert node._seed_source_stream_idx == 2
        assert node._source_frame_ids == [65, 66, 67]
        assert mock_model.add_mask.call_count == 1
        assert mock_model.add_mask.call_args.kwargs["frame_idx"] == 0
        assert mock_model.add_mask.call_args.kwargs["obj_id"] == 4

    def test_later_mask_updates_use_current_internal_frame(self) -> None:
        node = SAM3MaskPropagation(name="test_mask_update_frame_idx")
        mock_model = _make_mock_model()
        node._model = mock_model
        node._ensure_model = MagicMock()

        first_mask = torch.zeros(1, 10, 12, dtype=torch.int32)
        first_mask[:, 1:4, 2:6] = 2
        later_mask = torch.zeros(1, 10, 12, dtype=torch.int32)
        later_mask[:, 5:9, 7:10] = 7

        _run_streaming(
            node,
            num_frames=4,
            h=10,
            w=12,
            frame_ids=[10, 11, 12, 13],
            masks=[None, first_mask, None, later_mask],
        )

        add_mask_calls = mock_model.add_mask.call_args_list
        assert len(add_mask_calls) == 2
        assert add_mask_calls[0].kwargs["frame_idx"] == 0
        assert add_mask_calls[0].kwargs["obj_id"] == 2
        assert add_mask_calls[1].kwargs["frame_idx"] == 2
        assert add_mask_calls[1].kwargs["obj_id"] == 7

    def test_multi_label_mask_uses_raw_object_ids(self) -> None:
        node = SAM3MaskPropagation(name="test_mask_multi_label")
        mock_model = _make_mock_model()
        node._model = mock_model
        node._ensure_model = MagicMock()

        prompt_mask = torch.zeros(1, 10, 12, dtype=torch.int32)
        prompt_mask[:, 1:4, 1:4] = 5
        prompt_mask[:, 5:8, 6:10] = 9

        node.forward(torch.rand(1, 10, 12, 3, dtype=torch.float32), mask=prompt_mask)

        add_mask_calls = mock_model.add_mask.call_args_list
        add_prompt_calls = mock_model.add_prompt.call_args_list
        assert [call.kwargs["obj_id"] for call in add_mask_calls] == [5, 9]
        assert [call.kwargs["frame_idx"] for call in add_mask_calls] == [0, 0]
        assert [call.kwargs["obj_id"] for call in add_prompt_calls] == [5, 9]
        assert [call.kwargs["frame_idx"] for call in add_prompt_calls] == [0, 0]

    def test_mask_input_spec_is_optional_and_constructor_is_runtime_only(self) -> None:
        node = SAM3MaskPropagation(name="test_mask_specs")

        assert "mask" in SAM3MaskPropagation.INPUT_SPECS
        assert SAM3MaskPropagation.INPUT_SPECS["mask"].optional is True
        assert "prompt_mask_path" not in node.hparams
        assert "prompt_obj_id" not in node.hparams


# ---------------------------------------------------------------------------
# Validation & shared behavior tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_output_specs_compatible(self) -> None:
        """Output specs should match TrackingOverlayNode / TrackingCocoJsonNode inputs."""
        for cls in [
            SAM3TextPropagation,
            SAM3BboxPropagation,
            SAM3PointPropagation,
            SAM3MaskPropagation,
        ]:
            specs = cls.OUTPUT_SPECS
            assert "mask" in specs
            assert "object_ids" in specs
            assert "detection_scores" in specs
            assert specs["mask"].dtype == torch.int32
            assert specs["object_ids"].dtype == torch.int64

    def test_detector_guard_installed_on_ensure_model(self) -> None:
        node = SAM3TextPropagation(prompt_text="person", name="test_guard")
        mock_model = _make_mock_model()
        mock_detector = MagicMock()
        mock_detector._streaming_guard_installed = False
        mock_model.detector = mock_detector
        node._model = mock_model
        node._install_streaming_detector_guard()
        assert mock_detector._streaming_guard_installed is True
