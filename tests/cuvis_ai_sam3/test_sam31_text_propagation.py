"""Tests for the SAM 3.1 text propagation node with a stand-in for the multiplex model."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from cuvis_ai_sam3.node import sam31_text_propagation as node_module
from cuvis_ai_sam3.node.sam3_streaming_propagation import SAM3TextPropagation
from cuvis_ai_sam3.node.sam31_text_propagation import (
    SAM31TextPropagation,
    _FrameBuffer31,
    _FrameStages,
    _missing_checkpoint_keys,
)

pytestmark = pytest.mark.unit

HEIGHT, WIDTH = 10, 12
SAMPLE_VIDEO = Path(__file__).resolve().parents[2] / "assets" / "videos" / "0001"


class _FakeMultiplexModel:
    """Stands in for ``Sam3MultiplexTrackingWithInteractivity``: same call surface, tiny tensors.

    ``add_prompt`` has the SAM 3.1 signature (no ``reset_state``) and starts over like the real
    model; ``propagate_in_video`` yields one frame per step, reads the frame and its find stage
    from the state (so a node that steps before buffering fails), and leaves the per-frame
    entries the real tracker leaves behind.
    """

    def __init__(self, image_size: int = 8, fail_at_frame: int | None = None) -> None:
        self.image_size = image_size
        self.fail_at_frame = fail_at_frame
        # Builder defaults the node has to change for a frame-by-frame stream.
        self.hotstart_delay = 15
        self.postprocess_batch_size = 16
        self.use_batched_grounding = True
        self.score_threshold_detection = 0.0
        self.new_det_thresh = 0.0
        self.det_nms_thresh = 0.0
        self.suppress_overlapping_based_on_recent_occlusion_threshold = 0.0
        self.tracker = SimpleNamespace(num_maskmem=7, max_obj_ptrs_in_encoder=16)
        self.detector_calls: list[dict[str, Any]] = []
        self.detector = SimpleNamespace(forward_video_grounding_multigpu=self._detect)
        self._param = torch.nn.Parameter(torch.zeros(1))
        self.prompts: list[tuple[int, str]] = []
        self.steps: list[int] = []

    def parameters(self):  # noqa: ANN201 - mirrors nn.Module.parameters
        return iter([self._param])

    def _detect(self, **kwargs: Any) -> tuple[str, str]:
        self.detector_calls.append(kwargs)
        return "detections", "backbone"

    def _frame_output(self, state: dict[str, Any]) -> dict[str, torch.Tensor]:
        height, width = state["orig_height"], state["orig_width"]
        masks = torch.zeros(2, height, width, dtype=torch.bool)
        masks[0, : height // 2] = True
        masks[1, height // 2 :] = True
        return {
            "out_obj_ids": torch.tensor([0, 1], dtype=torch.int64),
            "out_probs": torch.tensor([0.9, 0.8], dtype=torch.float32),
            "out_binary_masks": masks,
            "out_boxes_xywh": torch.zeros(2, 4, dtype=torch.float32),
        }

    def add_prompt(
        self, inference_state: dict[str, Any], frame_idx: int, text_str: str | None = None
    ):
        assert 0 <= frame_idx < inference_state["num_frames"]
        self.prompts.append((int(frame_idx), str(text_str)))
        inference_state["text_prompt"] = text_str
        inference_state["tracker_metadata"].clear()
        inference_state["cached_frame_outputs"] = {}
        inference_state["sam2_inference_states"] = [
            {
                "num_frames": inference_state["num_frames"],
                "obj_ids": [0, 1],
                "output_dict": {"cond_frame_outputs": {0: {}}, "non_cond_frame_outputs": {}},
                "output_dict_per_obj": {
                    0: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
                },
                "temp_output_dict_per_obj": {
                    0: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
                },
                "consolidated_frame_inds": {
                    "cond_frame_outputs": {0},
                    "non_cond_frame_outputs": set(),
                },
                "frames_already_tracked": {},
                "mask_inputs_per_obj": {0: {0: torch.zeros(1)}},
                "point_inputs_per_obj": {0: {}},
            }
        ]
        inference_state["action_history"].append({"type": "add"})
        return frame_idx, self._frame_output(inference_state)

    def propagate_in_video(
        self,
        inference_state: dict[str, Any],
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
    ):
        assert max_frame_num_to_track is None and reverse is False
        frame_idx = int(start_frame_idx or 0)
        while True:
            if self.fail_at_frame is not None and frame_idx == self.fail_at_frame:
                raise RuntimeError("boom")
            batch = inference_state["input_batch"]
            assert batch.img_batch[frame_idx].shape == (3, self.image_size, self.image_size)
            assert batch.img_batch.tensors[frame_idx] is batch.img_batch[frame_idx]
            assert batch.find_inputs[frame_idx % len(batch.find_inputs)] is not None
            tracker_state = inference_state["sam2_inference_states"][0]
            tracker_state["output_dict"]["non_cond_frame_outputs"][frame_idx] = {"m": 1}
            tracker_state["output_dict_per_obj"][0]["non_cond_frame_outputs"][frame_idx] = {}
            tracker_state["frames_already_tracked"][frame_idx] = {"reverse": False}
            if frame_idx > 0 and frame_idx % 16 == 0:  # periodic reconditioning
                tracker_state["mask_inputs_per_obj"][0][frame_idx] = torch.zeros(1)
                tracker_state["consolidated_frame_inds"]["non_cond_frame_outputs"].add(frame_idx)
            metadata = inference_state["tracker_metadata"]
            metadata.setdefault("obj_id_to_sam2_score_frame_wise", {})[frame_idx] = {0: 0.9}
            metadata.setdefault("rank0_metadata", {}).setdefault("suppressed_obj_ids", {})[
                frame_idx
            ] = set()
            inference_state["cached_frame_outputs"][frame_idx] = {0: None}
            self.steps.append(frame_idx)
            yield frame_idx, self._frame_output(inference_state)
            frame_idx += 1


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> _FakeMultiplexModel:
    """The model the node gets instead of the 3.5 GB checkpoint."""
    model = _FakeMultiplexModel()
    builds = MagicMock(side_effect=lambda *_args, **_kwargs: model)
    monkeypatch.setattr(node_module, "_build_multiplex_model", builds)
    model.builds = builds  # type: ignore[attr-defined]
    return model


def _frame(height: int = HEIGHT, width: int = WIDTH) -> torch.Tensor:
    return torch.rand(1, height, width, 3, dtype=torch.float32)


def _semantics(output: dict[str, torch.Tensor]) -> dict[str, str]:
    return json.loads(bytes(output["category_semantics"].tolist()).decode("utf-8"))


def _run(node: SAM31TextPropagation, prompts: dict[int, str], n_frames: int) -> list[dict]:
    return [
        node.forward(
            rgb_image=_frame(),
            text_prompt=prompts.get(i),
            frame_id=torch.tensor([100 + i], dtype=torch.int64),
        )
        for i in range(n_frames)
    ]


# ---------------------------------------------------------------------------
# Port contract
# ---------------------------------------------------------------------------


def test_ports_are_those_of_sam3_text_propagation() -> None:
    for specs, reference in (
        (SAM31TextPropagation.INPUT_SPECS, SAM3TextPropagation.INPUT_SPECS),
        (SAM31TextPropagation.OUTPUT_SPECS, SAM3TextPropagation.OUTPUT_SPECS),
    ):
        assert set(specs) == set(reference)
        for name, spec in specs.items():
            other = reference[name]
            assert (spec.dtype, spec.shape, spec.optional) == (
                other.dtype,
                other.shape,
                other.optional,
            ), name


def test_node_is_exported_and_not_tagged_exportable() -> None:
    from cuvis_ai_schemas.enums import NodeTag

    from cuvis_ai_sam3.node import SAM31TextPropagation as exported

    assert exported is SAM31TextPropagation
    assert NodeTag.ONNX not in SAM31TextPropagation._tags


def test_frames_before_a_prompt_are_empty_and_build_nothing(fake_model) -> None:
    node = SAM31TextPropagation()
    (output,) = _run(node, {}, 1)

    assert output["mask"].shape == (1, HEIGHT, WIDTH)
    assert output["mask"].dtype == torch.int32
    assert int(output["mask"].sum()) == 0
    assert output["object_ids"].shape == (1, 0) and output["object_ids"].dtype == torch.int64
    assert output["detection_scores"].shape == (1, 0)
    assert output["detection_scores"].dtype == torch.float32
    assert output["category_ids"].shape == (1, 0) and output["category_ids"].dtype == torch.int64
    assert output["category_semantics"].dtype == torch.uint8
    assert _semantics(output) == {}
    fake_model.builds.assert_not_called()


# ---------------------------------------------------------------------------
# Prompting and tracking
# ---------------------------------------------------------------------------


def test_first_prompt_starts_tracking(fake_model) -> None:
    node = SAM31TextPropagation()
    (output,) = _run(node, {0: "person"}, 1)

    assert fake_model.prompts == [(0, "person")]
    assert output["mask"].shape == (1, HEIGHT, WIDTH) and output["mask"].dtype == torch.int32
    # Internal ids 0 and 1 leave as 1 and 2: 0 is the background of the label map.
    assert output["object_ids"].tolist() == [[1, 2]]
    assert sorted(output["mask"].unique().tolist()) == [1, 2]
    assert torch.equal(output["mask"][0, : HEIGHT // 2], torch.full((HEIGHT // 2, WIDTH), 1))
    assert output["detection_scores"][0].tolist() == pytest.approx([0.9, 0.8])
    assert output["category_ids"].tolist() == [[1, 1]]
    assert _semantics(output) == {"1": "person"}


def test_each_following_frame_advances_the_tracker_by_one_step(fake_model) -> None:
    node = SAM31TextPropagation()
    outputs = _run(node, {0: "person"}, 5)

    assert fake_model.steps == [0, 1, 2, 3, 4]
    assert len(fake_model.prompts) == 1
    assert all(out["object_ids"].tolist() == [[1, 2]] for out in outputs)
    fake_model.builds.assert_called_once()


def test_model_is_set_up_for_a_frame_by_frame_stream(fake_model) -> None:
    node = SAM31TextPropagation(
        score_threshold_detection=0.3,
        new_det_thresh=0.6,
        det_nms_thresh=0.2,
        overlap_suppress_thresh=0.8,
        max_num_objects=32,
    )
    _run(node, {0: "person"}, 1)

    assert fake_model.hotstart_delay == 0
    assert fake_model.postprocess_batch_size == 1
    assert fake_model.use_batched_grounding is False
    assert fake_model.score_threshold_detection == 0.3
    assert fake_model.new_det_thresh == 0.6
    assert fake_model.det_nms_thresh == 0.2
    assert fake_model.suppress_overlapping_based_on_recent_occlusion_threshold == 0.8
    assert fake_model.builds.call_args.kwargs["max_num_objects"] == 32


def test_the_same_prompt_again_keeps_the_running_tracks(fake_model) -> None:
    node = SAM31TextPropagation()
    outputs = _run(node, {0: "person", 3: "  person "}, 5)

    assert fake_model.prompts == [(0, "person")]
    assert fake_model.steps == [0, 1, 2, 3, 4]
    assert outputs[-1]["object_ids"].tolist() == [[1, 2]]


def test_a_new_prompt_restarts_tracking_under_a_new_category(fake_model) -> None:
    node = SAM31TextPropagation()
    outputs = _run(node, {0: "person", 3: "bus"}, 5)

    # SAM 3.1 starts over on a text prompt: the new prompt opens a fresh stream at its frame.
    assert fake_model.prompts == [(0, "person"), (0, "bus")]
    assert fake_model.steps == [0, 1, 2, 0, 1]
    assert outputs[2]["object_ids"].tolist() == [[1, 2]]
    assert outputs[3]["object_ids"].tolist() == [[3, 4]]  # ids are never reused in a stream
    assert outputs[3]["category_ids"].tolist() == [[2, 2]]
    assert _semantics(outputs[4]) == {"1": "person", "2": "bus"}
    fake_model.builds.assert_called_once()


def test_blank_prompt_counts_as_no_prompt(fake_model) -> None:
    node = SAM31TextPropagation()
    (output,) = _run(node, {0: "   "}, 1)

    assert output["object_ids"].shape == (1, 0)
    assert fake_model.prompts == []


def test_prompt_must_be_text(fake_model) -> None:
    node = SAM31TextPropagation()
    with pytest.raises(ValueError, match="text_prompt"):
        node.forward(rgb_image=_frame(), text_prompt=7)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_reset_keeps_the_model_and_forgets_the_stream(fake_model) -> None:
    node = SAM31TextPropagation()
    _run(node, {0: "person"}, 3)
    node.reset()

    (idle,) = _run(node, {}, 1)
    assert idle["object_ids"].shape == (1, 0)
    assert _semantics(idle) == {}

    (again,) = _run(node, {0: "bus"}, 1)
    assert again["object_ids"].tolist() == [[1, 2]]
    assert again["category_ids"].tolist() == [[1, 1]]
    assert _semantics(again) == {"1": "bus"}
    fake_model.builds.assert_called_once()


def test_cleanup_drops_the_model(fake_model) -> None:
    node = SAM31TextPropagation()
    _run(node, {0: "person"}, 2)
    node.cleanup()

    assert node._model is None
    _run(node, {0: "person"}, 1)
    assert fake_model.builds.call_count == 2


def test_frame_size_change_in_a_running_stream_is_refused(fake_model) -> None:
    node = SAM31TextPropagation()
    _run(node, {0: "person"}, 2)

    with pytest.raises(ValueError, match="frame size"):
        node.forward(rgb_image=_frame(HEIGHT + 2, WIDTH))


def test_a_tracker_failure_surfaces_once_and_the_next_prompt_starts_fresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _FakeMultiplexModel(fail_at_frame=2)
    monkeypatch.setattr(node_module, "_build_multiplex_model", lambda *_a, **_k: model)
    node = SAM31TextPropagation()
    _run(node, {0: "person"}, 2)

    with pytest.raises(RuntimeError, match="boom"):
        node.forward(rgb_image=_frame())

    (idle,) = _run(node, {}, 1)
    assert idle["object_ids"].shape == (1, 0)

    model.fail_at_frame = None
    (again,) = _run(node, {0: "person"}, 1)  # the same words re-seed a torn-down stream
    assert again["object_ids"].shape == (1, 2)
    assert model.prompts == [(0, "person"), (0, "person")]


# ---------------------------------------------------------------------------
# Bounded state on long streams
# ---------------------------------------------------------------------------


def test_state_stays_bounded_on_a_long_stream(fake_model) -> None:
    node = SAM31TextPropagation()
    n_frames = 60
    _run(node, {0: "person"}, n_frames)

    state = node._inference_state
    window = 16  # max(num_maskmem, max_obj_ptrs_in_encoder) of the model
    last = n_frames - 1
    tracker_state = state["sam2_inference_states"][0]
    per_frame = {
        "non_cond_frame_outputs": tracker_state["output_dict"]["non_cond_frame_outputs"],
        "non_cond_frame_outputs per object": tracker_state["output_dict_per_obj"][0][
            "non_cond_frame_outputs"
        ],
        "frames_already_tracked": tracker_state["frames_already_tracked"],
        "sam2 scores": state["tracker_metadata"]["obj_id_to_sam2_score_frame_wise"],
        "suppressed ids": state["tracker_metadata"]["rank0_metadata"]["suppressed_obj_ids"],
        "cached_frame_outputs": state["cached_frame_outputs"],
        "previous_stages_out": state["previous_stages_out"],
        "find_inputs": dict(state["input_batch"].find_inputs),
    }
    for name, container in per_frame.items():
        assert len(container) <= window, name
        assert last in container, name
        assert min(container) == last - window + 1, name
    assert sorted(state["input_batch"].img_batch._frames) == [last - 1, last]
    # The conditioning frame of the prompt is kept; it is what the tracker attends to.
    assert 0 in tracker_state["output_dict"]["cond_frame_outputs"]


def test_mask_inputs_follow_the_frames_the_tracker_still_holds(fake_model) -> None:
    node = SAM31TextPropagation()
    _run(node, {0: "person"}, 60)

    tracker_state = node._inference_state["sam2_inference_states"][0]
    mask_inputs = tracker_state["mask_inputs_per_obj"][0]
    # Frame 0 is a consolidated conditioning frame; 16 and 32 left the window; 48 is inside it.
    assert sorted(mask_inputs) == [0, 48]
    assert tracker_state["consolidated_frame_inds"]["non_cond_frame_outputs"] == {48}


def test_stages_report_a_length_that_covers_the_latest_frame() -> None:
    stages = _FrameStages()
    for frame_idx in range(40):
        stages[frame_idx] = f"stage {frame_idx}"
    for frame_idx in range(24):
        del stages[frame_idx]

    # The SAM 3.1 detector reads find_inputs[frame_idx % len(find_inputs)].
    assert len(stages) == 40
    assert stages[39 % len(stages)] == "stage 39"
    assert sorted(stages) == list(range(24, 40))
    assert len(_FrameStages()) == 0


def test_frame_buffer_exposes_the_tensors_view_the_model_reads() -> None:
    buffer = _FrameBuffer31(image_size=4, device=torch.device("cpu"))
    buffer.add(np.zeros((3, 3, 3), dtype=np.float32))

    assert buffer.tensors[0] is buffer[0]


def test_detector_lookahead_is_clamped_to_the_buffered_frames(fake_model) -> None:
    node = SAM31TextPropagation()
    _run(node, {0: "person"}, 3)

    lookahead = {"stale": "chunk"}
    result = fake_model.detector.forward_video_grounding_multigpu(
        frame_idx=2, num_frames=10**7, multigpu_buffer=lookahead
    )

    assert result == ("detections", "backbone")
    assert fake_model.detector_calls[-1]["num_frames"] == 3
    assert lookahead == {}


# ---------------------------------------------------------------------------
# Model build
# ---------------------------------------------------------------------------


class _BuiltModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.register_buffer("freqs_cis_real", torch.zeros(1))


def _patch_build(monkeypatch: pytest.MonkeyPatch, checkpoint_keys: set[str]) -> MagicMock:
    import sam3.model_builder as model_builder

    builder = MagicMock(return_value=SimpleNamespace(model=_BuiltModel()))
    monkeypatch.setattr(model_builder, "build_sam3_multiplex_video_predictor", builder)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(node_module, "_checkpoint_keys", lambda _path: checkpoint_keys)
    return builder


def test_build_uses_the_registry_checkpoint_and_streaming_safe_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cuvis_ai_core.data.model_weights import ModelWeights

    builder = _patch_build(monkeypatch, {"weight"})
    resolved: list[str] = []

    def resolve(name: str) -> str:
        resolved.append(name)
        return "registry/sam3.1_multiplex.pt"

    monkeypatch.setattr(ModelWeights, "resolve", staticmethod(resolve))

    model = node_module._build_multiplex_model(None, weight_name="sam3.1", max_num_objects=64)

    assert isinstance(model, _BuiltModel)
    assert resolved == ["sam3.1"]
    kwargs = builder.call_args.kwargs
    assert kwargs["checkpoint_path"] == "registry/sam3.1_multiplex.pt"
    assert kwargs["max_num_objects"] == 64
    assert kwargs["use_fa3"] is False
    assert kwargs["warm_up"] is False


def test_build_takes_an_explicit_checkpoint_without_asking_the_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cuvis_ai_core.data.model_weights import ModelWeights

    builder = _patch_build(monkeypatch, {"weight"})
    monkeypatch.setattr(
        ModelWeights, "resolve", staticmethod(lambda _name: pytest.fail("registry was asked"))
    )

    node_module._build_multiplex_model(
        "local/sam3.1_multiplex.pt", weight_name="sam3.1", max_num_objects=16
    )

    assert builder.call_args.kwargs["checkpoint_path"] == "local/sam3.1_multiplex.pt"


def test_build_resolves_the_weights_the_node_names(monkeypatch: pytest.MonkeyPatch) -> None:
    from cuvis_ai_core.data.model_weights import ModelWeights

    builder = _patch_build(monkeypatch, {"weight"})
    monkeypatch.setattr(ModelWeights, "resolve", staticmethod(lambda name: f"registry/{name}.pt"))

    node_module._build_multiplex_model(None, weight_name="another-row", max_num_objects=16)

    assert builder.call_args.kwargs["checkpoint_path"] == "registry/another-row.pt"


def test_the_node_hands_its_weight_name_to_the_build(fake_model) -> None:
    node = SAM31TextPropagation(weight_name="another-row")

    _run(node, {0: "person"}, 1)

    assert fake_model.builds.call_args.kwargs["weight_name"] == "another-row"
    assert node.hparams["weight_name"] == "another-row"


def test_the_default_weight_name_is_the_row_the_plugin_declares() -> None:
    from cuvis_ai_sam3.weights import WEIGHTS

    node = SAM31TextPropagation()

    (row,) = [entry for entry in WEIGHTS if entry.name == node.hparams["weight_name"]]
    assert row.selected_by == "weight_name"
    assert "checkpoint_path" in row.explicit_path_hparams


def test_an_unknown_weight_name_fails_with_the_registry_error_and_keeps_no_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cuvis_ai_core.data.model_weights import ModelDownloadError

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    node = SAM31TextPropagation(weight_name="no-such-weights")

    with pytest.raises(ModelDownloadError, match="no-such-weights"):
        node.forward(rgb_image=_frame(), text_prompt="person")

    assert node._model is None and node._inference_state is None


def test_build_needs_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA"):
        node_module._build_multiplex_model(
            "local/sam3.1_multiplex.pt", weight_name="sam3.1", max_num_objects=16
        )


def test_build_refuses_a_checkpoint_that_leaves_weights_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_build(monkeypatch, {"something_else"})

    with pytest.raises(RuntimeError, match="weight"):
        node_module._build_multiplex_model(
            "local/sam3.pt", weight_name="sam3.1", max_num_objects=16
        )


def test_missing_checkpoint_keys_ignores_the_computed_rotary_buffers() -> None:
    model_keys = [
        "a.weight",
        "blocks.0.attn.freqs_cis_real",
        "blocks.0.attn.freqs_cis_imag",
        "b.bias",
    ]

    assert _missing_checkpoint_keys(model_keys, {"a.weight", "b.bias"}) == []
    assert _missing_checkpoint_keys(model_keys, {"a.weight"}) == ["b.bias"]


def test_checkpoint_prefix_renames_are_the_ones_the_vendored_loader_applies() -> None:
    """The weight check repeats the loader's renames; this fails when the loader moves on."""
    import inspect

    import sam3.model_builder as model_builder

    source = inspect.getsource(model_builder.build_sam3_multiplex_video_predictor)
    for old, new in node_module._CHECKPOINT_KEY_RENAMES:
        assert f'k.startswith("{old}")' in source
        assert f'"{new}" + k[len("{old}")' in source


def test_a_model_built_for_several_ranks_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A multi-rank detector reads a whole chunk of frames; the stream keeps the last two."""
    model = _FakeMultiplexModel()
    model.world_size = 4  # type: ignore[attr-defined]
    monkeypatch.setattr(node_module, "_build_multiplex_model", lambda *_a, **_k: model)
    node = SAM31TextPropagation()

    with pytest.raises(RuntimeError, match="WORLD_SIZE"):
        node.forward(rgb_image=_frame(), text_prompt="person")
    assert node._model is None and node._inference_state is None


def test_a_failed_build_leaves_the_node_ready_for_the_next_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[int] = []
    model = _FakeMultiplexModel()

    def build(*_args: Any, **_kwargs: Any) -> _FakeMultiplexModel:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("no checkpoint")
        return model

    monkeypatch.setattr(node_module, "_build_multiplex_model", build)
    node = SAM31TextPropagation()

    with pytest.raises(RuntimeError, match="no checkpoint"):
        node.forward(rgb_image=_frame(), text_prompt="person")
    assert node._model is None and node._inference_state is None

    (output,) = _run(node, {0: "person"}, 1)
    assert output["object_ids"].shape == (1, 2)


def test_a_failure_while_opening_the_stream_leaves_no_half_built_stream(
    fake_model, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prompt whose stream cannot be set up must not leave state the next frame trips over."""
    add_frame = SAM31TextPropagation._add_frame_to_state
    calls: list[int] = []

    def fail_on_the_first_frame(self: SAM31TextPropagation, frame_idx: int) -> None:
        calls.append(frame_idx)
        if len(calls) == 1:
            raise RuntimeError("out of memory")
        add_frame(self, frame_idx)

    monkeypatch.setattr(SAM31TextPropagation, "_add_frame_to_state", fail_on_the_first_frame)
    node = SAM31TextPropagation()

    with pytest.raises(RuntimeError, match="out of memory"):
        node.forward(rgb_image=_frame(), text_prompt="person")
    assert node._inference_state is None and node._frame_buffer is None

    unprompted = node.forward(rgb_image=_frame())
    assert unprompted["object_ids"].shape == (1, 0)

    (output,) = _run(node, {0: "person"}, 1)
    assert output["object_ids"].shape == (1, 2)


def test_build_leaves_the_autocast_state_as_it_found_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """The sam3 library enters a bfloat16 autocast in its constructors and never leaves it."""
    builder = _patch_build(monkeypatch, {"weight"})

    def leaking_builder(**_kwargs: Any) -> SimpleNamespace:
        # The library enters the cuda device type; on a cpu-only torch that is a near no-op,
        # so the stand-in leaks the cpu device type to keep the assertion real.
        torch.autocast(device_type="cpu", dtype=torch.bfloat16).__enter__()
        return SimpleNamespace(model=_BuiltModel())

    builder.side_effect = leaking_builder
    enabled, dtype = torch.is_autocast_enabled("cpu"), torch.get_autocast_dtype("cpu")
    try:
        node_module._build_multiplex_model(
            "local/sam3.1_multiplex.pt", weight_name="sam3.1", max_num_objects=16
        )
        assert torch.is_autocast_enabled("cpu") is enabled
        assert torch.get_autocast_dtype("cpu") == dtype
    finally:
        torch.set_autocast_enabled("cpu", enabled)
        torch.set_autocast_dtype("cpu", dtype)


def test_moving_the_pipeline_to_a_gpu_loads_the_model(
    fake_model, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cuvis_ai_sam3 import shared_backbone

    monkeypatch.setattr(
        shared_backbone, "resolve_apply_target_device", lambda _fn: torch.device("cuda")
    )
    node = SAM31TextPropagation()

    node._apply(lambda tensor: tensor)

    assert node._model is fake_model
    fake_model.builds.assert_called_once()


def test_moving_the_pipeline_to_a_gpu_builds_on_that_gpu(
    fake_model, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cuvis_ai_sam3 import shared_backbone

    monkeypatch.setattr(
        shared_backbone, "resolve_apply_target_device", lambda _fn: torch.device("cuda", 1)
    )
    node = SAM31TextPropagation()

    node._apply(lambda tensor: tensor)

    assert fake_model.builds.call_args.kwargs["device"] == torch.device("cuda", 1)


def test_moving_a_running_stream_to_another_device_starts_over(
    fake_model, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stream's tensors live on the device it was opened on; a moved pipeline starts fresh."""
    from cuvis_ai_sam3 import shared_backbone

    target = {"device": torch.device("cuda", 0)}
    monkeypatch.setattr(
        shared_backbone, "resolve_apply_target_device", lambda _fn: target["device"]
    )
    node = SAM31TextPropagation()
    node._apply(lambda tensor: tensor)
    _run(node, {0: "person"}, 2)

    node._apply(lambda tensor: tensor)  # the same device again: nothing changes
    assert node._inference_state is not None
    assert fake_model.builds.call_count == 1

    target["device"] = torch.device("cuda", 1)
    node._apply(lambda tensor: tensor)

    assert node._inference_state is None
    assert fake_model.builds.call_count == 2
    assert fake_model.builds.call_args.kwargs["device"] == torch.device("cuda", 1)
    unprompted = node.forward(rgb_image=_frame())
    assert unprompted["object_ids"].shape == (1, 0)


def test_build_enters_the_requested_cuda_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """The upstream builder calls ``.cuda()`` with no index, so the build selects the device."""
    builder = _patch_build(monkeypatch, {"weight"})
    events: list[str] = []

    class _RecordedDevice:
        def __init__(self, index: int) -> None:
            self.index = index

        def __enter__(self) -> None:
            events.append(f"enter {self.index}")

        def __exit__(self, *_exc: Any) -> None:
            events.append(f"exit {self.index}")

    def build(**_kwargs: Any) -> SimpleNamespace:
        events.append("build")
        return SimpleNamespace(model=_BuiltModel())

    builder.side_effect = build
    monkeypatch.setattr(torch.cuda, "device", _RecordedDevice)

    node_module._build_multiplex_model(
        "local/sam3.1_multiplex.pt",
        weight_name="sam3.1",
        max_num_objects=16,
        device=torch.device("cuda", 1),
    )
    assert events == ["enter 1", "build", "exit 1"]

    events.clear()
    node_module._build_multiplex_model(
        "local/sam3.1_multiplex.pt",
        weight_name="sam3.1",
        max_num_objects=16,
        device=torch.device("cuda"),
    )
    assert events == ["build"]


def test_model_calls_run_with_the_models_gpu_as_the_current_one(
    fake_model, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The vendored tracker allocates state on a bare ``cuda``, which is the current device."""
    entered: list[int] = []

    class _RecordedDevice:
        def __init__(self, index: int) -> None:
            self.index = index

        def __enter__(self) -> None:
            entered.append(self.index)

        def __exit__(self, *_exc: Any) -> None:
            return None

    monkeypatch.setattr(torch.cuda, "device", _RecordedDevice)
    node = SAM31TextPropagation()
    node._ensure_model()

    with node._inference_context():  # the stand-in model lives on the cpu
        pass
    assert entered == []

    monkeypatch.setattr(node, "_model_device", lambda: torch.device("cuda", 1))
    with node._inference_context():
        pass
    assert entered == [1]


def test_a_model_that_cannot_load_on_the_move_waits_for_the_first_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cuvis_ai_sam3 import shared_backbone

    def build(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("no checkpoint")

    monkeypatch.setattr(node_module, "_build_multiplex_model", build)
    monkeypatch.setattr(
        shared_backbone, "resolve_apply_target_device", lambda _fn: torch.device("cuda")
    )
    node = SAM31TextPropagation()

    node._apply(lambda tensor: tensor)  # must not raise: pipeline.to(device) has to succeed

    assert node._model is None


# ---------------------------------------------------------------------------
# Real model
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda and the sam3.1 weights")
def test_real_model_tracks_people_past_its_memory_window() -> None:
    """The vendored SAM 3.1 model still runs under the streaming adapter.

    Takes the registered ``sam3.1`` weights, or the file ``SAM31_CHECKPOINT`` names.
    """
    import cv2

    node = SAM31TextPropagation(checkpoint_path=os.environ.get("SAM31_CHECKPOINT") or None)
    outputs = []
    for frame_idx in range(24):  # the tracker keeps 16 frames; go past them
        bgr = cv2.imread(str(SAMPLE_VIDEO / f"{frame_idx}.jpg"))
        rgb = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).float().div(255.0)
        outputs.append(
            node.forward(
                rgb_image=rgb.unsqueeze(0).cuda(),
                text_prompt="person" if frame_idx == 0 else None,
            )
        )

    first, last = outputs[0], outputs[-1]
    first_ids = set(first["object_ids"][0].tolist())
    last_ids = set(last["object_ids"][0].tolist())
    assert first_ids, "no person found on the prompt frame"
    assert first_ids & last_ids, "no track of the prompt frame is alive 23 frames later"
    assert last["mask"].shape == (1, 720, 1280) and last["mask"].dtype == torch.int32
    assert {out.device.type for out in last.values()} == {"cuda"}
    assert set(last["mask"].unique().tolist()) - {0} <= last_ids
    assert last["category_ids"][0].tolist() == [1] * len(last_ids)
    assert _semantics(last) == {"1": "person"}
    tracker_state = node._inference_state["sam2_inference_states"][0]
    assert len(tracker_state["output_dict"]["non_cond_frame_outputs"]) <= 16
    assert not torch.is_autocast_enabled("cuda")
    node.cleanup()
