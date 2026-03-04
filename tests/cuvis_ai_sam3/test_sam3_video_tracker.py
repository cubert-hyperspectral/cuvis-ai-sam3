from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cuvis_ai_sam3.node import sam3_video_tracker as tracker_mod


class _DummySAM3Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.detector = SimpleNamespace(world_size=1)
        self.rank = 0
        self.image_size = 8
        self.image_mean = (0.5, 0.5, 0.5)
        self.image_std = (0.5, 0.5, 0.5)
        self._compiled = False
        self._param = torch.nn.Parameter(torch.zeros(1))

    def _compile_model(self) -> None:
        self._compiled = True

    def add_prompt(
        self,
        inference_state: dict,
        frame_idx: int,
        text_str: str | None = None,
        **_: object,
    ) -> tuple[int, dict]:
        inference_state["text_prompt"] = text_str
        raw = self._run_single_frame_inference(inference_state, frame_idx, reverse=False)
        out = self._postprocess_output(
            inference_state,
            raw,
            removed_obj_ids=raw.get("removed_obj_ids"),
            suppressed_obj_ids=raw.get("suppressed_obj_ids"),
            unconfirmed_obj_ids=[],
        )
        return frame_idx, out

    def _run_single_frame_inference(
        self,
        inference_state: dict,
        frame_idx: int,
        reverse: bool = False,  # noqa: ARG002
    ) -> dict:
        h = int(inference_state["orig_height"])
        w = int(inference_state["orig_width"])

        tracker_md = inference_state.setdefault("tracker_metadata", {})
        score_map = tracker_md.setdefault("obj_id_to_tracker_score_frame_wise", {})
        rank0 = tracker_md.setdefault("rank0_metadata", {})
        suppressed = rank0.setdefault("suppressed_obj_ids", {})
        suppressed[frame_idx] = []

        if frame_idx in (0, 1):
            mask = torch.zeros((1, h, w), dtype=torch.bool, device=self._param.device)
            mask[:, : max(1, h // 2), : max(1, w // 2)] = True
            obj_id_to_mask = {1: mask}
            obj_id_to_score = {1: 0.95}
            score_map[frame_idx] = {1: 0.9}
        else:
            obj_id_to_mask = {}
            obj_id_to_score = {}
            score_map[frame_idx] = {}

        inference_state["previous_stages_out"][frame_idx] = "_THIS_FRAME_HAS_OUTPUTS_"
        return {
            "obj_id_to_mask": obj_id_to_mask,
            "obj_id_to_score": obj_id_to_score,
            "obj_id_to_tracker_score": score_map[frame_idx],
            "removed_obj_ids": [],
            "suppressed_obj_ids": [],
            "frame_stats": None,
        }

    def _postprocess_output(
        self,
        inference_state: dict,
        out: dict,
        removed_obj_ids: list[int] | None = None,  # noqa: ARG002
        suppressed_obj_ids: list[int] | None = None,  # noqa: ARG002
        unconfirmed_obj_ids: list[int] | None = None,  # noqa: ARG002
    ) -> dict:
        h = int(inference_state["orig_height"])
        w = int(inference_state["orig_width"])
        obj_ids = sorted(out["obj_id_to_mask"].keys())
        if not obj_ids:
            return {
                "out_obj_ids": np.zeros((0,), dtype=np.int64),
                "out_probs": np.zeros((0,), dtype=np.float32),
                "out_boxes_xywh": np.zeros((0, 4), dtype=np.float32),
                "out_binary_masks": np.zeros((0, h, w), dtype=bool),
                "frame_stats": out.get("frame_stats"),
            }

        masks = torch.cat([out["obj_id_to_mask"][obj_id] for obj_id in obj_ids], dim=0)
        probs = np.asarray([out["obj_id_to_score"][obj_id] for obj_id in obj_ids], dtype=np.float32)
        return {
            "out_obj_ids": np.asarray(obj_ids, dtype=np.int64),
            "out_probs": probs,
            "out_boxes_xywh": np.zeros((len(obj_ids), 4), dtype=np.float32),
            "out_binary_masks": masks.detach().cpu().numpy().astype(bool, copy=False),
            "frame_stats": out.get("frame_stats"),
        }

    def _tracker_remove_objects(
        self, tracker_states: list[dict[str, object]], obj_ids_to_remove: list[int]
    ) -> None:
        removed = {int(v) for v in obj_ids_to_remove}
        keep_states: list[dict[str, object]] = []
        for ts in tracker_states:
            obj_ids = [int(v) for v in ts.get("obj_ids", []) if int(v) not in removed]
            ts["obj_ids"] = obj_ids
            if obj_ids:
                keep_states.append(ts)
        tracker_states[:] = keep_states


@pytest.fixture
def tracker_node(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[tracker_mod.SAM3TrackerInference, _DummySAM3Model]:
    dummy_model = _DummySAM3Model()
    monkeypatch.setattr(
        tracker_mod,
        "build_sam3_video_model",
        lambda **_: dummy_model,
    )
    node = tracker_mod.SAM3TrackerInference(
        masklet_confirmation_consecutive_det_thresh=2,
        confirmation_warmup_frames=0,
        confirmation_warmup_thresh=2,
        confirmation_high_confidence_thresh=1.1,
    )
    return node, dummy_model


def test_tracker_forward_emits_tentative_and_confirmed_streams(
    tracker_node: tuple[tracker_mod.SAM3TrackerInference, _DummySAM3Model],
) -> None:
    node, _ = tracker_node
    frame = torch.rand((1, 12, 10, 3), dtype=torch.float32)

    out0 = node.forward(rgb_frame=frame)
    assert out0["frame_id"].tolist() == [0]
    assert out0["object_ids"].tolist() == [[1]]
    assert out0["confirmed_object_ids"].shape == (1, 0)

    out1 = node.forward(rgb_frame=frame)
    assert out1["frame_id"].tolist() == [1]
    assert out1["confirmed_object_ids"].tolist() == [[1]]

    out2 = node.forward(rgb_frame=frame)
    assert out2["frame_id"].tolist() == [2]
    assert out2["object_ids"].shape == (1, 0)
    assert out2["confirmed_object_ids"].shape == (1, 0)


def test_tracker_reset_clears_state(
    tracker_node: tuple[tracker_mod.SAM3TrackerInference, _DummySAM3Model],
) -> None:
    node, _ = tracker_node
    frame = torch.rand((1, 8, 8, 3), dtype=torch.float32)
    _ = node.forward(rgb_frame=frame)

    node.reset()

    assert node._inference_state is None
    assert node._frame_buffer is None
    assert node._frame_id == 0
    assert node._det_count == {}


def test_compile_flag_triggers_model_compile(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_model = _DummySAM3Model()
    monkeypatch.setattr(
        tracker_mod,
        "build_sam3_video_model",
        lambda **_: dummy_model,
    )
    node = tracker_mod.SAM3TrackerInference(compile_model=True)
    frame = torch.rand((1, 8, 8, 3), dtype=torch.float32)
    _ = node.forward(rgb_frame=frame)

    assert dummy_model._compiled is True


def test_extend_state_for_frame_syncs_tracker_inference_num_frames(
    tracker_node: tuple[tracker_mod.SAM3TrackerInference, _DummySAM3Model],
) -> None:
    node, _ = tracker_node
    frame = torch.rand((1, 12, 10, 3), dtype=torch.float32)

    _ = node.forward(rgb_frame=frame)
    assert node._inference_state is not None

    node._inference_state["tracker_inference_states"] = [{"obj_ids": [1], "num_frames": 1}]

    _ = node.forward(rgb_frame=frame)

    assert node._inference_state["num_frames"] == 2
    tracker_states = node._inference_state["tracker_inference_states"]
    assert len(tracker_states) == 1
    assert tracker_states[0]["num_frames"] == 2


def test_default_max_tracker_states_is_five(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_model = _DummySAM3Model()
    monkeypatch.setattr(
        tracker_mod,
        "build_sam3_video_model",
        lambda **_: dummy_model,
    )
    node = tracker_mod.SAM3TrackerInference()
    assert node.max_tracker_states == 5


def test_should_log_progress_every_fifty_frames() -> None:
    assert tracker_mod.SAM3TrackerInference._should_log_progress(frame_idx=0, interval=50) is True
    assert tracker_mod.SAM3TrackerInference._should_log_progress(frame_idx=48, interval=50) is False
    assert tracker_mod.SAM3TrackerInference._should_log_progress(frame_idx=49, interval=50) is True
    assert tracker_mod.SAM3TrackerInference._should_log_progress(frame_idx=99, interval=50) is True
    assert tracker_mod.SAM3TrackerInference._should_log_progress(frame_idx=100, interval=0) is False


def test_evict_excess_tracker_states_enforces_cap_and_metadata(
    tracker_node: tuple[tracker_mod.SAM3TrackerInference, _DummySAM3Model],
) -> None:
    node, _ = tracker_node
    node.max_tracker_states = 2
    node._inference_state = {
        "tracker_inference_states": [
            {"obj_ids": [1, 2]},  # primary state (kept)
            {"obj_ids": [3]},
            {"obj_ids": [4]},
        ],
        "tracker_metadata": {
            "obj_ids_per_gpu": [np.asarray([1, 2, 3, 4], dtype=np.int64)],
            "obj_ids_all_gpu": np.asarray([1, 2, 3, 4], dtype=np.int64),
            "num_obj_per_gpu": np.asarray([4], dtype=np.int64),
            "obj_id_to_score": {1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6},
            "rank0_metadata": {
                "obj_first_frame_idx": {1: 0, 2: 0, 3: 1, 4: 2},
                "trk_keep_alive": {1: 4, 2: 4, 3: 2, 4: 2},
                "unmatched_frame_inds": {3: [10], 4: [11]},
                "removed_obj_ids": set(),
                "overlap_pair_to_frame_inds": {(1, 3): [10], (2, 4): [11]},
                "masklet_confirmation": {
                    "status": np.asarray([1, 1, 1, 1], dtype=np.int32),
                    "consecutive_det_num": np.asarray([3, 3, 1, 1], dtype=np.int32),
                },
            },
        },
    }

    node._evict_excess_tracker_states(frame_idx=25)

    states = node._inference_state["tracker_inference_states"]
    assert len(states) <= 2
    assert [int(v) for v in node._inference_state["tracker_metadata"]["obj_ids_all_gpu"]] == [
        1,
        2,
        4,
    ]
    assert 3 not in node._inference_state["tracker_metadata"]["obj_id_to_score"]
