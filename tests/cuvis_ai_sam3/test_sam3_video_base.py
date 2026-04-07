from __future__ import annotations

from types import SimpleNamespace

import pytest

from sam3.model.sam3_video_base import Sam3VideoBase


def test_propagate_tracker_one_frame_empty_generator_raises_assertion_not_unboundlocal() -> None:
    model = Sam3VideoBase.__new__(Sam3VideoBase)
    model.tracker = SimpleNamespace(
        propagate_in_video=lambda *args, **kwargs: iter(()),
        low_res_mask_size=8,
    )
    model.fill_hole_area = 0

    inference_states = [{"obj_ids": [1]}]
    with pytest.raises(AssertionError) as exc_info:
        model._propogate_tracker_one_frame_local_gpu(
            inference_states=inference_states,
            frame_idx=1,
            reverse=False,
            run_mem_encoder=False,
        )

    message = str(exc_info.value)
    assert "num_frames_propagated: 0" in message
    assert "out_frame_idx: None" in message
    assert "frame_idx: 1" in message
