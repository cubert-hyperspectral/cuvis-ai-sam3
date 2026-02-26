from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai_sam3.node.sam3_video_tracker import _FrameBuffer


def test_frame_buffer_add_and_getitem() -> None:
    buffer = _FrameBuffer(
        image_size=8,
        device=torch.device("cpu"),
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
    )

    frame = np.random.randint(0, 256, size=(6, 5, 3), dtype=np.uint8)
    frame_idx = buffer.add(frame)

    assert frame_idx == 0
    assert len(buffer) == 1

    out = buffer[0]
    assert out.shape == (3, 8, 8)
    assert out.dtype == torch.float16


def test_frame_buffer_tensor_index_and_prune() -> None:
    buffer = _FrameBuffer(
        image_size=4,
        device=torch.device("cpu"),
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
    )

    frame = np.zeros((3, 3, 3), dtype=np.uint8)
    buffer.add(frame)
    buffer.add(frame)

    stacked = buffer[torch.tensor([0, 1], dtype=torch.int64)]
    assert stacked.shape == (2, 3, 4, 4)

    buffer.prune_before(1)
    with pytest.raises(IndexError):
        _ = buffer[0]
    _ = buffer[1]
