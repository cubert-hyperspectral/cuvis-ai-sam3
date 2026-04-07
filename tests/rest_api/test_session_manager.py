"""T2-T3: SessionManager lifecycle and expiry."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from rest_api.session_manager import SessionManager


def _make_inference_state(num_frames: int = 100) -> dict:
    video = MagicMock()
    video.frame_rate = 30.0
    return {
        "num_frames": num_frames,
        "orig_height": 720,
        "orig_width": 1280,
        "_video_iterator": video,
    }


def test_session_create_get_close():
    """T2: Basic create/get/close lifecycle."""
    mgr = SessionManager()
    state = _make_inference_state()
    info = mgr.register("s1", "/video.mp4", state)

    assert info.session_id == "s1"
    assert info.num_frames == 100
    assert info.frame_rate == 30.0
    assert info.width == 1280
    assert info.height == 720
    assert mgr.active_count == 1

    retrieved = mgr.get("s1")
    assert retrieved.session_id == "s1"

    mgr.remove("s1")
    assert mgr.active_count == 0

    with pytest.raises(KeyError):
        mgr.get("s1")


def test_session_expiry():
    """T3: Stale sessions get cleaned up."""
    mgr = SessionManager()
    state = _make_inference_state()
    info = mgr.register("s-old", "/video.mp4", state)
    # Backdate the creation time
    info.created_at = time.time() - 7200

    predictor = MagicMock()
    expired = mgr.cleanup_expired(timeout_seconds=3600, predictor=predictor)

    assert expired == ["s-old"]
    assert mgr.active_count == 0
    predictor.close_session.assert_called_once_with("s-old")


def test_remove_idempotent():
    """Removing a non-existent session does not raise."""
    mgr = SessionManager()
    mgr.remove("nonexistent")  # should not raise
