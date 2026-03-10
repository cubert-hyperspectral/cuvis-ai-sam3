"""Shared fixtures for REST API tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rest_api.config import ServerConfig
from rest_api.middleware import add_middleware
from rest_api.models import AddPromptResponse, ObjectResult, StartSessionResponse
from rest_api.routes import api_router


def _make_mock_outputs(n_objects: int = 2) -> dict:
    """Create a SAM3-style outputs dict with *n_objects*."""
    return {
        "out_obj_ids": np.arange(n_objects, dtype=np.int64),
        "out_probs": np.full(n_objects, 0.95, dtype=np.float32),
        "out_boxes_xywh": np.tile([0.1, 0.2, 0.3, 0.4], (n_objects, 1)).astype(np.float32),
        "out_binary_masks": np.ones((n_objects, 64, 64), dtype=bool),
    }


@pytest.fixture()
def mock_predictor():
    """Return a mocked ``Sam3VideoPredictor``-like object."""
    predictor = MagicMock()
    predictor.start_session.return_value = {"session_id": "test-session-123"}
    predictor._ALL_INFERENCE_STATES = {
        "test-session-123": {
            "state": {
                "num_frames": 100,
                "orig_height": 720,
                "orig_width": 1280,
                "_video_iterator": MagicMock(frame_rate=30.0),
            },
            "session_id": "test-session-123",
        }
    }
    predictor.add_prompt.return_value = {"frame_index": 0, "outputs": _make_mock_outputs(2)}

    def _mock_propagate(**kwargs):
        for i in range(3):
            yield {"frame_index": i, "outputs": _make_mock_outputs(1)}

    predictor.propagate_in_video.side_effect = _mock_propagate
    predictor.remove_object.return_value = {"is_success": True}
    predictor.reset_session.return_value = {"is_success": True}
    predictor.close_session.return_value = {"is_success": True}
    return predictor


async def _async_passthrough(fn, *args, **kwargs):
    """Run a sync function directly (no executor) for test simplicity."""
    return fn(*args, **kwargs)


@pytest.fixture()
def app_client(mock_predictor):
    """Create a FastAPI test client with a mocked predictor service (no lifespan)."""
    config = ServerConfig(device="cpu")

    # Build app without lifespan to avoid importing SAM3 model.
    app = FastAPI(title="SAM3 REST API (test)")
    app.state.config = config
    add_middleware(app)
    app.include_router(api_router)

    # Build mock service.
    mock_service = MagicMock()
    mock_service._predictor = mock_predictor
    mock_service._config = config
    mock_service.session_manager = MagicMock()
    mock_service.session_manager.active_count = 0

    mock_service.start_session.return_value = StartSessionResponse(
        session_id="test-session-123",
        num_frames=100,
        frame_rate=30.0,
        width=1280,
        height=720,
    )
    mock_service.add_prompt.return_value = AddPromptResponse(
        frame_index=0,
        objects=[
            ObjectResult(obj_id=0, bbox_xywh=[0.1, 0.2, 0.3, 0.4], score=0.95),
            ObjectResult(obj_id=1, bbox_xywh=[0.5, 0.5, 0.2, 0.3], score=0.90),
        ],
    )
    mock_service.propagate_in_video.return_value = [
        {
            "frame_index": i,
            "objects": [
                {"obj_id": 0, "bbox_xywh": [0.1, 0.2, 0.3, 0.4], "score": 0.95, "mask_rle": None}
            ],
        }
        for i in range(3)
    ]
    mock_service.run_sync = _async_passthrough

    app.state.predictor_service = mock_service

    with TestClient(app) as client:
        yield client
