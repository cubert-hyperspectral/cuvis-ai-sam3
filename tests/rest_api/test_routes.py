"""T5-T15: HTTP endpoint tests with mocked predictor service."""

from __future__ import annotations


def test_health_endpoint(app_client):
    """T15: GET /health returns 200."""
    resp = app_client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_start_session(app_client):
    """T5: POST /sessions returns session_id + video metadata."""
    resp = app_client.post(
        "/api/v1/sessions",
        json={"video_path": "/path/to/video.mp4"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "test-session-123"
    assert data["num_frames"] == 100
    assert data["frame_rate"] == 30.0
    assert data["width"] == 1280
    assert data["height"] == 720


def test_get_session(app_client):
    """GET /sessions/{id} returns session info."""
    from rest_api.session_manager import SessionInfo

    service = app_client.app.state.predictor_service
    service.session_manager.get.return_value = SessionInfo(
        session_id="test-session-123",
        video_path="/path/to/video.mp4",
        num_frames=100,
        frame_rate=30.0,
        width=1280,
        height=720,
    )
    resp = app_client.get("/api/v1/sessions/test-session-123")
    assert resp.status_code == 200
    assert resp.json()["session_id"] == "test-session-123"


def test_close_session(app_client):
    """DELETE /sessions/{id} returns success."""
    resp = app_client.delete("/api/v1/sessions/test-session-123")
    assert resp.status_code == 200
    assert resp.json()["success"] is True


def test_reset_session(app_client):
    """POST /sessions/{id}/reset returns success."""
    resp = app_client.post("/api/v1/sessions/test-session-123/reset")
    assert resp.status_code == 200
    assert resp.json()["success"] is True


def test_prompt_text_pcs(app_client):
    """T6: Text-only prompt routes to PCS."""
    resp = app_client.post(
        "/api/v1/sessions/test-session-123/prompts",
        json={"frame_index": 0, "text": "person"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["frame_index"] == 0
    assert len(data["objects"]) == 2


def test_prompt_point_pvs(app_client):
    """T7: Points + obj_id prompt routes to PVS."""
    resp = app_client.post(
        "/api/v1/sessions/test-session-123/prompts",
        json={
            "frame_index": 5,
            "points": [[0.5, 0.3]],
            "point_labels": [1],
            "obj_id": 0,
        },
    )
    assert resp.status_code == 200


def test_prompt_bbox_pvs(app_client):
    """T8: Box prompt routes to PVS."""
    resp = app_client.post(
        "/api/v1/sessions/test-session-123/prompts",
        json={
            "frame_index": 0,
            "bounding_boxes": [[0.1, 0.2, 0.3, 0.5]],
            "bounding_box_labels": [1],
        },
    )
    assert resp.status_code == 200


def test_prompt_no_prompt_type(app_client):
    """Prompt with no text/points/boxes returns 422."""
    resp = app_client.post(
        "/api/v1/sessions/test-session-123/prompts",
        json={"frame_index": 0},
    )
    assert resp.status_code == 422


def test_propagate_forward(app_client):
    """T9: direction=forward propagation."""
    resp = app_client.post(
        "/api/v1/sessions/test-session-123/propagate",
        json={"direction": "forward", "max_frames": 50},
    )
    assert resp.status_code == 200
    # SSE responses come as text/event-stream
    assert "text/event-stream" in resp.headers["content-type"]


def test_remove_object(app_client):
    """T13: DELETE /sessions/{id}/objects/{obj_id} returns success."""
    resp = app_client.delete("/api/v1/sessions/test-session-123/objects/0")
    assert resp.status_code == 200
    assert resp.json()["success"] is True


def test_session_not_found(app_client):
    """T14: Missing session returns 404."""
    service = app_client.app.state.predictor_service
    service.session_manager.get.side_effect = KeyError("Session 'nonexistent' not found")
    resp = app_client.get("/api/v1/sessions/nonexistent")
    assert resp.status_code == 404
