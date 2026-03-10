"""Session lifecycle endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from rest_api.models import SessionInfoResponse, StartSessionRequest, StartSessionResponse

router = APIRouter(prefix="/sessions", tags=["sessions"])


def _get_service(request: Request):  # noqa: ANN202
    return request.app.state.predictor_service


@router.post("", response_model=StartSessionResponse)
async def start_session(body: StartSessionRequest, request: Request) -> StartSessionResponse:
    service = _get_service(request)
    if service.session_manager.active_count >= service._config.max_sessions:
        raise HTTPException(status_code=429, detail="Maximum number of sessions reached")
    return await service.run_sync(service.start_session, body.video_path, body.session_id)


@router.get("/{session_id}", response_model=SessionInfoResponse)
async def get_session(session_id: str, request: Request) -> SessionInfoResponse:
    service = _get_service(request)
    try:
        info = service.session_manager.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found") from None
    return SessionInfoResponse(
        session_id=info.session_id,
        video_path=info.video_path,
        num_frames=info.num_frames,
        frame_rate=info.frame_rate,
        width=info.width,
        height=info.height,
    )


@router.delete("/{session_id}")
async def close_session(session_id: str, request: Request) -> dict:
    service = _get_service(request)
    await service.run_sync(service.close_session, session_id)
    return {"success": True}


@router.post("/{session_id}/reset")
async def reset_session(session_id: str, request: Request) -> dict:
    service = _get_service(request)
    await service.run_sync(service.reset_session, session_id)
    return {"success": True}
