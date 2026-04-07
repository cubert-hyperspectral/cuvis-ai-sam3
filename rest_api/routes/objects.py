"""Object management endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(prefix="/sessions", tags=["objects"])


@router.delete("/{session_id}/objects/{obj_id}")
async def remove_object(session_id: str, obj_id: int, request: Request) -> dict:
    service = request.app.state.predictor_service
    await service.run_sync(service.remove_object, session_id, obj_id)
    return {"success": True}
