"""Prompt endpoint — add PCS (text) or PVS (point/box) prompts."""

from __future__ import annotations

from fastapi import APIRouter, Request

from rest_api.models import AddPromptRequest, AddPromptResponse

router = APIRouter(prefix="/sessions", tags=["prompts"])


@router.post("/{session_id}/prompts", response_model=AddPromptResponse)
async def add_prompt(
    session_id: str,
    body: AddPromptRequest,
    request: Request,
    include_masks: bool = False,
) -> AddPromptResponse:
    service = request.app.state.predictor_service
    return await service.run_sync(
        service.add_prompt,
        session_id,
        body,
        include_masks=include_masks,
    )
