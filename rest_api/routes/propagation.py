"""Propagation endpoint with SSE streaming."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from rest_api.models import PropagateRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions", tags=["propagation"])


@router.post("/{session_id}/propagate")
async def propagate(
    session_id: str,
    body: PropagateRequest,
    request: Request,
    include_masks: bool = False,
) -> EventSourceResponse:
    service = request.app.state.predictor_service

    async def _event_generator() -> AsyncGenerator[dict, None]:
        loop = asyncio.get_running_loop()

        # Run the sync generator in a thread and iterate over results.
        gen = await loop.run_in_executor(
            None,
            lambda: list(
                service.propagate_in_video(
                    session_id,
                    body,
                    include_masks=include_masks,
                )
            ),
        )

        for frame_result in gen:
            yield {"event": "frame", "data": json.dumps(frame_result)}

        yield {"event": "done", "data": "{}"}

    return EventSourceResponse(_event_generator())
