"""Error handling and CORS middleware."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def add_middleware(app: FastAPI) -> None:
    """Attach CORS and error-handling middleware to *app*."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(RuntimeError)
    async def _runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
        msg = str(exc)
        if "cannot find session" in msg.lower() or "does not exist" in msg.lower():
            return JSONResponse(status_code=404, content={"detail": msg})
        logger.exception("Unhandled RuntimeError")
        return JSONResponse(status_code=400, content={"detail": msg})

    @app.exception_handler(KeyError)
    async def _key_error_handler(request: Request, exc: KeyError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})
