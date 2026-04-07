"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from rest_api.config import ServerConfig
from rest_api.middleware import add_middleware
from rest_api.routes import api_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown lifecycle."""
    config: ServerConfig = app.state.config

    # --- startup ---
    from rest_api.predictor_service import PredictorService

    service = PredictorService(config)
    app.state.predictor_service = service

    # Background session cleanup task.
    async def _cleanup_loop() -> None:
        while True:
            await asyncio.sleep(60)
            expired = service.session_manager.cleanup_expired(
                config.session_timeout_seconds,
                service._predictor,
            )
            if expired:
                logger.info("Expired sessions: %s", expired)

    cleanup_task = asyncio.create_task(_cleanup_loop())

    yield

    # --- shutdown ---
    cleanup_task.cancel()
    service.shutdown()
    logger.info("SAM3 REST API shut down")


def create_app(config: ServerConfig | None = None) -> FastAPI:
    """Build and return the FastAPI application."""
    if config is None:
        config = ServerConfig()

    app = FastAPI(
        title="SAM3 REST API",
        version="0.1.0",
        lifespan=_lifespan,
    )
    app.state.config = config

    add_middleware(app)
    app.include_router(api_router)

    return app
