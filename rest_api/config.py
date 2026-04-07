"""Server configuration via environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class ServerConfig(BaseSettings):
    """SAM3 REST API server settings.

    All fields can be set via environment variables prefixed with ``SAM3_``,
    e.g. ``SAM3_PORT=9000``.
    """

    model_config = {"env_prefix": "SAM3_"}

    checkpoint_path: str | None = None
    bpe_path: str | None = None
    device: str = "cuda"
    host: str = "0.0.0.0"
    port: int = 8100
    session_timeout_seconds: int = 3600
    max_sessions: int = 10
    compile_model: bool = False
