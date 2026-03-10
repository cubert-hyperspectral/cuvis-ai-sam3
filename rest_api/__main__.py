"""CLI entry point: ``python -m rest_api`` or ``sam3-rest-api``."""

from __future__ import annotations

import uvicorn

from rest_api.config import ServerConfig


def main() -> None:
    config = ServerConfig()
    uvicorn.run(
        "rest_api.app:create_app",
        host=config.host,
        port=config.port,
        factory=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
