"""CLI entry point: ``python -m rest_api`` or ``sam3-rest-api``."""

from __future__ import annotations

import sys

import uvicorn

from rest_api.config import ServerConfig


def main() -> None:
    config = ServerConfig()
    host = config.host
    port = config.port

    if len(sys.argv) == 2:
        address = sys.argv[1]
        h, _, p = address.rpartition(":")
        if h and p:
            host = h
            port = int(p)

    uvicorn.run(
        "rest_api.app:create_app",
        host=host,
        port=port,
        factory=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
