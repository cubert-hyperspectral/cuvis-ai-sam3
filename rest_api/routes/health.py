"""Health check endpoint."""

from __future__ import annotations

import torch
from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "allocated_mib": torch.cuda.memory_allocated() // (1024 * 1024),
            "reserved_mib": torch.cuda.memory_reserved() // (1024 * 1024),
            "max_allocated_mib": torch.cuda.max_memory_allocated() // (1024 * 1024),
        }
    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "gpu_memory": gpu_info,
    }
