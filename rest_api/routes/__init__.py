"""Assemble all API routers under ``/api/v1``."""

from fastapi import APIRouter

from rest_api.routes.health import router as health_router
from rest_api.routes.objects import router as objects_router
from rest_api.routes.prompts import router as prompts_router
from rest_api.routes.propagation import router as propagation_router
from rest_api.routes.sessions import router as sessions_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(sessions_router)
api_router.include_router(prompts_router)
api_router.include_router(propagation_router)
api_router.include_router(objects_router)
api_router.include_router(health_router)
