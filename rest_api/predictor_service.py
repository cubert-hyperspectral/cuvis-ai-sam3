"""SAM3 model loading and inference dispatch."""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Generator
from contextlib import nullcontext
from typing import Any

from rest_api.config import ServerConfig
from rest_api.models import (
    AddPromptRequest,
    AddPromptResponse,
    PropagateRequest,
    StartSessionResponse,
    outputs_to_object_results,
)
from rest_api.session_manager import SessionManager
from rest_api.video_adapter import Sam3VideoInferenceWithVideoIterator
from sam3.model.sam3_video_predictor import Sam3VideoPredictor

logger = logging.getLogger(__name__)


class Sam3VideoPredictorLazy(Sam3VideoPredictor):
    """Variant of ``Sam3VideoPredictor`` that accepts a pre-built model."""

    def __init__(self, model: Any, **kwargs: Any) -> None:  # noqa: ANN401
        # Skip the parent's __init__ which builds the model internally.
        self.model = model
        self.async_loading_frames = kwargs.get("async_loading_frames", False)
        self.video_loader_type = kwargs.get("video_loader_type", "cv2")


class PredictorService:
    """High-level service wrapping SAM3 predictor with metadata management."""

    def __init__(self, config: ServerConfig) -> None:
        from sam3.model_builder import build_sam3_video_model

        logger.info("Loading SAM3 model (checkpoint=%s) ...", config.checkpoint_path)
        model = build_sam3_video_model(
            checkpoint_path=config.checkpoint_path,
            bpe_path=config.bpe_path,
            compile=config.compile_model,
        )
        # Swap in the lazy-loading subclass.
        model.__class__ = Sam3VideoInferenceWithVideoIterator
        model = model.to(config.device).eval()

        self._predictor = Sam3VideoPredictorLazy(model)
        self.session_manager = SessionManager()
        self._config = config
        # Autocast must be entered in the thread where inference runs.
        self._use_cuda_autocast = str(config.device).lower().startswith("cuda")
        logger.info("SAM3 model loaded on %s", config.device)

    def _inference_autocast(self):  # noqa: ANN201
        """Return per-call autocast context for inference.

        Torch autocast is thread-local, so entering it at startup does not affect
        work executed in `run_in_executor` worker threads.
        """
        if not self._use_cuda_autocast:
            return nullcontext()

        import torch

        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    # -- session lifecycle --------------------------------------------------

    def start_session(
        self,
        video_path: str,
        session_id: str | None = None,
    ) -> StartSessionResponse:
        with self._inference_autocast():
            result = self._predictor.start_session(
                resource_path=video_path,
                session_id=session_id,
            )
        sid = result["session_id"]
        state = self._predictor._ALL_INFERENCE_STATES[sid]["state"]
        info = self.session_manager.register(sid, video_path, state)
        return StartSessionResponse(
            session_id=sid,
            num_frames=info.num_frames,
            frame_rate=info.frame_rate,
            width=info.width,
            height=info.height,
        )

    def close_session(self, session_id: str) -> None:
        self._predictor.close_session(session_id)
        self.session_manager.remove(session_id)

    def reset_session(self, session_id: str) -> None:
        self._predictor.reset_session(session_id)

    # -- prompts ------------------------------------------------------------

    def add_prompt(
        self,
        session_id: str,
        request: AddPromptRequest,
        include_masks: bool = False,
    ) -> AddPromptResponse:
        with self._inference_autocast():
            result = self._predictor.add_prompt(
                session_id=session_id,
                frame_idx=request.frame_index,
                text=request.text,
                points=request.points,
                point_labels=request.point_labels,
                bounding_boxes=request.bounding_boxes,
                bounding_box_labels=request.bounding_box_labels,
                obj_id=request.obj_id,
            )
        objects = outputs_to_object_results(result["outputs"], include_masks=include_masks)
        return AddPromptResponse(frame_index=result["frame_index"], objects=objects)

    # -- propagation --------------------------------------------------------

    def propagate_in_video(
        self,
        session_id: str,
        request: PropagateRequest,
        include_masks: bool = False,
    ) -> Generator[dict, None, None]:
        """Yield per-frame results as dicts (JSON-serializable)."""
        with self._inference_autocast():
            for frame_result in self._predictor.propagate_in_video(
                session_id=session_id,
                propagation_direction=request.direction,
                start_frame_idx=request.start_frame_index,
                max_frame_num_to_track=request.max_frames,
            ):
                objects = outputs_to_object_results(
                    frame_result["outputs"],
                    include_masks=include_masks,
                )
                yield {
                    "frame_index": frame_result["frame_index"],
                    "objects": [obj.model_dump() for obj in objects],
                }

    # -- object management --------------------------------------------------

    def remove_object(self, session_id: str, obj_id: int) -> None:
        self._predictor.remove_object(session_id=session_id, obj_id=obj_id)

    # -- lifecycle ----------------------------------------------------------

    def shutdown(self) -> None:
        """Release all sessions and model resources."""
        if hasattr(self._predictor, "shutdown"):
            self._predictor.shutdown()

    # -- async helpers ------------------------------------------------------

    async def run_sync(self, fn, *args, **kwargs):  # noqa: ANN201
        """Run a sync function in the default executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))
