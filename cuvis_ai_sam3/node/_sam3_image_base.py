"""Shared base for single-frame SAM3 image-predictor nodes.

Both ``SAM3SegmentEverything`` and ``SAM3PointExpansion`` wrap the SAM3
interactive image predictor: they build the model with
``build_sam3_image_model`` plus ``Sam3Processor``, run the heavy ViT embedding
through ``set_image`` and the light mask decoder through ``predict_inst``, and
share the same RGB-frame normalization, autocast eval context, and empty-output
shape. This module factors those out so the concrete nodes only carry their own
prompting logic and port specs.

The module name is underscore-prefixed so ``NodeRegistry.auto_register_package``
skips it: ``_Sam3ImageNode`` is an abstract base (it defines no ``forward``) and
must never be registered or instantiated on its own.
"""

from __future__ import annotations

import contextlib
from typing import Any, ClassVar

import numpy as np
import torch
from cuvis_ai_core.node import Node
from loguru import logger

from cuvis_ai_sam3 import shared_backbone as _shared


class _Sam3ImageNode(Node):
    """Abstract base owning SAM3 image-model construction and frame helpers.

    Subclasses define their own ``INPUT_SPECS`` / ``OUTPUT_SPECS``, hyper-
    parameters, and ``forward``. They must call ``super().__init__`` with at
    least ``checkpoint_path`` / ``device`` / ``compile_model`` (plus their own
    hparams as keyword arguments) so the base resolves the device and stores the
    lazily-built model handles.
    """

    _AUTOCAST_DTYPE: ClassVar[dict[str, torch.dtype]] = {
        "cuda": torch.bfloat16,
        "cpu": torch.bfloat16,
    }

    def __init__(
        self,
        checkpoint_path: str | None = None,
        device: str | None = None,
        compile_model: bool = False,
        **kwargs: Any,
    ) -> None:
        """Resolve the device and initialize the (lazily built) model handles.

        Args:
            checkpoint_path: Optional SAM3 image-model checkpoint path.
            device: Torch device override; defaults to cuda when available, else cpu.
            compile_model: Whether to ``torch.compile`` the image model.
            kwargs: Remaining node hyper-parameters, forwarded to ``Node``.
        """
        self._checkpoint_path = checkpoint_path
        self._requested_device = device
        self._resolved_device = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._compile_model = bool(compile_model)
        self._model: Any | None = None
        self._processor: Any | None = None
        self._backbone_claim: _shared.BackboneClaim | None = None
        self._claim_shared_backbone()

        super().__init__(
            checkpoint_path=checkpoint_path,
            device=device,
            compile_model=compile_model,
            **kwargs,
        )

    def _claim_shared_backbone(self) -> None:
        """(Re-)claim the shared-backbone entry unless sharing is bypassed."""
        if self._backbone_claim is not None and not self._backbone_claim.released:
            return
        self._backbone_claim = _shared.claim_for_node(
            self,
            checkpoint_path=self._checkpoint_path or None,
            compile_model=self._compile_model,
        )

    def _ensure_model(self) -> None:
        """Lazily build the SAM3 image model and interactive processor."""
        if self._model is not None and self._processor is not None:
            return

        from sam3.model.sam3_image_processor import Sam3Processor

        self._claim_shared_backbone()
        self._model = _shared.build_image_model_shared(
            checkpoint_path=self._checkpoint_path or None,
            device=self._resolved_device,
            enable_inst_interactivity=True,
            compile_model=self._compile_model,
            claim=self._backbone_claim,
        )
        self._processor = Sam3Processor(
            self._model, device=self._resolved_device, confidence_threshold=0.0
        )
        logger.info("{} image model loaded (device={})", type(self).__name__, self._resolved_device)

    def _apply(self, fn: Any, recurse: bool = True) -> _Sam3ImageNode:
        """Track framework device moves; ``pipeline.to`` reaches nodes via ``_apply``."""
        module = super()._apply(fn, recurse)
        target = _shared.resolve_apply_target_device(fn)
        if target is not None:
            self._on_pipeline_device(target)
        return module

    def _on_pipeline_device(self, device: torch.device) -> None:
        """Adopt the framework-assigned device and eagerly build on CUDA targets.

        ``super()._apply`` has already moved any built heads; the pinned shared
        backbone never moves, so a device change drops the model and rebuilds
        it against the new registry key. Without a CUDA target the build stays
        lazy (today's first-forward behavior).
        """
        new_key = _shared.device_key(device)
        if self._model is not None:
            if _shared.device_key(self._resolved_device) == new_key:
                return
            self._model = None
            self._processor = None
            reset = getattr(self, "reset", None)
            if callable(reset):
                reset()
        self._resolved_device = str(device)
        if device.type == "cuda":
            self._try_eager_build()

    def _try_eager_build(self) -> None:
        """Build the model now if possible; failures defer to the first forward."""
        if _shared.sharing_bypassed(self._compile_model):
            return
        try:
            self._ensure_model()
        except Exception as exc:
            logger.warning(
                "{} eager SAM3 build failed; deferring to first forward: {}",
                type(self).__name__,
                exc,
            )

    def _model_eval_context(self) -> contextlib.AbstractContextManager[None]:
        """Return the autocast context the SAM3 image stack expects, or a no-op.

        The SAM3 image stack emits bfloat16 activations in its vision path and
        relies on autocast to reconcile those with float32 weights.
        """
        device_type = str(self._resolved_device).split(":")[0]
        dtype = self._AUTOCAST_DTYPE.get(device_type)
        if dtype is not None:
            return torch.autocast(device_type=device_type, dtype=dtype)
        return contextlib.nullcontext()

    def cleanup(self) -> None:
        """Release the loaded SAM3 image model, processor, and backbone claim.

        Subclasses that hold extra per-frame state override this, call
        ``super().cleanup()``, and then drop their own caches. Only the node's
        own claim is released; the shared registry is never cleared here.
        """
        self._model = None
        self._processor = None
        if self._backbone_claim is not None:
            self._backbone_claim.release()
            self._backbone_claim = None

    @staticmethod
    def _normalize_frame(rgb_image: torch.Tensor) -> np.ndarray:
        """Validate the RGB frame and return an [H,W,3] float32 array clipped to [0,1]."""
        if rgb_image.ndim != 4 or int(rgb_image.shape[0]) != 1:
            raise ValueError(
                f"Expected rgb_image shape [1,H,W,3], got {tuple(int(v) for v in rgb_image.shape)}."
            )
        frame_np = np.asarray(rgb_image[0].detach().cpu().numpy(), dtype=np.float32)
        if frame_np.ndim != 3 or int(frame_np.shape[2]) != 3:
            raise ValueError(f"Expected RGB frame with shape [H,W,3], got {tuple(frame_np.shape)}.")
        return np.clip(frame_np, 0.0, 1.0)

    @staticmethod
    def _empty_output(height: int, width: int) -> dict[str, torch.Tensor]:
        """Return the empty-mask output (no object) for an HxW frame."""
        return {
            "mask": torch.zeros((1, int(height), int(width)), dtype=torch.int32),
            "object_ids": torch.zeros((1, 0), dtype=torch.int64),
            "detection_scores": torch.zeros((1, 0), dtype=torch.float32),
        }


__all__ = ["_Sam3ImageNode"]
