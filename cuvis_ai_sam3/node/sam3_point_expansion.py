"""Interactive single-frame SAM3 point-expansion node.

Unlike the streaming propagation nodes (which advance an internal frame index on
every ``forward`` call), this node segments ONE object on a SINGLE frame from
positive/negative click points, and is re-promptable in place: re-sending an
updated point set for the same frame refreshes the mask without re-embedding the
image. That makes interactive refinement (click, see mask, add a correction
point, see a better mask) real-time.

The node wraps the SAM3 interactive image predictor the same way
``SAM3SegmentEverything`` does: ``build_sam3_image_model`` plus ``Sam3Processor``,
with ``set_image`` computing the (expensive) ViT embedding once per frame and
``predict_inst`` running only the lightweight mask decoder per click.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PortSpec
from loguru import logger
from PIL import Image

from ._sam3_image_base import _Sam3ImageNode

_POSITIVE = "positive"
_NEGATIVE = "negative"
_NEUTRAL = "neutral"


class SAM3PointExpansion(_Sam3ImageNode):
    """Expand positive/negative click points into one object mask on a single frame.

    The optional ``points`` input is a per-frame list of prompt dicts with keys
    ``element_id``, ``x``, ``y`` (pixel coordinates), and ``type`` (one of
    ``positive``, ``negative``, ``neutral``). Positive points mark the object,
    negative points mark background, neutral points are ignored. All points
    address a single object whose id is the ``prompt_obj_id`` hparam.

    Frames before the first prompt (and frames with no positive point) emit an
    empty mask. The image embedding is cached by ``frame_id`` so re-prompting the
    same frame only re-runs the mask decoder; the mask is a deterministic function
    of the current point set (no low-res feedback), so the same points always yield
    the same mask.
    """

    _category = NodeCategory.MODEL
    _tags = frozenset(
        {
            NodeTag.RGB,
            NodeTag.IMAGE,
            NodeTag.MASK,
            NodeTag.KEYPOINTS,
            NodeTag.SEGMENTATION,
            NodeTag.INFERENCE,
            NodeTag.LEARNABLE,
            NodeTag.BATCHED,
            NodeTag.TORCH,
        }
    )

    INPUT_SPECS = {
        "rgb_frame": PortSpec(
            dtype=torch.float32,
            shape=(1, -1, -1, 3),
            description="RGB frame [1,H,W,3] in float32 with values in [0,1].",
        ),
        "points": PortSpec(
            dtype=list,
            shape=(),
            description=(
                "Optional per-frame list of point prompt dicts with keys element_id, x, y, type "
                "(type in {positive, negative, neutral}). Positive=object, negative=background."
            ),
            optional=True,
        ),
        "frame_id": PortSpec(
            dtype=torch.int64,
            shape=(1,),
            description="Optional source frame index [1]; keys the image-embedding cache.",
            optional=True,
        ),
    }
    OUTPUT_SPECS = {
        "mask": PortSpec(dtype=torch.int32, shape=(1, -1, -1)),
        "object_ids": PortSpec(dtype=torch.int64, shape=(1, -1)),
        "detection_scores": PortSpec(dtype=torch.float32, shape=(1, -1)),
    }

    def __init__(
        self,
        checkpoint_path: str | None = None,
        device: str | None = None,
        compile_model: bool = False,
        prompt_obj_id: int = 1,
        multimask_output: bool = True,
        mask_threshold: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Configure the interactive point-expansion node.

        Args:
            checkpoint_path: Optional SAM3 image-model checkpoint path.
            device: Torch device override; defaults to cuda when available, else cpu.
            compile_model: Whether to ``torch.compile`` the image model.
            prompt_obj_id: Object id written into the output label map (> 0).
            multimask_output: Request up to 3 candidate masks on a single positive
                click and keep the highest-scoring one.
            mask_threshold: Logit threshold for binarizing the predicted mask.
        """
        if int(prompt_obj_id) <= 0:
            raise ValueError(f"prompt_obj_id must be > 0, got {prompt_obj_id}.")

        self._prompt_obj_id = int(prompt_obj_id)
        self._multimask_output = bool(multimask_output)
        self._mask_threshold = float(mask_threshold)

        self._cached_frame_id: int | None = None
        self._cached_state: Any | None = None
        self._warned_missing_frame_id = False

        super().__init__(
            checkpoint_path=checkpoint_path,
            device=device,
            compile_model=compile_model,
            prompt_obj_id=prompt_obj_id,
            multimask_output=multimask_output,
            mask_threshold=mask_threshold,
            **kwargs,
        )

    def cleanup(self) -> None:
        """Release the SAM3 model/processor and drop the cached image embedding."""
        super().cleanup()
        self._cached_frame_id = None
        self._cached_state = None

    def reset(self) -> None:
        """Drop the cached image embedding so the next predict run re-embeds.

        The base ``Node`` defines no ``reset``; the framework's per-run reset calls
        this only to clear our cross-frame cache (frame ids repeat across sessions).
        """
        self._cached_frame_id = None
        self._cached_state = None

    @staticmethod
    def _resolve_frame_id(frame_id: torch.Tensor | None) -> int | None:
        """Extract the scalar source frame id, or None when not provided."""
        if frame_id is None or frame_id.numel() == 0:
            return None
        return int(frame_id.reshape(-1)[0].item())

    @classmethod
    def _parse_points(cls, points: list[dict[str, Any]] | None) -> tuple[np.ndarray, np.ndarray]:
        """Convert the runtime point list into pixel coords [M,2] and labels [M] (1=pos, 0=neg).

        Neutral points are dropped. Raises if the structure is malformed or no positive
        point is present.
        """
        if not points:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int32)
        if not isinstance(points, list):
            raise ValueError(f"Expected points to be a list of dicts, got {type(points).__name__}.")

        coords: list[list[float]] = []
        labels: list[int] = []
        for idx, raw in enumerate(points):
            if not isinstance(raw, dict):
                raise ValueError(
                    f"Expected point prompt at index {idx} to be a dict, got {type(raw).__name__}."
                )
            point_type = str(raw.get("type", _POSITIVE)).lower()
            if point_type == _NEUTRAL:
                continue
            if point_type not in (_POSITIVE, _NEGATIVE):
                raise ValueError(
                    f"Point prompt at index {idx} has unknown type {point_type!r}; "
                    "expected positive, negative, or neutral."
                )
            if "x" not in raw or "y" not in raw:
                raise ValueError(f"Point prompt at index {idx} is missing 'x' or 'y'.")
            coords.append([float(raw["x"]), float(raw["y"])])
            labels.append(1 if point_type == _POSITIVE else 0)

        if not any(label == 1 for label in labels):
            raise ValueError("SAM3PointExpansion requires at least one positive point.")

        return (
            np.asarray(coords, dtype=np.float32),
            np.asarray(labels, dtype=np.int32),
        )

    def forward(
        self,
        rgb_frame: torch.Tensor,
        points: list[dict[str, Any]] | None = None,
        frame_id: torch.Tensor | None = None,
        context: Context | None = None,  # noqa: ARG002
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Expand the current point set into one object mask on this frame."""
        frame_np = self._normalize_frame(rgb_frame)
        height, width = int(frame_np.shape[0]), int(frame_np.shape[1])

        # Drop neutral points; bail out early when there is nothing to segment.
        has_positive = bool(
            points and any(str(p.get("type", _POSITIVE)).lower() == _POSITIVE for p in points)
        )
        if not has_positive:
            return self._empty_output(height, width)

        point_coords, point_labels = self._parse_points(points)

        self._ensure_model()
        inference_state = self._embed_frame(frame_np, frame_id)

        multimask = bool(self._multimask_output) and point_coords.shape[0] == 1
        with self._model_eval_context():
            masks_np, scores_np, _ = self._model.predict_inst(
                inference_state,
                point_coords=point_coords[None, :, :],
                point_labels=point_labels[None, :],
                multimask_output=multimask,
                return_logits=True,
                normalize_coords=True,
            )

        return self._pack_output(masks_np, scores_np, height=height, width=width)

    def _embed_frame(self, frame_np: np.ndarray, frame_id: torch.Tensor | None) -> Any:
        """Return the cached image embedding state, re-embedding only when the frame changed."""
        current_frame_id = self._resolve_frame_id(frame_id)
        if (
            current_frame_id is not None
            and current_frame_id == self._cached_frame_id
            and self._cached_state is not None
        ):
            return self._cached_state

        if current_frame_id is None and not self._warned_missing_frame_id:
            logger.warning(
                "SAM3PointExpansion received no frame_id; re-embedding every call "
                "(wire frame_id for real-time refinement)."
            )
            self._warned_missing_frame_id = True

        image = Image.fromarray((frame_np * 255.0).clip(0.0, 255.0).astype(np.uint8))
        with self._model_eval_context():
            inference_state = self._processor.set_image(image)

        self._cached_frame_id = current_frame_id
        self._cached_state = inference_state
        return inference_state

    def _pack_output(
        self,
        masks_np: np.ndarray,
        scores_np: np.ndarray,
        *,
        height: int,
        width: int,
    ) -> dict[str, torch.Tensor]:
        """Pick the highest-scoring candidate and rasterize it to an int32 label map."""
        mask_logits = torch.as_tensor(masks_np, dtype=torch.float32)
        if mask_logits.ndim == 3:
            mask_logits = mask_logits.unsqueeze(0)
        if mask_logits.ndim != 4:
            raise ValueError(
                f"SAM3PointExpansion expected masks with shape [B,C,H,W], got {tuple(mask_logits.shape)}."
            )

        scores = torch.as_tensor(scores_np, dtype=torch.float32).reshape(-1)
        candidates = mask_logits[0]
        if candidates.shape[0] == 0 or scores.numel() == 0:
            return self._empty_output(height, width)

        best_idx = int(torch.argmax(scores).item())
        best_logits = candidates[best_idx]
        best_score = float(scores[best_idx].item())

        binary = best_logits > self._mask_threshold
        if not bool(binary.any()):
            return self._empty_output(height, width)

        label_map = torch.zeros((1, height, width), dtype=torch.int32)
        label_map[0][binary] = int(self._prompt_obj_id)
        return {
            "mask": label_map,
            "object_ids": torch.tensor([[int(self._prompt_obj_id)]], dtype=torch.int64),
            "detection_scores": torch.tensor([[best_score]], dtype=torch.float32),
        }


__all__ = ["SAM3PointExpansion"]
