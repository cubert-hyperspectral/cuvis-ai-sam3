"""Lazy VideoIterator adapter and SAM3 model subclass for on-demand frame loading."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from cuvis_ai_core.data.video import VideoIterator

from sam3.model.sam3_video_inference import Sam3VideoInferenceWithInstanceInteractivity


class VideoIteratorWithSAM3Preparation(VideoIterator):
    """Extends cuvis-ai ``VideoIterator`` with SAM3-specific preprocessing.

    Each ``__getitem__`` call lazily loads and preprocesses a single frame,
    matching the pipeline used by ``LazyCv2VideoFrameLoader``
    (``sam3/model/io_utils.py``).
    """

    def __init__(
        self,
        source_path: str,
        image_size: int,
        compute_device: torch.device | str,
    ) -> None:
        super().__init__(source_path)
        self.image_size = image_size
        self.device = torch.device(compute_device) if isinstance(compute_device, str) else compute_device

        # SAM3 normalization constants — match sam3/model/io_utils.py
        self._img_mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.bfloat16).view(3, 1, 1).to(self.device)
        self._img_std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.bfloat16).view(3, 1, 1).to(self.device)

    # -- frame preprocessing ------------------------------------------------

    def _read_frame(self, frame_id: int) -> torch.Tensor:
        """Load, preprocess, and return a single frame as a (C, H, W) bfloat16 tensor."""
        frame_bgr = self.get_frame(frame_id)["image"]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(
            frame_rgb,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_CUBIC,
        )
        # CRITICAL: Do NOT divide by 255.  SAM3 cv2 video pipeline keeps
        # values in 0-255 float range before normalization.
        frame_np = frame_resized.astype(np.float32)
        frame_tensor = torch.from_numpy(frame_np).permute(2, 0, 1)  # (C, H, W)
        frame_tensor = frame_tensor.to(device=self.device, dtype=torch.bfloat16)
        frame_tensor -= self._img_mean
        frame_tensor /= self._img_std
        return frame_tensor

    # -- indexing protocol (mirrors LazyCv2VideoFrameLoader) ----------------

    def __getitem__(self, index) -> torch.Tensor:  # type: ignore[override]
        if isinstance(index, (int, np.integer)):
            return self._read_frame(int(index))
        if isinstance(index, torch.Tensor):
            if index.numel() == 1:
                return self._read_frame(index.item()).unsqueeze(0)
            return torch.stack([self._read_frame(i) for i in index.tolist()])
        raise TypeError(
            f"VideoIteratorWithSAM3Preparation indices must be int or torch.Tensor, got {type(index)}"
        )

    def __len__(self) -> int:
        return self.num_frames

    def to(self, device, *args, **kwargs):  # noqa: ANN201
        """Satisfy ``copy_data_to_device`` protocol."""
        self.device = torch.device(device) if isinstance(device, str) else device
        self._img_mean = self._img_mean.to(self.device)
        self._img_std = self._img_std.to(self.device)
        return self


class Sam3VideoInferenceWithVideoIterator(Sam3VideoInferenceWithInstanceInteractivity):
    """Override ``init_state`` to use lazy ``VideoIterator`` instead of eager frame loading."""

    @torch.inference_mode()
    def init_state(
        self,
        resource_path,
        offload_video_to_cpu=False,
        async_loading_frames=False,
        video_loader_type="cv2",
    ) -> dict:
        """Initialize inference state with lazy frame loading via VideoIterator."""
        video = VideoIteratorWithSAM3Preparation(
            source_path=resource_path,
            image_size=self.image_size,
            compute_device=self.device,
        )

        inference_state = {}
        inference_state["image_size"] = self.image_size
        inference_state["num_frames"] = len(video)
        inference_state["orig_height"] = video.image_height
        inference_state["orig_width"] = video.image_width
        inference_state["constants"] = {}

        # Pass lazy iterator as images — SAM3 handles non-tensor img_batch
        # via per-index fallback in sam3_image.py:143-150
        self._construct_initial_input_batch(inference_state, video)

        inference_state["tracker_inference_states"] = []
        inference_state["tracker_metadata"] = {}
        inference_state["feature_cache"] = {}
        inference_state["cached_frame_outputs"] = {}
        inference_state["action_history"] = []
        inference_state["is_image_only"] = False

        # Stash reference for metadata access (frame_rate, dimensions)
        inference_state["_video_iterator"] = video

        return inference_state
