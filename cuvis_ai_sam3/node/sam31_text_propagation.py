"""SAM 3.1 text propagation: one RGB frame per ``forward()``, tracked on the Object Multiplex model.

SAM 3.1 tracks all objects of a prompt jointly in fixed-size buckets, so a frame costs about
the same for one object as for a dozen, where the SAM 3 tracker of
:class:`~cuvis_ai_sam3.node.sam3_streaming_propagation.SAM3TextPropagation` pays for every
object. The model keeps the dict-state call surface of SAM 3 (``add_prompt``,
``propagate_in_video``) but differs in the details the SAM 3 streaming code relies on: a text
prompt always starts over, the tracker state lives under other keys, the builder batches
sixteen frames before it yields, and the detector indexes its per-frame inputs modulo their
count. This module therefore drives the model with its own, smaller streaming adapter.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.execution import Context
from loguru import logger

from cuvis_ai_sam3 import shared_backbone as _shared

from .sam3_streaming_propagation import _STREAMING_SENTINEL, SAM3TextPropagation, _FrameBuffer

_DEFAULT_WEIGHT_NAME = "sam3.1"
"""Registry name of the SAM 3.1 checkpoint the node loads unless ``weight_name`` says otherwise."""

_COMPUTED_BUFFER_SUFFIXES = ("freqs_cis_real", "freqs_cis_imag")
"""Rotary-encoding buffers the model computes at construction; no checkpoint stores them."""

_CHECKPOINT_KEY_RENAMES = (("sam3_model.", "detector."), ("sam2_predictor.", "tracker."))
"""Prefixes of checkpoints saved before the public naming, as ``sam3.model_builder`` maps them."""

_BUFFERED_FRAMES = 2
"""Preprocessed frames kept on the device: the current one and its predecessor."""

_PER_FRAME_STATE_KEYS = (
    "previous_stages_out",
    "per_frame_raw_point_input",
    "per_frame_raw_box_input",
    "per_frame_visual_prompt",
    "per_frame_geometric_prompt",
    "per_frame_cur_step",
    "cached_frame_outputs",
)
"""Entries of the inference state that hold one item per frame."""


class _FrameBuffer31(_FrameBuffer):
    """Frame buffer that also answers ``img_batch.tensors[frame_idx]``, as SAM 3.1 reads it."""

    @property
    def tensors(self) -> _FrameBuffer31:
        """The buffer itself: the model indexes ``tensors`` by frame."""
        return self


class _FrameStages(dict):
    """Per-frame find stages of a pruned stream, with the length of the unpruned list.

    The SAM 3.1 detector reads ``find_inputs[frame_idx % len(find_inputs)]``, which assumes a
    list with one stage per frame of the video. A stream keeps only the recent stages, so the
    plain count would wrap the index onto a stage that is gone. Reporting the last frame index
    plus one keeps the modulo a no-op for every frame the stream still holds.
    """

    def __len__(self) -> int:
        """Last frame index plus one, or 0 for an empty stream."""
        return max(self) + 1 if dict.__len__(self) else 0


def _drop_frames_before(container: Any, keep_from: int) -> None:
    """Remove the frame indices below ``keep_from`` from a per-frame dict or set."""
    if isinstance(container, dict):
        for frame_idx in [k for k in container if isinstance(k, int) and k < keep_from]:
            del container[frame_idx]
    elif isinstance(container, set):
        container.difference_update([k for k in container if isinstance(k, int) and k < keep_from])


def _checkpoint_keys(checkpoint_path: str) -> set[str]:
    """Return the weight names a SAM 3.1 checkpoint stores, named as the assembled model does.

    The file is memory-mapped, so only its directory is read, not the 3.5 GB of tensors.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    if isinstance(checkpoint.get("model"), dict):
        checkpoint = checkpoint["model"]
    keys: set[str] = set()
    for key in checkpoint:
        for old, new in _CHECKPOINT_KEY_RENAMES:
            if key.startswith(old):
                key = new + key[len(old) :]
                break
        keys.add(key)
    return keys


def _missing_checkpoint_keys(model_keys: Iterable[str], checkpoint_keys: set[str]) -> list[str]:
    """Return the model weights a checkpoint does not provide, computed buffers aside."""
    return [
        key
        for key in model_keys
        if key not in checkpoint_keys and not key.endswith(_COMPUTED_BUFFER_SUFFIXES)
    ]


def _build_multiplex_model(
    checkpoint_path: str | None, *, weight_name: str, max_num_objects: int
) -> Any:
    """Build the SAM 3.1 Object Multiplex video model on the GPU and check its weights.

    ``sam3.model_builder`` loads the checkpoint non-strictly and only prints what is missing,
    so a wrong file (the SAM 3 checkpoint, for one) would run with untrained weights. The
    stored names are therefore compared with the model's own after the build.

    Parameters
    ----------
    checkpoint_path : str | None
        Local ``sam3.1_multiplex.pt``; ``None`` resolves ``weight_name`` through cuvis-ai-core's
        registry.
    weight_name : str
        Registry name of the checkpoint; not read when ``checkpoint_path`` is given.
    max_num_objects : int
        Most objects the tracker holds at once; see :class:`SAM31TextPropagation`.

    Returns
    -------
    Any
        The ``Sam3MultiplexTrackingWithInteractivity`` model in eval mode on CUDA.

    Raises
    ------
    RuntimeError
        Without CUDA, or when the checkpoint leaves model weights unset.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "SAM31TextPropagation needs a CUDA device: the SAM 3.1 model runs on the GPU only."
        )
    if checkpoint_path is None:
        from cuvis_ai_core.data.model_weights import ModelWeights

        checkpoint_path = str(ModelWeights.resolve(weight_name))

    from sam3.model_builder import build_sam3_multiplex_video_predictor

    # The sam3 library switches a bfloat16 autocast on in its constructors and never switches
    # it off; keep the thread's state as it was before the build.
    with _shared.preserve_autocast_state():
        predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=checkpoint_path,
            max_num_objects=max_num_objects,
            # FlashAttention 3 is a separate package the plugin does not depend on.
            use_fa3=False,
            warm_up=False,
            async_loading_frames=False,
        )
    model = predictor.model

    missing = _missing_checkpoint_keys(model.state_dict().keys(), _checkpoint_keys(checkpoint_path))
    if missing:
        raise RuntimeError(
            f"{checkpoint_path} is not a complete SAM 3.1 checkpoint: it leaves {len(missing)} "
            f"model weights unset (first: {', '.join(missing[:3])})."
        )
    return model


class SAM31TextPropagation(Node):
    """Track the objects a text prompt names across RGB frames with SAM 3.1 Object Multiplex.

    Same ports as :class:`~cuvis_ai_sam3.node.sam3_streaming_propagation.SAM3TextPropagation`,
    so a pipeline changes the class name only. Frames before the first ``text_prompt`` return
    an empty label map. A prompt starts the tracker on its frame; every later frame advances
    it by one step and keeps finding new objects of the prompt.

    Where it differs from the SAM 3 node:

    - A frame costs about the same for one object as for sixteen (one bucket), so it is the
      faster node once several objects are tracked and no faster for a single one.
    - A different prompt replaces the running one: tracking starts over on that frame under a
      new category, with new object ids. SAM 3.1 has no way to add a second prompt to a
      running stream. Sending the same words again changes nothing.
    - Objects are never released. Every object ever tracked keeps its slot, so once
      ``max_num_objects`` have been seen the stream stops picking up new ones until the next
      prompt or :meth:`reset`. Size ``max_num_objects`` for the scene.
    - It needs a CUDA device and the ``sam3.1`` checkpoint (about 3.5 GB).

    The node keeps tracker state between calls and drives a model that is not a fixed graph of
    tensor operations, so it cannot be part of an exported span.

    Parameters
    ----------
    checkpoint_path : str | None
        Local ``sam3.1_multiplex.pt``. ``None`` takes the registered weights ``weight_name``
        names.
    weight_name : str
        Registry name of the checkpoint. A pipeline that sets it lets a host application see
        which weights the pipeline needs before it loads; ``checkpoint_path`` wins over it.
    score_threshold_detection : float
        Lowest detection score that counts as a detection of the prompt.
    new_det_thresh : float
        Lowest detection score that opens a new track.
    det_nms_thresh : float
        IoU above which the weaker of two detections is dropped.
    overlap_suppress_thresh : float
        Overlap above which the more recently occluded of two tracks is hidden.
    max_num_objects : int
        Most objects tracked in one stream. A bucket of sixteen costs the same whether it is
        full or not; each further bucket in use adds time and memory.
    """

    _category = NodeCategory.MODEL
    _tags = frozenset(
        {
            NodeTag.VIDEO,
            NodeTag.RGB,
            NodeTag.MASK,
            NodeTag.TEXT,
            NodeTag.TRACKING,
            NodeTag.SEGMENTATION,
            NodeTag.INFERENCE,
            NodeTag.LEARNABLE,
            NodeTag.BATCHED,
            NodeTag.STATEFUL,
            NodeTag.TORCH,
        }
    )

    INPUT_SPECS = dict(SAM3TextPropagation.INPUT_SPECS)
    OUTPUT_SPECS = dict(SAM3TextPropagation.OUTPUT_SPECS)

    def __init__(
        self,
        checkpoint_path: str | None = None,
        weight_name: str = _DEFAULT_WEIGHT_NAME,
        score_threshold_detection: float = 0.4,
        new_det_thresh: float = 0.65,
        det_nms_thresh: float = 0.1,
        overlap_suppress_thresh: float = 0.7,
        max_num_objects: int = 64,
        **kwargs: Any,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._weight_name = weight_name
        self._score_threshold_detection = score_threshold_detection
        self._new_det_thresh = new_det_thresh
        self._det_nms_thresh = det_nms_thresh
        self._overlap_suppress_thresh = overlap_suppress_thresh
        self._max_num_objects = max_num_objects

        self._model: Any = None
        # Frames of tracker memory to keep; read off the model when it is built.
        self._keep_recent: int = 16
        self._frame_buffer: _FrameBuffer31 | None = None
        self._inference_state: dict[str, Any] | None = None
        self._generator: Any = None
        self._active_prompt: str | None = None
        self._active_category_id: int = 0
        self._category_ids: dict[str, int] = {}
        self._export_ids: dict[int, int] = {}
        self._next_export_id: int = 1

        super().__init__(
            checkpoint_path=checkpoint_path,
            weight_name=weight_name,
            score_threshold_detection=score_threshold_detection,
            new_det_thresh=new_det_thresh,
            det_nms_thresh=det_nms_thresh,
            overlap_suppress_thresh=overlap_suppress_thresh,
            max_num_objects=max_num_objects,
            **kwargs,
        )

    # -- Model ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Build the model once and set it up for a frame-by-frame stream."""
        if self._model is not None:
            return
        model = _build_multiplex_model(
            self._checkpoint_path or None,
            weight_name=self._weight_name,
            max_num_objects=self._max_num_objects,
        )
        # The builder is set up for whole videos: it holds outputs back for fifteen frames to
        # drop short-lived tracks, hands results out sixteen at a time and runs the detector
        # on batches of frames. A stream has one frame at a time and needs its mask now.
        model.hotstart_delay = 0
        model.postprocess_batch_size = 1
        model.use_batched_grounding = False
        model.score_threshold_detection = self._score_threshold_detection
        model.new_det_thresh = self._new_det_thresh
        model.det_nms_thresh = self._det_nms_thresh
        model.suppress_overlapping_based_on_recent_occlusion_threshold = (
            self._overlap_suppress_thresh
        )
        tracker = getattr(model, "tracker", None)
        self._keep_recent = max(
            int(getattr(tracker, "max_obj_ptrs_in_encoder", 16)),
            int(getattr(tracker, "num_maskmem", 7)),
        )
        self._guard_detector_lookahead(model)
        self._model = model
        logger.info(
            "SAM 3.1 model loaded (image_size={}, max_num_objects={}, memory window={} frames)",
            getattr(model, "image_size", "?"),
            self._max_num_objects,
            self._keep_recent,
        )

    def _guard_detector_lookahead(self, model: Any) -> None:
        """Keep the detector from computing frames the stream has not received yet.

        The detector works a chunk ahead and caches it; told the true number of buffered
        frames it has nothing to look ahead to, and an emptied cache cannot serve a chunk
        that was computed for another frame.
        """
        detector = getattr(model, "detector", None)
        if detector is None or getattr(detector, "_streaming_guard_installed", False):
            return
        forward_grounding = detector.forward_video_grounding_multigpu

        def forward_grounding_on_buffered_frames(*args: Any, **kwargs: Any) -> Any:
            """Run the detector with the buffered frame count and an empty chunk cache."""
            buffered = len(self._frame_buffer) if self._frame_buffer is not None else 0
            if buffered and kwargs.get("num_frames") is not None:
                kwargs["num_frames"] = min(buffered, int(kwargs["num_frames"]))
            if isinstance(kwargs.get("multigpu_buffer"), dict):
                kwargs["multigpu_buffer"].clear()
            return forward_grounding(*args, **kwargs)

        detector.forward_video_grounding_multigpu = forward_grounding_on_buffered_frames
        detector._streaming_guard_installed = True

    def _model_device(self) -> torch.device:
        """Device the loaded model lives on."""
        try:
            return next(self._model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _inference_context(self) -> torch.autocast:
        """Autocast the model was trained to run under (bfloat16)."""
        return torch.autocast(device_type=self._model_device().type, dtype=torch.bfloat16)

    def _apply(self, fn: Any, recurse: bool = True) -> SAM31TextPropagation:
        """Load the model when the pipeline moves to a GPU, not on the first prompt."""
        module = super()._apply(fn, recurse)
        target = _shared.resolve_apply_target_device(fn)
        if target is not None and target.type == "cuda" and self._model is None:
            try:
                self._ensure_model()
            except Exception as exc:
                logger.warning(
                    "SAM31TextPropagation could not load its model yet; "
                    "it will try again on the first prompt: {}",
                    exc,
                )
        return module

    # -- Lifecycle --------------------------------------------------------------

    def _drop_stream(self) -> None:
        """Forget the running stream; the model, the categories and the id counter stay."""
        self._generator = None
        self._frame_buffer = None
        self._inference_state = None
        self._active_prompt = None
        self._active_category_id = 0

    def reset(self) -> None:
        """Start over for a new video: no prompt, no tracks, ids from 1. The model stays loaded."""
        self._drop_stream()
        self._category_ids.clear()
        self._export_ids.clear()
        self._next_export_id = 1

    def cleanup(self) -> None:
        """Release the stream and the loaded model."""
        self.reset()
        self._model = None

    # -- Stream state -----------------------------------------------------------

    def _build_state(self, height: int, width: int) -> dict[str, Any]:
        """Inference state of an empty stream, laid out as the model's ``init_state`` does."""
        from sam3.model.data_misc import BatchedDatapoint
        from sam3.model.geometry_encoders import Prompt

        device = self._model_device()
        state: dict[str, Any] = {
            "image_size": int(self._model.image_size),
            "num_frames": 0,
            "device": device,
            "orig_height": int(height),
            "orig_width": int(width),
            "constants": {
                "empty_geometric_prompt": Prompt(
                    box_embeddings=torch.zeros(0, 1, 4, device=device),
                    box_mask=torch.zeros(1, 0, device=device, dtype=torch.bool),
                    box_labels=torch.zeros(0, 1, device=device, dtype=torch.long),
                    point_embeddings=torch.zeros(0, 1, 2, device=device),
                    point_mask=torch.zeros(1, 0, device=device, dtype=torch.bool),
                    point_labels=torch.zeros(0, 1, device=device, dtype=torch.long),
                )
            },
            "input_batch": BatchedDatapoint(
                img_batch=self._frame_buffer,
                find_text_batch=["<text placeholder>", "visual", "geometric"],
                find_inputs=_FrameStages(),
                find_targets={},
                find_metadatas={},
            ),
            "text_prompt": None,
            "backbone_out": None,
            "visual_prompt_embed": None,
            "visual_prompt_mask": None,
            "sam2_inference_states": [],
            "tracker_metadata": {},
            "feature_cache": {},
            "action_history": [],
            "is_image_only": False,
        }
        for key in _PER_FRAME_STATE_KEYS:
            state[key] = {}
        return state

    def _add_frame_to_state(self, frame_idx: int) -> None:
        """Give the state the per-frame entries the model expects for a new frame."""
        from sam3.model.data_misc import BatchedPointer, FindStage, convert_my_tensors
        from sam3.model.utils.misc import copy_data_to_device

        state = self._inference_state
        no_pointers = BatchedPointer(
            stage_ids=[], query_ids=[], object_ids=[], ptr_mask=[], ptr_types=[]
        )
        stage = FindStage(
            img_ids=[frame_idx],
            img_ids_np=np.array([frame_idx]),
            text_ids=[0],
            input_boxes=[torch.zeros(258)],
            input_boxes_before_embed=[torch.empty(0, 4)],
            input_boxes_mask=[torch.empty(0, dtype=torch.bool)],
            input_boxes_label=[torch.empty(0, dtype=torch.long)],
            input_points=[torch.empty(0, 257)],
            input_points_before_embed=[torch.empty(0, 3)],
            input_points_mask=[torch.empty(0)],
            ptrs=no_pointers,
            ptrs_seg=no_pointers,
            object_ids=[],
        )
        stage = copy_data_to_device(
            convert_my_tensors(stage), self._model_device(), non_blocking=True
        )
        batch = state["input_batch"]
        batch.find_inputs[frame_idx] = stage
        batch.find_targets[frame_idx] = None
        batch.find_metadatas[frame_idx] = None
        for key in _PER_FRAME_STATE_KEYS:
            if key != "cached_frame_outputs":
                state[key][frame_idx] = 0 if key == "per_frame_cur_step" else None
        state["num_frames"] = max(int(state["num_frames"]), frame_idx + 1)

    def _start_stream(self, frame: np.ndarray, prompt: str) -> int:
        """Open a stream on ``frame`` with ``prompt`` and return its frame index (0).

        SAM 3.1 clears its tracker on every text prompt, so a prompt always opens a new
        stream; a running one is dropped first.
        """
        self._ensure_model()
        self._drop_stream()
        self._frame_buffer = _FrameBuffer31(
            image_size=int(self._model.image_size), device=self._model_device()
        )
        self._frame_buffer.add(frame)
        state = self._inference_state = self._build_state(frame.shape[0], frame.shape[1])
        self._add_frame_to_state(0)
        try:
            with self._inference_context():
                self._model.add_prompt(state, frame_idx=0, text_str=prompt)
                # An empty history makes the model track every object of the prompt.
                state["action_history"].clear()
                # The stream has no last frame: the model iterates until the node stops asking.
                state["num_frames"] = _STREAMING_SENTINEL
                for tracker_state in state["sam2_inference_states"]:
                    tracker_state["num_frames"] = _STREAMING_SENTINEL
                self._generator = self._model.propagate_in_video(
                    state, start_frame_idx=0, max_frame_num_to_track=None, reverse=False
                )
        except Exception:
            self._drop_stream()
            raise
        self._active_prompt = prompt
        self._active_category_id = self._category_ids.setdefault(
            prompt, len(self._category_ids) + 1
        )
        # The model numbers the objects of a stream from 0 again; exported ids keep counting.
        self._export_ids.clear()
        return 0

    def _append_frame(self, frame: np.ndarray) -> int:
        """Buffer the next frame of the running stream and return its frame index."""
        state = self._inference_state
        size = (int(frame.shape[0]), int(frame.shape[1]))
        expected = (state["orig_height"], state["orig_width"])
        if size != expected:
            raise ValueError(
                f"SAM31TextPropagation: frame size changed from {expected} to {size} in a "
                "running stream; reset the node or send a new prompt."
            )
        frame_idx = self._frame_buffer.add(frame)
        self._add_frame_to_state(frame_idx)
        return frame_idx

    def _next_output(self) -> dict[str, Any] | None:
        """Advance the tracker by one frame; a failure ends the stream and surfaces once."""
        try:
            with self._inference_context():
                _, postprocessed = next(self._generator)
        except StopIteration:
            self._drop_stream()
            raise RuntimeError(
                "SAM31TextPropagation: the tracker stopped before the stream did."
            ) from None
        except Exception:
            # A generator that raised is closed for good; the next prompt opens a new stream.
            self._drop_stream()
            raise
        return postprocessed

    def _prune_state(self, frame_idx: int) -> None:
        """Drop what the tracker no longer reads, so a long stream stays bounded in memory."""
        state = self._inference_state
        keep_from = frame_idx - self._keep_recent + 1
        self._frame_buffer.prune_before(frame_idx - _BUFFERED_FRAMES + 1)

        batch = state["input_batch"]
        metadata = state["tracker_metadata"]
        per_frame = [state[key] for key in _PER_FRAME_STATE_KEYS]
        per_frame += [batch.find_inputs, batch.find_targets, batch.find_metadatas]
        per_frame.append(metadata.get("obj_id_to_sam2_score_frame_wise"))
        per_frame.append(metadata.get("rank0_metadata", {}).get("suppressed_obj_ids"))
        for container in per_frame:
            _drop_frames_before(container, keep_from)

        for tracker_state in state["sam2_inference_states"]:
            self._prune_tracker_state(tracker_state, keep_from)

    @staticmethod
    def _prune_tracker_state(tracker_state: dict[str, Any], keep_from: int) -> None:
        """Keep the memory window of one tracker bucket state.

        Conditioning frames stay: the tracker attends to them for as long as the stream runs.
        """
        temp_outputs = list(tracker_state["temp_output_dict_per_obj"].values())
        outputs = [tracker_state["output_dict"], *tracker_state["output_dict_per_obj"].values()]
        for output in outputs + temp_outputs:
            _drop_frames_before(output["non_cond_frame_outputs"], keep_from)
        consolidated = tracker_state["consolidated_frame_inds"]
        _drop_frames_before(consolidated["non_cond_frame_outputs"], keep_from)
        _drop_frames_before(tracker_state["frames_already_tracked"], keep_from)

        # The tracker asserts that every frame with a mask or point input is consolidated or
        # still pending, so an input goes only once its frame has left both.
        held: set[int] = set().union(*consolidated.values())
        for temp_output in temp_outputs:
            held.update(*temp_output.values())
        for inputs_key in ("mask_inputs_per_obj", "point_inputs_per_obj"):
            for inputs in tracker_state[inputs_key].values():
                for frame in [f for f in inputs if f < keep_from and f not in held]:
                    del inputs[frame]

    # -- Outputs ----------------------------------------------------------------

    def _category_semantics(self) -> torch.Tensor:
        """UTF-8 JSON bytes of the category-id-to-text mapping of this stream."""
        payload = json.dumps(
            {str(category_id): text for text, category_id in self._category_ids.items()},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return torch.tensor(list(payload), dtype=torch.uint8)

    def _empty_output(self, height: int, width: int) -> dict[str, torch.Tensor]:
        """Outputs of a frame without objects: a zero label map and empty id lists."""
        return {
            "mask": torch.zeros(1, height, width, dtype=torch.int32),
            "object_ids": torch.zeros(1, 0, dtype=torch.int64),
            "detection_scores": torch.zeros(1, 0, dtype=torch.float32),
            "category_ids": torch.zeros(1, 0, dtype=torch.int64),
            "category_semantics": self._category_semantics(),
        }

    def _pack_output(
        self, postprocessed: dict[str, Any] | None, frame_shape: tuple[int, int]
    ) -> dict[str, torch.Tensor]:
        """Turn the model's per-object masks into one label map with stable ids from 1."""
        if postprocessed is None or len(postprocessed["out_obj_ids"]) == 0:
            return self._empty_output(*frame_shape)

        export_ids = []
        for internal_id in torch.as_tensor(postprocessed["out_obj_ids"]).reshape(-1).tolist():
            if internal_id not in self._export_ids:
                self._export_ids[internal_id] = self._next_export_id
                self._next_export_id += 1
            export_ids.append(self._export_ids[internal_id])

        masks = torch.as_tensor(postprocessed["out_binary_masks"], dtype=torch.bool)
        label_map = torch.zeros(masks.shape[-2:], dtype=torch.int32, device=masks.device)
        for export_id, mask in zip(export_ids, masks, strict=True):
            label_map[mask] = export_id

        scores = torch.as_tensor(postprocessed["out_probs"], dtype=torch.float32)
        return {
            "mask": label_map.unsqueeze(0),
            "object_ids": torch.tensor([export_ids], dtype=torch.int64),
            "detection_scores": scores.reshape(1, -1),
            "category_ids": torch.full(
                (1, len(export_ids)), self._active_category_id, dtype=torch.int64
            ),
            "category_semantics": self._category_semantics(),
        }

    # -- Forward ----------------------------------------------------------------

    def forward(
        self,
        rgb_image: torch.Tensor,
        text_prompt: str | None = None,
        frame_id: torch.Tensor | None = None,  # noqa: ARG002 -- read by the sinks, not here
        context: Context | None = None,  # noqa: ARG002
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Process one RGB frame, starting or restarting the tracker when a prompt arrives.

        Parameters
        ----------
        rgb_image : torch.Tensor
            Shape ``[1, H, W, 3]``, float32, values in ``[0, 1]``.
        text_prompt : str | None
            What to track. A prompt that differs from the running one starts tracking over on
            this frame; ``None``, blank text or the running prompt keeps the stream as it is.
        frame_id : torch.Tensor | None
            Source frame index, accepted so the node wires like the SAM 3 node; the stream
            counts its own frames.

        Returns
        -------
        dict[str, torch.Tensor]
            ``mask`` (int32 label map ``[1, H, W]``, 0 is background), ``object_ids``,
            ``detection_scores``, ``category_ids`` (all ``[1, N]``) and ``category_semantics``
            (UTF-8 JSON bytes of the category-id-to-text mapping).
        """
        frame = rgb_image[0].detach().cpu().numpy()
        frame_shape = (int(frame.shape[0]), int(frame.shape[1]))
        prompt = SAM3TextPropagation._normalize_runtime_text_prompt(text_prompt)

        if prompt is not None and prompt != self._active_prompt:
            frame_idx = self._start_stream(frame, prompt)
        elif self._inference_state is None:
            return _shared.outputs_on_device(self._empty_output(*frame_shape), rgb_image.device)
        else:
            frame_idx = self._append_frame(frame)

        result = self._pack_output(self._next_output(), frame_shape)
        self._prune_state(frame_idx)
        if frame_idx > 0 and frame_idx % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return _shared.outputs_on_device(result, rgb_image.device)
