"""Process-wide sharing of the SAM3 vision-language backbone across nodes.

The 3.45GB SAM3 checkpoint is dominated by the ViT-based vision-language
backbone (~3.27GB); the detector/tracker heads account for the remaining
~0.18GB. The image and video model families construct the exact same backbone
architecture, so a single GPU-resident instance can serve every SAM3 node in a
process. This module keeps a registry of built backbones keyed by architecture
and checkpoint identity, together with the CPU-side state-dict splits needed to
rebuild models without re-reading the multi-GB checkpoint from disk.

Lifetime is claim-based: nodes claim their logical entry at construction and
release it in ``cleanup()`` (or implicitly when garbage-collected). When the
last claim is released, every device-resident backbone instance is freed to
return VRAM; the CPU state dicts are retained so a later build re-warms without
disk IO.

Device placement contract: a shared backbone is built directly on its target
device and then pinned there. Its ``train()`` and ``_apply()`` are overridden
so containing models can neither flip it out of eval mode nor move it across
devices or dtypes; model builders receiving an injected backbone therefore
never relocate it, and every device gets its own instance.

Set ``CUVIS_SAM3_NO_BACKBONE_SHARING=1`` (or request model compilation) to
bypass sharing entirely and build standalone models exactly as before.
"""

from __future__ import annotations

import os
import threading
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch import nn

_BACKBONE_PREFIX = "detector.backbone."
_BYPASS_ENV_VAR = "CUVIS_SAM3_NO_BACKBONE_SHARING"
# Sentinel checkpoint identity for the default Hugging Face checkpoint. Claims
# taken at node construction must stay network-free, so the actual cache path
# is resolved lazily at first build.
_HF_DEFAULT_IDENTITY: tuple[str, str] = ("hf", "sam3")

_LOCK = threading.RLock()
_ENTRIES: dict[tuple[str, tuple[Any, ...]], _Entry] = {}
_PINNED_CLASSES: dict[type, type] = {}


def sharing_bypassed(compile_model: bool = False) -> bool:
    """Return whether backbone sharing is bypassed for the current call.

    Sharing is bypassed when model compilation is requested (compiled models
    monkey-patch their backbone's ``forward``) or when the
    ``CUVIS_SAM3_NO_BACKBONE_SHARING`` environment variable is truthy. The
    environment is re-read on every call.
    """
    if compile_model:
        return True
    return os.environ.get(_BYPASS_ENV_VAR, "").strip().lower() in {"1", "true", "yes", "on"}


def _vit_architecture_id(enable_inst_interactivity: bool) -> str:
    """Return the identifier of the ViT backbone variant the builders construct.

    Encodes the architecture-affecting builder flags: the ViT variant
    (embed dim 1024, depth 32, 1008px input) and whether the SAM2 instance
    neck is attached (``enable_inst_interactivity`` changes the neck).
    """
    base = "vit1024d32_1008"
    return f"{base}-instneck" if enable_inst_interactivity else base


def _checkpoint_identity(checkpoint_path: str | None) -> tuple[Any, ...]:
    """Resolve a checkpoint spec to a stable identity tuple.

    Explicit paths resolve to ``(absolute_path, mtime_ns, size)`` so a replaced
    checkpoint file yields a new logical entry. ``None`` (the default Hugging
    Face checkpoint) maps to a fixed sentinel so claims never touch the network.
    """
    if checkpoint_path is None:
        return _HF_DEFAULT_IDENTITY
    resolved = Path(checkpoint_path).resolve()
    stat = resolved.stat()
    return (str(resolved), stat.st_mtime_ns, stat.st_size)


def device_key(device: str | torch.device) -> str:
    """Normalize a device spec to a stable registry key (cuda gets an index)."""
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        index = torch.cuda.current_device() if torch.cuda.is_available() else 0
        resolved = torch.device("cuda", index)
    return str(resolved)


def resolve_apply_target_device(fn: Any) -> torch.device | None:
    """Infer the target device of an ``nn.Module._apply`` conversion function.

    ``pipeline.to(device)`` reaches nodes through ``_apply`` recursion, not
    through ``Node.to``, so the target device must be probed from the
    conversion closure. Returns ``None`` when the function does not move
    tensors between devices (for example a dtype-only conversion), so callers
    can ignore those passes.
    """
    try:
        cpu_probe = fn(torch.empty(0))
    except Exception:
        return None
    if not isinstance(cpu_probe, torch.Tensor):
        return None
    if cpu_probe.device.type != "cpu":
        return cpu_probe.device
    # cpu -> cpu is ambiguous: dtype-only conversions look identical to an
    # explicit move to cpu. A cuda probe settles it whenever cuda exists.
    if torch.cuda.is_available():
        try:
            cuda_probe = fn(torch.empty(0, device="cuda"))
        except Exception:
            return None
        if isinstance(cuda_probe, torch.Tensor) and cuda_probe.device.type == "cpu":
            return torch.device("cpu")
    return None


@dataclass
class _DeviceSlot:
    """One device-resident shared backbone instance and its claimant count."""

    backbone: nn.Module
    users: int = 0


@dataclass
class _Entry:
    """Registry state for one (architecture, checkpoint) logical backbone."""

    key: tuple[str, tuple[Any, ...]]
    architecture_id: str
    enable_inst_interactivity: bool
    checkpoint_path: str | None
    heads_sd: dict[str, torch.Tensor] | None = None
    backbone_sd: dict[str, torch.Tensor] | None = None
    slots: dict[str, _DeviceSlot] = field(default_factory=dict)
    claim_count: int = 0
    build_count: int = 0
    reuse_count: int = 0
    loader_calls: int = 0


class BackboneClaim:
    """Handle on a shared-backbone logical entry, held by one SAM3 node.

    Claims keep the entry's device-resident backbone instances alive; when the
    last claim is released the instances are freed (the CPU state dicts stay).
    Nodes release explicitly from ``cleanup()``; a ``weakref.finalize`` hook
    registered in :func:`claim_backbone` releases at garbage collection as a
    backstop. ``release`` is idempotent.
    """

    def __init__(self, key: tuple[str, tuple[Any, ...]]) -> None:
        """Record the logical entry key; claim counting happens in claim_backbone."""
        self._key = key
        self._released = False
        self._device_key: str | None = None

    @property
    def released(self) -> bool:
        """Whether this claim has already been released."""
        return self._released

    def release(self) -> None:
        """Release the claim (idempotent); frees device instances at zero claims."""
        with _LOCK:
            if self._released:
                return
            self._released = True
            entry = _ENTRIES.get(self._key)
            if entry is None:
                return
            if self._device_key is not None:
                slot = entry.slots.get(self._device_key)
                if slot is not None:
                    slot.users = max(0, slot.users - 1)
                self._device_key = None
            entry.claim_count = max(0, entry.claim_count - 1)
            if entry.claim_count == 0:
                _free_device_slots(entry)


def claim_backbone(
    node: Any,
    checkpoint_path: str | None = None,
    enable_inst_interactivity: bool = True,
) -> BackboneClaim:
    """Claim the shared-backbone logical entry for a node.

    Registers a ``weakref.finalize`` hook on ``node`` (when given) so the claim
    is released even if the node is garbage-collected without ``cleanup()``.
    Raises ``OSError`` when an explicit ``checkpoint_path`` cannot be resolved.
    """
    key = (
        _vit_architecture_id(enable_inst_interactivity),
        _checkpoint_identity(checkpoint_path),
    )
    with _LOCK:
        entry = _ENTRIES.get(key)
        if entry is None:
            entry = _Entry(
                key=key,
                architecture_id=key[0],
                enable_inst_interactivity=enable_inst_interactivity,
                checkpoint_path=None if checkpoint_path is None else key[1][0],
            )
            _ENTRIES[key] = entry
        entry.claim_count += 1
    handle = BackboneClaim(key)
    if node is not None:
        weakref.finalize(node, handle.release)
    return handle


def claim_for_node(
    node: Any,
    checkpoint_path: str | None = None,
    compile_model: bool = False,
    enable_inst_interactivity: bool = True,
) -> BackboneClaim | None:
    """Node-facing claim helper: returns ``None`` (and logs) instead of raising.

    Returns ``None`` when sharing is bypassed or when the claim cannot be taken
    yet (for example the checkpoint file does not exist at construction time);
    the node then re-claims lazily at build time.
    """
    if sharing_bypassed(compile_model):
        return None
    try:
        return claim_backbone(
            node,
            checkpoint_path=checkpoint_path,
            enable_inst_interactivity=enable_inst_interactivity,
        )
    except Exception as exc:
        logger.warning("SAM3 shared-backbone claim deferred to first build: {}", exc)
        return None


def _load_raw_checkpoint(checkpoint_path: str) -> dict[str, torch.Tensor]:
    """Read the raw SAM3 checkpoint onto CPU (monkeypatch point for tests)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    return ckpt


def _construct_backbone_module(enable_inst_interactivity: bool) -> nn.Module:
    """Construct an unloaded SAM3VLBackbone matching both SAM3 model families."""
    import pkg_resources

    from sam3 import model_builder
    from sam3.model.vl_combiner import SAM3VLBackbone

    bpe_path = pkg_resources.resource_filename("sam3", "assets/bpe_simple_vocab_16e6.txt.gz")
    visual = model_builder._create_vision_backbone(
        enable_inst_interactivity=enable_inst_interactivity
    )
    text = model_builder._create_text_encoder(bpe_path)
    return SAM3VLBackbone(scalp=1, visual=visual, text=text)


def _pin_shared_module(module: nn.Module) -> nn.Module:
    """Pin a shared module to eval mode and to its current device placement.

    Swaps the instance onto a dynamically created subclass whose ``train`` and
    ``_apply`` overrides survive recursion from containing models: a claimant's
    ``model.train()`` or ``model.to()/.half()`` reaches children through those
    two methods, and both become no-ops here (``train(True)`` logs a warning).
    """
    cls = type(module)
    pinned = _PINNED_CLASSES.get(cls)
    if pinned is None:

        def train(self: nn.Module, mode: bool = True) -> nn.Module:
            """Keep the shared backbone in eval mode; ``train(True)`` is refused."""
            if mode:
                logger.warning("shared ViT backbone is pinned to eval mode; train(True) ignored")
                return self
            return nn.Module.train(self, False)

        def _apply(self: nn.Module, fn: Any, recurse: bool = True) -> nn.Module:
            """Ignore device and dtype conversions coming from containing models."""
            del fn, recurse
            return self

        pinned = type(f"_Pinned{cls.__name__}", (cls,), {"train": train, "_apply": _apply})
        _PINNED_CLASSES[cls] = pinned
    module.__class__ = pinned
    return module


def _resolve_entry_checkpoint(entry: _Entry) -> str:
    """Return the entry's on-disk checkpoint path, resolving it through core once.

    Without an explicit ``checkpoint_path`` the weights come from cuvis-ai-core's
    registry (the ``cubert-gmbh/sam3`` mirror): the cached file when present, a
    download when online, or ``ModelWeightsMissingError`` naming the provisioning
    command in the offline child.
    """
    if entry.checkpoint_path is None:
        from cuvis_ai_core.data.model_weights import ModelWeights

        entry.checkpoint_path = str(ModelWeights.resolve("sam3"))
    return entry.checkpoint_path


def _ensure_state_dicts(entry: _Entry) -> None:
    """Load and split the raw checkpoint into backbone/heads CPU state dicts once.

    Both splits are retained for the lifetime of the entry: ``heads_sd`` makes
    head-only model rebuilds cheap, ``backbone_sd`` lets a freed backbone
    re-warm on any device without another multi-GB checkpoint read.
    """
    if entry.backbone_sd is not None and entry.heads_sd is not None:
        return
    checkpoint_path = _resolve_entry_checkpoint(entry)
    raw = _load_raw_checkpoint(checkpoint_path)
    entry.loader_calls += 1
    backbone_sd: dict[str, torch.Tensor] = {}
    heads_sd: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        if key.startswith(_BACKBONE_PREFIX):
            backbone_sd[key[len(_BACKBONE_PREFIX) :]] = value
        else:
            heads_sd[key] = value
    entry.backbone_sd = backbone_sd
    entry.heads_sd = heads_sd


def _free_device_slots(entry: _Entry) -> None:
    """Drop every device-resident backbone instance of an entry (keeps state dicts)."""
    if not entry.slots:
        return
    devices = list(entry.slots)
    entry.slots.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info(
        "freed shared ViT backbone instance(s) ({}, devices={})",
        entry.architecture_id,
        devices,
    )


def _acquire_backbone(entry: _Entry, dev_key: str) -> nn.Module:
    """Return the entry's backbone for a device, building and publishing on miss.

    Caller must hold the registry lock. The built instance is loaded strictly
    from the backbone state-dict split, moved to the target device, frozen, and
    pinned before publication.
    """
    slot = entry.slots.get(dev_key)
    if slot is not None:
        entry.reuse_count += 1
        logger.info("reusing resident ViT backbone ({}, {})", entry.architecture_id, dev_key)
        return slot.backbone
    _ensure_state_dicts(entry)
    backbone = _construct_backbone_module(entry.enable_inst_interactivity)
    backbone.load_state_dict(entry.backbone_sd, strict=True)
    backbone.to(torch.device(dev_key))
    backbone.eval()
    backbone.requires_grad_(False)
    _pin_shared_module(backbone)
    entry.slots[dev_key] = _DeviceSlot(backbone=backbone)
    entry.build_count += 1
    logger.info("built shared ViT backbone ({}, {})", entry.architecture_id, dev_key)
    return backbone


def _mark_device_use(entry: _Entry, claim: BackboneClaim, dev_key: str) -> None:
    """Track which device a claimant uses; free an old-device instance it abandons."""
    previous = claim._device_key
    if previous == dev_key:
        return
    slot = entry.slots.get(dev_key)
    if slot is not None:
        slot.users += 1
    claim._device_key = dev_key
    if previous is None:
        return
    old_slot = entry.slots.get(previous)
    if old_slot is None:
        return
    old_slot.users = max(0, old_slot.users - 1)
    if old_slot.users == 0:
        del entry.slots[previous]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(
            "dropped shared ViT backbone instance ({}, {})",
            entry.architecture_id,
            previous,
        )


def _acquire_entry_backbone(
    checkpoint_path: str | None,
    enable_inst_interactivity: bool,
    dev_key: str,
    claim: BackboneClaim | None,
) -> tuple[_Entry, nn.Module, dict[str, torch.Tensor]]:
    """Get-or-build under the registry lock with publish-on-success semantics.

    A failed first build leaves the registry unchanged (freshly created entries
    are removed, freshly loaded state dicts are dropped) and the error
    propagates; the next call retries from scratch.
    """
    key = (
        _vit_architecture_id(enable_inst_interactivity),
        _checkpoint_identity(checkpoint_path),
    )
    with _LOCK:
        entry = _ENTRIES.get(key)
        if entry is None:
            entry = _Entry(
                key=key,
                architecture_id=key[0],
                enable_inst_interactivity=enable_inst_interactivity,
                checkpoint_path=None if checkpoint_path is None else key[1][0],
            )
            _ENTRIES[key] = entry
        had_state_dicts = entry.backbone_sd is not None and entry.heads_sd is not None
        try:
            backbone = _acquire_backbone(entry, dev_key)
        except BaseException:
            if not entry.slots and not had_state_dicts:
                entry.backbone_sd = None
                entry.heads_sd = None
            if entry.claim_count == 0 and not entry.slots and entry.backbone_sd is None:
                _ENTRIES.pop(key, None)
            raise
        if claim is not None and not claim.released and claim._key == key:
            _mark_device_use(entry, claim, dev_key)
        assert entry.heads_sd is not None
        return entry, backbone, entry.heads_sd


def _resolve_device(device: str | torch.device | None) -> str:
    """Default the target device like the upstream builders do."""
    if device is not None:
        return str(device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_image_model_shared(
    *,
    checkpoint_path: str | None = None,
    device: str | torch.device | None = None,
    enable_inst_interactivity: bool = True,
    compile_model: bool = False,
    claim: BackboneClaim | None = None,
) -> Any:
    """Build a SAM3 image model that reuses the process-wide shared backbone.

    Falls back to a fully standalone ``build_sam3_image_model`` call (no
    registry interaction) when sharing is bypassed: compilation requested or
    the bypass environment variable set.
    """
    from sam3 import model_builder

    resolved_device = _resolve_device(device)
    if sharing_bypassed(compile_model):
        build_kwargs: dict[str, Any] = {
            "device": resolved_device,
            "enable_inst_interactivity": enable_inst_interactivity,
        }
        if checkpoint_path:
            build_kwargs["checkpoint_path"] = checkpoint_path
        if compile_model:
            build_kwargs["compile"] = True
        return model_builder.build_sam3_image_model(**build_kwargs)

    dev_key = device_key(resolved_device)
    _entry, backbone, heads_sd = _acquire_entry_backbone(
        checkpoint_path, enable_inst_interactivity, dev_key, claim
    )
    model = model_builder.build_sam3_image_model(
        device=resolved_device,
        eval_mode=True,
        checkpoint_path=None,
        load_from_HF=False,
        enable_inst_interactivity=enable_inst_interactivity,
        backbone=backbone,
        state_dict=heads_sd,
    )
    # Guarantee head placement for device specs the upstream builder does not
    # move to (for example "cuda:1"); the pinned backbone is unaffected.
    model.to(torch.device(dev_key))
    return model


def build_video_model_shared(
    *,
    checkpoint_path: str | None = None,
    device: str | torch.device | None = None,
    compile_model: bool = False,
    claim: BackboneClaim | None = None,
) -> Any:
    """Build a SAM3 video model that reuses the process-wide shared backbone.

    The video builder always constructs the instance-interactive backbone
    variant, so the shared entry is keyed accordingly. Falls back to a fully
    standalone ``build_sam3_video_model`` call when sharing is bypassed.
    """
    from sam3 import model_builder

    resolved_device = _resolve_device(device)
    if sharing_bypassed(compile_model):
        build_kwargs: dict[str, Any] = {"device": resolved_device}
        if checkpoint_path:
            build_kwargs["checkpoint_path"] = checkpoint_path
        if compile_model:
            build_kwargs["compile"] = True
        return model_builder.build_sam3_video_model(**build_kwargs)

    dev_key = device_key(resolved_device)
    _entry, backbone, heads_sd = _acquire_entry_backbone(checkpoint_path, True, dev_key, claim)
    model = model_builder.build_sam3_video_model(
        checkpoint_path=None,
        load_from_HF=False,
        device=resolved_device,
        backbone=backbone,
        state_dict=heads_sd,
    )
    model.to(torch.device(dev_key))
    return model


def release_shared_backbone() -> None:
    """Clear the whole registry: every backbone instance and cached state dict.

    Test and diagnostic helper only — nodes must never call this; they release
    their own claim instead. Outstanding claims become no-ops on release.
    """
    with _LOCK:
        for entry in _ENTRIES.values():
            entry.slots.clear()
        _ENTRIES.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _module_resident_bytes(module: nn.Module) -> int:
    """Sum the storage bytes of a module's parameters and buffers."""
    total = 0
    seen: set[int] = set()
    for tensor in list(module.parameters()) + list(module.buffers()):
        marker = id(tensor)
        if marker in seen:
            continue
        seen.add(marker)
        total += tensor.numel() * tensor.element_size()
    return total


def shared_backbone_info() -> dict[str, Any]:
    """Snapshot registry state: claims, build/reuse counters, resident bytes.

    Returns a dict with ``bypass_env`` (whether the bypass environment variable
    is currently set) and ``entries``, one item per logical entry with its
    architecture id, checkpoint, claim/build/reuse/loader counters, state-dict
    cache flags, and per-device ``{resident_bytes, users}``.
    """
    with _LOCK:
        entries = []
        for entry in _ENTRIES.values():
            entries.append(
                {
                    "architecture_id": entry.architecture_id,
                    "checkpoint": entry.checkpoint_path or "hf:sam3",
                    "claim_count": entry.claim_count,
                    "build_count": entry.build_count,
                    "reuse_count": entry.reuse_count,
                    "loader_calls": entry.loader_calls,
                    "heads_sd_cached": entry.heads_sd is not None,
                    "backbone_sd_cached": entry.backbone_sd is not None,
                    "devices": {
                        dev: {
                            "resident_bytes": _module_resident_bytes(slot.backbone),
                            "users": slot.users,
                        }
                        for dev, slot in entry.slots.items()
                    },
                }
            )
        return {"bypass_env": sharing_bypassed(), "entries": entries}


__all__ = [
    "BackboneClaim",
    "build_image_model_shared",
    "build_video_model_shared",
    "claim_backbone",
    "claim_for_node",
    "device_key",
    "release_shared_backbone",
    "resolve_apply_target_device",
    "shared_backbone_info",
    "sharing_bypassed",
]
