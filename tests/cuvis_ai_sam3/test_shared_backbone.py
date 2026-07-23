"""Tests for the process-wide shared SAM3 vision-language backbone registry.

Unit tests use tiny fake modules and state dicts (no downloads, no real model
construction); the single ``gpu``-marked test exercises the real checkpoint.
"""

from __future__ import annotations

import gc
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from loguru import logger
from torch import nn

import cuvis_ai_sam3.shared_backbone as sb
import sam3.model_builder as mb

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _TinyBackbone(nn.Module):
    """Stands in for SAM3VLBackbone; key set mirrors the stripped backbone sd."""

    def __init__(self) -> None:
        super().__init__()
        self.vision_backbone = nn.Linear(2, 2, bias=False)
        self.language_backbone = nn.Linear(2, 2, bias=False)


class _FakeImageModel(nn.Module):
    """Minimal image-model stand-in exposing the injected backbone."""

    def __init__(self, backbone: nn.Module | None) -> None:
        super().__init__()
        self.backbone = backbone


class _FakeVideoModel(nn.Module):
    """Minimal video-model stand-in exposing detector.backbone."""

    def __init__(self, backbone: nn.Module | None) -> None:
        super().__init__()
        self.detector = nn.Module()
        self.detector.backbone = backbone


class _SeamImageModel(nn.Module):
    """Stand-in with exactly the attributes ``_load_checkpoint`` touches."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(2, 2, bias=False)
        self.transformer = nn.Linear(2, 2, bias=False)
        self.inst_interactive_predictor = None


class _Owner:
    """Plain claim owner used in place of a full Node."""


def _fake_raw_sd() -> dict[str, torch.Tensor]:
    return {
        "detector.backbone.vision_backbone.weight": torch.full((2, 2), 2.0),
        "detector.backbone.language_backbone.weight": torch.full((2, 2), 3.0),
        "detector.transformer.weight": torch.full((2, 2), 4.0),
        "tracker.head.weight": torch.full((2, 2), 5.0),
    }


@pytest.fixture(autouse=True)
def _clean_registry(monkeypatch: pytest.MonkeyPatch):
    """Isolate each test from prior registry state and the bypass env var."""
    monkeypatch.delenv("CUVIS_SAM3_NO_BACKBONE_SHARING", raising=False)
    sb.release_shared_backbone()
    yield
    sb.release_shared_backbone()


@pytest.fixture
def fake_stack(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Route the registry's loader/factory and the model builders to tiny fakes."""
    calls = {"loader": 0, "construct": 0, "image": [], "video": []}
    ckpt = tmp_path / "sam3.pt"
    ckpt.write_bytes(b"\x00" * 16)

    def loader(path: str) -> dict[str, torch.Tensor]:
        calls["loader"] += 1
        return _fake_raw_sd()

    def construct(enable_inst_interactivity: bool) -> nn.Module:
        calls["construct"] += 1
        return _TinyBackbone()

    def image_builder(**kwargs) -> nn.Module:
        calls["image"].append(kwargs)
        return _FakeImageModel(kwargs.get("backbone"))

    def video_builder(**kwargs) -> nn.Module:
        calls["video"].append(kwargs)
        return _FakeVideoModel(kwargs.get("backbone"))

    monkeypatch.setattr(sb, "_load_raw_checkpoint", loader)
    monkeypatch.setattr(sb, "_construct_backbone_module", construct)
    monkeypatch.setattr(mb, "build_sam3_image_model", image_builder)
    monkeypatch.setattr(mb, "build_sam3_video_model", video_builder)
    return SimpleNamespace(calls=calls, ckpt=str(ckpt))


def _single_entry() -> dict:
    entries = sb.shared_backbone_info()["entries"]
    assert len(entries) == 1
    return entries[0]


# ---------------------------------------------------------------------------
# Registry basics
# ---------------------------------------------------------------------------


class TestRegistryBuilds:
    def test_first_build_publishes(self, fake_stack) -> None:
        model = sb.build_image_model_shared(checkpoint_path=fake_stack.ckpt, device="cpu")
        entry = _single_entry()
        assert entry["build_count"] == 1
        assert entry["loader_calls"] == 1
        assert entry["heads_sd_cached"] and entry["backbone_sd_cached"]
        assert "cpu" in entry["devices"]
        assert isinstance(model.backbone, nn.Module)
        # weights actually reached the shared instance
        assert torch.equal(model.backbone.vision_backbone.weight, torch.full((2, 2), 2.0))

    def test_mid_build_failure_leaves_registry_empty_then_retries(
        self, fake_stack, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attempts = {"n": 0}

        def flaky(enable_inst_interactivity: bool) -> nn.Module:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("boom")
            return _TinyBackbone()

        monkeypatch.setattr(sb, "_construct_backbone_module", flaky)
        with pytest.raises(RuntimeError, match="boom"):
            sb.build_image_model_shared(checkpoint_path=fake_stack.ckpt, device="cpu")
        assert sb.shared_backbone_info()["entries"] == []

        model = sb.build_image_model_shared(checkpoint_path=fake_stack.ckpt, device="cpu")
        assert isinstance(model.backbone, nn.Module)
        assert _single_entry()["build_count"] == 1

    def test_image_then_video_share_one_backbone_and_one_load(self, fake_stack) -> None:
        image = sb.build_image_model_shared(checkpoint_path=fake_stack.ckpt, device="cpu")
        video = sb.build_video_model_shared(checkpoint_path=fake_stack.ckpt, device="cpu")

        assert video.detector.backbone is image.backbone
        assert fake_stack.calls["loader"] == 1
        assert fake_stack.calls["construct"] == 1

        image_kwargs = fake_stack.calls["image"][0]
        video_kwargs = fake_stack.calls["video"][0]
        assert image_kwargs["backbone"] is image.backbone
        assert video_kwargs["backbone"] is image.backbone
        assert image_kwargs["load_from_HF"] is False
        assert video_kwargs["load_from_HF"] is False
        assert set(image_kwargs["state_dict"]) == {
            "detector.transformer.weight",
            "tracker.head.weight",
        }
        assert video_kwargs["state_dict"] is image_kwargs["state_dict"]
        assert _single_entry()["reuse_count"] == 1

    def test_concurrent_get_or_build_builds_once(
        self, fake_stack, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def slow_construct(enable_inst_interactivity: bool) -> nn.Module:
            time.sleep(0.05)
            fake_stack.calls["construct"] += 1
            return _TinyBackbone()

        monkeypatch.setattr(sb, "_construct_backbone_module", slow_construct)
        results: list[nn.Module] = []

        def worker() -> None:
            model = sb.build_image_model_shared(checkpoint_path=fake_stack.ckpt, device="cpu")
            results.append(model.backbone)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert fake_stack.calls["construct"] == 1
        assert len(results) == 2
        assert results[0] is results[1]
        assert _single_entry()["build_count"] == 1


# ---------------------------------------------------------------------------
# Claim lifetime
# ---------------------------------------------------------------------------


class TestClaims:
    def test_claim_counts_across_pipeline_switch_backbone_never_freed(self, fake_stack) -> None:
        old_node = _Owner()
        old_claim = sb.claim_backbone(old_node, checkpoint_path=fake_stack.ckpt)
        sb.build_image_model_shared(checkpoint_path=fake_stack.ckpt, device="cpu", claim=old_claim)
        assert _single_entry()["claim_count"] == 1

        # Core constructs the new pipeline's nodes before old cleanup runs.
        new_node = _Owner()
        new_claim = sb.claim_backbone(new_node, checkpoint_path=fake_stack.ckpt)
        assert _single_entry()["claim_count"] == 2

        old_claim.release()
        entry = _single_entry()
        assert entry["claim_count"] == 1
        assert "cpu" in entry["devices"], "backbone must survive a SAM3-to-SAM3 switch"

        old_claim.release()  # idempotent
        assert _single_entry()["claim_count"] == 1
        new_claim.release()

    def test_zero_claims_frees_instances_and_rewarms_without_loader(self, fake_stack) -> None:
        node = _Owner()
        claim = sb.claim_backbone(node, checkpoint_path=fake_stack.ckpt)
        sb.build_image_model_shared(checkpoint_path=fake_stack.ckpt, device="cpu", claim=claim)
        assert fake_stack.calls["loader"] == 1

        claim.release()
        entry = _single_entry()
        assert entry["claim_count"] == 0
        assert entry["devices"] == {}, "device instances freed at zero claims"
        assert entry["heads_sd_cached"] and entry["backbone_sd_cached"]

        sb.build_image_model_shared(checkpoint_path=fake_stack.ckpt, device="cpu")
        entry = _single_entry()
        assert fake_stack.calls["loader"] == 1, "re-warm must not re-read the checkpoint"
        assert entry["build_count"] == 2
        assert "cpu" in entry["devices"]

    def test_gc_of_claimed_node_releases_claim(self, fake_stack) -> None:
        node = _Owner()
        sb.claim_backbone(node, checkpoint_path=fake_stack.ckpt)
        assert _single_entry()["claim_count"] == 1

        del node
        gc.collect()
        assert _single_entry()["claim_count"] == 0

    def test_move_to_new_device_rewarms_and_drops_abandoned_instance(self, fake_stack) -> None:
        node = _Owner()
        claim = sb.claim_backbone(node, checkpoint_path=fake_stack.ckpt)
        first = sb.build_image_model_shared(
            checkpoint_path=fake_stack.ckpt, device="cpu", claim=claim
        )
        second = sb.build_image_model_shared(
            checkpoint_path=fake_stack.ckpt, device="meta", claim=claim
        )

        assert second.backbone is not first.backbone
        entry = _single_entry()
        assert entry["build_count"] == 2
        assert set(entry["devices"]) == {"meta"}, "sole claimant moved; cpu instance dropped"
        claim.release()


# ---------------------------------------------------------------------------
# Eval/device pin
# ---------------------------------------------------------------------------


class TestPinnedBackbone:
    def test_containing_train_cannot_flip_shared_backbone(self, fake_stack) -> None:
        model = sb.build_image_model_shared(checkpoint_path=fake_stack.ckpt, device="cpu")
        backbone = model.backbone
        assert backbone.training is False

        messages: list[str] = []
        sink_id = logger.add(lambda message: messages.append(str(message)), level="WARNING")
        try:
            model.train()
        finally:
            logger.remove(sink_id)

        assert backbone.training is False
        assert any("pinned to eval" in message for message in messages)
        assert all(not p.requires_grad for p in backbone.parameters())

    def test_containing_to_cannot_move_shared_backbone(self, fake_stack) -> None:
        model = sb.build_image_model_shared(checkpoint_path=fake_stack.ckpt, device="cpu")
        backbone = model.backbone
        model.to("meta")
        assert next(backbone.parameters()).device.type == "cpu"


# ---------------------------------------------------------------------------
# Bypasses
# ---------------------------------------------------------------------------


class TestBypasses:
    def test_env_bypass_skips_registry(self, fake_stack, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CUVIS_SAM3_NO_BACKBONE_SHARING", "1")
        sb.build_image_model_shared(checkpoint_path=fake_stack.ckpt, device="cpu")

        assert fake_stack.calls["loader"] == 0
        assert sb.shared_backbone_info()["entries"] == []
        kwargs = fake_stack.calls["image"][0]
        assert "backbone" not in kwargs
        assert "state_dict" not in kwargs
        assert kwargs["checkpoint_path"] == fake_stack.ckpt

    def test_compile_bypass_skips_registry(self, fake_stack) -> None:
        sb.build_video_model_shared(
            checkpoint_path=fake_stack.ckpt, device="cpu", compile_model=True
        )

        assert fake_stack.calls["loader"] == 0
        assert sb.shared_backbone_info()["entries"] == []
        kwargs = fake_stack.calls["video"][0]
        assert kwargs["compile"] is True
        assert "backbone" not in kwargs
        assert "state_dict" not in kwargs


# ---------------------------------------------------------------------------
# _apply device probing
# ---------------------------------------------------------------------------


class TestApplyDeviceProbe:
    def test_resolve_apply_target_device(self) -> None:
        meta_target = sb.resolve_apply_target_device(lambda t: t.to("meta"))
        assert meta_target is not None and meta_target.type == "meta"

        assert sb.resolve_apply_target_device(lambda t: t.half()) is None

        cpu_target = sb.resolve_apply_target_device(lambda t: t.to("cpu"))
        if torch.cuda.is_available():
            assert cpu_target is not None and cpu_target.type == "cpu"
        else:
            assert cpu_target is None


# ---------------------------------------------------------------------------
# model_builder seam
# ---------------------------------------------------------------------------


class TestLoadCheckpointSeam:
    def test_state_dict_path_does_no_file_io_and_never_mutates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _no_file_io(*args, **kwargs):
            raise AssertionError("state_dict path must not open files")

        monkeypatch.setattr(mb.g_pathmgr, "open", _no_file_io)

        model = _SeamImageModel()
        sd = {
            "detector.transformer.weight": torch.ones(2, 2),
            "detector.backbone.weight": torch.full((2, 2), 3.0),
        }
        before = dict(sd)
        mb._load_checkpoint(model, None, state_dict=sd)

        assert torch.equal(model.transformer.weight, torch.ones(2, 2))
        assert torch.equal(model.backbone.weight, torch.full((2, 2), 3.0))
        assert sd == before

    def test_skip_prefixes_filters_keys_and_silences_missing_report(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        sd = {
            "detector.transformer.weight": torch.ones(2, 2),
            "detector.backbone.weight": torch.full((2, 2), 7.0),
        }

        model = _SeamImageModel()
        untouched = model.backbone.weight.clone()
        mb._load_checkpoint(model, None, state_dict=sd, skip_prefixes=("backbone.",))
        assert torch.equal(model.backbone.weight, untouched), "skip prefix must filter keys"
        assert capsys.readouterr().out == "", "backbone missing keys must be silenced"

        # Negative control: without skip_prefixes the missing-key report prints.
        model2 = _SeamImageModel()
        mb._load_checkpoint(
            model2, None, state_dict={"detector.transformer.weight": torch.ones(2, 2)}
        )
        assert "backbone.weight" in capsys.readouterr().out

    def test_video_builder_raises_on_unexpected_missing_keys(self) -> None:
        with pytest.raises(RuntimeError, match="non-backbone keys missing"):
            mb.build_sam3_video_model(
                checkpoint_path=None,
                load_from_HF=False,
                device="cpu",
                backbone=_TinyBackbone(),
                state_dict={"detector.bogus.weight": torch.zeros(1)},
            )


# ---------------------------------------------------------------------------
# Node wiring
# ---------------------------------------------------------------------------


class TestNodeWiring:
    def test_image_node_ensure_model_routes_through_shared(
        self, fake_stack, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cuvis_ai_sam3.node.sam3_point_expansion import SAM3PointExpansion

        recorded: dict = {}

        def fake_build(**kwargs) -> SimpleNamespace:
            recorded.update(kwargs)
            return SimpleNamespace()

        monkeypatch.setattr(sb, "build_image_model_shared", fake_build)
        monkeypatch.setattr(
            "sam3.model.sam3_image_processor.Sam3Processor",
            lambda model, device, confidence_threshold: SimpleNamespace(model=model),
        )

        node = SAM3PointExpansion(name="wiring_img", checkpoint_path=fake_stack.ckpt)
        assert node._backbone_claim is not None
        node._ensure_model()

        assert recorded["claim"] is node._backbone_claim
        assert recorded["checkpoint_path"] == fake_stack.ckpt
        assert recorded["enable_inst_interactivity"] is True
        assert recorded["device"] == node._resolved_device

    def test_streaming_node_ensure_model_routes_through_shared(
        self, fake_stack, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cuvis_ai_sam3.node.sam3_streaming_propagation import SAM3TextPropagation

        recorded: dict = {}

        def fake_build(**kwargs) -> MagicMock:
            recorded.update(kwargs)
            model = MagicMock()
            param = nn.Parameter(torch.zeros(1))
            model.parameters.side_effect = lambda: iter([param])
            return model

        monkeypatch.setattr(sb, "build_video_model_shared", fake_build)

        node = SAM3TextPropagation(name="wiring_vid", checkpoint_path=fake_stack.ckpt)
        assert node._backbone_claim is not None
        node._ensure_model()

        assert recorded["claim"] is node._backbone_claim
        assert recorded["checkpoint_path"] == fake_stack.ckpt
        assert node._model.hotstart_delay == 0

    def test_node_cleanup_releases_claim_but_keeps_registry(self, fake_stack) -> None:
        from cuvis_ai_sam3.node.sam3_point_expansion import SAM3PointExpansion

        node = SAM3PointExpansion(name="wiring_cleanup", checkpoint_path=fake_stack.ckpt)
        assert _single_entry()["claim_count"] == 1

        node.cleanup()
        entry = _single_entry()  # entry still listed: cleanup never clears the registry
        assert entry["claim_count"] == 0
        assert node._backbone_claim is None

    def test_pipeline_to_cuda_triggers_eager_build(
        self, fake_stack, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cuvis_ai_sam3.node.sam3_point_expansion import SAM3PointExpansion

        built = {"n": 0}

        def fake_build(**kwargs) -> SimpleNamespace:
            built["n"] += 1
            return SimpleNamespace()

        monkeypatch.setattr(sb, "build_image_model_shared", fake_build)
        monkeypatch.setattr(
            "sam3.model.sam3_image_processor.Sam3Processor",
            lambda model, device, confidence_threshold: SimpleNamespace(),
        )

        node = SAM3PointExpansion(name="wiring_eager", checkpoint_path=fake_stack.ckpt)
        node._on_pipeline_device(torch.device("cuda", 0))

        assert built["n"] == 1
        assert node._resolved_device == "cuda:0"

    def test_apply_records_device_and_stays_lazy_off_cuda(self, fake_stack) -> None:
        from cuvis_ai_sam3.node.sam3_point_expansion import SAM3PointExpansion

        node = SAM3PointExpansion(name="wiring_apply", checkpoint_path=fake_stack.ckpt)
        nn.ModuleList([node]).to("meta")

        assert node._resolved_device == "meta"
        assert node._model is None


# ---------------------------------------------------------------------------
# GPU integration (real checkpoint, run with -m gpu)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_gpu_shared_backbone_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build image + video models sharing one CUDA-resident real backbone."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    load_calls = {"n": 0}
    real_load = torch.load

    def counting_load(*args, **kwargs):
        load_calls["n"] += 1
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", counting_load)

    image = sb.build_image_model_shared(device="cuda")
    video = sb.build_video_model_shared(device="cuda")
    try:
        assert video.detector.backbone is image.backbone
        assert load_calls["n"] == 1, "raw checkpoint must be torch.loaded exactly once"

        info = sb.shared_backbone_info()
        assert len(info["entries"]) == 1
        entry = info["entries"][0]
        assert entry["build_count"] == 1
        assert entry["reuse_count"] == 1
        assert len(entry["devices"]) == 1
        (device_stats,) = entry["devices"].values()
        assert device_stats["resident_bytes"] > 2_000_000_000
    finally:
        del image, video
        sb.release_shared_backbone()
        torch.cuda.empty_cache()
