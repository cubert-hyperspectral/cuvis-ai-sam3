"""The shared backbone resolves the SAM3 checkpoint through core's weight registry.

No network, no real checkpoint: ``ModelWeights.resolve`` is stubbed, and the upstream
``download_ckpt_from_hf`` is patched to fail so a regression back to the gated
``facebook/sam3`` download is caught.
"""

from __future__ import annotations

import pytest
from cuvis_ai_core.data.model_weights import ModelWeights, ModelWeightsMissingError

from cuvis_ai_sam3 import shared_backbone


def _entry(checkpoint_path: str | None = None) -> shared_backbone._Entry:
    return shared_backbone._Entry(
        key=("sam3", ()),
        architecture_id="sam3",
        enable_inst_interactivity=False,
        checkpoint_path=checkpoint_path,
    )


def _never_upstream(monkeypatch) -> None:
    import sam3.model_builder as model_builder

    def _fail(*args, **kwargs):
        pytest.fail("the upstream facebook/sam3 download must not be used")

    monkeypatch.setattr(model_builder, "download_ckpt_from_hf", _fail)


def test_resolves_through_core_registry_once(monkeypatch, tmp_path) -> None:
    _never_upstream(monkeypatch)
    calls: list[str] = []

    def fake_resolve(cls, name, **kwargs):
        calls.append(name)
        return tmp_path / "sam3.pt"

    monkeypatch.setattr(ModelWeights, "resolve", classmethod(fake_resolve))
    entry = _entry()

    assert shared_backbone._resolve_entry_checkpoint(entry) == str(tmp_path / "sam3.pt")
    assert shared_backbone._resolve_entry_checkpoint(entry) == str(tmp_path / "sam3.pt")
    assert calls == ["sam3"]  # cached on the entry after the first resolution


def test_explicit_checkpoint_path_wins(monkeypatch) -> None:
    _never_upstream(monkeypatch)

    def _fail(cls, *args, **kwargs):
        pytest.fail("resolve must not be called when checkpoint_path is given")

    monkeypatch.setattr(ModelWeights, "resolve", classmethod(_fail))

    assert shared_backbone._resolve_entry_checkpoint(_entry("/local/sam3.pt")) == "/local/sam3.pt"


def test_missing_weights_error_reaches_the_caller(monkeypatch) -> None:
    """Offline without provisioned weights: core's actionable error is not swallowed."""
    _never_upstream(monkeypatch)

    def _missing(cls, name, **kwargs):
        raise ModelWeightsMissingError(
            f"'{name}' is not in the model cache. Provision it with: "
            f"uv run download-model download {name}"
        )

    monkeypatch.setattr(ModelWeights, "resolve", classmethod(_missing))

    with pytest.raises(ModelWeightsMissingError, match="download-model download sam3"):
        shared_backbone._resolve_entry_checkpoint(_entry())
