"""The plugin's weight declarations and their registration with core's registry.

No network and no checkpoint: the declarations are data, the registry is in-process,
and the one resolution test stubs ``ModelWeights.resolve``.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

from cuvis_ai_core.data.model_weights import ModelWeights
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.plugin import PluginWeightEntry

import cuvis_ai_sam3
import cuvis_ai_sam3.node as node_pkg
import cuvis_ai_sam3.weights as weights_mod
from cuvis_ai_sam3 import shared_backbone
from cuvis_ai_sam3.weights import PLUGIN_NAME, WEIGHTS


def _node_classes() -> list[type[Node]]:
    return [
        cls
        for cls in vars(node_pkg).values()
        if isinstance(cls, type) and issubclass(cls, Node) and cls is not Node
    ]


def _constructor_params(cls: type) -> set[str]:
    return set(inspect.signature(cls.__init__).parameters) - {"self", "args", "kwargs"}


def test_declares_the_sam3_checkpoint_with_full_pins() -> None:
    assert [entry.name for entry in WEIGHTS] == ["sam3"]
    (entry,) = WEIGHTS
    assert isinstance(entry, PluginWeightEntry)
    assert entry.repo_id == "cubert-gmbh/sam3"
    assert entry.filename == "sam3.pt"
    assert len(entry.revision) == 40 and len(entry.sha256) == 64
    assert entry.size_bytes > 3_000_000_000
    assert [aux.path for aux in entry.aux_files] == ["config.json"]
    assert entry.selected_by is None and entry.default is False
    assert entry.explicit_path_hparams == ["checkpoint_path"]
    assert entry.license == "SAM License" and entry.license_file == "LICENSE"
    assert entry.used_for and entry.summary and entry.description


def test_register_called_at_import() -> None:
    row = ModelWeights.get("sam3")
    assert PLUGIN_NAME == "sam3"
    assert row.plugin == PLUGIN_NAME
    assert row.source == "plugin"
    assert row.entry == WEIGHTS[0]
    assert cuvis_ai_sam3.WEIGHTS is WEIGHTS


def test_registry_name_is_what_the_shared_backbone_resolves(monkeypatch, tmp_path) -> None:
    seen: list[str] = []

    def fake_resolve(cls, name, **kwargs):
        seen.append(name)
        return tmp_path / "sam3.pt"

    monkeypatch.setattr(ModelWeights, "resolve", classmethod(fake_resolve))
    entry = shared_backbone._Entry(
        key=("sam3", ()),
        architecture_id="sam3",
        enable_inst_interactivity=False,
        checkpoint_path=None,
    )
    shared_backbone._resolve_entry_checkpoint(entry)
    assert seen == [WEIGHTS[0].name]


def test_explicit_path_hparams_are_node_constructor_params() -> None:
    """What emit_metadata validates: every declared hparam exists on at least one node."""
    classes = _node_classes()
    assert classes
    for entry in WEIGHTS:
        hparams = [*entry.explicit_path_hparams]
        if entry.selected_by is not None:
            hparams.append(entry.selected_by)
        for hparam in hparams:
            assert any(hparam in _constructor_params(cls) for cls in classes), hparam


def test_weights_module_is_side_effect_free() -> None:
    """The module declares only: loading it alone must not import torch, core or the plugin."""
    path = Path(weights_mod.__file__)
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('weights_probe', {str(path)!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "assert len(mod.WEIGHTS) == 1\n"
        "for heavy in ('torch', 'cuvis_ai_core', 'sam3', 'cuvis_ai_sam3'):\n"
        "    assert heavy not in sys.modules, heavy\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=180, check=False
    )
    assert result.returncode == 0, result.stderr
