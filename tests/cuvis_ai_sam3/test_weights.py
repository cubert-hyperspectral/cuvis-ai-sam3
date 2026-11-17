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
    assert [entry.name for entry in WEIGHTS] == ["sam3", "sam3.1"]
    entry = WEIGHTS[0]
    assert isinstance(entry, PluginWeightEntry)
    assert entry.repo_id == "cubert-gmbh/sam3"
    assert entry.filename == "sam3.pt"
    assert len(entry.revision) == 40 and len(entry.sha256) == 64
    assert entry.size_bytes > 3_000_000_000
    assert [aux.path for aux in entry.aux_files] == ["config.json"]
    assert entry.selected_by == "weight_name" and entry.default is True
    assert entry.explicit_path_hparams == ["checkpoint_path"]
    assert entry.license == "SAM License" and entry.license_file == "LICENSE"
    assert entry.used_for and entry.summary and entry.description


def test_declares_the_sam31_checkpoint_as_a_row_a_node_selects() -> None:
    entry = WEIGHTS[1]
    assert isinstance(entry, PluginWeightEntry)
    assert entry.name == "sam3.1"
    assert entry.repo_id == "cubert-gmbh/sam3.1"
    assert entry.filename == "sam3.1_multiplex.pt"
    assert entry.revision == "8f2b686066130b1bc20e94f94d0b2ac69aa7f362"
    assert entry.sha256 == "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"
    assert entry.size_bytes == 3_502_755_717
    assert [aux.path for aux in entry.aux_files] == ["config.json"]
    assert entry.selected_by == "weight_name" and entry.default is False
    assert entry.explicit_path_hparams == ["checkpoint_path"]
    assert entry.license == "SAM License" and entry.license_file == "LICENSE"
    assert entry.used_for and entry.summary and entry.description


def test_weight_name_picks_one_checkpoint_per_node_and_sam3_is_the_default() -> None:
    """Both rows sit in the ``weight_name`` group and ``sam3`` is its default row.

    A consumer gives a node that does not set ``weight_name`` the default row and a node that
    sets it the row it names. The SAM 3 nodes have no such hparam, so they ask for ``sam3``
    alone; a ``SAM31TextPropagation`` that names ``sam3.1`` asks for ``sam3.1`` alone. A row
    without a selector would be asked of every node of the plugin, the SAM 3.1 node included.
    """
    assert [entry.name for entry in WEIGHTS if entry.selected_by is None] == []
    assert {entry.selected_by for entry in WEIGHTS} == {"weight_name"}
    assert [entry.name for entry in WEIGHTS if entry.default] == ["sam3"]
    assert ModelWeights.get("sam3").plugin_default is True
    assert ModelWeights.get("sam3.1").plugin_default is False


def test_only_the_sam31_node_takes_weight_name() -> None:
    """The group default is right for every node that cannot name a row itself.

    If a SAM 3 node gained ``weight_name``, a pipeline could point it at the SAM 3.1 file, and
    if the SAM 3.1 node lost it, a consumer could no longer tell which checkpoint it loads.
    """
    takes_it = sorted(
        cls.__name__ for cls in _node_classes() if "weight_name" in _constructor_params(cls)
    )
    assert takes_it == ["SAM31TextPropagation"]


def test_register_called_at_import() -> None:
    assert PLUGIN_NAME == "sam3"
    for entry in WEIGHTS:
        row = ModelWeights.get(entry.name)
        assert row.plugin == PLUGIN_NAME
        assert row.source == "plugin"
        assert row.entry == entry
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
        "assert len(mod.WEIGHTS) == 2\n"
        "for heavy in ('torch', 'cuvis_ai_core', 'sam3', 'cuvis_ai_sam3'):\n"
        "    assert heavy not in sys.modules, heavy\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=180, check=False
    )
    assert result.returncode == 0, result.stderr
