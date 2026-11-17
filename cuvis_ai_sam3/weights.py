"""Weight declarations of the sam3 plugin.

Side-effect free on purpose: this module only declares. ``cuvis_ai_sam3/__init__``
registers the tuple with cuvis-ai-core's ``ModelWeights`` at import, and cuvis-ai's
``emit_metadata`` projects it into the plugin manifest's ``weights:`` block, so
CuvisNEXT and the installer know what to provision without importing the plugin.
The pins come from ``tools/mirror_weights.py`` in cuvis-ai-core (the mirrors are
``cubert-gmbh/sam3``, a byte-identical copy of ``facebook/sam3`` at 3c879f39, and
``cubert-gmbh/sam3.1``, the same for ``facebook/sam3.1`` at daa63191).

A consumer provisions every row without ``selected_by`` for every node of the plugin.
``sam3`` is such a row. ``sam3.1`` serves one node, so it is selected by that node's
``weight_name`` and the group has no default row: a pipeline needs it only where its
node names it.
"""

from __future__ import annotations

from cuvis_ai_schemas.plugin import AuxFile, PluginWeightEntry

PLUGIN_NAME = "sam3"
"""The manifest name of this plugin (what pipelines list under ``plugins:``)."""

WEIGHTS: tuple[PluginWeightEntry, ...] = (
    PluginWeightEntry(
        name="sam3",
        display_name="SAM3",
        summary="Magic wand, propagation, text prompts",
        used_for=["Point expansion", "Propagation", "Text prompts", "Segment everything"],
        repo_id="cubert-gmbh/sam3",
        filename="sam3.pt",
        revision="6d25af14a085ff9d3e1342c35bae7c87de4811f4",
        sha256="9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e",
        size_bytes=3_450_062_241,
        aux_files=[
            AuxFile(
                path="config.json",
                size_bytes=25_843,
                sha256="4616385e4b21f2e5e22c875b65679185cbccfa95de42542b9166f7dc3d57160f",
            )
        ],
        license="SAM License",
        license_file="LICENSE",
        explicit_path_hparams=["checkpoint_path"],
        description=(
            "SAM3 checkpoint for the magic-wand point expansion, text / box / mask "
            "propagation and segment-everything nodes; every sam3 node needs it unless "
            "checkpoint_path points at a local file (mirror of facebook/sam3 at 3c879f39)."
        ),
    ),
    PluginWeightEntry(
        name="sam3.1",
        display_name="SAM 3.1",
        summary="Text prompts with many objects (Object Multiplex)",
        used_for=["Text prompts"],
        repo_id="cubert-gmbh/sam3.1",
        filename="sam3.1_multiplex.pt",
        revision="8f2b686066130b1bc20e94f94d0b2ac69aa7f362",
        sha256="0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6",
        size_bytes=3_502_755_717,
        aux_files=[
            AuxFile(
                path="config.json",
                size_bytes=25_843,
                sha256="4616385e4b21f2e5e22c875b65679185cbccfa95de42542b9166f7dc3d57160f",
            )
        ],
        license="SAM License",
        license_file="LICENSE",
        selected_by="weight_name",
        explicit_path_hparams=["checkpoint_path"],
        description=(
            "SAM 3.1 Object Multiplex checkpoint for SAM31TextPropagation, the text-prompt "
            "tracker that costs about the same for one object as for sixteen; needed where a "
            "pipeline sets weight_name to sam3.1, unless checkpoint_path points at a local "
            "file (mirror of facebook/sam3.1 at daa63191)."
        ),
    ),
)
"""Every weight the sam3 nodes load, keyed by the name ``download-model`` knows."""
