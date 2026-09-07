"""Weight declarations of the sam3 plugin.

Side-effect free on purpose: this module only declares. ``cuvis_ai_sam3/__init__``
registers the tuple with cuvis-ai-core's ``ModelWeights`` at import, and cuvis-ai's
``emit_metadata`` projects it into the plugin manifest's ``weights:`` block, so
CuvisNEXT and the installer know what to provision without importing the plugin.
The pins come from ``tools/mirror_weights.py`` in cuvis-ai-core (the mirror is
``cubert-gmbh/sam3``, a byte-identical copy of ``facebook/sam3`` at 3c879f39).
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
)
"""Every weight the sam3 nodes load, keyed by the name ``download-model`` knows."""
