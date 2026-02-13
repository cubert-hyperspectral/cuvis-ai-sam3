"""cuvis_ai_sam3: SAM3 wrapper and cuvis.ai plugin package.

This package lives inside the forked SAM3 repository and provides:

- Access to the upstream SAM3 model builders
  (see :mod:`sam3.model_builder`).
- cuvis.ai-compatible Nodes for video object tracking
  (see :mod:`cuvis_ai_sam3.node`).
"""

from sam3.model_builder import (  # noqa: F401
    build_sam3_image_model,
    build_sam3_video_model,
    build_sam3_video_predictor,
)


def register_all_nodes() -> int:
    """Register all cuvis_ai_sam3 nodes in the cuvis.ai NodeRegistry.

    Returns
    -------
    int
        The number of node classes that were registered.
    """
    package_name = "cuvis_ai_sam3.node"

    # Plugin workflows are instance-based in cuvis_ai_core.
    from cuvis_ai_core.utils.node_registry import NodeRegistry

    registry = NodeRegistry()
    return registry.auto_register_package(package_name)


__all__ = [
    "build_sam3_image_model",
    "build_sam3_video_model",
    "build_sam3_video_predictor",
    "register_all_nodes",
]
