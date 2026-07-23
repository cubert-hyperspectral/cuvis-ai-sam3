"""cuvis_ai_sam3: SAM3 wrapper and cuvis.ai plugin package.

This package lives inside the forked SAM3 repository and provides:

- Access to the upstream SAM3 model builders
  (see :mod:`sam3.model_builder`).
- cuvis.ai-compatible Nodes for video object tracking
  (see :mod:`cuvis_ai_sam3.node`).
"""

from importlib.metadata import PackageNotFoundError, version
from typing import cast

try:
    __version__ = version("cuvis-ai-sam3")
except PackageNotFoundError:
    # Package is not installed, likely in development mode
    __version__ = "dev"

from cuvis_ai_sam3.shared_backbone import (  # noqa: F401
    build_image_model_shared,
    build_video_model_shared,
    claim_backbone,
    release_shared_backbone,
    shared_backbone_info,
)
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
    return cast(int, registry.auto_register_package(package_name))


__all__ = [
    "__version__",
    "build_image_model_shared",
    "build_sam3_image_model",
    "build_sam3_video_model",
    "build_sam3_video_predictor",
    "build_video_model_shared",
    "claim_backbone",
    "register_all_nodes",
    "release_shared_backbone",
    "shared_backbone_info",
]
