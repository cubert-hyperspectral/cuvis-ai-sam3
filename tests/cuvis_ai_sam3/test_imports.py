"""Smoke tests: verify the cuvis_ai_sam3 package is importable."""

from __future__ import annotations


def test_package_importable() -> None:
    """The cuvis_ai_sam3 package can be imported."""
    import cuvis_ai_sam3

    assert hasattr(cuvis_ai_sam3, "__version__")


def test_version_string() -> None:
    """__version__ is a non-empty string."""
    from cuvis_ai_sam3 import __version__

    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_register_all_nodes_callable() -> None:
    """register_all_nodes is exposed and callable."""
    from cuvis_ai_sam3 import register_all_nodes

    assert callable(register_all_nodes)


def test_sam3_tracker_node_importable() -> None:
    """SAM3TrackerInference is exported from the node package."""
    from cuvis_ai_sam3.node import SAM3TrackerInference

    assert SAM3TrackerInference.__name__ == "SAM3TrackerInference"


def test_propagation_nodes_importable() -> None:
    """Specialized propagation nodes are exported from the node package."""
    from cuvis_ai_sam3.node import (
        SAM3BboxPropagation,
        SAM3MaskPropagation,
        SAM3PointPropagation,
        SAM3TextPropagation,
    )

    assert SAM3TextPropagation.__name__ == "SAM3TextPropagation"
    assert SAM3BboxPropagation.__name__ == "SAM3BboxPropagation"
    assert SAM3PointPropagation.__name__ == "SAM3PointPropagation"
    assert SAM3MaskPropagation.__name__ == "SAM3MaskPropagation"
