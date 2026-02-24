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
