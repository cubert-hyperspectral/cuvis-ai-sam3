"""Regression test: the plugin and REST API import without decord installed.

decord is declared with a platform marker (linux x86_64 / windows amd64 only), so on
other platforms (e.g. aarch64 child environments) it is simply absent. The platform
marker only proves wheel selection; this test proves no import path in the shipped
packages reaches decord at import time. It runs the import walk in a fresh
subprocess so no previously-imported module can mask an eager decord import.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

_IMPORT_WALK_SCRIPT = textwrap.dedent(
    """
    import importlib
    import pkgutil
    import sys

    class DecordBlocker:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "decord" or fullname.startswith("decord."):
                raise ImportError("decord import blocked: must not be required")
            return None

    sys.meta_path.insert(0, DecordBlocker())

    failures = []
    for package_name in ("cuvis_ai_sam3", "rest_api"):
        package = importlib.import_module(package_name)
        for module_info in pkgutil.walk_packages(package.__path__, package_name + "."):
            if module_info.name.endswith(".__main__"):
                continue
            try:
                importlib.import_module(module_info.name)
            except ImportError as error:
                failures.append(f"{module_info.name}: {error}")

    # Upstream loader modules used by the tracking predictor and the REST video
    # adapter: decord lives behind function-local imports there, so importing the
    # modules themselves must succeed.
    for module_name in ("sam3.model.io_utils", "sam3.model.utils.sam2_utils"):
        try:
            importlib.import_module(module_name)
        except ImportError as error:
            failures.append(f"{module_name}: {error}")

    if failures:
        print("modules requiring decord at import time:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        sys.exit(1)
    print("OK")
    """
)


def test_imports_survive_missing_decord() -> None:
    """Every shipped module imports with decord blocked from a fresh interpreter."""
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_WALK_SCRIPT],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"import walk failed with decord blocked:\n{result.stderr}\n{result.stdout}"
    )
    assert "OK" in result.stdout
