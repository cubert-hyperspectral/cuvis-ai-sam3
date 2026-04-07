# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for SAM3 REST API server."""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all

block_cipher = None

SPEC_DIR = Path(SPECPATH)
PROJECT_ROOT = SPEC_DIR.parent
LAUNCHER = SPEC_DIR / "launcher.py"
CUVIS_AI_CORE_ROOT = PROJECT_ROOT.parent.parent / "cuvis-ai-core" / "cuvis-ai-core-sam3"

rest_datas, rest_binaries, rest_hiddenimports = collect_all("rest_api")
sam3_datas, sam3_binaries, sam3_hiddenimports = collect_all("sam3")
package_datas, package_binaries, package_hiddenimports = collect_all("cuvis_ai_sam3")
torch_datas, torch_binaries, torch_hiddenimports = collect_all("torch")
triton_datas, triton_binaries, triton_hiddenimports = collect_all("triton")
timm_datas, timm_binaries, timm_hiddenimports = collect_all("timm")
hf_datas, hf_binaries, hf_hiddenimports = collect_all("huggingface_hub")
einops_datas, einops_binaries, einops_hiddenimports = collect_all("einops")

datas = (
    rest_datas
    + sam3_datas
    + package_datas
    + torch_datas
    + triton_datas
    + timm_datas
    + hf_datas
    + einops_datas
)
binaries = (
    rest_binaries
    + sam3_binaries
    + package_binaries
    + torch_binaries
    + triton_binaries
    + timm_binaries
    + hf_binaries
    + einops_binaries
)

hiddenimports = (
    rest_hiddenimports
    + sam3_hiddenimports
    + package_hiddenimports
    + torch_hiddenimports
    + triton_hiddenimports
    + timm_hiddenimports
    + hf_hiddenimports
    + einops_hiddenimports
    + [
        "fastapi",
        "uvicorn",
        "uvicorn.logging",
        "uvicorn.loops.auto",
        "uvicorn.protocols.http.auto",
        "sse_starlette",
        "pydantic",
        "pydantic_core",
        "loguru",
        "rest_api.routes.health",
        "rest_api.routes.sessions",
        "rest_api.routes.prompts",
        "rest_api.routes.propagation",
        "rest_api.routes.objects",
    ]
)

excludes = [
    "PySide6",
    "PyQt5",
    "PyQt6",
    "pytest",
    "pytest_cov",
    "pytest_asyncio",
    "mypy",
    "ruff",
    "gradio",
    "jupyter",
    "tkinter",
]

a = Analysis(
    [str(LAUNCHER)],
    pathex=[str(PROJECT_ROOT), str(CUVIS_AI_CORE_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="sam3-rest-api",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="sam3-server",
)

