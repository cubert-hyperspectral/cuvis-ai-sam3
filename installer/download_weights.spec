# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for SAM3 weight downloader utility."""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all

block_cipher = None

SPEC_DIR = Path(SPECPATH)
PROJECT_ROOT = SPEC_DIR.parent
LAUNCHER = SPEC_DIR / "download_weights.py"

hf_datas, hf_binaries, hf_hiddenimports = collect_all("huggingface_hub")
tqdm_datas, tqdm_binaries, tqdm_hiddenimports = collect_all("tqdm")

a = Analysis(
    [str(LAUNCHER)],
    pathex=[str(PROJECT_ROOT)],
    binaries=hf_binaries + tqdm_binaries,
    datas=hf_datas + tqdm_datas,
    hiddenimports=hf_hiddenimports + tqdm_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "mypy", "ruff", "PySide6"],
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
    name="download-weights",
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
    name="download-weights",
)

