# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for SAM3 tray launcher."""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all

block_cipher = None

SPEC_DIR = Path(SPECPATH)
PROJECT_ROOT = SPEC_DIR.parent
LAUNCHER = SPEC_DIR / "tray_launcher.py"
ICON_FILE = SPEC_DIR / "app_icon.ico"

pystray_datas, pystray_binaries, pystray_hiddenimports = collect_all("pystray")
pil_datas, pil_binaries, pil_hiddenimports = collect_all("PIL")

datas = pystray_datas + pil_datas
binaries = pystray_binaries + pil_binaries
hiddenimports = pystray_hiddenimports + pil_hiddenimports + ["pystray._win32"]

excludes = [
    "torch",
    "triton",
    "timm",
    "pytest",
    "mypy",
    "ruff",
    "tkinter",
]

a = Analysis(
    [str(LAUNCHER)],
    pathex=[str(PROJECT_ROOT)],
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
    name="sam3-tray",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon=str(ICON_FILE) if ICON_FILE.exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="sam3-tray",
)

