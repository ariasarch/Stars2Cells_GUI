# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Stars2Cells.

Build (from the repo root):
    pyinstaller --noconfirm --clean packaging/stars2cells.spec

Produces:
    dist/Stars2Cells/            one-folder app (all platforms)
    dist/Stars2Cells.app         additionally, on macOS

One-folder (onedir) mode is deliberate: the pipeline spawns multiprocessing
workers, and onedir keeps worker start-up fast and avoids the temp-unpack
issues of onefile builds.
"""

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH).resolve().parent
ICON_DIR = ROOT / "packaging" / "icons"

APP_NAME = "Stars2Cells"
VERSION = os.environ.get("S2C_VERSION", "1.0.0")

# Data files the app reads at runtime. utilities/utils.py resolves
# S2C_logo.png relative to __file__'s grandparent, which maps to the
# bundle root (_internal/) in a frozen app — so ship it at '.'.
datas = [
    (str(ROOT / "S2C_logo.png"), "."),
    (str(ROOT / "base_data_requirements.txt"), "."),
    (str(ROOT / "exporting_data.txt"), "."),
    (str(ROOT / "LICENSE"), "."),
]

binaries = []
hiddenimports = []

# faiss is imported lazily inside functions, so PyInstaller's static
# analysis misses it. It is optional at runtime (sklearn KDTree fallback),
# so tolerate its absence at build time too.
try:
    faiss_datas, faiss_binaries, faiss_hidden = collect_all("faiss")
    datas += faiss_datas
    binaries += faiss_binaries
    hiddenimports += faiss_hidden
except Exception as exc:  # noqa: BLE001
    print(f"NOTE: faiss not bundled ({exc}); app will use sklearn fallback")

a = Analysis(
    [str(ROOT / "stars2cells.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={
        # Trim unused matplotlib backends; the app uses the Qt backend.
        "matplotlib": {"backends": ["QtAgg", "Qt5Agg", "Agg"]},
    },
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "PyQt6",
        "PySide2",
        "PySide6",
        "IPython",
        "jupyter",
        "notebook",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

if sys.platform == "win32":
    exe_icon = str(ICON_DIR / "S2C_logo.ico")
elif sys.platform == "darwin":
    exe_icon = str(ICON_DIR / "S2C_logo.icns")
else:
    exe_icon = None

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    strip=False,
    upx=False,
    console=False,
    icon=exe_icon if exe_icon and Path(exe_icon).exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name=APP_NAME,
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name=f"{APP_NAME}.app",
        icon=exe_icon if Path(exe_icon).exists() else None,
        bundle_identifier="com.neumaierlab.stars2cells",
        version=VERSION,
        info_plist={
            "CFBundleName": APP_NAME,
            "CFBundleDisplayName": APP_NAME,
            "CFBundleShortVersionString": VERSION,
            "NSHighResolutionCapable": True,
            "NSRequiresAquaSystemAppearance": False,
            "LSMinimumSystemVersion": "11.0",
        },
    )
