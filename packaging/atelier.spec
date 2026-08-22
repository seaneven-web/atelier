# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Atelier desktop app — one platform-aware, onedir, windowed recipe.

    macOS    -> dist/Atelier.app            (BUNDLE, icon.icns)
    Windows  -> dist/Atelier/Atelier.exe    (icon.ico)
    Linux    -> dist/Atelier/Atelier        (COLLECT only)

Build via packaging/build_macos.sh | build_windows.ps1 | build_linux.sh, or directly:
    python -m PyInstaller --noconfirm --distpath dist --workpath packaging/build packaging/atelier.spec

Bundled: atelier.py, atelier_neural.py, atelier_claude.py (importable), web/ (the UI), and the
package data of torch / torchvision / diffusers / transformers / huggingface_hub / safetensors /
accelerate / pillow_heif (collect_all — they ship config files and dynamic libs PyInstaller's
static analysis misses). The pretrained weights are NOT bundled: the app downloads them once into
its sandbox folder.
"""
import os
import re
import sys

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

REPO_ROOT = os.path.dirname(SPECPATH)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
IS_MAC = sys.platform == "darwin"
IS_WIN = sys.platform == "win32"

def _version():
    with open(os.path.join(REPO_ROOT, "atelier_app.py"), encoding="utf-8") as fh:
        m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', fh.read(), re.M)
    return m.group(1) if m else "0.0.0"
VERSION = _version()
print("atelier.spec: version %s, platform %s" % (VERSION, sys.platform))

datas, binaries, hiddenimports = [], [], []
for pkg in ("torch", "torchvision", "diffusers", "transformers", "huggingface_hub", "safetensors", "accelerate",
            "tokenizers", "regex", "pillow_heif", "certifi", "tqdm", "filelock", "packaging", "yaml", "numpy", "PIL"):
    try:
        d, b, h = collect_all(pkg)
        datas += d; binaries += b; hiddenimports += h
    except Exception as e:  # optional packages (anthropic, pywebview glue) are allowed to be absent
        print("atelier.spec: skip %s (%s)" % (pkg, e))
for pkg in ("anthropic", "webview"):
    try:
        datas += collect_data_files(pkg); hiddenimports += collect_submodules(pkg)
    except Exception as e:
        print("atelier.spec: optional %s not bundled (%s)" % (pkg, e))
hiddenimports += ["atelier", "atelier_neural", "atelier_claude", "pillow_heif", "PIL.Image", "PIL.ImageOps", "PIL.ImageDraw"]

# the UI
web_dir = os.path.join(REPO_ROOT, "web")
for fn in os.listdir(web_dir):
    if not fn.startswith("."):
        datas.append((os.path.join(web_dir, fn), "web"))

block_cipher = None
a = Analysis(
    [os.path.join(REPO_ROOT, "atelier_app.py")],
    pathex=[REPO_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "scipy", "pandas", "IPython", "notebook", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

icon = None
if IS_MAC and os.path.exists(os.path.join(SPECPATH, "icon.icns")): icon = os.path.join(SPECPATH, "icon.icns")
if IS_WIN and os.path.exists(os.path.join(SPECPATH, "icon.ico")): icon = os.path.join(SPECPATH, "icon.ico")

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="Atelier",
    debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
    console=False,                         # windowed: the app opens its own window / the browser
    icon=icon,
)
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, strip=False, upx=False, name="Atelier")

if IS_MAC:
    app = BUNDLE(
        coll, name="Atelier.app", icon=icon, bundle_identifier="com.seaneven.atelier",
        info_plist={
            "CFBundleName": "Atelier", "CFBundleDisplayName": "Atelier", "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION, "NSHighResolutionCapable": True, "LSMinimumSystemVersion": "12.0",
            "NSHumanReadableCopyright": "Your art stays on this computer.",
        },
    )
