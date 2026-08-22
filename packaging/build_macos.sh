#!/bin/bash
# Build Atelier on macOS: icon -> PyInstaller -> dist/Atelier.app -> dist/Atelier-mac-<arch>.zip (+ .dmg) -> smoke test.
# Needs requirements.txt + requirements-desktop.txt installed in .venv (or $PYTHON). Unsigned (ad-hoc) — see packaging/README.md.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(dirname "$HERE")"; DIST="$ROOT/dist"
PY="${PYTHON:-$ROOT/.venv/bin/python}"; [ -x "$PY" ] || PY="$(command -v python3)"
"$PY" -c "import PyInstaller" || { echo "pip install -r requirements.txt -r requirements-desktop.txt first" >&2; exit 1; }
case "$(uname -m)" in arm64|aarch64) ARCH=arm64 ;; *) ARCH=x86_64 ;; esac
VERSION="$(sed -n 's/^__version__ *= *["'"'"']\([^"'"'"']*\)["'"'"'].*/\1/p' "$ROOT/atelier_app.py" | head -1)"
echo "==> Atelier ${VERSION:-0.0.0} — macOS $ARCH — $PY"
[ -f "$HERE/icon.icns" ] || "$PY" "$HERE/make_icon.py"
rm -rf "$DIST/Atelier" "$DIST/Atelier.app"; mkdir -p "$DIST"
( cd "$ROOT" && "$PY" -m PyInstaller --noconfirm --distpath "$DIST" --workpath "$HERE/build" "$HERE/atelier.spec" )
codesign --force --deep --sign - "$DIST/Atelier.app" 2>/dev/null || true
( cd "$DIST" && rm -f "Atelier-mac-$ARCH.zip" && ditto -c -k --keepParent Atelier.app "Atelier-mac-$ARCH.zip" )
if [ -z "${SKIP_DMG:-}" ]; then
  rm -f "$DIST/Atelier-mac-$ARCH.dmg"; STAGE="$(mktemp -d)"; cp -R "$DIST/Atelier.app" "$STAGE/"; ln -s /Applications "$STAGE/Applications"
  hdiutil create -quiet -volname "Atelier" -srcfolder "$STAGE" -ov -format UDZO "$DIST/Atelier-mac-$ARCH.dmg"; rm -rf "$STAGE"
fi
[ -n "${SKIP_SMOKE:-}" ] || "$PY" "$HERE/smoke_test.py" "$DIST/Atelier.app/Contents/MacOS/Atelier" --frozen
ls -la "$DIST" | grep -E "Atelier-mac" || true
