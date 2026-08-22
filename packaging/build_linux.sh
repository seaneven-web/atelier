#!/bin/bash
# Build Atelier on Linux: icons -> PyInstaller -> dist/Atelier/ -> dist/Atelier-linux-<arch>.tar.gz -> smoke test.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(dirname "$HERE")"; DIST="$ROOT/dist"
PY="${PYTHON:-$ROOT/.venv/bin/python}"; [ -x "$PY" ] || PY="$(command -v python3)"
"$PY" -c "import PyInstaller" || { echo "pip install -r requirements.txt -r requirements-desktop.txt first" >&2; exit 1; }
case "$(uname -m)" in aarch64|arm64) ARCH=arm64 ;; *) ARCH=x86_64 ;; esac
VERSION="$(sed -n 's/^__version__ *= *["'"'"']\([^"'"'"']*\)["'"'"'].*/\1/p' "$ROOT/atelier_app.py" | head -1)"
echo "==> Atelier ${VERSION:-0.0.0} — Linux $ARCH — $PY"
[ -f "$HERE/linux/icon-256.png" ] || "$PY" "$HERE/make_icon.py" --png
rm -rf "$DIST/Atelier"; mkdir -p "$DIST"
( cd "$ROOT" && "$PY" -m PyInstaller --noconfirm --distpath "$DIST" --workpath "$HERE/build" "$HERE/atelier.spec" )
cp "$HERE/linux/atelier.desktop" "$HERE/linux/install.sh" "$HERE/linux/icon-256.png" "$HERE/linux/icon-512.png" "$DIST/Atelier/" 2>/dev/null || true
chmod +x "$DIST/Atelier/install.sh" 2>/dev/null || true
( cd "$DIST" && rm -f "Atelier-linux-$ARCH.tar.gz" && tar -czf "Atelier-linux-$ARCH.tar.gz" Atelier )
[ -n "${SKIP_SMOKE:-}" ] || "$PY" "$HERE/smoke_test.py" "$DIST/Atelier/Atelier" --frozen
ls -la "$DIST" | grep -E "Atelier-linux" || true
