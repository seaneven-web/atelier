#!/bin/bash
# Per-user install: registers the app in the menu (no root). Run from the unpacked Atelier/ folder.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p ~/.local/share/applications ~/.local/share/icons/hicolor/256x256/apps
cp "$HERE/icon-256.png" ~/.local/share/icons/hicolor/256x256/apps/atelier.png
sed "s|__EXEC__|$HERE/Atelier|; s|__ICON__|atelier|" "$HERE/atelier.desktop" > ~/.local/share/applications/atelier.desktop
echo "Atelier is in your application menu. (Uninstall: remove ~/.local/share/applications/atelier.desktop and this folder.)"
