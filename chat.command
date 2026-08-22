#!/bin/bash
# Double-click launcher (macOS) for the Claude-directed studio. Needs ANTHROPIC_API_KEY (or `ant auth login`).
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo "Run start.command once first (it builds the environment)."; read -r; exit 1; }
.venv/bin/pip install -q anthropic
.venv/bin/python atelier_claude.py "$@"
