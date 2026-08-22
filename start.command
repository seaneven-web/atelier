#!/bin/bash
# Double-click launcher (macOS). Creates the environment on first run, then opens Atelier.
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "First run — setting up (a few minutes)…"
  python3 -m venv .venv && .venv/bin/pip install -q --upgrade pip
  if [ "$(uname -m)" = "x86_64" ] && [ "$(uname)" = "Darwin" ]; then
    .venv/bin/pip install -q "torch==2.2.2" "torchvision==0.17.2" "numpy<2" pillow pillow-heif "diffusers==0.30.3" "transformers==4.44.2" "accelerate==0.34.2" "huggingface_hub<0.27" safetensors
  else
    .venv/bin/pip install -q torch torchvision numpy pillow pillow-heif diffusers transformers accelerate safetensors
  fi
fi
.venv/bin/python atelier.py "$@"
