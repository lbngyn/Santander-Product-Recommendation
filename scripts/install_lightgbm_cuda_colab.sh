#!/usr/bin/env bash
# Build the CUDA implementation of LightGBM for a Colab Linux GPU runtime.
set -euo pipefail

if ! command -v nvcc >/dev/null 2>&1; then
  echo "CUDA compiler not found. In Colab, select a GPU runtime before running this script." >&2
  exit 1
fi

apt-get update -qq
apt-get install -y -qq build-essential cmake
python -m pip uninstall -y lightgbm
python -m pip install --no-cache-dir --no-binary lightgbm \
  --config-settings=cmake.define.USE_CUDA=ON \
  --config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=native \
  "lightgbm>=4.7.0"
