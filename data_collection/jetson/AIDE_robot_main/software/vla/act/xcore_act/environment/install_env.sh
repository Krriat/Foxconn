#!/usr/bin/env bash
set -euo pipefail

ACT_PYTHON=/home/niic/miniconda3/envs/act/bin/python
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ACT_ENV_ROOT="$(cd -- "$(dirname -- "${ACT_PYTHON}")/.." && pwd)"
CACHE_DIR=/tmp/pip-act-cache
ENV_SITE=/home/niic/miniconda3/envs/act/lib/python3.10/site-packages
JETSON_SITE=/home/niic/.local/lib/python3.10/site-packages
TORCHVISION_EGG=/home/niic/.local/lib/python3.10/site-packages/torchvision-0.20.0-py3.10-linux-aarch64.egg
INSTALL_PYTHONPATH="${ENV_SITE}:${JETSON_SITE}:${TORCHVISION_EGG}"

mkdir -p "${ACT_ENV_ROOT}/etc/conda/activate.d" "${ACT_ENV_ROOT}/etc/conda/deactivate.d"
ln -sfn "${SCRIPT_DIR}/conda_activate.sh" \
  "${ACT_ENV_ROOT}/etc/conda/activate.d/xcore_act.sh"
ln -sfn "${SCRIPT_DIR}/conda_deactivate.sh" \
  "${ACT_ENV_ROOT}/etc/conda/deactivate.d/xcore_act.sh"

PYTHONNOUSERSITE=1 PYTHONPATH="${INSTALL_PYTHONPATH}" \
  "${ACT_PYTHON}" -m pip install \
  --cache-dir "${CACHE_DIR}" \
  -r "${SCRIPT_DIR}/requirements-act-jetson.txt"

# lerobot-record imports Rerun even when display_data=false. Its metadata asks
# for NumPy 2, but its runtime works with 1.26; NVIDIA's Jetson torch 2.5 needs
# the NumPy 1 ABI.
PYTHONNOUSERSITE=1 PYTHONPATH="${INSTALL_PYTHONPATH}" \
  "${ACT_PYTHON}" -m pip install \
  --cache-dir "${CACHE_DIR}" --no-deps rerun-sdk==0.26.2
PYTHONNOUSERSITE=1 PYTHONPATH="${INSTALL_PYTHONPATH}" \
  "${ACT_PYTHON}" -m pip install \
  --cache-dir "${CACHE_DIR}" --no-deps lerobot==0.4.3
PYTHONNOUSERSITE=1 PYTHONPATH="${INSTALL_PYTHONPATH}" \
  "${ACT_PYTHON}" -m pip install \
  --no-deps --editable "${PROJECT_ROOT}"
