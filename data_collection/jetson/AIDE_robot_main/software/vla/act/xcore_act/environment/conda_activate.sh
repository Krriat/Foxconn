#!/usr/bin/env bash

export _XCORE_ACT_OLD_PYTHONNOUSERSITE="${PYTHONNOUSERSITE-}"
export _XCORE_ACT_OLD_PYTHONPATH="${PYTHONPATH-}"
export PYTHONNOUSERSITE=1

_XCORE_ACT_ENV_SITE="${CONDA_PREFIX}/lib/python3.10/site-packages"
_XCORE_ACT_JETSON_SITE="/home/niic/.local/lib/python3.10/site-packages"
_XCORE_ACT_TORCHVISION="/home/niic/.local/lib/python3.10/site-packages/torchvision-0.20.0-py3.10-linux-aarch64.egg"
_XCORE_ACT_SDK_SITE="/home/niic/xcoresdk_python-v0.7.1.ar_6/Release/linux/arm"
_XCORE_ACT_REALSENSE_SITE="/usr/local/lib/python3.10/dist-packages"
export PYTHONPATH="${_XCORE_ACT_ENV_SITE}:${_XCORE_ACT_SDK_SITE}:${_XCORE_ACT_JETSON_SITE}:${_XCORE_ACT_TORCHVISION}:${_XCORE_ACT_REALSENSE_SITE}:${PYTHONPATH-}"

unset _XCORE_ACT_ENV_SITE _XCORE_ACT_JETSON_SITE _XCORE_ACT_TORCHVISION _XCORE_ACT_SDK_SITE _XCORE_ACT_REALSENSE_SITE
