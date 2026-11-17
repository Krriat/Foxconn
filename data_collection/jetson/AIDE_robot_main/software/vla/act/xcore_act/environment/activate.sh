#!/usr/bin/env bash

# This file must be sourced so activation persists in the caller. Do not use
# `set -e` here: it would leak errexit into the interactive terminal and make
# the shell close when a later recorder command reports an ordinary error.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Run this file with: source ${BASH_SOURCE[0]}" >&2
  exit 2
fi

_XCORE_ACT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export XCORE_ACT_ROOT="${_XCORE_ACT_ROOT}"
export AIDE_ROOT="${AIDE_ROOT:-/home/niic/aide}"

_XCORE_ROS_ENV="${AIDE_ROOT}/startup/ros_environment_debug.sh"
if [[ -f "${_XCORE_ROS_ENV}" ]]; then
  source "${_XCORE_ROS_ENV}" || return 1
fi
source /opt/ros/humble/setup.bash || return 1
_XCORE_STS_SETUP="${AIDE_ROOT}/hardware/end_effector/sts3215_ws/install/setup.bash"
if [[ -f "${_XCORE_STS_SETUP}" ]]; then
  source "${_XCORE_STS_SETUP}" || return 1
fi
source /home/niic/miniconda3/etc/profile.d/conda.sh || return 1
conda activate act || return 1

_XCORE_ACT_SRC="${_XCORE_ACT_ROOT}/src"
case ":${PYTHONPATH-}:" in
  *":${_XCORE_ACT_SRC}:"*) ;;
  *) export PYTHONPATH="${_XCORE_ACT_SRC}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac
unset _XCORE_ACT_ROOT _XCORE_ACT_SRC _XCORE_STS_SETUP _XCORE_ROS_ENV

# The conda activation hook installed with this project puts the act packages
# before the shared Jetson PyTorch site and adds the xCore binding.
export TOKENIZERS_PARALLELISM=false

echo "ACT environment: ${CONDA_PREFIX}"
python -c 'import platform, torch; print(f"arch={platform.machine()} torch={torch.__version__} cuda={torch.cuda.is_available()}")' || return 1
