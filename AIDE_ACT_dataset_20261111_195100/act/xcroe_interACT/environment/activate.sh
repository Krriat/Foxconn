#!/usr/bin/env bash

# Source this file; executing it cannot modify the caller's environment.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Run with: source ${BASH_SOURCE[0]}" >&2
  exit 2
fi

_XCORE_INTERACT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export XCORE_INTERACT_ROOT="${_XCORE_INTERACT_ROOT}"
export AIDE_ROOT="${AIDE_ROOT:-/home/niic/aide}"

if [[ -f "${AIDE_ROOT}/startup/ros_environment_debug.sh" ]]; then
  source "${AIDE_ROOT}/startup/ros_environment_debug.sh" || return 1
fi
source /opt/ros/humble/setup.bash || return 1
if [[ -f "${AIDE_ROOT}/hardware/end_effector/sts3215_ws/install/setup.bash" ]]; then
  source "${AIDE_ROOT}/hardware/end_effector/sts3215_ws/install/setup.bash" || return 1
fi
source /home/niic/miniconda3/etc/profile.d/conda.sh || return 1
conda activate act || return 1

case ":${PYTHONPATH-}:" in
  *":${_XCORE_INTERACT_ROOT}/src:"*) ;;
  *) export PYTHONPATH="${_XCORE_INTERACT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac

_XCORE_SDK_LIB="${AIDE_ROOT}/hardware/arm/xCoreSDK-v0.7.1.ar_6/lib/Linux/aarch64"
case ":${LD_LIBRARY_PATH-}:" in
  *":${_XCORE_SDK_LIB}:"*) ;;
  *) export LD_LIBRARY_PATH="${_XCORE_SDK_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
esac
export TOKENIZERS_PARALLELISM=false

unset _XCORE_INTERACT_ROOT _XCORE_SDK_LIB
echo "InterACT3D environment: root=${XCORE_INTERACT_ROOT} conda=${CONDA_PREFIX}"
python -c 'import platform, torch, lerobot; print(f"arch={platform.machine()} torch={torch.__version__} lerobot={lerobot.__version__} cuda={torch.cuda.is_available()}")' || return 1
