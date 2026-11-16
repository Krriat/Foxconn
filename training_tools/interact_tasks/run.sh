#!/usr/bin/env bash
set -euo pipefail
INTERACT_TASK_WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TORCH_HOME="$INTERACT_TASK_WORKSPACE/.cache/torch" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec "$INTERACT_TASK_WORKSPACE/.venv-act/bin/python" -u "$INTERACT_TASK_WORKSPACE/training_tools/interact_tasks/train.py" "$@"
