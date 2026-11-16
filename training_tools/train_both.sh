#!/usr/bin/env bash
set -euo pipefail
ACT_TRAINING_TOOLS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$ACT_TRAINING_TOOLS/../.venv-act/bin/python" "$ACT_TRAINING_TOOLS/run_training.py" both "$@"
