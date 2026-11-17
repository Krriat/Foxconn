#!/usr/bin/env bash
# Compatibility alias for the unified state and SDK entry point.
set -e

workspace_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$workspace_dir/tools/start_upperlimb.sh" "$@"
