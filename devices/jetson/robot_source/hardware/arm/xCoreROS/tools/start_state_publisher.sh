#!/usr/bin/env bash
# Compatibility alias for the arm interfaces and head RGB-D entry point.
set -e

workspace_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$workspace_dir/tools/start_upperlimb.sh" "$@"
