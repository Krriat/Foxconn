#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
asset_dir="${1:-$repo_dir/downloads}"
mkdir -p "$asset_dir"
gh release download snapshot-20261117-jetson-act-qwen --repo Krriat/Foxconn \
  --dir "$asset_dir" --pattern 'foxconn-*.tar.gz'
cd "$asset_dir"
sha256sum --check "$repo_dir/manifests/assets.sha256"
