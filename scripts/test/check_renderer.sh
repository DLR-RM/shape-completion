#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$REPO_ROOT"

echo "==> renderer preflight"
if ! uv run python -c "import importlib.util, sys; mods=('open3d','pyrender','torch','pytorch3d.renderer'); missing=[m for m in mods if importlib.util.find_spec(m) is None]; print('SKIP renderer lane: missing modules: ' + ', '.join(missing)) if missing else None; sys.exit(3 if missing else 0)"; then
    status=$?
    if [ "$status" -eq 3 ]; then
        exit 0
    fi
    exit "$status"
fi

if ! uv run python -c "import sys, torch; print('SKIP renderer lane: CUDA not available') if not torch.cuda.is_available() else None; sys.exit(3 if not torch.cuda.is_available() else 0)"; then
    status=$?
    if [ "$status" -eq 3 ]; then
        exit 0
    fi
    exit "$status"
fi

echo "==> pytest (renderer lane)"
SC_RUN_RENDERER_TESTS=1 uv run pytest visualize/tests/test_renderer.py -m renderer -q
