#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

EXTRA="full"
INSTALL_TIER1=true
INSTALL_TIER2=false

usage() {
    cat <<'EOF'
Bootstrap local/CI development environment for linting and tests.

Usage:
  ./scripts/bootstrap_dev.sh [options]

Options:
  --extra <name>      uv optional dependency extra to sync (default: full)
  --no-tier1-libs     skip `python libs/libmanager.py install`
  --with-tier2-libs   run `./scripts/compile_cuda_libs.sh` after tier-1 libs
  -h, --help          show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --extra)
            EXTRA="${2:-}"
            shift 2
            ;;
        --no-tier1-libs)
            INSTALL_TIER1=false
            shift
            ;;
        --with-tier2-libs)
            INSTALL_TIER2=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$EXTRA" ]]; then
    echo "--extra requires a non-empty value" >&2
    exit 1
fi

echo "==> Syncing Python dependencies (uv sync --inexact --extra $EXTRA)"
uv sync --inexact --extra "$EXTRA"

if [[ "$INSTALL_TIER1" == true ]]; then
    echo "==> Installing Tier-1 custom libs (libs/libmanager.py install)"
    uv run python libs/libmanager.py install
else
    echo "==> Skipping Tier-1 custom libs"
fi

if [[ "$INSTALL_TIER2" == true ]]; then
    echo "==> Installing Tier-2 CUDA libs (scripts/compile_cuda_libs.sh)"
    uv run bash ./scripts/compile_cuda_libs.sh
else
    echo "==> Skipping Tier-2 CUDA libs"
fi

echo "Bootstrap complete."
echo "Suggested check: uv run pytest -m \"not integration\""
