#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$REPO_ROOT"

tmp_files=()
cleanup() {
    for f in "${tmp_files[@]}"; do
        rm -f "$f"
    done
}
trap cleanup EXIT

static_failed=0

ruff_log="$(mktemp)"
tmp_files+=("$ruff_log")
echo "==> ruff (full repo, non-blocking)"
if uv run ruff check . >"$ruff_log" 2>&1; then
    echo "ruff full-repo: pass"
else
    static_failed=1
    echo "ruff full-repo: FAIL (non-blocking). Showing first 40 lines:"
    sed -n '1,40p' "$ruff_log"
fi

pyright_log="$(mktemp)"
tmp_files+=("$pyright_log")
echo "==> pyright (full repo, non-blocking)"
if uv run pyright >"$pyright_log" 2>&1; then
    echo "pyright full-repo: pass"
else
    static_failed=1
    echo "pyright full-repo: FAIL (non-blocking). Showing first 40 lines:"
    sed -n '1,40p' "$pyright_log"
fi

echo "==> Nightly pytest (full suite except quarantined renderer tests)"
uv run pytest

if [ "$static_failed" -ne 0 ]; then
    echo "==> NOTE: Non-blocking static checks reported issues (nightly remains green if pytest passes)."
fi
