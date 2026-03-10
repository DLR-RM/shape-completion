#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$REPO_ROOT"

echo "==> ruff (top-level)"
uv run ruff check .

echo "==> pyright (top-level)"
uv run pyright

echo "==> pytest + coverage (top-level; headline coverage is informational)"
uv run pytest \
  --cov=libs \
  --cov=utils \
  --cov=dataset \
  --cov=models \
  --cov=train \
  --cov=eval \
  --cov=inference \
  --cov=visualize \
  --cov=process \
  --cov-fail-under=0 \
  --cov-report=term

"$SCRIPT_DIR/check_source_coverage.sh"
