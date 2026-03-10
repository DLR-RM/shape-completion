#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$REPO_ROOT"

MIN_COVERAGE="${SOURCE_ONLY_COVERAGE_MIN:-35}"
OMIT_PATTERNS="${SOURCE_ONLY_COVERAGE_OMIT:-*/tests/*,*/setup.py,*/test_*.py}"

if [ ! -f ".coverage" ]; then
    echo "error: .coverage not found. Run pytest with --cov before checking source-only coverage." >&2
    exit 1
fi

echo "==> coverage (source-only)"
# Ignore missing-source errors from transient third-party temp modules outside the repo.
report_output="$(uv run coverage report --ignore-errors --fail-under=0 --omit="$OMIT_PATTERNS")"
printf '%s\n' "$report_output"

total_line="$(printf '%s\n' "$report_output" | awk '$1=="TOTAL" {print $2" "$3}')"
if [ -z "$total_line" ]; then
    echo "error: failed to parse TOTAL line from coverage report." >&2
    exit 1
fi

read -r total_stmts total_miss <<<"$total_line"
covered=$((total_stmts - total_miss))
exact_pct="$(
    uv run python - "$total_stmts" "$total_miss" <<'PY'
import sys

stmts = int(sys.argv[1])
miss = int(sys.argv[2])
print((stmts - miss) / stmts * 100)
PY
)"

printf 'Exact source-only coverage: %.4f%% (%d / %d)\n' "$exact_pct" "$covered" "$total_stmts"

uv run python - "$exact_pct" "$MIN_COVERAGE" <<'PY'
import sys

exact = float(sys.argv[1])
threshold = float(sys.argv[2])

if exact < threshold:
    print(
        f"error: source-only coverage {exact:.4f}% is below required {threshold:.4f}%",
        file=sys.stderr,
    )
    raise SystemExit(1)

print(f"source-only coverage gate: pass ({exact:.4f}% >= {threshold:.4f}%)")
PY
