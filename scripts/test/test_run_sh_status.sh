#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT
mkdir -p "$tmp_dir/bin" "$tmp_dir/scripts"

printf '#!/usr/bin/env bash\nreturn 0\n' > "$tmp_dir/scripts/activate_env.sh"
printf '#!/usr/bin/env bash\nexit 0\n' > "$tmp_dir/bin/nvidia-smi"
printf '#!/usr/bin/env bash\n"$@"\n' > "$tmp_dir/bin/srun"
chmod +x "$tmp_dir/bin/nvidia-smi" "$tmp_dir/bin/srun"

set +e
PATH="$tmp_dir/bin:$PATH" \
SLURM_SUBMIT_DIR="$tmp_dir" \
SLURM_JOB_NAME="status-test" \
"$repo_root/scripts/run.sh" --raw bash -c 'exit 7' >/dev/null 2>&1
status=$?
set -e

if [ "$status" -ne 7 ]; then
    printf 'Expected run.sh to return 7, got %s\n' "$status" >&2
    exit 1
fi
