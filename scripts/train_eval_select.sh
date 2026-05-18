#!/usr/bin/env bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT"
for category in "02691156" "02958343" "03001627" "04090263" "04379243"; do
    ./scripts/run.sh gen_eval -cn "$@" test.split=train test.metrics="[clip_fid,kid,prdc]" data.categories="[${category}]"
    ./scripts/run.sh gen_eval -cn "$@" test.split=train test.metrics="[clip_fid,kid,prdc]" data.train_ds="[shapenet_v1]" data.categories="[${category}]"
done
