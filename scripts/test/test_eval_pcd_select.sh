#!/usr/bin/env bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT"
for category in "02691156" "02958343" "03001627" "04090263" "04379243"; do
    ./scripts/run.sh gen_eval -cn "$@" test.metrics="[chamfer,fpd]" data.categories="[${category}]" test.batch_size=1024
    ./scripts/run.sh gen_eval -cn "$@" test.metrics="[chamfer,fpd]" files.mesh=model.obj data.train_ds="[shapenet_v1]" pointcloud.from_mesh=True data.categories="[${category}]" test.batch_size=1024
done
