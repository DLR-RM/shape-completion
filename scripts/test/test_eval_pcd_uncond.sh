#!/usr/bin/env bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT"
./scripts/run.sh gen_eval -cn "$@" test.metrics="[chamfer,fpd]" test.batch_size=512
./scripts/run.sh gen_eval -cn "$@" test.metrics="[chamfer,fpd]" files.mesh=model.obj data.train_ds="[shapenet_v1]" pointcloud.from_mesh=True test.batch_size=512
