#!/usr/bin/env bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT"
for category in "02691156" "02958343" "03001627" "04090263" "04379243"; do
    #./scripts/run.sh generate -cn "$@" data.categories="[${category}]" test.batch_size=1 +sample=10
    #./scripts/run.sh mesh_eval -cn "$@" data.categories="[${category}]" test.merge=True
    ./scripts/run.sh mesh_eval -cn "$@" data.categories="[${category}]" test.merge=True files.{points.test=null,mesh=model.obj} data.train_ds="[shapenet_v1]" pointcloud.from_mesh=True
done
