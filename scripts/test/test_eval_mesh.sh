#!/usr/bin/env bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT"
./scripts/run.sh generate -cn "$@"
./scripts/run.sh mesh_eval -cn "$@"
./scripts/run.sh mesh_eval -cn "$@" files.{points.test=null,mesh=model.obj} data.train_ds="[shapenet_v1]" pointcloud.from_mesh=True
