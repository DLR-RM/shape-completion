#!/usr/bin/env bash
cd git/shape-completion || true
./scripts/run.sh generate -cn "$@"
./scripts/run.sh mesh_eval -cn "$@"
./scripts/run.sh mesh_eval -cn "$@" files.{points.test=null,mesh=model.obj} data.train_ds="[shapenet_v1]" pointcloud.from_mesh=True