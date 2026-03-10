#!/usr/bin/env bash
cd git/shape-completion || true
./scripts/run.sh gen_eval -cn "$@" test.metrics="[chamfer,fpd]" test.batch_size=512
./scripts/run.sh gen_eval -cn "$@" test.metrics="[chamfer,fpd]" files.mesh=model.obj data.train_ds="[shapenet_v1]" pointcloud.from_mesh=True test.batch_size=512
