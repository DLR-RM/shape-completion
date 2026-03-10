#!/usr/bin/env bash
cd git/shape-completion || true
./scripts/run.sh evaluate -cn "$@"
./scripts/run.sh generate -cn "$@"
./scripts/run.sh gen_eval -cn "$@" test.metrics="[chamfer,fpd,clip_fid,kid,prdc]" test.batch_size=512
./scripts/run.sh gen_eval -cn "$@" test.metrics="[chamfer,fpd,clip_fid,kid,prdc]" files.mesh=model.obj data.train_ds="[shapenet_v1]" pointcloud.from_mesh=True test.batch_size=512
