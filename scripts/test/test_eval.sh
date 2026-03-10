#!/usr/bin/env bash
cd git/shape-completion || true
./scripts/run.sh evaluate -cn "$@"
./scripts/run.sh generate -cn "$@"
./scripts/run.sh mesh_eval -cn "$@"
./scripts/run.sh mesh_eval -cn "$@" files.{points.test=null,mesh=model.obj} data.train_ds="[shapenet_v1]" pointcloud.from_mesh=True
./scripts/run.sh gen_eval -cn "$@" test.metrics="[chamfer,fpd,clip_fid,kid,prdc]" test.batch_size=512
./scripts/run.sh gen_eval -cn "$@" test.metrics="[chamfer,fpd,clip_fid,kid,prdc]" files.mesh=model.obj data.train_ds="[shapenet_v1]" pointcloud.from_mesh=True test.batch_size=512

for category in "02691156" "02958343" "03001627" "04090263" "04379243"; do
    ./scripts/run.sh gen_eval -cn "$@" test.metrics="[chamfer,fpd,clip_fid,kid,prdc]" data.categories="[${category}]" test.batch_size=1024
    ./scripts/run.sh gen_eval -cn "$@" test.metrics="[chamfer,fpd,clip_fid,kid,prdc]" files.mesh=model.obj data.train_ds="[shapenet_v1]" pointcloud.from_mesh=True data.categories="[${category}]" test.batch_size=1024
done