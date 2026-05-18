# Reproduction recipes

This repository contains public, host-neutral recipes for the experiments behind the advertised papers and the instance-completion work. They are meant to catch broken public commands and config drift before a user spends hours on a run.

The recipes do not ship datasets or checkpoints. They render commands against the public Hydra configs and fail early when a checkpoint, mesh, output path, or BOP split has to be supplied.

## Usage

List available recipes:

```bash
python3 scripts/reproduce_experiment.py --list
python3 scripts/reproduce_experiment.py --list --suite 3dv2026
```

Render a command without running it:

```bash
python3 scripts/reproduce_experiment.py --recipe iros2023:mugs-main-train
python3 scripts/reproduce_experiment.py --recipe 3dv2026:ar-uncond-eval \
    --set weights=cvpr_2025/pcd_larm_v1/model_ema.pt \
    --set vae_weights=cvpr_2025_vae/pcd_vqvae_16k_long/model_best.pt
```

Execute one recipe:

```bash
python3 scripts/reproduce_experiment.py --execute --recipe iros2023:mugs-main-train
```

Installed environments can also use the console entry point:

```bash
reproduce_experiment --list --suite inst3d2026
```

## Dataset keys

The commands use `./scripts/run.sh`, which appends `dirs=default`. Set the needed roots in `conf/dirs/default.yaml` before running experiments:

| Key | Used by |
|-----|---------|
| `log` | checkpoint lookup and training outputs |
| `shapenet_v1_onet` | IROS 2023 mug experiments |
| `shapenet_v1_fused` | 3DV 2026 latent generative experiments |
| `automatica` | Humanoids 2023 grasping experiments |
| `tabletop` | Inst3D joint and pipeline training |
| `tabletop_pile` | Inst3D pile fine-tuning and pile evaluation |
| `bop` | Inst3D BOP scene evaluation |

Keep local machine paths in your untracked dirs config. Public docs and scripts should stay independent of a particular cluster, workstation, or package profile.

## Paper suites

### IROS 2023: uncertain regions

Core recipes:

```bash
python3 scripts/reproduce_experiment.py --list --suite iros2023
```

The suite covers uncertain-label preprocessing, the ConvONet mug model, the MCDropout/PSSNet/RealNVP baselines, and `gen_eval` for a trained checkpoint:

```bash
python3 scripts/reproduce_experiment.py --recipe iros2023:watertight-meshes \
    --set mesh_root=ShapeNetCore.v1 \
    --set watertight_root=shapenet_watertight \
    --set n_jobs=8

python3 scripts/reproduce_experiment.py --recipe iros2023:sample-points \
    --set watertight_root=shapenet_watertight \
    --set samples_root=shapenet_processed \
    --set n_jobs=8

python3 scripts/reproduce_experiment.py --recipe iros2023:mugs-main-eval \
    --set weights=mugs_paper/run_name/model_best.pt
```

When using the Zenodo extract, uncertain labels are already present; run `iros2023:uncertain-labels` only when regenerating labels from raw rendered depth.

### Humanoids 2023: completion for grasping

Core recipes:

```bash
python3 scripts/reproduce_experiment.py --list --suite humanoids2023
```

The suite covers Kinect-style rendering, VQDIF training/evaluation, and the PointTransformer and ShapeFormer comparison models:

```bash
python3 scripts/reproduce_experiment.py --recipe humanoids2023:watertight-meshes \
    --set mesh_root=automatica_meshes \
    --set watertight_root=automatica_watertight \
    --set n_jobs=8

python3 scripts/reproduce_experiment.py --recipe humanoids2023:render-kinect \
    --set mesh=mesh.obj \
    --set depth_root=depth_renders

python3 scripts/reproduce_experiment.py --recipe humanoids2023:stable-poses \
    --set watertight_root=automatica_watertight \
    --set poses_root=automatica_poses \
    --set n_jobs=8
```

### 3DV 2026: latent generative completion

Core recipes:

```bash
python3 scripts/reproduce_experiment.py --list --suite 3dv2026
```

The suite covers VAE/VQ-VAE stage-one training, LDM/AR training, conditional Kinect variants, and representative generation evaluation. LDM and AR checkpoints do not include the VAE/VQ-VAE weights, so pass `vae_weights` again at evaluation time.

The data-generation recipes cover the public preprocessing chain: watertight meshes, point/occupancy samples, Kinect-style partial observations, and optional HDF5 packing:

```bash
python3 scripts/reproduce_experiment.py --recipe 3dv2026:watertight-meshes \
    --set mesh_root=ShapeNetCore.v1 \
    --set watertight_root=shapenet_watertight \
    --set n_jobs=8

python3 scripts/reproduce_experiment.py --recipe 3dv2026:sample-points \
    --set watertight_root=shapenet_watertight \
    --set samples_root=shapenet_processed \
    --set n_jobs=8

python3 scripts/reproduce_experiment.py --recipe 3dv2026:render-kinect-parallel \
    --set watertight_root=shapenet_watertight \
    --set depth_root=shapenet_kinect \
    --set n_jobs=8
```

The AR recipes intentionally use `cvpr_2025` with overrides:

```text
model.arch=larm model.compile=False ++vae_arch=3dshape2vecset_vqvae
```

Evaluation recipes also pass `norm.center=True norm.scale=True` explicitly because the trained runs used those resolved values.

### Inst3D/ECCV: instance completion

Core recipes:

```bash
python3 scripts/reproduce_experiment.py --list --suite inst3d2026
```

The suite covers joint training, pipeline training, pile fine-tuning/evaluation, and BOP camera-frame scene evaluation. Pile recipes use `inputs.type=kinect_sim`, `data.train_ds=[tabletop_pile]`, and `test.no_aug=True`. BOP recipes use camera-frame settings and disable the filtering/refinement switches that were not part of the public sanity path:

```bash
python3 scripts/reproduce_experiment.py --recipe inst3d2026:render-tabletop-packed \
    --set num_scenes=100 \
    --set num_views=10 \
    --set shard=0 \
    --set tabletop_dataset=tabletop.v3

python3 scripts/reproduce_experiment.py --recipe inst3d2026:add-kinect-sim \
    --set hdf5_root=tabletop.v3/train/0 \
    --set shapenet_root=ShapeNetCore.v1

python3 scripts/reproduce_experiment.py --recipe inst3d2026:bop-pipeline-eval \
    --set bop_split=bopscene_hb_val_primesense \
    --set occ_weights=tabletop_pile/comp_cam_kinect_10_epochs/model_best.pt \
    --set inst_weights=tabletop_pile/instseg_cam_kinect_10_epochs/model_best.pt
```

Use camera-frame checkpoints for BOP recipes. World-frame pile checkpoints are not compatible with BOP scene geometry.

The tabletop rendering wrapper expects `DATA_ROOT` in the execution environment. Keep that value local; it is intentionally not part of the public recipe command.

## Sanity checks

The public sanity checks render every recipe and Hydra-compose every recipe that goes through `scripts/run.sh`. They are not a replacement for full training or evaluation:

```bash
pytest experiments/tests/test_reproduction.py
pytest experiments/tests/test_reproduction_hydra.py
```

These tests should pass without private paths, Slurm, or local CUDA setup. GPU-node runs are useful for final local validation, but those commands belong outside the public repository.
