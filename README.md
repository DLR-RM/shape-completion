# Shape Completion

A modular research platform for 3D shape completion using deep learning. Implements 30+ architectures spanning implicit functions, point cloud completion, diffusion models, transformers, and autoencoders.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)

## Publications

| Paper | Venue | Year | Links |
|-------|-------|------|-------|
| Shape Completion with Prediction of Uncertain Regions | IROS | 2023 | [project](https://hummat.github.io/2023-iros-uncertain/) / [arXiv](https://arxiv.org/abs/2308.00377) / [IEEE](https://doi.org/10.1109/IROS55552.2023.10342487) |
| Combining Shape Completion and Grasp Prediction for Fast and Versatile Grasping with a Multi-Fingered Hand | Humanoids | 2023 | [project](https://hummat.github.io/2023-humanoids-completion/) / [arXiv](https://arxiv.org/abs/2310.20350) / [IEEE](https://doi.org/10.1109/Humanoids57100.2023.10375210) |
| Evaluating Latent Generative Paradigms for High-Fidelity 3D Shape Completion from a Single Depth Image | 3DV | 2026 | [project](https://hummat.github.io/2026-3dv-genz/) / [arXiv](https://arxiv.org/abs/2511.11074) |

## Features

- **30+ Model Architectures**: ONet, ConvONet, IF-Net, PCN, SnowflakeNet, ShapeFormer, diffusion models, VAE/VQ-VAE, and more
- **Multiple Input Modalities**: Point clouds, depth images, RGB-D, simulated Kinect
- **Flexible Data Pipeline**: Field-based loading with 55+ transforms for augmentation
- **Distributed Training**: PyTorch Lightning with SLURM/DDP support
- **Comprehensive Evaluation**: Chamfer, EMD, F1, IoU, FID metrics
- **Multi-Backend Rendering**: PyRender, Open3D, PyTorch3D, Blender (path tracing)

## Quick Start

```bash
git clone https://github.com/DLR-RM/shape-completion.git
cd shape-completion
uv sync --inexact --extra ml       # install Python deps (creates .venv automatically)
source .venv/bin/activate          # activate the venv
python libs/libmanager.py install  # compile C++/CUDA libraries (Chamfer, EMD, ...)

# configure dataset paths, then train
# edit conf/dirs/default.yaml first
train -cn shapenet_v1 model.arch=onet
```

## Architecture

```
shape-completion/
├── models/       # 30+ ML architectures (ONet, ConvONet, diffusion, transformers...)
├── train/        # PyTorch Lightning training infrastructure
├── dataset/      # Data loading with Fields + Transforms system
├── conf/         # Hydra configuration files
├── eval/         # Mesh generation and evaluation metrics
├── inference/    # Real-world inference with preprocessing
├── visualize/    # Mesh extraction (MISE) and multi-backend rendering
├── process/      # Data preprocessing (watertight meshes, Kinect simulation)
├── utils/        # Shared utilities (80+ functions)
└── libs/         # C++/CUDA libraries (Chamfer, EMD, MISE, Kinect sim)
```

### Data Flow

```
Config (conf/)
    │
    ├── Dataset (dataset/)
    │   ├── Fields → Load data (point clouds, meshes, images)
    │   └── Transforms → Augmentation pipeline
    │
    ├── Model (models/)
    │   └── get_model(cfg) → Architecture selection
    │
    ├── Training (train/)
    │   ├── LitModel → Lightning wrapper
    │   └── Callbacks → Mesh generation, evaluation, visualization
    │
    └── Evaluation (eval/)
        ├── generate → Extract meshes via MISE
        └── mesh_eval → Compute metrics
```

## Available Models

### Implicit Shape Completion

| Model | Config `arch` | Paper |
|-------|---------------|-------|
| ONet | `onet` | [Occupancy Networks](https://arxiv.org/abs/1812.03828) |
| ConvONet | `conv_onet*` | [Convolutional Occupancy Networks](https://arxiv.org/abs/2003.04618) |
| IF-Net | `if_net` | [IF-Net](https://arxiv.org/abs/2003.01456) |
| VQDIF | `vqdif` | [ShapeFormer](https://arxiv.org/abs/2201.10326) |
| ShapeFormer | `shapeformer` | [ShapeFormer](https://arxiv.org/abs/2201.10326) |
| DMTet | `dmtet*` | [DMTet](https://arxiv.org/abs/2111.04276) |
| CompleTr | `completr` | -- |
| PointTransformer | `point_tr*` | -- |
| ImplicitNetwork | `idr` | [IDR](https://arxiv.org/abs/2003.09852) |
| MCDropoutNet | `dropout*` | [Lundell et al. 2019](https://doi.org/10.1109/IROS40897.2019.8967816) |
| PSSNet | `pssnet` | [Saund & Berenson 2020](https://proceedings.mlr.press/v155/saund21a.html) |
| RealNVP | `realnvp` | [Dinh et al. 2017](https://arxiv.org/abs/1605.08803) |

### Point Cloud Completion

| Model | Config `arch` | Paper |
|-------|---------------|-------|
| PCN | `pcn` | [Point Completion Network](https://arxiv.org/abs/1808.00671) |
| PSGN | `psgn` | [PSGN](https://arxiv.org/abs/1612.00603) |
| SnowflakeNet | `snowflakenet` | [SnowflakeNet](https://arxiv.org/abs/2108.04444) |
| PVD | `pvd` | [PVD](https://arxiv.org/abs/2104.03670) |

### Generative Models

| Model | Config `arch` | Paper |
|-------|---------------|-------|
| Shape3D2VecSet | `3dshape2vecset` | [3DShape2VecSet](https://arxiv.org/abs/2301.11445) / [3DV 2026](https://arxiv.org/abs/2511.11074) |
| Shape3D2VecSetVAE | `3dshape2vecset_vae` | [3DV 2026](https://arxiv.org/abs/2511.11074) |
| Shape3D2VecSetVQVAE | `3dshape2vecset_vqvae` | [3DV 2026](https://arxiv.org/abs/2511.11074) |
| LatentDiffusionModel | `ldm` | [3DV 2026](https://arxiv.org/abs/2511.11074) |
| GridDiffusionModel | `grid_diffusion` | -- |
| LatentAutoregressiveModel | `larm` | -- |

See [models/README.md](models/README.md) for the complete list of 30+ architectures including DINOv2 variants, point feature extractors, and configuration options.

## Installation

### Requirements

- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- CUDA toolkit (for GPU-accelerated libraries)

Python 3.11 is specified in `.python-version`; `uv sync` auto-creates the venv with the right interpreter.

### 1. Clone and install

```bash
git clone https://github.com/DLR-RM/shape-completion.git
cd shape-completion
./scripts/bootstrap_dev.sh      # canonical dev/test bootstrap (uses --extra full)
source .venv/bin/activate       # optional, uv run works without manual activation
```

`bootstrap_dev.sh` runs `uv sync --inexact --extra <extra>` and installs Tier-1 custom libs via `libs/libmanager.py`.
Use `./scripts/bootstrap_dev.sh --extra ml` for a lighter profile and `--with-tier2-libs` when you want PyTorch3D/tiny-cuda-nn/torch-scatter builds too.

### 2. Choose your extras

| Extra | What's included |
|-------|-----------------|
| `torch` | PyTorch, torchvision, Lightning |
| `ml` | torch + train + eval + inference + git deps |
| `full` | ml + visualize + process + test + wandb + plotly + fid |
| `all` | full + blenderproc + pathtracing + diffusion + rembg + 8bit + hypergradient + compile |

Pick one:

```bash
uv sync --inexact --extra torch --extra train     # just training
uv sync --inexact --extra ml                      # train + eval + inference
uv sync --inexact --extra full                    # common dependencies
uv sync --inexact --extra all                     # everything
```

> **Why `--inexact`?** By default, `uv sync` removes packages not in the lockfile — including compiled C++/CUDA libs installed via `uv pip install` (libmanager, compile_cuda_libs.sh). `--inexact` keeps them.

Individual module extras (`dataset`, `models`, `eval`, `inference`, `visualize`, `process`) are also available for minimal installs.

> **Note:** The `diffusion` extra installs a [custom fork of diffusers](https://github.com/hummat/diffusers/tree/vdm-scheduler) with VDM scheduler support. This is required for latent diffusion models.

<details>
<summary>Using pip instead of uv</summary>

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[ml]"       # or full, all, etc.
```

</details>

### 3. Compile C++/CUDA libraries

Three tiers of compiled dependencies, each building on the previous:

#### Tier 1: Custom libs (Chamfer, EMD, MISE, Kinect sim, ...)

```bash
python libs/libmanager.py install                # all libraries
python libs/libmanager.py install chamfer emd    # or specific ones
```

`install` skips already-installed libs. Use `upgrade` or `force-reinstall` to overwrite.
See [libs/README.md](libs/README.md) for details and system dependencies (libkinect needs CGAL, Eigen, OpenCV, assimp, libnoise).

#### Tier 2: PyTorch3D, tiny-cuda-nn, xFormers

These require specific compiler and CUDA arch settings that can't be expressed in the lockfile, so they're installed separately via `uv pip install`:

```bash
# Edit CC/CXX and TORCH_CUDA_ARCH_LIST for your system first
./scripts/compile_cuda_libs.sh
```

| Library | Used by | Install |
|---------|---------|---------|
| [torch-scatter](https://github.com/rusty1s/pytorch_scatter) | Grid-based models (ConvONet, IF-Net, CompleTr, ...) | Prebuilt wheel |
| [torch-cluster](https://github.com/rusty1s/pytorch_cluster) | FPS sampling (3DShape2VecSet) | Prebuilt wheel |
| [Detectron2](https://github.com/facebookresearch/detectron2) | Mask R-CNN, PointRend point sampling | Source build |
| [PyTorch3D](https://github.com/facebookresearch/pytorch3d) | Heterogeneous batching, 3D ops | Source build |
| [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) | NeRF-like positional encoding (tcnn backend) | Source build |
| [xFormers](https://github.com/facebookresearch/xformers) | Memory-efficient attention | Wheel (torch >=2.10) |

All are optional — the code falls back to pure-PyTorch alternatives when unavailable.

### 4. Configure paths

Edit `conf/dirs/default.yaml` with your dataset and log paths:

```yaml
log: /path/to/training/logs
shapenet_v1_fused: /path/to/shapenet/ShapeNetCore.v1.fused
# ... other datasets
```

### 5. Validation lanes

Use the built-in scripts for fast PR checks vs broader nightly checks:

```bash
./scripts/test/check_pr.sh       # fast lane: scoped ruff + scoped pyright + CPU-safe pytest subset
./scripts/test/check_all.sh      # full lane: repo-wide ruff/pyright + full pytest --cov (headline informational) + source-only gate
./scripts/test/check_nightly.sh  # broad lane: full pytest with default markers (renderer quarantined)
```

Quality policy and coverage interpretation live in [docs/quality.md](docs/quality.md). Supported vs experimental workflow status lives in [docs/status.md](docs/status.md).

## Usage

### Training

```bash
# Basic training
train -cn shapenet_v1

# With model selection
train -cn shapenet_v1 model.arch=conv_onet

# With options
train -cn shapenet_v1 \
    model.arch=conv_onet \
    train.lr=1e-4 \
    train.epochs=200 \
    log.wandb=true
```

See [train/README.md](train/README.md) for training options, callbacks, and distributed training.

### Evaluation

```bash
# Generate meshes from trained model
generate -cn shapenet_v1 model.weights=path/to/model_best.pt

# Evaluate generated meshes
mesh_eval -cn shapenet_v1 test.dir=path/to/meshes

# Combined generation + evaluation
gen_eval -cn shapenet_v1 model.weights=path/to/model_best.pt
```

See [eval/README.md](eval/README.md) for metrics and evaluation options.

### Inference

```bash
# On point cloud file
inference input.ply onet -w model.pt -o results/

# On depth image
inference depth.png conv_onet -w model.pt --intrinsic 582.7 582.7 320.8 245.3

# With plane removal (real-world scenes)
inference scene.ply onet -w model.pt --remove_plane --show
```

See [inference/README.md](inference/README.md) for preprocessing options.

### Visualization

```bash
# Visualize data loading and augmentation
vis_data -cn shapenet_v1 vis.show=true vis.mesh=true

# Render meshes
render path/to/meshes -o renders/ --method cycles
```

See [visualize/README.md](visualize/README.md) for Generator and Renderer options.

### Data Processing

```bash
# Generate watertight meshes
make_watertight path/to/ShapeNetCore.v1

# Render Kinect-style depth
render_kinect mesh.obj -o depth.png --noise perlin

# Find uncertain regions
find_uncertain_regions -cn shapenet_uncertain
```

See [process/README.md](process/README.md) for preprocessing scripts.

## Reproducing Paper Results

### C1: Shape Completion with Prediction of Uncertain Regions (IROS 2023)

[Project page](https://hummat.github.io/2023-iros-uncertain/) / [arXiv](https://arxiv.org/abs/2308.00377) / [IEEE](https://doi.org/10.1109/IROS55552.2023.10342487)

**Configs:**

| Config | Role |
|--------|------|
| `mugs.yaml` | Base: ConvONet grid, depth back-projection, mugs category |
| `mugs_paper.yaml` | Paper ablation driver: `num_classes=2`, `with_handle.lst` split |
| `mugs_no_aug.yaml` | No-augmentation evaluation (loads full mesh, all points, disables normalization) |
| `mugs_overfit.yaml` | Overfit debug |
| `mugs_dropout.yaml` | MCDropout baseline (`mcdropoutnet`) |
| `mugs_pssnet.yaml` | PSSNet baseline |
| `mugs_realnvp.yaml` | RealNVP normalizing flow baseline |
| `mugs_sim2real_norm.yaml` | Sim-to-real: normalization variant |
| `mugs_sim2real_scale.yaml` | Sim-to-real: scale augmentation variant |
| `mugs_300p_norm.yaml` | 300-point real-world input variant |
| `shapenet_uncertain.yaml` | ShapeNet-scale uncertain region evaluation |

**Key source files:**

| File | Purpose |
|------|---------|
| `models/src/conv_onet.py` | ConvONet with `num_classes` param (2=occupied+uncertain, 3=+free) |
| `models/src/mc_dropout_net.py` | MCDropout baseline |
| `models/src/pssnet.py` | PSSNet baseline |
| `models/src/realnvp.py` | RealNVP baseline |
| `process/scripts/find_uncertain_regions.py` | Computes per-query uncertain labels from multi-view depth |
| `dataset/src/transforms.py` | `FindUncertainPoints`, `LoadUncertain` transforms |

**Dataset:** ShapeNet mugs category (`03797390`), available on [Zenodo](https://zenodo.org/records/10284230).

**Commands:**

```bash
# Preprocess: compute uncertain region labels
find_uncertain_regions -cn shapenet_uncertain

# Train main model
train -cn mugs_paper

# Train baselines
train -cn mugs_dropout    # MCDropout
train -cn mugs_pssnet     # PSSNet
train -cn mugs_realnvp    # RealNVP

# Evaluate
gen_eval -cn mugs_paper model.weights=path/to/model_best.pt
```

### C2: Combining Shape Completion and Grasp Prediction (Humanoids 2023)

[Project page](https://hummat.github.io/2023-humanoids-completion/) / [arXiv](https://arxiv.org/abs/2310.20350) / [IEEE](https://doi.org/10.1109/Humanoids57100.2023.10375210)

**Configs:**

| Config | Role |
|--------|------|
| `humanoids_2023.yaml` | Top-level: VQDIF on automatica + kinect input |
| `automatica_2023.yaml` | Base: depth input on automatica objects |
| `automatica_2023_kinect.yaml` | Kinect-simulated depth on automatica |
| `automatica_2023_kinect_gpt2.yaml` | PointTransformer comparison (named `_gpt2` after the Karpathy-style transformer implementation, not GPT-2) |
| `shapenet_v1_depth_shapeformer.yaml` | ShapeFormer comparison |

**Key source files:**

| File | Purpose |
|------|---------|
| `models/src/vqdif.py` | VQDIF: voxel-based discrete implicit field (primary model) |
| `models/src/shapeformer.py` | ShapeFormer: transformer on VQDIF latent codes |
| `models/src/point_transformer.py` | PointTransformer comparison model |

**Dataset:** Automatica objects (BOP-style graspable objects), kinect-simulated depth.

**Commands:**

```bash
# Simulate Kinect depth
render_kinect mesh.obj -o depth.png --noise perlin

# Train primary model
train -cn humanoids_2023

# Train comparisons
train -cn automatica_2023_kinect_gpt2    # PointTransformer
train -cn shapenet_v1_depth_shapeformer  # ShapeFormer

# Evaluate
gen_eval -cn humanoids_2023 model.weights=path/to/model_best.pt
```

### C3: Evaluating Latent Generative Paradigms (3DV 2026)

[Project page](https://hummat.github.io/2026-3dv-genz/) / [arXiv](https://arxiv.org/abs/2511.11074)

> **Note:** C3 configs are named `cvpr_2025*`, preserving the development naming despite the 3DV 2026 venue.

**Configs:**

| Config | Role |
|--------|------|
| `cvpr_2025.yaml` | Main: conditional LDM with 3dshape2vecset_vae encoder |
| `cvpr_2025_vae.yaml` | Stage 1: VAE training (mixed pointcloud+depth) |
| `cvpr_2025_cond.yaml` | Depth-conditioned LDM variant |
| `cvpr_2025_depth.yaml` | Depth-only input variant |
| `cvpr_2025_depth_cls.yaml` | Classification ablation (57 classes) |
| `cvpr_2025_depth_dvr.yaml` | DVR comparison baseline |
| `shapenet_v1_ldm.yaml` | Base unconditional LDM |
| `shapenet_v1_vae_for_ldm.yaml` | VAE stage for LDM (mixed inputs) |
| `shapenet_v1_vae.yaml` | Standalone VAE |
| `shapenet_v1_vae_dif.yaml` | LDM on VAE latents |
| `shapenet_v1_latent.yaml` | LDM on autoencoder latents |
| `shapenet_v1_autoenc.yaml` | 3dshape2vecset autoencoder |
| `automatica_2024_kinect.yaml` | Real-robot eval with 3dshape2vecset |

**Key source files:**

| File | Purpose |
|------|---------|
| `models/src/shape3d2vecset.py` | Core transformer: AE, VAE, VQ-VAE, classification variants |
| `models/src/diffusion/latent.py` | Latent diffusion model |
| `models/src/diffusion/transformer.py` | Diffusion transformer architecture |
| `models/src/diffusion/edm.py` | EDM preconditioning |
| `models/src/vae.py` | VAE and VQ-VAE wrappers |
| `models/src/autoregression/latent.py` | Latent autoregressive model |

**Dataset:** ShapeNet v1 (all categories), automatica objects for real-robot evaluation.

**Two-stage training:**

```bash
# Stage 1: Train VAE
train -cn cvpr_2025_vae

# Stage 2: Train LDM (requires VAE checkpoint from stage 1)
# +vae_weights is relative to dirs.log (set in conf/dirs/default.yaml)
train -cn cvpr_2025 +vae_weights=cvpr_2025_vae/<run_name>/model_best.pt

# Evaluate (must pass +vae_weights again — LDM checkpoints don't include VAE weights)
gen_eval -cn cvpr_2025 model.weights=path/to/ldm/model_best.pt \
    +vae_weights=cvpr_2025_vae/<run_name>/model_best.pt
```

## Configuration

All experiments use [Hydra](https://hydra.cc/) configs from `conf/`. The base config (`config.yaml`) defines all options with defaults.

```bash
# Use a config
train -cn shapenet_v1

# Override values
train -cn shapenet_v1 model.arch=conv_onet train.lr=1e-4

# Multi-run sweep
train -cn shapenet_v1 model.arch=onet,conv_onet,if_net --multirun
```

See [conf/README.md](conf/README.md) for the complete configuration schema.

## Custom Libraries

| Library | Description |
|---------|-------------|
| `libchamfer` | Chamfer distance (CUDA) |
| `libemd` | Earth Mover's Distance (CUDA) |
| `libfusion` | GPU-accelerated TSDF fusion for watertight mesh generation |
| `libintersect` | Mesh intersection and inside-mesh queries |
| `libkinect` | Kinect depth sensor simulator (structured light stereo, configurable noise) |
| `libmise` | Multi-resolution isosurface extraction (MISE) |
| `libpointnet` | PointNet2 CUDA operations |
| `libsimplify` | Fast quadric edge collapse mesh decimation |

Install via `python libs/libmanager.py install`. See [libs/README.md](libs/README.md) for per-library build dependencies.

## Modules

| Module | Description | Key Features |
|-----------|-------------|--------------|
| [**models/**](models/README.md) | ML architectures | 30+ models, `get_model()` factory, attention backends |
| [**train/**](train/README.md) | Training infrastructure | LitModel, callbacks, schedulers, DDP/SLURM |
| [**dataset/**](dataset/README.md) | Data loading | Fields system, 55+ transforms, 7 datasets |
| [**conf/**](conf/README.md) | Configuration | Hydra configs, complete schema reference |
| [**eval/**](eval/README.md) | Evaluation | Mesh generation, 16+ metrics |
| [**inference/**](inference/README.md) | Inference | Real-world preprocessing, plane removal |
| [**visualize/**](visualize/README.md) | Visualization | MISE extraction, 5 rendering backends |
| [**process/**](process/README.md) | Preprocessing | Watertight meshes, Kinect simulation |
| [**utils/**](utils/README.md) | Utilities | 80+ functions, coordinate transforms |
| [**libs/**](libs/README.md) | C++/CUDA libraries | Chamfer, EMD, MISE, Kinect sim |


## Example Workflows

### Train ONet on ShapeNet

```bash
# 1. Train
train -cn shapenet_v1 model.arch=onet train.epochs=100

# 2. Generate meshes
generate -cn shapenet_v1 \
    model.weights=logs/shapenet_v1/onet/version_0/model_best.pt

# 3. Evaluate
mesh_eval -cn shapenet_v1 \
    test.dir=logs/shapenet_v1/onet/version_0/meshes
```

### Real-World Inference

```bash
# Inference on Kinect scan with plane removal
inference kinect_scene.ply conv_onet \
    -w model.pt \
    --remove_plane \
    --cluster \
    -o completed/
```

### Prepare Custom Dataset

```bash
# 1. Generate watertight meshes
make_watertight path/to/meshes --out_dir path/to/watertight

# 2. Create config
# conf/my_dataset.yaml with data paths

# 3. Train
train -cn my_dataset model.arch=onet
```

## FAQ

**How to install CUDA without root?**

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
chmod +x cuda_12.4.0_550.54.14_linux.run
./cuda_12.4.0_550.54.14_linux.run --toolkit --toolkitpath=/path/to/custom/directory
```

**How to use multiple GPUs?**

```bash
# Local multi-GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 train -cn shapenet_v1

# SLURM
srun --gres=gpu:4 train -cn shapenet_v1
```

**How to resume training?**

```bash
train -cn shapenet_v1 train.resume=true
# or specify checkpoint
train -cn shapenet_v1 model.checkpoint=path/to/last.ckpt
```

**How to use a different attention backend?**

```bash
train -cn shapenet_v1 model.attn_backend=xformers  # or torch, einops
```

## Citation

If you use this codebase, please cite the relevant paper(s):

```bibtex
@inproceedings{humt2023shape,
  author    = {Humt, Matthias and Winkelbauer, Dominik and Hillenbrand, Ulrich},
  title     = {Shape Completion with Prediction of Uncertain Regions},
  booktitle = {2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2023},
  pages     = {1215--1221},
  doi       = {10.1109/IROS55552.2023.10342487},
}

@inproceedings{humt2023combining,
  author       = {Humt, Matthias and Winkelbauer, Dominik and Hillenbrand, Ulrich and B{\"a}uml, Berthold},
  title        = {Combining Shape Completion and Grasp Prediction for Fast and Versatile Grasping with a Multi-Fingered Hand},
  booktitle    = {2023 IEEE-RAS 22nd International Conference on Humanoid Robots (Humanoids)},
  year         = {2023},
  pages        = {1--8},
  doi          = {10.1109/Humanoids57100.2023.10375210},
  organization = {IEEE},
}

@inproceedings{humt2026generative,
  author        = {Humt, Matthias and Hillenbrand, Ulrich and Triebel, Rudolph},
  title         = {Evaluating Latent Generative Paradigms for High-Fidelity {3D} Shape Completion from a Single Depth Image},
  booktitle     = {International Conference on 3D Vision (3DV)},
  year          = {2026},
  doi           = {10.48550/arXiv.2511.11074},
  eprint        = {2511.11074},
  archiveprefix = {arXiv},
  primaryclass  = {cs.CV},
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
