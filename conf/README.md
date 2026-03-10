# shape-completion-conf

Hydra configuration files for shape completion experiments.

## Installation

```bash
# As submodule
```

## Quick Start

```bash
# Train with a config
train -cn shapenet_v1

# Override specific values
train -cn shapenet_v1 model.arch=conv_onet train.lr=1e-4

# Use different directory config
train -cn shapenet_v1 +dirs=dlr
```

## File Structure

```
conf/
├── config.yaml                        # Base config: complete schema with all defaults
├── dirs/                              # Directory path configs (dataset/log locations)
│   ├── default.yaml                   # Your local paths (edit this first)
│   ├── dlr.yaml                       # DLR cluster paths
│   ├── dlr_alt.yaml                   # DLR alternate paths
│   ├── dlr_hdf5.yaml                  # DLR paths for HDF5 datasets
│   ├── lrz.yaml                       # LRZ cluster paths
│   ├── laptop.yaml                    # Local laptop paths
│   └── notebook.yaml                  # Jupyter notebook paths
│
│── # ── ShapeNet v1 ──────────────────
├── shapenet_v1.yaml                   # Point cloud completion (base for most experiments)
├── shapenet_v1_depth.yaml             # Depth image input
├── shapenet_v1_depth_like.yaml        # Depth-like partial views
├── shapenet_v1_depth_image.yaml       # Depth + image input
├── shapenet_v1_depth_pcd.yaml         # Depth point cloud input
├── shapenet_v1_depth_shapeformer.yaml # ShapeFormer-style depth
├── shapenet_v1_pcd.yaml               # Point cloud completion variant
├── shapenet_v1_autoenc.yaml           # Autoencoder training
├── shapenet_v1_vae.yaml               # VAE training
├── shapenet_v1_vae_dif.yaml           # VAE with diffusion
├── shapenet_v1_vae_for_ldm.yaml       # VAE pretrained for LDM
├── shapenet_v1_ldm.yaml               # Latent diffusion model
├── shapenet_v1_latent.yaml            # Latent space experiments
├── shapenet_v1_cls.yaml               # Classification
├── shapenet_v1_cls_kinect.yaml        # Classification with Kinect input
├── shapenet_v1_cls_seg_*.yaml         # Joint classification + segmentation
├── shapenet_v1_seg_inputs.yaml        # Segmentation on inputs
├── shapenet_v1_seg_inputs_depth.yaml  # Segmentation on depth inputs
├── shapenet_v1_seg_inputs_kinect.yaml # Segmentation on Kinect inputs
├── shapenet_v1_seg_points.yaml        # Segmentation on query points
├── shapenet_v1_seg_points_baseline.yaml # Segmentation baseline
├── shapenet_v1_seg_points_depth.yaml  # Segmentation on depth query points
├── shapenet_v1_seg_points_kinect.yaml # Segmentation on Kinect query points
├── shapenet_v1_no_surface.yaml        # Training without surface points
├── shapenet_v1_core.yaml              # ShapeNet core subset
│
│── # ── ShapeNet v2 / other formats ──
├── shapenet_v2.yaml                   # ShapeNet v2
├── shapenet_v2_depth.yaml             # ShapeNet v2 with depth
├── shapenet.yaml                      # ShapeNet (ONet format, base for mugs)
├── shapenet_core.yaml                 # ShapeNet core
├── shapenet_depth_like.yaml           # Depth-like on ShapeNet
├── shapenet_uncertain.yaml            # Uncertainty-aware ShapeNet
├── onet.yaml                          # Occupancy Networks config
├── onet_shapenet_v1.yaml              # ONet on ShapeNet v1
│
│── # ── Benchmarks & Other Datasets ──
├── completion3d.yaml                  # Completion3D benchmark
├── modelnet40.yaml                    # ModelNet40 classification
├── dtu_dvr.yaml                       # DTU with differentiable volume rendering
├── dtu_dvr_fast.yaml                  # DTU DVR fast variant
│
│── # ── Tabletop ─────────────────────
├── tabletop.yaml                      # Tabletop scene completion
├── tabletop_3d.yaml                   # Tabletop 3D variant
├── tabletop_inst_seg.yaml             # Instance segmentation
├── tabletop_inst_seg_3d_canonical.yaml
├── tabletop_inst_seg_pcd.yaml
├── tabletop_inst_seg_pcd_3d.yaml
│
│── # ── Real-world / Demo ────────────
├── automatica.yaml                    # Automatica base config
├── automatica_depth.yaml              # Automatica depth variant
├── automatica_dvr.yaml                # Automatica DVR variant
├── automatica_2023.yaml               # Automatica 2023 demo
├── automatica_2023_kinect.yaml        # Automatica 2023 Kinect
├── automatica_2023_kinect_gpt2.yaml   # Automatica 2023 GPT-2 backbone
├── automatica_2024_kinect.yaml        # Automatica 2024 Kinect demo
├── mugs.yaml                          # Mug completion (depth_bp input)
├── mugs_*.yaml                        # Mug variants (paper, overfit, dropout, etc.)
│
│── # ── Publications ─────────────────
├── humanoids_2023.yaml                # Humanoids 2023
├── cvpr_2025.yaml                     # CVPR 2025 (latent diffusion)
├── cvpr_2025_vae.yaml                 # CVPR 2025 VAE
├── cvpr_2025_cond.yaml                # CVPR 2025 conditional
├── cvpr_2025_depth.yaml               # CVPR 2025 depth
├── cvpr_2025_depth_cls.yaml           # CVPR 2025 depth classification
├── cvpr_2025_depth_dvr.yaml           # CVPR 2025 depth DVR
│
│── # ── Image Diffusion ──────────────
├── mnist_diffusion.yaml               # MNIST diffusion
├── fmnist_diffusion.yaml              # FashionMNIST diffusion
├── cifar10_diffusion.yaml             # CIFAR-10 diffusion
│
│── # ── Debug / Overfitting ──────────
├── overfit.yaml                       # Single-mesh overfitting (bunny)
├── overfit_bunny.yaml                 # Bunny overfit variant
├── overfit_automatica.yaml            # Automatica overfit
└── viz_overfit_bunny.yaml             # Visualization of bunny overfit
```

## Configuration System

### Inheritance

Every experiment config inherits from `config.yaml` via Hydra's `defaults` list. The base config defines the complete schema with all default values. Experiment configs only need to override the keys they change.

**Direct inheritance** (most configs):

```yaml
# shapenet_v1.yaml
defaults:
  - config          # Inherit all defaults from config.yaml
  - _self_          # Then apply this file's overrides
```

**Multi-level inheritance** (specialized configs inherit from other experiments):

```
config.yaml
├── shapenet_v1.yaml
│   ├── automatica.yaml
│   ├── shapenet_v1_depth.yaml
│   ├── shapenet_v1_vae.yaml
│   │   └── shapenet_v1_vae_for_ldm.yaml
│   │       ├── shapenet_v1_ldm.yaml
│   │       │   └── cvpr_2025.yaml
│   │       └── cvpr_2025_vae.yaml
│   └── ...
├── shapenet.yaml                      # ONet-format ShapeNet
│   └── mugs.yaml
│       └── mugs_*.yaml
├── overfit.yaml
└── ...
```

Hydra resolves these chains top-down: `config.yaml` defaults are loaded first, then each level applies its overrides. The `_self_` entry controls where the current file's keys are applied relative to inherited defaults (last = current file wins).

### Directory Configuration

Directory configs (`dirs/*.yaml`) map dataset names to filesystem paths. The base config loads `dirs/default.yaml` automatically:

```yaml
# config.yaml (excerpt)
defaults:
  - dirs: default     # Load dirs/default.yaml into the 'dirs' namespace
```

**First-time setup:** Edit `dirs/default.yaml` to point to your local dataset locations:

```yaml
# dirs/default.yaml
log: /path/to/training/logs
backup:                             # Optional backup log location

# Dataset root paths (referenced by data.train_ds in experiment configs)
shapenet_v1: /path/to/ShapeNetCore.v1
shapenet_v1_onet: /path/to/occupancy_networks
shapenet_v1_fused: /path/to/ShapeNetCore.v1.fused
shapenet_v1_fused_kinect: /path/to/ShapeNetCore.v1.fused.kinect
completion3d: /path/to/completion3d
ycb: /path/to/ycb/objects
bop: /path/to/bop
modelnet40: /path/to/ModelNet40
tabletop: /path/to/tabletop
tabletop_pile: /path/to/tabletop.pile
dtu: /path/to/DTU
automatica: /path/to/automatica
graspnet: /path/to/graspnet
# Image datasets (for diffusion experiments)
cifar10: /path/to/CIFAR10
mnist: /path/to/MNIST
fmnist: /path/to/FashionMNIST
```

The keys in `dirs/*.yaml` are referenced by `data.train_ds` in experiment configs. For example, `data.train_ds: [shapenet_v1_fused]` looks up the `shapenet_v1_fused` path from the active dirs config.

**Multiple environments:** Override the dirs config group on the command line:

```bash
train -cn shapenet_v1 +dirs=dlr      # DLR cluster paths
train -cn shapenet_v1 +dirs=laptop   # Local laptop paths
train -cn shapenet_v1 +dirs=lrz      # LRZ cluster paths
```

## Hydra Override Syntax

| Syntax | Meaning | Example |
|--------|---------|---------|
| `key=value` | Override an existing key | `train.lr=1e-4` |
| `+key=value` | Add a new key (error if exists) | `+dirs=dlr` |
| `++key=value` | Force set (add or override) | `++model.custom_param=42` |
| `~key` | Remove a key | `~aug.noise` |
| `key=[a,b]` | Override with list | `data.categories=[chair,table]` |
| `--multirun` | Sweep over comma-separated values | `model.arch=onet,conv_onet --multirun` |

The `+dirs=dlr` pattern overrides the default config group selection. Since `config.yaml` already declares `- dirs: default`, the `+` adds a second dirs entry which takes precedence. Use `++` when you need to force-set a key that might not exist in the schema.

## Quick Reference: Key Config Parameters

### Most commonly overridden keys

| Key | Default | Description |
|-----|---------|-------------|
| `model.arch` | *(none)* | Model architecture (e.g., `onet`, `conv_onet`, `conv_onet_grid`, `if_net`, `point_transformer`, `ldm`) |
| `data.train_ds` | *(none)* | Training dataset(s) as list of dirs keys |
| `data.categories` | *(all)* | ShapeNet category IDs to train on |
| `inputs.type` | `pointcloud` | Input modality: `pointcloud`, `depth`, `depth_bp`, `rgbd`, `kinect`, `image`, `partial`, `depth_like` |
| `inputs.num_points` | `300` | Number of input points |
| `train.epochs` | `10` | Training epochs |
| `train.batch_size` | `32` | Batch size |
| `train.lr` | `3.125e-6` | Learning rate (designed for batch 32; auto-scaled if `scale_lr: true`) |
| `train.precision` | `32-true` | Training precision: `32-true`, `16-mixed`, `bf16-mixed` |
| `train.optimizer` | `AdamW` | Optimizer: `AdamW`, `Adam`, `SGD`, `AdamW8bit` |
| `train.scheduler` | *(none)* | LR scheduler: `LinearWarmupCosineAnnealingLR`, `ReduceLROnPlateau`, `StepLR` |
| `log.wandb` | `false` | Enable Weights & Biases logging |
| `model.compile` | `false` | Enable `torch.compile()` |

### Learning rate scaling

The default LR (`3.125e-6`) is calibrated for batch size 32 on 1 GPU. When `train.scale_lr: true` (default), the effective LR is scaled by `(batch_size * num_gpus * accumulate_grad_batches) / 32`. Set `scale_lr: false` when specifying an absolute LR.

## Complete Config Schema

### Core Groups

| Group | Description |
|-------|-------------|
| `model` | Architecture, weights, compilation, network options |
| `data` | Dataset selection, categories, coordinate frame |
| `inputs` | Input modality and preprocessing |
| `points` | Query points for implicit function supervision |
| `pointcloud` | Surface point cloud loading |
| `mesh` | Mesh loading options |
| `train` | Training hyperparameters, optimizer, scheduler |
| `val` | Validation settings and visualization |
| `test` | Test/evaluation settings |
| `predict` | Prediction settings |
| `vis` | Visualization / mesh extraction settings |
| `aug` | Data augmentation |
| `norm` | Spatial normalization |
| `implicit` | Implicit function and DVR settings |
| `cls` | Classification head settings |
| `seg` | Segmentation head settings |
| `log` | Logging, W&B, checkpointing |
| `load` | DataLoader settings |
| `files` | File paths within each dataset sample |
| `misc` | Miscellaneous (seed) |

### Model Configuration

```yaml
model:
  arch: null                    # Architecture name (see models/README.md)
  weights: null                 # Path to pretrained weights (.pt)
  checkpoint: null              # Path to checkpoint (.ckpt)
  load_best: false              # Load model_best.pt from log dir
  compile: false                # torch.compile() for speed

  # Network options
  norm: null                    # Normalization: null|batch|layer|group|rms
  activation: relu              # Activation: relu|gelu|silu|mish
  dropout: 0.0                  # Dropout rate
  bias: true                    # Use bias in linear layers

  # Attention (transformers)
  attn: null                    # Enable attention layers
  attn_backend: null            # torch|xformers|einops
  attn_mode: null               # Attention mode override

  # Up/downsampling
  up: null                      # Upsampling method
  down: null                    # Downsampling method
  use_ws: null                  # Weight standardization

  # Model averaging
  average: null                 # ema|swa|null
  ema_decay: null               # EMA decay (auto if null)
  swr_lr: null                  # SWA learning rate

  # Loss
  reduction: mean               # mean|sum|none
```

### Data Configuration

```yaml
data:
  train_ds: null                # Training dataset(s): list of dirs/*.yaml keys
  val_ds: ${data.train_ds}      # Validation (defaults to train)
  test_ds: ${data.val_ds}       # Test (defaults to val)

  categories: null              # Category filter: list of IDs or null for all
  objects: null                  # Specific objects (YCB, BOP)

  # Sharding (multi-file datasets)
  num_files:                    # Number of data files per split
    train: 1
    val: 1
    test: 1
  num_shards:                   # Number of shards per split
    train: 1
    val: 1
    test: 1

  # Coordinate frame transforms
  frame: world                  # world|cam|net
  convention: opencv            # opencv|opengl
  unscale: true                 # Undo object scaling applied during rendering
  undistort: true               # Undo BlenderProc distortion (legacy)
  unrotate: true                # Undo object rotation applied during rendering
  scale: false                  # Rescale all data fields with inputs scale
  scale_multiplier: 1           # Scale multiplier
  rotate: false                 # Re-apply inputs rotation to all data fields
  sdf_from_occ: false           # Convert occupancy labels to SDF
  rot: null                     # Rotation matrix
  rot_x: null                   # X-axis rotation
  rot_y: null                   # Y-axis rotation
  rot_z: null                   # Z-axis rotation

  # Performance
  cache: false                  # Cache loaded data in memory
  hash_items: false             # Hash-based caching
  share_memory: false           # Shared memory for DDP
  dither: false                 # Dithering
  split: false                  # Split dataset
```

### Input Configuration

```yaml
inputs:
  type: pointcloud              # Input type (see table below)
  type_p: null                  # Type probabilities for mixed inputs
  type_key: null                # Type key override
  dim: 3                        # Dimension (3=xyz, 6=xyz+normals)
  num_points: 300               # Points to sample/load
  precision: 16                 # Input data precision

  # Bounds
  min_num_points: 1
  max_num_points: 100000

  # Processing
  nerf: false                   # NeRF positional encoding
  project: true                 # Project depth to point cloud
  cache: false                  # Cache loaded/rendered inputs
  load_random: false            # Random view each epoch
  crop: false                   # Crop inputs
  permute: false                # Permute point order
  bbox: false                   # Include bounding box

  # Image inputs
  width: null                   # Image width
  height: null                  # Image height
  normals: false                # Load normal maps
  resize: null                  # Resize factor

  # Directory overrides
  data_dir: null                # Custom data directory
  image_dir: null               # Custom image directory
  cam_dir: null                 # Custom camera directory

  # Point processing
  voxelize: null                # Voxel grid resolution
  fps:
    num_points: null            # Farthest point sampling count
    method: gpu                 # gpu|cpu
  bps:                          # Basis Point Set encoding
    num_points: null
    resolution: 32
    feature: delta
    basis: sphere               # Basis shape: sphere
    method: kdtree              # Nearest neighbor method
```

**Input Types:**

| Type | Description |
|------|-------------|
| `pointcloud` | Load point cloud from file |
| `depth` | Rendered/loaded depth map |
| `depth_bp` | Depth from BlenderProc rendering |
| `rgbd` | RGB-D images |
| `kinect` | Simulated Kinect depth |
| `image` | RGB images |
| `partial` | Planar-cut partial point cloud |
| `depth_like` | Depth-like partial from full cloud |
| `render_*` | On-the-fly rendering variants |

Mixed input types can be specified as a list with probabilities:

```yaml
inputs:
  type: [pointcloud, depth]    # Randomly choose between types
  type_p: [0.9, 0.1]          # 90% pointcloud, 10% depth
```

### Query Points Configuration

```yaml
points:
  dim: 3                        # Point dimension
  min_num_points: 1
  max_num_points: null
  nerf: false                   # NeRF positional encoding
  voxelize: null                # Voxel grid resolution
  bbox: false                   # Include bounding box
  cache: false                  # Cache loaded points
  permute: false                # Permute point order
  data_dir: null                # Custom data directory

  # Loading
  load_all: true                # Load all available points files
  load_random: false            # Random subset each epoch
  load_surface: false           # Load surface points
  load_uncertain: false         # Load uncertainty labels
  subsample: true               # Subsample to train.num_query_points
  crop: false                   # Crop to input bounds
  from_mesh: false              # Sample from mesh on-the-fly
  from_pointcloud: false        # Sample from point cloud

  # Sampling ratio
  in_out_ratio: null            # Inside/outside point ratio
```

### Surface Point Cloud Configuration

```yaml
pointcloud:
  num_points: null              # Points to load (global default)
  train:
    num_points: ${pointcloud.num_points}
  val:
    num_points: 100000
  test:
    num_points: ${pointcloud.val.num_points}
  min_num_points: 1
  max_num_points: null
  normals: false                # Load normals
  bbox: false                   # Include bounding box
  cache: false
  data_dir: null
  nerf: false
  voxelize: null
  permute: false
  from_mesh: false              # Sample from mesh on-the-fly
  fps:
    num_points: null            # Farthest point sampling
    method: gpu
```

### Training Configuration

```yaml
train:
  epochs: 10                    # Maximum epochs
  batch_size: 32                # Training batch size
  num_query_points: 2048        # Query points per sample
  num_views: 1                  # Number of input views

  # Optimizer
  optimizer: AdamW              # AdamW|Adam|SGD|AdamW8bit
  lr: 3.125e-6                  # Learning rate (1e-4 at batch 32)
  min_lr: null                  # Minimum LR for scheduler
  weight_decay: 0.01            # L2 regularization
  betas: [0.9, 0.999]           # Adam betas
  scale_lr: true                # Scale LR with batch size and GPUs
  find_lr: false                # Auto learning rate finder

  # Scheduler
  scheduler: null               # LinearWarmupCosineAnnealingLR|ReduceLROnPlateau|StepLR
  warmup_frac: 0.0033           # Warmup fraction of total steps
  lr_reduction_factor: 0.5      # ReduceLROnPlateau factor
  lr_step_size: 10              # StepLR step size
  lr_gamma: 0.9                 # StepLR gamma

  # Training options
  loss: null                    # Loss function override
  reduction: mean               # Loss reduction
  gradient_clip_val: null       # Gradient clipping value
  accumulate_grad_batches: 1    # Gradient accumulation steps
  precision: 32-true            # 32-true|16-mixed|bf16-mixed
  hypergradients: false         # Hypergradient optimization
  load_surface_points: null     # Load surface points for training
  num_batches: null             # Limit training batches per epoch
  shuffle: true                 # Shuffle training data

  # Checkpointing
  resume: false                 # Resume from checkpoint
  skip: false                   # Skip training entirely

  # Early stopping
  early_stopping: false
  model_selection_metric: val/f1
  patience_factor: 10           # Patience = epochs / factor

  # Debug
  overfit_batches: false        # Overfit on N batches
  fast_dev_run: false           # Quick sanity run
  detect_anomaly: false         # Detect NaN/Inf gradients
  no_aug: false                 # Disable all augmentation
  show: false                   # Show batch visualization
  find_batch_size: false        # Auto batch size finder
```

### Validation Configuration

```yaml
val:
  batch_size: ${train.batch_size}
  num_query_points: 100000      # More points for accurate evaluation
  load_surface_points: null
  freq: 1                       # Validate every N epochs (fractional OK)
  num_batches: ${train.num_batches}
  num_sanity: 2                 # Sanity validation steps at start
  reduction: ${train.reduction}
  precision: ${train.precision}
  shuffle: false

  # Visualization
  visualize: false              # Enable visualization callback
  vis_n_eval: 5                 # Visualize every N validations
  vis_n: null                   # Total samples to visualize
  vis_n_category: 4             # Samples per category

  # Evaluation
  mesh: false                   # Mesh metrics (or "fid" for FID score)
  voxels: false                 # Evaluate on voxels
  no_aug: false                 # Disable augmentation during validation
```

### Test Configuration

```yaml
test:
  batch_size: null
  num_query_points: null
  load_surface_points: ${val.load_surface_points}
  run: false                    # Run test after training
  split: test                   # Split to evaluate
  dir: null                     # Output directory
  filename: null                # Output filename
  overwrite: false              # Overwrite existing results
  merge: false                  # Merge results from shards
  basic: true                   # Basic metrics only
  no_aug: ${val.no_aug}
  reduction: ${val.reduction}
  metrics: null                 # Specific metrics to compute
  precision: ${val.precision}
  eval_meshes: true             # Evaluate extracted meshes
  num_instances: null            # Limit instances to evaluate
  shuffle: false
```

### Visualization Configuration

```yaml
vis:
  split: train                  # Split to visualize
  inputs: true                  # Show inputs
  occupancy: true               # Show occupancy field
  points: false                 # Show query points
  pointcloud: false             # Show surface point cloud
  voxels: false                 # Show voxels
  mesh: false                   # Show mesh
  box: true                     # Show bounding box
  bbox: false                   # Show axis-aligned bbox
  cam: false                    # Show camera
  frame: true                   # Show coordinate frame
  show: false                   # Open interactive viewer
  use_loader: false             # Use DataLoader
  save: false                   # Save visualizations to disk
  num_query_points: 2097152     # 128^3 for marching cubes
  resolution: 128               # Marching cubes resolution
  upsampling_steps: null
  refinement_steps: 0
  normals: false
  colors: false
  simplify: null                # Mesh simplification target
  renderer: cycles              # Blender renderer
  num_instances: null
  render: null                  # Render settings
  index: null                   # Specific sample index
```

### Augmentation Configuration

```yaml
aug:
  # Geometric
  rotate: null                  # Rotation axes: z|xyz|cam|null
  principal_rotations: false    # Use 90-degree principal rotations
  scale: null                   # Scale range or factor
  translate: null               # Translation range
  resize: null                  # Resize augmentation

  # Point cloud noise
  noise: null                   # Gaussian noise std
  clip_noise: null              # Clip noise magnitude
  downsample: null              # Downsampling method
  remove_outlier: false         # Remove statistical outliers

  # Depth-specific
  edge_noise: null              # Edge noise parameters
  border_noise: false           # Image border noise
  remove_angle: false           # Remove by surface incidence angle

  # View sampling
  upper_hemisphere: true        # Constrain camera to upper hemisphere
  angle_from_index: false       # Deterministic view angle from index

  # Cropping / occlusion
  cut_plane: null               # Planar cutting parameters
  cut_sphere: null              # Spherical cutting
  move_sphere: null             # Sphere-based occlusion
  bbox_jitter: null             # Bounding box jitter
  voxel_size: null              # Voxel size for downsampling
  flip: true                    # Random flipping
```

### Implicit Function Configuration

```yaml
implicit:
  threshold: 0.5                # Occupancy decision threshold
  sdf: false                    # Use SDF instead of occupancy

  # DVR (Differentiable Volume Rendering)
  dvr: false                    # Enable DVR wrapper
  near: 0.2                    # Near plane distance
  far: 2.4                     # Far plane distance
  step_func: equal              # Step function: equal|linear|exponential
  num_steps: [16, 32, 64, 128] # Progressive ray marching steps
  num_pixels: 1024              # Pixels per batch
  max_batch_size: 8192          # Max query batch size

  # Surface refinement
  num_refine_steps: 8
  refine_mode: secant           # secant|bisection
  crop: true                    # Crop queries to bounds
```

### Classification and Segmentation

```yaml
cls:
  occupancy: true               # Classify based on occupancy
  num_classes: null              # Number of classes
  weight: 0                     # Classification loss weight

seg:
  inputs: false                 # Segment inputs
  points: false                 # Segment query points
  pointcloud: false             # Segment surface point cloud
  mesh: false                   # Segment mesh
  num_classes: null
  weight: 0                     # Segmentation loss weight
```

### Logging Configuration

```yaml
log:
  freq: 10                      # Log every N steps
  verbose: false                # Verbose output
  progress: rich                # Progress bar: rich|tqdm|false
  pretty: false                 # Pretty print config

  # Weights & Biases
  wandb: false                  # Enable W&B
  offline: false                # Offline mode
  project: ${hydra:job.config_name}  # W&B project (defaults to config name)
  name: ${model.arch}           # W&B run name (defaults to architecture)
  id: null                      # Resume run by ID
  version: null                 # Version override

  # W&B extras
  gradients: false              # Log gradient histograms
  parameters: false             # Log parameter histograms
  graph: false                  # Log model graph
  model: false                  # Log model checkpoints to W&B
  profile: false                # Enable profiling
  metrics: null                 # Custom metrics to log

  # Checkpointing
  top_k: 1                     # Keep top K checkpoints
  summary_depth: 2             # Model summary depth
```

### DataLoader Configuration

```yaml
load:
  num_workers: -1               # DataLoader workers (-1 = auto)
  pin_memory: true              # Pin memory for GPU transfer
  prefetch_factor: 2            # Prefetch batches per worker
  precision: 32                 # Data loading precision
  res_modifier: 2               # Resolution modifier
  weighted: null                # Weighted sampling strategy
  hdf5: false                   # Use HDF5 data format

  keys_to_keep:                 # Filter batch dict to these keys
    - index
    - inputs
    - inputs.path
    - inputs.name
```

### File Paths Configuration

```yaml
files:
  pointcloud: null              # Surface point cloud filename
  mesh: null                    # Mesh filename
  normals: null                 # Normals filename
  voxels: null                  # Voxels filename

  points:                       # Query points filenames (per split)
    train: null                 # Single file or list of files
    val: ${files.points.train}
    test: ${files.points.val}

  split:                        # Custom train/val/test split files
    train: null
    val: null
    test: null

  suffix: ""                    # File suffix filter
```

### Normalization Configuration

```yaml
norm:
  padding: 0.1                  # Mesh bounding box padding
  center: null                  # Center method: null|true|xyz
  scale: false                  # Scale to unit cube/sphere
  method: cube                  # Normalization shape: cube|sphere
  reference: inputs             # Reference data for normalization
  offset: null                  # Manual offset
  to_front: false               # Rotate canonical front to -Z
  true_height: false            # Use true object height
  scale_factor: 1.0             # Additional scale multiplier
  bounds: [-0.5, 0.5]           # Normalization bounds
```

### Miscellaneous

```yaml
script: main                    # Entry point script
misc:
  seed: 0                       # Random seed
```

## Creating a New Config

### Minimal Example

```yaml
# conf/my_experiment.yaml
defaults:
  - config
  - _self_

data:
  train_ds:
    - shapenet_v1_fused

model:
  arch: onet

train:
  epochs: 100
  lr: 1e-4
```

### Inheriting from an Existing Experiment

To create a variant of an existing config, inherit from it instead of `config`:

```yaml
# conf/my_depth_experiment.yaml
defaults:
  - shapenet_v1_depth        # Inherit from shapenet_v1_depth (which inherits from shapenet_v1 -> config)
  - _self_

model:
  arch: conv_onet_grid

train:
  epochs: 200
  precision: 16-mixed
```

### Full Example with Overrides

```yaml
# conf/shapenet_depth_transformer.yaml
defaults:
  - config
  - _self_

data:
  train_ds:
    - shapenet_v1_fused
  categories: [chair, table, sofa]

model:
  arch: point_transformer
  norm: layer
  activation: gelu
  compile: true

inputs:
  type: depth
  project: true
  num_points: 2048

points:
  load_surface: true
  crop: true

train:
  epochs: 200
  batch_size: 16
  lr: 1e-4
  scheduler: LinearWarmupCosineAnnealingLR
  warmup_frac: 0.05
  precision: 16-mixed

aug:
  rotate: z
  noise: 0.005
  scale: [0.95, 1.05]

log:
  wandb: true
  project: shapenet-transformer
```

## Command Line Overrides

```bash
# Override single value
train -cn shapenet_v1 model.arch=conv_onet

# Override nested value
train -cn shapenet_v1 train.optimizer=Adam train.lr=1e-3

# Override list
train -cn shapenet_v1 data.categories=[chair,table]

# Add key not in schema
train -cn shapenet_v1 ++model.custom_flag=true

# Remove a key
train -cn shapenet_v1 ~aug.noise

# Override directory config
train -cn shapenet_v1 +dirs=dlr

# Multi-run sweep
train -cn shapenet_v1 model.arch=onet,conv_onet,if_net --multirun
```

## Config Patterns

### Variable Interpolation

Hydra supports `${...}` interpolation to reference other config values:

```yaml
val:
  batch_size: ${train.batch_size}       # Same as training batch size
  reduction: ${train.reduction}         # Same loss reduction
test:
  precision: ${val.precision}           # Chains through val -> train
```

### Per-Split File Configuration

Query point files and surface point clouds can differ across splits:

```yaml
files:
  points:
    train:
      - samples/surface_random.npz
      - samples/uniform_sphere.npz
    val:
      - samples/uniform_random.npz     # Denser, uniform sampling for evaluation
    test: ${files.points.val}           # Test uses same files as val
```

### Environment-Specific Overrides

```bash
# Development: fast iteration
train -cn shapenet_v1 train.fast_dev_run=true log.wandb=false

# Debug: overfit a single batch with anomaly detection
train -cn shapenet_v1 train.overfit_batches=1 train.detect_anomaly=true

# Production: mixed precision, compilation, full logging
train -cn shapenet_v1 train.precision=16-mixed model.compile=true log.wandb=true
```

## Common Configurations

| Config | Parent | Dataset | Input | Use Case |
|--------|--------|---------|-------|----------|
| `shapenet_v1` | `config` | ShapeNet v1 | Point cloud | Standard occupancy training |
| `shapenet_v1_depth` | `shapenet_v1` | ShapeNet v1 | Depth | Depth-based completion |
| `shapenet_v1_vae` | `shapenet_v1` | ShapeNet v1 | Point cloud | VAE training |
| `shapenet_v1_ldm` | `shapenet_v1_vae_for_ldm` | ShapeNet v1 | Point cloud | Latent diffusion |
| `shapenet_v1_cls` | `shapenet_v1` | ShapeNet v1 | Point cloud | Shape classification |
| `shapenet_v1_seg_*` | `shapenet_v1` | ShapeNet v1 | Various | Part segmentation |
| `shapenet_v1_autoenc` | `shapenet_v1` | ShapeNet v1 | Point cloud | Autoencoder |
| `completion3d` | `config` | Completion3D | Partial PCD | Benchmark evaluation |
| `modelnet40` | `config` | ModelNet40 | Point cloud | Classification benchmark |
| `tabletop` | `config` | TableTop | RGB-D | Scene completion |
| `tabletop_inst_seg*` | `tabletop` | TableTop | RGB-D | Instance segmentation |
| `dtu_dvr` | `config` | DTU | Multi-view | Differentiable volume rendering |
| `mugs` | `shapenet` | Custom mugs | Depth (BP) | Sim-to-real mug completion |
| `automatica*` | various | Custom | Kinect/depth | Real-world demos |
| `cvpr_2025` | `shapenet_v1_ldm` | ShapeNet v1 | Point cloud | CVPR 2025 diffusion experiments |
| `overfit` | `config` | Single mesh | Point cloud | Debug / sanity check |
| `*_diffusion` | `config` | MNIST/CIFAR | Image | 2D diffusion baselines |
