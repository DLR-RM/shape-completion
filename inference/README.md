# shape-completion-inference

Inference scripts for running trained shape completion models on real-world point cloud data. Supports depth images, point clouds, and includes preprocessing utilities for plane removal and object segmentation.

## Installation

```bash
# As submodule

# Dependencies (from main repo)
uv sync --extra inference
```

## Quick Start

```bash
# Basic inference on point cloud
inference data.ply onet -w path/to/model_best.pt

# From depth image with camera parameters
inference depth.png conv_onet -w model.pt --intrinsic 582.7 582.7 320.8 245.3

# With plane removal and visualization
inference scene.ply onet -w model.pt --remove_plane --show
```

## Inference vs Evaluation

The `inference` command and the `evaluate` command serve different purposes:

- **`inference`** runs a trained model on arbitrary real-world input (depth images, point clouds, scene captures). It handles camera unprojection, plane removal, clustering, and normalization -- everything needed to go from raw sensor data to a completed mesh. No dataset structure or ground truth is required.
- **`evaluate`** (`eval`) runs on structured dataset splits (train/val/test) using Hydra configs. It expects pre-processed data in the dataset format and computes metrics across the full split. Use `evaluate` for benchmarking; use `inference` for deploying on new data.

The `inference` command can optionally compute metrics against a ground truth mesh via `--mesh` and `--eval`, but this is per-file rather than dataset-wide.

## Architecture

```
inference/
├── __init__.py
├── conf/                         # Inference-specific configs
├── scripts/
│   ├── inference_pointcloud.py   # Main inference script (entry point)
│   └── inference_mesh_tutorial.py # Tutorial: mesh -> partial -> completion
├── src/
│   └── utils.py                  # Preprocessing utilities
└── tests/
```

## Main Script: inference_pointcloud

### Basic Usage

```bash
inference <input> <model> -w <weights> [options]

# Required arguments:
#   input           Path to point cloud file or directory
#   model           Model architecture (case-insensitive, see below)
#   -w/--weights    Path to model weights (.pt)      # mutually exclusive
#   -c/--checkpoint Path to Lightning checkpoint (.ckpt) # with -w
```

### Supported Model Architectures

The `model` argument selects the architecture. The following are supported for inference mesh generation:

| Architecture | `model` argument | Output |
|---|---|---|
| Occupancy Network | `onet` | Mesh (occupancy) |
| Convolutional Occupancy Network | `conv_onet`, `conv_onet_grid` | Mesh (occupancy) |
| IF-Net | `if_net` | Mesh (occupancy) |
| VQDIF | `vqdif` | Mesh (occupancy) |
| 3D Shape2VecSet | `3dshape2vecset` | Mesh (occupancy) |
| 3D Shape2VecSet VAE | `3dshape2vecset_vae` | Mesh (occupancy) |
| PCN | `pcn` | Point cloud |
| SnowflakeNet | `snowflakenet` | Point cloud |

Use `--sdf` if the model was trained with SDF supervision instead of occupancy.

### Input Formats

| Format | Extensions | Notes |
|--------|-----------|-------|
| Point cloud | `.ply`, `.obj`, `.off` | Loaded directly via Open3D |
| Depth image | `.png`, `.jpg` | Requires camera intrinsics; unprojected via Open3D |
| EXR depth | `.exr` | Requires `opencv-contrib-python` (`uv sync --extra exr`) |
| Kinect raw depth | `.npy`, `.npz` | Assumes Kinect v1 calibration; uses custom unprojection with disparity model |

**Important distinctions:**

- `.ply`/`.obj`/`.off` are loaded as 3D geometry directly. No camera parameters needed.
- `.png`/`.jpg`/`.exr` are treated as depth images and require `--intrinsic` for accurate unprojection. Without `--intrinsic`, default Kinect v1 parameters are used.
- `.npy`/`.npz` are treated as raw Kinect depth arrays and unprojected using a sensor-specific disparity model (see `unproject_kinect_depth` in `src/utils.py`). The `--intrinsic` flag is ignored for these formats since the Kinect calibration is built-in.

### Generation Options

```bash
--threshold 0.5      # Occupancy threshold (default: 0.5)
                     # For SDF models: auto-set to 0 unless explicitly overridden
--resolution 128     # Voxel grid resolution (default: 128)
--n_points -1        # Query points per batch (-1 = resolution^3)
--n_up -1            # MISE upsampling steps (-1 = auto: log2(resolution) - log2(32))
--padding 0.1        # Normalization padding (default: 0.1, must match training)
--sdf                # Model outputs SDF values instead of occupancy logits
```

### Camera Parameters

For depth image inputs (`.png`, `.jpg`, `.exr`), camera intrinsics control unprojection accuracy. The `--intrinsic` values are passed through to Open3D's RGBD unprojection via `easy_o3d`.

```bash
# As 4 values: fx fy cx cy
--intrinsic 582.7 582.7 320.8 245.3

# As 5 values: fx fy cx cy s (skew)
--intrinsic 582.7 582.7 320.8 245.3 0

# As 9 values: full 3x3 matrix (row-major)
--intrinsic 582.7 0 320.8 0 582.7 245.3 0 0 1
```

Camera extrinsic (pose) can also be specified. The rotation is decomposed into Euler angles and applied to orient the point cloud into a canonical frame:

```bash
# As 7 values (quaternion + translation): qw qx qy qz tx ty tz
--extrinsic 1 0 0 0 0 0 0

# As 9 values (rotation matrix, row-major)
# As 12 values (3x4 [R|t] matrix, row-major)
```

### Depth Processing

```bash
--depth_scale 1000.0   # Depth units (1000 = mm -> m, default: 1000)
--depth_trunc 1.1      # Truncate depth beyond this distance in meters (default: 1.1)
```

### Preprocessing Pipeline

When processing real-world sensor data, the preprocessing pipeline runs in this order:

1. **Load** -- Read input file and unproject to 3D if needed
2. **Crop** (`--pcd_crop`) -- Crop point cloud to a bounding box (default: `[-0.4, -0.5, -0.5, 0.4, 0.5, 0.5]`)
3. **Plane removal** (`--remove_plane`) -- RANSAC plane segmentation, keep points above plane
4. **Clustering** (`--cluster`) -- DBSCAN to isolate individual objects
5. **Outlier removal** -- Statistical and radius-based filtering
6. **Hull cropping** (`--crop`) -- Include plane points within object's convex hull
7. **Center and scale** -- Normalize to unit cube with padding
8. **Voxelize** -- Grid voxelization for IF-Net (automatic when model is IFNet)

### Plane Removal & Segmentation

For real-world scenes, remove table/ground plane and isolate objects:

```bash
inference scene.ply onet -w model.pt \
    --remove_plane \
    --distance_threshold 0.006 \
    --ransac_iterations 1000 \
    --cluster \
    --outlier_neighbors 50 \
    --outlier_radius 0.1
```

| Flag | Default | Description |
|------|---------|-------------|
| `--remove_plane` | off | Enable RANSAC plane segmentation |
| `--distance_threshold` | `0.006` | RANSAC inlier distance (meters) |
| `--ransac_iterations` | `1000` | RANSAC iterations |
| `--cluster` | off | DBSCAN clustering after plane removal |
| `--outlier_neighbors` | `50` | Neighbor count for outlier removal |
| `--outlier_radius` | `0.1` | Radius for radius-based outlier removal |
| `--outlier_std` | `10.0` | Std ratio for statistical outlier removal |
| `--on_plane` | off | Offset completed mesh to sit on the detected plane |
| `--crop` | off | Include plane points within the object's convex hull |
| `--crop_scale` | `1.0` | Scale factor for the convex hull used in cropping |
| `--up_axis` | `1` | Gravity axis for hull cropping (0=x, 1=y, 2=z) |

### Input/Output Options

```bash
-f .ply              # Input file extension filter (default: .ply)
--recursion_depth 2  # Directory recursion depth
-o output/           # Output directory
--sort               # Sort files before processing
--denormalize        # Denormalize output mesh to original scale/position
--pcd_crop -0.4 -0.5 -0.5 0.4 0.5 0.5  # Bounding box crop (6 floats)
```

### Evaluation Options

```bash
-m/--mesh gt_mesh.obj   # Ground truth mesh for comparison
-p/--pose 0 0 0.5       # Transform to align GT mesh (3, 4, 7, 9, or 12 values)
--eval                   # Compute metrics against ground truth
```

The `--pose`/`--transform` flag accepts multiple formats via `easy_o3d`: 3 values (translation), 4 (axis-angle), 7 (quaternion + translation), 9 (rotation matrix), or 12 (3x4 transform). If a `.npy` pose file matching the input filename pattern exists in the input directory, it is loaded automatically.

### Visualization & Miscellaneous

```bash
--show               # Show interactive 3D visualization
--verbose            # Debug logging with intermediate visualizations
--seed 0             # Random seed (default: 0)
--scale 1.0          # Fixed scale for input points (default: 1, uses auto-scale)
```

## Python API

### Load Model

```python
from argparse import Namespace
from pathlib import Path

from inference.scripts.inference_pointcloud import load_model

args = Namespace(
    model="conv_onet",
    weights=Path("model_best.pt"),
    checkpoint=None,
    sdf=False,
    padding=0.1,
)
model = load_model(args)  # Returns nn.Module on CUDA (if available), in eval mode
```

### Preprocessing Utilities

The preprocessing functions live in `inference.src.utils`:

```python
from pathlib import Path

from inference.src.utils import (
    get_point_cloud,
    remove_plane,
    get_input_data_from_point_cloud,
    unproject_kinect_depth,
)
```

#### Loading Point Clouds

`get_point_cloud` handles all supported input formats and returns an Open3D point cloud plus camera matrices:

```python
import numpy as np

# From a depth image with known intrinsics
pcd, intrinsic, extrinsic = get_point_cloud(
    in_path=Path("depth.png"),
    depth_scale=1000.0,
    depth_trunc=1.1,
    intrinsic=np.array([[582.7, 0, 320.8], [0, 582.7, 245.3], [0, 0, 1]]),
    extrinsic=None,
    pcd_crop=[-0.5, -0.5, -0.5, 0.5, 0.5, 0.5],
)

# From a PLY file (intrinsic/extrinsic ignored for mesh formats)
pcd, intrinsic, extrinsic = get_point_cloud(in_path=Path("scene.ply"))

# From raw Kinect depth (uses built-in Kinect v1 calibration)
pcd, intrinsic, extrinsic = get_point_cloud(in_path=Path("kinect_raw.npy"))
```

The `intrinsic` parameter accepts:
- `None` -- uses default Kinect v1 parameters
- `str` path to a `.npy`/`.npz` file containing the 3x3 matrix
- `np.ndarray` -- a 3x3 intrinsic matrix directly

#### Plane Removal

```python
# Remove table plane and cluster objects
objects, plane_model = remove_plane(
    pcd,
    distance_threshold=0.006,  # RANSAC inlier distance
    num_iterations=1000,       # RANSAC iterations
    cluster=True,              # DBSCAN clustering
    cluster_eps=0.075,         # DBSCAN neighborhood radius
    cluster_min_points=1000,   # DBSCAN minimum cluster size
    outlier_neighbors=50,      # Outlier removal neighbor count
    outlier_radius=0.1,        # Radius outlier removal
    outlier_std=10.0,          # Statistical outlier removal std ratio
    crop=True,                 # Include plane points under object hull
    crop_scale=1.0,            # Hull scale factor
    crop_up_axis=1,            # Gravity axis (0=x, 1=y, 2=z)
)
# objects: list of Open3D PointCloud (one per cluster, or one if cluster=False)
# plane_model: np.ndarray [a, b, c, d] plane coefficients (ax + by + cz + d = 0)
```

#### Preparing Model Input

`get_input_data_from_point_cloud` normalizes an Open3D point cloud to the unit cube and returns the data ready for the model:

```python
input_data, loc, scale = get_input_data_from_point_cloud(
    objects[0],              # Open3D PointCloud
    center=True,             # Center to bounding box midpoint
    scale=True,              # Normalize to unit scale (or pass a float for fixed scale)
    num_input_points=2048,   # Subsample to N points (None = keep all)
    crop=0.55,               # Crop to box of this half-width (None = no crop)
    padding=0.1,             # Padding for voxelization (must match training)
    voxelize=None,           # Grid resolution for voxelization (used by IFNet)
    noise_std=None,          # Add Gaussian noise (e.g. 0.005)
    rotate_z=None,           # Rotate around z-axis (radians)
    offset_y=0,              # Y offset before centering (for on-plane placement)
    transform=None,          # Additional rigid transform
)
# input_data: np.ndarray (N, 3) float32 -- normalized points
# loc: np.ndarray (3,) float32 -- center offset applied
# scale: float -- scale factor applied (for denormalization: mesh.apply_scale(scale))
```

### Full Inference Pipeline

```python
import numpy as np
import open3d as o3d
import torch
from argparse import Namespace
from pathlib import Path

from inference.scripts.inference_pointcloud import load_model
from inference.src.utils import get_input_data_from_point_cloud
from visualize import Generator

# Load model
args = Namespace(
    model="conv_onet",
    weights=Path("model_best.pt"),
    checkpoint=None,
    sdf=False,
    padding=0.1,
)
model = load_model(args)

# Load and preprocess point cloud
pcd = o3d.io.read_point_cloud("partial_object.ply")
input_data, loc, scale = get_input_data_from_point_cloud(
    pcd, center=True, scale=True, padding=0.1,
)

# Run inference
device = next(model.parameters()).device
inputs = torch.from_numpy(input_data).float().unsqueeze(0).to(device)

generator = Generator(model, threshold=0.5, padding=0.1, resolution=128)
grid, _, c = generator.generate_grid({"inputs": inputs})
mesh = generator.extract_mesh(grid, c)

# Denormalize to original coordinate frame
mesh.apply_scale(scale)
mesh.apply_translation(loc)
mesh.export("completed.ply")
```

## Command Line Examples

### Single Object Completion

```bash
# From a PLY point cloud
inference partial_chair.ply onet -w model.pt -o results/

# From a depth image (Realsense camera)
inference depth.png conv_onet -w model.pt \
    --intrinsic 615.3 615.3 320.0 240.0 \
    --depth_scale 1000 \
    -o results/

# Using a Lightning checkpoint instead of raw weights
inference input.ply conv_onet -c lightning_logs/version_0/checkpoints/last.ckpt \
    -o results/
```

### Tabletop Scene Processing

```bash
# Full pipeline: unproject depth, remove table, cluster objects, complete shapes
inference kinect_scene.ply conv_onet -w model.pt \
    --remove_plane \
    --distance_threshold 0.008 \
    --cluster \
    --on_plane \
    --denormalize \
    --show \
    -o completed_objects/
```

### Batch Processing

```bash
# Process all PLY files in a directory tree
inference data_dir/ onet -w model.pt \
    -f .ply \
    --recursion_depth 2 \
    --sort \
    -o results/
```

### With Ground Truth Comparison

```bash
# Evaluate against ground truth mesh
inference partial.ply onet -w model.pt \
    --mesh gt_mesh.obj \
    --pose 0 0 0.1 \
    --eval \
    --show
```

### SDF Model with High Resolution

```bash
inference input.ply conv_onet -w sdf_model.pt \
    --sdf \
    --resolution 256 \
    --n_up 3 \
    --threshold 0 \
    -o high_res/
```

## Output Files

When using `-o output/`, results are saved per input file. The output subdirectory name is the parent directory stem of the input file:

```
output/
└── <input_parent_stem>/
    ├── mesh.ply          # Completed mesh (for ONet/ConvONet/IFNet/VQDIF/Shape2VecSet)
    ├── pcd.ply           # Completed point cloud (for PCN/SnowflakeNet, only if no mesh)
    ├── inputs.ply        # Preprocessed input points (always saved)
    ├── plane.npy         # Plane model [a,b,c,d] (if --remove_plane was used)
    └── result.npy/.txt   # Evaluation metrics (if --eval was used)
```

Additionally, if `--remove_plane` is used, `object_<input_name>` is saved next to the input file containing the segmented object points.

## Troubleshooting

### Memory Issues

```bash
# Reduce grid resolution (fewer voxels = less memory)
--resolution 64

# Reduce query batch size (processes grid in smaller chunks)
--n_points 32768
```

### Poor Results

1. **Padding mismatch.** The `--padding` value must match what the model was trained with:
   ```bash
   --padding 0.1  # Check your training config's norm.padding
   ```

2. **Wrong architecture name.** The model argument must exactly match the architecture:
   ```bash
   # Common variants:
   inference input.ply conv_onet -w model.pt      # Standard ConvONet
   inference input.ply conv_onet_grid -w model.pt  # Grid-based ConvONet
   inference input.ply if_net -w model.pt          # IF-Net
   ```

3. **Threshold tuning.** For occupancy models, the threshold is applied in log-space. For SDF models, use 0:
   ```bash
   --threshold 0.3  # Lower = larger mesh (more aggressive)
   --threshold 0.7  # Higher = smaller mesh (more conservative)
   --sdf --threshold 0  # SDF: extract at zero level set
   ```

### Plane Removal Failing

```bash
# Increase inlier distance for noisy depth sensors
--distance_threshold 0.01

# More RANSAC iterations for challenging scenes
--ransac_iterations 2000

# Less aggressive outlier removal
--outlier_std 5.0
--outlier_radius 0.2
```
