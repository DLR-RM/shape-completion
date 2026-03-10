# shape-completion-utils

Shared utility functions and classes used across all shape-completion submodules. Provides I/O, tensor operations, coordinate transforms, camera geometry, logging, configuration helpers, and visualization utilities.

**65+ public functions and classes** across four modules: `utils.py`, `logging.py`, `voxelizer.py`, and `runtime.py`.

## Installation

```bash
# As submodule (required by all other submodules)
```

## Quick Start

```python
from utils import (
    load_mesh, save_mesh,
    to_tensor, to_numpy,
    normalize_mesh,
    setup_logger,
    setup_config,
)

# Load and process mesh
vertices, faces = load_mesh("model.obj")
mesh = trimesh.Trimesh(vertices, faces)
mesh = normalize_mesh(mesh)
save_mesh("normalized.ply", mesh.vertices, mesh.faces)

# Tensor conversions
tensor = to_tensor(numpy_array, device="cuda")
array = to_numpy(tensor)
```

## Architecture

```
utils/
├── __init__.py              # Re-exports everything from src/
├── src/
│   ├── __init__.py          # Aggregates exports from all modules
│   ├── utils.py             # Core utility functions (63 functions, 1 class)
│   ├── logging.py           # Logger with custom debug levels
│   ├── voxelizer.py         # Voxelizer class (simple, kdtree, open3d backends)
│   └── runtime.py           # Optional dependency checks and warning suppression
└── tests/
    ├── test_utils.py
    └── test_runtime_logging.py
```

## Function Reference

### I/O Functions

| Function | Description |
|----------|-------------|
| `load_mesh(path, load_with=)` | Load mesh vertices and faces; auto-falls-back through trimesh, Open3D, PyMeshLab |
| `save_mesh(path, vertices, faces, colors=, normals=)` | Save mesh to `.obj`, `.ply`, `.off`, or `.stl` with optional vertex colors/normals |
| `load_from_binary_hdf5(path, file_names, file_dirs=)` | Load `.npy`, `.npz`, mesh, or image data packed inside an HDF5 container |
| `save_command_and_args_to_file(path, args=)` | Persist the CLI invocation and parsed arguments to a text file |

```python
from utils import load_mesh, save_mesh

# Load mesh (auto-detects format, falls back through backends)
vertices, faces = load_mesh("model.obj")
vertices, faces = load_mesh("model.ply", load_with="trimesh")

# Save mesh with vertex colors
save_mesh("output.ply", vertices, faces, colors=vertex_colors)
```

### Tensor and Array Operations

| Function | Description |
|----------|-------------|
| `to_tensor(x, unsqueeze=, device=)` | Convert scalar, list, ndarray, or Tensor to a CUDA/CPU Tensor (recursively handles lists) |
| `to_numpy(x, squeeze=)` | Convert Tensor to ndarray (handles GPU, detach, squeeze); passes through non-tensors |
| `to_scalar(x)` | Extract a Python scalar from any 0-d Tensor or ndarray |
| `unsqueeze_as(tensor, other)` | Unsqueeze trailing dims of `tensor` to match `other`'s ndim |
| `normalize(val, a=, b=, p_min=, p_max=)` | Min-max normalize to `[a, b]` with optional percentile clipping; works on both Tensor and ndarray |
| `filter_dict(d, keep=, remove=)` | Filter a dict by whitelisted or blacklisted key sets |
| `subsample_indices(data, n, replace=)` | Random subsample of `n` indices from `data`, with optional replacement |
| `binary_from_multi_class(logits, free_label=, occ_label=)` | Convert multi-class logits to binary (free vs. occupied) via log-sum-exp |

```python
from utils import to_tensor, to_numpy, normalize

# NumPy -> Tensor (auto-unsqueezes batch dim, moves to CUDA)
tensor = to_tensor(array, device="cuda")

# Tensor -> NumPy (handles GPU tensors automatically)
array = to_numpy(tensor)

# Normalize with percentile clipping
normed = normalize(depth_values, a=0, b=1, p_min=0.01, p_max=0.99)
```

### Coordinate Transforms

| Function | Description |
|----------|-------------|
| `inv_trafo(trafo)` | Invert a `(4,4)` or `(B,4,4)` homogeneous transform; uses `R^T` when rotation is orthonormal, else full inverse |
| `apply_trafo(points, trafo)` | Apply a `(4,4)` or `(B,4,4)` rigid transform to `(N,3)` or `(B,N,3)` points |
| `look_at(eye, target, up=)` | Compute a 4x4 camera pose (inverse extrinsic) in OpenGL convention |
| `convert_coordinates(pts, in_fmt, out_fmt)` | Convert 3D point coordinates between `"opencv"`, `"opengl"`, and `"blender"` conventions |
| `convert_extrinsic(ext, in_fmt, out_fmt)` | Convert a 4x4 camera extrinsic between coordinate conventions |
| `pitch_from_trafo(trafo, roll=, degrees=)` | Extract pitch angle (x-axis rotation) from a 4x4 transform, correcting for gimbal lock |
| `rot_from_euler(axes, upper_hemisphere)` | Random Euler rotation matrix with optional upper-hemisphere constraint |

```python
from utils import look_at, apply_trafo, inv_trafo, convert_coordinates
import numpy as np

# Camera look-at matrix (OpenGL convention)
cam_pose = look_at(eye=np.array([0, 0, 2]), target=np.array([0, 0, 0]))

# Transform points from camera to world frame
points_world = apply_trafo(points_cam, inv_trafo(cam_pose))

# Convert between coordinate conventions
points_gl = convert_coordinates(points_cv, "opencv", "opengl")
```

Supported coordinate conventions:
- **OpenCV**: x-right, y-down, z-forward
- **OpenGL**: x-right, y-up, z-backward
- **Blender**: x-right, y-forward, z-up

### Camera and Intrinsic Utilities

| Function | Description |
|----------|-------------|
| `invert_intrinsic(K)` | Analytically invert a 3x3 intrinsic matrix (fast path for zero-skew case) |
| `adjust_intrinsic(K, width, height, box=, size=)` | Adjust intrinsic matrix after crop and/or resize; supports batched `K` |
| `draw_camera(intrinsic, extrinsic, width, height, scale=, color=)` | Create Open3D geometry (frustum + axes) for camera visualization |

```python
from utils import adjust_intrinsic, draw_camera

# Adjust intrinsic after cropping and resizing
K_new = adjust_intrinsic(K, width=640, height=480, box=(100, 50, 540, 430), size=256)

# Visualize camera frustum
camera_geoms = draw_camera(K, extrinsic_cv, width=640, height=480, scale=0.1)
```

### Depth and Point Cloud Operations

| Function | Description |
|----------|-------------|
| `depth_to_points(depth, intrinsic, depth_scale=, depth_trunc=)` | Unproject a depth map to 3D points; supports both Tensor and ndarray |
| `points_to_depth(points, intrinsic, width, height)` | Project 3D points to a depth map image |
| `points_to_uv(points, intrinsic, extrinsic=, width=, height=)` | Project 3D points to pixel coordinates `(u, v)`; optionally returns valid-pixel mask |
| `depth_to_image(depth, cmap=, a=, b=)` | Convert a depth map to a colormapped PIL Image for visualization |

```python
from utils import depth_to_points, points_to_uv, depth_to_image
import numpy as np

# Unproject depth to point cloud
intrinsic = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])
points = depth_to_points(depth_map, intrinsic)

# Project points to image coordinates
u, v, mask = points_to_uv(points, intrinsic, width=640, height=480)

# Visualize depth with turbo colormap
depth_vis = depth_to_image(depth_map, cmap="turbo")
```

### Ray and Frustum Operations

| Function | Description |
|----------|-------------|
| `get_rays(mask, intrinsic, extrinsic, normalize=, num_samples=)` | Generate ray origins and directions for masked pixels (for NeRF-style sampling) |
| `sample_distances(n_points, near, far, n_steps=, method=, uniform_in_volume=)` | Sample depths along rays (linear or random, with optional volume-uniform cubic CDF) |
| `is_in_frustum(points, intrinsic, extrinsic, width, height, near=, far=)` | Boolean mask for which points lie inside a camera frustum |

```python
from utils import get_rays, sample_distances

# Generate rays from masked pixels
ray_origins, ray_dirs, u, v = get_rays(mask, K, extrinsic, normalize=True)

# Sample distances along rays (uniform in volume)
distances = sample_distances(n_points=1024, near=0.2, far=2.4, n_steps=64, uniform_in_volume=True)
```

### Grid and Voxel Operations

| Function | Description |
|----------|-------------|
| `make_3d_grid(bb_min, bb_max, shape)` | Create a dense 3D query grid as `(N, 3)` Tensor; supports per-axis resolution |
| `points_to_coordinates(points, max_value=, plane=, intrinsic=)` | Convert 3D points to normalized `[0,1)` coordinates (optionally project to xy/xz/yz/uv) |
| `coordinates_to_index(coordinates, resolution)` | Convert `[0,1]` coordinates to flat row-major voxel indices |
| `Voxelizer(resolution=, padding=, method=, bounds=)` | Class for voxelizing point clouds via `simple`, `kdtree`, or `open3d` backends |

```python
from utils import make_3d_grid, Voxelizer

# Create query grid for implicit function evaluation
grid = make_3d_grid(bb_min=-0.5, bb_max=0.5, shape=128)  # (128**3, 3)

# Voxelize a point cloud
voxelizer = Voxelizer(resolution=64, bounds=(-0.5, 0.5), method="simple")
occupancy, voxel_indices = voxelizer(points)  # occupancy: (64, 64, 64)
centers = voxelizer.grid_points                # (64**3, 3)
```

### Mesh Operations

| Function | Description |
|----------|-------------|
| `normalize_mesh(mesh, center=, scale=, cube_or_sphere=)` | Normalize a trimesh to unit cube or unit sphere |
| `get_points(n_views=)` | Generate `n` evenly-distributed points on a unit sphere (Fibonacci spiral) |
| `generate_random_basis(n_points=, n_dims=, radius=, seed=)` | Sample points uniformly inside a ball |

```python
from utils import normalize_mesh, get_points
import trimesh

mesh = trimesh.load("model.obj")
mesh = normalize_mesh(mesh, cube_or_sphere="sphere")

# 100 viewpoints evenly distributed on a sphere
viewpoints = get_points(n_views=100)
```

### Image Utilities

| Function | Description |
|----------|-------------|
| `stack_images(images)` | Stack a list of images into a square grid (auto-computes rows/cols) |
| `crop_and_resize_image(img, box=, size=, color=, interpolation=)` | Crop to bounding box and/or resize (largest edge to `size`); handles ndarray, PIL, and Tensor |
| `bbox_from_mask(mask, padding=)` | Compute a square bounding box from a binary mask with optional padding |

```python
from utils import stack_images, bbox_from_mask, crop_and_resize_image

# Create 2x2 image grid
grid = stack_images([img1, img2, img3, img4])

# Crop to object region
bbox = bbox_from_mask(segmentation_mask, padding=0.1)
cropped = crop_and_resize_image(image, box=bbox, size=256)
```

### Configuration and Experiment Management

| Function | Description |
|----------|-------------|
| `setup_config(cfg, seed_workers=, functions=)` | Initialize config: detect GPU capabilities, set precision, seed, log level, run validators |
| `check_cfg(cfg)` | Validate config consistency (point loading, NeRF flags, SDF thresholds, etc.) |
| `resolve_save_dir(cfg)` | Compute experiment save directory from config (`dirs.log / project / name`) |
| `resolve_backup_dir(cfg)` | Compute backup directory path |
| `resolve_weights_path(cfg, weights_path=)` | Search multiple candidate paths for a model weights file |
| `resolve_checkpoint_path(cfg)` | Search for a checkpoint `.ckpt` file across save/backup dirs |
| `config_hash(cfg, length=, ignore=)` | Deterministic MD5 hash of config (ignoring runtime/logging keys) for caching |
| `TrackingDictConfig` | DictConfig subclass that records which keys are accessed (for detecting unused config) |

```python
from utils import setup_config, resolve_save_dir, config_hash

cfg = setup_config(cfg)
save_dir = resolve_save_dir(cfg)
run_hash = config_hash(cfg, length=8)  # e.g. "a3f1c02b"
```

### Logging

| Symbol | Description |
|--------|-------------|
| `setup_logger(name)` | Create a `ShapeCompletionLogger` with rank-zero-only output (for distributed training) |
| `set_log_level(level)` | Set log level on all loggers created via `setup_logger` |
| `DEBUG_LEVEL_1`, `DEBUG_LEVEL_2` | Custom log levels between `DEBUG` (10) and `INFO` (20): level 12 and 11 |

```python
from utils import setup_logger, set_log_level

logger = setup_logger(__name__)
logger.info("Training started")
logger.debug_level_1("Detailed debug info")   # Only shown at verbose >= 1
logger.debug_level_2("Very detailed debug")   # Only shown at verbose >= 2

set_log_level("DEBUG")  # Enable all debug output
```

### Path Utilities

| Function | Description |
|----------|-------------|
| `resolve_path(path)` | Expand `~` and resolve to absolute `Path` |
| `git_show_toplevel(cwd=)` | Get the root of the current git repo |
| `git_show_superproject_working_tree(cwd=)` | Get the superproject root (for submodules) |
| `get_git_root()` | Get the top-level project root (superproject if inside a submodule) |
| `git_submodule_path(name_or_path)` | Resolve a submodule name or file path to its git root |
| `resolve_out_dir(in_path, in_dir, out_dir, shard=)` | Compute an output path mirroring the input directory structure |
| `eval_input(inputs, in_format=, recursion_depth=)` | Glob/resolve input file paths; shuffles by default |

```python
from utils import git_submodule_path, resolve_path, eval_input

models_path = git_submodule_path("models")
full_path = resolve_path("~/data/meshes")
mesh_files = eval_input(Path("data/"), in_format=".obj", sort=True)
```

### Decorators

| Decorator | Description |
|-----------|-------------|
| `@measure_runtime` | Log function execution time; also usable as context manager. Supports custom log levels and CUDA sync. |
| `@default_on_exception(default, exceptions=)` | Return a default value on exception; logs the error; clears CUDA cache on OOM |
| `@get_args(print_args=)` | Capture and optionally print function arguments as a dict |

```python
from utils import measure_runtime, default_on_exception

@measure_runtime
def train_epoch():
    ...  # Logs: "train_epoch takes 45.2000s"

# Also works as a context manager
with measure_runtime("data loading"):
    load_data()  # Logs: "data loading took 1.2345s"

@default_on_exception(default=[], exceptions=(FileNotFoundError, ValueError))
def load_data(path):
    ...  # Returns [] on FileNotFoundError or ValueError
```

### Hardware and Distributed Utilities

| Function | Description |
|----------|-------------|
| `count_gpus()` | Count available GPUs via `nvidia-smi` |
| `get_num_workers(n=)` | Auto-determine DataLoader workers (reads `SLURM_CPUS_PER_TASK`; disables multithreading if workers > 1) |
| `get_device_info()` | Get lowest GPU compute capability; sets float32 matmul precision to `"high"` on capability > 7 |
| `is_distributed()` | Check if running in a distributed (multi-GPU) setup |
| `disable_multithreading()` | Set `OPENBLAS/MKL/OMP/TBB_NUM_THREADS=1` |

```python
from utils import get_num_workers, is_distributed

num_workers = get_num_workers()  # Auto from SLURM or CPU count
if is_distributed():
    ...  # Multi-GPU logic
```

### Runtime and Dependency Management

| Function | Description |
|----------|-------------|
| `suppress_known_optional_dependency_warnings()` | Suppress xFormers and Triton redefinition warnings |
| `log_optional_dependency_summary(logger, cfg=)` | Log which optional dependencies (xformers, bpy, torch_cluster, etc.) are available vs. needed |

### Miscellaneous

| Function | Description |
|----------|-------------|
| `cosine_anneal(start, stop, steps, current_step)` | Cosine annealing schedule value |
| `resolve_dtype(precision, integer=, unsigned=)` | Map numeric precision (8/16/32/64) to NumPy dtype |
| `monkey_patch(instance, method_name, new_method)` | Replace a method on an object instance at runtime |
| `stdout_redirected(to=, enabled=)` | Context manager to redirect all stdout (including C-level) to a file or `/dev/null` |

```python
from utils import stdout_redirected, cosine_anneal
import os

# Suppress noisy library output
with stdout_redirected(to=os.devnull):
    noisy_library_call()

# Cosine annealing
lr = cosine_anneal(start=1e-3, stop=1e-5, steps=1000, current_step=500)
```

## Constants

```python
from utils import PARTNET_COLORS, PLOTLY_COLORS, DEBUG_LEVEL_1, DEBUG_LEVEL_2

# Color palettes for part segmentation visualization
colors = PARTNET_COLORS  # (50, 3) ndarray, PartNet segmentation colors
colors = PLOTLY_COLORS   # Plotly qualitative Pastel + Pastel1 + Pastel2 + Plotly colors

# Custom debug levels (between DEBUG=10 and INFO=20)
# DEBUG_LEVEL_2 = 11, DEBUG_LEVEL_1 = 12
```
