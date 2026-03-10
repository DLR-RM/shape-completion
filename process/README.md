# shape-completion-process

Data processing and preprocessing scripts for shape completion. Includes watertight mesh generation, depth rendering with Kinect simulation, physics-based pose generation, uncertain region identification, SDF computation, and YCB photogrammetry reconstruction.

## Installation

```bash
# As submodule

# Dependencies (from main repo)
uv sync --extra process

# For BlenderProc rendering
uv sync --extra blenderproc
```

## Quick Start

```bash
# Generate watertight meshes via TSDF fusion
make_watertight path/to/ShapeNetCore.v1

# Render Kinect-style depth for a single mesh
render_kinect --in_file path/to/mesh.off

# Render Kinect-style depth for a directory of meshes in parallel
render_kinect_parallel path/to/meshes --out_dir path/to/output

# Find uncertain regions (Hydra config)
find_uncertain_regions -cn shapenet_uncertain

# Generate stable resting poses via physics simulation
find_pose path/to/meshes

# Add Kinect simulation to existing HDF5 files
add_kinect_sim --input-dir path/to/hdf5s --shapenet-dir path/to/shapenet
```

## Architecture

```
process/
├── __init__.py
├── scripts/
│   ├── make_watertight.py        # Watertight mesh generation (fuse/carve/fill)
│   ├── render_kinect.py          # Single-mesh Kinect depth simulation
│   ├── render_kinect_parallel.py # Parallel Kinect rendering over directories
│   ├── render_blenderproc.py     # BlenderProc scene rendering (no entry point)
│   ├── find_uncertain_regions.py # Uncertainty identification via Hydra
│   ├── generate_physics_poses.py # Physics-based pose generation (PyBullet)
│   ├── process_mesh.py           # Point sampling and mesh normalization
│   ├── sdf_from_mesh.py          # SDF computation and sampling
│   ├── add_kinect_sim.py         # Add Kinect sim to existing HDF5 data
│   ├── blender_load.py           # Blender scene import for exports
│   └── ycb_sfm.py                # YCB photogrammetry via COLMAP
├── src/
│   ├── __init__.py
│   ├── utils.py                  # Point cloud normalization, MeshLab filters
│   ├── fuse.py                   # TSDF fusion pipeline (libfusion)
│   ├── fill.py                   # Kaolin voxelization hole filling
│   └── carve.py                  # Open3D voxel carving
├── assets/
│   ├── meshlab_filter_scripts/   # MeshLab XML filter scripts
│   │   ├── clean.mlx             # Mesh cleaning filters
│   │   └── simplify.mlx          # Mesh simplification filters
│   └── sphere.ply                # Reference sphere mesh
└── tests/
```

### Registered Entry Points

These commands are available after `uv sync --extra process` (defined in `pyproject.toml`):

| Command | Script | Purpose |
|---------|--------|---------|
| `make_watertight` | `make_watertight.py` | Watertight mesh generation |
| `render_kinect` | `render_kinect.py` | Single-mesh Kinect depth rendering |
| `render_kinect_parallel` | `render_kinect_parallel.py` | Parallel Kinect depth rendering |
| `find_uncertain_regions` | `find_uncertain_regions.py` | Uncertain region identification |
| `find_pose` | `generate_physics_poses.py` | Physics-based stable pose generation |
| `add_kinect_sim` | `add_kinect_sim.py` | Add Kinect sim to HDF5 datasets |

Scripts without entry points are invoked via `python -m process.scripts.<name>`:

| Script | Purpose |
|--------|---------|
| `process_mesh.py` | Point cloud sampling and mesh normalization |
| `sdf_from_mesh.py` | SDF computation, marching cubes, and SDF-based sampling |
| `render_blenderproc.py` | BlenderProc multi-object scene rendering |
| `blender_load.py` | Import shape-completion exports into Blender |
| `ycb_sfm.py` | YCB object reconstruction via COLMAP photogrammetry |

## Scripts

### make_watertight -- Watertight Mesh Generation

Converts arbitrary triangle soups into watertight meshes. Supports three backends: TSDF fusion (default, requires `libfusion`), voxel carving (requires Open3D), and Kaolin hole filling (requires PyTorch + Kaolin). Based on [mesh-fusion](https://github.com/davidstutz/mesh-fusion).

**Modes:**

- `fuse` (default): Render depth maps from multiple views, fuse into TSDF volume, extract mesh via marching cubes. Requires `libfusion` from `libs/`.
- `carve`: Voxel carving via Open3D. Requires `open3d`.
- `fill`: Voxelization and hole filling via NVIDIA Kaolin. Requires `torch` and `kaolin`.
- `script`: Apply only MeshLab filter scripts (no watertighting).

**Pipeline (fuse mode):**

1. Load and normalize mesh to unit cube with padding
2. Render depth maps from N viewpoints on a sphere
3. Optionally erode depth maps to preserve thin structures
4. Fuse depths into a TSDF volume via `libfusion`
5. Extract surface via marching cubes
6. Apply MeshLab filter scripts (clean + simplify)
7. De-normalize back to original scale

```bash
# Basic usage -- processes all .obj files recursively
make_watertight path/to/ShapeNetCore.v1

# With explicit options
make_watertight path/to/meshes \
    --out_dir path/to/output \
    --resolution 256 \
    --depth_offset 1.5 \
    --n_views 100 \
    --n_jobs 8 \
    --out_format .off \
    --mode fuse

# Using Kaolin hole filling instead of TSDF fusion
make_watertight path/to/meshes --mode fill --try_cpu

# Voxel carving mode
make_watertight path/to/meshes --mode carve

# Only apply MeshLab filters (no watertighting)
make_watertight path/to/meshes --mode script \
    --script_dir assets/meshlab_filter_scripts

# Verify watertightness of existing results
make_watertight path/to/output --check --check_watertight

# Re-process failed results
make_watertight path/to/output --check --fix
```

**Key options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `fuse` | `fuse`, `carve`, `fill`, or `script` |
| `--resolution` | 256 | TSDF voxel grid resolution |
| `--n_views` | 100 | Number of depth views for fusion |
| `--depth_offset` | 1.5 | Depth offset to thicken objects |
| `--padding` | 0.1 | Relative padding during normalization |
| `--n_jobs` | all CPUs | Parallel workers |
| `--out_format` | `.off` | Output format: `.obj`, `.off`, `.ply`, `.stl` |
| `--precision` | 16 | Output float precision: 16, 32, or 64 |
| `--script_dir` | `assets/meshlab_filter_scripts` | MeshLab XML scripts to apply |
| `--no_normalization` | false | Skip normalize-then-denormalize step |
| `--check_watertight` | false | Verify output mesh is watertight |

**Dependencies:** `trimesh`, `numpy`, `scipy`, `joblib`, `pyrender`, `pymeshlab`, and one of: `libfusion` (fuse), `open3d` (carve), or `torch` + `kaolin` (fill).

### render_kinect -- Kinect Depth Simulation

Renders depth images with simulated structured-light Kinect v1 artifacts (IR projector occlusion shadows, missing depth at silhouettes) using `libkinect`. Also renders surface normal maps. For each view, the script samples a random camera position on a partial sphere, renders a clean depth + normal map via `pyrender`, then applies the Kinect simulation via `libkinect`.

```bash
# Single mesh
render_kinect --in_file mesh.off

# With output directory and camera parameters
render_kinect --in_file mesh.off \
    --out_dir path/to/output \
    --n_views 10 \
    --width 640 --height 480 \
    --fx 582.7 --fy 582.7 \
    --cx 320.8 --cy 245.3

# With random scale and rotation augmentation
render_kinect --in_file mesh.off \
    --scale_object --rotate_object

# Verify existing renders
render_kinect --in_file mesh.off --check --fix
```

**Output structure:**

```
output_dir/
├── depth/
│   ├── 00000.png           # Clean depth maps (16-bit PNG)
│   └── ...
├── normal/
│   ├── 00000.jpg           # Surface normal maps (RGB JPEG)
│   └── ...
├── kinect/
│   ├── 00000.png           # Kinect-simulated depth (16-bit PNG)
│   └── ...
└── parameters.npz          # Intrinsics, extrinsics, scales, max depths
```

**Key options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--n_views` | 100 | Number of viewpoints |
| `--width` / `--height` | 640 / 480 | Image resolution |
| `--fx`, `--fy` | 582.7 | Focal lengths |
| `--cx`, `--cy` | 320.8 / 245.3 | Principal point |
| `--znear` / `--zfar` | 0.5 / 6.0 | Depth clipping planes |
| `--depth_precision` | 16 | 8- or 16-bit depth images |
| `--scale_object` | false | Randomly scale the object per view |
| `--rotate_object` | false | Randomly rotate the object per view |
| `--inplane_rotation` | 0.0 | Fixed in-plane camera rotation (degrees) |

**Dependencies:** `trimesh`, `pyrender`, `open3d`, `easy_o3d`, `scipy`, `libkinect` (from `libs/`).

### render_kinect_parallel -- Parallel Kinect Rendering

Wraps `render_kinect` to process a directory of meshes in parallel using `joblib`. Supports sharding for very large datasets.

```bash
render_kinect_parallel path/to/meshes \
    --out_dir path/to/output \
    --in_format .off \
    --n_jobs 8 \
    --n_views 10

# Multiple shards (creates separate output directories per shard)
render_kinect_parallel path/to/meshes \
    --out_dir path/to/output \
    --n_shards 4 \
    --n_jobs 8
```

Inherits all camera and rendering options from `render_kinect`.

**Dependencies:** Same as `render_kinect`, plus `joblib`.

### render_blenderproc -- BlenderProc Rendering

Renders multi-object scenes using BlenderProc with configurable camera sampling, object placement, lighting, and support for both Cycles and Eevee renderers. Produces HDF5 files with depth, normals, segmentation, and camera parameters. Configured via `tyro` (dataclass CLI).

No registered entry point; run directly:

```bash
python -m process.scripts.render_blenderproc \
    --shapenet-dir path/to/ShapeNetCore.v1 \
    --out-dir path/to/output \
    --n-scenes 100 \
    --n-views 5
```

**Dependencies:** `blenderproc`, `bpy`, `opencv-contrib-python`, `trimesh`, `tyro`, `loguru`. Install via `uv sync --extra blenderproc`.

### find_uncertain_regions -- Uncertainty Identification

Identifies regions with ambiguous occupancy given partial depth observations. For each query point, renders depth maps from many viewpoints and determines whether the point's occupancy is consistently observable or ambiguous. Implements the method from [Shape Completion with Prediction of Uncertain Regions](https://arxiv.org/abs/2308.00377).

Uses Hydra for configuration (config directory: `conf/`).

```bash
# Using Hydra config
find_uncertain_regions -cn shapenet_uncertain

# With overrides
find_uncertain_regions -cn shapenet_uncertain \
    vis.split=val \
    log.verbose=true \
    vis.show=true \
    test.overwrite=true
```

**Output:** Per-sample `.npy` files containing bitpacked uncertainty labels for query points.

**Dependencies:** `open3d`, `pyrender`, `hydra-core`, `omegaconf`, `joblib`. Also requires the `dataset` submodule.

### generate_physics_poses (find_pose) -- Physics-Based Poses

Generates physically plausible stable resting poses for objects using PyBullet rigid body simulation. Drops objects from initial orientations and records final stable poses. Supports optional V-HACD convex decomposition for better collision geometry.

```bash
# Find stable poses using principal axis rotations (default)
find_pose path/to/meshes

# Random initial orientations
find_pose path/to/meshes \
    --rotate random \
    --num_poses 100

# With mesh simplification and V-HACD collision mesh
find_pose path/to/meshes \
    --simplify 10000 \
    --vhacd

# Visualize the simulation
find_pose path/to/mesh.obj --show

# Visualize identified poses
find_pose path/to/mesh.obj --show poses
```

**Output:** `poses.npy` in each mesh's directory, containing an array of 4x4 rotation matrices for stable poses.

**Key options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--rotate` | `principal` | `principal` (6 axis rotations) or `random` |
| `--num_poses` | 100 | Number of random initial orientations |
| `--simplify` | false | Simplify mesh first (bool, int target faces, or float ratio) |
| `--solidify` | false | Thicken thin surfaces via Blender solidify modifier |
| `--vhacd` | false | Use V-HACD convex decomposition for collision |
| `--gravity` | -9.81 | Gravity in z-direction |
| `--up_axis` | 2 (z) | Up axis for duplicate pose filtering |

**Dependencies:** `pybullet`, `pybullet_data`, `pybullet_utils`, `open3d`, `trimesh`, `scipy`. Optional: `bpy` (for `--solidify`), `gitpython` (for V-HACD auto-build).

### process_mesh -- Point Sampling and Mesh Normalization

Two tasks: `sample` (sample point clouds, surface points, and occupancy queries from watertight meshes) and `normalize` (center and scale meshes to unit cube).

No registered entry point; run directly:

```bash
# Sample points from watertight meshes
python -m process.scripts.process_mesh path/to/meshes sample \
    --out_dir path/to/output \
    --in_format .off \
    --num_points 100000 \
    --normalize \
    --n_jobs 8

# Normalize meshes
python -m process.scripts.process_mesh path/to/meshes normalize \
    --out_dir path/to/output \
    --center --scale
```

**Sample output structure:**

```
output/samples/
├── surface.npz           # Surface points + normals
├── uniform_random.npz    # Uniform random points + occupancy
├── surface_random.npz    # Near-surface points + occupancy
└── uniform_sphere_*.npz  # Sphere-sampled points + occupancy
```

**Dependencies:** `trimesh`, `open3d`, `joblib`, `libmesh` (from `libs/`, for `check_mesh_contains`).

### sdf_from_mesh -- SDF Computation

Computes signed distance fields from meshes using `libsdf` (from `libs/`), extracts meshes via marching cubes, and samples training data (points + SDF values) in various distributions.

No registered entry point; run directly:

```bash
# Compute SDF and sample points
python -m process.scripts.sdf_from_mesh path/to/mesh.obj 256

# With ShapeNet directory structure
python -m process.scripts.sdf_from_mesh "path/to/ShapeNet/**/*.obj" 256 \
    --shapenet --version 1 \
    -o path/to/output

# Store SDF grid and extracted mesh
python -m process.scripts.sdf_from_mesh mesh.obj 256 \
    --sdf --mesh \
    -o path/to/output

# Parallel processing
python -m process.scripts.sdf_from_mesh "path/to/*.obj" 256 \
    --parallel 8

# Specific sampling strategies
python -m process.scripts.sdf_from_mesh mesh.obj 256 \
    --samples surface_random.npy deepsdf.npy if_net.npy \
    --num_points 100000
```

**Available sampling strategies:**

| Sample file | Description |
|-------------|-------------|
| `uniform_grid.npy` | Uniform samples from the SDF voxel grid |
| `equal_grid.npy` | Equal inside/outside samples from grid |
| `surface_grid.npy` | Near-surface grid + uniform samples |
| `uniform_random.npy` | Uniform random in bounding volume (default) |
| `equal_random.npy` | Equal inside/outside with noise |
| `surface_random.npy` | Multi-sigma near-surface with interpolated SDF |
| `deepsdf.npy` | DeepSDF-style sampling |
| `if_net.npy` | IF-Net-style sampling |
| `uniform_sphere.npy` | Uniform random on spheres at configurable radii (default) |
| `surface.npy` | Surface points (always generated) |
| `normals.npy` | Corresponding surface normals (always generated) |

**Dependencies:** `trimesh`, `open3d`, `scipy`, `scikit-image`, `pymcubes`, `joblib`, `libsdf` and `libmcubes` (from `libs/`).

### add_kinect_sim -- Add Kinect Simulation to HDF5 Data

Post-processes existing HDF5 files (produced by `render_blenderproc`) to add the `kinect_sim` depth modality. Loads meshes from ShapeNet, transforms them into camera coordinates, and runs the Kinect structured-light simulation. Configured via `tyro` (dataclass CLI).

```bash
add_kinect_sim \
    --input-dir path/to/hdf5s \
    --shapenet-dir path/to/ShapeNetCore.v1 \
    --noise perlin

# Overwrite existing kinect_sim data
add_kinect_sim \
    --input-dir path/to/hdf5s \
    --shapenet-dir path/to/shapenet \
    --overwrite
```

**Noise types:** `none`, `gaussian`, `perlin` (default), `simplex`.

**Dependencies:** `h5py`, `trimesh`, `tyro`, `loguru`, `libkinect` (from `libs/`).

### ycb_sfm -- YCB Photogrammetry Reconstruction

End-to-end COLMAP-based 3D reconstruction pipeline for YCB objects from multi-view images. Handles image preparation, mask extraction (via `rembg`), sparse reconstruction (feature extraction, matching, triangulation), dense MVS (patch matching, stereo fusion), and Poisson surface reconstruction.

No registered entry point; run directly:

```bash
# Prepare images and extract masks
python -m process.scripts.ycb_sfm path/to/ycb_object \
    -gh path/to/colmap_github \
    --data

# Run sparse reconstruction
python -m process.scripts.ycb_sfm path/to/ycb_object \
    -gh path/to/colmap_github \
    --sparse

# Run dense reconstruction
python -m process.scripts.ycb_sfm path/to/ycb_object \
    -gh path/to/colmap_github \
    --dense

# Full pipeline (data + sparse + dense + mesh)
python -m process.scripts.ycb_sfm path/to/ycb_object \
    -gh path/to/colmap_github \
    --data --sparse --dense --mesh

# With custom camera model and IDs
python -m process.scripts.ycb_sfm path/to/ycb_object \
    -gh path/to/colmap_github \
    --data --sparse --dense \
    --camera_model OPENCV \
    --camera_ids 1 2 3
```

**Dependencies:** `h5py`, `rembg`, `scipy`, `colmap` (external binary), COLMAP Python scripts (from COLMAP GitHub repo).

### blender_load -- Blender Scene Import

Loads shape-completion export data (meshes, cameras, depth maps, point clouds) into Blender 4/5 scenes for visualization and rendering. Run inside Blender:

```bash
blender --python process/scripts/blender_load.py -- \
    --dir path/to/exported/output \
    --stem <filename_stem>
```

**Dependencies:** `bpy`, `bmesh`, `mathutils`, `numpy`, `loguru`.

## Python API

### Public Exports

The `process` package exports these functions (via `process/__init__.py`):

```python
from process import (
    apply_meshlab_filters,   # Apply MeshLab XML filter scripts
    get_points,              # Get 3D query points
    get_views,               # Generate camera viewpoints on sphere
    modify_simplify,         # Modify MeshLab simplify script parameters
    normalize_mesh,          # Normalize mesh to unit cube
    normalize_pointcloud,    # Normalize point cloud to unit cube/sphere
    sample_pointcloud,       # Sample surface points from trimesh
)
```

### Watertight Mesh Generation

```python
from process.scripts.make_watertight import load, normalize, process, save
from process.src.fuse import pyfusion_pipeline

# Load mesh
mesh = load(Path("input.obj"), loader="pymeshlab", return_type="dict")

# Normalize to unit cube
mesh, translation, scale = normalize(mesh, padding=0.1)

# Run TSDF fusion (requires args namespace with resolution, n_views, etc.)
mesh = pyfusion_pipeline(mesh, args)

# De-normalize and save
mesh, _, _ = normalize(mesh, translation=-translation / scale, scale=1 / scale)
save(mesh, Path("output.off"))
```

### Point Cloud Normalization

```python
from process.src.utils import normalize_pointcloud

# Normalize to unit cube [-1, 1]
points = normalize_pointcloud(points, center=True, scale=True, cube_or_sphere="cube")

# Normalize to unit sphere
points = normalize_pointcloud(points, center=True, scale=True, cube_or_sphere="sphere")
```

### MeshLab Filters

```python
from process.src.utils import apply_meshlab_filters, load_scripts, modify_simplify

# Load and apply MeshLab filter scripts
scripts = load_scripts(Path("assets/meshlab_filter_scripts"), num_vertices=10000)
vertices, faces = apply_meshlab_filters(vertices, faces, scripts)
```

## Data Processing Pipeline

A typical end-to-end pipeline for preparing ShapeNet data:

```bash
# 1. Generate watertight meshes from raw ShapeNet
make_watertight path/to/ShapeNetCore.v1 \
    --out_dir path/to/watertight \
    --resolution 256 --n_views 100 --n_jobs 8

# 2. Compute SDFs and sample training points
python -m process.scripts.sdf_from_mesh \
    "path/to/watertight/**/*.off" 256 \
    --shapenet -o path/to/sdf_data \
    --samples uniform_random.npy surface_random.npy uniform_sphere.npy \
    --parallel 8

# 3. Sample surface point clouds and occupancy queries
python -m process.scripts.process_mesh \
    path/to/watertight sample \
    --out_dir path/to/samples \
    --normalize --n_jobs 8

# 4. Render Kinect-style depth images
render_kinect_parallel path/to/watertight \
    --out_dir path/to/renders \
    --n_views 10 --n_jobs 8

# 5. Find stable resting poses
find_pose path/to/watertight \
    --out_dir path/to/poses \
    --rotate principal --n_jobs 8

# 6. Identify uncertain regions
find_uncertain_regions -cn shapenet_uncertain \
    vis.split=train
```

For BlenderProc-based rendering (multi-object scenes):

```bash
# 1. Render scenes
python -m process.scripts.render_blenderproc \
    --shapenet-dir path/to/ShapeNetCore.v1 \
    --out-dir path/to/scenes \
    --n-scenes 1000 --n-views 5

# 2. Add Kinect simulation to rendered scenes
add_kinect_sim \
    --input-dir path/to/scenes \
    --shapenet-dir path/to/ShapeNetCore.v1
```

## Dependencies

### Core (installed via `uv sync --extra process`)

- `trimesh` -- mesh I/O and processing
- `numpy`, `scipy` -- numerical computation
- `joblib`, `tqdm` -- parallelism and progress bars
- `pymcubes` -- marching cubes mesh extraction
- `tyro` -- dataclass-based CLI (for `add_kinect_sim`, `render_blenderproc`)
- `loguru` -- structured logging (for `add_kinect_sim`, `render_blenderproc`)

### From `libs/` (compiled C/Cython extensions)

| Library | Used by | Purpose |
|---------|---------|---------|
| `libfusion` | `make_watertight` (fuse mode) | TSDF fusion |
| `libkinect` | `render_kinect`, `add_kinect_sim` | Kinect structured-light simulation |
| `libmesh` | `process_mesh` | Occupancy queries (`check_mesh_contains`) |
| `libsdf` | `sdf_from_mesh` | Distance field computation |
| `libmcubes` | `sdf_from_mesh` | Marching cubes (Vega FEM) |
| `libsimplify` | `generate_physics_poses` | Mesh simplification |

### Optional Python Dependencies

| Feature | Extra | Dependencies |
|---------|-------|--------------|
| TSDF fusion | `process` | `pyrender`, `pymeshlab` |
| Voxel carving | -- | `open3d` |
| Kaolin hole filling | -- | `torch`, `kaolin` |
| BlenderProc rendering | `blenderproc` | `blenderproc`, `bpy`, `opencv-contrib-python`, `coacd` |
| Physics simulation | `blenderproc` | `pybullet` |
| Kinect rendering | `git` | `easy_o3d`, `open3d`, `pyrender` |
| YCB reconstruction | -- | `h5py`, `rembg`, `colmap` (external) |
| Blender import | -- | `bpy` (Blender's bundled Python) |
| Uncertain regions | -- | `open3d`, `hydra-core`, `omegaconf` |

## Performance Tips

1. **Parallel watertight generation** -- scale to available cores:
   ```bash
   make_watertight data/ --n_jobs $(nproc)
   ```

2. **GPU-accelerated fusion** -- install the GPU variant of libfusion:
   ```bash
   uv pip install libs/libfusion_gpu --no-build-isolation
   ```

3. **Batch Kinect rendering** -- use the parallel wrapper:
   ```bash
   render_kinect_parallel data/ --n_jobs 8
   ```

4. **Lower precision for faster I/O** -- use `--precision 16` where supported.

5. **Headless rendering** -- scripts automatically set `PYOPENGL_PLATFORM=egl` unless `--show` is used.

## References

- Watertight mesh generation: Based on [mesh-fusion](https://github.com/davidstutz/mesh-fusion)
- Uncertain regions: [Shape Completion with Prediction of Uncertain Regions](https://arxiv.org/abs/2308.00377) (IROS 2023)
