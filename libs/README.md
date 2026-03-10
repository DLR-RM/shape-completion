# Libraries

Several libraries for geometry and 3D manipulation, including operations on point clouds,
meshes, voxel grids, and depth images. All Python-installable libraries are managed through
`libmanager.py`; standalone binaries (vega, binvox, v-hacd) are used directly.

## Quick start

```bash
# Install all pip-installable libraries (skips already-installed ones)
python libs/libmanager.py install

# Install specific libraries by short name (without the "lib" prefix)
python libs/libmanager.py install chamfer emd kinect
```

See [Installation](#installation) for full details.

## List of available libraries

### Pip-installable libraries

These are managed by `libmanager.py` and installed as Python packages into the active
virtual environment. Each has a `pyproject.toml` declaring its build dependencies.

| Library | Package name | Type | Build deps | System deps | GPU required |
|---|---|---|---|---|---|
| libchamfer | `chamfer_ext` | CUDA/Torch | torch>=2.0 | CUDA toolkit | Yes |
| libemd | `emd_ext` | CUDA/Torch | torch>=2.0 | CUDA toolkit | Yes |
| libfusion/cpu | `pyfusion_cpu` | Cython | cython>=3.0, numpy>=1.20 | — | No |
| libfusion/gpu | `pyfusion_gpu` | Cython+CUDA | cython>=3.0, numpy>=1.20, torch>=2.0 | CUDA toolkit | Yes |
| libintersect | `intersect_ext` | Cython | cython>=3.0, numpy>=1.20 | — | No |
| libkinect | `kinect_ext` | Cython+C++ | cython>=3.0, numpy>=1.20 | CGAL, Eigen3, OpenCV, assimp, libnoise, OpenMP (optional) | No |
| libmise | `mise_ext` | Cython | cython>=3.0 | — | No |
| libpointnet | `pointnet_ext` | CUDA/Torch | torch>=2.0 | CUDA toolkit | Yes |
| libpvconv | `pvconv_ext` | CUDA/Torch | torch>=2.0 | CUDA toolkit | Yes |
| libsimplify | `simplify_ext` | Cython | cython>=3.0, numpy>=1.20 | — | No |

### Standalone binaries

These are not installed via pip. They are invoked as executables or imported directly.

| Name | Description | How to use |
|---|---|---|
| **vega** | Pre-compiled binaries for distance field computation and marching cubes isosurface extraction (based on [Vega FEM](http://run.usc.edu/vega/)). Provides `computeDistanceField` and `isosurfaceMesher` executables. | Import `from libs.vega import compute_distance_field, compute_marching_cubes`. Requires `LD_LIBRARY_PATH` to include the vega directory (set automatically on import). Depends on bundled `libtbb.so.2` and `libtcmalloc.so.4`. |
| **binvox** | Pre-compiled binary for voxelizing 3D meshes into binary voxel grids. | Run `libs/binvox <mesh_file>` directly. |
| **v-hacd** | [V-HACD](https://github.com/kmammou/v-hacd) — Volumetric Hierarchical Approximate Convex Decomposition for decomposing meshes into convex parts. Used for physics simulation (e.g., PyBullet) where convex collision shapes are required. | Build from source with CMake (C++11): `cd libs/v-hacd/app && cmake . && make`. Run `TestVHACD` executable. |

## Library details

### libchamfer

Chamfer distance between two point sets, computed on GPU via custom CUDA kernels.

### libemd

Earth Mover's Distance (EMD) between two point sets, computed on GPU via custom CUDA kernels.

### libfusion

GPU-accelerated TSDF fusion of depth images to generate watertight meshes from triangle soups.
Available in two variants:

- **libfusion/cpu**: Cython-based CPU implementation (no GPU required)
- **libfusion/gpu**: CUDA-accelerated implementation (requires torch + CUDA)

The `libs.libfusion` package automatically selects the GPU variant if installed, falling back
to CPU.

### libintersect

Functions and classes for mesh containment queries — determines whether points lie inside
a triangle mesh using triangle hashing and ray intersection.

### libkinect

Kinect depth sensor simulator using structured light stereo matching. Simulates realistic
depth sensing behavior including stereo occlusion artifacts and configurable noise patterns.

**Features:**
- Ray casting from IR camera with mesh intersection
- Stereo visibility checking (IR projector occlusion)
- Disparity-to-depth conversion
- Configurable noise: None, Gaussian, Perlin, Simplex

**Python API:**
```python
from libkinect import KinectSimCython, NoiseType

sim = KinectSimCython()
depth = sim.simulate(
    vertices,              # (N, 3) float32 mesh vertices in camera coords
    faces,                 # (M, 3) int32 face indices
    width=640, height=480, # Image resolution
    fx=582.7, fy=582.7,    # Focal lengths
    cx=320.8, cy=245.3,    # Principal point
    z_near=0.5, z_far=4.0, # Depth range (meters)
    baseline=0.075,        # IR projector-camera baseline (meters)
    noise=NoiseType.PERLIN # Noise type
)
```

**System dependencies:** CGAL (header-only, v5+), Eigen3, OpenCV (core + highgui + imgcodecs),
assimp, libnoise (bundled in `lib/libnoise/`). Optional: OpenMP for parallelization. Requires C++17.

### libmise

Multi-resolution ISOsurface extraction (MISE). Pure Cython, no system dependencies.

### libpointnet

PointNet2 operations for PyTorch with CUDA backend: ball query, grouping, interpolation,
furthest point sampling.

### libpvconv

Point-Voxel Convolution (PVConv) operations for PyTorch with CUDA backend. Implements
voxelization, trilinear devoxelization, ball query, shared MLPs, and squeeze-and-excitation
blocks used by PVCNN-based models.

### libsimplify

Fast quadric edge-collapse decimation for triangle meshes. Pure Cython.

## Optional vs required

All libraries use lazy imports with graceful fallback — if a library is not installed, its
symbols are set to `None` and a warning is logged at import time. No library is strictly
required for the package to import. Which libraries you need depends on your use case:

- **Training / evaluation with point cloud metrics**: libchamfer, libemd
- **PointNet2-based models**: libpointnet
- **PVCNN-based models**: libpvconv
- **Occupancy-based models** (mesh extraction): libmise, libintersect
- **TSDF fusion pipeline**: libfusion (cpu or gpu)
- **Watertight mesh generation**: vega, libsimplify
- **Depth simulation**: libkinect
- **Voxelization**: binvox
- **Physics simulation**: v-hacd

Note: libkinect is **not** imported through the `libs` package `__init__.py` — it is imported
directly as `from libkinect import KinectSimCython`.

## Installation

### Using libmanager (recommended)

`libmanager.py` wraps `pip install` (or `uv pip install` when uv is detected) with
convenience features: skip-if-installed logic, CUDA architecture selection, and batch
installation.

```bash
# Install all libraries
python libs/libmanager.py install

# Install specific libraries (use short names without "lib" prefix)
python libs/libmanager.py install kinect chamfer emd

# Set CUDA architecture for your GPU (default covers most GPUs)
python libs/libmanager.py install --cuda_archs "8.9"

# Upgrade already-installed libraries (pip -U)
python libs/libmanager.py upgrade chamfer emd

# Force-reinstall (pip --force-reinstall)
python libs/libmanager.py force-reinstall kinect

# Uninstall
python libs/libmanager.py uninstall kinect

# Clean build artifacts (build/, dist/, *.egg-info, generated .cpp from .pyx)
python libs/libmanager.py clean kinect
```

**Commands:**

| Command | Behavior |
|---|---|
| `install` | Skips already-installed libraries. Checks both pip metadata and on-disk `.so` files. |
| `upgrade` | Runs `pip install -U` — upgrades even if already installed. |
| `force-reinstall` | Runs `pip install --force-reinstall` — rebuilds from scratch. |
| `uninstall` | Runs `pip uninstall -y <package_name>`. |
| `clean` | Removes `build/`, `dist/`, `*.egg-info`, generated `.cpp` files, and optionally JIT cache (`torch_extensions`). |

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--cuda_archs` | `"5.0+PTX;6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.6;8.9+PTX"` | CUDA architectures to compile for. Set to your GPU's compute capability for faster builds. Conflicts with `TORCH_CUDA_ARCH_LIST` env var if both are set to different values. |
| `--verbose` | off | Pass `-v` to pip for verbose build output. |
| `--no-uv` | off | Force pip instead of auto-detected uv. |
| `--build-isolation` | off | Enable PEP 517 build isolation. Default is off (`--no-build-isolation`), which reuses torch/numpy from your venv for much faster CUDA builds. |

### Direct installation with uv/pip

```bash
# Standard install (downloads build deps like torch — slower first time)
uv pip install libs/libchamfer

# Fast install (reuses torch from venv — requires torch already installed)
uv pip install libs/libchamfer --no-build-isolation

# Editable install (keeps .so in source dir — useful for development)
uv pip install -e libs/libkinect --no-build-isolation
```

**Performance tip:** CUDA libraries (chamfer, emd, pointnet, pvconv, fusion_gpu) require
torch at build time. Using `--no-build-isolation` avoids re-downloading torch for each
library, making builds significantly faster.

### Build dependencies

Each library's `pyproject.toml` declares its Python build dependencies:

- **Cython-only** (mise): `cython>=3.0`
- **Cython + NumPy** (intersect, simplify, fusion_cpu): `cython>=3.0`, `numpy>=1.20`
- **CUDA/Torch** (chamfer, emd, pointnet, pvconv): `torch>=2.0`
- **Cython + NumPy + Torch** (fusion_gpu): `cython>=3.0`, `numpy>=1.20`, `torch>=2.0`
- **Cython + NumPy + system libs** (kinect): `cython>=3.0`, `numpy>=1.20` + CGAL, Eigen3, OpenCV, assimp, libnoise

### System dependencies

#### All CUDA libraries

- NVIDIA GPU with CUDA toolkit installed
- `nvcc` on PATH (or set via `CUDA_HOME`)
- Compatible PyTorch with CUDA support

#### libkinect

Install on Ubuntu/Debian:

```bash
sudo apt install libeigen3-dev libcgal-dev libopencv-dev libassimp-dev
```

libnoise is bundled in `libs/libkinect/lib/libnoise/`.

#### vega

Pre-compiled Linux x86_64 binaries. Ships with bundled `libtbb.so.2` and `libtcmalloc.so.4`.
The `LD_LIBRARY_PATH` is set automatically when importing `libs.vega`. If running the
binaries directly, set it manually:

```bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/libs/vega"
```

## Troubleshooting

### CUDA libraries fail to build

**`nvcc not found` or `No CUDA toolkits found`**
- Ensure CUDA toolkit is installed and `nvcc` is on your PATH.
- Or set `CUDA_HOME=/usr/local/cuda` (or wherever your toolkit lives).

**`error: 'AT_CHECK' was not declared` or similar PyTorch API errors**
- PyTorch API changes across versions. Ensure your torch version matches the one the
  library was developed against (torch>=2.0).

**Build is extremely slow**
- Use `--no-build-isolation` (default in libmanager) to avoid re-downloading torch.
- Set `--cuda_archs` to only your GPU's compute capability instead of the broad default
  (e.g., `--cuda_archs "8.9"` for RTX 4090).

### libkinect fails to build

**`CGAL_DIR not found` / `Could NOT find CGAL`**
- Install CGAL: `sudo apt install libcgal-dev` (Ubuntu) or build from source.
- CGAL 5+ is header-only; just needs headers discoverable by CMake.

**`Eigen/Dense: No such file or directory`**
- Install Eigen3: `sudo apt install libeigen3-dev`.
- The setup.py hardcodes `/usr/include/eigen3`. If Eigen is installed elsewhere, edit
  `libs/libkinect/setup.py` to update the include path.

**`opencv2/core.hpp: No such file or directory`**
- Install OpenCV dev packages: `sudo apt install libopencv-dev`.
- Verify headers are at `/usr/include/opencv4` (the path in setup.py).

### `uv sync` breaks installed libraries

`uv sync` can delete `.so` files while leaving `.dist-info` metadata intact.
`libmanager.py` handles this by checking for actual `.so` file existence, not just metadata.
If a library shows as installed but fails to import, run:

```bash
python libs/libmanager.py force-reinstall <library_name>
```

### vega binaries fail with shared library errors

```
error while loading shared libraries: libtbb.so.2: cannot open shared object file
```

Set `LD_LIBRARY_PATH` to include the vega directory:

```bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/libs/vega"
```

### `clean` command

If builds leave stale artifacts causing import errors or rebuild failures:

```bash
python libs/libmanager.py clean <library_name>
# Then reinstall
python libs/libmanager.py force-reinstall <library_name>
```

The `clean` command removes `build/`, `dist/`, `*.egg-info` directories, Cython-generated
`.cpp` files, and (by default) the `torch_extensions` JIT cache.

## Testing

Unit tests live in `libs/tests/`. Run with pytest:

```bash
pytest libs/tests/ -v
```

Tests exist for: libchamfer, libfusion, libintersect, libkinect, libmise, libsimplify, and
furthest point sampling (FPS).
