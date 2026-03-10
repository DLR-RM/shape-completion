# shape-completion-visualize

Mesh extraction and rendering infrastructure for shape completion. Provides the `Generator` class for extracting meshes from implicit models and the `Renderer` class supporting multiple backends (Blender, PyRender, Open3D, PyTorch3D, Plotly). A higher-level `Visualizer` class composes both into a single interface for generate-and-render workflows.

## Installation

```bash
# As submodule

# Dependencies (from main repo)
uv sync --extra visualize

# For path-traced rendering (optional)
uv sync --extra pathtracing
```

### BlenderProc Setup (for path tracing)

```bash
git clone git@github.com:DLR-RM/BlenderProc.git
cd BlenderProc
uv pip install -e .
```

## Quick Start

```python
from visualize import Generator, Renderer

# Extract mesh from trained model
generator = Generator(model, resolution=128, threshold=0.5)
mesh = generator.generate_mesh({"inputs": point_cloud_tensor})

# Render mesh
renderer = Renderer(method="pyrender", width=512, height=512)
image = renderer.render(mesh.vertices, mesh.faces)
```

## Architecture

```
visualize/
├── __init__.py                     # Lazy-loads Renderer to avoid heavy imports
├── src/
│   ├── __init__.py
│   ├── generator.py                # Mesh extraction from implicit models
│   ├── renderer.py                 # Multi-backend rendering
│   └── visualizer.py               # High-level Generator + Renderer wrapper
├── scripts/
│   ├── vis_data.py                 # Visualize data loading/augmentation
│   ├── vis_inference.py            # Visualize inference results + metrics
│   ├── render.py                   # Render meshes/point clouds to images
│   ├── render_gen.py               # Render generated meshes for figures
│   ├── load_mesh.py                # Interactive mesh inspection (Open3D)
│   ├── vis_generation_process.py   # Visualize generation process steps
│   ├── render_generation_process.py
│   ├── vis_ar_quality_sweep.py     # Autoregressive quality sweep
│   ├── vis_cross_attention_queries.py
│   ├── vis_latent_embedding_compare.py
│   └── vis_latent_pca_rgb.py
├── assets/
│   └── backdrop.ply                # Backdrop mesh for rendered scenes
└── tests/
```

## Generator vs Renderer

**Generator** converts implicit function outputs (occupancy or SDF grids) into explicit geometry (meshes, point clouds). It handles the model-specific inference logic (VAE sampling, diffusion generation, autoregressive decoding, MC dropout) and runs Marching Cubes or MISE to produce `trimesh.Trimesh` objects.

**Renderer** takes explicit geometry (vertices, faces, colors) and produces images. It dispatches to one of several rendering backends depending on speed/quality requirements. The Renderer knows nothing about models or implicit functions.

**Visualizer** composes a Generator and a Renderer behind a single interface. It lazily initializes both, automatically selects MISE for high resolutions (>128), and provides `get_mesh()` and `get_image()` convenience methods. Useful when you want a one-liner for generate-then-render without managing both objects.

```python
from visualize.src.visualizer import Visualizer

vis = Visualizer(resolution=128, method="pyrender", width=512, height=512)
vis.generator = model          # Internally creates a Generator
mesh = vis.get_mesh(item)      # generate_mesh wrapper
image = vis.get_image([mesh])  # render wrapper
vis.save(path, mesh=mesh, image=image)
```

## Generator

The `Generator` class extracts meshes from implicit function models using Marching Cubes (via PyMCubes or scikit-image) or MISE (Multi-resolution IsoSurface Extraction).

### Constructor Parameters

```python
Generator(
    model,                          # Trained implicit model (models.Model subclass)
    points_batch_size=None,         # Query batch size; defaults to full grid
    threshold=0.5,                  # Occupancy threshold (or SDF iso-level)
    extraction_class=1,             # Class label to extract (for multi-class)
    refinement_steps=0,             # Gradient-based surface refinement iterations
    resolution=128,                 # Grid resolution per axis
    upsampling_steps=0,             # MISE upsampling (0=plain Marching Cubes)
    estimate_normals=False,         # Compute vertex normals via gradient
    predict_colors=False,           # Query model for per-vertex colors
    padding=0.1,                    # Bounding box padding
    scale_factor=1.0,               # Bounding box scale
    simplify=None,                  # Mesh simplification (bool, int, or float)
    use_skimage=False,              # Use skimage instead of PyMCubes
    sdf=False,                      # Model outputs SDF (not occupancy logits)
    bounds=(-0.5, 0.5),             # Spatial bounds; scalar pair or per-axis tuple
)
```

### Key Behaviors

- **Bounding box**: Computed from `bounds`, `padding`, and `scale_factor`. The grid uses uniform voxel spacing determined by the largest axis, so per-axis resolutions may differ for non-cubic bounds.
- **Threshold interpretation**: For occupancy models, `threshold` is converted to logit space via `log(t / (1-t))`. For SDF models, `threshold` is used directly as the iso-level.
- **Marching Cubes backend**: PyMCubes (default) pads the grid by 1 to ensure watertight output. scikit-image (`use_skimage=True`) supports gradient direction control and vertex normals but may produce open meshes.
- **Simplification**: `True` uses `simplify_mesh(target_percent=10)`, an `int` sets target face count, a `float` sets target percentage. Prefers MeshLab scripts from the `process` submodule when available; falls back to `libsimplify`.

### MISE (Multi-resolution IsoSurface Extraction)

MISE starts at resolution 32 and adaptively refines regions near the surface. The final resolution is `2^(log2(32) + upsampling_steps)`, so the `resolution` parameter is overridden when `upsampling_steps > 0`.

```python
# upsampling_steps=1 -> 64, =2 -> 128, =3 -> 256, =4 -> 512
generator = Generator(model, resolution=32, upsampling_steps=3)  # effective 256
```

MISE is only available for binary extraction (`extraction_class=1`) and requires the `libs.MISE` module.

### Surface Refinement

Iteratively moves mesh vertices toward the true iso-surface using RMSprop on the occupancy loss, with an optional normal consistency regularizer:

```python
generator = Generator(
    model,
    resolution=128,
    refinement_steps=10,     # 10 iterations of gradient-based refinement
)
```

Refinement requires the model encoder to produce features (calls `model.encode()` then `model.decode()` with gradient tracking).

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `generate_mesh(item, ...)` | `Trimesh \| list[Trimesh]` | Full pipeline: grid prediction, Marching Cubes, optional refinement/simplification |
| `generate_grid(item, ...)` | `(grid, points, features)` | Predict occupancy/SDF on the 3D grid without mesh extraction |
| `generate_grid_per_instance(item, ...)` | `list[dict]` | Per-instance grids with coarse-to-fine refinement and candidate selection |
| `generate_pcd(item, ...)` | `PointCloud` | Generate colored point cloud (multi-class or thresholded binary) |
| `extract_mesh(predictions, ...)` | `Trimesh` | Run Marching Cubes on a pre-computed grid |
| `extract_meshes(grid, ...)` | `Trimesh \| list[Trimesh]` | Extract from single grid, list of grids, or class-keyed dict |
| `estimate_vertex_normals(vertices, ...)` | `ndarray` | Compute normals via occupancy gradient (requires grad) |
| `refine_mesh(mesh, ...)` | `Trimesh` | Gradient-based vertex refinement toward iso-surface |
| `predict_vertex_colors(vertices, ...)` | `ndarray` | Query per-vertex RGB from model |

### Special Model Support

The Generator dispatches based on model type in `generate_grid()`:

| Model Type | Behavior |
|------------|----------|
| Standard implicit (`Model`) | Encode once, then decode on grid points in batches |
| `VAEModel` | Optionally sample latent; supports multi-sample generation |
| `DiffusionModel` | Calls `model.generate()` with configurable `steps` kwarg |
| `AutoregressiveModel` | Calls `model.generate()` with threshold-based decoding |
| `MCDropoutNet` | Multiple forward passes; returns `[mean_logits, variance]` |
| `PSSNet` | Multiple stochastic predictions; returns `[mean_logits, variance]` |
| `ShapeFormer` | Calls `model.generate_grids()` directly |

For models with a `predict()` method and no MISE/refinement, the Generator skips encoding and calls `model.predict()` directly on grid points.

### Mesh Simplification

```python
generator = Generator(model, resolution=256, simplify=10000)   # target 10k faces
generator = Generator(model, resolution=256, simplify=0.1)     # target 10% of faces
generator = Generator(model, resolution=256, simplify=True)    # target 10% (default)
```

## Renderer

Multi-backend renderer. All backends share a unified `render()` / `__call__()` interface that accepts vertices, faces, colors, normals, intrinsic, and extrinsic arrays and returns a dict with `"color"`, `"depth"`, and/or `"normal"` keys.

### Rendering Backends

| Backend | Method String | Speed | Quality | GPU Required | Point Clouds | Depth | Normals | Differentiable | Key Trade-off |
|---------|--------------|-------|---------|-------------|-------------|-------|---------|----------------|---------------|
| **PyRender** | `"pyrender"` | Fast | Rasterization | No (EGL offscreen) | Yes (points or spheres) | Yes | Yes (custom shader) | No | Best for fast previews and batch processing |
| **Open3D** | `"open3d"` | Fast | Rasterization | No | Yes (spheres) | Yes | Onscreen only | No | Good for interactive inspection; offscreen normals not implemented |
| **PyTorch3D** | `"pytorch3d"` | Medium | Soft rasterization | Yes (CUDA) | Spheres only | No | No | Yes | Only option for gradient-based optimization through rendering |
| **Blender Cycles** | `"cycles"` | Slow | Path tracing | No (CPU) / Yes (OPTIX) | Yes (geometry nodes) | Yes | Yes | No | Publication-quality renders; requires BlenderProc |
| **Blender Eevee** | `"eevee"` | Medium | Real-time rasterization | No | Yes (particle system) | Yes | Yes | No | Faster Blender alternative without ray tracing |

When `method="auto"`, the Renderer selects a backend based on requested features:
1. `differentiable=True` -> PyTorch3D (requires CUDA)
2. `raytracing=True` -> Blender Cycles
3. Otherwise -> PyRender > Open3D > Blender > PyTorch3D (first available)

### Constructor Parameters

```python
Renderer(
    method="auto",                  # "auto", "pyrender", "open3d", "pytorch3d",
                                    # "blender", "cycles", "eevee"
    width=512,
    height=512,
    render_color=True,              # Produce RGB image
    render_depth=False,             # Produce depth map
    render_normal=False,            # Produce normal map
    offscreen=True,                 # Headless rendering (EGL for pyrender/open3d)
    differentiable=False,           # Enable gradient flow (pytorch3d only)
    raytracing=False,               # Path tracing (blender/cycles only)
    file_format="PNG",              # "PNG", "JPEG", or "EXR"
    transparent_background=False,   # Alpha channel (PNG only)
    show=False,                     # Display result with matplotlib
)
```

### Calling Convention

The Renderer is callable. Both `renderer.render(...)` and `renderer(...)` work identically:

```python
result = renderer(
    vertices=mesh.vertices,               # (N, 3) or list of arrays
    faces=mesh.faces,                      # (F, 3) or list; None for point clouds
    colors=np.array([0.3, 0.6, 0.7]),     # Per-mesh (3,), per-vertex (N, 3), or list
    intrinsic=K,                           # 3x3 camera matrix (optional, has default)
    extrinsic=pose,                        # 4x4 OpenGL camera-to-world (optional)
)
color_image = result["color"]              # H x W x 3 uint8 (or H x W x 4 if transparent)
depth_map = result.get("depth")            # H x W float32
normal_map = result.get("normal")          # H x W x 3 float32
```

Camera conventions: extrinsics are expected in **OpenGL convention** (Y-up, -Z forward). The Renderer converts to OpenCV internally for backends that need it (Open3D, PyTorch3D).

### Multi-Object Scenes

Pass lists to render multiple objects in a single scene:

```python
result = renderer(
    vertices=[mesh.vertices, pcd.vertices, plane.vertices],
    faces=[mesh.faces, None, plane.faces],          # None = point cloud
    colors=[np.array([0.3, 0.6, 0.7]),              # mesh color
            np.array([1.0, 0.5, 0.4]),              # point cloud color
            "shadow"],                                # Blender shadow catcher
)
```

The `"shadow"` color string (Blender only) creates a shadow-catching plane for compositing.

### Differentiable Rendering

```python
renderer = Renderer(method="pytorch3d", differentiable=True)

# Gradients flow through rendering
image = renderer.render(mesh.vertices, mesh.faces)
loss = compute_loss(image["color"], target)
loss.backward()
```

Requires CUDA and PyTorch3D. Depth and normal maps are not yet implemented for this backend.

### Path-Traced Rendering

```python
renderer = Renderer(method="cycles", raytracing=True, width=1024, height=1024)
```

Blender Cycles settings applied by default:
- OPTIX denoiser
- Noise threshold: 0.01
- Max samples: 100
- Filmic view transform with medium contrast
- Area light at `[-1, 2, -0.5]`, energy 105

### Default Camera and Colors

| Property | Value |
|----------|-------|
| `default_intrinsic` | `[[W, 0, W/2], [0, W, H/2], [0, 0, 1]]` |
| `default_extrinsic` | Look-at from `[1, 0.5, 1]` toward origin |
| `default_mesh_color` | `[0.333, 0.593, 0.666]` (teal) |
| `default_pcd_color` | `[1.0, 0.49, 0.435]` (coral) |
| `default_background_color` | `[0.27, 0.32, 0.37]` (dark slate) |

## Scripts

### vis_data -- Visualize Data Loading

Hydra-configured script that iterates a dataset split and visualizes items interactively (via Open3D `Visualize`) or saves them to disk.

```bash
# Interactive visualization of training data
vis_data -cn shapenet_v1 vis.show=true vis.mesh=true

# Save augmented data to disk
vis_data -cn shapenet_v1 vis.save=true vis.split=train

# Visualize specific indices
vis_data -cn shapenet_v1 vis.show=true vis.index=42
vis_data -cn shapenet_v1 vis.show=true vis.index="0,5,10"

# Use DataLoader with batching
vis_data -cn shapenet_v1 vis.show=true vis.use_loader=true
```

Toggleable channels: `vis.inputs`, `vis.occupancy`, `vis.points`, `vis.pointcloud`, `vis.mesh`, `vis.box`, `vis.bbox`, `vis.cam`, `vis.frame`.

### vis_inference -- Visualize Inference Results

Hydra-configured script that loads a trained model, runs generation, optionally computes metrics, and visualizes results.

```bash
# Generate meshes and visualize
vis_inference -cn shapenet_v1 vis.show=true model.weights=path/to/model.pt

# Generate and save meshes + metrics
vis_inference -cn shapenet_v1 vis.save=true model.weights=path/to/model.pt

# With uncertainty visualization (MC Dropout or PSSNet)
vis_inference -cn mugs_pssnet vis.show=true model.weights=path/to/model.pt
```

Supports occupancy contour plots (`vis.plot_contour=true`), uncertainty region extraction, and per-category metric aggregation.

### render -- Render Meshes to Images

Standalone argparse script (no Hydra). Renders `.ply` meshes and their associated input point clouds using Blender Cycles.

```bash
# Render all meshes in a directory
python -m visualize.scripts.render path/to/results/

# Render a single file
python -m visualize.scripts.render path/to/mesh.ply

# Render mesh and point cloud separately
python -m visualize.scripts.render path/to/results/ --individual

# Render point clouds
python -m visualize.scripts.render path/to/results/ --obj_type pcd --show
```

Expects naming convention: `{name}_mesh.ply`, `{name}_inputs.ply`, `{name}_gt.ply`. Produces front/back renders and a grid composite (`renders_front.png`, `renders_back.png`).

### render_gen -- Render Generated Meshes

Standalone script for rendering specific generated meshes for paper figures. Uses hardcoded paths (intended as a template for figure generation, not general use).

### load_mesh -- Interactive Mesh Inspection

Opens generated meshes alongside ground truth in an Open3D viewer.

```bash
python -m visualize.scripts.load_mesh path/to/generation/meshes/ -md path/to/ShapeNetCore.v1/category/
```

Color coding: gray = prediction, red = uncertain region, green = ground truth.

## Configuration Reference

The `vis` section in Hydra configs controls visualization and generation:

```yaml
vis:
  # Data selection
  split: train                    # Dataset split to visualize
  index: null                     # Specific index or comma-separated list
  use_loader: false               # Use DataLoader (enables batching, collation)

  # Display toggles
  show: false                     # Interactive Open3D display
  save: false                     # Save outputs to disk
  inputs: true                    # Show input point cloud
  occupancy: true                 # Show predicted occupancy
  points: false                   # Show query points
  pointcloud: false               # Show surface point cloud
  mesh: false                     # Show ground truth mesh
  voxels: false                   # Show voxel grid
  box: true                       # Show bounding box
  bbox: false                     # Show tight bounding box
  cam: false                      # Show camera frustum
  frame: true                     # Show coordinate frame

  # Mesh extraction (Generator parameters)
  resolution: 128                 # Grid resolution
  num_query_points: 2097152       # Points batch size (128^3)
  upsampling_steps: null          # MISE steps (null = disabled)
  refinement_steps: 0             # Surface refinement iterations
  normals: false                  # Estimate vertex normals
  colors: false                   # Predict vertex colors
  simplify: null                  # Mesh simplification target

  # Rendering
  renderer: cycles                # Backend: pyrender, open3d, pytorch3d, cycles, eevee
  num_instances: null             # Limit number of instances to process
  render: null                    # Additional render settings
```

## Python API Examples

### Generate and Render Pipeline

```python
import torch
from visualize import Generator, Renderer

model = get_model(cfg).cuda()

generator = Generator(
    model,
    resolution=128,
    threshold=0.5,
    padding=0.1,
)
renderer = Renderer(method="pyrender", width=512, height=512)

inputs = torch.from_numpy(point_cloud).cuda().unsqueeze(0)
mesh = generator.generate_mesh({"inputs": inputs})
image = renderer(mesh.vertices, mesh.faces)["color"]
```

### Batch Generation

```python
from visualize import Generator

generator = Generator(model, resolution=64)

meshes = []
for batch in dataloader:
    inputs = batch["inputs"].cuda()
    for i in range(inputs.shape[0]):
        mesh = generator.generate_mesh({"inputs": inputs[i:i+1]})
        meshes.append(mesh)
```

### Multi-Sample VAE Generation

```python
generator = Generator(model, resolution=128)

# Single sample from posterior
mesh = generator.generate_mesh({"inputs": inputs}, sample=1)

# Multiple samples from posterior
grids = generator.generate_grid({"inputs": inputs}, sample=5)

# Unconditional generation
mesh = generator.generate_mesh({"inputs": inputs}, sample=1, unconditional=True)
```

### Custom Camera Views

```python
import numpy as np
from visualize import Renderer

renderer = Renderer(method="pyrender")

intrinsic = np.array([
    [500, 0, 256],
    [0, 500, 256],
    [0, 0, 1]
])
pose = np.eye(4)
pose[:3, 3] = [0, 0, 2]  # 2m away, OpenGL convention

image = renderer(mesh.vertices, mesh.faces, intrinsic=intrinsic, extrinsic=pose)["color"]
```

### Per-Instance Grid Generation (Multi-Object Scenes)

```python
generator = Generator(model, resolution=128, padding=0.1)

# Coarse prediction -> per-instance bounding box -> fine prediction
instance_grids = generator.generate_grid_per_instance(
    {"inputs": inputs},
    threshold=0.5,
    return_meta=True,   # Returns dict with grid, voxel_size, center, instance_idx
)

# Extract meshes with per-instance colors
meshes = generator.extract_meshes(instance_grids)
```

### Using the Visualizer Convenience Class

```python
from visualize.src.visualizer import Visualizer

vis = Visualizer(
    output_dir=Path("output/"),
    resolution=128,
    method="pyrender",
    width=512,
    height=512,
)
vis.generator = model  # Initializes Generator internally

# Generate and render in one go
mesh = vis.get_mesh(item={"inputs": inputs})
image = vis.get_image([mesh])
vis.save(Path("output/result"), mesh=mesh, image=image)
```

## Performance Tips

1. **Use MISE for high resolution** -- avoids querying the full dense grid:
   ```python
   generator = Generator(model, resolution=32, upsampling_steps=3)  # effective 256
   ```

2. **Reduce batch size for memory** -- splits grid point queries:
   ```python
   generator = Generator(model, points_batch_size=16384)
   ```

3. **Use PyRender for fast previews** -- rasterization, no GPU needed:
   ```python
   renderer = Renderer(method="pyrender", offscreen=True)
   ```

4. **Use Cycles only for final renders** -- path tracing is 10-100x slower:
   ```python
   renderer = Renderer(method="cycles", raytracing=True)
   ```

5. **scikit-image Marching Cubes** -- when you need gradient direction control or vertex normals from the isosurface extraction itself:
   ```python
   generator = Generator(model, use_skimage=True)
   ```
