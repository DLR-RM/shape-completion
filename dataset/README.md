# shape-completion-dataset

Dataset loading and transformation infrastructure for 3D shape completion. Provides a unified interface for loading diverse 3D datasets with configurable data fields and augmentation pipelines.

## Installation

```bash
# As submodule

# Dependencies (from main repo)
uv sync --extra dataset
```

## Quick Start

```python
from dataset import get_dataset, get_transformations

# Using factory (recommended)
datasets = get_dataset(cfg, splits=("train", "val"))
train_dataset = datasets["train"]
val_dataset = datasets["val"]

# With transforms
transforms = get_transformations(cfg, split="train")
```

## Architecture

### Core Components

```
dataset/
├── __init__.py           # get_dataset(), get_transformations() factories
├── src/
│   ├── __init__.py       # Public exports
│   ├── fields.py         # Field classes for data loading (15 fields)
│   ├── transforms.py     # Transform classes for augmentation (59 transforms)
│   ├── shared.py         # SharedDataset, SharedDataLoader (shared-memory data loading)
│   ├── utils.py          # Helper functions, TorchvisionDatasetWrapper
│   ├── tv_transforms.py  # Torchvision transforms (NormalizeDepth, CenterPad, CameraIntrinsic)
│   │
│   ├── shapenet.py       # ShapeNet dataset
│   ├── bop.py            # BOP challenge datasets
│   ├── ycb.py            # YCB object dataset
│   ├── tabletop.py       # Tabletop scenes
│   ├── completion3d.py   # Completion3D benchmark
│   ├── modelnet.py       # ModelNet dataset
│   ├── graspnet.py       # GraspNet dataset
│   ├── coco.py           # COCO instance segmentation
│   └── image.py          # ImageFolderDataset (from NVlabs/edm)
└── tests/
```

### Data Flow

```
Dataset.__getitem__(idx)
    │
    ├── Field.load(obj_dir, index, category)   # Load raw data (point cloud, mesh, image)
    │   └── Caching (optional)                 # functools.cache decorator with deepcopy
    │
    ├── Transform.__call__(data)               # Augmentation pipeline
    │   ├── before_apply(data)                 # Hook before applying to each key
    │   ├── apply(data, key)                   # Core logic, filtered by apply_to
    │   └── after_apply(data)                  # Hook after applying to all keys
    │
    └── Return dict                            # {"inputs": ..., "points": ..., "occ": ...}
```

## Fields System

Fields define how to load specific data types from disk. All fields inherit from the `Field` ABC.

### Base Class

```python
from dataset.src.fields import Field

class Field(ABC):
    def __init__(self, cachable: bool = True, cache: bool = False):
        """
        Args:
            cachable: Whether this field supports caching.
            cache: Enable functools.cache on load() (returns deepcopy of cached result).
        """
        ...

    @abstractmethod
    def load(self, obj_dir: str | Path, index: int, category: int | None) -> dict:
        """Load data from obj_dir. Override in subclasses."""
        raise NotImplementedError
```

### Available Fields (15 total)

#### Data Loading Fields

| Field | Description | Output |
|-------|-------------|--------|
| `PointCloudField` | Load point clouds (.npz, .npy) | `{None: (N,3), "normals": (N,3)}` |
| `PointsField` | Load query points with occupancy | `{None: (M,3), "occ": (M,)}` |
| `MeshField` | Load meshes (.obj, .off, .ply) | `{None: trimesh.Trimesh}` |
| `ImageField` | Load images | `{None: (H,W,C)}` |
| `DepthField` | Load depth maps with camera params | `{None: (H,W), "intrinsic": ..., "extrinsic": ...}` |
| `RGBDField` | Load RGB-D pairs | `{"image": (H,W,3), "depth": (H,W)}` |
| `BlenderProcRGBDField` | BlenderProc rendered RGB-D | Multiple views with camera params |
| `BOPField` | BOP format loader | Scene data dict |
| `VoxelsField` | Load voxel grids (.binvox) | `{None: (D,H,W)}` |
| `PartNetField` | PartNet part annotations | `{"points": ..., "labels": ...}` |
| `DTUField` | DTU multi-view dataset loader | Multi-view images with cameras |

#### Utility Fields

| Field | Description |
|-------|-------------|
| `RandomField` | Return random subset from wrapped field (with optional weights) |
| `MixedField` | Combine multiple fields (merge or priority-select) |
| `EmptyField` | Return empty dict |
| `IndexField` | Return item index, name, and path |

### Field Configuration

```yaml
data:
  fields:
    inputs:
      type: PointCloudField
      file_name: pointcloud.npz
      file_key: points
      with_normals: false

    points:
      type: PointsField
      file_name: points.npz
      points_key: points
      occ_key: occupancies
      num_samples: 2048

    mesh:
      type: MeshField
      file_name: mesh.obj
```

### Custom Field Example

```python
from dataset.src.fields import Field

class MyField(Field):
    def __init__(self, file_name: str, cache: bool = False):
        super().__init__(cachable=True, cache=cache)
        self.file_name = file_name

    def load(self, obj_dir: str | Path, index: int, category: int | None = None) -> dict:
        file_path = Path(obj_dir) / self.file_name
        data = np.load(file_path)
        return {None: torch.from_numpy(data)}
```

## Transforms System

Transforms augment and process data. All transforms inherit from the `Transform` ABC. There are **59 concrete transforms** organized into categories below.

### Base Class

```python
from dataset.src.transforms import Transform

class Transform(ABC):
    def __init__(
        self,
        apply_to: str | list[str] | tuple[str, ...] | None = None,
        allowed: str | list[str] | tuple[str, ...] | None = None,
        cachable: bool = False,
    ):
        """
        Args:
            apply_to: Key(s) to apply transform to. None = apply to all keys.
                      Automatically converted to a set for filtering.
            allowed: Valid keys for apply_to. Defaults to standard set:
                     {"inputs", "inputs.depth", "inputs.normals", "inputs.image",
                      "points", "pointcloud", "pointcloud.normals",
                      "mesh.vertices", "mesh.normals", "voxels", "bbox",
                      "partnet.points"}
            cachable: Whether this transform's output can be cached.
        """

    def before_apply(self, data: dict) -> dict:
        """Hook before applying to each key."""
        return data

    @abstractmethod
    def apply(self, data: dict, key: str | None) -> dict:
        """Override to implement transform logic."""
        raise NotImplementedError

    def after_apply(self, data: dict) -> dict:
        """Hook after applying to all keys."""
        return data
```

### Transform Categories

#### Geometric Transforms

| Transform | Description | Key Params |
|-----------|-------------|------------|
| `Rotate` | Random rotation | `axes`, `angles`, `from_inputs` |
| `Affine` | Affine transformation (from extrinsics) | `replace` |
| `Translate` | Translation | offset range |
| `Scale` | Uniform/non-uniform scaling | scale range |
| `Normalize` | Center and scale to unit sphere | `center`, `scale`, `to_front`, `reference`, `scale_method` |
| `ApplyPose` | Apply rigid pose transform | |
| `RefinePose` | Refine pose via ICP | ICP params |
| `RefinePosePerInstance` | Per-instance pose refinement | ICP params |

#### Point Cloud Augmentation

| Transform | Description | Key Params |
|-----------|-------------|------------|
| `SubsamplePointcloud` | Random subsampling | `num_samples` |
| `SubsamplePoints` | Subsample query points | `num_samples`, `in_out_ratio` |
| `AddGaussianNoise` | Add Gaussian noise | `std` |
| `CropPointcloud` | Crop to bounding box | bounds |
| `CropPointcloudWithMesh` | Crop point cloud using mesh | |
| `CropPoints` | Crop query points | `padding` |
| `AxesCutPointcloud` | Planar cut | `axes`, `cut_ratio`, `rotate_object` |
| `SphereCutPointcloud` | Spherical cut | `radius` |
| `SphereMovePointcloud` | Move points along sphere surface | |
| `ProcessPointcloud` | Downsample + outlier removal | `downsample`, `remove_outlier` |
| `RemoveHiddenPointsFromInputs` | Hidden point removal | viewpoint |
| `DepthLikePointcloud` | Simulate depth-sensor partial view | `rotate_object`, `upper_hemisphere` |
| `RotatePointcloud` | Rotate point cloud specifically | `axes`, `angles` |

#### Rendering Transforms

| Transform | Description | Key Params |
|-----------|-------------|------------|
| `Render` | Render mesh to images (pyrender) | `width`, `height` |
| `RenderPointcloud` | Render point cloud | resolution |
| `RenderDepthMaps` | Multi-view depth rendering | `num_views` |
| `DepthToPointcloud` | Unproject depth to 3D | intrinsics |
| `ShadingImageFromNormals` | Generate shading from normals | |

#### Mesh Transforms

| Transform | Description |
|-----------|-------------|
| `NormalizeMesh` | Normalize mesh to unit cube |
| `RotateMesh` | Rotate mesh |
| `PointcloudFromMesh` | Sample surface points from mesh |
| `PointsFromMesh` | Sample query points (with occupancy) from mesh |
| `PointsFromPointcloud` | Sample query points from point cloud |

#### Noise and Artifacts

| Transform | Description |
|-----------|-------------|
| `EdgeNoise` | Add noise at depth discontinuities |
| `ImageBorderNoise` | Add noise at image borders |
| `AngleOfIncidenceRemoval` | Remove points by viewing angle |

#### Image Transforms

| Transform | Description |
|-----------|-------------|
| `ImageToTensor` | Convert images to normalized tensors (with optional resize/crop) |
| `Torchvision` | Wrap any torchvision transform |

#### Spatial Encoding Transforms

| Transform | Description |
|-----------|-------------|
| `VoxelizePointcloud` | Voxelize point cloud |
| `VoxelizePoints` | Voxelize query points |
| `BPS` | Basis Point Set encoding |
| `BoundingBox` | Compute bounding box from reference |
| `BoundingBoxJitter` | Jitter bounding box |

#### Data Transforms

| Transform | Description |
|-----------|-------------|
| `SdfFromOcc` | Convert occupancy to SDF/TSDF |
| `SegmentationFromPartNet` | Generate segmentation labels from PartNet |
| `NormalsCameraCosineSimilarity` | Compute normal-camera cosine similarity |
| `InputsNormalsFromPointcloud` | Extract normals for inputs from point cloud |
| `Permute` | Permute tensor dimensions |
| `MinMaxNumPoints` | Enforce min/max point counts (pad or subsample) |
| `LoadUncertain` | Load uncertainty data |
| `FindUncertainPoints` | Identify uncertain query points |
| `SplitData` | Split data dict into sub-dicts |
| `Compress` | Compress arrays (float16, packbits) |
| `Unpack` | Unpack compressed arrays |
| `CheckDtype` | Validate/convert tensor dtypes |

#### Control Flow Transforms

| Transform | Description |
|-----------|-------------|
| `Return` | Return specific keys only |
| `RandomChoice` | Randomly select one transform from a list |
| `RandomApply` | Randomly apply a transform with probability |
| `KeysToKeep` | Filter output to specified keys |

#### Debug Transforms

| Transform | Description |
|-----------|-------------|
| `Visualize` | Debug visualization (plotly) |
| `SaveData` | Save intermediate results to disk |

### Transform Pipeline Example

```python
from dataset.src.transforms import (
    SubsamplePointcloud,
    AddGaussianNoise,
    Rotate,
    Normalize,
)

transforms = [
    # Subsample input point cloud to 2048 points
    SubsamplePointcloud(apply_to="inputs", num_samples=2048),

    # Add noise only during training
    AddGaussianNoise(apply_to="inputs", std=0.01),

    # Random rotation around Z axis
    Rotate(apply_to=["inputs", "points"], axes="z", angles=360),

    # Normalize to unit sphere
    Normalize(apply_to=["inputs", "points", "mesh.vertices"]),
]
```

### Using `apply_transforms`

```python
from dataset.src.transforms import apply_transforms

# Apply a list of transforms with per-transform timing logs
data = apply_transforms(data, transforms)
```

## Available Datasets

The `get_dataset()` factory routes by `cfg.data.train_ds` / `cfg.data.val_ds` / `cfg.data.test_ds` names. Supported dataset identifiers:

| Identifier | Class | Notes |
|------------|-------|-------|
| `shapenet*` | `ShapeNet` | Any name containing "shapenet" (e.g., `shapenet_v1`, `shapenet_v2`) |
| `completion3d` | `Completion3D` | Stanford Completion3D benchmark |
| `ycb` | `YCB` | YCB object dataset (train/val/test + real-data mode) |
| `modelnet40` | `ModelNet` | ModelNet40 classification dataset |
| `mnist`, `fmnist`, `cifar10` | torchvision wrappers | Image classification datasets |
| `coco` | `CocoInstanceSegmentation` | COCO instance segmentation |
| `tabletop*` | `TableTop` | Any name containing "tabletop" |
| `graspnet*` | `GraspNetEval` | Any name starting with "graspnet" |
| `bop_*` | `BOP` | BOP challenge (test only): `bop_ycbv`, `bop_lm`, `bop_hb`, `bop_tyol` |
| Other | `ShapeNet` | Falls back to ShapeNet-style loading with custom `cfg.dirs[ds]` |

Additional dataset classes available for direct use:
- `ImageFolderDataset` — zip/folder image dataset (from NVlabs/edm)
- `SharedDataset` / `SharedDataLoader` — shared-memory wrappers for distributed training

### ShapeNet

```python
from dataset.src.shapenet import ShapeNet

dataset = ShapeNet(
    root="/path/to/shapenet",
    split="train",              # train | val | test
    categories=["chair", "table"],  # or None for all
    fields={"inputs": field, "points": field},
    transforms=transforms,
)
```

**Categories:** 57 ShapeNet categories supported (see `CATEGORIES_MAP` in shapenet.py)

### BOP (Benchmark for 6D Object Pose)

```python
from dataset.src.bop import BOP

dataset = BOP(
    root="/path/to/bop",
    dataset="ycbv",            # ycbv | lm | tless | itodd | ...
    split="train_pbr",
    fields=fields,
)
```

### YCB

```python
from dataset.src.ycb import YCB

dataset = YCB(
    root="/path/to/ycb",
    split="train",
    objects=["002_master_chef_can", "003_cracker_box"],
)
```

### TableTop

Custom tabletop scene dataset with rendered views.

```python
from dataset.src.tabletop import TableTop

dataset = TableTop(
    root="/path/to/tabletop",
    split="train",
    scene_type="single",       # single | multi | clutter
)
```

### Completion3D

Stanford Completion3D benchmark dataset.

```python
from dataset.src.completion3d import Completion3D

dataset = Completion3D(
    root="/path/to/completion3d",
    split="train",
)
```

### ModelNet

```python
from dataset.src.modelnet import ModelNet

dataset = ModelNet(
    root="/path/to/modelnet",
    version=40,                # 10 | 40
    split="train",
)
```

### GraspNet

```python
from dataset.src.graspnet import GraspNetEval

dataset = GraspNetEval(
    root="/path/to/graspnet",
    split="test",
)
```

## Configuration

All configuration uses [Hydra](https://hydra.cc/). Config files live in `conf/` at the main repo root.

### Dataset Factory Config

```yaml
data:
  train_ds: shapenet_v1        # Dataset identifier(s) — can be a list for multi-dataset
  val_ds: null                 # Defaults to train_ds if null
  test_ds: null                # Defaults to val_ds if null
  categories: [chair, table]   # Categories to load (null = all)
  cache: false                 # Cache loaded data
  hash_items: false            # Use hashed item paths
  sdf_from_occ: false          # Convert occupancy to SDF
  dither: false                # Dither float32 tensors during training
```

### Input Configuration

```yaml
inputs:
  type: pointcloud            # pointcloud | depth | image | rgbd | partial | depth_like |
                              # kinect | shading | normals | render
  dim: 3                      # Point dimension (3=xyz, 6=xyz+normals)
  num_points: 2048            # Points to load
  project: false              # Project depth to point cloud
  cache: false                # Cache rendered inputs
  load_random: true           # Random view selection
  voxelize: 0                 # Voxelize inputs (0 = disabled, else resolution)
  permute: false              # Permute tensor dimensions
  bbox: false                 # Compute bounding box
  min_num_points: 0           # Minimum points (pad if fewer)
  max_num_points: 0           # Maximum points (subsample if more)

  # Image inputs
  width: 640
  height: 480
  resize: null                # Resize dimensions
  crop: 0                     # Center crop size
  normals: false              # Load normal maps

  # BPS encoding
  bps:
    num_points: 0
    resolution: 0
    method: null
    feature: null
    basis: null

  # FPS sampling
  fps:
    num_points: 0
```

### Point Cloud Configuration

```yaml
pointcloud:
  from_mesh: false            # Sample from mesh instead of loading
  normals: false              # Load normals
  bbox: false                 # Compute bounding box
  train:
    num_points: 100000        # Surface points for training
  val:
    num_points: 100000
```

### Query Points Configuration

```yaml
points:
  dim: 3                      # Query point dimension
  from_mesh: false            # Sample from mesh
  from_pointcloud: false      # Sample from point cloud
  subsample: true             # Enable subsampling
  crop: false                 # Crop to bounds
  voxelize: 0                 # Voxelize (0 = disabled)
  cache: false                # Cache
  bbox: false                 # Compute bounding box
  min_num_points: 0           # Minimum points
  train:
    num_samples: 2048         # Query points per sample
    ratio: 0.5                # Surface vs volume ratio
  val:
    num_samples: 100000
```

### Augmentation Configuration

```yaml
aug:
  rotate: z                   # Rotation axes (x|y|z|xy|xyz|cam|none)
  scale: [0.9, 1.1]           # Scale range
  translate: 0.1              # Translation range
  noise: 0.01                 # Gaussian noise std

  # Point cloud specific
  downsample: false           # Enable downsampling
  remove_hidden: false        # Remove occluded points
  upper_hemisphere: true      # Camera in upper hemisphere only
  remove_outlier: false       # Statistical outlier removal
  move_sphere: false          # Sphere-based point movement
  bbox_jitter: 0              # Bounding box jitter amount

  # Depth specific
  edge_noise: false           # Add edge artifacts
  remove_angle: false         # Remove by incidence angle
  border_noise: false         # Add border noise to images

train:
  no_aug: false               # Disable augmentation
val:
  no_aug: true
test:
  no_aug: true
```

### Normalization Configuration

```yaml
norm:
  center: ""                  # Center axes (e.g., "xyz")
  scale: false                # Scale to unit sphere
  to_front: false             # Rotate to front
  offset: null                # Translation offset
  true_height: false          # Use true height for normalization
  reference: null             # Reference for normalization (null | "mesh" | "pointcloud")
  method: null                # Scale method
  padding: 0.1                # Padding for bounding box
```

### Mesh Configuration

```yaml
mesh:
  norm: true                  # Normalize mesh
  rot: null                   # Pre-rotation angles [x, y, z]
  bbox: false                 # Compute bounding box
```

## Adding a New Dataset

### Step 1: Create the dataset module

Create `dataset/src/mydataset.py`:

```python
from pathlib import Path
from torch.utils.data import Dataset
from .transforms import Transform, apply_transforms

class MyDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        fields: dict | None = None,
        transforms: list[Transform] | None = None,
        categories: list | None = None,
    ):
        self.root = Path(root)
        self.split = split
        self.fields = fields or {}
        self.transforms = transforms or []

        # Build item list
        self.items = self._load_split()

    def _load_split(self):
        """Load split file or scan directory."""
        split_file = self.root / f"{self.split}.lst"
        if split_file.exists():
            return split_file.read_text().strip().split("\n")
        return list(self.root.glob("*"))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        path = self.root / item

        # Load fields
        data = {}
        for name, field in self.fields.items():
            data[name] = field.load(str(path), idx, category=None)

        # Apply transforms (with timing logs)
        data = apply_transforms(data, self.transforms)

        return data
```

### Step 2: Export in `dataset/src/__init__.py`

```python
from .mydataset import MyDataset
```

### Step 3: Add routing in `dataset/__init__.py:get_dataset()`

The factory routes by matching `cfg.data.train_ds` string patterns:

```python
# In the train/val/test loops inside get_dataset():
elif ds == "mydataset":
    data = MyDataset(
        root=cfg.dirs[ds],
        split=split,
        fields=fields,
        transforms=get_transformations(cfg, split),
    )
```

Pattern matching rules in `get_dataset()`:
- `"shapenet" in ds` — any name containing "shapenet"
- `"tabletop" in ds` — any name containing "tabletop"
- `ds.startswith("graspnet")` — any name starting with "graspnet"
- `"bop" in ds` — BOP datasets (test split only)
- Exact match for `completion3d`, `ycb`, `modelnet40`, `mnist`, `fmnist`, `cifar10`, `coco`
- Fallback: treats as ShapeNet-style dataset with `cfg.dirs[ds]`

### Step 4: Create Hydra config

Create `conf/mydataset.yaml`:

```yaml
defaults:
  - config
  - _self_

data:
  train_ds: mydataset
  val_ds: mydataset
```

Add the data root to `conf/dirs/default.yaml`:

```yaml
mydataset: /path/to/mydataset
```

## Adding a New Transform

### Step 1: Implement the transform

Add to `dataset/src/transforms.py`:

```python
class MyTransform(Transform):
    @get_args()  # Captures constructor args for serialization/logging
    def __init__(
        self,
        my_param: float = 1.0,
        apply_to: str | list[str] | None = None,
        cachable: bool = False,
    ):
        super().__init__(apply_to=apply_to, cachable=cachable)
        self.my_param = my_param

    def apply(self, data: DataDict, key: str | None) -> DataDict:
        # key is None when apply_to=None (applies to whole dict)
        # key is a string when apply_to is set (e.g., "inputs", "points")
        if key is not None:
            data[key] = data[key] * self.my_param
        return data
```

### Step 2: Add to `__all__`

At the bottom of `transforms.py`, add `"MyTransform"` to the `__all__` list. This auto-exports it through `dataset/src/__init__.py`.

### Step 3: Import in `dataset/__init__.py`

Add to the explicit import block:

```python
from .src.transforms import MyTransform
```

### Step 4: Wire into `get_transformations()` (optional)

If the transform should be automatically included based on config flags, add it to the `get_transformations()` function in `dataset/__init__.py`:

```python
if cfg.aug.my_flag:
    transformations.append(MyTransform(apply_to="inputs", my_param=cfg.aug.my_param))
```

## Caching

### Field-level caching

Fields support automatic caching via the `cache` constructor parameter. The cache stores results with `functools.cache` and returns deep copies to prevent mutation:

```python
field = PointCloudField(file_name="pointcloud.npz", cache=True)
# First call loads from disk, subsequent calls return deepcopy of cached result
```

### Transform-level caching

Some transforms support the `cachable` flag, which signals to the `SharedDataset` infrastructure that their output can be stored in shared memory:

```python
transform = VoxelizePointcloud(apply_to="inputs", resolution=32, cachable=True)
```

## Shared-Memory Data Loading

`SharedDataset` and `SharedDataLoader` enable caching dataset items in shared memory for distributed training:

```python
from dataset.src.shared import SharedDataset, SharedDataLoader

shared_ds = SharedDataset(dataset)
loader = SharedDataLoader(shared_ds, batch_size=32)
```

Configure via:
```yaml
load:
  share_memory: true
```

## Performance Tips

1. **Use weighted sampling** for imbalanced categories:
   ```yaml
   load:
     weighted: true
   ```

2. **Enable data caching** for small datasets:
   ```yaml
   data:
     cache: true
   ```

3. **Reduce query points** during validation:
   ```yaml
   points:
     train:
       num_samples: 2048
     val:
       num_samples: 100000  # More for accurate eval
   ```

4. **Use SharedDataLoader** for distributed training:
   ```yaml
   load:
     share_memory: true
   ```

5. **Use voxelization** for fixed-size inputs:
   ```yaml
   inputs:
     voxelize: 32            # Produces (32, 32, 32) voxel grid
   ```

6. **Filter output keys** to reduce memory:
   ```yaml
   load:
     keys_to_keep: [inputs, points, occ]
   ```
