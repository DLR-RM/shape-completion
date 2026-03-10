# shape-completion-models

Machine learning models for 3D shape completion. This submodule implements 30+ architectures spanning implicit functions, point cloud completion, diffusion models, transformers, and autoencoders.

## Installation

```bash
# As submodule

# Dependencies (from main repo)
uv sync --extra models
```

## Quick Start

```python
from models import get_model, ONet, ConvONet, PCN

# Using the factory (recommended)
model = get_model(cfg)  # Routes based on cfg.model.arch

# Direct instantiation
model = ONet(arch="onet", dim=3, inputs_type="pointcloud")
```

## Architecture

### Model Factory

All model instantiation flows through `get_model(cfg)` in `__init__.py`. This 600+ line function:

1. Reads `cfg.model.arch` to select architecture
2. Applies architecture-specific config options
3. Optionally patches NeRF positional encoding
4. Patches attention backend (torch/xformers/einops)
5. Loads weights/checkpoint if specified
6. Wraps in DVR (differentiable volume rendering) if `cfg.implicit.dvr`

### Base Class

All models inherit from `Model` (`src/model.py`), which extends `nn.Module` with:

- `setup()` / `teardown()` lifecycle hooks
- Fuzzy state dict loading (handles renamed keys)
- Logging dict for metrics
- EMA statistics tracking

### Key Interface

For implicit models (occupancy prediction):
```python
def forward(self, inputs: Tensor, points: Tensor) -> Tensor:
    """
    Args:
        inputs: Conditioning input (B, N, D) - point cloud, image features, etc.
        points: Query locations (B, M, 3) where to predict occupancy
    Returns:
        logits: Occupancy predictions (B, M, 1) or (B, M, C) for multi-class
    """
```

For point completion models:
```python
def forward(self, inputs: Tensor) -> Tensor:
    """
    Args:
        inputs: Partial point cloud (B, N, 3)
    Returns:
        completed: Dense point cloud (B, M, 3)
    """
```

## Available Models

> **Arch string matching:** `get_model()` uses substring/prefix matching for some architectures.
> For example, `conv_onet` matches any arch containing `"conv_onet"`, `dino_inst` matches any
> arch containing `"dino_inst"`, and `onet` matches any arch containing `"onet"` (but is checked
> after `conv_onet`). Order matters -- see `__init__.py` for the full dispatch chain.

### Implicit Shape Completion (Occupancy/SDF)

| Model | Config `arch` | Paper | Description |
|-------|---------------|-------|-------------|
| `ONet` | `onet` | [Occupancy Networks](https://arxiv.org/abs/1812.03828) | Point cloud encoder + MLP decoder |
| `ConvONet` | `conv_onet*` | [Convolutional Occupancy Networks](https://arxiv.org/abs/2003.04618) | Multi-scale 3D convolutions |
| `IFNet` | `if_net` | [IF-Net](https://arxiv.org/abs/2003.01456) | Multi-scale features with displacements |
| `VQDIF` | `vqdif` | [ShapeFormer](https://arxiv.org/abs/2201.10326) | VQ-encoded implicit function |
| `ShapeFormer` | `shapeformer` | [ShapeFormer](https://arxiv.org/abs/2201.10326) | Transformer on VQ codes |
| `DMTet` | `dmtet*` | [DMTet](https://arxiv.org/abs/2111.04276) | Differentiable marching tetrahedra |
| `CompleTr` | `completr` | Custom | XDConv encoder + transformer decoder |
| `PointTransformer` | `point_tr*` | Custom | Self-attention on point queries |
| `ImplicitNetwork` | `idr` | [IDR](https://arxiv.org/abs/2003.09852) | Implicit differentiable renderer |

### Point Cloud Completion

| Model | Config `arch` | Paper | Description |
|-------|---------------|-------|-------------|
| `PCN` | `pcn` | [Point Completion Network](https://arxiv.org/abs/1808.00671) | Coarse-to-fine generation |
| `PSGN` | `psgn` | [PSGN](https://arxiv.org/abs/1612.00603) | Point set generation |
| `SnowflakeNet` | `snowflakenet` | [SnowflakeNet](https://arxiv.org/abs/2108.04444) | Hierarchical point splitting |
| `PVDModel` | `pvd`, `pcd_diffusion`, `point_diffusion` | [PVD](https://arxiv.org/abs/2104.03670) | Point cloud diffusion |

### Diffusion Models

| Model | Config `arch` | Description |
|-------|---------------|-------------|
| `DiffusersModel` | `diffusers` | HuggingFace diffusers integration |
| `UNetModel` | `unet` | 3D UNet with timestep conditioning |
| `GridDiffusionModel` | `grid_diffusion` | Voxel grid diffusion |
| `LatentDiffusionModel` | `ldm`, `latent_diffusion` | VAE + transformer diffusion |
| `EDMPrecond` | `ldm` (with `ldm_arch: precond`) | EDM preconditioning wrapper around VAE |
| `EDMTransformer` | (used internally by LDM) | Elucidated diffusion transformer denoiser |

### Autoregressive / VQ Models

| Model | Config `arch` | Description |
|-------|---------------|-------------|
| `Shape3D2VecSet` | `3dshape2vecset` | Set-to-set transformer for occupancy prediction |
| `Shape3D2VecSetCls` | `3dshape2vecset_cls` | Shape classification variant |
| `Shape3D2VecSetVAE` | `3dshape2vecset_vae` | VAE variant for latent shape representation |
| `Shape3D2VecSetVQVAE` | `3dshape2vecset_vqvae` | VQ-VAE variant with discrete codebook |
| `LatentAutoregressiveModel` | `larm`, `latent_ar_model`, `latent_autoregressive_model` | GPT-based autoregressive generation on VQ codes |
| `LatentGPT` | (used by LARM) | Causal transformer backbone for LARM |
| `VAEModel` | (base class) | Variational autoencoder base class |
| `VQVAEModel` | (base class) | Vector-quantized VAE base class |

### Vision Transformer Variants (DINOv2 backbone)

The `dino*` and `dino_inst*` arch strings route to different classes based on `cfg.inputs.type` and other flags. The backbone defaults to `dinov2_vits14` but can be overridden via the `backbone` config key.

| Model | Config `arch` | Routing condition | Description |
|-------|---------------|-------------------|-------------|
| `Dino3D` | `dino*` | depth/kinect + `project: true` | 3D occupancy from projected depth features |
| `DinoRGB` | `dino*` | image/rgb or `project: false` | Occupancy prediction from RGB images |
| `DinoRGBD` | `dino*` | rgbd | RGB-D fusion for occupancy |
| `DinoInstSeg` | `dino_inst*` | image (default) | 2D instance segmentation with DINOv2 features |
| `DinoInstSeg3D` | `dino_inst*` | depth/kinect + `project: true` | 3D instance segmentation from projected depth |
| `DinoInstSegRGBD` | `dino_inst*` | rgbd | Instance segmentation with RGB-D fusion |
| `DinoInst3D` | `dino_inst*` | `load_3d: true` | Multi-view 3D instance segmentation |
| `DinoCls` | (direct import only) | N/A | DINOv2 classification head (not in `get_model`) |
| `InstOccPipeline` | `inst_pipe*` | N/A | Pipeline combining instance segmentation + occupancy |

### Classification / Segmentation

| Model | Config `arch` | Paper | Description |
|-------|---------------|-------|-------------|
| `PointNetCls` | `pointnet` (with `cls.num_classes`) | [PointNet](https://arxiv.org/abs/1612.00593) | Point cloud classification |
| `PointNetSeg` | `pointnet` (with `seg.num_classes`) | [PointNet](https://arxiv.org/abs/1612.00593) | Point cloud segmentation |
| `MaskRCNN` | `mask_rcnn` | [Mask R-CNN](https://arxiv.org/abs/1703.06870) | Instance segmentation (torchvision, optional import) |

### Point Feature Extractors (encoder components)

These are not standalone models accessed via `get_model()` but encoder modules used internally by other architectures.

| Module | Used by | Paper |
|--------|---------|-------|
| `DGCNN` | PVD, other encoders | [DGCNN](https://arxiv.org/abs/1801.07829) |
| `PVCNN` | PVD point-voxel encoder | [PVCNN](https://arxiv.org/abs/1907.03739) |
| `PointNet++` | Various encoders | [PointNet++](https://arxiv.org/abs/1706.02413) |

### Uncertainty / Flow Models

| Model | Config `arch` | Description |
|-------|---------------|-------------|
| `MCDropoutNet` | `dropout*` | Monte Carlo dropout for uncertainty |
| `PSSNet` | `pssnet` | Probabilistic symmetric shape |
| `RealNVP` | `realnvp` | Normalizing flow |

### Other

| Model | Config `arch` | Description |
|-------|---------------|-------------|
| `PIFu` | `pifu` | Pixel-aligned implicit functions for human reconstruction |
| `DVR` | (wrapper via `implicit.dvr: true`) | Differentiable volume rendering, wraps any implicit model |

## Configuration Options

### Core Model Config

```yaml
model:
  arch: onet              # Architecture name (see tables above)
  weights: null           # Path to pretrained weights
  checkpoint: null        # Path to training checkpoint
  load_best: false        # Load model_best.pt from log dir

  # Architecture options
  norm: null              # Normalization: null | batch | layer | group | rms
  activation: relu        # Activation: relu | gelu | silu | mish
  dropout: 0.0            # Dropout rate
  bias: true              # Use bias in linear layers

  # Attention (for transformer models)
  attn_backend: torch     # torch | xformers | einops
  attn_mode: null         # Attention mode override
  attn: true              # Enable attention layers

  # Compilation
  compile: false          # Use torch.compile

  # Loss
  reduction: mean         # Loss reduction: mean | sum | none
```

### Input Config

```yaml
inputs:
  type: pointcloud        # pointcloud | depth | image | rgbd | kinect
  dim: 3                  # Input dimension (3 for xyz, 6 for xyz+normals)
  num_points: 2048        # Number of input points
  nerf: false             # Apply NeRF positional encoding to inputs
  project: false          # Project depth to 3D points

  fps:
    num_points: 512       # FPS downsampling target
```

### Query Points Config

```yaml
points:
  dim: 3                  # Query point dimension
  nerf: false             # NeRF encoding for query points
  voxelize: null          # Voxel grid resolution (e.g., 32, 64)
```

### Implicit Function Config

```yaml
implicit:
  threshold: 0.5          # Occupancy threshold for mesh extraction
  dvr: false              # Wrap model in DVR

  # DVR options (when dvr: true)
  near: 0.1               # Near plane
  far: 2.0                # Far plane
  num_steps: 128          # Ray marching steps
  step_func: linear       # Step function: linear | exponential
```

## Architecture-Specific Options

### ConvONet (`conv_onet*`)

```yaml
condition: add            # Feature conditioning: add | concat
sample_mode: bilinear     # Grid sampling: bilinear | nearest
padding_mode: zeros       # Padding: zeros | border | reflection
```

### IFNet (`if_net`)

```yaml
displacements: true       # Use displacement vectors
multires: true            # Multi-resolution features
pvconv: false             # Use PVConv encoder
```

### CompleTr (`completr`)

```yaml
encoder: unetxd           # Encoder type
decoder: transformer      # Decoder type
n_layer: 5                # Transformer layers
n_head: 4                 # Attention heads
self_attn: false          # Self-attention in decoder
cross_attn: true          # Cross-attention to encoder
```

### PointTransformer (`point_tr*`)

```yaml
n_embd: 512               # Embedding dimension
n_layer: 8                # Transformer layers
n_head: 8                 # Attention heads
enc_type: enc             # Encoder type
dec_type: dec             # Decoder type
use_linear_attn: false    # Use linear attention
```

### Diffusion Models (`unet`, `diffusers`)

```yaml
scheduler: ddpm           # ddpm | ddim | edm
num_train_timesteps: 1000
num_inference_steps: 100
beta_schedule: linear     # linear | cosine | squaredcos_cap_v2
prediction_type: epsilon  # epsilon | v_prediction | sample
self_condition: false     # Self-conditioning
zero_snr: false           # Zero SNR terminal
```

### Latent Diffusion (`ldm`)

```yaml
vae_arch: 3dshape2vecset_vae  # VAE architecture
vae_weights: path/to/vae.pt   # Pretrained VAE
ldm_arch: transformer         # Denoiser: transformer | precond
vae_freeze: true              # Freeze VAE during training
bit_diffusion: false          # Diffusion on VQ indices
```

### Shape3D2VecSet variants

```yaml
n_latent: 512             # Latent dimension
n_layer: 24               # Transformer layers
n_embd: 512               # Embedding dimension
n_queries: 512            # Number of query tokens
activation: geglu         # Activation function

# VQ-VAE specific
n_code: 16384             # Codebook size
quantizer: vq             # vq | fsq | lfq
decay: 0.8                # EMA decay for codebook
```

### Latent Autoregressive (`larm`)

```yaml
vae_arch: 3dshape2vecset_vqvae  # Discretizer architecture
vae_weights: path/to/vqvae.pt   # Pretrained VQ-VAE
ar_arch: transformer             # Autoregressor: transformer
vae_freeze: true                 # Freeze discretizer during training
objective: causal                # Training objective
```

### Grid Diffusion (`grid_diffusion`)

```yaml
ndim: 3                   # Spatial dimensions (2 or 3)
channels: 1               # Input channels
resolution: 32            # Voxel grid resolution
rescale_skip: true        # Rescaled skip connections
```

## Adding a New Model

1. Create `models/src/mymodel.py`:

```python
from torch import nn, Tensor
from .model import Model

class MyModel(Model):
    def __init__(self, dim: int = 3, **kwargs):
        super().__init__()
        self.encoder = ...
        self.decoder = ...

    def encode(self, inputs: Tensor) -> Tensor:
        return self.encoder(inputs)

    def decode(self, points: Tensor, features: Tensor) -> Tensor:
        return self.decoder(points, features)

    def forward(self, inputs: Tensor, points: Tensor) -> Tensor:
        features = self.encode(inputs)
        return self.decode(points, features)
```

2. Export in `models/src/__init__.py`:

```python
from .mymodel import MyModel
```

3. Add routing in `models/__init__.py:get_model()`:

```python
elif arch == "mymodel":
    model = MyModel(dim=cfg.inputs.dim, **kwargs)
```

4. Create config in `conf/`:

```yaml
defaults:
  - config
  - _self_

model:
  arch: mymodel
```

## File Structure

```
models/
├── __init__.py              # get_model() factory, weight loading
├── src/
│   ├── __init__.py          # Public exports
│   ├── model.py             # Base Model class
│   ├── mixins.py            # MultiEvalMixin, MultiLossMixin
│   ├── transformer.py       # Attention, NeRFEncoding, backends
│   ├── utils.py             # Helper functions (loss fns, activation, patch_attention)
│   │
│   │  # Implicit shape completion
│   ├── onet.py              # ONet
│   ├── conv_onet.py         # ConvONet
│   ├── if_net.py            # IFNet
│   ├── vqdif.py             # VQDIF
│   ├── shapeformer.py       # ShapeFormer
│   ├── dmtet.py             # DMTet
│   ├── completr.py          # CompleTr
│   ├── point_transformer.py # PointTransformer
│   ├── idr.py               # ImplicitNetwork (IDR)
│   ├── pifu.py              # PIFu
│   │
│   │  # Point cloud completion
│   ├── pcn.py               # PCN
│   ├── psgn.py              # PSGN
│   ├── snowflakenet.py      # SnowflakeNet
│   │
│   │  # Diffusion models
│   ├── diffusion/
│   │   ├── __init__.py
│   │   ├── model.py         #   Base DiffusionModel class + bit encoding utils
│   │   ├── unet.py          #   3D UNet denoiser
│   │   ├── diffusers.py     #   HuggingFace diffusers integration (optional import)
│   │   ├── latent.py        #   Latent diffusion model
│   │   ├── grid.py          #   Grid diffusion model
│   │   ├── edm.py           #   EDM preconditioning (standalone)
│   │   ├── transformer.py   #   EDMTransformer denoiser
│   │   ├── shape3d2vecset.py #  EDMPrecond (wraps VAE + EDM)
│   │   ├── pcd.py           #   PVDModel entry point
│   │   ├── pvd/             #   Point-Voxel Diffusion internals
│   │   │   ├── pvd.py       #     PVD model implementation
│   │   │   ├── modules.py   #     PVD network modules
│   │   │   └── utils.py     #     PVD utilities
│   │   ├── blocks.py        #   Shared UNet building blocks
│   │   └── utils.py         #   Noise schedule and diffusion utilities
│   │
│   │  # Autoregressive / VQ models
│   ├── autoregression/
│   │   ├── __init__.py
│   │   ├── model.py         #   AutoregressiveModel base class
│   │   ├── latent.py        #   LatentAutoregressiveModel
│   │   └── transformer.py   #   LatentGPT causal transformer
│   ├── vae.py               # VAEModel / VQVAEModel base classes
│   ├── shape3d2vecset.py    # Shape3D2VecSet, Cls, VAE, VQVAE variants
│   │
│   │  # DINOv2-based vision models
│   ├── dinov2.py            # Dino3D, DinoRGB, DinoRGBD, DinoInstSeg*, InstOccPipeline
│   ├── dvr.py               # DVR wrapper + RayMarchingConfig
│   │
│   │  # Uncertainty / flow models
│   ├── mc_dropout_net.py    # MCDropoutNet
│   ├── pssnet.py            # PSSNet
│   ├── realnvp.py           # RealNVP
│   │
│   │  # Encoder / backbone modules (used internally)
│   ├── pointnet.py          # PointNetCls, PointNetSeg
│   ├── pointnetpp.py        # PointNet++ wrapper
│   ├── dgcnn.py             # DGCNN encoder
│   ├── pvcnn.py             # PVCNN encoder
│   ├── resnet.py            # ResNet backbone
│   ├── dpt.py               # Dense Prediction Transformer
│   ├── fpn.py               # Feature Pyramid Network
│   ├── hourglass.py         # Stacked hourglass network
│   ├── mask_rcnn.py         # Mask R-CNN (torchvision, optional import)
│   ├── grid.py              # Grid feature extraction
│   ├── voxel.py             # Voxel-based operations
│   └── xdconf.py            # XDConv encoder for CompleTr
│
└── tests/                   # Per-module unit tests
```

## Attention Backend System

The `transformer.py` module provides a unified attention interface with multiple backends:

```python
from models.src.transformer import Attention, patch_attention

# Attention auto-selects best available backend
attn = Attention(dim=512, num_heads=8)

# Or patch an entire model
model = patch_attention(model, backend="xformers")
```

**Backends:**
- `torch`: PyTorch native (always available)
- `xformers`: Memory-efficient attention (requires xformers, CUDA SM >= 7.0)
- `einops`: einops-based implementation

**NeRF Encoding:**

```python
from models.src.transformer import NeRFEncoding, TCNN_EXISTS

# Positional encoding for coordinates
enc = NeRFEncoding(in_dim=3, num_frequencies=6)
encoded = enc(points)  # (B, N, 3) -> (B, N, 39)

# TCNN backend (faster, requires tiny-cuda-nn)
if TCNN_EXISTS:
    enc = NeRFEncoding(in_dim=3, backend="tcnn")
```
