# shape-completion-train

PyTorch Lightning-based training infrastructure for shape completion models. Provides data loading, training loops, callbacks for mesh generation/evaluation, and distributed training support.

## Installation

```bash
# As submodule

# Dependencies (from main repo)
uv sync --extra train
```

## Quick Start

```bash
# Train a model
train -cn shapenet_v1

# Resume from checkpoint
train -cn shapenet_v1 train.resume=true

# Distributed training (SLURM)
srun --nodes=2 --gpus-per-node=4 train -cn shapenet_v1
```

The `train` CLI entry point is defined in `pyproject.toml` as `train.scripts.train:main`. The script uses [Hydra](https://hydra.cc/) for configuration, with the base config at `conf/config.yaml` and experiment-specific configs in `conf/` subdirectories (selected via `-cn`).

## Architecture

### Core Components

```
train/
├── scripts/
│   └── train.py              # Main training entry point (Hydra)
├── src/
│   ├── __init__.py            # Package exports
│   ├── model.py               # LitModel - Lightning wrapper
│   ├── data_module.py         # LitDataModule - data loading
│   ├── schedulers.py          # Learning rate schedulers
│   ├── utils.py               # Helpers: collate fns, save/load, test dataset
│   └── callbacks/
│       ├── __init__.py        # Re-exports all callbacks
│       ├── every_n.py         # EveryNCallback - base class for periodic callbacks
│       ├── ema.py             # EMACallback - NVIDIA NeMo EMA (optimizer-level)
│       ├── generate_meshes.py # GenerateMeshesCallback - mesh extraction via MISE
│       ├── visualize.py       # VisualizeCallback - rendering + W&B/TensorBoard upload
│       ├── eval_meshes.py     # EvalMeshesCallback - mesh quality metrics
│       └── test.py            # TestMeshesCallback - test-set mesh evaluation
└── tests/
```

### LitModel (`src/model.py`)

The `LitModel` class wraps any model from the `models/` submodule and handles:

- **Training loop**: Loss computation via model's `loss()` method or built-in per-architecture dispatch
- **Validation loop**: Evaluation via model's `evaluate()` method or built-in dispatch
- **EMA tracking**: Exponential moving average via `torch.optim.swa_utils.AveragedModel`
- **Optimizer config**: Configured externally by the training script (defaults to AdamW)
- **Gradient monitoring**: Logs gradient and weight norms at configurable verbosity levels
- **Non-finite loss detection**: Raises on NaN/Inf loss in `on_before_backward`

```python
from train.src.model import LitModel

lit_model = LitModel(
    name="onet",                     # Architecture name
    output_dir="logs/onet",          # Save directory
    model=model,                     # From models.get_model()
    optimizer=optimizer,             # Pre-configured optimizer (default: AdamW)
    scheduler=scheduler,             # Optional LR scheduler
    interval="epoch",                # Scheduler step interval: "epoch" | "step"
    frequency=1,                     # Scheduler step frequency
    hypergradients=False,            # Enable hypergradient optimizer (gdtuo)
    monitor="val/f1",                # Metric for checkpointing/early stopping
    metrics=None,                    # Restrict logged metrics to this list
    threshold=0.5,                   # Occupancy threshold
    regression=False,                # SDF regression mode (vs. binary occupancy)
    loss=None,                       # Loss function name passed to model.loss()
    reduction="mean",                # Loss reduction method
    points_batch_size=None,          # Query points per batch for model.loss()
    sync_dist=False,                 # Sync metrics across GPUs (auto-set for multi-GPU)
    ema=0.999,                       # EMA decay rate (None to disable)
)
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `training_step()` | Calls `model.loss()` if available, otherwise dispatches per architecture. Logs `train/loss` and any model-reported metrics. |
| `validation_step()` | Calls `model.evaluate()` if available, otherwise dispatches per architecture. Uses EMA model when available. |
| `configure_optimizers()` | Returns optimizer (and scheduler config if set). Supports per-param-group LR adjustment. |
| `on_before_optimizer_step()` | Logs gradient/weight norms (requires `DEBUG_LEVEL_1` or higher verbosity). |
| `on_before_zero_grad()` | Updates EMA model parameters. |
| `on_before_backward()` | Validates loss is finite; raises `ValueError` on NaN/Inf. |
| `forward()` | Routes inference through EMA model when not training. |
| `on_save_checkpoint()` / `on_load_checkpoint()` | Persists and restores EMA state dict. |

### LitDataModule (`src/data_module.py`)

Wraps train/val/test datasets into a PyTorch Lightning data module:

```python
from train.src.data_module import LitDataModule

datamodule = LitDataModule(
    train=train_dataset,
    val=val_dataset,              # Optional (required if overfit=False)
    test=test_dataset,            # Optional (required for test_dataloader)
    batch_size=32,
    batch_size_val=32,
    num_workers=8,
    num_workers_val=8,            # Separate worker count for validation
    shuffle_val=False,            # Shuffle validation data
    overfit=False,                # Use training set for validation (debug)
    prefetch_factor=2,            # Prefetch batches per worker
    pin_memory=False,             # Pin memory for GPU transfer
    weighted=False,               # Weighted sampling by category
    seed=0,                       # RNG seed for reproducibility
    collate_fn=collate_fn,        # Custom collation function
    cache=False,                  # Use SharedDataLoader for in-memory caching
    hash_items=False,             # Hash items for SharedDataLoader cache keys
    share_memory=False,           # Share arrays across workers (distributed)
)
```

**Features:**
- Weighted random sampling for imbalanced datasets (requires `category_weights` on the dataset)
- `SharedDataLoader` for in-memory caching with optional cross-worker shared memory
- Persistent workers with configurable prefetching
- Custom collate functions: `common_collate` (intersection of keys), `heterogeneous_collate` (PyTorch3D padding for variable-length inputs), `coco_collate`
- Test dataloader always uses `batch_size=1`

### Learning Rate Schedulers (`src/schedulers.py`)

**LinearWarmupCosineAnnealingLR:**

```python
from train.src.schedulers import LinearWarmupCosineAnnealingLR

scheduler = LinearWarmupCosineAnnealingLR(
    optimizer,
    warmup_iters=1000,           # Linear warmup steps
    total_iters=100000,          # Total training steps
    warmup_start_lr=0.0,         # Starting LR (default: 0)
    min_lr=1e-6,                 # Final LR after cosine decay (float or list)
)
```

`min_lr` accepts a float (applied proportionally to all param groups) or a list (one value per param group). When a single float is given and param groups have different base LRs, the ratio `min_lr / max(base_lrs)` is applied to each group.

The training script also supports `ReduceLROnPlateau` and `StepLR` from PyTorch, selected via the `train.scheduler` config key.

## Callbacks

### Callback Hierarchy

All periodic callbacks inherit from `EveryNCallback`, which provides flexible scheduling:

```
EveryNCallback          # Base: run every N steps, epochs, or evaluations
  └─ GenerateMeshesCallback   # Mesh extraction via MISE
       └─ VisualizeCallback   # Rendering + upload
            ├─ EvalMeshesCallback  # Mesh quality metrics
            └─ TestMeshesCallback  # Test-set evaluation
```

`EveryNCallback` supports:
- `n_steps`: Run every N training steps
- `n_epochs`: Run every N epochs
- `n_evals`: Run every N validation runs (int) or at specific eval indices (list of ints)
- `first` / `last`: Control whether to run on the first/last occurrence

### GenerateMeshesCallback

Generates meshes from model predictions using MISE (Multi-resolution ISOsurface Extraction):

```python
from train.src.callbacks import GenerateMeshesCallback

callback = GenerateMeshesCallback(
    every_n_evals=5,             # Generate every N validation runs
    resolution=128,              # MISE grid resolution
    padding=0.1,                 # Mesh padding
    threshold=0.5,               # Occupancy threshold
    points_batch_size=None,      # Query points per batch (default: resolution^3)
    precision=None,              # Override autocast precision
)
```

Automatically adjusts resolution if the model has a `resolution` attribute. Uses `lightning.Fabric` for precision control during generation.

### VisualizeCallback

Renders meshes, point clouds, and model outputs; logs to TensorBoard or uploads to W&B:

```python
from train.src.callbacks import VisualizeCallback

callback = VisualizeCallback(
    every_n_evals=5,
    n_per_category=4,            # Samples per category (must be 1 or even)
    n_total=None,                # Total samples cap (at least one of n_per_category/n_total required)
    meshes=True,                 # Render generated meshes
    inputs=True,                 # Render input point clouds alongside meshes
    logits=False,                # Render model logits as colored point clouds
    render=None,                 # Model render mode: "color" | "normals" | "mesh" | None
    front=True,                  # Render front view
    back=False,                  # Render back view
    upload_to_wandb=False,       # Upload to W&B (otherwise saves to disk + TensorBoard)
    width=512,                   # Render width in pixels
    height=512,                  # Render height in pixels
    show=False,                  # Interactive matplotlib display
    progress=False,              # Show progress bars during generation
    resolution=128,
    padding=0.1,
    points_batch_size=None,
    threshold=0.5,
    precision=None,
)
```

Stacks images per category when `n_per_category >= 4`. Supports distributed gathering of items across GPUs.

### EvalMeshesCallback

Evaluates generated meshes against ground truth:

```python
from train.src.callbacks import EvalMeshesCallback

callback = EvalMeshesCallback(
    every_n_evals=5,
    n_per_category=None,         # Samples per category
    n_total=456,                 # Total samples (default: 57*8; must satisfy n_total*12 >= 2048 for FID)
    upload_to_wandb=False,
    metrics="all",               # String: "all" | "mesh" | "pcd" | "fid" | "kid" | "clip"
    fid_stats_name=None,         # Pre-computed FID reference stats name (cleanfid)
    prefix="val/mesh/",          # Metric logging prefix
    num_workers=None,             # Parallel workers for mesh evaluation (joblib)
    progress=False,
    resolution=128,
    padding=0.1,
    points_batch_size=None,
    threshold=0.5,
    precision=None,
)
```

**Evaluation modes:**
- `"mesh"` or `"all"`: Chamfer distance, F-score, precision, recall against ground truth meshes
- `"pcd"`: Point cloud metrics against ground truth point clouds
- `"fid"` / `"kid"`: Render-based FID/KID using cleanfid (requires pre-computed stats; uses icosphere viewpoints)
- `"clip"`: CLIP-based FID using `clip_vit_b_32`

Mesh evaluation runs in parallel via joblib. FID renders are stored in a temporary directory cleaned up on callback destruction.

### EMACallback

Optimizer-level EMA implementation (from NVIDIA NeMo). Wraps each optimizer in an `EMAOptimizer` that maintains a shadow copy of parameters:

```python
from train.src.callbacks import EMACallback

callback = EMACallback(
    decay=0.999,
    validate_original_weights=False,  # If True, validate with original weights (not EMA)
    every_n_steps=1,                  # Apply EMA update every N optimizer steps
    cpu_offload=False,                # Offload EMA weights to CPU
)
```

**Note:** The training script currently uses `torch.optim.swa_utils.AveragedModel` for EMA (configured via `model.average=ema` in the config), with the `EMACallback` import present but commented out. Both implementations are available; the callback approach is useful when optimizer-level weight swapping is needed.

### TestMeshesCallback

Runs test-set mesh evaluation and visualization after training:

```python
from train.src.callbacks import TestMeshesCallback

callback = TestMeshesCallback(
    test=True,                   # Run mesh evaluation against ground truth
    meshes=True,                 # Generate meshes
    inputs=True,                 # Include input visualization
    front=True,                  # Render front view
    back=True,                   # Render back view
    upload_to_wandb=False,
    width=512,
    height=512,
    show=False,
    resolution=128,
    padding=0.1,
    points_batch_size=None,
    threshold=0.5,
    precision=None,
)
```

Evaluates each test sample individually (batch_size=1). Looks for pose files and ground truth mesh paths in the input directory. Results are printed as a table (via `tabulate`) and optionally uploaded to W&B or saved as `test/stats.txt`.

## Configuration

All defaults are in `conf/config.yaml`. Experiment configs in `conf/` subdirectories override these via Hydra.

### Training Config

```yaml
train:
  epochs: 10                    # Max epochs
  batch_size: 32                # Training batch size
  lr: 3.125e-6                  # Base learning rate (scaled by batch_size * GPUs * nodes * accum if scale_lr=true)
  min_lr:                       # Minimum LR for schedulers (default: lr / 10)
  scale_lr: true                # Scale LR with effective batch size

  # Optimizer
  optimizer: AdamW              # AdamW | Adam | SGD | AdamW8bit
  betas: [0.9, 0.999]           # Adam betas
  weight_decay: 0.01            # L2 regularization (uses param groups: decay vs no-decay)

  # Scheduler
  scheduler:                    # LinearWarmupCosineAnnealingLR | ReduceLROnPlateau | StepLR | null
  warmup_frac: 0.0033           # Warmup fraction of total steps (for LinearWarmupCosineAnnealingLR)
  lr_reduction_factor: 0.5      # Factor for ReduceLROnPlateau
  lr_step_size: 10              # Step size for StepLR
  lr_gamma: 0.9                 # Gamma for StepLR

  # Training options
  loss:                         # Loss function name (passed to model.loss())
  reduction: mean               # Loss reduction
  gradient_clip_val:            # Gradient clipping (norm-based; null = disabled)
  accumulate_grad_batches: 1
  precision: 32-true            # 32-true | 16-mixed | bf16-mixed
  num_batches:                  # Limit training batches per epoch (null = all)

  # Checkpointing / resume
  resume: false                 # Resume from last checkpoint
  skip: false                   # Skip training, just run test
  model_selection_metric: val/f1 # Metric for checkpointing and early stopping

  # Early stopping
  early_stopping: false         # Enable early stopping
  patience_factor: 10           # Patience = patience_factor * val_freq (clamped to [5, epochs/2])

  # Auto-tuning
  find_lr: false                # Run Lightning LR finder before training
  find_batch_size: false        # Run Lightning batch size scaler before training

  # Debug
  overfit_batches: false        # Overfit on N batches (or false)
  fast_dev_run: false           # Quick test run (1 train + 1 val batch)
  detect_anomaly: false         # Enable autograd anomaly detection
  hypergradients: false         # Use hypergradient optimizer (gdtuo)
```

### Validation Config

```yaml
val:
  batch_size: ${train.batch_size}  # Validation batch size (defaults to training batch size)
  freq: 1                       # Validate every N epochs (or fraction < 1 for intra-epoch)
  num_batches: ${train.num_batches}  # Limit validation batches (null = all)
  num_sanity: 2                 # Sanity validation steps before training

  # Visualization
  visualize: false              # Enable VisualizeCallback
  vis_n:                        # Total samples to visualize (null = unlimited)
  vis_n_category: 4             # Samples per category
  vis_n_eval: 5                 # Visualize every N validation runs

  # Mesh evaluation
  mesh: false                   # Mesh metrics: false | list of ["chamfer", "f1", "fid", "kid", "all", ...]
  num_query_points: 100000      # Points for evaluation
```

### Model Averaging Config

```yaml
model:
  average:                      # ema | swa | null
  ema_decay:                    # EMA decay (auto-computed from total steps if null)
  swr_lr:                       # SWA learning rate (default: 100x base LR)
  compile: false                # torch.compile (mode="max-autotune")
```

When `model.average=ema` and `ema_decay` is null, the decay is automatically computed as `1 - 100 / steps_rounded` (clamped to [0.9, 0.9999]).

### Logging Config

```yaml
log:
  wandb: false                  # Use Weights & Biases (otherwise TensorBoard)
  offline: false                # Offline W&B logging
  project: ${hydra:job.config_name}  # W&B project name (defaults to config file name)
  name: ${model.arch}           # Run name (defaults to model architecture)
  id:                           # W&B run ID (for resuming)
  version:                      # TensorBoard version string

  freq: 10                      # Log every N steps
  verbose: false                # Verbosity level (false/0, true/1, 2, 3)
  progress: rich                # Progress bar: rich | tqdm | false
  profile: false                # Enable PyTorch Lightning profiler

  # W&B extras
  gradients: false              # Log gradient histograms
  parameters: false             # Log parameter histograms
  graph: false                  # Log model graph
  model: false                  # Log model checkpoints to W&B

  # Checkpointing
  top_k: 1                      # Keep top K checkpoints
  summary_depth: 2              # Model summary depth
  metrics:                      # Restrict logged metrics to this list (null = all)
```

### Data Loading Config

```yaml
load:
  num_workers: -1               # DataLoader workers (-1 = auto from cpu_count)
  prefetch_factor: 2            # Prefetch batches per worker
  pin_memory: true              # Pin memory for GPU transfer
  weighted:                     # Weighted sampling by category

data:
  cache: false                  # Cache data in memory (SharedDataLoader)
  hash_items: false             # Hash items for caching
  share_memory: false           # Share memory across workers in distributed setting
```

## Distributed Training

### SLURM Integration

The training script auto-detects SLURM via `SLURM_JOB_NAME` and `SLURM_JOB_NUM_NODES` environment variables:

```bash
# Single node, 4 GPUs
srun --gres=gpu:4 --ntasks-per-node=4 train -cn shapenet_v1

# Multi-node (2 nodes, 4 GPUs each)
srun --nodes=2 --gres=gpu:4 --ntasks-per-node=4 train -cn shapenet_v1
```

**Auto-scaling:**
- Effective batch size = `batch_size * gpus * nodes * accumulate_grad_batches`
- Learning rate scales with effective batch size when `train.scale_lr=true` (enabled by default)
- Metrics sync across GPUs with `sync_dist=true` (auto-enabled when `num_nodes * num_gpus > 1`)

### DDP Configuration

The DDP strategy is selected automatically:

```python
# Default: "auto" (standard DDP)
# Falls back to "ddp_find_unused_parameters_true" when:
#   - self_condition=true AND self_cond_grad=false
#   - OR vae_freeze=false / cond_freeze=false
trainer = pl.Trainer(
    devices="auto",
    accelerator="auto",
    strategy=strategy,  # "auto" or "ddp_find_unused_parameters_true"
    num_nodes=num_nodes,
    plugins=SLURMEnvironment(),  # Only when SLURM detected
)
```

## Training Script Flow

The main training script (`scripts/train.py`) follows this flow:

```
 1. Load config via Hydra
    |
 2. Create datasets (train/val/test splits)
    |
 3. Create LitDataModule with dataloaders + collate function
    |
 4. Create model via get_model(cfg)
    |
 5. Configure optimizer (AdamW, 8-bit, hypergradients, or model-provided)
    |   - Weight decay uses param groups (decay vs no-decay)
    |   - Optionally scales LR with effective batch size
    |
 6. Configure scheduler (warmup + cosine, plateau, step, or none)
    |
 7. Optionally compile model with torch.compile(mode="max-autotune")
    |
 8. Wrap in LitModel with EMA (if model.average=ema)
    |
 9. Configure callbacks:
    |   - ModelCheckpoint (always)
    |   - RichModelSummary (always)
    |   - LearningRateMonitor (if scheduler)
    |   - RichProgressBar (if log.progress=rich)
    |   - EarlyStopping (if enabled and patience < epochs/3)
    |   - StochasticWeightAveraging (if model.average=swa)
    |   - VisualizeCallback (if val.visualize=true)
    |   - EvalMeshesCallback (if val.mesh is set)
    |
10. Set up logger (WandbLogger or TensorBoardLogger)
    |
11. Create Trainer with distributed config
    |
12. Optionally find LR / batch size (Lightning Tuner)
    |
13. trainer.fit() - main training loop
    |
14. Save best model weights (model_best.pt) + EMA weights (model_ema.pt)
    |
15. Optionally run test evaluation (TestMeshesCallback, single GPU)
```

## Extending

### Adding a Custom Callback

```python
from lightning.pytorch.callbacks import Callback

class MyCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        # Access the underlying model (unwrap compile + LitModel)
        model = pl_module.model.orig_mod

        # Access EMA model (if configured)
        if pl_module.ema_model is not None:
            ema_model = pl_module.ema_model.module.orig_mod

        # Log metrics
        pl_module.log("my_metric", value)
```

### Custom Loss in LitModel

For models with custom loss computation, implement a `loss()` method on the model. LitModel detects `hasattr(self.model, "loss")` and calls it:

```python
class MyModel(Model):
    def loss(self, batch, regression=False, name=None, reduction="mean",
             points_batch_size=None, log_freq=50, global_step=0, total_steps=1000):
        loss = ...

        # Log additional metrics (displayed based on logger verbosity)
        self.add_log("my_metric", value, level=logging.INFO)

        return loss
```

### Custom Evaluation

For models with custom evaluation, implement an `evaluate()` method. LitModel detects `hasattr(self.model, "evaluate")` and calls it:

```python
class MyModel(Model):
    def evaluate(self, batch, name=None, threshold=0.5, regression=False,
                 reduction="mean", metrics=None, points_batch_size=None,
                 global_step=0, total_steps=1000):
        result = {
            "loss": loss_value,
            "iou": iou_value,
            "f1": f1_value,
        }
        return result
```

## Common Recipes

### Train ONet on ShapeNet

```bash
train -cn shapenet_v1 model.arch=onet
```

### Train with EMA and visualization

```bash
train -cn shapenet_v1 model.average=ema val.visualize=true log.wandb=true
```

### Resume training from checkpoint

```bash
# Resume from last checkpoint (auto-detected)
train -cn shapenet_v1 train.resume=true

# Resume from a specific checkpoint
train -cn shapenet_v1 model.checkpoint=path/to/checkpoint.ckpt
```

### Fine-tune from pretrained weights

```bash
train -cn shapenet_v1 model.weights=path/to/model_best.pt train.lr=1e-5
```

### Debug with overfit

```bash
train -cn shapenet_v1 train.overfit_batches=1 train.epochs=100
```

### Multi-GPU training

```bash
# Local (auto-detects all visible GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 train -cn shapenet_v1

# SLURM
srun --gres=gpu:4 --ntasks-per-node=4 train -cn shapenet_v1
```

### Log to Weights & Biases

```bash
# Online logging
train -cn shapenet_v1 log.wandb=true log.project=my-project log.name=my-run

# Offline logging (sync later with `wandb sync`)
train -cn shapenet_v1 log.wandb=true log.offline=true

# Log gradient and parameter histograms
train -cn shapenet_v1 log.wandb=true log.gradients=true log.parameters=true
```

### 8-bit optimizer for large models

```bash
# Requires bitsandbytes
train -cn shapenet_v1 train.optimizer=AdamW8bit
```

### Use cosine annealing with warmup

```bash
train -cn shapenet_v1 train.scheduler=LinearWarmupCosineAnnealingLR train.warmup_frac=0.01
```

### Mixed precision training

```bash
# FP16 mixed precision
train -cn shapenet_v1 train.precision=16-mixed

# BFloat16 mixed precision
train -cn shapenet_v1 train.precision=bf16-mixed
```

### Skip training, run test only

```bash
train -cn shapenet_v1 train.skip=true model.checkpoint=path/to/checkpoint.ckpt test.run=true
```

### Auto-tune learning rate or batch size

```bash
# Find optimal learning rate
train -cn shapenet_v1 train.find_lr=true

# Find maximum batch size
train -cn shapenet_v1 train.find_batch_size=true
```

### Enable mesh evaluation during validation

```bash
# Chamfer + F1 every 5 validations
train -cn shapenet_v1 val.mesh='[chamfer,f1]' val.vis_n_eval=5

# All metrics including FID
train -cn shapenet_v1 val.mesh='[all]' val.vis_n_eval=10
```

## Output Structure

```
logs/
└── shapenet_v1/
    └── onet/
        ├── version_0/
        │   ├── checkpoints/
        │   │   ├── epoch=99-step=10000-val_loss=0.05.ckpt
        │   │   └── last.ckpt
        │   ├── model_best.pt       # Best model weights (unwrapped state dict)
        │   ├── model_ema.pt        # EMA weights (if enabled)
        │   ├── hparams.yaml        # Hyperparameters
        │   ├── vis/                 # Visualization output (if not using W&B)
        │   │   └── step_N/
        │   │       └── category/
        │   │           ├── images/  # Rendered images
        │   │           └── meshes/  # Generated meshes
        │   ├── test/               # Test results (if test.run=true)
        │   │   └── stats.txt       # Per-object and mean metrics
        │   └── events.out.tfevents.* # TensorBoard logs
        └── wandb/                  # W&B logs (if log.wandb=true)
```
