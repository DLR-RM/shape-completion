# shape-completion-eval

Evaluation and mesh generation infrastructure for shape completion models. Provides scripts for generating meshes from trained models and computing standard 3D reconstruction metrics.

## Installation

```bash
# As submodule

# Dependencies (from main repo)
uv sync --extra eval
```

## Quick Start

```bash
# Generate meshes from trained model
generate -cn shapenet_v1 model.weights=path/to/model_best.pt

# Evaluate generated meshes against ground truth
mesh_eval -cn shapenet_v1 test.dir=path/to/meshes

# Combined generation + evaluation
gen_eval -cn shapenet_v1 model.weights=path/to/model_best.pt
```

## Architecture

```
eval/
├── __init__.py               # Public exports (eval_mesh_pcd, src.utils.*)
├── scripts/
│   ├── generate.py           # Mesh generation from model (entry point: generate)
│   ├── mesh_eval.py          # Per-mesh quality evaluation (entry point: mesh_eval)
│   ├── gen_eval.py           # Generation quality metrics (entry point: gen_eval)
│   ├── eval.py               # Occupancy-level evaluation (entry point: evaluate)
│   └── compare_eval.py       # Compare results across experiments (no entry point)
├── src/
│   ├── __init__.py            # Re-exports src.utils.*
│   ├── gen_metrics.py         # Distance functions, COV/MMD, 1-NNA, Hausdorff, ECD
│   ├── prdc.py                # Precision/Recall/Density/Coverage (improved, NAVER)
│   ├── prd.py                 # Precision-Recall Distribution curves (Google)
│   └── utils.py               # eval_pointcloud, eval_occupancy, render_for_fid, helpers
└── tests/
```

## Pipeline Overview

The evaluation pipeline has two stages that can be run independently or combined:

```
                ┌──────────────┐     ┌──────────────┐
Trained Model → │   generate   │ ──→ │  mesh_eval   │ → per-mesh metrics (CSV)
                └──────────────┘     └──────────────┘
                    meshes/              chamfer, IoU, F1, ...

                ┌──────────────┐
Trained Model → │   evaluate   │ → occupancy metrics (CSV)
                └──────────────┘
                    no mesh extraction needed

                ┌──────────────┐
    meshes/   → │   gen_eval   │ → generation quality metrics (TXT)
                └──────────────┘
                    FID, COV, MMD, 1-NNA, ...
```

**`generate`** runs the model on test data and extracts meshes using MISE (Multiresolution IsoSurface Extraction). MISE starts at a coarse resolution (typically 32^3), evaluates the occupancy function, then recursively subdivides voxels near the decision boundary for `upsampling_steps` iterations. Marching Cubes extracts the final mesh from the refined grid. Higher `vis.resolution` and more `vis.upsampling_steps` produce more detailed meshes at the cost of more model queries.

**`mesh_eval`** loads the generated meshes and ground truth, then computes per-instance reconstruction metrics (Chamfer distance, IoU, F1, etc.). Results are saved as both a full pickle and a per-category CSV summary.

**`gen_eval`** evaluates generation quality (diversity and fidelity) by comparing distributions of generated and reference shapes. It supports point-cloud-based metrics (COV, MMD, 1-NNA) and image-based metrics (FID, KID) computed from multi-view renderings.

**`evaluate`** runs model inference directly on query points without mesh extraction, computing occupancy-level metrics. Faster than `generate` + `mesh_eval` for rapid validation.

## Scripts

### generate (Mesh Generation)

Generates meshes from a trained model using MISE + Marching Cubes.

```bash
# Basic usage
generate -cn shapenet_v1 model.weights=path/to/model_best.pt

# With options
generate -cn shapenet_v1 \
    model.weights=path/to/model_best.pt \
    vis.resolution=128 \
    vis.upsampling_steps=2 \
    vis.refinement_steps=0 \
    vis.simplify=10000 \
    vis.normals=true \
    vis.colors=true \
    test.split=test \
    test.overwrite=true \
    vis.num_instances=10
```

If `vis.upsampling_steps` is not set and resolution > 128, it is computed automatically as `log2(resolution) - log2(32)`. The effective resolution becomes `2^(log2(32) + upsampling_steps)`.

**Output structure:**
```
logs/<project>/<name>/generation/<split>/
├── meshes/
│   └── <category>/
│       ├── object_name.ply
│       └── ...
├── inputs/                     # Copy of input point clouds
│   └── <category>/
│       └── object_name.ply
└── vis/                        # Visual examples (first few per category)
    └── <category>/
        ├── 00_mesh.ply
        └── 00_inputs.ply
```

### mesh_eval (Per-Mesh Evaluation)

Evaluates generated meshes against ground truth point clouds and occupancy.

```bash
# Basic evaluation
mesh_eval -cn shapenet_v1

# Evaluate meshes from a specific directory
mesh_eval -cn shapenet_v1 test.dir=path/to/meshes

# Show individual results interactively (requires display)
mesh_eval -cn shapenet_v1 vis.show=true
```

Requires that `generate` has been run first (or meshes exist at the expected path). Computes all available per-mesh metrics: point cloud metrics (Chamfer-L1/L2, F1, precision, recall, normals) and occupancy metrics (IoU, F1, precision, recall).

When multiple mesh variants exist per instance (e.g., `model_0.ply`, `model_1.ply`, ...), the script evaluates all variants and reports the best result, plus computes TMD (Total Mutual Difference) and UHD (Unidirectional Hausdorff Distance) across variants.

**Output files:**
- `<ds>_<split>_mesh_eval_full_<threshold>.pkl` -- full per-instance results
- `<ds>_<split>_mesh_eval_<threshold>.csv` -- per-category summary
- `<ds>_<split>_mesh_eval_<threshold>.txt` -- formatted table

### gen_eval (Generation Quality)

Evaluates generation quality by comparing distributions of generated and reference shapes. Supports both point-cloud-based and image-based metrics.

```bash
# Default: Chamfer-based COV/MMD/1-NNA + FID
gen_eval -cn shapenet_v1 model.weights=path/to/model_best.pt

# Specific metrics
gen_eval -cn shapenet_v1 \
    model.weights=path/to/model_best.pt \
    test.metrics=[chamfer,emd,fid,kid]

# Feature-based metrics (requires trained feature extractor)
gen_eval -cn shapenet_v1 \
    model.weights=path/to/model_best.pt \
    test.metrics=[fpd]

# With CLIP-based FID and PRD/PRDC
gen_eval -cn shapenet_v1 \
    model.weights=path/to/model_best.pt \
    test.metrics=[clip_fid,prdc]

# Pre-compute reference FID stats only (useful for shared baselines)
gen_eval -cn shapenet_v1 stats_only=true

# Control rendering views for FID
gen_eval -cn shapenet_v1 \
    model.weights=path/to/model_best.pt \
    views=sdfstylegan  # Options: icosphere (12), dodecahedron/sdfstylegan (20), 3dshape2vecset (10)
```

**Available `test.metrics` values:**
- `chamfer` -- Chamfer-based COV, MMD, 1-NNA, ECD
- `emd` -- EMD-based COV, MMD, 1-NNA, ECD
- `feat` -- Feature-space COV, MMD, 1-NNA, ECD (requires trained encoder)
- `fpd` -- FPD, KPD, 3D Precision/Recall, 3D PRDC (requires trained encoder)
- `fid` -- Frechet Inception Distance from multi-view renderings
- `kid` -- Kernel Inception Distance from multi-view renderings
- `clip_fid` -- CLIP-based FID
- `prd` -- Precision-Recall Distribution from multi-view features
- `prdc` -- improved Precision/Recall/Density/Coverage from multi-view features

### eval (Occupancy Evaluation)

Evaluates model predictions directly on query points without mesh extraction.

```bash
evaluate -cn shapenet_v1 \
    model.weights=path/to/model_best.pt

# With extended metrics
evaluate -cn shapenet_v1 \
    model.weights=path/to/model_best.pt \
    test.basic=false \
    vis.num_query_points=100000
```

Setting `test.basic=false` adds AUPRC (area under precision-recall curve) and ECE (expected calibration error).

### compare_eval (Compare Results)

Visualizes and compares evaluation results across experiments using Plotly. No console entry point; run as a module.

```bash
# Compare by searching a directory recursively
python -m eval.scripts.compare_eval \
    --directory logs/ \
    --cls mean \
    --metric f1

# Compare specific result files
python -m eval.scripts.compare_eval \
    --files model1/test_eval_0.50.csv model2/test_eval_0.50.csv \
    --metric iou
```

## Metrics Reference

### Per-Mesh Point Cloud Metrics

Computed by `eval_pointcloud()` in `eval.src.utils`. Samples points from predicted mesh and compares to ground truth point cloud.

| Metric | Key | Description |
|--------|-----|-------------|
| Chamfer-L1 | `chamfer-l1` | L1 Chamfer distance: mean of completeness + accuracy (lower is better) |
| Chamfer-L2 | `chamfer-l2` | L2 Chamfer distance: mean of squared completeness + squared accuracy (lower is better) |
| F1 Score | `pcd_f1` | Harmonic mean of precision and recall at distance threshold (default 0.01) |
| Precision | `pcd_precision` | Fraction of predicted points within threshold of a GT point |
| Recall | `pcd_recall` | Fraction of GT points within threshold of a predicted point |
| Normal Consistency | `normals` | Mean absolute dot product between predicted and GT normals (requires normals) |

### Per-Mesh Occupancy Metrics

Computed by `eval_occupancy()` in `eval.src.utils`. Checks whether query points are inside/outside the predicted mesh.

| Metric | Key | Description |
|--------|-----|-------------|
| IoU | `iou` | Intersection over Union of predicted vs. GT occupancy |
| F1 Score | `f1` | Harmonic mean of occupancy precision and recall |
| Precision | `precision` | Fraction of predicted occupied points that are truly occupied |
| Recall | `recall` | Fraction of truly occupied points that are predicted occupied |
| Accuracy | `acc` | Overall classification accuracy |

### Multi-Shape Diversity Metrics (mesh_eval)

Computed when multiple mesh variants exist per instance.

| Metric | Key | Description |
|--------|-----|-------------|
| TMD | `tmd` | Total Mutual Difference: average pairwise Chamfer distance between variants |
| UHD | `uhd` | Unidirectional Hausdorff Distance: max nearest-neighbor distance from input to predictions |

### Generation Quality Metrics (gen_eval)

Computed by `eval.src.gen_metrics` and `eval.scripts.gen_eval`. Compare distributions of generated vs. reference shapes.

| Metric | Key | Description |
|--------|-----|-------------|
| Coverage | `cov` | Fraction of reference shapes matched by at least one generated shape (higher is better) |
| MMD | `mmd` | Minimum Matching Distance: mean nearest-neighbor distance from reference to generated (lower is better) |
| 1-NNA | `1-nna` | 1-Nearest Neighbor Accuracy: 0.5 = indistinguishable distributions (closer to 0.5 is better) |
| ECD | `ecd` | Edge Count Difference: two-sample test based on minimum spanning trees |
| Hausdorff | `hausdorff` | Directed Hausdorff distance between point clouds |
| FID | `fid` | Frechet Inception Distance from multi-view renderings (lower is better) |
| KID | `kid` | Kernel Inception Distance from multi-view renderings (lower is better) |
| CLIP-FID | `clip_fid` | FID computed with CLIP features instead of Inception |
| FPD | `fpd` | Frechet Point-cloud Distance: FID in 3D feature space (lower is better) |
| KPD | `kpd` | Kernel Point-cloud Distance: KID in 3D feature space (lower is better) |
| PRD | `prd` | Precision-Recall Distribution curves from embeddings ([Sajjadi et al. 2018](https://arxiv.org/abs/1806.00035)) |
| PRDC | `prdc` | Improved Precision/Recall/Density/Coverage ([Naeem et al. 2020](https://arxiv.org/abs/2002.09797)) |

## Python API

### Evaluating a Mesh

```python
from eval import eval_mesh_pcd
from trimesh import Trimesh

mesh = Trimesh(vertices, faces)
item = {
    "pointcloud": gt_pointcloud,  # (N, 3) ground truth
    "points": query_points,       # (M, 3) query points
    "points.occ": occupancies,    # (M,) ground truth occupancy
}

results = eval_mesh_pcd(mesh, item, n_points=100000)
# Returns: {'chamfer-l1': ..., 'chamfer-l2': ..., 'iou': ..., 'f1': ..., ...}
```

### Computing Distance Metrics

```python
from eval.src.gen_metrics import distance_fn, DistanceMetrics

# Chamfer distance between two point clouds
cd = distance_fn(cloud1, cloud2, metric=DistanceMetrics.CHAMFER)

# Earth Mover's Distance
emd = distance_fn(cloud1, cloud2, metric=DistanceMetrics.EMD)

# F1 Score (negative, for use in distance matrices)
f1 = distance_fn(cloud1, cloud2, metric=DistanceMetrics.F1)
```

### Coverage and MMD

```python
from eval.src.gen_metrics import paired_distances, cov_mmd, DistanceMetrics

# Compute pairwise distances between generated and reference sets
distances = paired_distances(generated_clouds, reference_clouds,
                            metric=DistanceMetrics.CHAMFER)

# Compute COV and MMD from distance matrix
cov, mmd = cov_mmd(distances, num_points=2048)
```

### Point Cloud Evaluation

```python
from eval.src.utils import eval_pointcloud

result = eval_pointcloud(
    pointcloud_pred,    # (N, 3) predicted
    pointcloud_gt,      # (M, 3) ground truth
    normals_pred=None,  # (N, 3) optional
    normals_gt=None,    # (M, 3) optional
    method="tensor",    # "tensor" (default, GPU) | "kdtree" | "faiss"
)
# Returns: {'chamfer-l1': ..., 'chamfer-l2': ..., 'f1': ..., 'precision': ..., 'recall': ..., ...}
```

### Occupancy Evaluation

```python
from eval.src.utils import eval_occupancy
import torch

result = eval_occupancy(
    occ_pred,   # (N,) predicted occupancy probabilities
    occ_gt,     # (N,) ground truth occupancy (0/1)
    threshold=0.5,
)
# Returns: {'iou': ..., 'f1': ..., 'precision': ..., 'recall': ..., 'acc': ..., 'tp': ..., ...}
```

### PRDC (Precision/Recall/Density/Coverage)

```python
from eval.src.prdc import compute_prdc

metrics = compute_prdc(
    real_features,   # (N, D) feature embeddings of real samples
    fake_features,   # (N, D) feature embeddings of generated samples
    nearest_k=5,
)
# Returns: {'precision': ..., 'recall': ..., 'density': ..., 'coverage': ...}
```

## Configuration

### Test Configuration

```yaml
test:
  run: true                   # Run test evaluation
  split: test                 # Dataset split
  dir: null                   # Directory with generated meshes (mesh_eval)
  filename: null              # Results filename
  overwrite: false            # Overwrite existing results
  merge: false                # Merge with existing results
  metrics: [chamfer, fid]     # Metrics for gen_eval
  batch_size: null            # Batch size (auto: 512 for EMD, 1024 otherwise)
  num_instances: null         # Limit instances per category
  basic: true                 # false adds AUPRC and ECE (eval only)
```

### Visualization / Generation Configuration

```yaml
vis:
  resolution: 128             # MISE target resolution (32, 64, 128, 256)
  upsampling_steps: null      # MISE upsampling steps (auto if null and resolution > 128)
  refinement_steps: 0         # Surface refinement iterations post-extraction
  num_query_points: 2097152   # Points per batch for model queries (default: resolution^3)
  simplify: null              # Mesh simplification target face count (null = no simplification)
  normals: false              # Estimate and save vertex normals
  colors: false               # Predict and save vertex colors
  num_instances: null         # Max instances per category to generate
  index: null                 # Specific dataset indices to generate (int, comma-separated, or list)
  show: false                 # Interactive visualization (requires display)
  save: false                 # Save raw data for later visualization
```

### Implicit Function Configuration

```yaml
implicit:
  threshold: 0.5              # Occupancy decision boundary (0.5 for occupancy, 0.0 for SDF)
  sdf: false                  # Whether model predicts signed distance (vs. occupancy)
```

## Output Format

### Mesh Evaluation Results

Results are saved as CSV with columns:

```
category,name,chamfer-l1,chamfer-l2,pcd_f1,pcd_precision,pcd_recall,iou,f1,precision,recall
chair,model_001,0.0023,0.0001,0.892,0.901,0.883,0.785,0.832,0.845,0.819
...
```

### Summary Statistics

Per-category and overall statistics (macro = mean of category means, micro = mean of all instances):

```
                chamfer-l1  chamfer-l2    pcd_f1       iou
category
chair             0.00234     0.00012   0.8921    0.7854
table             0.00189     0.00009   0.9102    0.8123
...
mean (macro)      0.00215     0.00011   0.9012    0.7989
mean (micro)      0.00210     0.00010   0.8998    0.7950
```

## Parallel Evaluation

Mesh evaluation supports parallel processing via `joblib`:

```python
from joblib import Parallel, delayed
from eval.scripts.mesh_eval import single_eval

results = Parallel(n_jobs=-1)(
    delayed(single_eval)(item, save_dir, dataset)
    for item in tqdm(dataloader)
)
```

## Command Line Examples

### Full Evaluation Pipeline

```bash
# 1. Train model
train -cn shapenet_v1 model.arch=onet

# 2. Generate meshes
generate -cn shapenet_v1 \
    model.weights=logs/shapenet_v1/onet/version_0/model_best.pt \
    vis.resolution=128

# 3. Evaluate meshes (per-instance reconstruction quality)
mesh_eval -cn shapenet_v1 \
    test.dir=logs/shapenet_v1/onet/version_0/meshes

# 4. Evaluate generation quality (distribution-level)
gen_eval -cn shapenet_v1 \
    model.weights=logs/shapenet_v1/onet/version_0/model_best.pt \
    test.metrics=[chamfer,fid]
```

### Multi-Resolution Evaluation

```bash
for res in 32 64 128; do
    generate -cn shapenet_v1 \
        model.weights=path/to/model.pt \
        vis.resolution=$res \
        test.dir=meshes_${res}
done
```

### Compare Models

```bash
python -m eval.scripts.compare_eval \
    --directory logs/ \
    --metric iou \
    --cls "mean (macro)"
```
