# PyFusion

PyFusion is a Python framework for volumetric depth fusion.
It contains simple occupancy and TSDF fusion methods that can be executed on a CPU as well as on a GPU.

## Installation

```bash
pip install .
```
## Usage

```python
from pyfusion_gpu import PyViews, tsdf_gpu

# create a views object
# depthmaps: a NxHxW numpy float tensor of N depthmaps, invalid depth values are marked by negative numbers
# Ks: the camera intric matrices, Nx3x3 float tensor
# Rs: the camera rotation matrices, Nx3x3 float tensor
# Ts: the camera translation vectors, Nx3 float tensor
views = PyViews(depthmaps, Ks,Rs,Ts)

# afterwards you can fuse the depth maps for example by
# depth,height,width: number of voxels in each dimension
# truncation: TSDF truncation value
tsdf = tsdf_gpu(views, depth,height,width, vx_size, truncation, False)

# the same code can also be run on the CPU
from pyfusion_cpu import PyViews, tsdf_cpu
tsdf = tsdf_cpu(views, depth,height,width, vx_size, truncation, False, n_threads=8)
```
