# Libraries

Several libraries that assist in geometry and 3D manipulation, including operations on point
clouds, meshes, and more. The available libraries and functionalities are listed below.

## List of available libraries

### libchamfer

A library for computing the Chamfer distance between two sets of points.

- **Build Directory**: Contains compiled objects and binaries.
- **Chamfer CUDA Implementation**: Provides CUDA support for the Chamfer operation.

### libemd

The Earth Mover's Distance (EMD) library in CUDA.

- **Build Directory**: Contains compiled objects and binaries.
- **EMD CUDA Implementation**: Provides CUDA support for the EMD calculation.

### libfusion

GPU accelerated TSDF fusion of depth images to generated watertight meshes from triangle soups

### libintersect

Functions and classes to calculate and manipulate intersections in mesh structures.

- **Build Directory**: Compiled objects and binaries.
- **Inside Mesh and Triangle Hash**: Provides support for mesh intersection and inside mesh calculations.

### libkinect

Kinect depth map simulator.

### libmise

Multi-resolution ISO surface extraction (MISE) library.

### libpointnet

PointNet2 operations for PyTorch, including CUDA implementations.

- **Build Directory**: Contains compiled objects and binaries.
- **PointNet2 Operations**: A collection of operations for PointNet2.

### libsimplify

Fast quadric edge collapse decimation for triangle meshes.

## Installation

The preferred method of installation is to run the following command from the root directory of this repository:

```bash
python lib_manager.py install
```

Selective install through comma-separated list of library names. Commands to `uninstall`, `update`
, (force-)`reinstall` and `clean` (removes build files) are also available.

Alternatively, you can install each library individually by navigating to its directory and running:

```bash
pip install .
```

or

```bash
pip install <library_name>
```

from the root directory of this repository.
