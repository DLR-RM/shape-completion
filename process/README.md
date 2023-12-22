# Data Processing Scripts

Various scripts for (pre-)processing of data. Functionality includes [rendering](#rendering), identification of [physically plausible object poses](#find-physically-plausible-poses)
generation of [watertight meshes](#generate-watertight-meshes) from arbitrary triangle soups, identification of [uncertain regions](#find-uncertain-regions) as introduced in [Shape Completion with Prediction of Uncertain Regions](https://arxiv.org/abs/2308.00377) and more.

## Overview

### Rendering

* `render_kinect.py`: Render depth images from a given camera pose and intrinsics. Also simulates Kinect depth noise.
* `render_kinect_parallel.py`: Runs `render_kinect.py` in parallel for multiple camera poses and intrinsics.
* `render_data.py` Uses [`BlenderProc`](https://github.com/DLR-RM/BlenderProc) for the rendering.

### Find physically plausible poses

The script `generate_physics_poses.py` generates physically plausible poses for sets of objects. It uses the [Bullet Physics Engine](https://pybullet.org/wordpress/) to simulate the object falling onto a plane.

### Generate watertight meshes

The script `make_watertight.py` generates watertight meshes from triangle soups. It is based on [mesh-fusion](https://github.com/davidstutz/mesh-fusion) with various improvements for quality, robustness and speed.

**Usage**
```bash
make_watertight path/to/ShapeNetCore.v1
```

Use `--out_dir /path/to/output/directory` to specify the output directory.

### Find uncertain regions

The script `find_uncertain_regions.py` identifies regions with ambiguous occupancy given the current point of view.
For more details refer to [Shape Completion with Prediction of Uncertain Regions](https://arxiv.org/abs/2308.00377).

**Usage**
```bash
find_uncertain_regions -cn shapenet_uncertain
```

Add `vis.split=val` or `vis.split=test` to find uncertain regions for the validation or test set, respectively.
Use `log.verbose=True` to get detailed information about the process and `vis.show=True` to visualize the results.
