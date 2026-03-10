# pyright: reportMissingImports=false

import logging

import numpy as np
import torch
from torch import Tensor

try:
    logging.debug("Importing from libchamfer")
    from .libchamfer import ChamferDistanceL2
except (ImportError, ModuleNotFoundError):
    logging.warning("Could not import from libchamfer. 'ChamferDistanceL2' will not be available.")
    ChamferDistanceL2 = None

try:
    logging.debug("Importing from libemd")
    from .libemd import EarthMoversDistance
except (ImportError, ModuleNotFoundError):
    logging.warning("Could not import from libemd. 'EarthMoversDistance' will not be available.")
    EarthMoversDistance = None

try:
    logging.debug("Importing from libintersect")
    from .libintersect import check_mesh_contains
except (ImportError, ModuleNotFoundError):
    logging.warning("Could not import from libintersect. 'check_mesh_contains' will not be available.")
    check_mesh_contains = None

try:
    logging.debug("Importing from libpointnet")
    from .libpointnet import furthest_point_sample as fps_pointnet
    from .libpointnet import gather, group
except (ImportError, ModuleNotFoundError):
    logging.warning(
        "Could not import from libpointnet. 'gather_operation' and 'grouping_operation' will not be available."
    )
    gather = None
    group = None
    fps_pointnet = None

try:
    logging.debug("Importing from libconv")
    from .libpvconv import (
        PVConv,
        SE3d,
        SharedMLP,
        Voxelization,
        get_voxel_coords,
        trilinear_devoxelize,
    )
    from .libpvconv import (
        furthest_point_sample as fps_pvconv,
    )
except (ImportError, ModuleNotFoundError) as e:
    logging.warning(f"Could not import from libpvconv. Some functionality will not be available: {e}")
    fps_pvconv = None
    trilinear_devoxelize = None
    Voxelization = None
    get_voxel_coords = None
    PVConv = None
    SE3d = None
    SharedMLP = None

try:
    logging.debug("Importing from libsimplify")
    from .libsimplify import simplify_mesh
except (ImportError, ModuleNotFoundError):
    logging.warning("Could not import from libsimplify. 'simplify_mesh' will not be available.")
    simplify_mesh = None

try:
    logging.debug("Importing from libmise")
    from .libmise import MISE
except (ImportError, ModuleNotFoundError):
    logging.warning("Could not import from libmise. 'MISE' will not be available.")
    MISE = None

try:
    logging.debug("Importing from libfusion")
    from .libfusion import PyViews, tsdf_fusion
except (ImportError, ModuleNotFoundError):
    logging.warning("Could not import from libfusion. 'PyViews' and 'tsdf_fusion' will not be available.")
    PyViews = None
    tsdf_fusion = None

try:
    logging.debug("Importing from vega")
    from .vega import compute_distance_field, compute_marching_cubes
except (ImportError, ModuleNotFoundError):
    logging.warning(
        "Could not import from vega. 'compute_distance_field' and 'compute_marching_cubes' will not be available."
    )
    compute_distance_field = None
    compute_marching_cubes = None


@torch.no_grad()
def furthest_point_sample(
    points: Tensor | np.ndarray, num_samples: int, backend: str = "pvconv", **kwargs
) -> Tensor | np.ndarray:
    """
    Samples points from a given point cloud by selecting the furthest points based on the
    specified backend algorithm. This function can handle both PyTorch tensors and NumPy arrays.

    Parameters:
    - points (Union[Tensor, np.ndarray]): The input point cloud data as a 3D tensor (B, N, C) for batches
      or a 2D array (N, 3) for a single point cloud, where B is the batch size, N is the number of points,
      and C[:3] represents the XYZ coordinates, i.e. the first three channels must be the point positions.
    - num_samples (int): The number of points to sample from the point cloud.
    - backend (str, optional): The backend algorithm used for sampling. Supported backends are "pvconv",
      "pointnet", and "torch_cluster". Default is "pvconv".

    Returns:
    - Union[Tensor, np.ndarray]: The down-sampled point cloud data, which will be of the same type (Tensor or array)
      as the input, containing the furthest num_samples points.

    Raises:
    - ImportError: If the necessary module for the selected backend is not available.
    - ValueError: If an unknown backend is specified.
    - AssertionError: If the input points do not meet the expected dimensions or format.

    Examples:
    >>> points = torch.randn(10, 100, 3)  # Example tensor with 10 samples, each containing 100 points.
    >>> sampled_points = furthest_point_sample(points, 10)
    >>> print(sampled_points.shape)
    torch.Size([10, 10, 3])

    >>> points_np = np.random.rand(100, 3)  # Example NumPy array with 100 points.
    >>> sampled_points_np = furthest_point_sample(points_np, 20, backend='open3d')
    >>> print(sampled_points_np.shape)
    (20, 3)
    """
    if torch.is_tensor(points):
        assert points.ndim == 3 and points.size(2) >= 3, f"Expected points to have shape (B, N, C), got {points.shape}."

        if backend == "pvconv":
            assert fps_pvconv is not None, "Could not import 'furthest_point_sample' from libpvconv."
            return fps_pvconv(points, num_samples, **kwargs).contiguous()
        if backend == "pointnet":
            assert fps_pointnet is not None, "Could not import 'furthest_point_sample' from libpointnet."
            return fps_pointnet(points, num_samples).contiguous()
        if backend == "torch_cluster":
            try:
                from torch_cluster import fps

                points_flat = points.view(-1, points.size(-1))
                batch = torch.arange(points.size(0)).to(points.device)
                batch = torch.repeat_interleave(batch, points.size(1))
                idx = fps(points_flat[..., :3], batch, ratio=num_samples / points.size(1), random_start=False)
                return points_flat[idx].view(points.size(0), num_samples, points.size(-1))
            except (ImportError, ModuleNotFoundError) as err:
                raise ImportError("Could not import 'fps' from torch_cluster.") from err

        raise ValueError(f"Unknown backend: {backend}.")
    elif isinstance(points, np.ndarray):
        assert points.ndim == 2 and points.shape[1] == 3, f"Expected points to have shape (N, 3), got {points.shape}."

        import open3d as o3d

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        return np.asarray(pcd.farthest_point_down_sample(num_samples).points)
