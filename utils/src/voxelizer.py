from functools import cached_property
from typing import Literal

import numpy as np
import torch
from einops import rearrange
from pykdtree.kdtree import KDTree

from .utils import make_3d_grid


class Voxelizer:
    def __init__(
        self,
        resolution: int = 128,
        padding: float = 0.1,
        method: Literal["simple", "kdtree", "open3d"] = "simple",
        round: bool = False,
        scale_factor: float = 1.0,
        bounds: tuple[float, float] | tuple[tuple[float, float, float], tuple[float, float, float]] = (-0.5, 0.5),
    ):
        assert method.lower() in ["simple", "kdtree", "open3d"], f"Unknown voxelization method: {method}."

        self.res = resolution
        self.method = method.lower()
        self.padding = float(padding)
        self.round = round
        self.scale_factor = float(scale_factor)

        # Bounds handling consistent with visualize/src/generator.py
        if isinstance(bounds[0], (int, float)) and isinstance(bounds[1], (int, float)):
            mins = np.array([float(bounds[0])] * 3, dtype=np.float32)
            maxs = np.array([float(bounds[1])] * 3, dtype=np.float32)
        else:
            mins = np.array(bounds[0], dtype=np.float32)
            maxs = np.array(bounds[1], dtype=np.float32)
        self._mins = mins
        self._maxs = maxs

    @cached_property
    def center(self) -> np.ndarray:
        return ((self._mins + self._maxs) / 2.0).astype(np.float32)

    @cached_property
    def axis_extent(self) -> np.ndarray:
        return (self._maxs - self._mins) * float(self.scale_factor)

    @cached_property
    def axis_box_size(self) -> np.ndarray:
        return self.axis_extent + float(self.padding)

    @cached_property
    def box_size(self) -> float:
        return float(np.max(self.axis_box_size))

    @cached_property
    def voxel_size(self) -> float:
        return self.box_size / float(self.res)

    @cached_property
    def grid_shape(self) -> tuple[int, int, int]:
        shape = np.maximum(2, np.round(self.axis_box_size / self.voxel_size).astype(int))
        return int(shape[0]), int(shape[1]), int(shape[2])

    @cached_property
    def _grid_origin_center(self) -> np.ndarray:
        # First voxel center along each axis
        nx, ny, nz = self.grid_shape
        vs = float(self.voxel_size)
        total = np.array([nx, ny, nz], dtype=np.float32) * vs
        offset = (total - vs) / 2.0
        return (self.center - offset).astype(np.float32)

    @cached_property
    def _grid_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        # Boundary (min/max) of the voxel grid (half-voxel outside centers)
        nx, ny, nz = self.grid_shape
        vs = float(self.voxel_size)
        origin_c = self._grid_origin_center
        mins_b = origin_c - 0.5 * vs
        maxs_b = origin_c + np.array([nx - 1, ny - 1, nz - 1], dtype=np.float32) * vs + 0.5 * vs
        return mins_b, maxs_b

    @cached_property
    def grid_points(self) -> np.ndarray:
        x0, y0, z0 = self._grid_origin_center.tolist()
        nx, ny, nz = self.grid_shape
        vs = float(self.voxel_size)
        return (
            make_3d_grid(
                (x0, y0, z0),
                (x0 + vs * (nx - 1), y0 + vs * (ny - 1), z0 + vs * (nz - 1)),
                (nx, ny, nz),
                dtype=torch.float64,
            )
            .numpy()
            .astype(np.float32)
        )

    @cached_property
    def kdtree(self) -> KDTree:
        return KDTree(self.grid_points)

    def __call__(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nx, ny, nz = self.grid_shape
        vs = float(self.voxel_size)
        origin_c = self._grid_origin_center
        mins_b, maxs_b = self._grid_bounds

        if self.method == "simple":
            idx_f = (points - origin_c) / vs
            idx = np.rint(idx_f) if self.round else np.floor(idx_f)
            idx = idx.astype(np.int64)
            # Valid inside-bounds mask for occupancy
            valid = np.all(points >= mins_b, axis=1) & np.all(points <= maxs_b, axis=1)
            # Clip indices for return to keep mapping defined for all points
            idx[:, 0] = np.clip(idx[:, 0], 0, nx - 1)
            idx[:, 1] = np.clip(idx[:, 1], 0, ny - 1)
            idx[:, 2] = np.clip(idx[:, 2], 0, nz - 1)
            occupancy = np.zeros((nx, ny, nz), dtype=bool)
            if np.any(valid):
                v = idx[valid]
                occupancy[(v[:, 0], v[:, 1], v[:, 2])] = True
            voxel_indices = np.ravel_multi_index((idx[:, 0], idx[:, 1], idx[:, 2]), (nx, ny, nz))
        elif self.method == "kdtree":
            _, voxel_indices = self.kdtree.query(
                points.astype(self.grid_points.dtype),
                k=1,
                eps=0,
                distance_upper_bound=self.voxel_size / 2 if self.round else None,
            )
            voxel_indices = voxel_indices.clip(0, len(self.grid_points) - 1)
            occupancy = np.zeros(len(self.grid_points), dtype=bool)
            valid = np.all(points >= mins_b, axis=1) & np.all(points <= maxs_b, axis=1)
            if np.any(valid):
                occupancy[voxel_indices[valid]] = True
            occupancy = rearrange(occupancy, "(x y z) -> x y z", x=nx, y=ny, z=nz)
            # Per-point indices consistent with simple/open3d quantization
            idx_f = (points - origin_c) / vs
            idx = np.rint(idx_f) if self.round else np.floor(idx_f)
            idx = idx.astype(np.int64)
            idx[:, 0] = np.clip(idx[:, 0], 0, nx - 1)
            idx[:, 1] = np.clip(idx[:, 1], 0, ny - 1)
            idx[:, 2] = np.clip(idx[:, 2], 0, nz - 1)
            voxel_indices = np.ravel_multi_index((idx[:, 0], idx[:, 1], idx[:, 2]), (nx, ny, nz))
        elif self.method == "open3d":
            import open3d as o3d

            assert not self.round, "Rounding not supported for Open3D voxelization."
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            voxel_grid = o3d.geometry.VoxelGrid().create_from_point_cloud_within_bounds(
                input=pcd,
                voxel_size=self.voxel_size,
                min_bound=mins_b,
                max_bound=maxs_b,
            )
            occupancy = np.zeros((nx, ny, nz), dtype=bool)
            voxels = voxel_grid.get_voxels()
            if len(voxels) > 0:
                occ_idx = np.asarray([v.grid_index for v in voxels], dtype=np.int64)
                occ_idx[:, 0] = np.clip(occ_idx[:, 0], 0, nx - 1)
                occ_idx[:, 1] = np.clip(occ_idx[:, 1], 0, ny - 1)
                occ_idx[:, 2] = np.clip(occ_idx[:, 2], 0, nz - 1)
                occupancy[(occ_idx[:, 0], occ_idx[:, 1], occ_idx[:, 2])] = True
            # Per-point mapping via quantization for consistency
            idx_f = (points - origin_c) / vs
            idx = np.rint(idx_f) if self.round else np.floor(idx_f)
            idx = idx.astype(np.int64)
            idx[:, 0] = np.clip(idx[:, 0], 0, nx - 1)
            idx[:, 1] = np.clip(idx[:, 1], 0, ny - 1)
            idx[:, 2] = np.clip(idx[:, 2], 0, nz - 1)
            voxel_indices = np.ravel_multi_index((idx[:, 0], idx[:, 1], idx[:, 2]), (nx, ny, nz))
        else:
            raise ValueError(f"Unknown voxelization method: {self.method}")

        return occupancy, voxel_indices
