from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import numpy as np
import pytest

from inference.src import utils as script


class _FakePointCloud:
    instances: ClassVar[list[_FakePointCloud]] = []

    def __init__(self, points: Any = None) -> None:
        self.points = np.asarray(points if points is not None else np.zeros((0, 3), dtype=np.float32), dtype=np.float32)
        self.rotate_calls: list[np.ndarray] = []
        self.__class__.instances.append(self)

    def rotate(self, rotation: np.ndarray, center: tuple[float, float, float] = (0, 0, 0)) -> _FakePointCloud:
        _ = center
        self.rotate_calls.append(np.asarray(rotation, dtype=np.float32))
        return self

    def translate(self, offset: np.ndarray) -> _FakePointCloud:
        self.points = self.points + np.asarray(offset, dtype=np.float32)
        return self

    def scale(self, factor: float, center: tuple[float, float, float] = (0, 0, 0)) -> _FakePointCloud:
        self.points = (self.points - np.asarray(center, dtype=np.float32)) * factor + np.asarray(center, dtype=np.float32)
        return self

    def transform(self, matrix: np.ndarray) -> _FakePointCloud:
        rotation = matrix[:3, :3]
        translation = matrix[:3, 3]
        self.points = self.points @ rotation.T + translation
        return self

    def crop(self, box: Any) -> _FakePointCloud:
        mask = np.all((self.points >= np.asarray(box.min_bound)) & (self.points <= np.asarray(box.max_bound)), axis=1)
        return _FakePointCloud(self.points[mask])

    def get_min_bound(self) -> np.ndarray:
        return self.points.min(axis=0)

    def get_max_bound(self) -> np.ndarray:
        return self.points.max(axis=0)


class _FakeAxisAlignedBoundingBox:
    def __init__(self, min_bound: tuple[float, float, float], max_bound: tuple[float, float, float]) -> None:
        self.min_bound = np.asarray(min_bound, dtype=np.float32)
        self.max_bound = np.asarray(max_bound, dtype=np.float32)


class _FakeRotation:
    def __init__(self, matrix: np.ndarray) -> None:
        self._matrix = matrix

    def as_matrix(self) -> np.ndarray:
        return self._matrix


class _FakeVoxelizer:
    last_init: tuple[int, float, str] | None = None
    last_points: np.ndarray | None = None

    def __init__(self, resolution: int, padding: float, method: str) -> None:
        self.__class__.last_init = (resolution, padding, method)

    def __call__(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.__class__.last_points = np.asarray(points, dtype=np.float32)
        return self.__class__.last_points * 2, np.array([0, 1], dtype=np.int64)


class _FakePlanePointCloud(_FakePointCloud):
    def __init__(
        self,
        points: Any = None,
        *,
        plane_model: np.ndarray | None = None,
        plane_indices: np.ndarray | None = None,
        cluster_labels: np.ndarray | None = None,
    ) -> None:
        super().__init__(points)
        self.plane_model = np.asarray(
            plane_model if plane_model is not None else np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.plane_indices = np.asarray(plane_indices if plane_indices is not None else np.array([], dtype=np.int64))
        self.cluster_labels = None if cluster_labels is None else np.asarray(cluster_labels, dtype=np.int64)
        self.statistical_calls: list[tuple[int, float]] = []
        self.radius_calls: list[tuple[int, float]] = []

    def segment_plane(self, *, distance_threshold: float, ransac_n: int, num_iterations: int) -> tuple[np.ndarray, list[int]]:
        _ = distance_threshold, ransac_n, num_iterations
        return self.plane_model, self.plane_indices.tolist()

    def select_by_index(self, indices: Any, invert: bool = False) -> _FakePlanePointCloud:
        flat_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        mask = np.zeros(len(self.points), dtype=bool)
        mask[flat_indices] = True
        if invert:
            mask = ~mask
        labels = None if self.cluster_labels is None else self.cluster_labels[mask]
        return _FakePlanePointCloud(self.points[mask], cluster_labels=labels)

    def cluster_dbscan(self, eps: float, min_points: int) -> np.ndarray:
        _ = eps, min_points
        if self.cluster_labels is None:
            return np.zeros(len(self.points), dtype=np.int64)
        return self.cluster_labels

    def remove_statistical_outlier(self, nb_neighbors: int, std_ratio: float) -> tuple[_FakePlanePointCloud, list[int]]:
        self.statistical_calls.append((nb_neighbors, std_ratio))
        return self, [len(self.points) - 1]

    def remove_radius_outlier(self, nb_points: int, radius: float) -> tuple[_FakePlanePointCloud, list[int]]:
        self.radius_calls.append((nb_points, radius))
        return self, [0]


class _FakeDelaunay:
    def __init__(self, hull_points: np.ndarray) -> None:
        self.hull_points = np.asarray(hull_points, dtype=np.float32)

    def find_simplex(self, points: np.ndarray) -> np.ndarray:
        _ = points
        return np.array([1, -1, 1, -1, -1], dtype=np.int64)


def test_unproject_kinect_depth_filters_invalid_and_returns_intrinsic() -> None:
    depth_raw = np.array([[9, 7], [2047, 8]], dtype=np.float32)
    points, intrinsic = script.unproject_kinect_depth(
        depth_raw=depth_raw,
        depth_scale=1.0,
        depth_trunc=None,
        invalid_depth_value=2047,
        fx=1.0,
        fy=1.0,
        cx=0.0,
        cy=0.0,
        baseline=1.0,
        ir_depth_offset=0.0,
        disparity_offset=10.0,
        disparity_precision=1.0,
    )

    assert points.shape[1] == 3
    assert len(points) == 3
    np.testing.assert_allclose(intrinsic, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))


def test_unproject_kinect_depth_applies_depth_trunc() -> None:
    depth_raw = np.array([[9, 8]], dtype=np.float32)
    points, _ = script.unproject_kinect_depth(
        depth_raw=depth_raw,
        depth_scale=1.0,
        depth_trunc=0.75,
        invalid_depth_value=2047,
        fx=1.0,
        fy=1.0,
        cx=0.0,
        cy=0.0,
        baseline=1.0,
        ir_depth_offset=0.0,
        disparity_offset=10.0,
        disparity_precision=1.0,
    )

    # raw=9 -> z=1.0 filtered out; raw=8 -> z=0.5 kept
    assert len(points) == 1
    np.testing.assert_allclose(points[0, 2], 0.5, rtol=0.0, atol=1e-8)


def test_unproject_kinect_depth_sanitizes_nan_and_inf_disparities() -> None:
    depth_raw = np.array([[10, 11], [2047, 9]], dtype=np.float32)
    points, _ = script.unproject_kinect_depth(
        depth_raw=depth_raw,
        depth_scale=1.0,
        depth_trunc=None,
        invalid_depth_value=2047,
        fx=1.0,
        fy=1.0,
        cx=0.0,
        cy=0.0,
        baseline=1.0,
        ir_depth_offset=0.0,
        disparity_offset=10.0,
        disparity_precision=1.0,
    )

    assert points.shape == (1, 3)
    np.testing.assert_allclose(points[0], np.array([1.5, 1.5, 1.0]), rtol=0.0, atol=1e-8)


def test_get_rot_from_extrinsic_identity_returns_expected_axes_rotations() -> None:
    extrinsic = np.eye(4)
    rot_x, rot_y, rot_z = script.get_rot_from_extrinsic(extrinsic)

    expected_rot_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        dtype=np.float64,
    )
    np.testing.assert_allclose(rot_x, expected_rot_x, rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(rot_y, np.eye(3), rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(rot_z, np.eye(3), rtol=0.0, atol=1e-8)


def test_get_point_cloud_depth_branch_applies_rotation_and_crop(monkeypatch: Any, tmp_path: Path) -> None:
    intrinsic_path = tmp_path / "intrinsic.npy"
    extrinsic_path = tmp_path / "extrinsic.npy"
    depth_path = tmp_path / "depth.npy"
    intrinsic_matrix = np.diag([2.0, 2.0, 1.0]).astype(np.float32)
    extrinsic_matrix = np.eye(4, dtype=np.float32)
    depth = np.array([[1.0]], dtype=np.float32)
    unproject_points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32)
    unproject_intrinsic = np.diag([3.0, 3.0, 1.0]).astype(np.float32)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    rot_y = np.eye(3, dtype=np.float32)
    rot_z = np.eye(3, dtype=np.float32)

    def _fake_load(path: str) -> np.ndarray:
        if path == str(intrinsic_path):
            return intrinsic_matrix
        if path == str(extrinsic_path):
            return extrinsic_matrix
        if path == str(depth_path):
            return depth
        raise AssertionError(path)

    fake_o3d = SimpleNamespace(
        geometry=SimpleNamespace(PointCloud=_FakePointCloud),
        utility=SimpleNamespace(Vector3dVector=lambda values: np.asarray(values, dtype=np.float32)),
    )
    _FakePointCloud.instances.clear()
    monkeypatch.setattr(script.np, "load", _fake_load)
    monkeypatch.setattr(script, "unproject_kinect_depth", lambda *_args, **_kwargs: (unproject_points, unproject_intrinsic))
    monkeypatch.setattr(script, "eval_transformation_data", lambda value: value)
    monkeypatch.setattr(script, "get_rot_from_extrinsic", lambda value: (rot_x, rot_y, rot_z))
    monkeypatch.setattr(script, "o3d", fake_o3d)

    pcd, intrinsic, extrinsic = script.get_point_cloud(
        depth_path,
        intrinsic=str(intrinsic_path),
        extrinsic=str(extrinsic_path),
        pcd_crop=[-0.1, -0.1, -0.1, 0.1, 0.1, 0.1],
    )

    np.testing.assert_allclose(intrinsic, unproject_intrinsic, rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(extrinsic[:3, :3], (rot_x @ rot_y).T, rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(pcd.points, np.array([[1.0, 1.0, 1.0]], dtype=np.float32), rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(_FakePointCloud.instances[0].rotate_calls[0], rot_x @ rot_y @ rot_z, rtol=0.0, atol=1e-8)


def test_get_point_cloud_image_branch_and_unsupported_suffix(monkeypatch: Any, tmp_path: Path) -> None:
    image_path = tmp_path / "depth.png"
    calls: dict[str, Any] = {}
    fake_pcd = _FakePointCloud([[0.0, 0.0, 0.0]])
    fake_o3d = SimpleNamespace(
        io=SimpleNamespace(read_image=lambda path: SimpleNamespace(numpy=lambda: np.array([[4.0]], dtype=np.float32))),
    )

    def _fake_eval_data(*, data: np.ndarray, camera_intrinsic: np.ndarray, depth_scale: float, depth_trunc: float) -> _FakePointCloud:
        calls["data"] = np.asarray(data)
        calls["intrinsic"] = np.asarray(camera_intrinsic)
        calls["depth_scale"] = depth_scale
        calls["depth_trunc"] = depth_trunc
        return fake_pcd

    monkeypatch.setattr(script, "o3d", fake_o3d)
    monkeypatch.setattr(script, "eval_data", _fake_eval_data)

    intrinsic = np.diag([4.0, 5.0, 1.0]).astype(np.float32)
    pcd, intrinsic_arr, extrinsic_arr = script.get_point_cloud(image_path, intrinsic=intrinsic, depth_scale=42.0, depth_trunc=0.9)

    assert pcd is fake_pcd
    np.testing.assert_allclose(calls["data"], np.array([[4.0]], dtype=np.float32), rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(calls["intrinsic"], intrinsic, rtol=0.0, atol=1e-8)
    assert calls["depth_scale"] == 42.0
    assert calls["depth_trunc"] == 0.9
    np.testing.assert_allclose(intrinsic_arr, intrinsic, rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(extrinsic_arr, np.eye(4), rtol=0.0, atol=1e-8)

    with pytest.raises(ValueError, match="Unsupported file format"):
        script.get_point_cloud(tmp_path / "bad.txt")


def test_get_input_data_from_point_cloud_centers_and_scales_extent(monkeypatch: Any) -> None:
    pcd = _FakePointCloud([[0.0, 0.0, 0.0], [2.0, 4.0, 2.0], [4.0, 2.0, 2.0]])
    monkeypatch.setattr(script, "o3d", SimpleNamespace(geometry=SimpleNamespace(AxisAlignedBoundingBox=_FakeAxisAlignedBoundingBox)))

    points, loc, scale_value = script.get_input_data_from_point_cloud(
        pcd,
        center=True,
        offset_y=2.0,
        scale=True,
        crop=None,
    )

    expected_points = np.array([[-2.0, -1.0, -1.0], [0.0, 3.0, 1.0], [2.0, 1.0, 1.0]], dtype=np.float32) / 6.0
    np.testing.assert_allclose(points, expected_points, rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(loc, np.array([2.0, 1.0, 1.0], dtype=np.float32), rtol=0.0, atol=1e-8)
    assert scale_value == 6.0


def test_get_input_data_from_point_cloud_subsamples_noises_and_voxelizes(monkeypatch: Any) -> None:
    pcd = _FakePointCloud([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [2.0, 0.0, 0.0]])
    transform = np.eye(4, dtype=np.float32)
    fake_o3d = SimpleNamespace(
        geometry=SimpleNamespace(AxisAlignedBoundingBox=_FakeAxisAlignedBoundingBox),
    )
    _FakeVoxelizer.last_init = None
    _FakeVoxelizer.last_points = None

    monkeypatch.setattr(script, "o3d", fake_o3d)
    monkeypatch.setattr(script, "eval_transformation_data", lambda value: value)
    monkeypatch.setattr(script, "subsample_indices", lambda points, n: np.array([1, 0], dtype=np.int64))
    monkeypatch.setattr(script.np.random, "randn", lambda *shape: np.ones(shape, dtype=np.float32))
    monkeypatch.setattr(script, "Voxelizer", _FakeVoxelizer)
    monkeypatch.setattr(script.R, "from_euler", lambda *_args, **_kwargs: _FakeRotation(np.eye(3, dtype=np.float32)))

    points, loc, scale_value = script.get_input_data_from_point_cloud(
        pcd,
        transform=transform,
        num_input_points=2,
        noise_std=0.1,
        rotate_z=0.5,
        crop=0.6,
        voxelize=8,
        padding=0.2,
    )

    assert _FakeVoxelizer.last_points is not None
    np.testing.assert_allclose(_FakeVoxelizer.last_points, np.array([[0.6, 0.1, 0.1], [0.1, 0.1, 0.1]], dtype=np.float32))
    assert _FakeVoxelizer.last_init == (8, 0.2, "kdtree")
    np.testing.assert_allclose(points, np.array([[1.2, 0.2, 0.2], [0.2, 0.2, 0.2]], dtype=np.float32), rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(loc, np.zeros(3, dtype=np.float32), rtol=0.0, atol=1e-8)
    assert scale_value == 1.0


def test_remove_plane_filters_outliers_and_merges_crop_hull(monkeypatch: Any) -> None:
    pcd = _FakePlanePointCloud(
        [
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.5, 0.0],
            [0.0, 0.9, 0.0],
            [0.0, 2.0, 0.0],
        ],
        plane_indices=np.array([0, 1], dtype=np.int64),
    )
    fake_o3d = SimpleNamespace(
        geometry=SimpleNamespace(
            PointCloud=lambda values: _FakePlanePointCloud(values, cluster_labels=np.array([0, 1, -1], dtype=np.int64))
        ),
        utility=SimpleNamespace(Vector3dVector=lambda values: np.asarray(values, dtype=np.float32)),
    )

    monkeypatch.setattr(script, "o3d", fake_o3d)
    monkeypatch.setattr(script, "Delaunay", _FakeDelaunay)
    monkeypatch.setattr(script, "check_mesh_contains", None)

    pcds, plane_model = script.remove_plane(
        pcd,
        distance_threshold=0.1,
        outlier_neighbors=4,
        outlier_radius=0.2,
        outlier_std=1.5,
        cluster=False,
        crop=True,
    )

    np.testing.assert_allclose(plane_model, np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float32), rtol=0.0, atol=1e-8)
    assert len(pcds) == 1
    np.testing.assert_allclose(
        pcds[0].points,
        np.array([[0.0, 1.0, 0.0], [0.0, 1.5, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32),
        rtol=0.0,
        atol=1e-8,
    )


def test_remove_plane_cluster_mode_splits_detected_components(monkeypatch: Any) -> None:
    pcd = _FakePlanePointCloud(
        [
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.5, 0.0],
            [0.0, 1.7, 0.0],
            [0.0, 2.0, 0.0],
        ],
        plane_indices=np.array([0, 1], dtype=np.int64),
        cluster_labels=np.array([-1, -1, 0, 1, -1], dtype=np.int64),
    )
    fake_o3d = SimpleNamespace(
        geometry=SimpleNamespace(
            PointCloud=lambda values: _FakePlanePointCloud(values, cluster_labels=np.array([0, 1, -1], dtype=np.int64))
        ),
        utility=SimpleNamespace(Vector3dVector=lambda values: np.asarray(values, dtype=np.float32)),
    )

    monkeypatch.setattr(script, "o3d", fake_o3d)

    pcds, _ = script.remove_plane(
        pcd,
        distance_threshold=0.1,
        outlier_neighbors=0,
        cluster=True,
        cluster_min_points=1,
        num_cluster=cast(Any, None),
        crop=False,
    )

    assert len(pcds) == 2
    np.testing.assert_allclose(pcds[0].points, np.array([[0.0, 1.5, 0.0]], dtype=np.float32), rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(pcds[1].points, np.array([[0.0, 1.7, 0.0]], dtype=np.float32), rtol=0.0, atol=1e-8)
