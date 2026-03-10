from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from process.scripts import process_mesh


def test_sample_pointcloud_shape_and_dtype() -> None:
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    args = SimpleNamespace(num_points=128, precision=32)

    points, normals = process_mesh.sample_pointcloud(mesh, args)

    assert points.shape == (128, 3)
    assert normals.shape == (128, 3)
    assert points.dtype == np.float32
    assert normals.dtype == np.float32
    assert np.isfinite(points).all()
    assert np.isfinite(normals).all()


def test_sample_points_shape_dtype_and_occupancy(monkeypatch: pytest.MonkeyPatch) -> None:
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    args = SimpleNamespace(num_points=100, padding=0.1, precision=32)
    expected_num_points = args.num_points * 8

    called_with_points: dict[str, int] = {}

    def fake_check_mesh_contains(_mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
        called_with_points["count"] = len(points)
        return np.zeros(len(points), dtype=np.bool_)

    monkeypatch.setattr(process_mesh, "check_mesh_contains", fake_check_mesh_contains)

    points, occupancy = process_mesh.sample_points(mesh, args)

    assert points.shape == (expected_num_points, 3)
    assert occupancy.shape == (expected_num_points,)
    assert points.dtype == np.float32
    assert occupancy.dtype == np.bool_
    assert called_with_points["count"] == expected_num_points
