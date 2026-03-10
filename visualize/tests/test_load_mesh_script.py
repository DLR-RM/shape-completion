from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from visualize.scripts import load_mesh as script


class _FakeMesh:
    def __init__(self) -> None:
        self.vertices = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        self.faces = np.array([[0, 0, 0]], dtype=np.int32)
        self.transforms: list[np.ndarray] = []
        self.translations: list[np.ndarray] = []
        self.scales: list[float] = []

    @property
    def bounding_box(self) -> SimpleNamespace:
        return SimpleNamespace(bounds=np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=np.float32))

    def apply_translation(self, offset: np.ndarray) -> None:
        self.translations.append(np.asarray(offset))

    def apply_scale(self, scale: float) -> None:
        self.scales.append(scale)

    def apply_transform(self, pose: np.ndarray) -> None:
        self.transforms.append(np.asarray(pose))


class _FakeO3DTriangleMesh:
    def __init__(self, vertices: Any = None, triangles: Any = None) -> None:
        self.vertices = np.asarray(vertices if vertices is not None else np.zeros((0, 3), dtype=np.float32))
        self.triangles = np.asarray(triangles if triangles is not None else np.zeros((0, 3), dtype=np.int32))
        self.color: tuple[float, float, float] | None = None
        self.normals_computed = False

    def create_coordinate_frame(self, size: float = 0.1) -> _FakeO3DTriangleMesh:
        _ = size
        return _FakeO3DTriangleMesh()

    def paint_uniform_color(self, color: tuple[float, float, float]) -> _FakeO3DTriangleMesh:
        self.color = color
        return self

    def compute_vertex_normals(self) -> _FakeO3DTriangleMesh:
        self.normals_computed = True
        return self


class _FakeNPZ:
    def __init__(self, scale: float | None, pose: np.ndarray | None) -> None:
        self.values = {"scale": scale, "pose": pose}

    def get(self, key: str) -> Any:
        return self.values.get(key)


def test_main_smoke_with_mesh_dir_and_transform(monkeypatch: Any, tmp_path: Path) -> None:
    data_dir = tmp_path / "sample" / "cat_eval" / "outputs"
    mesh_dir = tmp_path / "ShapeNetCore.v1"
    draw_calls: list[list[Any]] = []
    loaded_meshes: list[_FakeMesh] = []

    args = Namespace(data_dir=str(data_dir), mesh_dir=str(mesh_dir))

    monkeypatch.setattr(script.argparse.ArgumentParser, "parse_args", lambda self: args)
    monkeypatch.setattr(script.os, "listdir", lambda path: ["123_item.npz", "ignore.txt"])
    monkeypatch.setattr(script.os.path, "isfile", lambda path: str(path).endswith("_uncertain.ply"))
    monkeypatch.setattr(
        script.trimesh,
        "load_mesh",
        lambda path, process=False: loaded_meshes.append(_FakeMesh()) or loaded_meshes[-1],
    )
    monkeypatch.setattr(
        script.o3d,
        "io",
        SimpleNamespace(
            read_triangle_mesh=lambda path: SimpleNamespace(
                vertices=np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]], dtype=np.float32),
                triangles=np.array([[0, 1, 2]], dtype=np.int32),
            )
        ),
    )
    monkeypatch.setattr(script.np, "load", lambda path, allow_pickle=True: _FakeNPZ(2.0, np.eye(4, dtype=np.float32)))
    monkeypatch.setattr(
        script,
        "o3d",
        SimpleNamespace(
            io=script.o3d.io,
            geometry=SimpleNamespace(TriangleMesh=_FakeO3DTriangleMesh),
            utility=SimpleNamespace(
                Vector3dVector=lambda values: np.asarray(values, dtype=np.float32),
                Vector3iVector=lambda values: np.asarray(values, dtype=np.int32),
            ),
            visualization=SimpleNamespace(draw_geometries=lambda geometries, **kwargs: draw_calls.append(list(geometries))),
        ),
    )

    script.main()

    assert len(loaded_meshes) == 2
    assert len(draw_calls) == 1
    assert len(draw_calls[0]) == 4


def test_main_smoke_without_mesh_dir(monkeypatch: Any, tmp_path: Path) -> None:
    data_dir = tmp_path / "outputs"
    draw_calls: list[list[Any]] = []

    args = Namespace(data_dir=str(data_dir), mesh_dir=None)

    monkeypatch.setattr(script.argparse.ArgumentParser, "parse_args", lambda self: args)
    monkeypatch.setattr(script.os, "listdir", lambda path: ["555_item.npz"])
    monkeypatch.setattr(script.os.path, "isfile", lambda path: False)
    monkeypatch.setattr(script.trimesh, "load_mesh", lambda path, process=False: _FakeMesh())
    monkeypatch.setattr(
        script,
        "o3d",
        SimpleNamespace(
            geometry=SimpleNamespace(TriangleMesh=_FakeO3DTriangleMesh),
            utility=SimpleNamespace(
                Vector3dVector=lambda values: np.asarray(values, dtype=np.float32),
                Vector3iVector=lambda values: np.asarray(values, dtype=np.int32),
            ),
            visualization=SimpleNamespace(draw_geometries=lambda geometries, **kwargs: draw_calls.append(list(geometries))),
        ),
    )

    script.main()

    assert len(draw_calls) == 1
    assert len(draw_calls[0]) == 2
