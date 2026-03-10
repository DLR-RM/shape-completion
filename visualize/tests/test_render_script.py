from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from visualize.scripts import render as script


class _FakeMesh:
    def __init__(self, *, color: bool = False) -> None:
        self.vertices = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]], dtype=np.float32)
        self.faces = np.array([[0, 1, 2]], dtype=np.int32)
        self.colors = (
            np.array([[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]], dtype=np.uint8) if color else None
        )
        self.bounding_box = SimpleNamespace(extents=np.array([1.0, 1.0, 1.0], dtype=np.float32))
        self.centroid = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.transformations: list[np.ndarray] = []
        self.translations: list[np.ndarray] = []

    def apply_transformations(self, trafo: np.ndarray) -> None:
        self.transformations.append(np.asarray(trafo))

    def apply_transform(self, trafo: np.ndarray) -> None:
        self.transformations.append(np.asarray(trafo))

    def apply_translation(self, offset: list[float] | np.ndarray) -> None:
        self.translations.append(np.asarray(offset, dtype=np.float32))


class _FakePointCloud:
    def __init__(self) -> None:
        self.points = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32)
        self._distance_calls = 0

    def compute_nearest_neighbor_distance(self) -> np.ndarray:
        self._distance_calls += 1
        return np.array([0.0], dtype=np.float32) if self._distance_calls == 1 else np.array([0.05], dtype=np.float32)

    def farthest_point_down_sample(self, num_points: int) -> _FakePointCloud:
        _ = num_points
        downsampled = _FakePointCloud()
        downsampled.points = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32)
        downsampled._distance_calls = 1
        return downsampled


class _FakeRenderer:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.calls: list[dict[str, Any]] = []
        self.default_mesh_color = np.array([0.4, 0.4, 0.4], dtype=np.float32)
        self.default_pcd_color = np.array([0.1, 0.2, 0.8], dtype=np.float32)

    def __call__(self, **kwargs: Any) -> dict[str, list[np.ndarray]]:
        self.calls.append(kwargs)
        image = np.zeros((8, 8, 4), dtype=np.uint8)
        image[..., 3] = 255
        return {"color": [image]}


def test_main_smoke_mesh_dir_with_individual_outputs(monkeypatch: Any, tmp_path: Path) -> None:
    input_dir = tmp_path / "render_dir"
    obj_path = input_dir / "chairs" / "sample_mesh.ply"
    obj_path.parent.mkdir(parents=True)
    obj_path.write_text("mesh")
    (obj_path.parent / "sample_inputs.ply").write_text("inputs")
    (obj_path.parent / "sample_gt.ply").write_text("gt")
    renderer_holder: dict[str, _FakeRenderer] = {}

    def fake_renderer(*args: Any, **kwargs: Any) -> _FakeRenderer:
        renderer_holder["instance"] = _FakeRenderer(*args, **kwargs)
        return renderer_holder["instance"]

    monkeypatch.setattr(
        script.ArgumentParser,
        "parse_args",
        lambda self: Namespace(dir=input_dir, obj_type="mesh", individual=True, look_at="centroid", show=False, verbose=False),
    )
    monkeypatch.setattr(script, "suppress_known_optional_dependency_warnings", lambda: None)
    monkeypatch.setattr(script, "log_optional_dependency_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(script, "seed_everything", lambda seed: None)
    monkeypatch.setattr(script, "tqdm", lambda iterable, **kwargs: iterable)
    monkeypatch.setattr(script, "look_at", lambda *args, **kwargs: np.eye(4, dtype=np.float32))
    monkeypatch.setattr(script, "inv_trafo", lambda matrix: matrix)
    monkeypatch.setattr(script, "stack_images", lambda images: images[0] if images else np.zeros((8, 8, 4), dtype=np.uint8))
    monkeypatch.setattr(script, "Renderer", fake_renderer)
    monkeypatch.setattr(script, "Trimesh", _FakeMesh)
    monkeypatch.setattr(script.trimesh, "PointCloud", lambda points: _FakeMesh(color=True))
    monkeypatch.setattr(script.trimesh, "load", lambda path, **kwargs: _FakeMesh(color=not str(path).endswith("backdrop.ply")))
    monkeypatch.setattr(
        script.trimesh.transformations,
        "rotation_matrix",
        lambda angle, axis: np.eye(4, dtype=np.float32),
    )
    monkeypatch.setattr(script.o3d.io, "read_point_cloud", lambda path: _FakePointCloud())

    script.main()

    assert renderer_holder["instance"].kwargs["transparent_background"] is False
    assert (obj_path.parent / "sample_front.png").is_file()
    assert (obj_path.parent / "sample_back.png").is_file()
    assert (obj_path.parent / "sample_mesh_front.png").is_file()
    assert (obj_path.parent / "sample_input_front.png").is_file()
    assert (obj_path.parent / "sample_gt_front.png").is_file()
    assert (input_dir / "renders_front.png").is_file()
    assert (input_dir / "renders_back.png").is_file()


def test_main_smoke_single_pcd_file(monkeypatch: Any, tmp_path: Path) -> None:
    obj_path = tmp_path / "single_pcd.ply"
    obj_path.write_text("pcd")
    renderer_holder: dict[str, _FakeRenderer] = {}

    class _FakeColoredCloud:
        def __init__(self) -> None:
            self.vertices = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float32)
            self.faces = None
            self.colors = np.array([[255, 0, 0, 255], [0, 255, 0, 255]], dtype=np.uint8)
            self.bounding_box = SimpleNamespace(extents=np.array([1.0, 1.0, 1.0], dtype=np.float32))
            self.centroid = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.transformations: list[np.ndarray] = []
            self.translations: list[np.ndarray] = []

        def apply_transformations(self, trafo: np.ndarray) -> None:
            self.transformations.append(np.asarray(trafo))

        def apply_translation(self, offset: list[float] | np.ndarray) -> None:
            self.translations.append(np.asarray(offset, dtype=np.float32))

    def fake_renderer(*args: Any, **kwargs: Any) -> _FakeRenderer:
        renderer_holder["instance"] = _FakeRenderer(*args, **kwargs)
        return renderer_holder["instance"]

    monkeypatch.setattr(
        script.ArgumentParser,
        "parse_args",
        lambda self: Namespace(dir=obj_path, obj_type="pcd", individual=False, look_at="zero", show=False, verbose=True),
    )
    monkeypatch.setattr(script, "suppress_known_optional_dependency_warnings", lambda: None)
    monkeypatch.setattr(script, "log_optional_dependency_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(script, "seed_everything", lambda seed: None)
    monkeypatch.setattr(script, "tqdm", lambda iterable, **kwargs: iterable)
    monkeypatch.setattr(script, "look_at", lambda *args, **kwargs: np.eye(4, dtype=np.float32))
    monkeypatch.setattr(script, "inv_trafo", lambda matrix: matrix)
    monkeypatch.setattr(script, "stack_images", lambda images: images[0] if images else np.zeros((8, 8, 4), dtype=np.uint8))
    monkeypatch.setattr(script, "Renderer", fake_renderer)
    monkeypatch.setattr(script, "Trimesh", _FakeMesh)
    monkeypatch.setattr(script.trimesh, "PointCloud", lambda points: _FakeMesh(color=True))
    monkeypatch.setattr(script.trimesh, "load", lambda path, **kwargs: _FakeMesh() if str(path).endswith("backdrop.ply") else _FakeColoredCloud())
    monkeypatch.setattr(
        script.trimesh.transformations,
        "rotation_matrix",
        lambda angle, axis: np.eye(4, dtype=np.float32),
    )
    monkeypatch.setattr(script.o3d.io, "read_point_cloud", lambda path: _FakePointCloud())

    script.main()

    assert renderer_holder["instance"].kwargs["transparent_background"] is True
    assert (tmp_path / "single_front.png").is_file()
    assert (tmp_path / "single_back.png").is_file()
    assert (tmp_path / "renders_front.png").is_file()
    assert (tmp_path / "renders_back.png").is_file()


def test_main_rejects_invalid_input_path(monkeypatch: Any, tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    monkeypatch.setattr(
        script.ArgumentParser,
        "parse_args",
        lambda self: Namespace(dir=missing, obj_type="mesh", individual=False, look_at="centroid", show=False, verbose=False),
    )
    monkeypatch.setattr(script, "suppress_known_optional_dependency_warnings", lambda: None)
    monkeypatch.setattr(script, "log_optional_dependency_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(script, "seed_everything", lambda seed: None)

    with pytest.raises(ValueError, match="Invalid input path"):
        script.main()
