from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import torch

from inference.scripts import inference_mesh_tutorial as script


class _FakeDrawable:
    def __init__(self) -> None:
        self.color: np.ndarray | None = None
        self.rotate_calls: list[tuple[np.ndarray, tuple[float, float, float]]] = []
        self.scale_calls: list[tuple[float, tuple[float, float, float]]] = []
        self.translate_calls: list[np.ndarray] = []
        self.normals_computed = False

    def rotate(self, rotation: np.ndarray, center: tuple[float, float, float] = (0, 0, 0)) -> _FakeDrawable:
        self.rotate_calls.append((np.asarray(rotation), center))
        return self

    def scale(self, factor: float, center: tuple[float, float, float] = (0, 0, 0)) -> _FakeDrawable:
        self.scale_calls.append((factor, center))
        return self

    def translate(self, offset: np.ndarray) -> _FakeDrawable:
        self.translate_calls.append(np.asarray(offset))
        return self

    def compute_vertex_normals(self) -> _FakeDrawable:
        self.normals_computed = True
        return self

    def paint_uniform_color(self, color: list[float] | np.ndarray) -> _FakeDrawable:
        self.color = np.asarray(color, dtype=np.float32)
        return self


class _FakeTriangleMesh(_FakeDrawable):
    def __init__(self, vertices: Any = None, triangles: Any = None) -> None:
        super().__init__()
        self.vertices = np.asarray(vertices if vertices is not None else np.zeros((0, 3), dtype=np.float32))
        self.triangles = np.asarray(triangles if triangles is not None else np.zeros((0, 3), dtype=np.int32))

    def create_coordinate_frame(
        self, size: float = 1.0, origin: tuple[float, float, float] = (0, 0, 0)
    ) -> _FakeTriangleMesh:
        _ = size, origin
        return _FakeTriangleMesh()


class _FakeAxisAlignedBoundingBox:
    def __init__(self, min_bound: tuple[float, float, float], max_bound: tuple[float, float, float]) -> None:
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.color: np.ndarray | None = None


class _FakePointCloud:
    def __init__(self, points: Any = None) -> None:
        self.points = np.asarray(points if points is not None else np.zeros((0, 3), dtype=np.float32))
        self.normals: np.ndarray | None = None
        self.colors: np.ndarray | None = None


class _FakeGenerator:
    instances: ClassVar[list[_FakeGenerator]] = []

    def __init__(self, model: Any, resolution: int) -> None:
        self.model = model
        self.resolution = resolution
        self.generate_grid_calls: list[dict[str, torch.Tensor]] = []
        self.__class__.instances.append(self)

    def generate_grid(self, batch: dict[str, torch.Tensor]) -> list[np.ndarray]:
        self.generate_grid_calls.append(batch)
        return [np.zeros((2, 2, 2), dtype=np.float32)]

    def extract_mesh(self, grid: np.ndarray) -> SimpleNamespace:
        return SimpleNamespace(
            vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            faces=np.array([[0, 1, 2]], dtype=np.int32),
            grid_shape=grid.shape,
        )


class _FakeConvONet:
    def __init__(self, padding: float = 0.0) -> None:
        self.padding = padding
        self.loaded_state: dict[str, Any] | None = None
        self.eval_called = False
        self.device: torch.device | None = None

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.loaded_state = state_dict

    def eval(self) -> _FakeConvONet:
        self.eval_called = True
        return self

    def to(self, device: torch.device) -> _FakeConvONet:
        self.device = device
        return self


def _install_fake_o3d(monkeypatch: Any, draw_calls: list[list[Any]]) -> None:
    fake_o3d = SimpleNamespace(
        geometry=SimpleNamespace(
            PointCloud=_FakePointCloud,
            TriangleMesh=_FakeTriangleMesh,
            AxisAlignedBoundingBox=_FakeAxisAlignedBoundingBox,
        ),
        utility=SimpleNamespace(
            Vector3dVector=lambda values: np.asarray(values, dtype=np.float32),
            Vector3iVector=lambda values: np.asarray(values, dtype=np.int32),
        ),
        visualization=SimpleNamespace(draw_geometries=lambda geometries: draw_calls.append(list(geometries))),
    )
    monkeypatch.setattr(script, "o3d", fake_o3d)


def test_main_smoke_hidden_point_branch(monkeypatch: Any, tmp_path: Path) -> None:
    args = Namespace(
        input=tmp_path / "input.ply",
        weights=tmp_path / "weights.pt",
        render=False,
        resolution=12,
        vis=False,
        flip_yz=False,
        scale=0.0,
        padding=0.2,
    )
    points = np.array([[0.0, 0.0, 0.0], [0.2, 0.1, 0.3]], dtype=np.float32)
    normals = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    fake_mesh = _FakeTriangleMesh()

    _FakeGenerator.instances.clear()

    monkeypatch.setattr(script.ArgumentParser, "parse_args", lambda self: args)
    monkeypatch.setattr(script, "load_mesh", lambda path: fake_mesh)
    monkeypatch.setattr(script, "normalize_mesh", lambda mesh: mesh)
    monkeypatch.setattr(script, "rot_from_euler", lambda **_kwargs: (np.eye(3, dtype=np.float32), 0.0))
    monkeypatch.setattr(script, "hidden_point_removal", lambda mesh: (points, normals))
    monkeypatch.setattr(script, "augment_points", lambda pts, *_args, **_kwargs: pts + 1.0)
    monkeypatch.setattr(
        script,
        "normalize_points",
        lambda pts: (pts * 0.5, np.array([1.0, 2.0, 3.0], dtype=np.float32), 2.0),
    )
    monkeypatch.setattr(script.R, "from_euler", lambda *_args, **_kwargs: SimpleNamespace(as_matrix=lambda: np.eye(3)))
    monkeypatch.setattr(script, "load_model", lambda path, padding: {"weights": path, "padding": padding})
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    monkeypatch.setattr(script, "Generator", _FakeGenerator)

    script.main()

    assert len(_FakeGenerator.instances) == 1
    assert _FakeGenerator.instances[0].resolution == 12
    batch = _FakeGenerator.instances[0].generate_grid_calls[0]
    assert batch["inputs"].shape == (1, 2, 3)
    assert torch.allclose(batch["inputs"], torch.tensor([[[0.5, 0.5, 0.5], [0.6, 0.55, 0.65]]]))
    assert len(fake_mesh.rotate_calls) == 1
    assert len(fake_mesh.scale_calls) == 1


def test_main_smoke_render_scale_and_visualize(monkeypatch: Any, tmp_path: Path) -> None:
    args = Namespace(
        input=tmp_path / "input.ply",
        weights=tmp_path / "weights.pt",
        render=True,
        resolution=8,
        vis=True,
        flip_yz=True,
        scale=1.5,
        padding=0.3,
    )
    points = np.array([[0.0, 0.0, 0.2], [0.1, -0.2, 0.4]], dtype=np.float32)
    normals = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    fake_mesh = _FakeTriangleMesh()
    draw_calls: list[list[Any]] = []

    _FakeGenerator.instances.clear()
    _install_fake_o3d(monkeypatch, draw_calls)

    monkeypatch.setattr(script.ArgumentParser, "parse_args", lambda self: args)
    monkeypatch.setattr(script, "load_mesh", lambda path: fake_mesh)
    monkeypatch.setattr(script, "normalize_mesh", lambda mesh: mesh)
    monkeypatch.setattr(
        script,
        "rot_from_euler",
        lambda **_kwargs: (np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32), 12.0),
    )
    monkeypatch.setattr(script, "render", lambda mesh: (points, normals))
    monkeypatch.setattr(script, "augment_points", lambda pts, *_args, **_kwargs: pts)
    monkeypatch.setattr(
        script,
        "scale_points",
        lambda pts, scale: (pts / scale, np.array([0.1, 0.2, 0.3], dtype=np.float32), scale),
    )
    monkeypatch.setattr(script.R, "from_euler", lambda *_args, **_kwargs: SimpleNamespace(as_matrix=lambda: np.eye(3)))
    monkeypatch.setattr(script, "load_model", lambda path, padding: {"weights": path, "padding": padding})
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    monkeypatch.setattr(script, "Generator", _FakeGenerator)

    script.main()

    assert len(_FakeGenerator.instances) == 1
    assert _FakeGenerator.instances[0].resolution == 8
    assert len(draw_calls) == 2
    assert len(fake_mesh.rotate_calls) == 2


def test_load_model_restores_state_and_moves_to_cpu(monkeypatch: Any, tmp_path: Path) -> None:
    weights_path = tmp_path / "weights.ckpt"

    monkeypatch.setattr(script.torch, "load", lambda path, weights_only=False: {"model": {"layer": torch.ones(1)}})
    monkeypatch.setattr(script, "ConvONet", _FakeConvONet)
    monkeypatch.setattr(script.torch.cuda, "is_available", lambda: False)

    model = script.load_model(weights_path, padding=0.25)

    assert isinstance(model, _FakeConvONet)
    assert model.padding == 0.25
    assert model.loaded_state == {"layer": torch.ones(1)}
    assert model.eval_called is True
    assert model.device == torch.device("cpu")


def test_visualize_builds_preview_geometries(monkeypatch: Any) -> None:
    draw_calls: list[list[Any]] = []
    _install_fake_o3d(monkeypatch, draw_calls)

    points = np.array([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]], dtype=np.float32)
    normals = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    script.visualize(points, normals=normals, padding=0.4)

    assert len(draw_calls) == 1
    geometries = draw_calls[0]
    assert len(geometries) == 3
    assert isinstance(geometries[0], _FakePointCloud)
    assert geometries[1].__class__ is _FakeTriangleMesh
    assert isinstance(geometries[2], _FakeAxisAlignedBoundingBox)
