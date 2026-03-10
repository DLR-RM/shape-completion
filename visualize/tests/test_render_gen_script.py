from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np

from visualize.scripts import render_gen as script


class _FakeMesh:
    def __init__(self, vertices: np.ndarray | None = None, faces: np.ndarray | None = None, **_: Any) -> None:
        self.vertices = np.asarray(
            vertices if vertices is not None else [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]],
            dtype=np.float32,
        )
        self.faces = np.asarray(faces if faces is not None else [[0, 1, 2]], dtype=np.int32)
        self.bounds = np.array([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=np.float32)
        self.extents = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.transforms: list[np.ndarray] = []
        self.translations: list[np.ndarray] = []
        self.scales: list[float] = []

    def apply_transform(self, trafo: np.ndarray) -> None:
        self.transforms.append(np.asarray(trafo))

    def apply_translation(self, offset: list[float] | np.ndarray) -> None:
        self.translations.append(np.asarray(offset, dtype=np.float32))

    def apply_scale(self, scale: float) -> None:
        self.scales.append(scale)


class _FakePointCloud:
    def __init__(self, points: np.ndarray | None = None) -> None:
        self.vertices = np.asarray(points if points is not None else [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float32)
        self.transforms: list[np.ndarray] = []
        self.scales: list[float] = []

    def apply_transform(self, trafo: np.ndarray) -> None:
        self.transforms.append(np.asarray(trafo))

    def apply_scale(self, scale: float) -> None:
        self.scales.append(scale)


class _FakeOpen3DPointCloud:
    def __init__(self) -> None:
        self.points = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float32)


class _FakeRenderer:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.calls: list[dict[str, Any]] = []
        self.default_mesh_color = np.array([0.3, 0.4, 0.5], dtype=np.float32)
        self.default_pcd_color = np.array([0.1, 0.2, 0.8], dtype=np.float32)

    def __call__(self, **kwargs: Any) -> dict[str, np.ndarray]:
        self.calls.append(kwargs)
        image = np.zeros((8, 8, 4), dtype=np.uint8)
        image[..., 3] = 255
        return {"color": image}


def test_main_smoke_writes_generation_panels(monkeypatch: Any, tmp_path: Path) -> None:
    dataset_root = Path("/data/shapenet")
    train_root = Path("/data/train")
    figure_root = Path("/data/figures")
    renderer_holder: dict[str, _FakeRenderer] = {}
    set_levels: list[int] = []

    def fake_path(value: str | Path) -> Path:
        real = Path(value)
        text = str(real)
        if text.startswith(str(dataset_root)):
            return tmp_path / "datasets" / real.relative_to(dataset_root)
        if text.startswith(str(train_root)):
            return tmp_path / "train" / real.relative_to(train_root)
        if text.startswith(str(figure_root)):
            return tmp_path / "figures" / real.relative_to(figure_root)
        return real

    def fake_renderer(*args: Any, **kwargs: Any) -> _FakeRenderer:
        renderer_holder["instance"] = _FakeRenderer(*args, **kwargs)
        return renderer_holder["instance"]

    monkeypatch.setattr(
        script.ArgumentParser,
        "parse_args",
        lambda self: Namespace(individual=False, look_at="centroid", transparent=True, show=False, verbose=2),
    )
    monkeypatch.setattr(script, "Path", fake_path)
    monkeypatch.setattr(script, "seed_everything", lambda seed: None)
    monkeypatch.setattr(script, "set_log_level", lambda level: set_levels.append(level))
    monkeypatch.setattr(script, "Renderer", fake_renderer)
    monkeypatch.setattr(script, "Trimesh", _FakeMesh)
    monkeypatch.setattr(script.trimesh, "PointCloud", lambda points: _FakePointCloud(np.asarray(points, dtype=np.float32)))
    monkeypatch.setattr(script.trimesh, "load", lambda path: _FakeMesh())
    monkeypatch.setattr(
        script.trimesh.transformations,
        "rotation_matrix",
        lambda angle, axis: np.eye(4, dtype=np.float32),
    )
    monkeypatch.setattr(script, "load_mesh", lambda path: (_FakeMesh().vertices, _FakeMesh().faces))
    monkeypatch.setattr(script, "normalize_mesh", lambda mesh: mesh)
    monkeypatch.setattr(script, "look_at", lambda *args, **kwargs: np.eye(4, dtype=np.float32))
    monkeypatch.setattr(script, "inv_trafo", lambda matrix: matrix)
    monkeypatch.setattr(script.np, "load", lambda path: np.eye(4, dtype=np.float32))
    monkeypatch.setattr(script.o3d.io, "read_point_cloud", lambda path: _FakeOpen3DPointCloud())

    script.main()

    out_dir = tmp_path / "figures" / "03001627" / "e6b0b43093f105277997a53584de8fa7"
    assert renderer_holder["instance"].kwargs["transparent_background"] is True
    assert set_levels == [script.DEBUG_LEVEL_2]
    assert (out_dir / "inputs.png").is_file()
    assert (out_dir / "gt.png").is_file()
    assert (out_dir / "disc.png").is_file()
    assert (out_dir / "gen_0.png").is_file()
    assert (out_dir / "gen_9.png").is_file()
    assert len(renderer_holder["instance"].calls) == 13
