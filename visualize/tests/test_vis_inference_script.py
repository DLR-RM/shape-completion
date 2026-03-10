from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from visualize.scripts import vis_inference as script


class _FakeFabric:
    def __init__(self, precision: str) -> None:
        self.precision = precision

    def setup_module(self, module: Any) -> Any:
        return module

    def autocast(self) -> Any:
        return nullcontext()


class _FakeTransform:
    def __init__(self) -> None:
        self.added_keys: list[list[str]] = []

    def add_keys(self, keys: list[str]) -> None:
        self.added_keys.append(keys)


class _FakeDataset:
    def __init__(self, item: dict[str, Any]) -> None:
        self.item = item
        self.transform = [object(), _FakeTransform(), object()]
        self.accessed: list[int] = []
        self.objects = [{"name": "shape", "category": "dummy"}]
        self.metadata = {"dummy": {"name": "Dummy Category"}}

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, Any]:
        self.accessed.append(index)
        return self.item


class _FakeModel:
    def __init__(self) -> None:
        self.device: torch.device | None = None
        self.eval_called = False
        self.decode_calls: list[tuple[torch.Tensor, Any]] = []

    def eval(self) -> _FakeModel:
        self.eval_called = True
        return self

    def to(self, device: torch.device) -> _FakeModel:
        self.device = device
        return self

    def decode(self, points: torch.Tensor, feature: Any) -> torch.Tensor:
        self.decode_calls.append((points.detach().clone(), feature))
        return torch.linspace(-1.0, 1.0, points.shape[1], dtype=torch.float32).unsqueeze(0)


class _FakeMesh:
    def __init__(self, vertices: Any = None, faces: Any = None, process: bool = False) -> None:
        _ = process
        vertices_array = np.asarray(vertices if vertices is not None else np.zeros((0, 3), dtype=np.float32))
        self.vertices = vertices_array.reshape((-1, 3)) if vertices_array.size else np.zeros((0, 3), dtype=np.float32)
        faces_array = np.asarray(faces if faces is not None else np.zeros((0, 3), dtype=np.int32))
        self.faces = faces_array.reshape((-1, 3)) if faces_array.size else np.zeros((0, 3), dtype=np.int32)

    def sample(self, n: int, return_index: bool = False) -> Any:
        points = np.zeros((n, 3), dtype=np.float32)
        indices = np.zeros((n,), dtype=np.int32)
        if return_index:
            return points, indices
        return points

    @property
    def face_normals(self) -> np.ndarray:
        return np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (max(len(self.faces), 1), 1))


class _FakeGenerator:
    instances: ClassVar[list[_FakeGenerator]] = []

    def __init__(self, model: Any, **kwargs: Any) -> None:
        self.model = model
        self.kwargs = kwargs
        self.fabric = _FakeFabric("32-true")
        self.bounds = (-0.5, 0.5)
        self.resolution0 = kwargs["resolution"]
        self.generate_grid_calls: list[dict[str, Any]] = []
        self.extract_mesh_calls: list[tuple[np.ndarray, Any]] = []
        self.estimate_vertex_normals_calls: list[tuple[np.ndarray, Any]] = []
        self.__class__.instances.append(self)

    def generate_grid(self, item: dict[str, Any], extraction_class: int | None = None) -> tuple[np.ndarray, np.ndarray, str]:
        _ = extraction_class
        self.generate_grid_calls.append(item)
        grid = np.ones((2, 2, 2), dtype=np.float32)
        points = np.array(
            [
                [-0.25, -0.25, -0.25],
                [0.25, -0.25, -0.25],
                [-0.25, 0.25, -0.25],
                [0.25, 0.25, -0.25],
                [-0.25, -0.25, 0.25],
                [0.25, -0.25, 0.25],
                [-0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25],
            ],
            dtype=np.float32,
        )
        return grid, points, "feature"

    def extract_mesh(
        self,
        grid: np.ndarray,
        feature: Any = None,
        threshold: float | None = None,
        extraction_class: int | None = None,
    ) -> _FakeMesh:
        _ = threshold, extraction_class
        self.extract_mesh_calls.append((np.asarray(grid), feature))
        return _FakeMesh(
            vertices=np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]], dtype=np.float32),
            faces=np.array([[0, 1, 2]], dtype=np.int32),
        )

    def estimate_vertex_normals(self, points: np.ndarray, feature: Any, normalize: bool = False) -> np.ndarray:
        _ = normalize
        self.estimate_vertex_normals_calls.append((np.asarray(points), feature))
        return np.tile(np.array([[0.0, 0.0, 0.1]], dtype=np.float32), (len(points), 1))


class _FakeO3DTriangleMesh:
    last_instance: _FakeO3DTriangleMesh | None = None

    def __init__(self, vertices: Any = None, triangles: Any = None) -> None:
        self.vertices = np.asarray(vertices if vertices is not None else np.zeros((0, 3), dtype=np.float32))
        self.triangles = np.asarray(triangles if triangles is not None else np.zeros((0, 3), dtype=np.int32))
        self.colors: list[tuple[float, float, float]] = []
        self.normals_computed = False
        self.removed_masks: list[np.ndarray] = []
        self.__class__.last_instance = self

    def paint_uniform_color(self, color: tuple[float, float, float]) -> _FakeO3DTriangleMesh:
        self.colors.append(color)
        return self

    def compute_vertex_normals(self) -> _FakeO3DTriangleMesh:
        self.normals_computed = True
        return self

    def sample_points_uniformly(self, n: int) -> _FakeO3DPointCloud:
        points = np.tile(np.array([[0.0, 0.0, 0.0]], dtype=np.float32), (max(n, 3), 1))
        return _FakeO3DPointCloud(points)

    @classmethod
    def create_from_point_cloud_ball_pivoting(cls, pcd: Any, radii: Any) -> _FakeO3DTriangleMesh:
        _ = pcd, radii
        return cls(
            vertices=np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]], dtype=np.float32),
            triangles=np.array([[0, 1, 2]], dtype=np.int32),
        )

    @classmethod
    def create_from_point_cloud_poisson(
        cls, pcd: Any, depth: int = 5, n_threads: int | None = None
    ) -> tuple[_FakeO3DTriangleMesh, None]:
        _ = pcd, depth, n_threads
        return (
            cls(
                vertices=np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]], dtype=np.float32),
                triangles=np.array([[0, 1, 2]], dtype=np.int32),
            ),
            None,
        )

    def cluster_connected_triangles(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.array([0, 0, 1], dtype=np.int32),
            np.array([2, 1], dtype=np.int32),
            np.array([1.0, 0.5], dtype=np.float32),
        )

    def remove_triangles_by_mask(self, mask: np.ndarray) -> None:
        self.removed_masks.append(np.asarray(mask, dtype=bool))


class _FakeO3DPointCloud:
    def __init__(self, points: Any = None) -> None:
        self.vertices = np.asarray(points if points is not None else np.zeros((0, 3), dtype=np.float32))
        self.colors: list[tuple[float, float, float]] = []

    def paint_uniform_color(self, color: tuple[float, float, float]) -> _FakeO3DPointCloud:
        self.colors.append(color)
        return self


class _FakeFigure:
    def __init__(self) -> None:
        self.saved_paths: list[str] = []

    def savefig(self, path: Any) -> None:
        path_str = str(path)
        Path(path_str).parent.mkdir(parents=True, exist_ok=True)
        Path(path_str).touch()
        self.saved_paths.append(path_str)


def _base_cfg() -> DictConfig:
    return OmegaConf.create(
        {
            "data": {"test_ds": ["dummy"], "frame": "world"},
            "test": {"split": "test"},
            "model": {"weights": "stub", "checkpoint": None, "load_best": False},
            "val": {"precision": "32-true"},
            "cls": {"num_classes": 2},
            "aug": {"scale": None},
            "vis": {
                "num_query_points": 8,
                "refinement_steps": 0,
                "simplify": False,
                "resolution": 2,
                "normals": False,
                "inputs": False,
                "occupancy": False,
                "points": False,
                "frame": False,
                "pointcloud": False,
                "mesh": False,
                "box": False,
                "cam": False,
                "save": False,
                "show": False,
            },
            "implicit": {"threshold": 0.5, "sdf": False},
            "norm": {"padding": 0.1},
            "misc": {"seed": 0},
        }
    )


def test_occupancy_contour_plot_returns_three_axes() -> None:
    vals = np.linspace(-1.0, 1.0, 4, dtype=np.float32)
    xx, yy, zz = np.meshgrid(vals, vals, vals, indexing="ij")
    fig, axes = script.occupancy_contour_plot(xx + yy + zz)

    assert len(axes) == 3
    plt.close(fig)


def test_evaluate_handles_empty_and_mesh_metrics(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        script,
        "get_metrics",
        lambda probs, occ, threshold=None: {"score": float(torch.sum(probs) + torch.sum(occ))},
    )
    monkeypatch.setattr(script, "check_mesh_contains", lambda mesh, points: np.array([1, 0], dtype=np.int64))

    empty = script.evaluate(torch.zeros(2), torch.zeros(2), threshold=0.5)
    results = script.evaluate(
        torch.tensor([1.0, 0.0]),
        torch.tensor([1.0, 0.0]),
        points=np.zeros((2, 3), dtype=np.float32),
        mesh=cast(Any, _FakeMesh()),
        threshold=0.5,
    )

    assert empty == script.EMPTY_RESULTS_DICT
    assert results == {"score": 2.0, "m_score": 2.0}


def test_process_mesh_removes_minor_components(monkeypatch: Any) -> None:
    fake_o3d = SimpleNamespace(
        geometry=SimpleNamespace(TriangleMesh=_FakeO3DTriangleMesh),
        utility=SimpleNamespace(
            Vector3dVector=lambda values: np.asarray(values, dtype=np.float32),
            Vector3iVector=lambda values: np.asarray(values, dtype=np.int32),
        ),
    )
    monkeypatch.setattr(script, "o3d", fake_o3d)

    result = script.process_mesh(
        cast(
            Any,
            _FakeMesh(
            vertices=np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]], dtype=np.float32),
            faces=np.array([[0, 1, 2], [0, 2, 1], [1, 2, 0]], dtype=np.int32),
            ),
        ),
        min_num_triangles=2,
    )

    assert isinstance(result, script.Trimesh)
    assert _FakeO3DTriangleMesh.last_instance is not None
    assert np.array_equal(_FakeO3DTriangleMesh.last_instance.removed_masks[0], np.array([False, False, True]))


def test_main_smoke_runs_one_iteration(monkeypatch: Any, tmp_path: Any) -> None:
    item = {
        "inputs.name": "shape",
        "index": 0,
        "inputs": np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float32),
        "points": np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float32),
        "points.occ": np.array([1, 0], dtype=np.int64),
        "mesh.vertices": np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]], dtype=np.float32),
        "mesh.triangles": np.array([[0, 1, 2]], dtype=np.int32),
    }
    dataset = _FakeDataset(item)
    draw_calls: list[list[Any]] = []

    fake_o3d = SimpleNamespace(
        geometry=SimpleNamespace(TriangleMesh=_FakeO3DTriangleMesh, PointCloud=_FakeO3DPointCloud),
        utility=SimpleNamespace(
            Vector3dVector=lambda values: np.asarray(values, dtype=np.float32),
            Vector3iVector=lambda values: np.asarray(values, dtype=np.int32),
        ),
        visualization=SimpleNamespace(draw_geometries=lambda geometries, **kwargs: draw_calls.append(list(geometries))),
    )

    _FakeGenerator.instances.clear()

    monkeypatch.setattr(script, "setup_config", lambda cfg: cfg)
    monkeypatch.setattr(script, "suppress_known_optional_dependency_warnings", lambda: None)
    monkeypatch.setattr(script, "log_optional_dependency_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(script, "resolve_save_dir", lambda cfg: tmp_path)
    monkeypatch.setattr(script, "get_dataset", lambda cfg, splits: {"test": dataset})
    monkeypatch.setattr(script, "get_model", lambda cfg: _FakeModel())
    monkeypatch.setattr(script.lightning, "Fabric", _FakeFabric)
    monkeypatch.setattr(script, "Generator", _FakeGenerator)
    monkeypatch.setattr(script, "Visualize", lambda **kwargs: lambda item: None)
    monkeypatch.setattr(script, "Trimesh", _FakeMesh)
    monkeypatch.setattr(script, "o3d", fake_o3d)
    monkeypatch.setattr(script, "trange", lambda n: range(n))
    monkeypatch.setattr(script.torch.cuda, "is_available", lambda: False)

    script.main.__wrapped__(_base_cfg())

    assert dataset.accessed == [0]
    assert len(_FakeGenerator.instances) == 1
    assert _FakeGenerator.instances[0].kwargs["resolution"] == 2
    assert dataset.transform[-2].added_keys
    assert len(draw_calls) == 1
    assert len(draw_calls[0]) == 3


def test_main_save_metrics_and_uncertainty_paths(monkeypatch: Any, tmp_path: Any) -> None:
    item = {
        "inputs.name": "shape",
        "index": 0,
        "inputs": np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=np.float32),
        "points": np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=np.float32),
        "points.occ": np.array([1, 2, 0], dtype=np.int64),
        "mesh.vertices": np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]], dtype=np.float32),
        "mesh.triangles": np.array([[0, 1, 2]], dtype=np.int32),
        "inputs.scale": 1.5,
        "inputs.frame": np.eye(3, dtype=np.float32),
    }
    dataset = _FakeDataset(item)
    written_triangle_meshes: list[str] = []
    written_point_clouds: list[str] = []

    fake_o3d = SimpleNamespace(
        geometry=SimpleNamespace(TriangleMesh=_FakeO3DTriangleMesh, PointCloud=_FakeO3DPointCloud),
        utility=SimpleNamespace(
            Vector3dVector=lambda values: np.asarray(values, dtype=np.float32),
            Vector3iVector=lambda values: np.asarray(values, dtype=np.int32),
            DoubleVector=lambda values: list(values),
        ),
        io=SimpleNamespace(
            write_triangle_mesh=lambda path, mesh: (Path(path).touch(), written_triangle_meshes.append(path))[1],
            write_point_cloud=lambda path, mesh: (Path(path).touch(), written_point_clouds.append(path))[1],
        ),
        visualization=SimpleNamespace(draw_geometries=lambda *args, **kwargs: None),
    )

    cfg = _base_cfg()
    cfg.vis.save = True
    cfg.vis.plot_contour = True
    cfg.val.metrics = ["iou"]
    cfg.implicit.uncertain_threshold = 0.6

    _FakeGenerator.instances.clear()
    figures: list[_FakeFigure] = []
    model = _FakeModel()

    monkeypatch.setattr(script, "setup_config", lambda cfg: cfg)
    monkeypatch.setattr(script, "suppress_known_optional_dependency_warnings", lambda: None)
    monkeypatch.setattr(script, "log_optional_dependency_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(script, "resolve_save_dir", lambda cfg: tmp_path)
    monkeypatch.setattr(script, "get_dataset", lambda cfg, splits: {"test": dataset})
    monkeypatch.setattr(script, "get_model", lambda cfg: model)
    monkeypatch.setattr(script.lightning, "Fabric", _FakeFabric)
    monkeypatch.setattr(script, "Generator", _FakeGenerator)
    monkeypatch.setattr(script, "Visualize", lambda **kwargs: lambda item: None)
    monkeypatch.setattr(script, "Trimesh", _FakeMesh)
    monkeypatch.setattr(script, "o3d", fake_o3d)
    monkeypatch.setattr(script, "trange", lambda n: range(n))
    monkeypatch.setattr(script.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        script,
        "to_tensor",
        lambda x, unsqueeze=True, device=None: (
            torch.as_tensor(x).unsqueeze(0) if unsqueeze and isinstance(x, np.ndarray) else torch.as_tensor(x)
        ),
    )
    monkeypatch.setattr(script, "probs_from_logits", lambda logits: torch.tensor([[0.9, 0.7, 0.2]], dtype=torch.float32))
    monkeypatch.setattr(
        script,
        "check_mesh_contains",
        lambda mesh, points: np.pad(
            np.array([1], dtype=np.int64), (0, max(len(points) - 1, 0)), constant_values=0
        ),
    )
    monkeypatch.setattr(script, "eval_pointcloud", lambda *args, **kwargs: {"cd": 0.1})
    monkeypatch.setattr(script, "occupancy_contour_plot", lambda *args, **kwargs: (figures.append(_FakeFigure()) or figures[-1], [None, None, None]))

    script.main.__wrapped__(cfg)

    output_dir = tmp_path / "generation" / "meshes"
    metrics_dir = tmp_path / "generation" / "metrics"

    assert dataset.accessed == [0]
    assert len(_FakeGenerator.instances) == 1
    assert len(_FakeGenerator.instances[0].estimate_vertex_normals_calls) >= 2
    assert len(model.decode_calls) >= 1
    assert (output_dir / "shape_0_params.npz").exists()
    assert (output_dir / "shape_0_contour.png").exists()
    assert (output_dir / "shape_0_unc_contour.png").exists()
    assert (output_dir / "shape_0_grad_contour.png").exists()
    assert (metrics_dir / "test_eval_full_0.50_0.60.pkl").exists()
    assert (metrics_dir / "test_eval_0.50_0.60.csv").exists()
    assert any(path.endswith("shape_0_gt.ply") for path in written_triangle_meshes)
    assert any(path.endswith("shape_0_mesh.ply") for path in written_triangle_meshes)
    assert any(path.endswith("shape_0_uncertain.ply") for path in written_triangle_meshes)
    assert any(path.endswith("shape_0_uncertain_gt.ply") for path in written_triangle_meshes)
    assert any(path.endswith("shape_0_inputs.ply") for path in written_point_clouds)
