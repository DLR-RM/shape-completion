from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
from trimesh import Trimesh

from ..src.callbacks import eval_meshes as eval_meshes_module


def _stub_visualize_init(self, *_args: Any, **kwargs: Any) -> None:
    self.items = []
    self.progress = bool(kwargs.get("progress", False))


def _init_callback(monkeypatch, **kwargs: Any) -> eval_meshes_module.EvalMeshesCallback:
    monkeypatch.setattr(eval_meshes_module.VisualizeCallback, "__init__", _stub_visualize_init)
    monkeypatch.setattr(eval_meshes_module, "get_num_workers", lambda num_workers=None: 1)
    callback = eval_meshes_module.EvalMeshesCallback(**kwargs)
    monkeypatch.setattr(eval_meshes_module, "tqdm_joblib", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(eval_meshes_module, "delayed", lambda fn: (lambda *args, **kw: lambda: fn(*args, **kw)))
    callback.__dict__["parallel"] = lambda jobs: [job() for job in jobs]
    return callback


def _mesh() -> Trimesh:
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    return Trimesh(vertices=vertices, faces=faces, process=False)


def test_eval_meshes_returns_empty_without_items(monkeypatch) -> None:
    callback = _init_callback(monkeypatch, metrics="all")

    assert callback._eval_meshes(cast(Any, SimpleNamespace(device="cpu"))) == {}


def test_eval_meshes_pointcloud_path_computes_mean(monkeypatch) -> None:
    callback = _init_callback(monkeypatch, metrics="pcd")
    mesh = _mesh()
    callback.items = [
        {
            "category.name": "a",
            "inputs.name": "a0",
            "mesh": mesh,
            "pointcloud": np.zeros((2, 3), dtype=np.float32),
        },
        {
            "category.name": "b",
            "inputs.name": "b000",
            "mesh": mesh,
            "pointcloud": np.ones((2, 3), dtype=np.float32),
        },
    ]
    monkeypatch.setattr(
        eval_meshes_module,
        "eval_mesh_pcd",
        lambda _mesh_obj, item: {"pcd_metric": float(len(cast(str, item["inputs.name"])))},
    )

    results = callback._eval_meshes(cast(Any, SimpleNamespace(device="cpu")))
    assert results is not None

    assert results["pcd_metric"] == 3.0


def test_eval_meshes_mesh_path_passes_normalize_flag(monkeypatch) -> None:
    callback = _init_callback(monkeypatch, metrics="mesh")
    mesh = _mesh()
    callback.items = [
        {
            "category.name": "a",
            "inputs.name": "a0",
            "mesh": mesh,
            "mesh.vertices": np.asarray(mesh.vertices),
            "mesh.triangles": np.asarray(mesh.faces),
            "points": np.zeros((2, 3), dtype=np.float32),
        }
    ]
    normalize_flags: list[bool] = []

    def _fake_eval_mesh(_pred: Trimesh, _gt: Trimesh, _query: np.ndarray | None, normalize: bool) -> dict[str, float]:
        normalize_flags.append(normalize)
        return {"mesh_metric": 1.0}

    monkeypatch.setattr(eval_meshes_module, "eval_mesh", _fake_eval_mesh)

    results = callback._eval_meshes(cast(Any, SimpleNamespace(device="cpu")))
    assert results is not None

    assert normalize_flags == [False]
    assert results["mesh_metric"] == 1.0
