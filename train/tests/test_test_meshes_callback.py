from types import MethodType, SimpleNamespace
from typing import Any, cast

import numpy as np
import torch
from trimesh import PointCloud, Trimesh

from ..src.callbacks import test as test_callback_module


def _stub_visualize_init(self, *_args: Any, **kwargs: Any) -> None:
    self.inputs = bool(kwargs.get("inputs", True))
    self.upload_to_wandb = bool(kwargs.get("upload_to_wandb", False))
    self.images = {"front": [], "back": []}
    self.data = {"meshes": [], "inputs": []}
    self.items = []
    self._batch = []


def _init_callback(monkeypatch, **kwargs: Any) -> test_callback_module.TestMeshesCallback:
    monkeypatch.setattr(test_callback_module.VisualizeCallback, "__init__", _stub_visualize_init)
    return test_callback_module.TestMeshesCallback(**kwargs)


def _mesh() -> Trimesh:
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    return Trimesh(vertices=vertices, faces=faces, process=False)


def test_make_table_data_adds_mean_row() -> None:
    data = [{"cat": {"iou": 1.0, "chamfer": 2.0}}, {"dog": {"iou": 3.0, "chamfer": 4.0}}]

    table_data, headers = test_callback_module.TestMeshesCallback._make_table_data(data)

    assert headers == ["name", "iou", "chamfer"]
    assert table_data[-1][0] == "mean"
    assert table_data[-1][1] == 2.0
    assert table_data[-1][2] == 3.0


def test_eval_batch_skips_when_input_path_missing(monkeypatch) -> None:
    callback = _init_callback(monkeypatch)
    callback._batch = [{"mesh": _mesh(), "inputs.norm_offset": np.zeros(3)}]

    callback._eval_batch()

    assert callback.stats == []


def test_visualize_batch_wraps_xyz_inputs_as_pointcloud(monkeypatch) -> None:
    callback = _init_callback(monkeypatch, inputs=True)
    callback._batch = [{"mesh": _mesh(), "inputs": np.zeros((4, 3), dtype=np.float32)}]
    captured: dict[str, Any] = {}

    def _capture(meshes: list[Trimesh], pcds: list[PointCloud] | None) -> None:
        captured["meshes"] = meshes
        captured["pcds"] = pcds

    callback._render_meshes = _capture  # type: ignore[assignment]
    callback._visualize_batch()

    assert len(captured["meshes"]) == 1
    assert isinstance(captured["pcds"][0], PointCloud)


def test_on_test_batch_end_passes_model_to_process_batch(monkeypatch) -> None:
    callback = _init_callback(monkeypatch)
    seen: dict[str, Any] = {}
    callback._gather_items = MethodType(lambda _self, _world_size: None, callback)
    callback._add_item = MethodType(lambda _self, item: seen.setdefault("item", item), callback)
    callback._process_batch = MethodType(lambda _self, model: seen.setdefault("model", model), callback)
    callback._eval_batch = MethodType(lambda _self: seen.setdefault("eval_called", True), callback)
    callback._visualize_batch = MethodType(lambda _self: seen.setdefault("vis_called", True), callback)

    trainer = cast(Any, SimpleNamespace(world_size=1))
    pl_module = cast(Any, SimpleNamespace(model="dummy-model"))
    batch = {"inputs.path": ["/tmp/input"], "inputs": torch.zeros((1, 4, 3))}

    callback.on_test_batch_end(trainer, pl_module, None, cast(dict[str, Any], batch), batch_idx=0)

    assert seen["model"] == "dummy-model"
    assert seen["eval_called"] is True
    assert seen["vis_called"] is True


def test_on_test_epoch_end_gathers_distributed_stats(monkeypatch) -> None:
    callback = _init_callback(monkeypatch)
    callback.stats = [{"cat": {"iou": 1.0}}]

    def _gather(output: list[Any], local_stats: list[Any]) -> None:
        output[0] = local_stats
        output[1] = [{"dog": {"iou": 2.0}}]

    monkeypatch.setattr(torch.distributed, "all_gather_object", _gather)
    trainer = cast(Any, SimpleNamespace(world_size=2))

    callback.on_test_epoch_end(trainer, cast(Any, None))

    assert len(callback.stats) == 2
