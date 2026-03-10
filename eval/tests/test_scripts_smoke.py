from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

from eval.scripts import compare_eval as compare_eval_script
from eval.scripts import eval as eval_script
from eval.scripts import gen_eval as gen_eval_script
from eval.scripts import generate as generate_script
from eval.scripts import mesh_eval as mesh_eval_script


class _FakeFabric:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    @contextmanager
    def autocast(self) -> Any:
        yield


class _FakeModel:
    name = "FakeModel"

    def eval(self) -> _FakeModel:
        return self

    def to(self, device: torch.device) -> _FakeModel:
        self.device = device
        return self

    def on_validation_epoch_end(self) -> dict[str, float]:
        return {}


class _FakeScriptDataset(Dataset[dict[str, Any]]):
    categories: ClassVar[list[str]] = ["001", "002"]

    def __init__(self, input_path: Path) -> None:
        self._input_path = input_path

    def __len__(self) -> int:
        return 1

    def __getitem__(self, _index: int) -> dict[str, Any]:
        return {
            "index": torch.tensor(0),
            "inputs": torch.zeros(8, 3),
            "points": torch.zeros(8, 3),
            "points.occ": torch.zeros(8),
            "category.name": "cat",
            "category.id": "001",
            "inputs.name": "obj",
            "inputs.path": str(self._input_path),
        }


class _FakeGenerator:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.query_points = torch.zeros((1, 3))
        self.estimate_normals = False

    def generate_grid(
        self,
        _data: dict[str, Any],
        **_kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray, None]:
        return np.zeros((2, 2, 2), dtype=np.float32), np.zeros((1, 3), dtype=np.float32), None

    def generate_grid_per_instance(self, _data: dict[str, Any], **_kwargs: Any) -> np.ndarray:
        return np.zeros((2, 2, 2), dtype=np.float32)


class _FakeMeshVisual:
    def __init__(self) -> None:
        self.vertex_colors = np.array([[10, 20, 30, 255], [20, 30, 40, 255], [30, 40, 50, 255]], dtype=np.uint8)


class _FakeMesh:
    def __init__(self) -> None:
        self.vertices = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=np.float32)
        self.faces = np.array([[0, 1, 2]], dtype=np.int32)
        self.vertex_normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        self.visual = _FakeMeshVisual()

    def show(self, *_args: Any, **_kwargs: Any) -> None:
        pass


class _FakeProcessGenerator:
    def __init__(self, meshes: _FakeMesh | list[_FakeMesh]) -> None:
        self._meshes = meshes
        self.estimate_normals = False
        self.query_points = torch.zeros((1, 3))

    def extract_meshes(self, **_kwargs: Any) -> _FakeMesh | list[_FakeMesh]:
        return self._meshes


class _FakeParallel:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def __enter__(self) -> _FakeParallel:
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> bool:
        return False

    def __call__(self, tasks: Any) -> list[Any]:
        return list(tasks)


class _FakeFigure:
    def __init__(self) -> None:
        self.shown = False

    def show(self) -> None:
        self.shown = True


@contextmanager
def _null_context(*_args: Any, **_kwargs: Any) -> Any:
    yield


def _eval_cfg() -> DictConfig:
    return OmegaConf.create(
        {
            "test": {
                "split": "test",
                "batch_size": 1,
                "shuffle": False,
                "precision": "16-mixed",
                "basic": True,
                "overwrite": True,
            },
            "vis": {"save": False, "index": None, "num_query_points": 0, "show": False},
            "data": {"test_ds": ["dummy"], "split": False, "num_files": {"test": 1}, "num_shards": {"test": 1}},
            "files": {"suffix": ""},
            "points": {"load_uncertain": False},
            "implicit": {"threshold": 0.5, "sdf": False},
            "load": {"num_workers": 0, "prefetch_factor": None, "pin_memory": False},
            "misc": {"seed": 0},
            "model": {"weights": None, "checkpoint": None, "load_best": False},
            "log": {"name": "eval-smoke", "progress": False, "verbose": 0, "metrics": []},
            "train": {"model_selection_metric": "val/loss"},
        }
    )


def _generate_cfg() -> DictConfig:
    return OmegaConf.create(
        {
            "test": {"split": "test", "batch_size": 1, "shuffle": False, "precision": "16-mixed", "overwrite": True},
            "val": {"precision": "16-mixed"},
            "vis": {
                "index": None,
                "resolution": 8,
                "upsampling_steps": 0,
                "num_query_points": 8,
                "refinement_steps": 0,
                "normals": False,
                "colors": False,
                "simplify": False,
                "num_instances": None,
                "show": False,
                "save": False,
            },
            "implicit": {"threshold": 0.5, "sdf": False},
            "norm": {"padding": 0.1, "scale_factor": 1.0, "bounds": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]},
            "load": {"num_workers": 0, "prefetch_factor": None, "pin_memory": False},
            "misc": {"seed": 0},
            "model": {"weights": None, "checkpoint": None, "load_best": False},
            "predict": {"instances": False},
            "inputs": {"type": "depth", "project": False, "fps": {"num_points": 0}},
            "data": {"categories": [], "split": False, "num_files": {"test": 1}, "num_shards": {"test": 1}},
            "log": {"name": "generate-smoke", "progress": False, "verbose": 0},
        }
    )


def _process_cfg() -> DictConfig:
    return OmegaConf.create(
        {
            "vis": {"num_instances": None, "save": False, "show": False},
            "inputs": {"type": "depth", "project": False, "fps": {"num_points": 0}},
            "test": {"split": "test"},
            "data": {"num_files": {"test": 1}, "num_shards": {"test": 1}},
        }
    )


def _mesh_eval_cfg() -> DictConfig:
    return OmegaConf.create(
        {
            "test": {"split": "test", "merge": False, "overwrite": True},
            "data": {"test_ds": ["dummy"], "num_files": {"test": 1}, "num_shards": {"test": 1}},
            "implicit": {"threshold": 0.5},
            "files": {"suffix": ""},
            "aug": {"noise": 0, "edge_noise": 0, "remove_angle": False},
            "pointcloud": {"normals": False},
            "log": {"name": "mesh-eval-smoke", "verbose": 0, "progress": False},
            "load": {"num_workers": 0, "prefetch_factor": None, "pin_memory": False},
            "misc": {"seed": 0},
            "vis": {"show": False},
            "train": {"model_selection_metric": "val/f1"},
        }
    )


def _gen_eval_cfg(log_dir: Path) -> DictConfig:
    return OmegaConf.create(
        {
            "test": {
                "split": "test",
                "batch_size": 1,
                "metrics": ["invalid"],
                "overwrite": True,
                "num_instances": None,
            },
            "pointcloud": {"num_points": 8, "test": {"num_points": 8}},
            "load": {"num_workers": 0, "hdf5": False},
            "log": {"project": "proj", "name": "run", "verbose": 0, "progress": False},
            "vis": {"show": False},
            "data": {"test_ds": ["dummy"], "categories": []},
            "implicit": {"threshold": 0.5},
            "dirs": {"log": str(log_dir)},
            "misc": {"seed": 0},
        }
    )


def test_eval_main_smoke(monkeypatch: Any, tmp_path: Path) -> None:
    input_path = tmp_path / "input.ply"
    input_path.write_text("ply")
    dataset = _FakeScriptDataset(input_path)

    monkeypatch.setattr(eval_script, "CocoInstanceSegmentation", _FakeScriptDataset)
    monkeypatch.setattr(eval_script, "setup_config", lambda cfg: cfg)
    monkeypatch.setattr(eval_script, "suppress_known_optional_dependency_warnings", lambda: None)
    monkeypatch.setattr(eval_script, "log_optional_dependency_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(eval_script, "resolve_save_dir", lambda *_args, **_kwargs: tmp_path)
    monkeypatch.setattr(eval_script, "overwrite_results", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(eval_script, "config_hash", lambda *_args, **_kwargs: "smokeid")
    monkeypatch.setattr(eval_script, "get_num_workers", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(eval_script, "get_dataset", lambda *_args, **_kwargs: {"test": dataset})
    monkeypatch.setattr(eval_script, "get_model", lambda *_args, **_kwargs: _FakeModel())
    monkeypatch.setattr(eval_script.lightning, "Fabric", _FakeFabric)
    monkeypatch.setattr(eval_script, "summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(eval_script, "test_step", lambda *_args, **_kwargs: {"loss": 0.0})
    monkeypatch.setattr("train.get_collate_fn", lambda *_args, **_kwargs: None)

    eval_script.main.__wrapped__(_eval_cfg())

    assert (tmp_path / "dummy_test_eval_full_smokeid.pkl").is_file()
    assert (tmp_path / "dummy_test_eval_full_smokeid.csv").is_file()
    assert (tmp_path / "dummy_test_eval_full_smokeid.txt").is_file()


def test_generate_main_smoke(monkeypatch: Any, tmp_path: Path) -> None:
    input_path = tmp_path / "input.ply"
    input_path.write_text("ply")
    dataset = _FakeScriptDataset(input_path)
    process_calls: list[int] = []

    def _fake_process_item(
        _cfg: DictConfig,
        item: dict[str, Any],
        _feature: Any,
        _dataset: Dataset[dict[str, Any]],
        _generator: Any,
        _obj_counter: defaultdict[Any, int],
        _vis_dir: Path,
        _meshes_dir: Path,
        _inputs_dir: Path,
    ) -> None:
        process_calls.append(int(np.asarray(item["index"]).item()))

    monkeypatch.setattr(generate_script, "setup_config", lambda cfg: cfg)
    monkeypatch.setattr(generate_script, "suppress_known_optional_dependency_warnings", lambda: None)
    monkeypatch.setattr(generate_script, "log_optional_dependency_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(generate_script, "resolve_save_dir", lambda *_args, **_kwargs: tmp_path)
    monkeypatch.setattr(generate_script, "overwrite_results", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(generate_script, "get_num_workers", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(generate_script, "get_dataset", lambda *_args, **_kwargs: {"test": dataset})
    monkeypatch.setattr(generate_script, "get_model", lambda *_args, **_kwargs: _FakeModel())
    monkeypatch.setattr(generate_script.lightning, "Fabric", _FakeFabric)
    monkeypatch.setattr(generate_script, "summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(generate_script, "Generator", _FakeGenerator)
    monkeypatch.setattr(generate_script, "to_tensor", lambda value, **_kwargs: value)
    monkeypatch.setattr(
        generate_script,
        "to_numpy",
        lambda value: value.detach().cpu().numpy() if torch.is_tensor(value) else value,
    )
    monkeypatch.setattr(generate_script, "process_item", _fake_process_item)
    monkeypatch.setattr("train.get_collate_fn", lambda *_args, **_kwargs: None)

    generate_script.main.__wrapped__(_generate_cfg())

    assert process_calls == [0]
    assert (tmp_path / "generation" / "test" / "meshes").is_dir()
    assert (tmp_path / "generation" / "test" / "vis").is_dir()
    assert (tmp_path / "generation" / "test" / "inputs").is_dir()


def test_generate_process_item_scalar_category(monkeypatch: Any, tmp_path: Path) -> None:
    cfg = _process_cfg()
    input_path = tmp_path / "input.ply"
    input_path.write_text("ply")

    dataset = _FakeScriptDataset(input_path)
    monkeypatch.setattr(generate_script, "CocoInstanceSegmentation", _FakeScriptDataset)
    monkeypatch.setattr(generate_script, "save_mesh", lambda path, *_args: Path(path).write_text("mesh"))

    vis_dir = tmp_path / "vis"
    meshes_dir = tmp_path / "meshes"
    inputs_dir = tmp_path / "inputs"
    vis_dir.mkdir()
    meshes_dir.mkdir()
    inputs_dir.mkdir()

    item: dict[str, Any] = {
        "index": 0,
        "category.name": "cat",
        "category.id": "001",
        "inputs.name": "obj",
        "inputs.path": str(input_path),
        "grid": np.zeros((2, 2, 2), dtype=np.float32),
        "inputs.inv_extrinsic": np.eye(4),
    }
    obj_counter: defaultdict[str, int] = defaultdict(int)
    generator = _FakeProcessGenerator(_FakeMesh())

    generate_script.process_item(
        cfg, item, None, dataset, cast(Any, generator), obj_counter, vis_dir, meshes_dir, inputs_dir
    )

    assert obj_counter["001"] == 1
    assert (meshes_dir / "001" / "obj.ply").is_file()
    assert (vis_dir / "001_cat" / "00_mesh.ply").is_file()
    assert (vis_dir / "001_cat" / "00_inputs.ply").is_file()


def test_generate_process_item_list_category(monkeypatch: Any, tmp_path: Path) -> None:
    cfg = _process_cfg()
    input_path = tmp_path / "input.ply"
    input_path.write_text("ply")

    dataset = _FakeScriptDataset(input_path)
    monkeypatch.setattr(generate_script, "CocoInstanceSegmentation", _FakeScriptDataset)
    monkeypatch.setattr(generate_script, "save_mesh", lambda path, *_args: Path(path).write_text("mesh"))

    vis_dir = tmp_path / "vis"
    meshes_dir = tmp_path / "meshes"
    inputs_dir = tmp_path / "inputs"
    vis_dir.mkdir()
    meshes_dir.mkdir()
    inputs_dir.mkdir()

    item: dict[str, Any] = {
        "index": 0,
        "category.name": ["cat1", "cat2"],
        "category.id": ["001", "002"],
        "inputs.name": ["obj1", "obj2"],
        "inputs.path": str(input_path),
        "grid": np.zeros((2, 2, 2), dtype=np.float32),
        "inputs.inv_extrinsic": np.eye(4),
    }
    obj_counter: defaultdict[str, int] = defaultdict(int)
    generator = _FakeProcessGenerator([_FakeMesh(), _FakeMesh()])

    generate_script.process_item(
        cfg, item, None, dataset, cast(Any, generator), obj_counter, vis_dir, meshes_dir, inputs_dir
    )

    assert obj_counter["001"] == 1
    assert obj_counter["002"] == 1
    assert (meshes_dir / "001" / "obj1.ply").is_file()
    assert (meshes_dir / "002" / "obj2.ply").is_file()
    assert (vis_dir / "001_cat1" / "00_mesh.ply").is_file()
    assert any((vis_dir / "002_cat2").glob("*.ply"))


def test_mesh_eval_main_smoke(monkeypatch: Any, tmp_path: Path) -> None:
    cfg = _mesh_eval_cfg()

    eval_item: dict[str, Any] = {
        "index": 0,
        "path": "dummy/input.ply",
        "object category": "001",
        "category name": "cat",
        "object name": "obj",
        "f1": 1.0,
    }

    monkeypatch.setattr(mesh_eval_script, "setup_config", lambda cfg: cfg)
    monkeypatch.setattr(mesh_eval_script, "suppress_known_optional_dependency_warnings", lambda: None)
    monkeypatch.setattr(mesh_eval_script, "log_optional_dependency_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mesh_eval_script, "resolve_save_dir", lambda *_args, **_kwargs: tmp_path)
    monkeypatch.setattr(mesh_eval_script, "overwrite_results", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(mesh_eval_script, "get_dataset", lambda *_args, **_kwargs: {"test": object()})
    monkeypatch.setattr(mesh_eval_script, "get_num_workers", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(mesh_eval_script, "stdout_redirected", _null_context)
    monkeypatch.setattr(mesh_eval_script, "DataLoader", lambda *_args, **_kwargs: [{}])
    monkeypatch.setattr(mesh_eval_script, "single_eval", lambda *_args, **_kwargs: eval_item)
    monkeypatch.setattr(mesh_eval_script, "tqdm_joblib", _null_context)
    monkeypatch.setattr(mesh_eval_script, "tqdm", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mesh_eval_script, "Parallel", _FakeParallel)
    monkeypatch.setattr(mesh_eval_script, "delayed", lambda fn: fn)

    mesh_eval_script.main.__wrapped__(cfg)

    save_dir = tmp_path / "generation" / "test"
    assert (save_dir / "dummy_test_mesh_eval_full_0.50.pkl").is_file()
    assert (save_dir / "dummy_test_mesh_eval_0.50.csv").is_file()
    assert (save_dir / "dummy_test_mesh_eval_0.50.txt").is_file()


def test_gen_eval_main_smoke(monkeypatch: Any, tmp_path: Path) -> None:
    mesh_dir = tmp_path / "generation" / "test" / "meshes"
    mesh_dir.mkdir(parents=True)
    (mesh_dir / "sample.ply").write_text("ply")

    monkeypatch.setattr(gen_eval_script, "setup_config", lambda cfg: cfg)
    monkeypatch.setattr(gen_eval_script, "suppress_known_optional_dependency_warnings", lambda: None)
    monkeypatch.setattr(gen_eval_script, "log_optional_dependency_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(gen_eval_script, "resolve_save_dir", lambda *_args, **_kwargs: tmp_path)
    monkeypatch.setattr(gen_eval_script, "resolve_path", lambda path: Path(path))
    monkeypatch.setattr(gen_eval_script, "get_num_workers", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(gen_eval_script.fid, "test_stats_exists", lambda *_args, **_kwargs: False)

    gen_eval_script.main.__wrapped__(_gen_eval_cfg(tmp_path))

    assert (tmp_path / "generation" / "test" / "dummy_test_gen_eval_0.50_sdfstylegan.txt").is_file()


def test_compare_eval_main_smoke(monkeypatch: Any, tmp_path: Path) -> None:
    csv_path = tmp_path / "test_eval_0.50.csv"
    csv_path.write_text("category name,f1\nmean,0.9\n")

    parse_args = argparse.Namespace(directory=tmp_path, files=None, file="test_eval_0.50.csv", cls="mean", metric="f1")
    called: dict[str, Any] = {}

    monkeypatch.setattr(compare_eval_script.argparse.ArgumentParser, "parse_args", lambda _self: parse_args)
    monkeypatch.setattr(
        compare_eval_script,
        "visualize_csv_files",
        lambda directory_or_paths, file, cls, metric: called.update(
            {"directory_or_paths": directory_or_paths, "file": file, "cls": cls, "metric": metric}
        ),
    )

    compare_eval_script.main()

    assert called["directory_or_paths"] == tmp_path
    assert called["file"] == "test_eval_0.50.csv"
    assert called["cls"] == "mean"
    assert called["metric"] == "f1"


def test_compare_eval_visualize_csv_files_smoke(monkeypatch: Any, tmp_path: Path) -> None:
    run_dir = tmp_path / "exp" / "generation" / "test"
    run_dir.mkdir(parents=True)
    csv_path = run_dir / "test_eval_0.50.csv"
    csv_path.write_text("category name,f1\nmean,0.7\nchair,0.4\n")

    captured: dict[str, Any] = {}
    figure = _FakeFigure()

    def _fake_bar(df: Any, **kwargs: Any) -> _FakeFigure:
        captured["df"] = df
        captured["kwargs"] = kwargs
        return figure

    monkeypatch.setattr(compare_eval_script.px, "bar", _fake_bar)

    compare_eval_script.visualize_csv_files(tmp_path, file="test_eval_0.50.csv", cls="mean", metric="f1")

    assert figure.shown is True
    assert list(captured["df"]["Experiment"]) == ["generation"]
    assert list(captured["df"]["F1"]) == [0.7]
