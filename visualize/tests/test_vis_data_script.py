from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf


class _FakeDataset:
    def __init__(self) -> None:
        self.name = "fake-dataset"
        self.category_weights = [1.0]
        self.accessed: list[int] = []

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, Any]:
        self.accessed.append(index)
        return {"value": index}


def _make_dataset_module(dataset: _FakeDataset) -> types.ModuleType:
    module = types.ModuleType("dataset")
    module_any = cast(Any, module)
    module_any.SaveData = object
    module_any.SharedDataLoader = object
    module_any.SharedDataset = lambda dataset, shared_dict, shared_hash_map: dataset
    module_any.Visualize = object
    module_any.get_dataset = lambda cfg, splits: {"test": dataset}
    return module


def _load_vis_data_module(monkeypatch: Any, dataset_module: types.ModuleType, train_module: types.ModuleType) -> Any:
    module_name = "test_vis_data_script_module"
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "vis_data.py"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "dataset", dataset_module)
    monkeypatch.setitem(sys.modules, "train", train_module)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _base_cfg() -> DictConfig:
    return OmegaConf.create(
        {
            "vis": {"split": "test", "show": False, "save": False, "index": None, "use_loader": False},
            "data": {"cache": False},
            "test": {"shuffle": False},
            "train": {"fast_dev_run": 0, "epochs": 1},
            "log": {"verbose": 0, "progress": False},
        }
    )


def test_main_smoke_without_loader(monkeypatch: Any) -> None:
    dataset = _FakeDataset()
    dataset_module = _make_dataset_module(dataset)
    train_module = types.ModuleType("train")

    module = _load_vis_data_module(monkeypatch, dataset_module, train_module)
    monkeypatch.setattr(module, "setup_config", lambda cfg, **kwargs: cfg)
    monkeypatch.setattr(module, "suppress_known_optional_dependency_warnings", lambda: None)
    monkeypatch.setattr(module, "log_optional_dependency_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "tqdm", lambda iterable, **kwargs: iterable)

    module.main.__wrapped__(_base_cfg())

    assert dataset.accessed == [0]


def test_main_smoke_with_loader(monkeypatch: Any) -> None:
    dataset = _FakeDataset()
    dataset_module = _make_dataset_module(dataset)
    train_module = types.ModuleType("train")
    loader_calls: list[dict[str, Any]] = []

    module = _load_vis_data_module(monkeypatch, dataset_module, train_module)
    monkeypatch.setattr(module, "setup_config", lambda cfg, **kwargs: cfg)
    monkeypatch.setattr(module, "suppress_known_optional_dependency_warnings", lambda: None)
    monkeypatch.setattr(module, "log_optional_dependency_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "get_num_workers", lambda value: value)
    monkeypatch.setattr(module, "tqdm", lambda iterable, **kwargs: iterable)
    monkeypatch.setattr(
        module,
        "DataLoader",
        lambda *args, **kwargs: loader_calls.append(kwargs) or [{"index": [0]}],
    )

    cfg = OmegaConf.merge(
        _base_cfg(),
        OmegaConf.create(
            {
                "vis": {"use_loader": True, "index": "0"},
                "load": {"num_workers": 0, "weighted": True, "pin_memory": False, "prefetch_factor": None},
                "misc": {"seed": 0},
                "data": {"test_ds": ["dummy"], "cache": False},
                "test": {"shuffle": False},
            }
        ),
    )

    module.main.__wrapped__(cfg)

    assert loader_calls
    assert loader_calls[0]["batch_size"] == 1
    assert loader_calls[0]["sampler"] is not None
