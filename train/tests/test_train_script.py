from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import Dataset

from train.scripts import train as train_script


class _ToyDataset(Dataset[int]):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> int:
        return index


class _FakeTrainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.optimizer = None


class _FakeLitDataModule:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.test = kwargs.get("test")


class _FakeLitModel:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.model = kwargs["model"]


class _FakeTrainer:
    instances: ClassVar[list[_FakeTrainer]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.callbacks = list(kwargs.get("callbacks", []))
        self.logger = kwargs.get("logger")
        self.optimizers: list[Any] = []
        self.estimated_stepping_batches = 12
        self.fit_calls: list[tuple[Any, Any, Any]] = []
        self.test_calls: list[tuple[Any, Any, Any, Any]] = []
        self.checkpoint_callback = next(
            (cb for cb in self.callbacks if hasattr(cb, "best_model_path")),
            SimpleNamespace(best_model_path=None, last_model_path=None, best_model_score=None),
        )
        self.__class__.instances.append(self)

    def fit(self, model: Any, datamodule: Any, ckpt_path: Any = None) -> None:
        self.fit_calls.append((model, datamodule, ckpt_path))

    def test(self, model: Any, datamodule: Any, ckpt_path: Any = None, verbose: Any = None) -> None:
        self.test_calls.append((model, datamodule, ckpt_path, verbose))


class _FakeTuner:
    instances: ClassVar[list[_FakeTuner]] = []

    def __init__(self, trainer: _FakeTrainer) -> None:
        self.trainer = trainer
        self.scale_batch_size_calls = 0
        self.lr_find_calls = 0
        self.__class__.instances.append(self)

    def scale_batch_size(self, model: Any, datamodule: Any = None) -> None:
        self.scale_batch_size_calls += 1

    def lr_find(self, model: Any, datamodule: Any = None) -> None:
        self.lr_find_calls += 1


class _FakeTensorBoardLogger:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


class _FakeWandbConfig:
    def __init__(self) -> None:
        self.updates: list[tuple[Any, Any]] = []

    def update(self, values: Any, allow_val_change: bool = False) -> None:
        self.updates.append((values, allow_val_change))


class _FakeWandbExperiment:
    def __init__(self) -> None:
        self.config = _FakeWandbConfig()


class _FakeWandbLogger:
    instances: ClassVar[list[_FakeWandbLogger]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.experiment = _FakeWandbExperiment()
        self.watch_calls: list[tuple[Any, Any]] = []
        self.__class__.instances.append(self)

    def watch(self, model: Any, **kwargs: Any) -> None:
        self.watch_calls.append((model, kwargs))


class _FakeCallback:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


class _FakeModelCheckpoint(_FakeCallback):
    def __init__(self, checkpoint_path: Path, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.best_model_path = str(checkpoint_path)
        self.last_model_path = None
        self.best_model_score = 0.75


def _base_cfg() -> DictConfig:
    return OmegaConf.create(
        {
            "log": {
                "project": None,
                "name": None,
                "version": "v0",
                "id": None,
                "offline": False,
                "model": False,
                "gradients": False,
                "parameters": False,
                "graph": False,
                "freq": 1,
                "top_k": 1,
                "progress": "rich",
                "verbose": 2,
                "profile": False,
                "wandb": False,
                "metrics": [],
                "summary_depth": 1,
            },
            "model": {
                "arch": "fake_model",
                "weights": None,
                "checkpoint": None,
                "load_best": False,
                "compile": False,
                "average": "swa",
                "swa_lr": 0.05,
                "ema_decay": None,
            },
            "test": {"run": True, "dir": None, "filename": None, "split": "test"},
            "train": {
                "batch_size": 2,
                "accumulate_grad_batches": 1,
                "epochs": 20,
                "precision": "32",
                "gradient_clip_val": 0.0,
                "hypergradients": False,
                "lr": 1e-3,
                "min_lr": None,
                "scale_lr": False,
                "weight_decay": 0.1,
                "optimizer": "AdamW",
                "betas": [0.9, 0.95],
                "scheduler": "StepLR",
                "lr_step_size": 2,
                "lr_gamma": 0.5,
                "lr_reduction_factor": 0.1,
                "patience_factor": 1.0,
                "model_selection_metric": "val/loss",
                "overfit_batches": False,
                "skip": False,
                "find_batch_size": True,
                "find_lr": True,
                "early_stopping": True,
                "fast_dev_run": False,
                "num_batches": 1.0,
                "loss": None,
                "reduction": "mean",
                "detect_anomaly": False,
            },
            "val": {
                "batch_size": 2,
                "freq": 1,
                "precision": "32",
                "num_batches": 1.0,
                "num_sanity": 0,
                "visualize": False,
                "mesh": False,
                "vis_n_eval": 1,
                "vis_n_category": 0,
                "vis_n": 1,
                "num_query_points": 8,
            },
            "vis": {
                "inputs": True,
                "num_query_points": 8,
                "resolution": 8,
                "normals": False,
                "colors": False,
                "simplify": False,
                "refinement_steps": 0,
                "show": False,
                "mesh": False,
                "render": "",
            },
            "inputs": {"type": "pointcloud"},
            "norm": {"padding": 0.1},
            "load": {"num_workers": 0, "prefetch_factor": None, "pin_memory": False, "weighted": False},
            "data": {
                "cache": False,
                "hash_items": False,
                "share_memory": False,
                "num_files": {"train": 1},
            },
            "misc": {"seed": 0},
            "implicit": {"threshold": 0.5, "sdf": False},
        }
    )


def test_run_smoke_exercises_training_flow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    save_dir = tmp_path / "run"
    best_ckpt = tmp_path / "best.ckpt"
    best_ckpt.write_text("checkpoint")
    saved: dict[str, Any] = {}

    _FakeTrainer.instances.clear()
    _FakeTuner.instances.clear()

    monkeypatch.setattr(train_script, "setup_config", lambda cfg, **_kwargs: cfg)
    monkeypatch.setattr(train_script, "suppress_known_optional_dependency_warnings", lambda: None)
    monkeypatch.setattr(train_script, "log_optional_dependency_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_script, "resolve_save_dir", lambda _cfg: save_dir)
    monkeypatch.setattr(train_script, "resolve_checkpoint_path", lambda _cfg: None)
    monkeypatch.setattr(train_script, "resolve_path", lambda value: Path(value))
    monkeypatch.setattr(train_script.HydraConfig, "get", lambda: SimpleNamespace(job=SimpleNamespace(config_name="smoke")))
    monkeypatch.setattr(train_script, "get_dataset", lambda _cfg, splits: {split: _ToyDataset() for split in splits})
    monkeypatch.setattr(train_script, "get_num_workers", lambda value: value)
    monkeypatch.setattr(train_script, "get_collate_fn", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_script, "get_model", lambda _cfg: _FakeTrainModel())
    monkeypatch.setattr(
        train_script,
        "assign_params_groups",
        lambda model: ([model.linear.weight], [model.linear.bias]),
    )
    monkeypatch.setattr(train_script, "LitDataModule", _FakeLitDataModule)
    monkeypatch.setattr(train_script, "LitModel", _FakeLitModel)
    monkeypatch.setattr(train_script, "ModelCheckpoint", lambda *args, **kwargs: _FakeModelCheckpoint(best_ckpt, *args, **kwargs))
    monkeypatch.setattr(train_script, "RichModelSummary", _FakeCallback)
    monkeypatch.setattr(train_script, "LearningRateMonitor", _FakeCallback)
    monkeypatch.setattr(train_script, "RichProgressBar", _FakeCallback)
    monkeypatch.setattr(train_script, "EarlyStopping", _FakeCallback)
    monkeypatch.setattr(train_script, "StochasticWeightAveraging", _FakeCallback)
    monkeypatch.setattr(train_script, "TensorBoardLogger", _FakeTensorBoardLogger)
    monkeypatch.setattr(train_script.pl, "Trainer", _FakeTrainer)
    monkeypatch.setattr(train_script, "Tuner", _FakeTuner)
    monkeypatch.setattr(train_script.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(train_script.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        train_script,
        "save_best_model",
        lambda path, out_dir: (
            saved.__setitem__("best", (Path(path), out_dir)) or {"weight": torch.ones(1)}
        ),
    )
    monkeypatch.setattr(
        train_script,
        "save_ema_model",
        lambda trainer, model, out_dir, state_dict=None: saved.__setitem__("ema", (trainer, model, out_dir, state_dict)),
    )

    import train.src.callbacks as callbacks_module

    monkeypatch.setattr(callbacks_module, "TestMeshesCallback", _FakeCallback)

    score = train_script.run(_base_cfg())

    assert score == pytest.approx(0.75)
    assert len(_FakeTrainer.instances) == 2
    assert len(_FakeTrainer.instances[0].fit_calls) == 1
    assert len(_FakeTrainer.instances[1].test_calls) == 1
    assert _FakeTuner.instances[0].scale_batch_size_calls == 1
    assert _FakeTuner.instances[0].lr_find_calls == 1
    assert saved["best"][0] == best_ckpt
    assert saved["best"][1] == save_dir
    assert saved["ema"][2] == save_dir
    assert isinstance(_FakeTrainer.instances[0].kwargs["logger"], _FakeTensorBoardLogger)


def test_run_smoke_wandb_and_compile(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    save_dir = tmp_path / "wandb_run"
    finished: list[str] = []
    compiled: list[tuple[Any, str]] = []

    cfg = _base_cfg()
    cfg.test.run = False
    cfg.model.compile = True
    cfg.model.average = None
    cfg.train.skip = True
    cfg.train.find_batch_size = False
    cfg.train.find_lr = False
    cfg.train.early_stopping = False
    cfg.train.weight_decay = 0.0
    cfg.train.scheduler = None
    cfg.train.hypergradients = False
    cfg.log.wandb = True
    cfg.log.gradients = True
    cfg.log.graph = True

    _FakeTrainer.instances.clear()
    _FakeWandbLogger.instances.clear()

    monkeypatch.setattr(train_script, "setup_config", lambda cfg, **_kwargs: cfg)
    monkeypatch.setattr(train_script, "suppress_known_optional_dependency_warnings", lambda: None)
    monkeypatch.setattr(train_script, "log_optional_dependency_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_script, "resolve_save_dir", lambda _cfg: save_dir)
    monkeypatch.setattr(train_script.HydraConfig, "get", lambda: SimpleNamespace(job=SimpleNamespace(config_name="smoke")))
    monkeypatch.setattr(train_script, "get_dataset", lambda _cfg, splits: {split: _ToyDataset() for split in splits})
    monkeypatch.setattr(train_script, "get_num_workers", lambda value: value)
    monkeypatch.setattr(train_script, "get_collate_fn", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_script, "get_model", lambda _cfg: _FakeTrainModel())
    monkeypatch.setattr(train_script, "LitDataModule", _FakeLitDataModule)
    monkeypatch.setattr(train_script, "LitModel", _FakeLitModel)
    monkeypatch.setattr(train_script, "ModelCheckpoint", lambda *args, **kwargs: _FakeModelCheckpoint(tmp_path / "unused.ckpt", *args, **kwargs))
    monkeypatch.setattr(train_script, "RichModelSummary", _FakeCallback)
    monkeypatch.setattr(train_script, "RichProgressBar", _FakeCallback)
    monkeypatch.setattr(train_script, "WandbLogger", _FakeWandbLogger)
    monkeypatch.setattr(train_script.pl, "Trainer", _FakeTrainer)
    monkeypatch.setattr(train_script.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(train_script.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        train_script.torch,
        "compile",
        lambda model, mode="default": compiled.append((model, mode)) or model,
    )
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(finish=lambda: finished.append("done")))

    score = train_script.run(cfg)

    assert score == 0.75
    assert len(compiled) == 1
    assert len(_FakeWandbLogger.instances) == 1
    assert len(_FakeWandbLogger.instances[0].watch_calls) == 1
    assert len(_FakeWandbLogger.instances[0].experiment.config.updates) == 1
    assert finished == ["done"]
