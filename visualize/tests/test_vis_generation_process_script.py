from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import numpy as np
import torch
import trimesh
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from visualize.scripts import vis_generation_process as script


class _FakeFabric:
    def __init__(self, precision: str) -> None:
        self.precision = precision

    def setup_module(self, module: Any) -> Any:
        return module

    def autocast(self) -> Any:
        return nullcontext()


class _FakeGenerator:
    instances: ClassVar[list[_FakeGenerator]] = []

    def __init__(self, model: Any, **kwargs: Any) -> None:
        self.model = model
        self.kwargs = kwargs
        self.query_points = torch.zeros((8, 3), dtype=torch.float32)
        self.grid_shape = (2, 2, 2)
        self.__class__.instances.append(self)

    def extract_mesh(self, grid: np.ndarray) -> trimesh.Trimesh:
        _ = grid
        return trimesh.Trimesh(
            vertices=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=np.float32),
            faces=np.array([[0, 1, 2]], dtype=np.int32),
            process=False,
        )


class _FakeDataset:
    def __init__(self, item: dict[str, Any]) -> None:
        self.item = item

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, Any]:
        assert index == 0
        return self.item


class _FakeMetadataDataset:
    def __init__(self, objects: list[dict[str, str]]) -> None:
        self.objects = objects

    def __len__(self) -> int:
        return len(self.objects)


class _FakeDiffusionBase:
    pass


class _FakeAutoregressiveBase:
    pass


class _FakeDiffusionModel(_FakeDiffusionBase):
    def __init__(self) -> None:
        self.condition_key = "conditioning"
        self.device: torch.device | None = None
        self.generate_calls: list[dict[str, Any]] = []

    def to(self, device: torch.device) -> _FakeDiffusionModel:
        self.device = device
        return self

    def requires_grad_(self, requires_grad: bool) -> _FakeDiffusionModel:
        _ = requires_grad
        return self

    def generate(self, **kwargs: Any) -> tuple[torch.Tensor, list[torch.Tensor]]:
        self.generate_calls.append(kwargs)
        return torch.zeros((1, 1), dtype=torch.float32), [torch.ones((1, 4), dtype=torch.float32)]


class _FakeAutoregressor:
    n_block = 4


class _FakeAutoregressiveModel(_FakeAutoregressiveBase):
    def __init__(self) -> None:
        self.condition_key = "conditioning"
        self.device: torch.device | None = None
        self.generate_calls: list[dict[str, Any]] = []
        self._autoregressor = _FakeAutoregressor()

    def to(self, device: torch.device) -> _FakeAutoregressiveModel:
        self.device = device
        return self

    def requires_grad_(self, requires_grad: bool) -> _FakeAutoregressiveModel:
        _ = requires_grad
        return self

    def generate(self, **kwargs: Any) -> tuple[torch.Tensor, list[torch.Tensor]]:
        self.generate_calls.append(kwargs)
        return torch.zeros((1, 1), dtype=torch.float32), [torch.tensor([[1]], dtype=torch.long)]


class _EchoDecoder:
    def decode(self, values: torch.Tensor) -> torch.Tensor:
        return values


class _FakePredictor:
    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []

    def decode(self, *, points: torch.Tensor, feature: torch.Tensor) -> dict[str, torch.Tensor]:
        _ = feature
        self.calls.append(points)
        logits = torch.full((points.size(0), points.size(1), 1), 2.0, dtype=torch.float32)
        return {"logits": logits}


def _base_cfg() -> DictConfig:
    return OmegaConf.create(
        {
            "test": {"split": "test", "precision": "32-true"},
            "val": {"precision": "32-true"},
            "model": {"weights": None, "checkpoint": None, "load_best": False},
            "implicit": {"threshold": 0.5, "sdf": False},
            "norm": {"padding": 0.1, "bounds": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]},
            "vis": {
                "objects": [0],
                "implicit_render": False,
                "resolution": 2,
                "num_query_points": 8,
                "diff_steps": [0],
                "ar_steps": [1],
                "num_steps": 1,
                "temperature": 1.0,
            },
        }
    )


def _dataset_item() -> dict[str, Any]:
    return {
        "inputs.name": "shape",
        "category.id": "chairs",
        "inputs.file": 7,
        "inputs": torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=torch.float32),
        "conditioning": torch.tensor([[0.5, 0.5, 0.0], [0.6, 0.5, 0.0]], dtype=torch.float32),
        "mesh.vertices": np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]], dtype=np.float32),
        "mesh.triangles": np.array([[0, 1, 2]], dtype=np.int32),
    }


def test_default_object_indices_sorts_by_metadata() -> None:
    dataset = _FakeMetadataDataset(
        [
            {"category": "b", "name": "z"},
            {"category": "a", "name": "y"},
            {"category": "a", "name": "x"},
        ]
    )

    result = script._default_object_indices(dataset, limit=2)

    assert result == [2, 1]


def test_prepare_conditioning_and_select_input_points() -> None:
    inputs = torch.zeros((1, 2, 3), dtype=torch.float32)
    conditioning = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)

    prepared = script._prepare_conditioning(conditioning, inputs)
    selected = script._select_input_points_for_visualization(inputs, prepared, "conditioning")

    assert prepared is not None
    assert prepared.shape == (1, 2, 3)
    assert torch.equal(selected, prepared)


def test_render_points_preview_and_reference_panels(tmp_path: Path) -> None:
    image = script._render_points_preview(
        np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        intrinsic=np.eye(3, dtype=np.float32),
        extrinsic=np.eye(4, dtype=np.float32),
        image_size=8,
        color=np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )

    method_dir = tmp_path / "panels"
    method_dir.mkdir()
    implicit_renderer = SimpleNamespace(
        intrinsic=torch.eye(3, dtype=torch.float32).unsqueeze(0),
        extrinsic=torch.eye(4, dtype=torch.float32).unsqueeze(0),
        image_size=8,
    )
    gt_mesh = trimesh.Trimesh(
        vertices=np.array([[0.0, 0.0, 1.0], [0.2, 0.0, 1.0], [0.0, 0.2, 1.0]], dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.int32),
        process=False,
    )

    script._save_implicit_reference_panels(
        method_dir,
        cast(Any, implicit_renderer),
        np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        gt_mesh,
    )

    assert isinstance(image, Image.Image)
    assert (method_dir / "input.png").is_file()
    assert (method_dir / "gt.png").is_file()


def test_decode_ar_feature_and_logits_from_feature() -> None:
    partial_indices = torch.tensor([[1, 2]], dtype=torch.long)
    padded = script.decode_ar_feature(_EchoDecoder(), partial_indices, n_total=4)

    predictor = _FakePredictor()
    logits = script.logits_from_feature(
        predictor,
        feature=torch.ones((1, 2), dtype=torch.float32),
        points=torch.zeros((1, 5, 3), dtype=torch.float32),
        points_batch_size=2,
        sdf=True,
        sdf_tau=0.5,
        sdf_iso=1.0,
    )

    assert torch.equal(padded, torch.tensor([[1, 2, 0, 0]], dtype=torch.long))
    assert logits.shape == (1, 5)
    assert len(predictor.calls) == 3
    assert torch.allclose(logits, torch.full((1, 5), -2.0))


def test_filter_components_and_smooth_normals() -> None:
    mask = np.array(
        [
            [True, True, False],
            [False, False, False],
            [False, False, True],
        ],
        dtype=bool,
    )
    filtered = script._filter_components_2d(mask, keep_largest=True, min_pixels=1)

    normals = np.dstack([mask.astype(np.float32), np.zeros((3, 3), dtype=np.float32), np.ones((3, 3), dtype=np.float32)])
    smoothed = script._smooth_normals_map(normals, filtered, kernel_size=2, passes=1)

    assert filtered.sum() == 2
    assert smoothed.shape == normals.shape
    assert np.isfinite(smoothed).all()


def test_main_smoke_diffusion_writes_outputs(monkeypatch: Any, tmp_path: Path) -> None:
    cfg = _base_cfg()
    save_calls: list[Path] = []
    model = _FakeDiffusionModel()

    _FakeGenerator.instances.clear()

    monkeypatch.setattr(script, "setup_config", lambda cfg: cfg)
    monkeypatch.setattr(script, "suppress_known_optional_dependency_warnings", lambda: None)
    monkeypatch.setattr(script, "resolve_save_dir", lambda cfg: tmp_path)
    monkeypatch.setattr(script, "get_dataset", lambda cfg, splits: {"test": _FakeDataset(_dataset_item())})
    monkeypatch.setattr(script, "get_model", lambda cfg: model)
    monkeypatch.setattr(script, "patch_attention", lambda model, backend: model)
    monkeypatch.setattr(script.lightning, "Fabric", _FakeFabric)
    monkeypatch.setattr(script, "DiffusionModel", _FakeDiffusionBase)
    monkeypatch.setattr(script, "AutoregressiveModel", _FakeAutoregressiveBase)
    monkeypatch.setattr(script, "Generator", _FakeGenerator)
    monkeypatch.setattr(
        script,
        "decode_diffusion_intermediate",
        lambda model, latent, generator, points_batch_size=None: generator.extract_mesh(np.zeros((2, 2, 2))),
    )
    monkeypatch.setattr(script, "save_mesh", lambda path, vertices, faces: save_calls.append(Path(path)) or Path(path).write_text("mesh"))
    monkeypatch.setattr(script.torch.cuda, "is_available", lambda: False)

    script.main.__wrapped__(cfg)

    out_dir = tmp_path / "generation_process" / "chairs" / "shape_f00007"
    assert (out_dir / "input.ply").is_file()
    assert (out_dir / "gt.ply").is_file()
    assert save_calls == [out_dir / "diffusion" / "step_00.ply"]
    assert model.generate_calls[0]["num_steps"] == 1
    assert _FakeGenerator.instances[0].kwargs["resolution"] == 2


def test_main_smoke_autoregressive_writes_outputs(monkeypatch: Any, tmp_path: Path) -> None:
    cfg = _base_cfg()
    model = _FakeAutoregressiveModel()
    save_calls: list[Path] = []

    monkeypatch.setattr(script, "setup_config", lambda cfg: cfg)
    monkeypatch.setattr(script, "suppress_known_optional_dependency_warnings", lambda: None)
    monkeypatch.setattr(script, "resolve_save_dir", lambda cfg: tmp_path)
    monkeypatch.setattr(script, "get_dataset", lambda cfg, splits: {"test": _FakeDataset(_dataset_item())})
    monkeypatch.setattr(script, "get_model", lambda cfg: model)
    monkeypatch.setattr(script, "patch_attention", lambda model, backend: model)
    monkeypatch.setattr(script.lightning, "Fabric", _FakeFabric)
    monkeypatch.setattr(script, "DiffusionModel", _FakeDiffusionBase)
    monkeypatch.setattr(script, "AutoregressiveModel", _FakeAutoregressiveBase)
    monkeypatch.setattr(script, "Generator", _FakeGenerator)
    monkeypatch.setattr(
        script,
        "decode_ar_intermediate",
        lambda model, partial_indices, n_total, generator, points_batch_size=None: generator.extract_mesh(
            np.zeros((2, 2, 2))
        ),
    )
    monkeypatch.setattr(script, "save_mesh", lambda path, vertices, faces: save_calls.append(Path(path)) or Path(path).write_text("mesh"))
    monkeypatch.setattr(script.torch.cuda, "is_available", lambda: False)

    script.main.__wrapped__(cfg)

    out_dir = tmp_path / "generation_process" / "chairs" / "shape_f00007"
    assert save_calls == [out_dir / "ar" / "token_001.ply"]
    assert model.generate_calls[0]["temperature"] == 1.0
