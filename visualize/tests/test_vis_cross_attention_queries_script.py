from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from visualize.scripts import vis_cross_attention_queries as script


class _FakeFabric:
    def __init__(self, precision: str) -> None:
        self.precision = precision

    def setup_module(self, module: Any) -> Any:
        return module

    def autocast(self) -> Any:
        return nullcontext()


class _FakeDataset:
    def __init__(self, item: dict[str, Any], objects: list[dict[str, str]]) -> None:
        self.item = item
        self.objects = objects
        self.accessed: list[int] = []

    def __len__(self) -> int:
        return len(self.objects)

    def __getitem__(self, index: int) -> dict[str, Any]:
        self.accessed.append(index)
        return self.item


class _FakeModel:
    def __init__(self) -> None:
        self.device: torch.device | None = None
        self.requires_grad_flag: bool | None = None

    def to(self, device: torch.device) -> _FakeModel:
        self.device = device
        return self

    def requires_grad_(self, flag: bool) -> _FakeModel:
        self.requires_grad_flag = flag
        return self


class _FakeEncoder:
    def __init__(self) -> None:
        self.eval_called = False

    def eval(self) -> _FakeEncoder:
        self.eval_called = True
        return self


def _base_cfg(output_dir: Path) -> DictConfig:
    return OmegaConf.create(
        {
            "test": {"split": "test", "precision": "32-true"},
            "val": {"precision": "32-true"},
            "model": {"weights": "stub", "checkpoint": None, "load_best": False},
            "vis": {
                "cross_attn_output_dir": str(output_dir),
                "objects": [1],
                "max_input_points": 2,
                "cross_attn_proj_size": 16,
                "cross_attn_proj_radius": 1,
                "cross_attn_proj_cam_location": [0.0, 0.0, 0.0],
                "cross_attn_proj_cam_target": [0.0, 0.0, 1.0],
            },
        }
    )


def test_default_object_indices_and_encoder_resolution(monkeypatch: Any) -> None:
    dataset = _FakeDataset(
        item={"inputs": np.zeros((1, 3), dtype=np.float32)},
        objects=[
            {"category": "03001627", "name": "chair_b"},
            {"category": "02691156", "name": "plane_a"},
            {"category": "03001627", "name": "chair_a"},
        ],
    )

    assert script._default_object_indices(dataset, limit=2) == [1, 2]

    class _DummyShape:
        def __init__(self, tag: str) -> None:
            self.tag = tag

    monkeypatch.setattr(script, "Shape3D2VecSet", _DummyShape)
    resolved = script._resolve_shape3d2vecset_encoder(
        SimpleNamespace(
            _vae=SimpleNamespace(ae=_DummyShape("vae")),
            _discretizer=SimpleNamespace(ae=_DummyShape("disc")),
            ae=_DummyShape("direct"),
        )
    )

    assert isinstance(resolved, _DummyShape)
    assert resolved.tag == "vae"


def test_projection_and_color_helpers_draw_pixels(monkeypatch: Any) -> None:
    monkeypatch.setattr(script, "look_at", lambda eye, target: np.eye(4, dtype=np.float32))
    monkeypatch.setattr(script, "inv_trafo", lambda matrix: matrix)
    monkeypatch.setattr(script, "convert_extrinsic", lambda matrix, src, dst: matrix)

    query_colors = script._to_query_colors(3)
    uncertainty_colors = script._uncertainty_colors_from_entropy(np.array([0.0, 1.0], dtype=np.float32))
    image = script._project_colored_points_to_image(
        points=np.array([[0.0, 0.0, 2.0], [0.5, 0.0, 2.0]], dtype=np.float32),
        rgb=np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8),
        image_size=16,
        cam_location=(0.0, 0.0, 0.0),
        cam_target=(0.0, 0.0, 1.0),
        cam_focal=None,
        point_radius=1,
    )

    assert query_colors.shape == (3, 3)
    assert query_colors.dtype == np.uint8
    assert np.array_equal(uncertainty_colors[0], np.array([0, 0, 255], dtype=np.uint8))
    assert np.array_equal(uncertainty_colors[1], np.array([255, 255, 76], dtype=np.uint8))
    assert np.asarray(image).shape == (16, 16, 3)
    assert np.any(np.asarray(image) != 255)


def test_compute_attention_weights_returns_expected_shapes(monkeypatch: Any) -> None:
    class _FakeCrossAttn:
        n_head = 1

        @staticmethod
        def to_q(tensor: torch.Tensor) -> torch.Tensor:
            return tensor

        @staticmethod
        def to_kv(tensor: torch.Tensor) -> torch.Tensor:
            return torch.cat([tensor, tensor], dim=2)

    class _FakeBlock:
        def __init__(self) -> None:
            self.cross_attn = _FakeCrossAttn()

        @staticmethod
        def ln_2(tensor: torch.Tensor) -> torch.Tensor:
            return tensor

    fake_ae = SimpleNamespace(query_levels=[2], nerf_enc=lambda tensor: tensor, inputs_enc=_FakeBlock())

    monkeypatch.setattr(script, "furthest_point_sample", lambda inputs, num_samples: inputs[:, :num_samples, :])

    (input_points, query_points, attn_mean), owners, entropy_norm, margin = script._compute_attention_weights(
        cast(Any, fake_ae),
        torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]],
            dtype=torch.float32,
        ),
    )

    assert input_points.shape == (3, 2)
    assert query_points.shape == (2, 2)
    assert attn_mean.shape == (2, 3)
    assert owners.shape == (3,)
    assert entropy_norm.shape == (3,)
    assert margin.shape == (3,)


def test_main_smoke_writes_outputs(monkeypatch: Any, tmp_path: Path) -> None:
    item = {
        "inputs.name": "shape_a",
        "category.id": "02691156",
        "inputs": np.array(
            [
                [0.0, 0.0, 2.0],
                [0.3, 0.0, 2.0],
                [0.0, 0.3, 2.0],
                [0.3, 0.3, 2.0],
            ],
            dtype=np.float32,
        ),
    }
    dataset = _FakeDataset(
        item=item,
        objects=[
            {"category": "03001627", "name": "shape_b"},
            {"category": "02691156", "name": "shape_a"},
            {"category": "03001627", "name": "shape_a"},
        ],
    )
    fake_model = _FakeModel()
    fake_encoder = _FakeEncoder()
    attention_inputs: list[torch.Tensor] = []

    def fake_compute_attention_weights(ae: Any, inputs: torch.Tensor) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        assert ae is fake_encoder
        attention_inputs.append(inputs.detach().cpu().clone())
        input_points = inputs.squeeze(0).detach().cpu().numpy().astype(np.float32)
        query_points = input_points[:1]
        attn_mean = np.array([[0.8, 0.2]], dtype=np.float32)
        owners = np.array([0, 0], dtype=np.int32)
        entropy_norm = np.array([0.1, 0.7], dtype=np.float32)
        margin = np.array([0.6, 0.2], dtype=np.float32)
        return (input_points, query_points, attn_mean), owners, entropy_norm, margin

    monkeypatch.setattr(script, "setup_config", lambda cfg: cfg)
    monkeypatch.setattr(script, "get_dataset", lambda cfg, splits: {"test": dataset})
    monkeypatch.setattr(script, "get_model", lambda cfg: fake_model)
    monkeypatch.setattr(script, "patch_attention", lambda model, backend=None: model)
    monkeypatch.setattr(script.lightning, "Fabric", _FakeFabric)
    monkeypatch.setattr(script, "_resolve_shape3d2vecset_encoder", lambda model: fake_encoder)
    monkeypatch.setattr(script, "_compute_attention_weights", fake_compute_attention_weights)
    monkeypatch.setattr(script, "tqdm", lambda iterable, **kwargs: iterable)
    monkeypatch.setattr(script.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(script, "look_at", lambda eye, target: np.eye(4, dtype=np.float32))
    monkeypatch.setattr(script, "inv_trafo", lambda matrix: matrix)
    monkeypatch.setattr(script, "convert_extrinsic", lambda matrix, src, dst: matrix)

    out_dir = tmp_path / "cross_attn"
    script.main.__wrapped__(_base_cfg(out_dir))

    target_dir = out_dir / "02691156" / "shape_a" / "cross_attention"

    assert dataset.accessed == [1]
    assert fake_model.device == torch.device("cpu")
    assert fake_model.requires_grad_flag is False
    assert fake_encoder.eval_called is True
    assert len(attention_inputs) == 1
    assert attention_inputs[0].shape == (1, 2, 3)
    assert (target_dir / "query_ownership_input_points.ply").is_file()
    assert (target_dir / "queries_fps.ply").is_file()
    assert (target_dir / "uncertainty_input_points.ply").is_file()
    assert (target_dir / "query_ownership_input_points_proj.png").is_file()
    assert (target_dir / "uncertainty_input_points_proj.png").is_file()
    weights = np.load(target_dir / "attention_weights.npz")
    assert np.array_equal(weights["owner_per_input"], np.array([0, 0], dtype=np.int32))
