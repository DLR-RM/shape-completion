from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from visualize.scripts import vis_latent_embedding_compare as script


class _FakeFabric:
    def __init__(self, precision: str) -> None:
        self.precision = precision

    def setup_module(self, module: Any) -> Any:
        return module

    def autocast(self) -> Any:
        return nullcontext()


class _FakeDataset:
    def __init__(self, items: list[dict[str, Any]], objects: list[dict[str, str]]) -> None:
        self.items = items
        self.objects = objects
        self.accessed: list[int] = []

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        self.accessed.append(index)
        return self.items[index]


class _FakeModel:
    def __init__(self, offset: float) -> None:
        self.offset = offset
        self.device: torch.device | None = None
        self.requires_grad_flag: bool | None = None
        self.eval_called = False
        self.encode_inputs: list[torch.Tensor] = []

    def to(self, device: torch.device) -> _FakeModel:
        self.device = device
        return self

    def requires_grad_(self, flag: bool) -> _FakeModel:
        self.requires_grad_flag = flag
        return self

    def eval(self) -> _FakeModel:
        self.eval_called = True
        return self

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        self.encode_inputs.append(inputs.detach().cpu().clone())
        base = inputs.mean(dim=1, keepdim=True)
        return torch.cat([base + self.offset, base + self.offset + 1.0], dim=1)


def _base_cfg(output_dir: Path) -> DictConfig:
    return OmegaConf.create(
        {
            "test": {"split": "test", "precision": "32-true"},
            "val": {"precision": "32-true"},
            "model": {"weights": "stub", "checkpoint": None, "load_best": False},
            "vis": {
                "output_dir": str(output_dir),
                "objects": [1, 0],
                "projection": "auto",
                "save_csv": True,
                "seed": 7,
                "vae_arch": "vae_demo",
                "vqvae_arch": "vqvae_demo",
                "vae_weights": "vae.pt",
                "vqvae_weights": "vqvae.pt",
            },
        }
    )


def test_resolve_indices_filters_by_category() -> None:
    items = [
        {"category.id": "03001627", "inputs": np.zeros((2, 3), dtype=np.float32)},
        {"category.id": "02691156", "inputs": np.zeros((2, 3), dtype=np.float32)},
        {"category.id": "03001627", "inputs": np.zeros((2, 3), dtype=np.float32)},
    ]
    dataset = _FakeDataset(
        items=items,
        objects=[
            {"category": "03001627", "name": "chair_b"},
            {"category": "02691156", "name": "plane_a"},
            {"category": "03001627", "name": "chair_a"},
        ],
    )
    vis_cfg = OmegaConf.create({"categories": ["03001627"], "max_objects": 3})

    indices = script._resolve_indices(dataset, vis_cfg)

    assert indices == [2, 0]


def test_project_2d_small_inputs_and_tsne_path(monkeypatch: Any) -> None:
    single, method_single = script._project_2d(
        np.array([[1.0, 2.0]], dtype=np.float32),
        method="auto",
        seed=0,
        umap_neighbors=15,
        umap_min_dist=0.1,
        tsne_perplexity=30.0,
    )
    pair, method_pair = script._project_2d(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        method="auto",
        seed=0,
        umap_neighbors=15,
        umap_min_dist=0.1,
        tsne_perplexity=30.0,
    )

    class _FakeTSNE:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def fit_transform(self, features: np.ndarray) -> np.ndarray:
            return np.stack([features[:, 0] + 10.0, features[:, 1] + 20.0], axis=1)

    monkeypatch.setattr(script, "TSNE", _FakeTSNE)
    tsne_xy, method_tsne = script._project_2d(
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        method="tsne",
        seed=0,
        umap_neighbors=15,
        umap_min_dist=0.1,
        tsne_perplexity=30.0,
    )

    assert method_single == "identity"
    assert single.shape == (1, 2)
    assert method_pair == "identity"
    assert np.array_equal(pair[:, 0], np.array([1.0, 3.0], dtype=np.float32))
    assert method_tsne == "tsne"
    assert np.array_equal(tsne_xy[:, 0], np.array([11.0, 13.0, 15.0], dtype=np.float32))


def test_main_smoke_writes_outputs(monkeypatch: Any, tmp_path: Path) -> None:
    items = [
        {
            "inputs": np.array([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]], dtype=np.float32),
            "category.id": "03001627",
            "inputs.name": "chair_a",
        },
        {
            "inputs": np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], dtype=np.float32),
            "category.id": "02691156",
            "inputs.name": "plane_b",
        },
    ]
    dataset = _FakeDataset(
        items=items,
        objects=[
            {"category": "03001627", "name": "chair_a"},
            {"category": "02691156", "name": "plane_b"},
        ],
    )
    vae_model = _FakeModel(offset=0.5)
    vqvae_model = _FakeModel(offset=2.0)
    created_models: list[tuple[str | None, str | None, _FakeModel]] = []

    def fake_get_model(cfg: DictConfig, arch: str | None = None, weights_path: str | None = None, **kwargs: Any) -> _FakeModel:
        _ = cfg, kwargs
        model = vae_model if arch == "vae_demo" else vqvae_model
        created_models.append((arch, weights_path, model))
        return model

    monkeypatch.setattr(script, "setup_config", lambda cfg: cfg)
    monkeypatch.setattr(script, "suppress_known_optional_dependency_warnings", lambda: None)
    monkeypatch.setattr(script, "get_dataset", lambda cfg, splits: {"test": dataset})
    monkeypatch.setattr(script, "get_model", fake_get_model)
    monkeypatch.setattr(script, "patch_attention", lambda model, backend=None: model)
    monkeypatch.setattr(script.lightning, "Fabric", _FakeFabric)
    monkeypatch.setattr(script, "tqdm", lambda iterable, **kwargs: iterable)
    monkeypatch.setattr(script.torch.cuda, "is_available", lambda: False)

    output_dir = tmp_path / "latent_compare"
    script.main.__wrapped__(_base_cfg(output_dir))

    out_prefix = output_dir / "latent_embedding_compare_identity"

    assert dataset.accessed == [1, 0]
    assert created_models == [
        ("vae_demo", "vae.pt", vae_model),
        ("vqvae_demo", "vqvae.pt", vqvae_model),
    ]
    assert vae_model.device == torch.device("cpu")
    assert vqvae_model.device == torch.device("cpu")
    assert vae_model.requires_grad_flag is False
    assert vqvae_model.requires_grad_flag is False
    assert vae_model.eval_called is True
    assert vqvae_model.eval_called is True
    assert len(vae_model.encode_inputs) == 2
    assert len(vqvae_model.encode_inputs) == 2
    assert out_prefix.with_suffix(".png").is_file()
    assert out_prefix.with_suffix(".csv").is_file()

    data = np.load(out_prefix.with_suffix(".npz"))
    assert np.array_equal(data["sample_indices"], np.array([1, 0], dtype=np.int32))
    assert np.array_equal(data["categories"], np.array(["02691156", "03001627"]))
    assert data["vae_xy"].shape == (2, 2)
    assert data["vqvae_xy"].shape == (2, 2)
    assert data["projection_method_vae"].item() == "identity"
    assert data["projection_method_vqvae"].item() == "identity"

    csv_text = out_prefix.with_suffix(".csv").read_text(encoding="utf-8")
    assert "plane_b" in csv_text
    assert "chair_a" in csv_text
