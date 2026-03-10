from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from visualize.scripts import vis_latent_pca_rgb as script


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


class _FakeDiffusionBase:
    pass


class _FakeModel(_FakeDiffusionBase):
    def __init__(self) -> None:
        self.device: torch.device | None = None
        self.requires_grad_flag: bool | None = None

    def to(self, device: torch.device) -> _FakeModel:
        self.device = device
        return self

    def requires_grad_(self, flag: bool) -> _FakeModel:
        self.requires_grad_flag = flag
        return self


def _base_cfg(output_dir: Path) -> DictConfig:
    return OmegaConf.create(
        {
            "test": {"split": "test", "precision": "32-true"},
            "val": {"precision": "32-true"},
            "model": {"weights": "stub", "checkpoint": None, "load_best": False},
            "vis": {
                "pca_rgb_output_dir": str(output_dir),
                "pca_method": "diffusion",
                "num_steps": 4,
                "pca_rgb_steps": [0, 2],
                "pca_proj_size": 16,
                "pca_proj_radius": 1,
                "pca_proj_cam_location": [0.0, 0.0, 0.0],
                "pca_proj_cam_target": [0.0, 0.0, 1.0],
                "pca_proj_animation_format": "none",
                "pca_strip_height": 8,
                "pca_strip_token_width": 2,
            },
        }
    )


def test_prepare_conditioning_normalize_and_collect_diffusion_steps() -> None:
    inputs = torch.zeros((1, 4, 3), dtype=torch.float32)
    conditioned = script._prepare_conditioning(torch.tensor([1.0, 2.0, 3.0]), inputs)
    matrix_conditioned = script._prepare_conditioning(torch.tensor([[1.0, 2.0, 3.0]]), inputs)
    scalar = script._prepare_conditioning(torch.tensor(1.0), inputs)
    none_case = script._prepare_conditioning(None, inputs)
    normalized = script._normalize_to_uint8(
        np.array(
            [
                [1.0, 5.0, 5.0],
                [1.0, 5.0, 6.0],
                [1.0, 5.0, 7.0],
            ],
            dtype=np.float32,
        )
    )

    class _FakeGenModel:
        @staticmethod
        def generate(**kwargs: Any) -> tuple[None, list[torch.Tensor]]:
            _ = kwargs
            return None, [
                torch.tensor([[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]], dtype=torch.float32),
                torch.tensor([[[2.0, 3.0, 4.0], [4.0, 5.0, 6.0]]], dtype=torch.float32),
                torch.tensor([[[3.0, 4.0, 5.0], [5.0, 6.0, 7.0]]], dtype=torch.float32),
            ]

    step_latents = script._collect_diffusion_step_latents(
        model=_FakeGenModel(),
        inputs=inputs,
        conditioning=None,
        num_steps=4,
        selected_steps=[0, 5, 2],
    )

    assert conditioned is not None and conditioned.shape == (3,)
    assert matrix_conditioned is not None and matrix_conditioned.shape == (1, 1, 3)
    assert scalar is not None and scalar.shape == (1,)
    assert none_case is None
    assert normalized.dtype == np.uint8
    assert normalized.shape == (3, 3)
    assert step_latents.steps == [0, 2]
    assert len(step_latents.latents) == 2


def test_save_mp4_missing_ffmpeg_and_export_projection_animations(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setattr(script.shutil, "which", lambda name: None)
    assert script._save_mp4([], tmp_path / "missing.mp4", fps=12.0) is False

    frames = []
    for idx in range(2):
        path = tmp_path / f"frame_{idx}.png"
        Image.fromarray(np.full((8, 8, 3), idx * 40, dtype=np.uint8), mode="RGB").save(path)
        frames.append(path)

    mp4_calls: list[tuple[Path, float]] = []
    monkeypatch.setattr(script, "_save_mp4", lambda images, path, fps: mp4_calls.append((path, fps)) or True)

    script._export_projection_animations(
        frames,
        tmp_path,
        "demo",
        animation_format="both",
        gif_duration_ms=100,
        gif_loop=1,
        mp4_fps=5.0,
    )

    assert (tmp_path / "demo_pca_rgb_proj.gif").is_file()
    assert mp4_calls == [(tmp_path / "demo_pca_rgb_proj.mp4", 5.0)]


def test_main_diffusion_smoke_writes_outputs(monkeypatch: Any, tmp_path: Path) -> None:
    item = {
        "inputs": np.array(
            [
                [0.0, 0.0, 2.0],
                [0.3, 0.0, 2.0],
                [0.0, 0.3, 2.0],
                [0.3, 0.3, 2.0],
            ],
            dtype=np.float32,
        ),
        "category.id": "03001627",
        "inputs.name": "shape_a",
    }
    dataset = _FakeDataset(item=item, objects=[{"category": "03001627", "name": "shape_a"}])
    fake_model = _FakeModel()
    collect_calls: list[tuple[int, list[int]]] = []

    def fake_collect_diffusion_step_latents(
        model: Any,
        inputs: torch.Tensor,
        conditioning: torch.Tensor | None,
        num_steps: int,
        selected_steps: list[int],
    ) -> script.StepLatents:
        assert model is fake_model
        assert conditioning is None
        collect_calls.append((num_steps, selected_steps))
        _ = inputs
        return script.StepLatents(
            steps=[0, 2],
            latents=[
                np.array(
                    [
                        [1.0, 2.0, 3.0],
                        [2.0, 3.0, 4.0],
                        [3.0, 4.0, 5.0],
                        [4.0, 5.0, 6.0],
                    ],
                    dtype=np.float32,
                ),
                np.array(
                    [
                        [2.0, 1.0, 3.0],
                        [3.0, 2.0, 4.0],
                        [4.0, 3.0, 5.0],
                        [5.0, 4.0, 6.0],
                    ],
                    dtype=np.float32,
                ),
            ],
        )

    monkeypatch.setattr(script, "setup_config", lambda cfg: cfg)
    monkeypatch.setattr(script, "get_dataset", lambda cfg, splits: {"test": dataset})
    monkeypatch.setattr(script, "get_model", lambda cfg: fake_model)
    monkeypatch.setattr(script, "patch_attention", lambda model, backend=None: model)
    monkeypatch.setattr(script.lightning, "Fabric", _FakeFabric)
    monkeypatch.setattr(script, "DiffusionModel", _FakeDiffusionBase)
    monkeypatch.setattr(script, "_collect_diffusion_step_latents", fake_collect_diffusion_step_latents)
    monkeypatch.setattr(
        script,
        "_extract_query_points",
        lambda inputs, n_queries: np.array(
            [
                [0.0, 0.0, 2.0],
                [0.3, 0.0, 2.0],
                [0.0, 0.3, 2.0],
                [0.3, 0.3, 2.0],
            ],
            dtype=np.float32,
        ),
    )
    monkeypatch.setattr(script, "tqdm", lambda iterable, **kwargs: iterable)
    monkeypatch.setattr(script.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(script, "look_at", lambda eye, target: np.eye(4, dtype=np.float32))
    monkeypatch.setattr(script, "inv_trafo", lambda matrix: matrix)
    monkeypatch.setattr(script, "convert_extrinsic", lambda matrix, src, dst: matrix)

    output_dir = tmp_path / "pca_rgb"
    script.main.__wrapped__(_base_cfg(output_dir))

    target_dir = output_dir / "03001627" / "shape_a" / "diffusion_pca_rgb"

    assert dataset.accessed == [0]
    assert fake_model.device == torch.device("cpu")
    assert fake_model.requires_grad_flag is False
    assert collect_calls == [(4, [0, 2])]
    assert (target_dir / "step_000_pca_rgb_strip.png").is_file()
    assert (target_dir / "step_002_pca_rgb_strip.png").is_file()
    assert (target_dir / "step_000_pca_rgb.ply").is_file()
    assert (target_dir / "step_002_pca_rgb.ply").is_file()
    assert (target_dir / "step_000_pca_rgb_proj.png").is_file()
    assert (target_dir / "step_002_pca_rgb_proj.png").is_file()
