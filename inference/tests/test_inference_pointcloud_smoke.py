from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from inference.scripts import inference_pointcloud as script


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._p = nn.Parameter(torch.zeros(1))
        self.name = "ONet"

    def predict(self, inputs: torch.Tensor, query_points: torch.Tensor, points_batch_size: int | None = None) -> torch.Tensor:
        return torch.zeros((1, query_points.size(1)), device=inputs.device, dtype=inputs.dtype)


def _args(tmp_path: Path) -> Namespace:
    return Namespace(
        show=False,
        verbose=0,
        eval=False,
        output=None,
        depth_scale=1000.0,
        depth_trunc=1.1,
        intrinsic=None,
        extrinsic=None,
        pcd_crop=None,
        remove_plane=False,
        distance_threshold=0.006,
        ransac_iterations=1000,
        outlier_neighbors=50,
        outlier_radius=0.1,
        outlier_std=10.0,
        cluster=False,
        crop=True,
        crop_scale=1.0,
        up_axis=1,
        on_plane=False,
        scale=1.0,
        padding=0.1,
        n_points=32,
        resolution=16,
        sdf=False,
        threshold=0.5,
        n_up=0,
        denormalize=False,
        mesh=tmp_path / "gt_mesh.ply",
        pose=None,
    )


def test_run_smoke_minimal(monkeypatch: Any, tmp_path: Path) -> None:
    in_path = tmp_path / "sample.ply"
    in_path.write_text("ply")

    monkeypatch.setattr(
        script,
        "get_point_cloud",
        lambda *_args, **_kwargs: (script.o3d.geometry.PointCloud(), np.eye(3), np.eye(4)),
    )
    monkeypatch.setattr(
        script,
        "get_input_data_from_point_cloud",
        lambda *_args, **_kwargs: (np.zeros((16, 3), dtype=np.float32), np.zeros(3, dtype=np.float32), 1.0),
    )
    monkeypatch.setattr(
        script,
        "make_3d_grid",
        lambda *_args, **_kwargs: torch.zeros((16, 3), dtype=torch.float32),
    )

    script.run(_FakeModel(), in_path, _args(tmp_path))
