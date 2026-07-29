from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from dataset import _keys_to_keep_for_load, get_dataset


def _base_cfg(*, show: bool, save: bool, mesh: bool):
    return OmegaConf.create(
        {
            "vis": {
                "show": show,
                "save": save,
                "mesh": mesh,
                "occupancy": False,
                "points": False,
                "pointcloud": False,
                "voxels": False,
            },
            "data": {
                "train_ds": ["shapenet_v1"],
                "val_ds": ["shapenet_v1"],
                "test_ds": ["shapenet_v1"],
                "sdf_from_occ": False,
            },
            "dirs": {"shapenet_v1": "/tmp"},
            "files": {"points": {"train": False, "val": False, "test": False}},
            "cls": {"num_classes": None, "occupancy": False},
            "seg": {"num_classes": None},
            "model": {"arch": "convonet"},
            "points": {"from_mesh": False, "from_pointcloud": False},
            "pointcloud": {"bbox": False, "normals": False, "from_mesh": False},
            "aug": {"remove_angle": False, "edge_noise": False, "move_sphere": False},
            "inputs": {"type": "depth", "normals": False, "project": False},
            "val": {"mesh": None, "batch_size": 1},
            "norm": {"reference": "inputs", "true_height": False},
            "mesh": {"bbox": False},
        }
    )


@pytest.mark.parametrize(
    ("show", "save", "mesh", "expected_load_mesh"),
    [
        (False, False, True, False),
        (True, False, True, True),
        (False, True, True, True),
        (False, True, False, False),
    ],
)
def test_vis_mesh_save_requests_mesh_loading(
    monkeypatch: pytest.MonkeyPatch,
    show: bool,
    save: bool,
    mesh: bool,
    expected_load_mesh: bool,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_get_shapenet(
        cfg: Any,
        split: str,
        data_dir: Path,
        load_pointcloud: bool | str,
        load_points: bool,
        load_mesh: bool | str,
        load_cam: bool,
        load_normals: bool,
    ) -> list[Any]:
        calls.append(
            {
                "split": split,
                "data_dir": data_dir,
                "load_pointcloud": load_pointcloud,
                "load_points": load_points,
                "load_mesh": load_mesh,
                "load_cam": load_cam,
                "load_normals": load_normals,
            }
        )
        return []

    monkeypatch.setattr("dataset.get_shapenet", fake_get_shapenet)
    monkeypatch.setattr("dataset.print_dataset_info", lambda *args, **kwargs: None)

    get_dataset(_base_cfg(show=show, save=save, mesh=mesh), splits=("test",))

    assert calls
    assert calls[0]["load_mesh"] is expected_load_mesh


def test_save_mesh_extends_keys_to_keep_for_mesh_export() -> None:
    cfg = OmegaConf.create(
        {
            "vis": {"show": False, "save": True, "mesh": True},
            "load": {"keys_to_keep": ["index", "inputs"]},
        }
    )

    keys = _keys_to_keep_for_load(cfg)

    assert keys is not None
    assert "mesh.vertices" in keys
    assert "mesh.triangles" in keys


def test_get_dataset_routes_gc6d_to_scene_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _base_cfg(show=False, save=False, mesh=False)
    cfg.data.test_ds = ["gc6d_cross_object_test"]
    calls: list[tuple[str, str]] = []
    sentinel: list[object] = []

    def fake_get_bop_scene(_cfg: Any, ds: str, split: str) -> list[object]:
        calls.append((ds, split))
        return sentinel

    monkeypatch.setattr("dataset.get_bop_scene", fake_get_bop_scene)
    monkeypatch.setattr("dataset.print_dataset_info", lambda *args, **kwargs: None)

    datasets = get_dataset(cfg, splits=("test",))

    assert calls == [("gc6d_cross_object_test", "test")]
    assert datasets["test"] is sentinel
