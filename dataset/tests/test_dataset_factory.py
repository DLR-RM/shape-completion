from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from dataset import _keys_to_keep_for_load, get_bop_scene, get_dataset, get_graspnet, get_tabletop


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


def test_get_bop_scene_forwards_ann_id_mod_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    class FakeBOPSceneEval:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)

    monkeypatch.setattr("dataset.BOPSceneEval", FakeBOPSceneEval)

    cfg = OmegaConf.create(
        {
            "vis": {"show": False, "save": False, "mesh": False},
            "log": {"verbose": 0},
            "dirs": {
                "bop": "/tmp/bop",
                "bop_scene_models": "/tmp/bop/GraspClutter6D/models_eval",
            },
            "data": {
                "bop_scene_split": "scenes",
                "bop_scene_ids": [2, 3],
                "bop_ann_id_mod": 4,
                "bop_ann_id_remainder": 3,
                "bop_target_filename": None,
                "frame": "cam",
                "split": False,
                "dither": False,
            },
            "inputs": {"project": True, "crop": False, "num_points": 100000},
            "norm": {"center": False, "scale": False, "padding": 0.1, "scale_factor": 1.0, "reference": "inputs"},
            "points": {"crop": False, "subsample": True, "voxelize": False, "in_out_ratio": 0.5},
            "test": {"num_query_points": 100000},
            "load": {"keys_to_keep": []},
            "collate_3d": None,
            "stack_2d": False,
        }
    )

    dataset = get_bop_scene(cfg, "bopscene_GraspClutter6D_scenes", split="test")

    assert isinstance(dataset, FakeBOPSceneEval)
    assert calls[0]["ann_id_mod"] == 4
    assert calls[0]["ann_id_remainder"] == 3
    assert calls[0]["target_filename"] is None


def test_get_graspnet_keeps_unsplit_voxelized_mesh_data(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    class FakeGraspNetEval:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)

    monkeypatch.setattr("dataset.GraspNetEval", FakeGraspNetEval)

    cfg = OmegaConf.create(
        {
            "vis": {"show": False, "save": True},
            "log": {"verbose": 0},
            "dirs": {"graspnet": "/tmp/graspnet"},
            "data": {
                "graspnet_split": None,
                "graspnet_camera": "kinect",
                "graspnet_depth_scale": 1000.0,
                "graspnet_scene_ids": None,
                "graspnet_view_ids": [1],
                "graspnet_test_view_ids": [16, 80],
                "graspnet_point_seed": 42,
                "camera": "kinect",
                "split": True,
                "dither": False,
            },
            "inputs": {"type": "rgbd", "project": True, "crop": False, "num_points": None},
            "norm": {
                "bounds": [[-0.5, -0.5, -0.05], [0.5, 0.5, 0.5]],
                "center": False,
                "scale": False,
                "padding": 0.0,
                "scale_factor": 1.0,
                "reference": "inputs",
            },
            "points": {"crop": False, "subsample": False, "voxelize": 128, "in_out_ratio": 0.5},
            "test": {"num_query_points": 100000},
            "load": {"keys_to_keep": ["index", "mesh", "mesh.vertices", "mesh.triangles"]},
            "collate_3d": "stack",
            "stack_2d": True,
        }
    )

    dataset = get_graspnet(cfg, "graspnet", split="test")

    assert isinstance(dataset, FakeGraspNetEval)
    split_data = [t for t in calls[0]["transforms"] if t.__class__.__name__ == "SplitData"]
    assert split_data == []
    keys_to_keep = [t for t in calls[0]["transforms"] if t.__class__.__name__ == "KeysToKeep"]
    assert len(keys_to_keep) == 1
    assert "mesh.vertices" in keys_to_keep[0].keys
    assert "mesh.triangles" in keys_to_keep[0].keys
    assert calls[0]["load_color"] is True
    assert calls[0]["view_ids"] == [16, 80]
    assert calls[0]["point_seed_base"] == 42


def test_get_graspnet_does_not_load_color_for_depth_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    class FakeGraspNetEval:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)

    monkeypatch.setattr("dataset.GraspNetEval", FakeGraspNetEval)

    cfg = OmegaConf.create(
        {
            "vis": {"show": False, "save": False},
            "log": {"verbose": 0},
            "dirs": {"graspnet": "/tmp/graspnet"},
            "data": {
                "graspnet_split": None,
                "graspnet_camera": "kinect",
                "graspnet_depth_scale": 1000.0,
                "graspnet_scene_ids": None,
                "camera": "kinect",
                "split": False,
                "dither": False,
            },
            "inputs": {"type": "kinect", "project": True, "crop": False, "num_points": None},
            "norm": {
                "bounds": [[-0.5, -0.5, -0.05], [0.5, 0.5, 0.5]],
                "center": False,
                "scale": False,
                "padding": 0.0,
                "scale_factor": 1.0,
                "reference": "inputs",
            },
            "points": {"crop": False, "subsample": False, "voxelize": False, "in_out_ratio": 0.5},
            "test": {"num_query_points": 100000},
            "load": {"keys_to_keep": ["index", "inputs"]},
            "collate_3d": "stack",
            "stack_2d": True,
        }
    )

    dataset = get_graspnet(cfg, "graspnet", split="test")

    assert isinstance(dataset, FakeGraspNetEval)
    assert calls[0]["load_color"] is False


def test_get_tabletop_rgbd_can_load_kinect_depth_source(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    class FakeTableTop:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)

    monkeypatch.setattr("dataset.TableTop", FakeTableTop)
    monkeypatch.setattr("dataset.get_tabletop_transforms", lambda **kwargs: None)

    cfg = OmegaConf.create(
        {
            "dirs": {"tabletop": "/tmp/tabletop", "shapenet_v1_fused": "/tmp/shapenet"},
            "inputs": {
                "type": "rgbd",
                "depth_type": "kinect",
                "project": True,
                "min_num_points": 1,
                "max_num_points": None,
                "crop": False,
                "num_points": 100000,
                "normals": False,
            },
            "points": {
                "subsample": True,
                "min_num_points": 1,
                "max_num_points": None,
                "crop": False,
                "voxelize": False,
                "from_pointcloud": False,
                "cache": False,
            },
            "train": {"num_query_points": 10000},
            "val": {"num_query_points": 100000},
            "aug": {"rotate": False, "scale": False, "translate": False, "noise": False},
            "data": {
                "frame": "world",
                "dither": False,
                "split": False,
                "allow_empty_scenes": True,
                "exclude_manifest": "/tmp/excluded.json",
            },
            "norm": {"center": False, "scale": False, "scale_factor": 1.0, "reference": "inputs", "padding": 0.0},
            "implicit": {"near": None, "far": None},
            "load": {"keys_to_keep": ["inputs", "inputs.image", "inputs.colors", "inputs.pixel_coords"], "hdf5": False},
            "files": {"mesh": None, "pointcloud": None, "points": {"train": "samples/uniform_random.npz", "val": None}},
            "load_3d": False,
            "collate_3d": None,
            "filter": False,
            "pointcloud": {"normals": False},
            "patch_size": None,
            "scale": 1.0,
            "stack_2d": False,
            "sample_free": "cube",
        }
    )

    dataset = get_tabletop(cfg, split="train")

    assert isinstance(dataset, FakeTableTop)
    assert calls[0]["load_color"] is True
    assert calls[0]["load_depth"] == "kinect"
    assert calls[0]["allow_empty_scenes"] is True
    assert calls[0]["exclude_manifest"] == Path("/tmp/excluded.json")


def test_get_tabletop_uses_dataset_specific_3d_root(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    class FakeTableTop:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)

    monkeypatch.setattr("dataset.TableTop", FakeTableTop)
    monkeypatch.setattr("dataset.get_tabletop_transforms", lambda **kwargs: None)

    cfg = OmegaConf.create(
        {
            "dirs": {
                "tabletop": "/tmp/tabletop",
                "objaverse_rgbd_v2": "/tmp/objaverse",
                "objaverse_rgbd_v2_3d": "/tmp/objaverse_3d",
                "shapenet_v1_fused": "/tmp/shapenet",
            },
            "inputs": {
                "type": "depth",
                "project": False,
                "min_num_points": 1,
                "max_num_points": None,
                "num_points": None,
                "normals": False,
            },
            "points": {
                "subsample": False,
                "min_num_points": 1,
                "max_num_points": None,
                "crop": False,
                "voxelize": False,
                "from_pointcloud": False,
                "cache": False,
            },
            "train": {"num_query_points": 10000},
            "val": {"num_query_points": 100000},
            "aug": {"rotate": False, "scale": False, "translate": False, "noise": False},
            "data": {"frame": "world", "dither": False, "split": False},
            "norm": {
                "center": False,
                "scale": False,
                "scale_factor": 1.0,
                "reference": "inputs",
                "padding": 0.0,
            },
            "implicit": {"near": None, "far": None},
            "load": {"keys_to_keep": ["inputs", "points"], "hdf5": False},
            "files": {
                "mesh": None,
                "pointcloud": None,
                "points": {"train": "samples/uniform_random.npz", "val": None},
            },
            "load_3d": True,
            "collate_3d": None,
            "filter": False,
            "pointcloud": {"normals": False},
            "patch_size": None,
            "scale": 1.0,
            "stack_2d": False,
            "sample_free": "cube",
        }
    )

    get_tabletop(cfg, split="train", ds="objaverse_rgbd_v2")
    get_tabletop(cfg, split="train", ds="tabletop")

    assert calls[0]["data_dir"] == Path("/tmp/objaverse")
    assert calls[0]["data_dir_3d"] == Path("/tmp/objaverse_3d")
    assert calls[1]["data_dir"] == Path("/tmp/tabletop")
    assert calls[1]["data_dir_3d"] == Path("/tmp/shapenet")


def test_get_dataset_routes_objaverse_rgbd_to_tabletop(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_get_tabletop(cfg: Any, split: str, ds: str = "tabletop") -> list[Any]:
        calls.append((split, ds))
        return []

    cfg = _base_cfg(show=False, save=False, mesh=False)
    cfg.data.train_ds = ["objaverse_rgbd_v2"]
    cfg.dirs.objaverse_rgbd_v2 = "/tmp/objaverse"
    monkeypatch.setattr("dataset.get_tabletop", fake_get_tabletop)
    monkeypatch.setattr("dataset.print_dataset_info", lambda *args, **kwargs: None)

    datasets = get_dataset(cfg, splits=("train",))

    assert datasets["train"] == []
    assert calls == [("train", "objaverse_rgbd_v2")]


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


def test_get_dataset_rejects_missing_requested_split() -> None:
    cfg = _base_cfg(show=False, save=False, mesh=False)
    cfg.data.train_ds = None
    cfg.data.val_ds = None
    cfg.data.test_ds = None

    with pytest.raises(ValueError, match=r"data\.test_ds must be configured"):
        get_dataset(cfg, splits=("test",))
