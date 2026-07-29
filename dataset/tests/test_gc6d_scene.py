from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf
from PIL import Image

from dataset import GC6DSceneEval, get_bop_scene


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_scene(root: Path, scene_id: int, frame_ids: range) -> None:
    scene_dir = root / "scenes" / f"{scene_id:06d}"
    camera: dict[str, object] = {}
    ground_truth: dict[str, object] = {}
    for frame_id in frame_ids:
        frame_name = f"{frame_id:06d}"
        depth_path = scene_dir / "depth" / f"{frame_name}.png"
        depth_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((2, 3), 1000, dtype=np.uint16)).save(depth_path)
        camera[str(frame_id)] = {
            "cam_K": np.eye(3).reshape(-1).tolist(),
            "depth_scale": 1.0,
        }
        ground_truth[str(frame_id)] = []
    _write_json(scene_dir / "scene_camera.json", camera)
    _write_json(scene_dir / "scene_gt.json", ground_truth)


def _write_cross_object_test_fixture(root: Path) -> None:
    _write_json(root / "split_info" / "grasp_test_scene_ids.json", [1, 2])
    _write_scene(root, 1, range(1, 9))
    _write_scene(root, 2, range(1, 9))


def _factory_cfg(root: Path):
    return OmegaConf.create(
        {
            "dirs": {"gc6d": str(root)},
            "data": {
                "gc6d_split": "cross_object_test",
                "gc6d_camera": "azure_kinect",
                "gc6d_scene_ids": [1],
                "split": False,
                "dither": False,
                "bop_generate_points": False,
                "bop_load_mesh": False,
                "bop_load_label": False,
                "bop_background_plane_threshold": None,
            },
            "norm": {
                "center": None,
                "scale": False,
                "scale_factor": 1.0,
                "reference": "inputs",
                "padding": 0.1,
                "bounds": None,
            },
            "inputs": {"crop": False, "num_points": None, "project": False},
            "points": {
                "crop": False,
                "subsample": False,
                "voxelize": None,
                "in_out_ratio": None,
            },
            "test": {"num_query_points": None},
            "load": {"keys_to_keep": []},
            "vis": {"show": False, "save": False, "mesh": False},
            "log": {"verbose": 0},
            "collate_3d": None,
            "single_view": False,
        }
    )


def test_gc6d_scene_eval_uses_official_split_and_camera_cycle(tmp_path: Path) -> None:
    root = tmp_path / "gc6d"
    _write_cross_object_test_fixture(root)

    dataset = GC6DSceneEval(
        root=root,
        split="cross_object_test",
        camera="d435",
        load_mesh=False,
    )

    assert dataset.name == "graspclutter6d"
    assert dataset.split == "cross_object_test"
    assert dataset.camera == "d435"
    assert dataset.mask_dir == "mask_visib"
    assert [sample.ann_id for sample in dataset.samples] == ["000002", "000006", "000002", "000006"]

    item = dataset[0]
    assert item["category.id"] == "graspclutter6d/000001"
    assert item["frame.id"] == 2
    np.testing.assert_allclose(item["inputs"], np.ones((2, 3), dtype=np.float32))


def test_gc6d_scene_eval_filters_requested_scenes(tmp_path: Path) -> None:
    root = tmp_path / "gc6d"
    _write_cross_object_test_fixture(root)

    dataset = GC6DSceneEval(
        root=root,
        split="cross_object_test",
        scene_ids=[2],
        name="gc6d-release",
        load_mesh=False,
    )

    assert dataset.name == "gc6d-release"
    assert dataset.categories == ["000002"]
    assert len(dataset) == 8

    with pytest.raises(ValueError, match="not in GraspClutter6D split"):
        GC6DSceneEval(
            root=root,
            split="cross_object_test",
            scene_ids=[3],
            load_mesh=False,
        )


def test_gc6d_scene_eval_reports_missing_split_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"grasp_test_scene_ids\.json"):
        GC6DSceneEval(
            root=tmp_path,
            split="cross_object_test",
            load_mesh=False,
        )


def test_gc6d_scene_factory_builds_native_loader(tmp_path: Path) -> None:
    root = tmp_path / "gc6d"
    _write_cross_object_test_fixture(root)

    dataset = get_bop_scene(_factory_cfg(root), "gc6d", split="test")

    assert isinstance(dataset, GC6DSceneEval)
    assert dataset.gc6d_split == "cross_object_test"
    assert dataset.gc6d_camera == "azure_kinect"
    assert dataset.categories == ["000001"]
    assert [sample.ann_id for sample in dataset.samples] == ["000003", "000007"]
