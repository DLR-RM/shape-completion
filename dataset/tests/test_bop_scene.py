from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ..src import bop_scene as bop_scene_module
from ..src.bop_scene import BOPSceneEval


class _FakePointCloud:
    def __init__(self, points: np.ndarray) -> None:
        self.points = points


def _write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_bop_scene_eval_loads_full_scene_and_projects_visible_instance_labels(tmp_path, monkeypatch):
    root = tmp_path / "bop"
    scene = root / "tless" / "test_primesense" / "000001"
    (root / "tless" / "models_eval").mkdir(parents=True)
    (root / "tless" / "models_eval" / "obj_000001.ply").write_text("ply\n", encoding="utf-8")
    (root / "tless" / "models_eval" / "obj_000002.ply").write_text("ply\n", encoding="utf-8")

    depth = np.array([[1000, 1000], [1000, 0]], dtype=np.uint16)
    _write_png(scene / "depth" / "000000.png", depth)
    _write_png(scene / "mask_visib" / "000000_000000.png", np.array([[255, 0], [0, 0]], dtype=np.uint8))
    _write_png(scene / "mask_visib" / "000000_000001.png", np.array([[0, 255], [255, 0]], dtype=np.uint8))
    _write_json(
        scene / "scene_camera.json",
        {"0": {"cam_K": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], "depth_scale": 1.0}},
    )
    _write_json(
        scene / "scene_gt.json",
        {
            "0": [
                {"obj_id": 1, "cam_R_m2c": np.eye(3).reshape(-1).tolist(), "cam_t_m2c": [0.0, 0.0, 1000.0]},
                {"obj_id": 1, "cam_R_m2c": np.eye(3).reshape(-1).tolist(), "cam_t_m2c": [100.0, 0.0, 1000.0]},
            ]
        },
    )
    _write_json(
        scene / "scene_gt_info.json",
        {"0": [{"visib_fract": 1.0}, {"visib_fract": 0.5}]},
    )

    fake_points = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float32)
    monkeypatch.setattr(
        bop_scene_module,
        "convert_depth_image_to_point_cloud",
        lambda *_args, **_kwargs: _FakePointCloud(fake_points),
    )

    vertices = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    monkeypatch.setattr(bop_scene_module, "_load_mesh", lambda _path: (vertices, faces))

    dataset = BOPSceneEval(
        root=root,
        name="tless",
        split="test_primesense",
        project=True,
        load_label=True,
        stack_2d=True,
        load_mesh=True,
        mesh_simplify_fraction=None,
    )

    item = dataset[0]

    assert item["category.id"] == "tless/000001"
    assert item["scene.id"] == 1
    assert item["frame.id"] == 0
    assert item["inputs"].shape == (3, 3)
    np.testing.assert_array_equal(item["inputs.labels"], np.array([1, 2, 2], dtype=np.int64))
    np.testing.assert_array_equal(item["inputs.obj_id_order_2d"], np.array([1, 1], dtype=np.int32))
    np.testing.assert_array_equal(item["mesh.obj_id_order_3d"], np.array([1, 1], dtype=np.int32))
    assert item["alignment.status"] == "perfect"
    np.testing.assert_array_equal(item["alignment.gt_index_order_2d"], np.array([0, 1], dtype=np.int64))
    np.testing.assert_array_equal(item["alignment.gt_index_order_3d"], np.array([0, 1], dtype=np.int64))
    assert item["alignment.obj_id_to_2d_indices"] == {1: [0, 1]}
    assert item["alignment.obj_id_to_2d_index"] == {1: 0}
    assert item["mesh.vertices"].shape == (6, 3)
    assert item["mesh.num_vertices"] == [3, 3]
    np.testing.assert_allclose(np.ptp(item["mesh.vertices"], axis=0), np.array([0.11, 0.01, 0.0]), atol=1e-6)
    np.testing.assert_allclose(item["mesh.vertices"][:, 2], np.ones(6), atol=1e-6)


def test_bop_scene_eval_projected_labels_do_not_require_stacked_masks(tmp_path, monkeypatch):
    root = tmp_path / "bop"
    scene = root / "tless" / "test_primesense" / "000001"

    depth = np.array([[1000, 1000], [1000, 0]], dtype=np.uint16)
    _write_png(scene / "depth" / "000000.png", depth)
    _write_png(scene / "mask_visib" / "000000_000000.png", np.array([[255, 0], [0, 0]], dtype=np.uint8))
    _write_png(scene / "mask_visib" / "000000_000001.png", np.array([[0, 255], [255, 0]], dtype=np.uint8))
    _write_json(
        scene / "scene_camera.json",
        {"0": {"cam_K": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], "depth_scale": 1.0}},
    )
    _write_json(
        scene / "scene_gt.json",
        {
            "0": [
                {"obj_id": 1, "cam_R_m2c": np.eye(3).reshape(-1).tolist(), "cam_t_m2c": [0.0, 0.0, 1000.0]},
                {"obj_id": 2, "cam_R_m2c": np.eye(3).reshape(-1).tolist(), "cam_t_m2c": [100.0, 0.0, 1000.0]},
            ]
        },
    )

    fake_points = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float32)
    monkeypatch.setattr(
        bop_scene_module,
        "convert_depth_image_to_point_cloud",
        lambda *_args, **_kwargs: _FakePointCloud(fake_points),
    )

    dataset = BOPSceneEval(
        root=root,
        name="tless",
        split="test_primesense",
        project=True,
        load_label=True,
        stack_2d=False,
        load_mesh=False,
    )

    item = dataset[0]

    assert item["inputs"].shape == (3, 3)
    np.testing.assert_array_equal(item["inputs.labels"], np.array([1, 2, 2], dtype=np.int64))


def test_bop_scene_eval_can_load_alternate_depth_directory(tmp_path):
    root = tmp_path / "bop"
    scene = root / "hb" / "val_primesense" / "000001"
    _write_png(scene / "depth" / "000000.png", np.array([[1000]], dtype=np.uint16))
    _write_png(scene / "depth_da3" / "000000.png", np.array([[1500]], dtype=np.uint16))
    _write_json(
        scene / "scene_camera.json",
        {"0": {"cam_K": np.eye(3).reshape(-1).tolist(), "depth_scale": 1.0}},
    )
    _write_json(scene / "scene_gt.json", {"0": []})

    dataset = BOPSceneEval(root=root, name="hb", split="val_primesense", depth_dir="depth_da3", load_mesh=False)

    item = dataset[0]

    assert item["inputs.path"] == scene / "depth_da3" / "000000.png"
    assert item["inputs.name"] == "000000.png"
    np.testing.assert_allclose(item["inputs"], np.array([[1.5]], dtype=np.float32))


def test_bop_scene_eval_applies_depth_scale_and_pose_millimeters(tmp_path, monkeypatch):
    root = tmp_path / "bop"
    scene = root / "ycbv" / "test" / "000001"
    (root / "ycbv" / "models_eval").mkdir(parents=True)
    (root / "ycbv" / "models_eval" / "obj_000001.ply").write_text("ply\n", encoding="utf-8")
    _write_png(scene / "depth" / "000000.png", np.array([[15000]], dtype=np.uint16))
    _write_json(
        scene / "scene_camera.json",
        {"0": {"cam_K": np.eye(3).reshape(-1).tolist(), "depth_scale": 0.1}},
    )
    _write_json(
        scene / "scene_gt.json",
        {"0": [{"obj_id": 1, "cam_R_m2c": np.eye(3).reshape(-1).tolist(), "cam_t_m2c": [0.0, 0.0, 1000.0]}]},
    )
    vertices = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    monkeypatch.setattr(bop_scene_module, "_load_mesh", lambda _path: (vertices, faces))

    dataset = BOPSceneEval(root=root, name="ycbv", split="test", load_mesh=True, mesh_simplify_fraction=None)

    item = dataset[0]

    np.testing.assert_allclose(item["inputs"], np.array([[1.5]], dtype=np.float32))
    np.testing.assert_allclose(item["mesh.obj_poses_cam"][0, :3, 3], np.array([0.0, 0.0, 1.0], dtype=np.float32))


def test_bop_scene_eval_all_missing_masks_keeps_empty_mask_shape_and_counts(tmp_path):
    root = tmp_path / "bop"
    scene = root / "tless" / "test_primesense" / "000001"
    _write_png(scene / "depth" / "000000.png", np.ones((2, 3), dtype=np.uint16))
    _write_json(
        scene / "scene_camera.json",
        {"0": {"cam_K": np.eye(3).reshape(-1).tolist(), "depth_scale": 1.0}},
    )
    _write_json(
        scene / "scene_gt.json",
        {
            "0": [
                {"obj_id": 1, "cam_R_m2c": np.eye(3).reshape(-1).tolist(), "cam_t_m2c": [0.0, 0.0, 1000.0]},
                {"obj_id": 2, "cam_R_m2c": np.eye(3).reshape(-1).tolist(), "cam_t_m2c": [0.0, 0.0, 1000.0]},
            ]
        },
    )

    dataset = BOPSceneEval(root=root, name="tless", split="test_primesense", load_label=True, load_mesh=False)

    item = dataset[0]

    assert item["inputs.masks"].shape == (0, 2, 3)
    assert item["inputs.missing_mask_count"] == 2
    assert item["inputs.empty_mask_count"] == 0
    np.testing.assert_array_equal(item["scene.object_id_list"], np.zeros((0,), dtype=np.int32))


def test_bop_scene_eval_reports_json_parse_path_context(tmp_path):
    root = tmp_path / "bop"
    scene = root / "hb" / "val_primesense" / "000001"
    _write_png(scene / "depth" / "000000.png", np.ones((1, 1), dtype=np.uint16))
    (scene / "scene_camera.json").write_text("{bad json", encoding="utf-8")
    _write_json(scene / "scene_gt.json", {"0": []})

    dataset = BOPSceneEval(root=root, name="hb", split="val_primesense", load_mesh=False)

    with pytest.raises(ValueError, match=r"scene_camera\.json"):
        dataset[0]


def test_bop_sample_requires_zero_padded_ann_id():
    with pytest.raises(ValueError, match="6-digit"):
        bop_scene_module._BOPSample(scene_dir=Path("."), ann_id="1")


def test_bop_scene_eval_background_plane_filter_keeps_plane_and_labeled_objects():
    xs, zs = np.meshgrid(np.linspace(-0.1, 0.1, 5), np.linspace(0.8, 1.0, 5))
    plane_points = np.column_stack([xs.ravel(), np.zeros(xs.size), zs.ravel()]).astype(np.float32)
    off_plane_background = np.array([[0.0, 0.2, 0.9], [0.1, -0.15, 0.85]], dtype=np.float32)
    labeled_object = np.array([[0.0, 0.12, 0.9], [0.02, 0.16, 0.92]], dtype=np.float32)
    item = {
        "inputs": np.concatenate([plane_points, off_plane_background, labeled_object], axis=0),
        "inputs.labels": np.concatenate(
            [
                np.zeros(len(plane_points) + len(off_plane_background), dtype=np.int64),
                np.ones(len(labeled_object), dtype=np.int64),
            ]
        ),
    }
    dataset = BOPSceneEval.__new__(BOPSceneEval)
    dataset.filter_background = None
    dataset.background_plane_threshold = 0.02
    dataset.background_plane_iterations = 64
    dataset.background_plane_max_fit_points = 100
    dataset.background_plane_min_inliers = 10
    dataset.background_plane_show = False

    dataset._filter_background_points(item)

    assert "inputs.background_plane" in item
    assert item["inputs.background_plane_status"] == "ok"
    assert item["inputs.background_plane_num_background"] == len(plane_points) + len(off_plane_background)
    assert item["inputs.background_plane_num_background_kept"] == len(plane_points)
    assert item["inputs.background_plane_inlier_ratio"] == pytest.approx(
        len(plane_points) / (len(plane_points) + len(off_plane_background))
    )
    assert len(item["inputs"]) == len(plane_points) + len(labeled_object)
    assert int((item["inputs.labels"] == 1).sum()) == len(labeled_object)
    np.testing.assert_allclose(np.abs(item["inputs.background_plane"][1]), 1.0, atol=1e-5)


def test_bop_scene_eval_filter_background_z_threshold_removes_far_background_points():
    item = {
        "inputs": np.asarray(
            [[0.0, 0.0, 0.5], [0.0, 0.0, -0.6], [0.0, 0.0, 0.7], [0.0, 0.0, 1.2]],
            dtype=np.float32,
        ),
        "inputs.labels": np.asarray([0, 0, 1, 0], dtype=np.int64),
    }
    dataset = BOPSceneEval.__new__(BOPSceneEval)
    dataset.filter_background = 0.55
    dataset.background_plane_threshold = None

    dataset._filter_background_points(item)

    np.testing.assert_array_equal(item["inputs.labels"], np.asarray([0, 1], dtype=np.int64))
    np.testing.assert_allclose(item["inputs"][:, 2], np.asarray([0.5, 0.7], dtype=np.float32))


def test_bop_scene_eval_background_plane_filter_can_show_debug_view(monkeypatch):
    xs, zs = np.meshgrid(np.linspace(-0.1, 0.1, 5), np.linspace(0.8, 1.0, 5))
    plane_points = np.column_stack([xs.ravel(), np.zeros(xs.size), zs.ravel()]).astype(np.float32)
    off_plane_background = np.array([[0.0, 0.2, 0.9], [0.1, -0.15, 0.85]], dtype=np.float32)
    labeled_object = np.array([[0.0, 0.12, 0.9]], dtype=np.float32)
    item = {
        "inputs": np.concatenate([plane_points, off_plane_background, labeled_object], axis=0),
        "inputs.labels": np.concatenate(
            [
                np.zeros(len(plane_points) + len(off_plane_background), dtype=np.int64),
                np.ones(len(labeled_object), dtype=np.int64),
            ]
        ),
    }
    dataset = BOPSceneEval.__new__(BOPSceneEval)
    dataset.filter_background = None
    dataset.background_plane_threshold = 0.02
    dataset.background_plane_iterations = 64
    dataset.background_plane_max_fit_points = 100
    dataset.background_plane_min_inliers = 10
    dataset.background_plane_show = True
    calls = []

    def _fake_show(points, labels, keep_mask):
        calls.append((points.copy(), labels.copy(), keep_mask.copy()))

    monkeypatch.setattr(dataset, "_show_background_plane_filter", _fake_show)

    dataset._filter_background_points(item)

    assert len(calls) == 1
    points, labels, keep_mask = calls[0]
    assert points.shape == (len(plane_points) + len(off_plane_background) + len(labeled_object), 3)
    assert labels.shape == (len(points),)
    assert keep_mask.shape == (len(points),)
    assert keep_mask[labels == 1].all()
    assert not keep_mask[labels == 0][-len(off_plane_background) :].any()


def test_bop_scene_eval_background_plane_filter_reports_failure():
    item = {
        "inputs": np.asarray([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.2, 0.0, 1.0]], dtype=np.float32),
        "inputs.labels": np.zeros(3, dtype=np.int64),
    }
    dataset = BOPSceneEval.__new__(BOPSceneEval)
    dataset.filter_background = None
    dataset.background_plane_threshold = 0.02
    dataset.background_plane_iterations = 64
    dataset.background_plane_max_fit_points = 100
    dataset.background_plane_min_inliers = 10
    dataset.background_plane_show = False

    dataset._filter_background_points(item)

    assert "inputs.background_plane" not in item
    assert item["inputs.background_plane_status"] == "not_enough_background"
    assert item["inputs.background_plane_num_background"] == 3


def test_bop_scene_eval_point_filter_syncs_stacked_2d_labels():
    item = {
        "inputs": np.asarray([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.2, 0.0, 1.0]], dtype=np.float32),
        "inputs.labels": np.asarray([0, 1, 2], dtype=np.int64),
        "inputs.pixel_coords": np.asarray([[0, 0], [0, 1], [1, 0]], dtype=np.int32),
        "label": np.asarray([[0, 1], [2, 0]], dtype=np.int64),
        "inputs.masks": np.asarray([[0, 1], [2, 0]], dtype=np.int64),
    }
    dataset = BOPSceneEval.__new__(BOPSceneEval)
    dataset.stack_2d = True

    dataset._apply_input_point_keep_mask(item, np.asarray([True, False, True]))

    assert len(item["inputs"]) == 2
    np.testing.assert_array_equal(item["inputs.labels"], np.asarray([0, 2], dtype=np.int64))
    np.testing.assert_array_equal(item["inputs.pixel_coords"], np.asarray([[0, 0], [1, 0]], dtype=np.int32))
    np.testing.assert_array_equal(item["label"], np.asarray([[0, 0], [2, 0]], dtype=np.int64))
    np.testing.assert_array_equal(item["inputs.masks"], np.asarray([[0, 0], [2, 0]], dtype=np.int64))


def test_bop_scene_eval_point_filter_syncs_instance_2d_masks():
    item = {
        "inputs": np.asarray([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.2, 0.0, 1.0]], dtype=np.float32),
        "inputs.labels": np.asarray([0, 1, 2], dtype=np.int64),
        "inputs.pixel_coords": np.asarray([[0, 0], [0, 1], [1, 0]], dtype=np.int32),
        "label": np.asarray([[0, 1], [2, 0]], dtype=np.int64),
        "inputs.masks": np.asarray(
            [
                [[0, 1], [0, 0]],
                [[0, 0], [1, 0]],
            ],
            dtype=np.uint8,
        ),
    }
    dataset = BOPSceneEval.__new__(BOPSceneEval)
    dataset.stack_2d = False

    dataset._apply_input_point_keep_mask(item, np.asarray([True, False, True]))

    np.testing.assert_array_equal(item["label"], np.asarray([[0, 0], [2, 0]], dtype=np.int64))
    np.testing.assert_array_equal(
        item["inputs.masks"],
        np.asarray(
            [
                [[0, 0], [0, 0]],
                [[0, 0], [1, 0]],
            ],
            dtype=np.uint8,
        ),
    )


def test_bop_scene_eval_statistical_outlier_filter_runs_per_label(monkeypatch):
    item = {
        "inputs": np.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.01, 0.0, 1.0],
                [2.0, 0.0, 1.0],
                [0.2, 0.0, 1.0],
                [0.21, 0.0, 1.0],
                [3.0, 0.0, 1.0],
                [0.4, 0.0, 1.0],
                [4.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "inputs.labels": np.asarray([0, 0, 0, 1, 1, 1, 2, 2], dtype=np.int64),
    }
    dataset = BOPSceneEval.__new__(BOPSceneEval)
    dataset.statistical_outlier_removal = True
    dataset.statistical_outlier_neighbors = 2
    dataset.statistical_outlier_std_ratio = 1.0
    dataset.statistical_outlier_min_points = 3
    dataset.statistical_outlier_show = False
    calls = []

    def _fake_inliers(points, *, nb_neighbors, std_ratio):
        calls.append((points.copy(), nb_neighbors, std_ratio))
        return np.asarray([0, 1], dtype=np.int64)

    monkeypatch.setattr(dataset, "_statistical_inlier_indices", _fake_inliers)

    dataset._remove_statistical_outlier_points(item)

    assert len(calls) == 2
    assert [len(points) for points, _, _ in calls] == [3, 3]
    np.testing.assert_array_equal(item["inputs.labels"], np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64))
    assert item["inputs.outlier_removal_num_removed"] == 2
    assert item["inputs.outlier_removal_num_removed_background"] == 1
    assert item["inputs.outlier_removal_num_removed_objects"] == 1


def test_bop_scene_eval_statistical_outlier_filter_maps_label_local_indices(monkeypatch):
    item = {
        "inputs": np.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.01, 0.0, 1.0],
                [0.02, 0.0, 1.0],
                [0.2, 0.0, 1.0],
                [0.21, 0.0, 1.0],
                [0.22, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "inputs.labels": np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64),
    }
    dataset = BOPSceneEval.__new__(BOPSceneEval)
    dataset.statistical_outlier_removal = True
    dataset.statistical_outlier_neighbors = 2
    dataset.statistical_outlier_std_ratio = 1.0
    dataset.statistical_outlier_min_points = 3
    dataset.statistical_outlier_show = False
    monkeypatch.setattr(dataset, "_statistical_inlier_indices", lambda *_args, **_kwargs: np.asarray([1, 2]))

    dataset._remove_statistical_outlier_points(item)

    np.testing.assert_allclose(
        item["inputs"],
        np.asarray(
            [[0.01, 0.0, 1.0], [0.02, 0.0, 1.0], [0.21, 0.0, 1.0], [0.22, 0.0, 1.0]],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(item["inputs.labels"], np.asarray([0, 0, 1, 1], dtype=np.int64))


def test_bop_scene_eval_statistical_outlier_filter_can_show_debug_view(monkeypatch):
    item = {
        "inputs": np.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.01, 0.0, 1.0],
                [2.0, 0.0, 1.0],
                [0.2, 0.0, 1.0],
                [0.21, 0.0, 1.0],
                [3.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "inputs.labels": np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64),
    }
    dataset = BOPSceneEval.__new__(BOPSceneEval)
    dataset.statistical_outlier_removal = True
    dataset.statistical_outlier_neighbors = 2
    dataset.statistical_outlier_std_ratio = 1.0
    dataset.statistical_outlier_min_points = 3
    dataset.statistical_outlier_show = True
    calls = []

    def _fake_inliers(points, *, nb_neighbors, std_ratio):
        return np.asarray([0, 1], dtype=np.int64)

    def _fake_show(points, labels, keep_mask):
        calls.append((points.copy(), labels.copy(), keep_mask.copy()))

    monkeypatch.setattr(dataset, "_statistical_inlier_indices", _fake_inliers)
    monkeypatch.setattr(dataset, "_show_statistical_outlier_filter", _fake_show)

    dataset._remove_statistical_outlier_points(item)

    assert len(calls) == 1
    points, labels, keep_mask = calls[0]
    assert points.shape == (6, 3)
    np.testing.assert_array_equal(labels, np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64))
    np.testing.assert_array_equal(keep_mask, np.asarray([True, True, False, True, True, False]))


def test_bop_scene_eval_statistical_outlier_debug_view_runs_when_no_points_are_removed(monkeypatch):
    item = {
        "inputs": np.asarray([[0.0, 0.0, 1.0], [0.01, 0.0, 1.0], [0.02, 0.0, 1.0]], dtype=np.float32),
        "inputs.labels": np.asarray([0, 0, 0], dtype=np.int64),
    }
    dataset = BOPSceneEval.__new__(BOPSceneEval)
    dataset.statistical_outlier_removal = True
    dataset.statistical_outlier_neighbors = 2
    dataset.statistical_outlier_std_ratio = 1.0
    dataset.statistical_outlier_min_points = 3
    dataset.statistical_outlier_show = True
    calls = []

    monkeypatch.setattr(dataset, "_statistical_inlier_indices", lambda *_args, **_kwargs: np.asarray([0, 1, 2]))
    monkeypatch.setattr(dataset, "_show_statistical_outlier_filter", lambda *args: calls.append(args))

    dataset._remove_statistical_outlier_points(item)

    assert len(calls) == 1
    assert item["inputs.outlier_removal_num_removed"] == 0
    assert len(item["inputs"]) == 3


def test_bop_scene_eval_statistical_outlier_filter_is_disabled_by_default(monkeypatch):
    item = {
        "inputs": np.asarray([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0], [0.2, 0.0, 1.0]], dtype=np.float32),
        "inputs.labels": np.asarray([0, 0, 1], dtype=np.int64),
    }
    dataset = BOPSceneEval.__new__(BOPSceneEval)
    dataset.statistical_outlier_removal = False

    def _fail_inliers(*_args, **_kwargs):
        raise AssertionError("outlier removal should not run")

    monkeypatch.setattr(dataset, "_statistical_inlier_indices", _fail_inliers)

    dataset._remove_statistical_outlier_points(item)

    assert "inputs.outlier_removal_num_removed" not in item
    assert len(item["inputs"]) == 3


def test_bop_scene_eval_dbscan_filter_keeps_non_noise_object_components_by_default(monkeypatch):
    item = {
        "inputs": np.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.01, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.2, 0.0, 1.0],
                [0.21, 0.0, 1.0],
                [0.22, 0.0, 1.0],
                [2.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "inputs.labels": np.asarray([0, 0, 0, 1, 1, 1, 1], dtype=np.int64),
    }
    dataset = BOPSceneEval.__new__(BOPSceneEval)
    dataset.dbscan_filter = True
    dataset.dbscan_eps = 0.02
    dataset.dbscan_min_points = 2
    dataset.dbscan_keep = "non_noise"
    dataset.dbscan_include_background = False
    dataset.dbscan_show = False
    calls = []

    def _fake_clusters(points, *, eps, min_points):
        calls.append((points.copy(), eps, min_points))
        return np.asarray([0, 0, 1, -1], dtype=np.int64)

    monkeypatch.setattr(dataset, "_dbscan_cluster_ids", _fake_clusters)

    dataset._filter_dbscan_components(item)

    assert len(calls) == 1
    assert len(item["inputs"]) == 6
    np.testing.assert_array_equal(item["inputs.labels"], np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64))
    assert item["inputs.dbscan_num_removed"] == 1
    assert item["inputs.dbscan_num_removed_background"] == 0
    assert item["inputs.dbscan_num_removed_objects"] == 1
    assert item["inputs.dbscan_num_clustered_labels"] == 1


def test_bop_scene_eval_dbscan_filter_can_keep_largest_object_component(monkeypatch):
    item = {
        "inputs": np.asarray(
            [
                [0.2, 0.0, 1.0],
                [0.21, 0.0, 1.0],
                [0.22, 0.0, 1.0],
                [2.0, 0.0, 1.0],
                [2.01, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "inputs.labels": np.asarray([1, 1, 1, 1, 1], dtype=np.int64),
    }
    dataset = BOPSceneEval.__new__(BOPSceneEval)
    dataset.dbscan_filter = True
    dataset.dbscan_eps = 0.02
    dataset.dbscan_min_points = 2
    dataset.dbscan_keep = "largest"
    dataset.dbscan_include_background = False
    dataset.dbscan_show = False

    monkeypatch.setattr(dataset, "_dbscan_cluster_ids", lambda *_args, **_kwargs: np.asarray([0, 0, 0, 1, 1]))

    dataset._filter_dbscan_components(item)

    assert len(item["inputs"]) == 3
    np.testing.assert_array_equal(item["inputs.labels"], np.asarray([1, 1, 1], dtype=np.int64))
    assert item["inputs.dbscan_num_removed"] == 2
    assert item["inputs.dbscan_num_removed_objects"] == 2


def test_bop_scene_eval_dbscan_filter_can_include_background(monkeypatch):
    item = {
        "inputs": np.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.01, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.2, 0.0, 1.0],
                [0.21, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "inputs.labels": np.asarray([0, 0, 0, 1, 1], dtype=np.int64),
    }
    dataset = BOPSceneEval.__new__(BOPSceneEval)
    dataset.dbscan_filter = True
    dataset.dbscan_eps = 0.02
    dataset.dbscan_min_points = 2
    dataset.dbscan_keep = "non_noise"
    dataset.dbscan_include_background = True
    dataset.dbscan_show = False

    def _fake_clusters(points, *, eps, min_points):
        if len(points) == 3:
            return np.asarray([0, 0, -1], dtype=np.int64)
        return np.asarray([0, 0], dtype=np.int64)

    monkeypatch.setattr(dataset, "_dbscan_cluster_ids", _fake_clusters)

    dataset._filter_dbscan_components(item)

    np.testing.assert_array_equal(item["inputs.labels"], np.asarray([0, 0, 1, 1], dtype=np.int64))
    assert item["inputs.dbscan_num_removed"] == 1
    assert item["inputs.dbscan_num_removed_background"] == 1
    assert item["inputs.dbscan_num_removed_objects"] == 0
    assert item["inputs.dbscan_num_clustered_labels"] == 2


def test_bop_scene_eval_dbscan_filter_can_show_debug_view(monkeypatch):
    item = {
        "inputs": np.asarray([[0.0, 0.0, 1.0], [0.01, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32),
        "inputs.labels": np.asarray([1, 1, 1], dtype=np.int64),
    }
    dataset = BOPSceneEval.__new__(BOPSceneEval)
    dataset.dbscan_filter = True
    dataset.dbscan_eps = 0.02
    dataset.dbscan_min_points = 2
    dataset.dbscan_keep = "non_noise"
    dataset.dbscan_include_background = False
    dataset.dbscan_show = True
    calls = []

    monkeypatch.setattr(dataset, "_dbscan_cluster_ids", lambda *_args, **_kwargs: np.asarray([0, 0, -1]))
    monkeypatch.setattr(dataset, "_show_dbscan_filter", lambda *args: calls.append(args))

    dataset._filter_dbscan_components(item)

    assert len(calls) == 1
    points, labels, keep_mask, cluster_ids = calls[0]
    assert points.shape == (3, 3)
    np.testing.assert_array_equal(labels, np.asarray([1, 1, 1], dtype=np.int64))
    np.testing.assert_array_equal(keep_mask, np.asarray([True, True, False]))
    np.testing.assert_array_equal(cluster_ids, np.asarray([0, 0, -1], dtype=np.int64))


def test_bop_scene_eval_dbscan_removal_syncs_2d_masks(monkeypatch):
    item = {
        "inputs": np.asarray([[0.0, 0.0, 1.0], [0.01, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32),
        "inputs.labels": np.asarray([1, 1, 1], dtype=np.int64),
        "inputs.pixel_coords": np.asarray([[0, 0], [0, 1], [1, 0]], dtype=np.int32),
        "label": np.asarray([[1, 1], [1, 0]], dtype=np.int64),
        "inputs.masks": np.asarray([[1, 1], [1, 0]], dtype=np.int64),
    }
    dataset = BOPSceneEval.__new__(BOPSceneEval)
    dataset.stack_2d = True
    dataset.dbscan_filter = True
    dataset.dbscan_eps = 0.02
    dataset.dbscan_min_points = 2
    dataset.dbscan_keep = "non_noise"
    dataset.dbscan_include_background = False
    dataset.dbscan_show = False

    monkeypatch.setattr(dataset, "_dbscan_cluster_ids", lambda *_args, **_kwargs: np.asarray([0, 0, -1]))

    dataset._filter_dbscan_components(item)

    assert len(item["inputs"]) == 2
    np.testing.assert_array_equal(item["inputs.labels"], np.asarray([1, 1], dtype=np.int64))
    np.testing.assert_array_equal(item["label"], np.asarray([[1, 1], [0, 0]], dtype=np.int64))
    np.testing.assert_array_equal(item["inputs.masks"], np.asarray([[1, 1], [0, 0]], dtype=np.int64))
    assert item["inputs.dbscan_num_removed_objects"] == 1


def test_bop_scene_eval_uses_deterministic_view_when_sampling_one_per_scene(tmp_path):
    root = tmp_path / "bop"
    scene = root / "tless" / "test_primesense" / "000001"
    _write_png(scene / "depth" / "000000.png", np.ones((1, 1), dtype=np.uint16))
    _write_png(scene / "depth" / "000002.png", np.ones((1, 1), dtype=np.uint16))
    _write_png(scene / "depth" / "000004.png", np.ones((1, 1), dtype=np.uint16))
    _write_json(
        scene / "scene_camera.json",
        {
            "0": {"cam_K": np.eye(3).reshape(-1).tolist(), "depth_scale": 1.0},
            "2": {"cam_K": np.eye(3).reshape(-1).tolist(), "depth_scale": 1.0},
            "4": {"cam_K": np.eye(3).reshape(-1).tolist(), "depth_scale": 1.0},
        },
    )
    _write_json(scene / "scene_gt.json", {"0": [], "2": [], "4": []})

    first = BOPSceneEval(root=root, name="tless", split="test_primesense", one_view_per_scene=True, load_mesh=False)
    second = BOPSceneEval(root=root, name="tless", split="test_primesense", one_view_per_scene=True, load_mesh=False)

    assert [sample.ann_id for sample in first.samples] == ["000002"]
    assert [sample.ann_id for sample in second.samples] == ["000002"]


def test_bop_scene_eval_uses_bop_test_targets_when_available(tmp_path):
    root = tmp_path / "bop"
    scene = root / "tless" / "test_primesense" / "000001"
    for frame_id in ["000000", "000002", "000004"]:
        _write_png(scene / "depth" / f"{frame_id}.png", np.ones((1, 1), dtype=np.uint16))
    _write_json(
        root / "tless" / "test_targets_bop19.json",
        [
            {"scene_id": 1, "im_id": 0, "obj_id": 1},
            {"scene_id": 1, "im_id": 4, "obj_id": 1},
            {"scene_id": 1, "im_id": 4, "obj_id": 2},
        ],
    )
    _write_json(
        scene / "scene_camera.json",
        {
            "0": {"cam_K": np.eye(3).reshape(-1).tolist(), "depth_scale": 1.0},
            "2": {"cam_K": np.eye(3).reshape(-1).tolist(), "depth_scale": 1.0},
            "4": {"cam_K": np.eye(3).reshape(-1).tolist(), "depth_scale": 1.0},
        },
    )
    _write_json(scene / "scene_gt.json", {"0": [], "2": [], "4": []})

    dataset = BOPSceneEval(root=root, name="tless", split="test_primesense", load_mesh=False)

    assert [sample.ann_id for sample in dataset.samples] == ["000000", "000004"]


def test_bop_scene_eval_does_not_apply_test_targets_to_val_splits(tmp_path):
    root = tmp_path / "bop"
    scene = root / "hb" / "val_primesense" / "000001"
    for frame_id in ["000000", "000002", "000004"]:
        _write_png(scene / "depth" / f"{frame_id}.png", np.ones((1, 1), dtype=np.uint16))
    _write_json(
        root / "hb" / "test_targets_bop19.json",
        [
            {"scene_id": 1, "im_id": 4, "obj_id": 1},
        ],
    )
    _write_json(
        scene / "scene_camera.json",
        {
            "0": {"cam_K": np.eye(3).reshape(-1).tolist(), "depth_scale": 1.0},
            "2": {"cam_K": np.eye(3).reshape(-1).tolist(), "depth_scale": 1.0},
            "4": {"cam_K": np.eye(3).reshape(-1).tolist(), "depth_scale": 1.0},
        },
    )
    _write_json(scene / "scene_gt.json", {"0": [], "2": [], "4": []})

    dataset = BOPSceneEval(root=root, name="hb", split="val_primesense", load_mesh=False)

    assert [sample.ann_id for sample in dataset.samples] == ["000000", "000002", "000004"]


def test_bop_scene_eval_point_filter_cascade_keeps_3d_and_2d_labels_consistent(monkeypatch):
    item = {
        "inputs": np.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.01, 0.0, 1.0],
                [0.02, 0.0, 1.0],
                [0.03, 0.2, 1.0],
                [0.20, 0.2, 1.0],
                [0.21, 0.2, 1.0],
                [0.22, 0.2, 1.0],
            ],
            dtype=np.float32,
        ),
        "inputs.labels": np.asarray([0, 0, 0, 0, 1, 1, 1], dtype=np.int64),
        "inputs.pixel_coords": np.asarray(
            [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 2]],
            dtype=np.int32,
        ),
        "label": np.asarray([[0, 0, 0, 0], [1, 1, 1, 0]], dtype=np.int64),
        "inputs.masks": np.asarray([[0, 0, 0, 0], [1, 1, 1, 0]], dtype=np.int64),
    }
    dataset = BOPSceneEval.__new__(BOPSceneEval)
    dataset.stack_2d = True
    dataset.filter_background = None
    dataset.background_plane_threshold = 0.01
    dataset.background_plane_iterations = 64
    dataset.background_plane_max_fit_points = 100
    dataset.background_plane_min_inliers = 3
    dataset.background_plane_show = False
    dataset.statistical_outlier_removal = True
    dataset.statistical_outlier_neighbors = 1
    dataset.statistical_outlier_std_ratio = 5.0
    dataset.statistical_outlier_min_points = 2
    dataset.statistical_outlier_show = False
    dataset.dbscan_filter = True
    dataset.dbscan_eps = 0.02
    dataset.dbscan_min_points = 2
    dataset.dbscan_keep = "non_noise"
    dataset.dbscan_include_background = False
    dataset.dbscan_show = False

    monkeypatch.setattr(
        bop_scene_module,
        "_fit_plane_ransac",
        lambda *_args, **_kwargs: np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    )
    monkeypatch.setattr(dataset, "_statistical_inlier_indices", lambda *_args, **_kwargs: np.asarray([0, 1]))
    monkeypatch.setattr(dataset, "_dbscan_cluster_ids", lambda *_args, **_kwargs: np.asarray([0, -1]))

    dataset._filter_background_points(item)
    dataset._remove_statistical_outlier_points(item)
    dataset._filter_dbscan_components(item)

    assert item["inputs"].shape == (3, 3)
    assert item["inputs.labels"].shape == (3,)
    assert item["inputs.pixel_coords"].shape == (3, 2)
    np.testing.assert_array_equal(item["inputs.labels"], np.asarray([0, 0, 1], dtype=np.int64))
    np.testing.assert_array_equal(
        item["label"],
        np.asarray([[0, 0, 0, 0], [1, 0, 0, 0]], dtype=np.int64),
    )
    np.testing.assert_array_equal(item["inputs.masks"], item["label"])
    assert item["inputs.background_plane_num_background_kept"] == 3
    assert item["inputs.outlier_removal_num_removed"] == 2
    assert item["inputs.dbscan_num_removed"] == 1


def test_bop_scene_eval_missing_mesh_reports_searched_paths(tmp_path):
    root = tmp_path / "bop"
    scene = root / "tless" / "test_primesense" / "000001"
    (root / "tless" / "models_eval").mkdir(parents=True)
    _write_png(scene / "depth" / "000000.png", np.ones((1, 1), dtype=np.uint16))
    _write_json(
        scene / "scene_camera.json",
        {"0": {"cam_K": np.eye(3).reshape(-1).tolist(), "depth_scale": 1.0}},
    )
    _write_json(
        scene / "scene_gt.json",
        {"0": [{"obj_id": 7, "cam_R_m2c": np.eye(3).reshape(-1).tolist(), "cam_t_m2c": [0.0, 0.0, 1000.0]}]},
    )

    dataset = BOPSceneEval(root=root, name="tless", split="test_primesense", load_mesh=True)

    try:
        dataset[0]
    except FileNotFoundError as e:
        assert "obj_id=7" in str(e)
        assert "obj_000007.ply" in str(e)
    else:
        raise AssertionError("Expected missing BOP mesh to raise FileNotFoundError.")


def test_bop_scene_eval_mesh_decimation_import_error_does_not_cache_original_under_simplified_key(monkeypatch):
    class _FakeMesh:
        def __init__(self, vertices, faces, process=False):
            self.vertices = vertices
            self.faces = faces

        def simplify_quadric_decimation(self, _amount):
            raise ImportError("missing simplifier")

    dataset = BOPSceneEval.__new__(BOPSceneEval)
    dataset.mesh_simplify_fraction = 0.5
    dataset._mesh_decimation_enabled = True
    dataset.mesh_scale = 1.0
    dataset._mesh_cache = {}
    path = Path("obj_000001.ply")
    vertices = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    faces = np.asarray([[0, 1, 2], [0, 2, 1]], dtype=np.int64)
    monkeypatch.setattr(bop_scene_module, "_load_mesh", lambda _path: (vertices, faces))
    monkeypatch.setattr(bop_scene_module, "Trimesh", _FakeMesh)

    mesh = dataset._load_mesh_cached(path)

    assert mesh.faces is faces
    assert dataset._mesh_decimation_enabled is True
    assert (path, 0.5, 1.0) not in dataset._mesh_cache
    assert (path, None, 1.0) in dataset._mesh_cache


def test_bop_scene_eval_mesh_decimation_runtime_errors_propagate(monkeypatch):
    class _FakeMesh:
        def __init__(self, vertices, faces, process=False):
            self.vertices = vertices
            self.faces = faces

        def simplify_quadric_decimation(self, _amount):
            raise RuntimeError("bad mesh")

    dataset = BOPSceneEval.__new__(BOPSceneEval)
    dataset.mesh_simplify_fraction = 0.5
    dataset._mesh_decimation_enabled = True
    dataset.mesh_scale = 1.0
    dataset._mesh_cache = {}
    path = Path("obj_000001.ply")
    vertices = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    faces = np.asarray([[0, 1, 2], [0, 2, 1]], dtype=np.int64)
    monkeypatch.setattr(bop_scene_module, "_load_mesh", lambda _path: (vertices, faces))
    monkeypatch.setattr(bop_scene_module, "Trimesh", _FakeMesh)

    with pytest.raises(RuntimeError, match="bad mesh"):
        dataset._load_mesh_cached(path)

    assert dataset._mesh_cache == {}


def test_bop_scene_eval_real_mesh_and_depth_share_meter_scale():
    bop_root = os.environ.get("BOP_ROOT")
    if bop_root is None:
        pytest.skip("Set BOP_ROOT to run the real BOP scene scale check.")

    root = Path(bop_root)
    if not (root / "hb" / "val_primesense" / "000001").exists():
        pytest.skip("Real HB val_primesense scene 000001 is not available.")

    dataset = BOPSceneEval(
        root=root,
        name="hb",
        split="val_primesense",
        scene_ids=[1],
        project=True,
        load_label=True,
        stack_2d=True,
        load_mesh=True,
        crop_to_mesh=False,
        mesh_simplify_fraction=None,
        filter_background=None,
    )
    item = dataset[0]

    inputs = item["inputs"]
    vertices = item["mesh.vertices_cam"]
    mesh_extent = np.ptp(vertices, axis=0)
    intersection_extent = np.minimum(inputs.max(axis=0), vertices.max(axis=0)) - np.maximum(
        inputs.min(axis=0), vertices.min(axis=0)
    )

    assert mesh_extent.max() < 2.0
    assert np.all(intersection_extent > 0.0)
