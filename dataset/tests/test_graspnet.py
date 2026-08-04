from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from ..src import graspnet as graspnet_module
from ..src.graspnet import GraspNetEval, _labels_at_pixels, _Sample, _stable_point_seed


def test_mesh_decimation_failure_disables_once(monkeypatch, caplog):
    dataset = GraspNetEval.__new__(GraspNetEval)
    dataset.mesh_simplify_fraction = 0.1
    dataset._mesh_decimation_enabled = True
    dataset._mesh_decimation_disabled_reason = None
    dataset._mesh_cache = {}
    dataset._points_cache = {}

    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)

    monkeypatch.setattr(graspnet_module, "load_mesh", lambda _path: (vertices, faces))

    call_counter = {"count": 0}

    def _raise_missing_dependency(self, _target_fraction):
        call_counter["count"] += 1
        raise ModuleNotFoundError("No module named 'fast_simplification'")

    monkeypatch.setattr(graspnet_module.Trimesh, "simplify_quadric_decimation", _raise_missing_dependency)

    with caplog.at_level(logging.WARNING):
        dataset._load_mesh_cached(Path("obj_001.ply"))
        dataset._load_mesh_cached(Path("obj_002.ply"))

    warning_lines = [
        record.message for record in caplog.records if "Mesh decimation disabled after first failure" in record.message
    ]
    assert len(warning_lines) == 1
    assert call_counter["count"] == 1
    assert dataset._mesh_decimation_enabled is False
    assert dataset._mesh_decimation_disabled_reason is not None


def test_load_rgb_returns_chw_uint8(tmp_path: Path) -> None:
    rgb_dir = tmp_path / "scene_0100" / "kinect" / "rgb"
    rgb_dir.mkdir(parents=True)
    image = np.asarray(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    Image.fromarray(image).save(rgb_dir / "0000.png")

    dataset = GraspNetEval.__new__(GraspNetEval)
    dataset.camera = "kinect"
    sample = _Sample(scene_dir=tmp_path / "scene_0100", ann_id="0000")

    loaded = dataset._load_rgb(sample)

    assert loaded.shape == (3, 2, 2)
    assert loaded.dtype == np.uint8
    np.testing.assert_array_equal(loaded[:, 0, 0], np.asarray([255, 0, 0], dtype=np.uint8))


def test_point_colors_from_chw_image_are_aligned_and_unit_range() -> None:
    image = np.asarray(
        [
            [[255, 0], [0, 255]],
            [[0, 255], [0, 255]],
            [[0, 0], [255, 255]],
        ],
        dtype=np.uint8,
    )

    colors = GraspNetEval._point_colors_from_image(
        image,
        rows=np.asarray([0, 1]),
        cols=np.asarray([0, 0]),
    )

    np.testing.assert_allclose(colors, np.asarray([[1, 0, 0], [0, 0, 1]], dtype=np.float32))


def test_apply_input_point_keep_mask_reindexes_pixel_coords() -> None:
    dataset = GraspNetEval.__new__(GraspNetEval)
    item = {
        "inputs": np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
        "inputs.labels": np.asarray([0, 1, 2], dtype=np.int64),
        "inputs.pixel_coords": np.asarray([[0, 0], [0, 1], [1, 0]], dtype=np.int32),
        "inputs.colors": np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    }

    dataset._apply_input_point_keep_mask(item, np.asarray([True, False, True]))

    assert item["inputs"].shape == (2, 3)
    np.testing.assert_array_equal(item["inputs.labels"], np.asarray([0, 2], dtype=np.int64))
    np.testing.assert_array_equal(item["inputs.pixel_coords"], np.asarray([[0, 0], [1, 0]], dtype=np.int32))
    np.testing.assert_array_equal(
        item["inputs.colors"],
        np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    )


def test_labels_at_pixels_maps_empty_instance_set_to_background() -> None:
    labels = _labels_at_pixels(
        np.zeros((0, 2, 3), dtype=np.uint8),
        rows=np.asarray([0, 1, 1]),
        cols=np.asarray([1, 0, 2]),
    )

    np.testing.assert_array_equal(labels, np.zeros(3, dtype=np.int64))


def test_enumerate_samples_filters_explicit_view_ids(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene_0000"
    depth_dir = scene_dir / "kinect" / "depth"
    depth_dir.mkdir(parents=True)
    for view_id in (0, 16, 32, 64):
        (depth_dir / f"{view_id:04d}.png").touch()

    dataset = GraspNetEval.__new__(GraspNetEval)
    dataset.scene_dirs = [scene_dir]
    dataset.scene_ids = None
    dataset.camera = "kinect"
    dataset.one_view_per_scene = False
    dataset.view_ids = {"0000", "0032"}

    samples = dataset._enumerate_samples("train")

    assert [sample.ann_id for sample in samples] == ["0000", "0032"]


def test_stable_point_seed_matches_frozen_protocol_rule() -> None:
    assert _stable_point_seed(42, "scene_0000/kinect/0000") == 3398647467
