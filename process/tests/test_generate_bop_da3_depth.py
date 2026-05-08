from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image

from process.scripts import generate_bop_da3_depth as da3


def _write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_depth_m_to_bop_uint16_encodes_meters_as_millimeters() -> None:
    depth = np.asarray([[1.234, np.nan], [-1.0, 70.0]], dtype=np.float32)

    encoded = da3.depth_m_to_bop_uint16(depth)

    np.testing.assert_array_equal(encoded, np.asarray([[1234, 0], [0, 65535]], dtype=np.uint16))


def test_depth_m_to_bop_uint16_applies_bop_depth_scale() -> None:
    depth = np.asarray([[1.5]], dtype=np.float32)

    encoded = da3.depth_m_to_bop_uint16(depth, depth_scale=0.1)

    np.testing.assert_array_equal(encoded, np.asarray([[15000]], dtype=np.uint16))


def test_metric_conversion_falls_back_to_input_intrinsics_for_scalar_da3_intrinsics() -> None:
    depths = np.ones((1, 2, 2), dtype=np.float32)
    bop_intrinsics = [np.asarray([[600.0, 0.0, 0.0], [0.0, 300.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)]

    converted = da3._convert_metric_depths(depths, np.asarray(0.0, dtype=np.float32), bop_intrinsics)

    assert len(converted) == 1
    np.testing.assert_allclose(converted[0], np.full((2, 2), 1.5, dtype=np.float32))


def test_metric_conversion_scales_by_focal_against_da3_basis() -> None:
    depths = np.ones((1, 2, 2), dtype=np.float32)
    bop_intrinsics = [np.asarray([[600.0, 0.0, 0.0], [0.0, 600.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)]

    converted = da3._convert_metric_depths(depths, None, bop_intrinsics)

    np.testing.assert_allclose(converted[0], np.full((2, 2), 2.0, dtype=np.float32))


def test_scale_factor_least_squares_and_degenerate_denominator() -> None:
    pred = np.asarray([[1.0, 2.0]], dtype=np.float32)
    sensor = np.asarray([[2.0, 4.0]], dtype=np.float32)

    assert da3._scale_factor(pred, sensor, "least_squares") == 2.0
    assert da3._scale_factor(np.zeros((1, 1), dtype=np.float32), sensor[:, :1], "least_squares") == 1.0


def test_run_writes_da3_depth_directory_and_metadata(tmp_path: Path) -> None:
    root = tmp_path / "bop"
    scene = root / "hb" / "val_primesense" / "000001"
    _write_png(scene / "rgb" / "000000.png", np.zeros((2, 2, 3), dtype=np.uint8))
    _write_png(scene / "rgb" / "000001.png", np.ones((2, 2, 3), dtype=np.uint8))
    _write_png(scene / "depth" / "000000.png", np.asarray([[1000, 0], [2000, 2000]], dtype=np.uint16))
    _write_png(scene / "depth" / "000001.png", np.asarray([[3000, 3000], [0, 3000]], dtype=np.uint16))
    _write_json(
        scene / "scene_camera.json",
        {
            "0": {"cam_K": np.eye(3).reshape(-1).tolist(), "depth_scale": 1.0},
            "1": {"cam_K": np.eye(3).reshape(-1).tolist(), "depth_scale": 1.0},
        },
    )

    def fake_predict(_images: list[Path], _intrinsics: list[np.ndarray]) -> list[np.ndarray]:
        return [np.asarray([[0.5, 0.5], [1.0, 1.0]], dtype=np.float32)]

    args = Namespace(
        bop_root=root,
        dataset="hb",
        split="val_primesense",
        scene_ids=[1],
        frame_ids=[0],
        dst_depth_dir="depth_da3",
        valid_mask="sensor",
        align_scale="median",
        overwrite=False,
        model="fake-da3",
    )

    summary = da3.run(args, predict_depths=fake_predict)

    out = np.asarray(Image.open(scene / "depth_da3" / "000000.png"))
    np.testing.assert_array_equal(out, np.asarray([[1000, 0], [2000, 2000]], dtype=np.uint16))
    assert not (scene / "depth_da3" / "000001.png").exists()
    assert summary["num_written"] == 1
    assert summary["num_skipped"] == 0
    meta = json.loads((root / "hb" / "val_primesense" / "depth_da3_meta.json").read_text(encoding="utf-8"))
    assert meta["model"] == "fake-da3"
    assert meta["align_scale"] == "median"
    assert meta["frames"][0]["scale"] == 2.0


def test_run_writes_depths_with_scene_depth_scale(tmp_path: Path) -> None:
    root = tmp_path / "bop"
    scene = root / "ycbv" / "test" / "000001"
    _write_png(scene / "rgb" / "000000.png", np.zeros((1, 1, 3), dtype=np.uint8))
    _write_png(scene / "depth" / "000000.png", np.asarray([[15000]], dtype=np.uint16))
    _write_json(
        scene / "scene_camera.json",
        {"0": {"cam_K": np.eye(3).reshape(-1).tolist(), "depth_scale": 0.1}},
    )
    args = Namespace(
        bop_root=root,
        dataset="ycbv",
        split="test",
        scene_ids=[1],
        frame_ids=[0],
        dst_depth_dir="depth_da3",
        valid_mask="sensor",
        align_scale="none",
        overwrite=False,
        model="fake-da3",
    )

    summary = da3.run(args, predict_depths=lambda *_args: [np.asarray([[1.5]], dtype=np.float32)])

    out = np.asarray(Image.open(scene / "depth_da3" / "000000.png"))
    np.testing.assert_array_equal(out, np.asarray([[15000]], dtype=np.uint16))
    assert summary["metadata"]["frames"][0]["depth_scale"] == 0.1


def test_run_counts_existing_outputs_as_skipped(tmp_path: Path) -> None:
    root = tmp_path / "bop"
    scene = root / "hb" / "val_primesense" / "000001"
    _write_png(scene / "rgb" / "000000.png", np.zeros((1, 1, 3), dtype=np.uint8))
    _write_png(scene / "depth" / "000000.png", np.asarray([[1000]], dtype=np.uint16))
    _write_png(scene / "depth_da3" / "000000.png", np.asarray([[999]], dtype=np.uint16))
    _write_json(
        scene / "scene_camera.json",
        {"0": {"cam_K": np.eye(3).reshape(-1).tolist(), "depth_scale": 1.0}},
    )
    args = Namespace(
        bop_root=root,
        dataset="hb",
        split="val_primesense",
        scene_ids=[1],
        frame_ids=[0],
        dst_depth_dir="depth_da3",
        valid_mask="sensor",
        align_scale="none",
        overwrite=False,
        model="fake-da3",
    )

    summary = da3.run(args, predict_depths=lambda *_args: [np.asarray([[1.0]], dtype=np.float32)])

    assert summary["num_written"] == 0
    assert summary["num_skipped"] == 1
    np.testing.assert_array_equal(np.asarray(Image.open(scene / "depth_da3" / "000000.png")), np.asarray([[999]], dtype=np.uint16))
