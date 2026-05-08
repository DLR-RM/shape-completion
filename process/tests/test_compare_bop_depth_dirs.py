from __future__ import annotations

import csv
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from process.scripts import compare_bop_depth_dirs as compare_depth


def _write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_run_compares_predicted_depth_dir_to_sensor_depth(tmp_path: Path) -> None:
    root = tmp_path / "bop"
    scene = root / "hb" / "val_primesense" / "000001"
    _write_png(scene / "rgb" / "000000.png", np.zeros((2, 2, 3), dtype=np.uint8))
    _write_png(scene / "depth" / "000000.png", np.asarray([[1000, 2000], [0, 4000]], dtype=np.uint16))
    _write_png(scene / "depth_da3" / "000000.png", np.asarray([[1100, 1800], [3000, 0]], dtype=np.uint16))
    _write_json(
        scene / "scene_camera.json",
        {"0": {"cam_K": [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]], "depth_scale": 1.0}},
    )
    out_dir = tmp_path / "audit"

    summary = compare_depth.run(
        Namespace(
            bop_root=root,
            dataset="hb",
            split="val_primesense",
            scene_ids=[1],
            frame_ids=None,
            sensor_depth_dir="depth",
            pred_depth_dir="depth_da3",
            pred_depth_scale=None,
            out_dir=out_dir,
            num_visualizations=1,
        )
    )

    assert summary["num_frames"] == 1
    assert summary["summary"]["mae_m"]["mean"] == pytest.approx(0.15000003576278687)
    assert summary["summary"]["pred_over_sensor_valid_ratio"]["mean"] == 2 / 3
    assert summary["summary"]["hallucinated_valid_ratio"]["mean"] == 0.25
    assert (out_dir / "hb_val_primesense_depth_da3_metrics.csv").exists()
    assert (out_dir / "hb_val_primesense_depth_da3_summary.json").exists()
    assert (out_dir / "visualizations" / "hb_val_primesense_scene000001_frame000000.png").exists()

    with (out_dir / "hb_val_primesense_depth_da3_metrics.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["dataset"] == "hb"
    assert rows[0]["scene_id"] == "1"
    assert rows[0]["frame_id"] == "0"
    assert float(rows[0]["delta1"]) == pytest.approx(1.0)
    assert float(rows[0]["abs_rel"]) == pytest.approx((0.1 / 1.0 + 0.2 / 2.0) / 2.0)
    assert float(rows[0]["rmse_m"]) == pytest.approx(np.sqrt((0.1**2 + 0.2**2) / 2.0))
    assert float(rows[0]["bias_m"]) == pytest.approx(-0.05)
    expected_pcd_errors = np.asarray([0.1, 0.2 * np.sqrt(1.0 + 0.01**2)])
    assert float(rows[0]["pcd_mae_m"]) == pytest.approx(float(expected_pcd_errors.mean()))
    assert float(rows[0]["pcd_rmse_m"]) == pytest.approx(float(np.sqrt(np.mean(expected_pcd_errors**2))))
    assert float(rows[0]["median_scale_to_sensor"]) == pytest.approx(np.median([1.0 / 1.1, 2.0 / 1.8]))


def test_run_decodes_pred_depth_with_scene_depth_scale_by_default(tmp_path: Path) -> None:
    root = tmp_path / "bop"
    scene = root / "ycbv" / "test" / "000001"
    _write_png(scene / "rgb" / "000000.png", np.zeros((1, 1, 3), dtype=np.uint8))
    _write_png(scene / "depth" / "000000.png", np.asarray([[15000]], dtype=np.uint16))
    _write_png(scene / "depth_da3" / "000000.png", np.asarray([[15000]], dtype=np.uint16))
    _write_json(
        scene / "scene_camera.json",
        {"0": {"cam_K": np.eye(3).reshape(-1).tolist(), "depth_scale": 0.1}},
    )

    summary = compare_depth.run(
        Namespace(
            bop_root=root,
            dataset="ycbv",
            split="test",
            scene_ids=[1],
            frame_ids=None,
            sensor_depth_dir="depth",
            pred_depth_dir="depth_da3",
            pred_depth_scale=None,
            out_dir=tmp_path / "audit",
            num_visualizations=0,
        )
    )

    assert summary["summary"]["mae_m"]["mean"] == pytest.approx(0.0)


def test_run_can_override_pred_depth_scale_for_legacy_fixed_millimeter_outputs(tmp_path: Path) -> None:
    root = tmp_path / "bop"
    scene = root / "ycbv" / "test" / "000001"
    _write_png(scene / "rgb" / "000000.png", np.zeros((1, 1, 3), dtype=np.uint8))
    _write_png(scene / "depth" / "000000.png", np.asarray([[15000]], dtype=np.uint16))
    _write_png(scene / "depth_da3" / "000000.png", np.asarray([[1500]], dtype=np.uint16))
    _write_json(
        scene / "scene_camera.json",
        {"0": {"cam_K": np.eye(3).reshape(-1).tolist(), "depth_scale": 0.1}},
    )

    summary = compare_depth.run(
        Namespace(
            bop_root=root,
            dataset="ycbv",
            split="test",
            scene_ids=[1],
            frame_ids=None,
            sensor_depth_dir="depth",
            pred_depth_dir="depth_da3",
            pred_depth_scale=1.0,
            out_dir=tmp_path / "audit",
            num_visualizations=0,
        )
    )

    assert summary["summary"]["mae_m"]["mean"] == pytest.approx(0.0)
