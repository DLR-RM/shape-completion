import importlib
import sys
import types
from argparse import Namespace
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R


def _load_module() -> Any:
    # Keep tests independent from optional rembg installation.
    if "rembg" not in sys.modules:
        rembg = types.ModuleType("rembg")
        cast(Any, rembg).remove = lambda image, **_kwargs: np.ones(image.shape[:2], dtype=np.uint8) * 255
        sys.modules["rembg"] = rembg
    module = importlib.import_module("process.scripts.ycb_sfm")
    return importlib.reload(module)


def test_inv_trafo_inverts_rigid_transform() -> None:
    ycb_sfm = _load_module()
    trafo = np.eye(4, dtype=np.float64)
    trafo[:3, :3] = R.from_euler("xyz", [20, -10, 35], degrees=True).as_matrix()
    trafo[:3, 3] = np.array([0.2, -0.4, 1.3], dtype=np.float64)

    inv = ycb_sfm.inv_trafo(trafo)

    assert np.allclose(inv @ trafo, np.eye(4), atol=1e-8)
    assert np.allclose(trafo @ inv, np.eye(4), atol=1e-8)


def test_get_camera_opencv_layout() -> None:
    ycb_sfm = _load_module()
    calib = {
        "N1_rgb_K": np.array(
            [[500.0, 0.0, 320.0], [0.0, 510.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        "N1_rgb_d": np.array([0.1, -0.2, 0.01, -0.03, 0.05], dtype=np.float64),
    }

    camera = ycb_sfm.get_camera(
        cam_id=1,
        camera_model=ycb_sfm.camera_models[4],
        width=640,
        height=480,
        calibration=calib,
    )

    assert camera == [
        1,
        "OPENCV",
        640,
        480,
        500.0,
        510.0,
        320.0,
        240.0,
        0.1,
        -0.2,
        0.01,
        -0.03,
    ]


def test_get_image_builds_colmap_row(monkeypatch: Any) -> None:
    ycb_sfm = _load_module()

    class _PoseFile:
        def __init__(self, table_from_ref: np.ndarray) -> None:
            self._table_from_ref = table_from_ref

        def __enter__(self) -> "_PoseFile":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def __getitem__(self, key: str) -> np.ndarray:
            if key != "H_table_from_reference_camera":
                raise KeyError(key)
            return self._table_from_ref

        def close(self) -> None:
            return None

    table_from_ref = np.eye(4, dtype=np.float64)
    monkeypatch.setattr(ycb_sfm.h5py, "File", lambda *_args, **_kwargs: _PoseFile(table_from_ref))
    calibration = {"H_N1_from_NP5": np.eye(4, dtype=np.float64)}

    image = ycb_sfm.get_image(
        cam_id=1,
        angle=6,
        image_counter=3,
        data_dir="/tmp/data",
        calibration=calibration,
    )

    assert image == [3, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, "cam1/image003.jpg"]


def test_get_mask_returns_full_size_mask(tmp_path: Path, monkeypatch: Any) -> None:
    ycb_sfm = _load_module()
    ycb_sfm.args = Namespace(show_bbox=False)

    data_dir = tmp_path / "data"
    masks_dir = data_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    init_mask = np.ones((12, 12), dtype=np.uint8) * 255
    init_mask[3:9, 4:10] = 0
    Image.fromarray(init_mask).save(masks_dir / "N1_0_mask.pbm")

    image = np.ones((12, 12, 3), dtype=np.uint8) * 127
    monkeypatch.setattr(
        ycb_sfm,
        "remove",
        lambda rembg_input, **_kwargs: np.ones(rembg_input.shape[:2], dtype=np.uint8) * 255,
    )

    save_paths: list[str] = []

    def _capture_save(self: Image.Image, fp: Any, *_args: Any, **_kwargs: Any) -> None:
        save_paths.append(str(fp))

    monkeypatch.setattr(Image.Image, "save", _capture_save)

    mask = ycb_sfm.get_mask(
        data_dir=str(data_dir),
        image_path=str(data_dir / "N1_0.jpg"),
        pad=[0, 0, 0, 0],
        image=image,
    )

    assert mask.shape == (12, 12)
    assert mask.dtype == np.uint8
    assert np.count_nonzero(mask) > 0
    assert save_paths == []
