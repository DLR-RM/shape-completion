import gzip
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils import setup_logger

logger = setup_logger(__name__)


def _log_debug_level_1(message: str) -> None:
    log_fn = getattr(logger, "debug_level_1", logger.debug)
    log_fn(message)


def get_file(name_or_url: str | Path) -> Path:
    if name_or_url == "bunny":
        url = "http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz"
    elif name_or_url == "armadillo":
        url = "http://graphics.stanford.edu/pub/3Dscanrep/armadillo/Armadillo.ply.gz"
    elif name_or_url == "torus":
        raise NotImplementedError("Torus is not available for download.")
    elif name_or_url == "sphere":
        raise NotImplementedError("Sphere is not available for download.")
    else:
        url = name_or_url

    file = Path(tempfile.gettempdir()) / Path(url).name

    if name_or_url == "bunny":
        extract_file = file.parent / file.name.split(".")[0] / "reconstruction" / "bun_zipper.ply"
        out_file = file.parent / "bunny" / "mesh.ply"
        out_file.parent.mkdir(parents=True, exist_ok=True)

    elif name_or_url == "armadillo":
        extract_file = file.parent / "Armadillo.ply"
        out_file = file.parent / "armadillo" / "mesh.ply"
        out_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        raise NotImplementedError

    if not out_file.exists():
        if url is not None:
            _log_debug_level_1(f"Downloading {name_or_url} from {url} to {file}")
            urlretrieve(str(url), file)
            if ".tar" in file.suffixes:
                with tarfile.open(file) as tar:
                    tar.extractall(file.parent)
            elif file.suffix == ".gz":
                with gzip.open(file) as f_in:
                    with open(file.parent / file.stem, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
            shutil.move(extract_file, out_file)
            _log_debug_level_1(f"Extracted file to {out_file}")
    return out_file


def calibration_from_blender(
    az: float, el: float, distance_ratio: float, width: int = 127, height: int = 127, padding: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate 4x3 3D to 2D projection matrix given viewpoint parameters."""
    FOV = None
    F_MM = 35.0  # Focal length
    SENSOR_SIZE_MM = 32.0
    PIXEL_ASPECT_RATIO = 1.0  # pixel_aspect_x / pixel_aspect_y
    RESOLUTION_PCT = 100.0
    SKEW = 0.0
    CAM_MAX_DIST = 1.75
    CAM_ROT = np.asarray(
        [
            [1.910685676922942e-15, 4.371138828673793e-08, 1.0],
            [1.0, -4.371138828673793e-08, -0.0],
            [4.371138828673793e-08, 1.0, -4.371138828673793e-08],
        ]
    )

    # Adjust F_MM based on FOV if provided
    if FOV is not None:
        F_MM = 0.5 * SENSOR_SIZE_MM / np.tan(np.radians(FOV) / 2)

    # Calculate intrinsic matrix.
    scale = RESOLUTION_PCT / 100
    f_u = F_MM * (width + padding) * scale / SENSOR_SIZE_MM
    f_v = F_MM * (height + padding) * scale * PIXEL_ASPECT_RATIO / SENSOR_SIZE_MM
    u_0 = (width + padding) * scale / 2
    v_0 = (height + padding) * scale / 2
    K = np.array(((f_u, SKEW, u_0), (0.0, f_v, v_0), (0.0, 0.0, 1.0)), dtype=np.float64)

    # Calculate rotation and translation matrices.
    # Step 1: World coordinate to object coordinate.
    sa = np.sin(np.radians(-az))
    ca = np.cos(np.radians(-az))
    se = np.sin(np.radians(-el))
    ce = np.cos(np.radians(-el))
    R_world2obj = np.array(((ca * ce, -sa, ca * se), (sa * ce, ca, sa * se), (-se, 0.0, ce)), dtype=np.float64).T

    # Step 2: Object coordinate to camera coordinate.
    R_obj2cam = np.array(CAM_ROT, dtype=np.float64).T
    R_world2cam = R_obj2cam @ R_world2obj
    cam_location = np.array([[distance_ratio * CAM_MAX_DIST], [0.0], [0.0]], dtype=np.float64)
    # print('distance', distance_ratio * CAM_MAX_DIST)
    T_world2cam = -R_obj2cam @ cam_location

    # Step 3: Fix blender camera's y and z axis direction.
    R_camfix = np.array(((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)), dtype=np.float64)
    R_world2cam = R_camfix @ R_world2cam
    T_world2cam = R_camfix @ T_world2cam

    RT = np.hstack((R_world2cam, T_world2cam))

    return K, RT


class TorchvisionDatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.name = dataset.__class__.__name__

    def __getattr__(self, attr):
        return getattr(self.dataset, attr)

    def __setattr__(self, key, value):
        if key in ["dataset", "name"]:
            super().__setattr__(key, value)
        else:
            setattr(self.dataset, key, value)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        img, target = self.dataset[index]
        _, h, w = img.size()
        if h < 32 or w < 32:
            pad = (32 - w) // 2
            img = F.pad(img, (pad, pad, pad, pad), mode="constant", value=0)
        return {"index": index, "inputs": img, "inputs.labels": target}

    def __repr__(self):
        return self.name.upper()

    def __str__(self):
        return self.name
