from __future__ import annotations

import os
from pathlib import Path

import pytest

from ..src.shapenet import ShapeNet

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def shapenet_root() -> Path:
    root = os.environ.get("SHAPENET_V1_FUSED_SIMPLE")
    if not root:
        pytest.skip("Set SHAPENET_V1_FUSED_SIMPLE to run ShapeNet integration tests.")
    path = Path(root).expanduser().resolve()
    if not path.is_dir():
        pytest.skip(f"ShapeNet root does not exist: {path}")
    return path


def test_shapenet_sample_loads(shapenet_root: Path):
    inputs_dir = Path(os.environ.get("SHAPENET_V1_INPUTS", str(shapenet_root) + ".kinect"))
    if not inputs_dir.is_dir():
        pytest.skip(f"ShapeNet inputs directory not found: {inputs_dir}")

    dataset = ShapeNet(
        split="val",
        data_dir=shapenet_root,
        inputs_dir=inputs_dir,
        categories=["03797390"],
        unscale=True,
        undistort=True,
        unrotate=True,
        points_file="samples/uniform_random.npz",
        load_random_points=False,
        inputs_type="kinect",
    )
    item = dataset[0]
    assert item is not None
