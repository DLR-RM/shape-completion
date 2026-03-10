from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from utils import depth_to_points, points_to_depth

from ..src.fields import DepthField, DTUField, EmptyField, PointCloudField, PointsField


def test_init():
    for field in [EmptyField, PointsField, PointCloudField]:
        field()


@pytest.mark.integration
def test_dtu_field_loads_one_sample():
    dtu_dir = os.environ.get("DTU_SAMPLE_DIR")
    if not dtu_dir:
        pytest.skip("Set DTU_SAMPLE_DIR to run DTU integration tests.")

    path = Path(dtu_dir).expanduser().resolve()
    if not path.is_dir():
        pytest.skip(f"DTU sample directory not found: {path}")

    field = DTUField()
    data = field.load(path, index=0, category=None)
    assert None in data
    assert "intrinsic" in data
    assert "extrinsic" in data
    assert "width" in data
    assert "height" in data


@pytest.mark.integration
def test_depth_field_roundtrip():
    model_dir = os.environ.get("SHAPENET_SAMPLE_OBJECT_DIR")
    if not model_dir:
        pytest.skip("Set SHAPENET_SAMPLE_OBJECT_DIR to run depth field integration tests.")

    path = Path(model_dir).expanduser().resolve()
    if not path.is_dir():
        pytest.skip(f"ShapeNet sample object directory not found: {path}")

    depth_field = DepthField(project=False, crop=False)
    depth_data = depth_field.load(path, index=0)

    depth = depth_data[None]
    intrinsic = depth_data["intrinsic"]
    width = depth_data["width"]
    height = depth_data["height"]

    points = depth_to_points(depth, intrinsic, depth_scale=1.0, depth_trunc=6.0)
    depth_reconstructed = points_to_depth(points, intrinsic, width, height, depth_scale=1.0)

    assert depth_reconstructed.shape == depth.shape
    assert np.isfinite(depth_reconstructed).all()
