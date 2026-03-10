import numpy as np
import pytest

from ..src.utils import calibration_from_blender, get_file


@pytest.mark.integration
def test_get_file():
    for name in ["bunny", "armadillo"]:
        file = get_file(name)
        assert file.exists()


def test_calibration_from_blender():
    az, el, distance_ratio = 45.0, 30.0, 1.0
    intrinsic, extrinsic = calibration_from_blender(az, el, distance_ratio)
    assert intrinsic.shape == (3, 3)
    assert extrinsic.shape == (3, 4)
    assert np.isfinite(np.asarray(intrinsic)).all()
    assert np.isfinite(np.asarray(extrinsic)).all()
