import numpy as np

from process.src import carve


def test_xyz_spherical_forward_axis() -> None:
    r, r_x, r_y = carve.xyz_spherical(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    assert np.isclose(r, 1.0)
    assert np.isclose(r_x, 0.0)
    assert np.isclose(r_y, 0.0)


def test_get_rotation_matrix_identity() -> None:
    rotation = carve.get_rotation_matrix(0.0, 0.0)
    np.testing.assert_allclose(rotation, np.eye(3), atol=1e-7)


def test_get_extrinsic_has_expected_translation() -> None:
    extrinsic = carve.get_extrinsic(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    assert extrinsic.shape == (4, 4)
    np.testing.assert_allclose(extrinsic[:3, 3], np.array([0.0, 0.0, 2.0]), atol=1e-7)
