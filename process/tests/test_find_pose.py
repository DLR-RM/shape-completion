import numpy as np
from scipy.spatial.transform import Rotation as R

from ..generate_physics_poses import filter_matrices


def test_filter_matrices():
    rot = list(R.from_euler('x', [0, -np.pi / 2, np.pi / 2, np.pi]).as_matrix())
    rot += list(R.from_euler('y', [-np.pi / 2, np.pi / 2]).as_matrix())
    assert len(filter_matrices(rot)) == len(rot)

    rot += list(R.from_euler('z', np.random.rand(3) * 2 * np.pi).as_matrix())
    assert len(filter_matrices(rot)) == len(rot) - 3
