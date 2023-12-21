import numpy as np

from .. import PyViews, tsdf_fusion


def test_fusion():
    depthmaps = [np.random.rand(480, 640) for _ in range(10)]
    rotations = [np.eye(3) for _ in range(len(depthmaps))]

    Ks = np.array([[640, 0, 320],
                   [0, 640, 320],
                   [0, 0, 1]]).reshape((1, 3, 3))

    Ks = np.repeat(Ks, len(depthmaps), axis=0).astype(np.float32)
    Rs = np.array(rotations).astype(np.float32)
    Ts = np.array([np.array([0, 0, 1]) for _ in range(len(Rs))]).astype(np.float32)

    depthmaps = np.array(depthmaps).astype(np.float32)
    resolution = 256
    voxel_size = 1 / resolution

    views = PyViews(depthmaps, Ks, Rs, Ts)
    tsdf = tsdf_fusion(views,
                       depth=resolution,
                       height=resolution,
                       width=resolution,
                       vx_size=voxel_size,
                       truncation=10 * voxel_size,
                       unknown_is_free=False)[0].transpose((2, 1, 0))

    assert tsdf.shape == (resolution, resolution, resolution)
