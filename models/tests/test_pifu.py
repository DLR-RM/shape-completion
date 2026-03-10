import torch

from ..src.pifu import PIFu


def test_init():
    PIFu()


def test_forward():
    images = torch.rand(2, 3, 512, 512)
    points = torch.rand(2, 10000, 3)
    intrinsic = torch.rand(2, 3, 3)
    extrinsic = torch.rand(2, 4, 4)

    pifu = PIFu()
    pifu(images, points, intrinsic, extrinsic)
