import torch

from .. import furthest_point_sample, gather_operation


def test_furthest_point_sample():
    pcd = torch.rand((32, 3000, 3)).cuda()
    pcd = furthest_point_sample(pcd, 2048)
    assert pcd.shape == (32, 2048)


def test_gather_operation():
    pcd = torch.rand((32, 3000, 3)).cuda()
    idx = furthest_point_sample(pcd, 2048)
    pcd = gather_operation(pcd.transpose(1, 2).contiguous(), idx)
    assert pcd.transpose(1, 2).shape == (32, 2048, 3)
