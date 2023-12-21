import torch

from .. import ChamferDistanceL2


def test_chamfer_init():
    ChamferDistanceL2()


def test_chamfer_forward():
    chamfer = ChamferDistanceL2()
    x = torch.rand(1, 3, 100).cuda()
    y = torch.rand(1, 3, 100).cuda()
    chamfer(x, y)


def test_chamfer_backward():
    chamfer = ChamferDistanceL2()
    x = torch.rand(1, 3, 100).cuda()
    y = torch.rand(1, 3, 100, requires_grad=True).cuda()
    dist1, dist2 = chamfer(x, y)
    loss = dist1.mean() + dist2.mean()
    loss.backward()
