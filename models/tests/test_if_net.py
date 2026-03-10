import torch

from ..src.if_net import IFNet


class TestIFNet:
    def test_forward(self):
        inputs = torch.rand(8, 128, 128, 128).cuda()
        points = 1.1 * torch.rand(8, 2048, 3).cuda() - 0.55

        if_net = IFNet(displacements=False).cuda()
        if_net(inputs, points)
