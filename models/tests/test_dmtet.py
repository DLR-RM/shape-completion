import torch

from ..src.dmtet import DMTet


class TestDMTet:
    def test_init(self):
        DMTet()

    def test_forward(self):
        dmtet = DMTet()

        inputs = torch.randn(4, 3000, 3) - 0.5
        points = 1.1 * torch.randn(4, 2048, 3) - 0.55

        dmtet(inputs, points)
