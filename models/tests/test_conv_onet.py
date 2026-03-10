import torch

from ..src.conv_onet import ConvONet


class TestConvONet:
    def test_init_grid(self):
        ConvONet()

    def test_init_plane(self):
        ConvONet(arch="conv_onet_plane")

    def test_init_uv_plane(self):
        ConvONet(arch="conv_onet_uv_plane")

    def test_forward_grid(self):
        conv_onet = ConvONet()

        inputs = torch.randn(4, 3000, 3) - 0.5
        points = 1.1 * torch.randn(4, 2048, 3) - 0.55

        conv_onet(inputs, points)

    def test_forward_plane(self):
        conv_onet = ConvONet(arch="conv_onet_plane")

        inputs = torch.randn(4, 3000, 3) - 0.5
        points = 1.1 * torch.randn(4, 2048, 3) - 0.55

        conv_onet(inputs, points)

    def test_forward_uv_plane(self):
        conv_onet = ConvONet(arch="conv_onet_uv_plane")

        inputs = torch.randn(4, 3000, 3) - 0.5
        points = 1.1 * torch.randn(4, 2048, 3) - 0.55
        kwargs = {
            "inputs.width": 640 * torch.ones(4),
            "inputs.height": 480 * torch.ones(4),
            "inputs.intrinsic": torch.eye(3).unsqueeze(0).repeat(4, 1, 1),
        }

        conv_onet(inputs, points, **kwargs)
