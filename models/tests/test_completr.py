from typing import Any, cast

import torch
from torch import nn

from ..src.completr import CompleTr
from ..src.conv_onet import ConvONet


def test_big_conv_onet_like():
    completr = CompleTr(
        encoder_type="unet3d",
        decoder_type="resnet",
        decoder_kwargs={"hidden_dim": 256, "cross_attn": False},
        multires=False,
    )

    big_conv_onet = ConvONet(arch="big_conv_onet_grid")

    unet1 = cast(nn.Module, cast(Any, completr.encoder)[1])
    unet2 = cast(nn.Module, cast(Any, big_conv_onet.encoder).unet3d)
    for (n1, p1), (n2, p2) in zip(unet1.named_parameters(), unet2.named_parameters(), strict=False):
        assert p1.size() == p2.size(), f"Size mismatch: {n1} {n2}"

    resnet1 = cast(nn.Module, completr.decoder)
    resnet2 = cast(nn.Module, big_conv_onet.decoder)
    for (n1, p1), (n2, p2) in zip(resnet1.named_parameters(), resnet2.named_parameters(), strict=False):
        assert p1.size() == p2.size(), f"Size mismatch: {n1} {n2}"


def test_multires():
    completr = CompleTr(
        decoder_type="resnet", encoder_kwargs={"points": "single", "planes": None}, decoder_kwargs={"cross_attn": False}
    )

    inputs = torch.randn(1, 3000, 3)
    points = torch.randn(1, 2028, 3)

    completr(inputs, points)
