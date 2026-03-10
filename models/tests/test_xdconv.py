from typing import Any, cast

import pytest
import torch
from pytorch3dunet.unet3d.model import UNet3D
from torch import nn

from ..src.pvcnn import PVConv
from ..src.xdconf import UNetXD, XDConv


class TestXDConv:
    def test_init(self):
        XDConv(in_channels=32, out_channels=32)

    def test_forward(self):
        xdconv = XDConv(in_channels=8, out_channels=16)
        x = (torch.rand(1, 8, 100), None, None)
        p = None
        x = xdconv(x, p)
        points, planes, grid = x
        assert points.shape == (1, 16, 100)
        assert planes is None
        assert grid is None

        xdconv = XDConv(in_channels=8, out_channels=16, planes=("xy",))
        x = (torch.rand(1, 8, 100), torch.rand(1, 8, 64, 64), None)
        p = torch.rand(1, 100, 3)
        x = xdconv(x, p)
        points, planes, grid = x
        assert points.shape == (1, 16, 100)
        assert planes[0].shape == (1, 16, 64, 64)
        assert grid is None

        xdconv = XDConv(in_channels=8, out_channels=16, planes=("xy",), grid=cast(Any, True))
        x = (torch.rand(1, 8, 100), torch.rand(1, 8, 64, 64), torch.rand(1, 8, 32, 32, 32))
        p = torch.rand(1, 100, 3)
        x = xdconv(x, p)
        points, planes, grid = x
        assert points.shape == (1, 16, 100)
        assert planes[0].shape == (1, 16, 64, 64)
        assert grid.shape == (1, 16, 32, 32, 32)

    def test_from_points(self):
        xdconv = XDConv(
            in_channels=8, out_channels=16, planes=("xy",), grid=cast(Any, True), planes_res=64, grid_res=32
        )
        x = (torch.rand(1, 8, 100), None, None)
        p = torch.rand(1, 100, 3)
        x = xdconv(x, p)
        points, planes, grid = x
        assert points.shape == (1, 16, 100)
        assert planes[0].shape == (1, 16, 64, 64)
        assert grid.shape == (1, 16, 32, 32, 32)

    def test_grid_only(self):
        xdconv = XDConv(in_channels=8, out_channels=16, points=None, grid=cast(Any, True), grid_res=32)
        x = (None, None, torch.rand(1, 8, 32, 32, 32))
        p = None
        x = xdconv(x, p)
        points, planes, grid = x
        assert points is None
        assert planes is None
        assert grid.shape == (1, 16, 32, 32, 32)

    def test_pvconv_like(self):
        xdconv = XDConv(in_channels=8, out_channels=16, planes=None, grid=cast(Any, True), grid_res=32)
        x = (torch.rand(1, 8, 100), None, None)
        p = torch.rand(1, 100, 3)
        out_a = xdconv(x, p)
        points_a, planes, grid_a = out_a
        assert points_a.shape == (1, 16, 100)
        assert planes is None
        assert grid_a.shape == (1, 16, 32, 32, 32)

        pvconv = cast(
            Any, PVConv(in_channels=8, out_channels=16, fuse_voxel_feature=True, mlp=True, activation="relu", norm=None)
        )
        pvconv.mlp = xdconv.conv1d
        pvconv.cnn = xdconv.conv3d
        out_b = pvconv((x[0], None), p)
        points_b, grid_b = out_b
        assert points_b.shape == (1, 16, 100)
        assert grid_b.shape == (1, 16, 32, 32, 32)

        assert torch.allclose(points_a, points_b), "points not equal"
        assert torch.allclose(grid_a, grid_b), "grid not equal"

        xdconv.fuse_grid = False
        pvconv.fuse_voxel_feature = False
        out_a = xdconv(x, p)
        out_b = pvconv((x[0], None), p)
        points_a, planes, grid_a = out_a
        points_b, grid_b = out_b
        assert torch.allclose(points_a, points_b), "points not equal"
        assert torch.allclose(grid_a, grid_b), "grid not equal"

    def test_multi_planes(self):
        xdconv = XDConv(in_channels=8, out_channels=16, planes=("xy", "xz"))
        x = (torch.rand(1, 8, 100), [torch.rand(1, 8, 64, 64), torch.rand(1, 8, 64, 64)], None)
        p = torch.rand(1, 100, 3)
        x = xdconv(x, p)
        points, planes, grid = x
        assert points.shape == (1, 16, 100)
        assert planes[0].shape == (1, 16, 64, 64)
        assert planes[1].shape == (1, 16, 64, 64)
        assert grid is None

        assert torch.stack(planes, dim=-1).sum(dim=-1).shape == (1, 16, 64, 64)

    def test_multi_forward(self):
        xdconv = nn.ModuleList(
            [
                XDConv(in_channels=4, out_channels=8, planes=("xy",), grid=cast(Any, True)),
                XDConv(in_channels=8, out_channels=16, planes=("xy",), grid=cast(Any, True)),
            ]
        )
        x = (torch.rand(1, 4, 100), torch.rand(1, 4, 64, 64), torch.rand(1, 4, 32, 32, 32))
        p = torch.rand(1, 100, 3)
        for layer in xdconv:
            x = layer(x, p)
        points, planes, grid = x
        assert points.shape == (1, 16, 100)
        assert planes[0].shape == (1, 16, 64, 64)
        assert grid.shape == (1, 16, 32, 32, 32)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_self_attn(self):
        xdconv = XDConv(
            in_channels=8, out_channels=16, points="self_attn", planes=("xy", "xz", "yz"), grid=cast(Any, True)
        ).cuda()
        x = (torch.rand(1, 8, 100).cuda(), [torch.rand(1, 8, 64, 64).cuda()] * 3, torch.rand(1, 8, 32, 32, 32).cuda())
        p = torch.rand(1, 100, 3).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            x = xdconv(x, p)
        points, planes, grid = x
        assert points.shape == (1, 16, 100)
        assert planes[0].shape == (1, 16, 64, 64)
        assert planes[1].shape == (1, 16, 64, 64)
        assert planes[2].shape == (1, 16, 64, 64)
        assert grid.shape == (1, 16, 32, 32, 32)


class TestUNetXD:
    def test_init(self):
        unet1 = UNetXD(channels=(16, 32, 64))
        unet2 = UNetXD(channels=16, num_levels=3)

        for (n1, p1), (n2, p2) in zip(unet1.named_parameters(), unet2.named_parameters(), strict=False):
            assert p1.size() == p2.size(), f"shape mismatch: {n1} {n2}"

    def test_unet3d_like(self):
        unet3d = cast(Any, UNet3D(in_channels=16, out_channels=16, f_maps=32, num_levels=2, is_segmentation=False))
        unet3d.final_conv = None

        unetxd = UNetXD(channels=(16, 32, 64), points=None, planes=None)

        for (n1, p1), (n2, p2) in zip(unet3d.named_parameters(), unetxd.named_parameters(), strict=False):
            assert p1.size() == p2.size(), f"shape mismatch: {n1} {n2}"

    def test_forward(self):
        for fuse in [False, True]:
            unetxd = UNetXD(
                channels=(16, 32, 64, 128),
                in_channels=3,
                out_channels=16,
                planes=("xy", "xz", "yz"),
                grid=cast(Any, True),
                planes_res=64,
                grid_res=32,
                fuse_planes=fuse,
                fuse_grid=fuse,
                return_encoder_features=True,
                return_decoder_features=True,
            )
            p = torch.rand(1, 100, 3)
            x, _enc, _dec = unetxd(p)
            points, planes, grid = x
            assert points.shape == (1, 16, 100)
            assert planes[0].shape == (1, 16, 64, 64)
            assert planes[1].shape == (1, 16, 64, 64)
            assert planes[2].shape == (1, 16, 64, 64)
            assert grid.shape == (1, 16, 32, 32, 32)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_points_types(self):
        for points_type in [None, "single", "double", "self_attn"]:
            unetxd = UNetXD(
                channels=(16, 32, 64, 128),
                in_channels=3,
                out_channels=16,
                points=points_type,
                planes=("xy", "xz", "yz"),
                grid=cast(Any, True),
                planes_res=64,
                grid_res=32,
                return_encoder_features=True,
                return_decoder_features=True,
            ).cuda()
            p = torch.rand(1, 100, 3).cuda()
            with torch.autocast(
                device_type="cuda", dtype=torch.float16 if points_type == "self_attn" else torch.float32
            ):
                x, _enc, _dec = unetxd(p)
            points, planes, grid = x
            assert points is None if points_type is None else points.shape == (1, 16, 100)
            assert planes[0].shape == (1, 16, 64, 64)
            assert planes[1].shape == (1, 16, 64, 64)
            assert planes[2].shape == (1, 16, 64, 64)
            assert grid.shape == (1, 16, 32, 32, 32)
