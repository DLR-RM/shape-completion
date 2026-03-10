import logging

import pytest
import torch
from torch import Tensor

from ..src.pvcnn import PVCNN, PVConv, logger


class TestPVConv:
    def test_init(self):
        for in_channels in [3, 32]:
            for out_channels in [32, 64, 128, 256, 512]:
                for kernel_size in [1, 3]:
                    for res in [8, 16, 32]:
                        for padding in [0, 0.1]:
                            PVConv(in_channels, out_channels, kernel_size, res, padding=padding)

        for fuse_fuse_skip in [
            (False, False, False),
            (True, False, False),
            (True, True, False),
            (True, True, True),
            (False, True, False),
            (False, True, True),
            (False, False, True),
            (True, False, True),
        ]:
            fuse_point_feature, fuse_voxel_feature, skip = fuse_fuse_skip
            PVConv(3, 64, 3, 32, fuse_point_feature, fuse_voxel_feature, skip)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_forward(self):
        logger.setLevel(logging.DEBUG)
        points = torch.randn(16, 3000, 3).cuda() - 0.5
        feature = points.transpose(1, 2)
        for mlp in [True, False]:
            logger.debug(f"mlp={mlp}")
            for downsample in ["max", "avg", "stride"]:
                logger.debug(f"downsample={downsample}")
                for fuse_fuse_skip in [
                    (False, False, False),
                    (True, False, False),
                    (True, True, False),
                    (True, True, True),
                    (False, True, False),
                    (False, True, True),
                    (False, False, True),
                    (True, False, True),
                ]:
                    logger.debug(f"fuse_fuse_skip={fuse_fuse_skip}")
                    fuse_point_feature, fuse_voxel_feature, skip = fuse_fuse_skip
                    pvconv1 = (
                        PVConv(
                            in_channels=3,
                            out_channels=32,
                            kernel_size=3,
                            resolution=16,
                            fuse_point_feature=fuse_point_feature,
                            fuse_voxel_feature=fuse_voxel_feature,
                            skip=skip,
                            mlp=mlp,
                        )
                        .eval()
                        .cuda()
                    )
                    no_fusion = not fuse_point_feature and not fuse_voxel_feature
                    pvconv2 = (
                        PVConv(
                            in_channels=32,
                            out_channels=64,
                            kernel_size=3,
                            resolution=8,
                            fuse_point_feature=fuse_point_feature,
                            fuse_voxel_feature=fuse_voxel_feature,
                            skip=skip,
                            downsample=downsample if fuse_voxel_feature or no_fusion else None,
                            mlp=mlp,
                        )
                        .eval()
                        .cuda()
                    )
                    with torch.no_grad():
                        x = pvconv1(feature, points)
                        assert isinstance(x, tuple)
                        assert len(x) == 2
                        assert x[0].dim() == 3
                        assert x[1].dim() == 5
                        assert x[0].shape == (16, 32, 3000)
                        assert x[1].shape == (16, 32, 16, 16, 16)

                        x = pvconv2(x, points)
                        assert isinstance(x, tuple)
                        assert len(x) == 2
                        assert x[0].dim() == 3
                        assert x[1].dim() == 5
                        assert x[0].shape == (16, 64, 3000)
                        assert x[1].shape == (16, 64, 8, 8, 8)


class TestPVCNN:
    def test_init(self):
        for channels in [(3, 32, 32, 32, 32, 32), (3, 64, 256, 512)]:
            for resolutions in [(32, 32, 32, 32, 32), (32, 16, 8)]:
                if len(channels) - 1 == len(resolutions):
                    PVCNN(channels, resolutions)

        for fuse_fuse_skip in [
            (False, False, False),
            (True, False, False),
            (True, True, False),
            (True, True, True),
            (False, True, False),
            (False, True, True),
            (False, False, True),
            (True, False, True),
        ]:
            fuse_point_feature, fuse_voxel_feature, skip = fuse_fuse_skip
            PVCNN(fuse_point_feature=fuse_point_feature, fuse_voxel_feature=fuse_voxel_feature, skip=skip)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_forward(self):
        points = torch.randn(16, 3000, 3).cuda() - 0.5
        for unet3d in [False, True]:
            for fuse_fuse_skip in [
                (False, False, False),
                (True, False, False),
                (True, True, False),
                (True, True, True),
                (False, True, False),
                (False, True, True),
                (False, False, True),
                (True, False, True),
            ]:
                fuse_point_feature, fuse_voxel_feature, skip = fuse_fuse_skip
                pvcnn = (
                    PVCNN(
                        channels=(3, 32, 32, 32, 32, 32) if unet3d else (3, 64, 256, 512),
                        resolutions=(16, 16, 16, 16, 16) if unet3d else (32, 16, 8),
                        unet3d=unet3d,
                        unet3d_kwargs={"num_levels": 2, "f_maps": 32, "is_segmentation": False},
                        fuse_point_feature=fuse_point_feature,
                        fuse_voxel_feature=fuse_voxel_feature,
                        skip=skip,
                    )
                    .eval()
                    .cuda()
                )
                with torch.no_grad():
                    x = pvcnn(points)

                assert isinstance(x, list)
                if unet3d:
                    assert len(x) == 5
                    point_feat, grid_feat = x[-1]
                    assert isinstance(point_feat, Tensor)
                    assert isinstance(grid_feat, Tensor)
                    assert point_feat.shape == (16, 32, 3000)
                    assert grid_feat.shape == (16, 32, 16, 16, 16)
                else:
                    assert len(x) == 3
                    assert x[0][0].shape == (16, 64, 3000)
                    assert x[0][1].shape == (16, 64, 32, 32, 32)
                    assert x[1][0].shape == (16, 256, 3000)
                    assert x[1][1].shape == (16, 256, 16, 16, 16)
                    assert x[2][0].shape == (16, 512, 3000)
                    assert x[2][1].shape == (16, 512, 8, 8, 8)
