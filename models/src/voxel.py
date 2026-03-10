from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3dunet.unet3d.model import UNet3D

from utils import make_3d_grid

from .grid import GridEncoder


class GridVoxelEncoder(GridEncoder):
    def __init__(
        self,
        dim: int = 1,
        c_dim: int = 32,
        unet: bool = False,
        unet_kwargs: dict[str, Any] | None = None,
        unet3d: bool = False,
        unet3d_kwargs: dict[str, Any] | None = None,
        plane_resolution: int | None = None,
        grid_resolution: int | None = None,
        feature_type: tuple[str, ...] = ("grid",),
        padding: float = 0.1,
        kernel_size: int = 3,
        leaky: bool = False,
        **kwargs,
    ):
        super().__init__(
            c_dim=c_dim,
            unet=unet,
            unet_kwargs=unet_kwargs,
            unet3d=unet3d,
            unet3d_kwargs=unet3d_kwargs,
            plane_resolution=plane_resolution,
            grid_resolution=grid_resolution,
            feature_type=feature_type,
            padding=padding,
            **kwargs,
        )

        self.actvn = F.relu if not leaky else lambda x: F.leaky_relu(x, negative_slope=0.2)
        self.conv_in = nn.Conv3d(dim, c_dim, kernel_size, padding=1 if kernel_size > 1 else 0)

    def forward(self, points: torch.Tensor, **kwargs) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        x = points
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        elif len(x.shape) == 5:
            x = x.permute(0, 4, 1, 2, 3)
        else:
            raise ValueError

        c = self.actvn(self.conv_in(x))
        c = c.reshape(x.size(0), self.c_dim, -1)
        c = c.transpose(1, 2)

        grid_shape = (int(x.shape[2]), int(x.shape[3]), int(x.shape[4]))
        p = make_3d_grid((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5), grid_shape).expand(x.size(0), -1, -1).to(x.device)
        return self.generate_feature(c, self.get_index_dict(p))


class VoxelEncoder(nn.Module):
    def __init__(self, dim=3, c_dim=128):
        super().__init__()
        self.actvn = F.relu

        self.conv_in = nn.Conv3d(1, 32, 3, padding=1)

        self.conv_0 = nn.Conv3d(32, 64, 3, padding=1, stride=2)
        self.conv_1 = nn.Conv3d(64, 128, 3, padding=1, stride=2)
        self.conv_2 = nn.Conv3d(128, 256, 3, padding=1, stride=2)
        self.conv_3 = nn.Conv3d(256, 512, 3, padding=1, stride=2)
        self.fc = nn.Linear(512 * 2 * 2 * 2, c_dim)

    def forward(self, points: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        batch_size = points.size(0)

        x = points.unsqueeze(1)
        net = self.conv_in(x)
        net = self.conv_0(self.actvn(net))
        net = self.conv_1(self.actvn(net))
        net = self.conv_2(self.actvn(net))
        net = self.conv_3(self.actvn(net))

        hidden = net.view(batch_size, 512 * 2 * 2 * 2)
        c = self.fc(self.actvn(hidden))

        return {"grid": c.permute(0, 1, 4, 3, 2)}


class UNet3DEncoder(nn.Module):
    def __init__(self, dim: int = 1, c_dim: int = 32, unet3d_kwargs: dict[str, Any] | None = None):
        super().__init__()

        self.dim = dim
        self.c_dim = c_dim
        self.actvn = F.relu
        self.conv_in = nn.Conv3d(dim, c_dim, 3, padding=1)
        self.unet3d = UNet3D(in_channels=c_dim, out_channels=c_dim, **(unet3d_kwargs or {}))

    def forward(self, points: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        x = points
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        elif len(x.shape) == 5:
            x = x.permute(0, 4, 1, 2, 3)
        else:
            raise ValueError

        c = self.actvn(self.conv_in(x))
        c = self.unet3d(c)

        return {"grid": c.permute(0, 1, 4, 3, 2)}


class IFNetEncoder(nn.Module):
    def __init__(self, c_dim: int = 32, pool: bool = True):
        super().__init__()

        self.conv_in = nn.Conv3d(1, 16, 3, padding=1)
        self.conv_0 = nn.Conv3d(16, 32, 3, padding=1)
        self.conv_0_1 = nn.Conv3d(32, 32, 3, padding=1)
        self.conv_1 = nn.Conv3d(32, 64, 3, padding=1)
        self.conv_1_1 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv_2 = nn.Conv3d(64, c_dim, 3, padding=1)

        self.conv_in_bn = nn.BatchNorm3d(16)
        self.conv0_1_bn = nn.BatchNorm3d(32)
        self.conv1_1_bn = nn.BatchNorm3d(64)

        self.actvn = nn.ReLU()
        self.maxpool = nn.MaxPool3d(2) if pool else lambda x: x

    def forward(self, points: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        x = points
        x = x.unsqueeze(1)

        net = self.actvn(self.conv_in(x))
        net = self.conv_in_bn(net)
        net = self.maxpool(net)

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        net = self.maxpool(net)

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)

        net = self.conv_2(net)

        return {"grid": net.permute(0, 1, 4, 3, 2)}


class CoordVoxelEncoder(nn.Module):
    def __init__(self, dim=3, c_dim=128):
        super().__init__()
        self.actvn = F.relu

        self.conv_in = nn.Conv3d(4, 32, 3, padding=1)

        self.conv_0 = nn.Conv3d(32, 64, 3, padding=1, stride=2)
        self.conv_1 = nn.Conv3d(64, 128, 3, padding=1, stride=2)
        self.conv_2 = nn.Conv3d(128, 256, 3, padding=1, stride=2)
        self.conv_3 = nn.Conv3d(256, 512, 3, padding=1, stride=2)
        self.fc = nn.Linear(512 * 2 * 2 * 2, c_dim)

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device

        coord1 = torch.linspace(-0.5, 0.5, x.size(1)).to(device)
        coord2 = torch.linspace(-0.5, 0.5, x.size(2)).to(device)
        coord3 = torch.linspace(-0.5, 0.5, x.size(3)).to(device)

        coord1 = coord1.view(1, -1, 1, 1).expand_as(x)
        coord2 = coord2.view(1, 1, -1, 1).expand_as(x)
        coord3 = coord3.view(1, 1, 1, -1).expand_as(x)

        coords = torch.stack([coord1, coord2, coord3], dim=1)

        x = x.unsqueeze(1)
        net = torch.cat([x, coords], dim=1)
        net = self.conv_in(net)
        net = self.conv_0(self.actvn(net))
        net = self.conv_1(self.actvn(net))
        net = self.conv_2(self.actvn(net))
        net = self.conv_3(self.actvn(net))

        hidden = net.view(batch_size, 512 * 2 * 2 * 2)
        c = self.fc(self.actvn(hidden))

        return c
