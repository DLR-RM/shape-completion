from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, padding: int = 1, bias: bool = False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=bias)


class ConvBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, norm: str = "group"):
        super().__init__()
        self.conv1 = conv3x3(in_planes, out_planes // 2)
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if norm == "batch":
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
            self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
            self.bn4 = nn.BatchNorm2d(in_planes)
        elif norm == "group":
            self.bn1 = nn.GroupNorm(32, in_planes)
            self.bn2 = nn.GroupNorm(32, int(out_planes / 2))
            self.bn3 = nn.GroupNorm(32, int(out_planes / 4))
            self.bn4 = nn.GroupNorm(32, in_planes)

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                self.bn4, nn.ReLU(), nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3


class HourGlass(nn.Module):
    def __init__(self, num_modules: int, depth: int, num_features: int, norm: str = "group"):
        super().__init__()

        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.norm = norm

        self._generate_network(depth)

    def _generate_network(self, level: int):
        self.add_module("b1_" + str(level), ConvBlock(self.features, self.features, norm=self.norm))

        self.add_module("b2_" + str(level), ConvBlock(self.features, self.features, norm=self.norm))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module("b2_plus_" + str(level), ConvBlock(self.features, self.features, norm=self.norm))

        self.add_module("b3_" + str(level), ConvBlock(self.features, self.features, norm=self.norm))

    def _forward(self, x: Tensor, level: int) -> Tensor:
        b1 = cast(nn.Module, getattr(self, f"b1_{level}"))
        b2 = cast(nn.Module, getattr(self, f"b2_{level}"))
        b3 = cast(nn.Module, getattr(self, f"b3_{level}"))

        # Upper branch
        up1 = x
        up1 = b1(up1)

        # Lower branch
        low1 = F.avg_pool2d(x, 2, stride=2)
        low1 = b2(low1)

        if level > 1:
            low2 = self._forward(low1, level - 1)
        else:
            low2 = low1
            b2_plus = cast(nn.Module, getattr(self, f"b2_plus_{level}"))
            low2 = b2_plus(low2)

        low3 = low2
        low3 = b3(low3)

        # NOTE: for newer PyTorch (1.3~), it seems that training results are degraded due to implementation diff in F.grid_sample
        # if the pretrained model behaves weirdly, switch with the commented line.
        # NOTE: I also found that "bicubic" works better.
        up2 = F.interpolate(low3, scale_factor=2, mode="bicubic", align_corners=True)
        # up2 = F.interpolate(low3, scale_factor=2, mode='nearest)

        return up1 + up2

    def forward(self, x: Tensor):
        return self._forward(x, self.depth)


class HGFilter(nn.Module):
    def __init__(
        self,
        num_modules: int = 4,
        num_hourglass: int = 2,
        hourglass_dim: int = 256,
        norm: str = "group",
        hg_down: str = "ave_pool",
    ):
        super().__init__()

        self.num_modules = num_modules
        self.num_hourglass = num_hourglass
        self.hourglass_dim = hourglass_dim
        self.norm = norm
        self.hg_down = hg_down

        # Base part
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)

        if norm == "batch":
            self.bn1 = nn.BatchNorm2d(64)
        elif norm == "group":
            self.bn1 = nn.GroupNorm(32, 64)

        if hg_down == "conv64":
            self.conv2 = ConvBlock(64, 64, norm)
            self.down_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        elif hg_down == "conv128":
            self.conv2 = ConvBlock(64, 128, norm)
            self.down_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        elif hg_down == "ave_pool":
            self.conv2 = ConvBlock(64, 128, norm)
        else:
            raise NameError("Unknown Fan Filter setting!")

        self.conv3 = ConvBlock(128, 128, norm)
        self.conv4 = ConvBlock(128, 256, norm)

        # Stacking part
        for hg_module in range(num_modules):
            self.add_module("m" + str(hg_module), HourGlass(1, num_hourglass, 256, norm))

            self.add_module("top_m_" + str(hg_module), ConvBlock(256, 256, norm))
            self.add_module("conv_last" + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            if norm == "batch":
                self.add_module("bn_end" + str(hg_module), nn.BatchNorm2d(256))
            elif norm == "group":
                self.add_module("bn_end" + str(hg_module), nn.GroupNorm(32, 256))

            self.add_module("l" + str(hg_module), nn.Conv2d(256, hourglass_dim, kernel_size=1, stride=1, padding=0))

            if hg_module < num_modules - 1:
                self.add_module("bl" + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module(
                    "al" + str(hg_module), nn.Conv2d(hourglass_dim, 256, kernel_size=1, stride=1, padding=0)
                )

    def forward(self, x: Tensor) -> tuple[list[Tensor], Tensor, Tensor]:
        x = F.relu(self.bn1(self.conv1(x)))
        tmpx = x
        if self.hg_down == "ave_pool":
            x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        elif self.hg_down in ["conv64", "conv128"]:
            x = self.conv2(x)
            x = self.down_conv2(x)
        else:
            raise NameError("Unknown Fan Filter setting!")

        normx = x

        x = self.conv3(x)
        x = self.conv4(x)

        previous = x

        outputs = list()
        for i in range(self.num_modules):
            hg_module = cast(nn.Module, getattr(self, f"m{i}"))
            top_module = cast(nn.Module, getattr(self, f"top_m_{i}"))
            conv_last = cast(nn.Module, getattr(self, f"conv_last{i}"))
            bn_end = cast(nn.Module, getattr(self, f"bn_end{i}"))
            pred_head = cast(nn.Module, getattr(self, f"l{i}"))
            hg = hg_module(previous)

            ll = hg
            ll = top_module(ll)

            ll = F.relu(bn_end(conv_last(ll)))

            # Predict heatmaps
            tmp_out = pred_head(ll)
            outputs.append(tmp_out)

            if i < self.num_modules - 1:
                bl = cast(nn.Module, getattr(self, f"bl{i}"))
                al = cast(nn.Module, getattr(self, f"al{i}"))
                ll = bl(ll)
                tmp_out_ = al(tmp_out)
                previous = previous + ll + tmp_out_

        return outputs, tmpx.detach(), normx
