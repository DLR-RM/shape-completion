from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from utils import points_to_coordinates, setup_logger

from .mixins import MultiEvalMixin, MultiLossMixin, PredictMixin
from .model import Model
from .xdconf import XDCONV_TYPE, XDConv, pytorch_scatter

logger = setup_logger(__name__)


def _debug_level_1(message: str) -> None:
    debug_level_1 = getattr(cast(Any, logger), "debug_level_1", None)
    if callable(debug_level_1):
        debug_level_1(message)
        return
    logger.debug(message)


def _require_tensor(value: Tensor | None, name: str) -> Tensor:
    if value is None:
        raise ValueError(f"Missing {name}")
    return value


def _as_tensor(value: object, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor")
    return value


class IFNet(MultiEvalMixin, MultiLossMixin, PredictMixin, Model):
    def __init__(
        self,
        hidden_dim: int = 256,
        padding: float = 0.1,
        displacements: None | bool | float = 0.0722,
        multires: bool = True,
        pvconv: bool = False,
        xdconv: bool = False,
        **kwargs,
    ):
        super().__init__()
        if isinstance(displacements, bool) and displacements:
            displacements = 0.0722

        feature_size = 128
        channels = [1, 16, 32, 64, 128, 128]
        if pvconv or (xdconv and kwargs.get("points", True)):
            channels[0] = 3
        if multires:
            feature_size = sum(channels)
        if displacements:
            feature_size *= 7

        if pvconv:
            if kwargs.get("vanilla", False):
                assert not displacements, "Vanilla PVConv does not support displacements"
                from libs import PVConv, Voxelization, get_voxel_coords, trilinear_devoxelize

                self.input_voxelization = Voxelization(128)
                self.trilinear_devoxelize = trilinear_devoxelize
                self.get_voxel_coords = get_voxel_coords
            else:
                from .pvcnn import PVConv
            sct = kwargs.get("scatter_type", "mean")
            fuse_point_feature = kwargs.get("fuse_point_feature", True)
            downsample = kwargs.get("downsample", None)
            _debug_level_1(
                f"Using PVConv (scatter_type={sct}, fuse_point_feature={fuse_point_feature}, downsample={downsample})"
            )
            self.conv_in = PVConv(
                channels[0],
                channels[1],
                3,
                resolution=128,
                fuse_point_feature=fuse_point_feature,
                downsample=downsample,
                scatter_type=sct,
            )
            self.conv_0 = PVConv(
                channels[1],
                channels[2],
                3,
                resolution=64,
                fuse_point_feature=fuse_point_feature,
                downsample=downsample,
                scatter_type=sct,
            )
            self.conv_1 = PVConv(
                channels[2],
                channels[3],
                3,
                resolution=32,
                fuse_point_feature=fuse_point_feature,
                downsample=downsample,
                scatter_type=sct,
            )
            self.conv_2 = PVConv(
                channels[3],
                channels[4],
                3,
                resolution=16,
                fuse_point_feature=fuse_point_feature,
                downsample=downsample,
                scatter_type=sct,
            )
            self.conv_3 = PVConv(
                channels[4],
                channels[5],
                3,
                resolution=8,
                fuse_point_feature=fuse_point_feature,
                downsample=downsample,
                scatter_type=sct,
            )
        elif xdconv:
            _debug_level_1("Using XDConv")
            self.conv_in = XDConv(
                channels[0],
                channels[1],
                3,
                norm="batch",
                padding=padding,
                points=kwargs.get("points", True),
                planes=kwargs.get("planes", None),
                grid=kwargs.get("grid", False),
                planes_res=128 * 2,
            )
            self.conv_0 = XDConv(
                channels[1],
                channels[2],
                3,
                norm="batch",
                padding=padding,
                points=kwargs.get("points", True),
                planes=kwargs.get("planes", None),
                grid=kwargs.get("grid", False),
                planes_res=128 * 2 // 2,
            )
            self.conv_1 = XDConv(
                channels[2],
                channels[3],
                3,
                norm="batch",
                padding=padding,
                points=kwargs.get("points", True),
                planes=kwargs.get("planes", None),
                grid=kwargs.get("grid", False),
                planes_res=128 * 2 // 4,
            )
            self.conv_2 = XDConv(
                channels[3],
                channels[4],
                3,
                norm="batch",
                padding=padding,
                points=kwargs.get("points", True),
                planes=kwargs.get("planes", None),
                grid=kwargs.get("grid", False),
                planes_res=128 * 2 // 8,
            )
            self.conv_3 = XDConv(
                channels[4],
                channels[5],
                3,
                norm="batch",
                padding=padding,
                points=kwargs.get("points", True),
                planes=kwargs.get("planes", None),
                grid=kwargs.get("grid", False),
                planes_res=128 * 2 // 16,
            )
            if kwargs.get("planes", None):
                self.maxpool_2d = nn.MaxPool2d(2)
            if kwargs.get("grid", False):
                self.maxpool_3d = nn.MaxPool3d(2)
        else:
            self.conv_in = nn.Conv3d(channels[0], channels[1], 3, padding=1)
            self.conv_0 = nn.Conv3d(channels[1], channels[2], 3, padding=1)
            self.conv_0_1 = nn.Conv3d(channels[2], channels[2], 3, padding=1)
            self.conv_1 = nn.Conv3d(channels[2], channels[3], 3, padding=1)
            self.conv_1_1 = nn.Conv3d(channels[3], channels[3], 3, padding=1)
            self.conv_2 = nn.Conv3d(channels[3], channels[4], 3, padding=1)
            self.conv_2_1 = nn.Conv3d(channels[4], channels[4], 3, padding=1)
            self.conv_3 = nn.Conv3d(channels[4], channels[5], 3, padding=1)
            self.conv_3_1 = nn.Conv3d(channels[5], channels[5], 3, padding=1)

            self.maxpool_3d = nn.MaxPool3d(2)

            self.conv_in_bn = nn.BatchNorm3d(channels[1])
            self.conv0_1_bn = nn.BatchNorm3d(channels[2])
            self.conv1_1_bn = nn.BatchNorm3d(channels[3])
            self.conv2_1_bn = nn.BatchNorm3d(channels[4])
            self.conv3_1_bn = nn.BatchNorm3d(channels[5])

        self.fc_0 = nn.Conv1d(feature_size, hidden_dim, 1)
        self.fc_1 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim, 1, 1)
        self.activation = nn.ReLU(inplace=True)

        self.displacements = None
        if displacements:
            displacement_value = float(displacements)
            offsets: list[list[float]] = [[0.0, 0.0, 0.0]]
            for axis in range(3):
                for direction in [-1.0, 1.0]:
                    offset = [0.0, 0.0, 0.0]
                    offset[axis] = direction * displacement_value
                    offsets.append(offset)
            self.displacements = torch.tensor(offsets)

        self.padding = padding
        self.multires = multires
        self.pvconv = pvconv
        self.xdconv = xdconv

    def forward(self, inputs: Tensor, points: Tensor, **kwargs) -> dict[str, Tensor]:
        feature = self.encode(inputs)
        return self.decode(points, feature)

    def _encode_conv(self, inputs: Tensor, **kwargs) -> list[Tensor]:
        feature_0 = inputs.unsqueeze(1)  # (B,1,res,res,res)

        net = self.activation(self.conv_in(feature_0))
        net = self.conv_in_bn(net)
        feature_1 = net
        net = self.maxpool_3d(net)

        net = self.activation(self.conv_0(net))
        net = self.activation(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        feature_2 = net
        net = self.maxpool_3d(net)

        net = self.activation(self.conv_1(net))
        net = self.activation(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        feature_3 = net
        net = self.maxpool_3d(net)

        net = self.activation(self.conv_2(net))
        net = self.activation(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        feature_4 = net
        net = self.maxpool_3d(net)

        net = self.activation(self.conv_3(net))
        net = self.activation(self.conv_3_1(net))
        net = self.conv3_1_bn(net)
        feature_5 = net

        return [feature_0, feature_1, feature_2, feature_3, feature_4, feature_5]

    def _encode_pvconv(self, inputs: Tensor, **kwargs) -> list[Tensor]:
        net = inputs.transpose(1, 2)
        if hasattr(self, "input_voxelization"):
            feature_0, _ = self.input_voxelization(net, inputs)
        else:
            feature_0 = pytorch_scatter(net, inputs, 128, self.padding)
        net = self.conv_in(net, inputs)
        feature_1 = net[1]
        net = self.conv_0(net, inputs)
        feature_2 = net[1]
        net = self.conv_1(net, inputs)
        feature_3 = net[1]
        net = self.conv_2(net, inputs)
        feature_4 = net[1]
        net = self.conv_3(net, inputs)
        feature_5 = net[1]

        return [feature_0, feature_1, feature_2, feature_3, feature_4, feature_5]

    def _encode_xdconv(self, inputs: Tensor, **kwargs) -> list[Tensor]:
        points: Tensor | None = inputs
        net: XDCONV_TYPE
        if inputs.dim() in [4, 5]:
            feature_0 = inputs
            if inputs.dim() == 4:
                feature_0 = inputs.unsqueeze(1)
            points = None
            net = self.conv_in([None, None, feature_0], points)
        elif inputs.dim() == 3:
            point_feature = inputs.transpose(1, 2)
            feature_0 = pytorch_scatter(point_feature, inputs, 128, self.padding)
            net = self.conv_in([point_feature, None, feature_0], inputs)
        else:
            raise ValueError(f"Invalid input dimension {inputs.dim()}")
        feature_1 = _require_tensor(net[2], "xdconv feature_1")
        net = self._maxpool_xdconv(net)
        net = self.conv_0(net, points)
        feature_2 = _require_tensor(net[2], "xdconv feature_2")
        net = self._maxpool_xdconv(net)
        net = self.conv_1(net, points)
        feature_3 = _require_tensor(net[2], "xdconv feature_3")
        net = self._maxpool_xdconv(net)
        net = self.conv_2(net, points)
        feature_4 = _require_tensor(net[2], "xdconv feature_4")
        net = self._maxpool_xdconv(net)
        net = self.conv_3(net, points)
        feature_5 = _require_tensor(net[2], "xdconv feature_5")

        return [feature_0, feature_1, feature_2, feature_3, feature_4, feature_5]

    def _maxpool_xdconv(self, x: XDCONV_TYPE) -> XDCONV_TYPE:
        points_f, planes_f, grid_f = x
        if planes_f is not None:
            maxpool_2d = getattr(self, "maxpool_2d", None)
            if not callable(maxpool_2d):
                raise RuntimeError("Missing maxpool_2d for plane features")
            planes_f = [_as_tensor(maxpool_2d(pf), "maxpool_2d output") for pf in planes_f]
        if grid_f is not None:
            grid_f = self.maxpool_3d(grid_f)
        return points_f, planes_f, grid_f

    def encode(self, inputs: Tensor, **kwargs) -> list[Tensor]:
        if self.pvconv:
            return self._encode_pvconv(inputs.float(), **kwargs)
        elif self.xdconv:
            return self._encode_xdconv(inputs.float(), **kwargs)
        else:
            return self._encode_conv(inputs.float(), **kwargs)

    def decode(self, points: Tensor, feature: list[Tensor], **kwargs) -> dict[str, Tensor]:
        sampled_feature: Tensor
        if hasattr(self, "trilinear_devoxelize") and hasattr(self, "get_voxel_coords"):
            sampled_feature = self._get_pvconv_features(points, feature)
        else:
            coords = points_to_coordinates(points, max_value=1 + self.padding)
            if not isinstance(coords, Tensor):
                raise TypeError("points_to_coordinates(points) must return a tensor")
            flip_coords = not (self.pvconv or self.xdconv)
            if self.xdconv:
                conv_in = cast(XDConv, self.conv_in)
                flip_coords = not conv_in.points
            if flip_coords:
                coords = torch.flip(coords, dims=[-1])  # grid_sample expects zyx
            norm_coords = 2 * coords - 1  # [-1, 1]
            sample_coords: Tensor = norm_coords[:, None, None]

            if self.displacements is not None:
                displaced_coords = [sample_coords + d.to(sample_coords.device) for d in self.displacements]
                sample_coords = torch.cat(displaced_coords, dim=2)  # (B,1,7,N,3)

            if self.multires:
                sampled_feature = torch.cat(
                    [F.grid_sample(f, sample_coords, padding_mode="zeros", align_corners=True) for f in feature], dim=1
                )  # (B,1+16+32+64+128+128,7,N)
            else:
                sampled_feature = F.grid_sample(
                    feature[-1], sample_coords, padding_mode="zeros", align_corners=True
                )  # (B,128,7,N)
        sampled_feature = sampled_feature.view(sampled_feature.size(0), -1, sampled_feature.size(-1))  # (B,128*7,N)

        net = self.activation(self.fc_0(sampled_feature))
        net = self.activation(self.fc_1(net))
        net = self.activation(self.fc_2(net))

        out = self.fc_out(net).squeeze(1)

        return {"logits": out}

    def _get_pvconv_features(self, points: Tensor, features: list[Tensor]) -> Tensor:
        devoxelized_features: list[Tensor] = []
        for feature in features:
            devoxelized_feature = self.trilinear_devoxelize(
                feature,
                self.get_voxel_coords(points.transpose(1, 2), feature.size(-1)),
                feature.size(-1),
                self.training,
            )
            if not isinstance(devoxelized_feature, Tensor):
                raise TypeError("trilinear_devoxelize must return a tensor")
            devoxelized_features.append(devoxelized_feature)
        return torch.cat(devoxelized_features, dim=1)

    def loss(
        self,
        data: dict[str, list[str] | Tensor],
        regression: bool = False,
        name: str | None = None,
        reduction: str | None = "mean",
    ) -> Tensor:
        return super().loss(data, regression, name, reduction)["occ_loss"]
