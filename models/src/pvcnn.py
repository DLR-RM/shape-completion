from typing import Any, cast

from torch import Tensor, nn

from utils import setup_logger

from .grid import GridEncoder
from .utils import get_activation, get_norm
from .xdconf import DoubleConv, pytorch_sample, pytorch_scatter

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


class PVConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        resolution: int = 32,
        fuse_point_feature: bool = True,
        fuse_voxel_feature: bool = False,
        skip: bool = False,
        downsample: str | None = None,
        mlp: bool = False,
        activation: str = "leaky_relu",
        norm: str | None = "batch",
        padding: float = 0.1,
        scatter_type: str = "mean",
        **kwargs,
    ):
        super().__init__()
        self.downsample = None
        assert downsample in [None, "max", "avg", "stride"]
        if downsample == "max":
            self.downsample = nn.MaxPool3d(2)
        elif downsample == "avg":
            self.downsample = nn.AvgPool3d(2)
        elif downsample == "stride":
            self.downsample = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False)
        assert scatter_type in ["mean", "max"]
        self.scatter_type = scatter_type

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution
        self.fuse_point_feature = fuse_point_feature
        self.fuse_voxel_feature = fuse_voxel_feature
        self.padding = padding

        kwargs["activation_kwargs"] = kwargs.get("activation_kwargs", {"negative_slope": 0.1})
        self.cnn = DoubleConv(
            in_channels, out_channels, kernel_size, dim=3, activation=activation, norm=norm, layer_order="cna", **kwargs
        )
        if mlp:
            self.mlp = DoubleConv(
                in_channels,
                out_channels,
                kernel_size=1,
                dim=1,
                activation=activation,
                norm=norm,
                layer_order="cna",
                **kwargs,
            )
        else:
            activation = "relu"
            self.mlp = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=norm is None),
                get_norm(cast(Any, norm), out_channels, dim=1, **kwargs.get("norm_kwargs", {})),
                get_activation(activation, **kwargs.get("activation_kwargs", {})),
            )

        self.skip = skip
        self.skip_point = None
        self.skip_voxel = None
        if skip:
            if fuse_point_feature:
                self.skip_point = nn.Identity()
                if in_channels != out_channels:
                    self.skip_point = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            if fuse_voxel_feature:
                self.skip_voxel = nn.Identity()
                if in_channels != out_channels:
                    self.skip_voxel = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, features: Tensor | tuple[Tensor | None, Tensor | None], inputs: Tensor) -> tuple[Tensor, Tensor]:
        if isinstance(features, tuple):
            prev_point_feature, prev_voxel_feature = features
        else:
            prev_point_feature, prev_voxel_feature = features, None
        point_feature_in = _require_tensor(prev_point_feature, "prev_point_feature")
        if prev_voxel_feature is None:
            logger.debug("First PVConv layer")
            prev_voxel_feature = pytorch_scatter(
                point_feature_in, inputs, self.resolution, self.padding, scatter_type=self.scatter_type
            )
        else:
            logger.debug("Subsequent PVConv layer")
            if self.fuse_voxel_feature or (not self.fuse_point_feature and not self.fuse_voxel_feature):
                if self.resolution != prev_voxel_feature.size(-1):
                    assert self.downsample is not None, "Downsampling is required for change of resolution"
                    logger.debug("Downsample voxel feature")
                    prev_voxel_feature = self.downsample(prev_voxel_feature)
                    assert self.resolution == prev_voxel_feature.size(-1), "Unsupported change of resolution"
            else:
                prev_voxel_feature = pytorch_scatter(
                    point_feature_in, inputs, self.resolution, self.padding, scatter_type=self.scatter_type
                )

        voxel_feature_in = _require_tensor(prev_voxel_feature, "prev_voxel_feature")
        point_feature = self.mlp(point_feature_in)
        voxel_feature = self.cnn(voxel_feature_in)

        point_feature_out = point_feature
        voxel_feature_out = voxel_feature

        if self.fuse_point_feature:
            logger.debug("Fuse point feature")
            sampled_feature = pytorch_sample(voxel_feature, inputs, self.padding)
            point_feature_out = point_feature_out + sampled_feature

        if self.fuse_voxel_feature:
            logger.debug("Fuse voxel feature")
            scattered_feature = pytorch_scatter(
                point_feature, inputs, self.resolution, self.padding, scatter_type=self.scatter_type
            )
            voxel_feature_out = voxel_feature_out + scattered_feature

        if self.skip:
            logger.debug("Skip connection")
            if self.skip_point is not None:
                point_feature_out = point_feature_out + self.skip_point(point_feature_in)
            if self.skip_voxel is not None:
                voxel_feature_out = voxel_feature_out + self.skip_voxel(voxel_feature_in)

        return point_feature_out, voxel_feature_out


class PVCNN(nn.Module):
    def __init__(
        self,
        channels: tuple[int, ...] = (3, 64, 256, 512),
        resolutions: tuple[int, ...] = (32, 16, 8),
        norm: str = "batch",
        activation: str = "leaky_relu",
        padding: float = 0.1,
        use_vanilla: bool = False,
        resnet: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert len(resolutions) == len(channels) - 1, "Number of resolutions must be one less than number of channels"
        if len(set(resolutions)) > 1:
            fuse_point_feature = False
            fuse_voxel_feature = False
            if kwargs.get("fuse_point_feature", True):
                _debug_level_1("Fuse point feature")
                fuse_point_feature = True
            if kwargs.get("fuse_voxel_feature", False):
                _debug_level_1("Fuse voxel feature")
                fuse_voxel_feature = True
            if not fuse_point_feature and not fuse_voxel_feature:
                assert not use_vanilla, "PVCNN with vanilla PVConv does not support multires with downsampling"
                kwargs["downsample"] = kwargs.get("downsample", "max")
            if len(set(resolutions)) > 1 and fuse_voxel_feature:
                logger.warning("Fuse multires voxel feature requires downsampling. Using max pooling")
                kwargs["downsample"] = kwargs.get("downsample", "max")
            if kwargs.get("downsample") is not None or fuse_voxel_feature:
                assert all([resolutions[i] == resolutions[i - 1] // 2 for i in range(1, len(resolutions))])

        self.pvconvs = nn.ModuleList()
        skip = kwargs.pop("skip", resnet)
        if use_vanilla:
            assert not resnet, "Vanilla PVConv does not support ResNet"
            try:
                from libs.pvconv import PVConv as _PVConv  # pyright: ignore[reportMissingImports]

                for i in range(len(resolutions)):
                    self.pvconvs.append(
                        _PVConv(
                            in_channels=channels[i],
                            out_channels=channels[i + 1],
                            kernel_size=3,
                            resolution=resolutions[i],
                        )
                    )
            except ImportError:
                logger.exception("'libs/pvcnn' is not installed. Using 'PVConv' from 'models/src/pvcnn.py'")
                use_vanilla = False

        if not use_vanilla:
            for i in range(len(resolutions)):
                self.pvconvs.append(
                    PVConv(
                        channels[i],
                        channels[i + 1],
                        resolution=resolutions[i],
                        skip=skip,
                        activation=activation,
                        norm=norm,
                        padding=padding,
                        **kwargs,
                    )
                )

    def forward(self, inputs: Tensor, **kwargs) -> list[tuple[Tensor, Tensor]]:
        feature_list = list()
        net = (inputs.transpose(1, 2), None)
        for pvconv in self.pvconvs:
            net = pvconv(net, inputs)
            feature_list.append(net)
        return feature_list


class GridPVCNN(nn.Module):
    def __init__(
        self,
        channels: tuple[int, ...],
        resolutions: tuple[int, ...],
        norm: str = "batch",
        activation: str = "leaky_relu",
        padding: float = 0.1,
        use_vanilla: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.multires = len(set(resolutions)) > 1
        if self.multires:
            assert all([resolutions[i] == resolutions[i - 1] // 2 for i in range(1, len(resolutions))])
            assert kwargs.get("unet") is None and kwargs.get("unet3d") is None, (
                "UNets are not supported for multires PVCNN"
            )
        else:
            self.grid = GridEncoder(c_dim=channels[-1], padding=padding, **kwargs)

        self.pvcnn = PVCNN(
            channels=channels,
            resolutions=resolutions,
            norm=norm,
            activation=activation,
            padding=padding,
            use_vanilla=use_vanilla,
            **kwargs,
        )

    def forward(self, inputs: Tensor, **kwargs) -> dict[str, Tensor | list[Tensor]]:
        features = self.pvcnn(inputs)
        if self.multires:
            return {"grid": [feature[1] for feature in features]}
        coords = inputs[:, :, :3]
        index_dict = self.grid.get_index_dict(coords)
        feature = self.grid.generate_feature(features[-1][0], index_dict)
        return feature
