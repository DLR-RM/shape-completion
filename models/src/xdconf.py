from functools import partial
from typing import Any, cast

import torch
from pytorch3dunet.unet3d.utils import number_of_features_per_level
from torch import Tensor, nn
from torch.nn.functional import grid_sample, interpolate, max_pool2d, max_pool3d
from torch_scatter import scatter

from utils import coordinates_to_index, points_to_coordinates

from .transformer import DecoderBlock, EncoderBlock
from .utils import get_activation, get_norm, grid_sample_2d, grid_sample_3d

XDCONV_TYPE = tuple[Tensor | None, list[Tensor] | None, Tensor | None]


def _as_tensor(value: Any) -> Tensor:
    return cast(Tensor, value)


def _get_norm(name: str | None, num_channels: int, dim: int, **kwargs: Any) -> nn.Module | None:
    if name is None:
        return None
    return get_norm(cast(Any, name), num_channels, dim=dim, **kwargs)


def _get_activation(name: str, **kwargs: Any) -> nn.Module:
    return get_activation(cast(Any, name), **kwargs)


def _sum_tensors(tensors: list[Tensor]) -> Tensor:
    if not tensors:
        raise ValueError("Expected at least one tensor")
    total = tensors[0]
    for tensor in tensors[1:]:
        total = total + tensor
    return total


def pytorch_sample(
    plane_or_grid_feature: Tensor,
    points: Tensor,
    padding: float = 0.1,
    plane: str | None = None,
    sample_mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
    custom_grid_sampling: bool = False,
) -> Tensor:
    sample_fn = grid_sample
    if custom_grid_sampling:
        sample_fn = grid_sample_3d if plane is None else grid_sample_2d
    coords = _as_tensor(points_to_coordinates(points, max_value=1 + padding, plane=plane))
    grid = 2 * coords - 1
    grid = grid[:, :, None, None] if plane is None else grid[:, :, None]
    voxel_features = sample_fn(plane_or_grid_feature, grid, sample_mode, padding_mode, align_corners)
    return voxel_features.view(plane_or_grid_feature.size(0), plane_or_grid_feature.size(1), -1)  # (B, C, N)


def pytorch_scatter(
    point_feature: Tensor,
    points: Tensor,
    res: int = 32,
    padding: float = 0.1,
    plane: str | None = None,
    scatter_type: str = "mean",
) -> Tensor:
    coords = _as_tensor(points_to_coordinates(points, max_value=1 + padding, plane=plane))
    index = _as_tensor(coordinates_to_index(coords, res)).unsqueeze(1)
    dim_size = res**3 if plane is None else res**2
    plane_or_grid_feature = scatter(src=point_feature, index=index, dim_size=dim_size, reduce=scatter_type)
    if plane is None:
        return plane_or_grid_feature.view(point_feature.size(0), -1, res, res, res)  # (B, C, D, H, W)
    return plane_or_grid_feature.view(point_feature.size(0), -1, res, res)  # (B, C, H, W)


def xdconv_maxpool(x: XDCONV_TYPE, kernel_size: int = 2, stride: int | None = None) -> XDCONV_TYPE:
    points_f, planes_f, grid_f = x
    if planes_f is not None:
        planes_f = [max_pool2d(pf, kernel_size, stride) for pf in planes_f]
    if grid_f is not None:
        grid_f = max_pool3d(grid_f, kernel_size, stride)
    return points_f, planes_f, grid_f


def xdconv_upsample(x: XDCONV_TYPE, scale_factor: int = 2, align_corners=True) -> XDCONV_TYPE:
    points_f, planes_f, grid_f = x
    if planes_f is not None:
        planes_f = [
            interpolate(pf, scale_factor=(scale_factor,) * 2, mode="bilinear", align_corners=align_corners)
            for pf in planes_f
        ]
    if grid_f is not None:
        grid_f = interpolate(grid_f, scale_factor=(scale_factor,) * 3, mode="trilinear", align_corners=align_corners)
    return points_f, planes_f, grid_f


def xdconv_cat(x: XDCONV_TYPE, y: XDCONV_TYPE) -> XDCONV_TYPE:
    points_f_x, planes_f_x, grid_f_x = x
    points_f_y, planes_f_y, grid_f_y = y
    points_f = None
    if points_f_x is not None and points_f_y is not None:
        points_f = torch.cat([points_f_x, points_f_y], dim=1)
    planes_f = None
    if planes_f_x is not None and planes_f_y is not None:
        planes_f = [torch.cat([pf_x, pf_y], dim=1) for pf_x, pf_y in zip(planes_f_x, planes_f_y, strict=False)]
    grid_f = None
    if grid_f_x is not None and grid_f_y is not None:
        grid_f = torch.cat([grid_f_x, grid_f_y], dim=1)
    return points_f, planes_f, grid_f


class DoubleConv(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int | None = None,
        kernel_size: int | tuple[int, int] = 3,
        dim: int = 3,
        activation: str | tuple[str, str] = "relu",
        norm: str | tuple[str, str] | None = None,
        layer_order: str = "nca",
        **kwargs,
    ):
        super().__init__()
        assert dim in [1, 2, 3], f"Unsupported dimension: {dim}"
        conv = nn.Conv1d if dim == 1 else nn.Conv2d if dim == 2 else nn.Conv3d
        k1, k2 = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
        hidden_channels = hidden_channels or max(in_channels, out_channels // 2)
        activation1, activation2 = activation if isinstance(activation, (tuple, list)) else (activation, activation)
        norm1, norm2 = norm if isinstance(norm, (tuple, list)) else (norm, norm)

        norm_channels = in_channels if layer_order[0] == "n" else hidden_channels
        operations = {
            "n": _get_norm(norm1, norm_channels, dim, **kwargs.get("norm_kwargs", {})),
            "c": conv(
                in_channels,
                hidden_channels,
                k1,
                padding=0 if dim == 1 or k1 == 1 or k1 % 2 == 0 else 1,
                bias=norm is None,
            ),
            "a": _get_activation(activation1, **kwargs.get("activation_kwargs", {})),
        }
        self.extend(operations[o] for o in layer_order if operations[o] is not None)

        norm_channels = hidden_channels if layer_order[0] == "n" else out_channels
        operations = {
            "n": _get_norm(norm2, norm_channels, dim, **kwargs.get("norm_kwargs", {})),
            "c": conv(
                hidden_channels,
                out_channels,
                k2,
                padding=0 if dim == 1 or k2 == 1 or k2 % 2 == 0 else 1,
                bias=norm is None,
            ),
            "a": _get_activation(activation2, **kwargs.get("activation_kwargs", {})),
        }
        self.extend(operations[o] for o in layer_order if operations[o] is not None)


class MaxPoolXD(nn.Module):
    def __init__(self, kernel_size: int = 2, stride: int | None = None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: XDCONV_TYPE, kernel_size: int | None = None, stride: int | None = None) -> XDCONV_TYPE:
        return xdconv_maxpool(x, kernel_size or self.kernel_size, stride or self.stride)


class UpSampleXD(nn.Module):
    def __init__(self, scale_factor: int = 2, align_corners: bool = True):
        super().__init__()
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def forward(
        self, x: XDCONV_TYPE, scale_factor: int | None = None, align_corners: bool | None = None
    ) -> XDCONV_TYPE:
        return xdconv_upsample(x, scale_factor or self.scale_factor, align_corners or self.align_corners)


class EncoderBlockXD(EncoderBlock):
    def forward(self, src: Tensor) -> Tensor:
        return super().forward(src.transpose(1, 2)).transpose(1, 2)


class DecoderBlockXD(DecoderBlock):
    def forward(self, src: Tensor, memory: Tensor) -> Tensor:
        return super().forward(src.transpose(1, 2), memory.transpose(1, 2)).transpose(1, 2)


class XDConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int | None = None,
        kernel_size: int = 3,
        activation: str = "relu",
        norm: str | None = None,
        padding: float = 0.1,
        points: str | None = "single",
        planes: tuple[str, ...] | None = None,
        grid: str | None = None,
        fuse_points: bool = True,
        fuse_planes: bool = True,
        fuse_grid: bool = True,
        planes_res: int | None = None,
        grid_res: int | None = None,
        layer_order: str = "nca",
        scatter_type: str = "mean",
        sample_type: str = "bilinear",
        custom_grid_sampling: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.points = points
        self.planes = planes
        self.grid = grid
        self.fuse_points = fuse_points
        self.fuse_planes = fuse_planes
        self.fuse_grid = fuse_grid
        self.planes_res = planes_res
        self.grid_res = grid_res
        self.scatter = partial(pytorch_scatter, scatter_type=scatter_type)
        self.sample = partial(pytorch_sample, sample_mode=sample_type, custom_grid_sampling=custom_grid_sampling)

        # Backward compatibility: legacy configs/tests sometimes pass booleans for grid mode.
        if isinstance(grid, bool):
            grid = "single" if grid else None
            self.grid = grid

        assert points or planes or grid, "At least one of 'points', 'planes' or 'grid' must be selected"
        assert points in [None, "single", "double", "self_attn"], f"Unsupported points: {points}"
        assert grid in [None, "single", "double"], f"Unsupported grid: {grid}"
        norm = norm if "n" in layer_order else None
        self.conv1d = None
        if points:
            if points == "single":
                norm_channels = in_channels if layer_order[0] == "n" else out_channels
                operations = {
                    "n": _get_norm(norm, norm_channels, dim=1, **kwargs.get("norm_kwargs", {})),
                    "c": nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=norm is None),
                    "a": _get_activation(activation, **kwargs.get("activation_kwargs", {})),
                }
                self.conv1d = nn.Sequential(*(operations[o] for o in layer_order if operations[o] is not None))
            elif points == "double":
                self.conv1d = DoubleConv(
                    in_channels,
                    out_channels,
                    hidden_channels,
                    kernel_size=1,
                    dim=1,
                    activation=activation,
                    norm=norm,
                    norm_kwargs=kwargs.get("norm_kwargs", {}),
                    layer_order=layer_order,
                )
            elif points == "self_attn":
                conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=norm is None)
                enc = EncoderBlockXD(
                    n_embd=out_channels,
                    n_head=kwargs.get("n_head", min(8, max(1, out_channels // 16))),
                    dropout=kwargs.get("dropout", 0),
                    hidden_layer_multiplier=kwargs.get("hidden_layer_multiplier", 4),
                )
                self.conv1d = nn.Sequential(conv, enc) if in_channels != out_channels else enc
        if planes:
            self.conv2d = DoubleConv(
                in_channels,
                out_channels,
                hidden_channels,
                kernel_size=kernel_size,
                dim=2,
                activation=activation,
                norm=norm,
                norm_kwargs=kwargs.get("norm_kwargs", {}),
                layer_order=layer_order,
            )
        if grid:
            if grid == "single":
                norm_channels = in_channels if layer_order[0] == "n" else out_channels
                operations = {
                    "n": _get_norm(norm, norm_channels, dim=3, **kwargs.get("norm_kwargs", {})),
                    "c": nn.Conv3d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        padding=0 if kernel_size == 1 or kernel_size % 2 == 0 else 1,
                        bias=norm is None,
                    ),
                    "a": _get_activation(activation, **kwargs.get("activation_kwargs", {})),
                }
                self.conv3d = nn.Sequential(*(operations[o] for o in layer_order if operations[o] is not None))
            else:
                self.conv3d = DoubleConv(
                    in_channels,
                    out_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    dim=3,
                    activation=activation,
                    norm=norm,
                    norm_kwargs=kwargs.get("norm_kwargs", {}),
                    layer_order=layer_order,
                )

    def forward(
        self,
        features: XDCONV_TYPE,
        inputs: Tensor | None = None,
        planes_res: int | None = None,
        grid_res: int | None = None,
    ) -> XDCONV_TYPE:
        points_f, planes_f, grid_f = features
        if self.points and points_f is None:
            assert grid_f is not None or planes_f is not None or inputs is not None, (
                "Either 'features[2]', 'features[1]' or 'inputs' must be provided"
            )
            if grid_f is not None:
                assert inputs is not None, "Point coordinates are required for grid sampling"
                points_f = self.sample(grid_f, inputs, self.padding)
            elif planes_f is not None:
                assert inputs is not None, "Point coordinates are required for plane sampling"
                assert self.planes is not None, "Plane names are required when planes are provided"
                sampled_points = [
                    self.sample(pf, inputs, self.padding, plane)
                    for pf, plane in zip(planes_f, self.planes, strict=False)
                ]
                points_f = _sum_tensors(sampled_points)
            elif inputs is not None:
                points_f = inputs.transpose(1, 2)
        if self.planes:
            if planes_f is None:
                assert inputs is not None, "Either 'features[1]' or 'inputs' must be provided"
                assert points_f is not None, "Point features are required to scatter plane features"
                plane_res_value = planes_res if planes_res is not None else self.planes_res
                assert plane_res_value is not None, "Either 'features[1]' or 'planes_res' must be provided"
                planes_f = [
                    self.scatter(points_f, inputs, plane_res_value, self.padding, plane=plane) for plane in self.planes
                ]
            else:
                if isinstance(planes_f, Tensor):
                    planes_f = [planes_f]
                assert len(planes_f) == len(self.planes), "The number of planes must match"
        if self.grid and grid_f is None:
            assert inputs is not None, "Either 'features[2]' or 'inputs' must be provided"
            assert points_f is not None, "Point features are required to scatter grid features"
            grid_res_value = grid_res if grid_res is not None else self.grid_res
            assert grid_res_value is not None, "Either 'features[2]' or 'grid_res' must be provided"
            grid_f = self.scatter(points_f, inputs, grid_res_value, self.padding)

        points_f_from_planes_f, points_f_from_grid_f = None, None

        if self.points and points_f is not None:
            assert self.conv1d is not None, "Point convolution module must be initialized when points are enabled"
            points_f = self.conv1d(points_f)
        if self.planes and planes_f is not None:
            assert self.conv2d is not None, "Plane convolution module must be initialized when planes are enabled"
            planes_f = [self.conv2d(pf) for pf in planes_f]
            if self.points and inputs is not None:
                if self.fuse_points:
                    assert self.planes is not None, "Plane names are required when planes are enabled"
                    points_f_from_planes_f = [
                        self.sample(pf, inputs, self.padding, plane)
                        for pf, plane in zip(planes_f, self.planes, strict=False)
                    ]
                if self.fuse_planes and points_f is not None:
                    assert self.planes is not None, "Plane names are required when planes are enabled"
                    points_f_to_planes_f = [
                        self.scatter(points_f, inputs, res=pf.size(-1), padding=self.padding, plane=plane)
                        for pf, plane in zip(planes_f, self.planes, strict=False)
                    ]
                    planes_f = [pf + ptp for pf, ptp in zip(planes_f, points_f_to_planes_f, strict=False)]
        if self.grid and grid_f is not None:
            assert self.conv3d is not None, "Grid convolution module must be initialized when grid is enabled"
            grid_f = self.conv3d(grid_f)
            if self.points and inputs is not None:
                if self.fuse_points:
                    points_f_from_grid_f = self.sample(grid_f, inputs, self.padding)
                if self.fuse_grid and points_f is not None:
                    grid_f_from_points_f = self.scatter(points_f, inputs, res=grid_f.size(-1), padding=self.padding)
                    grid_f = grid_f + grid_f_from_points_f

        if self.points:
            if self.fuse_points and points_f_from_planes_f is not None:
                fused_points = _sum_tensors(points_f_from_planes_f)
                points_f = points_f + fused_points if points_f is not None else fused_points
            if self.fuse_points and points_f_from_grid_f is not None:
                points_f = points_f + points_f_from_grid_f if points_f is not None else points_f_from_grid_f

        if self.points is None:
            points_f = None
        if self.planes is None:
            planes_f = None
        if self.grid is None:
            grid_f = None

        return cast(XDCONV_TYPE, (points_f, planes_f, grid_f))


class UNetXD(nn.Module):
    def __init__(
        self,
        channels: int | tuple[int, ...] | list[int],
        in_channels: int | None = None,
        out_channels: int | None = None,
        kernel_size: int = 3,
        layer_order: str = "nca",
        num_groups: int | None = 8,
        num_levels: int | None = None,
        activation: str = "relu",
        norm: str | None = "group",
        padding: float = 0.1,
        points: str | None = "single",
        planes: tuple[str, ...] | None = ("xy", "yz", "zx"),
        grid: str = "double",
        fuse_points: bool = True,
        fuse_planes: bool = True,
        fuse_grid: bool = True,
        planes_res: int | None = None,
        grid_res: int | None = None,
        return_encoder_features: bool = False,
        return_decoder_features: bool = False,
        custom_grid_sampling: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert num_groups if norm == "group" else True, "Number of groups must be provided for group normalization"
        assert not (channels is None and num_levels is None), "Either 'channels' or 'num_levels' must be provided"
        if isinstance(channels, int):
            channels = number_of_features_per_level(channels, num_levels=num_levels)
        self.fuse_points = fuse_points
        self.fuse_planes = fuse_planes
        self.fuse_grid = fuse_grid
        self.return_encoder_features = return_encoder_features
        self.return_decoder_features = return_decoder_features

        down_range = range(len(channels) - 1)
        hidden_channels = None
        self.encoders = nn.ModuleList(
            [
                XDConv(
                    channels[i],
                    channels[i + 1],
                    hidden_channels,
                    kernel_size,
                    activation,
                    norm,
                    padding,
                    points or None,
                    planes or None,
                    grid,
                    fuse_points,
                    fuse_planes,
                    fuse_grid,
                    planes_res=None if planes_res is None else planes_res // (2**i),
                    grid_res=None if grid_res is None else grid_res // (2**i),
                    layer_order=layer_order,
                    norm_kwargs={"num_groups": num_groups},
                    custom_grid_sampling=custom_grid_sampling,
                )
                for i in down_range
            ]
        )

        up_range = reversed(range(1, len(channels) - 1))
        self.decoders = nn.ModuleList(
            [
                XDConv(
                    channels[i] + channels[i + 1],
                    channels[i],
                    channels[i],
                    kernel_size,
                    activation,
                    norm,
                    padding,
                    points or None,
                    planes or None,
                    grid,
                    fuse_points,
                    fuse_planes,
                    fuse_grid,
                    planes_res=None if planes_res is None else planes_res // (2 ** (i - 1)),
                    grid_res=None if grid_res is None else grid_res // (2 ** (i - 1)),
                    layer_order=layer_order,
                    norm_kwargs={"num_groups": num_groups},
                    custom_grid_sampling=custom_grid_sampling,
                )
                for i in up_range
            ]
        )

        self.conv_in = None
        if in_channels is not None:
            self.conv_in = nn.Conv1d(in_channels, channels[0], kernel_size=1)

        self.conv_out_points: nn.Conv1d | None = None
        self.conv_out_planes: nn.Conv2d | None = None
        self.conv_out_grid: nn.Conv3d | None = None
        self.conv_out: list[nn.Module | None] = [None, None, None]
        if out_channels is not None:
            if points:
                self.conv_out_points = nn.Conv1d(channels[1], out_channels, kernel_size=1)
                self.conv_out[0] = self.conv_out_points
            if planes:
                self.conv_out_planes = nn.Conv2d(channels[1], out_channels, kernel_size=1)
                self.conv_out[1] = self.conv_out_planes
            if grid:
                self.conv_out_grid = nn.Conv3d(channels[1], out_channels, kernel_size=1)
                self.conv_out[2] = self.conv_out_grid

    def _conv_out(self, x: XDCONV_TYPE) -> XDCONV_TYPE:
        if all(module is None for module in self.conv_out):
            return x
        points, planes, grid = x
        if points is not None and self.conv_out[0] is not None:
            points = cast(nn.Module, self.conv_out[0])(points)
        if planes is not None and self.conv_out[1] is not None:
            conv2d = cast(nn.Module, self.conv_out[1])
            planes = [cast(Tensor, conv2d(pf)) for pf in planes]
        if grid is not None and self.conv_out[2] is not None:
            grid = cast(Tensor, cast(nn.Module, self.conv_out[2])(grid))
        return points, planes, grid

    def _set_fuse(self, x: XDCONV_TYPE) -> XDCONV_TYPE:
        points, planes, grid = x
        if not self.fuse_points:
            points = None
        if not self.fuse_planes:
            planes = None
        if not self.fuse_grid:
            grid = None
        return points, planes, grid

    def encode(self, inputs: Tensor, **kwargs) -> list[XDCONV_TYPE]:
        x = None
        if self.conv_in is not None:
            x = self.conv_in(inputs.transpose(1, 2))
        x = (x, None, None)
        x = self.encoders[0](x, inputs)
        enc_feature = [x]
        x = self._set_fuse(x)
        for encoder in self.encoders[1:]:
            x = xdconv_maxpool(x)
            x = encoder(x, inputs)
            enc_feature.insert(0, x)
            x = self._set_fuse(x)

        return enc_feature

    def decode(self, enc_feature: list[XDCONV_TYPE], inputs: Tensor, **kwargs) -> tuple[XDCONV_TYPE, list[XDCONV_TYPE]]:
        dec_feature = list()
        x = enc_feature[0]
        for decoder, f in zip(self.decoders, enc_feature[1:], strict=False):
            x = self._set_fuse(x)
            x = decoder(xdconv_cat(f, xdconv_upsample(x)), inputs)
            dec_feature.append(x)
        return self._conv_out(x), dec_feature

    def forward(
        self, inputs: Tensor, **kwargs
    ) -> XDCONV_TYPE | tuple[XDCONV_TYPE, list[XDCONV_TYPE]] | tuple[XDCONV_TYPE, list[XDCONV_TYPE], list[XDCONV_TYPE]]:
        enc_feature = self.encode(inputs, **kwargs)
        x, dec_feature = self.decode(enc_feature, inputs, **kwargs)
        if self.return_encoder_features and self.return_decoder_features:
            return x, enc_feature, dec_feature
        elif self.return_encoder_features:
            return x, enc_feature
        elif self.return_decoder_features:
            return x, dec_feature
        return x
