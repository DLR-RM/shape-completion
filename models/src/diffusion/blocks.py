# From https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
import math
from collections.abc import Callable
from functools import partial
from typing import Any, cast

import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from torch import Tensor, nn

from ..transformer import MultiHeadAttention
from ..xdconf import XDCONV_TYPE, XDConv
from .utils import default, exists, get_convnd


class SinusoidalPositionEmbeddings(nn.Module):
    # Like https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/model.py#L12

    def __init__(self, dim: int, theta: int = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, time: Tensor) -> Tensor:
        half_dim = self.dim // 2
        embeddings = math.log(self.theta) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


def Upsample(dim: int, dim_out: int | None = None, ndim: int = 3, mode: str = "nearest") -> nn.Sequential:
    conv = get_convnd(ndim)
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode=mode, align_corners=None if mode == "nearest" else True),
        conv(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(
    dim: int, dim_out: int | None = None, ndim: int = 3, mode: str = "spd"
) -> nn.Conv3d | nn.Conv2d | nn.Conv1d | nn.Sequential:
    if mode == "spd":
        # Space-to-depth (SPD) Conv ("No More Strided Convolutions or Pooling": https://arxiv.org/abs/2208.03641)
        if ndim == 1:
            return nn.Sequential(
                Rearrange("b c (d p1) -> b (c p1) d", p1=2), nn.Conv1d(dim * 2**ndim, default(dim_out, dim), 1)
            )
        elif ndim == 2:
            return nn.Sequential(
                Rearrange("b c (d p1) (h p2) -> b (c p1 p2) d h", p1=2, p2=2),
                nn.Conv2d(dim * 2**ndim, default(dim_out, dim), 1),
            )
        elif ndim == 3:
            return nn.Sequential(
                Rearrange("b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w", p1=2, p2=2, p3=2),
                nn.Conv3d(dim * 2**ndim, default(dim_out, dim), 1),
            )
        else:
            raise NotImplementedError(f"SPD not implemented for {ndim}D")
    elif mode == "stride":
        conv = get_convnd(ndim)
        return conv(dim, default(dim_out, dim), 3, stride=2, padding=1)
    elif mode == "maxpool":
        maxpool = nn.MaxPool3d if ndim == 3 else nn.MaxPool2d if ndim == 2 else nn.MaxPool1d if ndim == 1 else None
        if maxpool is None:
            raise NotImplementedError(f"MaxPool not implemented for {ndim}D")
        conv = get_convnd(ndim)
        return nn.Sequential(maxpool(2), conv(dim, default(dim_out, dim), 3, padding=1))
    else:
        raise ValueError(f"Invalid method: {mode}")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, ndim: int = 3, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.d = dim
        self.scale = nn.Parameter(torch.ones(1, dim, *(1,) * ndim))
        self.register_parameter("scale", self.scale)

    def forward(self, x: Tensor) -> Tensor:
        norm_x = x.norm(2, dim=1, keepdim=True)
        rms_x = norm_x * self.d ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)
        return self.scale * x_normed


class WeightStandardizedConvNd(nn.Module):
    """Micro-Batch Training with Batch-Channel Normalization and Weight Standardization

    Paper: https://arxiv.org/abs/1903.10520
    Purportedly works synergistically with group normalization
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        ndim = kwargs.pop("ndim", 3)
        conv = get_convnd(ndim)
        self.conv = conv(*args, **kwargs)
        fn = F.conv3d if ndim == 3 else F.conv2d if ndim == 2 else F.conv1d if ndim == 1 else None
        self.pattern = "o ... -> o 1 1 1 1" if ndim == 3 else "o ... -> o 1 1 1" if ndim == 2 else "o ... -> o 1 1"

        if self.conv is None or fn is None:
            raise NotImplementedError(f"WeightStandardizedConv{ndim}d not implemented")
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        eps = 1e-7 if x.dtype == torch.float64 else 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.conv.weight
        mean = reduce(weight, self.pattern, "mean")
        var = reduce(weight, self.pattern, partial(torch.var, unbiased=False))
        weight = (weight - mean) * var.clamp(min=eps).rsqrt()

        return self.fn(
            x, weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups
        )


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        ndim: int = 3,
        groups: int = 8,
        use_weight_standardization: bool = False,
        use_rms_norm: bool = False,
        use_xdconv: bool = False,
        points: str | None = "single",
        grid: str | None = "single",
        dropout: float = 0.0,
    ):
        super().__init__()
        # Todo: Maybe use weight norm instead of group norm
        if use_weight_standardization:
            self.proj = WeightStandardizedConvNd(dim, dim_out, 3, padding=1, ndim=ndim)
        else:
            if use_xdconv:
                self.proj = XDConv(
                    dim,
                    dim_out,
                    norm="group",
                    norm_kwargs={"num_groups": groups},
                    points=points,
                    grid=grid,
                    # fuse_grid=False,
                    layer_order="cn",  # Individual points and grid normalization
                    custom_grid_sampling=False,
                )
            else:
                conv = get_convnd(ndim)
                self.proj = conv(dim, dim_out, 3, padding=1, bias=False)

        self.norm = None
        if not use_xdconv:
            self.norm = RMSNorm(dim_out, ndim) if use_rms_norm else nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor | tuple[XDCONV_TYPE, Tensor],
        scale_shift: tuple[Tensor, Tensor] | None = None,
        **kwargs,
    ) -> Tensor | XDCONV_TYPE:
        point_features: Tensor | None = None
        grid_features: Tensor | None
        if isinstance(self.proj, XDConv):
            xd_input, points = cast(tuple[XDCONV_TYPE, Tensor], x)
            point_features, _, grid_features = self.proj(xd_input, points, **kwargs)
        else:
            grid_features = self.proj(cast(Tensor, x))

        if self.norm is not None and grid_features is not None:
            grid_features = self.norm(grid_features)

        if scale_shift is not None:
            scale, shift = scale_shift
            if grid_features is not None:
                pattern = f"b c -> b c {' '.join(['1' for _ in range(grid_features.ndim - 2)])}"
                grid_features = grid_features * (rearrange(scale, pattern) + 1) + rearrange(shift, pattern)
            if isinstance(self.proj, XDConv) and point_features is not None:
                point_features = point_features * (rearrange(scale, "b c -> b c 1") + 1) + rearrange(
                    shift, "b c -> b c 1"
                )

        if isinstance(self.proj, XDConv):
            if point_features is not None:
                point_features = self.act(point_features)
            if grid_features is not None:
                grid_features = self.act(grid_features)
            x_out = self.dropout(point_features) if point_features is not None else None
            g_out = self.dropout(grid_features) if grid_features is not None else None
            return x_out, None, g_out
        if grid_features is None:
            raise ValueError("Conv block expected tensor feature")
        return self.dropout(self.act(grid_features))


class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(
        self,
        dim: int,
        dim_out: int,
        ndim: int = 3,
        n_time_embd: int | None = None,
        groups: int = 8,
        rescale_skip: bool = False,  # Rescale skip connection by sqrt(2) to get unit variance
        use_weight_standardization: bool = False,
        use_rms_norm: bool = False,
        use_xdconv: bool = False,
        points: str | None = "single",
        grid: str | None = "single",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(cast(int, n_time_embd), dim_out * 2)) if exists(n_time_embd) else None
        )
        self.use_xdconv = use_xdconv
        self.rescale_skip = rescale_skip

        use_ws = use_weight_standardization
        self.block1 = Block(dim, dim_out, ndim, groups, use_ws, use_rms_norm, use_xdconv, points, grid, dropout)
        self.block2 = Block(dim_out, dim_out, ndim, groups, use_ws, use_rms_norm, use_xdconv, points, grid)
        if not isinstance(self.block1.proj, XDConv):
            nn.init.zeros_(cast(Tensor, cast(Any, self.block2.proj).weight))

        if use_xdconv and points:
            if grid:
                self.res_conv = nn.ModuleList(
                    [
                        nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity(),
                        nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity(),
                    ]
                )
            else:
                self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        else:
            conv = get_convnd(ndim)
            self.res_conv = conv(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(
        self, x: Tensor | tuple[XDCONV_TYPE, Tensor], time_embd: Tensor | None = None, **kwargs
    ) -> Tensor | XDCONV_TYPE:
        scale_shift: tuple[Tensor, Tensor] | None = None
        if self.mlp is not None and time_embd is not None:
            projected = cast(Tensor, self.mlp(time_embd))
            scale_shift = cast(tuple[Tensor, Tensor], projected.chunk(2, dim=1))

        if self.use_xdconv:
            xd_input, p = cast(tuple[XDCONV_TYPE, Tensor], x)
            residual_x, _, residual_g = xd_input
            h = self.block1((xd_input, p), scale_shift=scale_shift, **kwargs)
            hx, _, hg = cast(XDCONV_TYPE, self.block2((cast(XDCONV_TYPE, h), p)))
            if hx is not None and residual_x is not None:
                if isinstance(self.res_conv, nn.ModuleList):
                    hx = hx + cast(Tensor, self.res_conv[0](residual_x))
                else:
                    hx = hx + cast(Tensor, self.res_conv(residual_x))
            if hg is not None and residual_g is not None:
                if isinstance(self.res_conv, nn.ModuleList):
                    hg = hg + cast(Tensor, self.res_conv[1](residual_g))
                else:
                    hg = hg + cast(Tensor, self.res_conv(residual_g))
            if self.rescale_skip:
                factor = math.sqrt(2)
                hx = hx / factor if hx is not None else None
                hg = hg / factor if hg is not None else None
                return hx, None, hg
            return hx, None, hg
        else:
            tensor_x = cast(Tensor, x)
            h = cast(Tensor, self.block1(tensor_x, scale_shift=scale_shift))
            h = cast(Tensor, self.block2(h)) + cast(Tensor, self.res_conv(tensor_x))
            if self.rescale_skip:
                return h / math.sqrt(2)
            return h


class Norm(nn.Module):
    def __init__(
        self,
        dim: int,
        fn: Callable,
        pre: bool = True,
        post: bool = True,
        ndim: int = 3,
        groups: int = 8,
        use_rms_norm: bool = False,
    ):
        super().__init__()
        self.fn = fn
        norm = partial(RMSNorm, ndim=ndim) if use_rms_norm else partial(nn.GroupNorm, groups)
        self.pre_norm = norm(dim) if pre else None
        self.post_norm = norm(dim) if post else None
        self.ndim = ndim

    def forward(self, x: Tensor) -> Tensor:
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        if isinstance(self.fn, MultiHeadAttention):
            if self.ndim == 3:
                _b, _c, d, h, w = x.shape
                x = rearrange(x, "b c d h w -> b (d h w) c").contiguous()
            elif self.ndim == 2:
                _b, _c, h, w = x.shape
                x = rearrange(x, "b c h w -> b (h w) c").contiguous()
            elif self.ndim == 1:
                x = x.transpose(1, 2)

            x = self.fn(x)

            if self.ndim == 3:
                x = rearrange(x, "b (d h w) c -> b c d h w", d=d, h=h, w=w).contiguous()
            elif self.ndim == 2:
                x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
            elif self.ndim == 1:
                x = x.transpose(1, 2)
        else:
            x = self.fn(x)
        if self.post_norm is not None:
            return self.post_norm(x)
        return x
