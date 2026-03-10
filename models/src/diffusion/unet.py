import inspect
from collections.abc import Callable
from functools import partial
from itertools import pairwise
from typing import cast

import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, nn

from ..mixins import MultiEvalMixin, PredictMixin
from ..model import Model
from ..transformer import XFORMERS_EXISTS, MultiHeadAttention
from ..xdconf import pytorch_sample
from .blocks import Downsample, Norm, ResnetBlock, SinusoidalPositionEmbeddings, Upsample
from .utils import default, get_convnd


class UNet(nn.Module):
    @classmethod
    def get_init_args(cls):
        return inspect.signature(cls.__init__).parameters.keys()

    def __init__(
        self,
        dim: int,
        init_dim: int | None = None,
        out_dim: int | None = None,
        dim_mults: tuple[int, ...] = (1, 2, 4),
        channels: int = 1,
        ndim: int = 3,
        resolution: int = 32,
        self_condition: bool = False,
        time_condition: bool = True,
        norm_groups: int = 8,
        use_attention: bool = True,
        rescale_skip: bool = False,
        use_weight_standardization: bool = False,
        use_rms_norm: bool = False,
        downsample: str = "spd",
        upsample: str = "nearest",
        use_xdconv: bool = False,
        points: str | None = "single",
        grid: str | None = "single",
        voxel: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        if use_xdconv and grid is None:
            use_attention = False
        self.use_xdconv = use_xdconv
        self.resolution = resolution
        self.voxel = voxel
        self.ndim = ndim

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = cast(int, default(init_dim, dim))
        conv = get_convnd(ndim)
        if ndim == 1 or use_xdconv:
            self.init_conv = nn.Conv1d(input_channels, init_dim, 1)
        elif ndim == 2:
            self.init_conv = conv(input_channels, init_dim, 3, padding=1)
        elif ndim == 3:
            self.init_conv = nn.Sequential(
                Rearrange("b c (d h w) -> b c d h w", d=resolution, h=resolution, w=resolution),
                conv(input_channels, init_dim, 1),
            )
        else:
            raise ValueError(f"Unsupported dimension: {ndim}")

        dims = [init_dim, *(dim * mult for mult in dim_mults)]
        in_out = list(pairwise(dims))

        n_time_embd: int | None = dim * 4 if time_condition else None
        make_block = partial(
            ResnetBlock,
            ndim=ndim,
            n_time_embd=n_time_embd,
            rescale_skip=rescale_skip,
            use_weight_standardization=use_weight_standardization,
            use_rms_norm=use_rms_norm,
            use_xdconv=use_xdconv,
            points=points,
            grid=grid,
            dropout=dropout,
        )
        make_norm = partial(
            Norm, ndim=ndim, groups=norm_groups, use_rms_norm=use_rms_norm and not use_weight_standardization
        )

        _make_down = partial(Downsample, ndim=ndim, mode=downsample)
        _make_up = partial(Upsample, ndim=ndim, mode=upsample)
        make_down: Callable[[int, int, bool], nn.Module]
        make_up: Callable[[int, int, bool], nn.Module]
        if use_xdconv and points:
            if grid:

                def _make_down_xd(dim_in: int, dim_out: int, is_last: bool) -> nn.Module:
                    if is_last:
                        return nn.ModuleList([nn.Conv1d(dim_in, dim_out, 1), conv(dim_in, dim_out, 3, padding=1)])
                    return nn.ModuleList([nn.Conv1d(dim_in, dim_out, 1), _make_down(dim_in, dim_out)])

                def _make_up_xd(dim_in: int, dim_out: int, is_last: bool) -> nn.Module:
                    if is_last:
                        return nn.ModuleList([nn.Conv1d(dim_in, dim_out, 1), conv(dim_in, dim_out, 3, padding=1)])
                    return nn.ModuleList([nn.Conv1d(dim_in, dim_out, 1), _make_up(dim_in, dim_out)])

                make_down = _make_down_xd
                make_up = _make_up_xd
            else:

                def _make_conv1(dim_in: int, dim_out: int, is_last: bool) -> nn.Module:
                    return nn.Conv1d(dim_in, dim_out, kernel_size=1)

                make_down = _make_conv1
                make_up = _make_conv1
        else:

            def _make_down_regular(dim_in: int, dim_out: int, is_last: bool) -> nn.Module:
                return conv(dim_in, dim_out, kernel_size=3, padding=1) if is_last else _make_down(dim_in, dim_out)

            def _make_up_regular(dim_in: int, dim_out: int, is_last: bool) -> nn.Module:
                return conv(dim_in, dim_out, kernel_size=3, padding=1) if is_last else _make_up(dim_in, dim_out)

            make_down = _make_down_regular
            make_up = _make_up_regular

        self.time_mlp: nn.Sequential | None = None
        if time_condition:
            n_time_embd_i = cast(int, n_time_embd)
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, n_time_embd_i),
                nn.GELU(),  # nn.SiLU in LDM/StableDiffusion
                nn.Linear(n_time_embd_i, n_time_embd_i),
            )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            too_big = resolution > 32 and ind < 2
            attn: nn.Module = nn.Identity()
            if use_attention and not too_big:
                attn = make_norm(
                    dim_in,
                    MultiHeadAttention(
                        n_embd=dim_in,
                        n_head=min(16, dim_in // 16),
                        backend="einops" if ndim > 2 else "xformers" if XFORMERS_EXISTS else "torch",
                        linear=ndim > 2,
                    ),
                )

            self.downs.append(
                nn.ModuleList(
                    [
                        make_block(dim_in, dim_in, groups=min(32, dim_in // 8)),
                        make_block(dim_in, dim_in, groups=min(32, dim_in // 8)),
                        attn,
                        make_down(dim_in, dim_out, is_last),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = make_block(mid_dim, mid_dim, groups=min(32, mid_dim // 8))
        self.mid_attn = (
            make_norm(mid_dim, MultiHeadAttention(n_embd=mid_dim, n_head=min(16, mid_dim // 16)), post=False)
            if use_attention
            else None
        )
        self.mid_block2 = make_block(mid_dim, mid_dim, groups=min(32, mid_dim // 8))

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            too_big = resolution > 32 and ind < 2
            attn: nn.Module = nn.Identity()
            if use_attention and not too_big:
                attn = make_norm(
                    dim_out,
                    MultiHeadAttention(
                        n_embd=dim_out,
                        n_head=min(16, dim_out // 16),
                        backend="einops" if ndim > 2 else "xformers" if XFORMERS_EXISTS else "torch",
                        linear=ndim > 2,
                    ),
                )

            self.ups.append(
                nn.ModuleList(
                    [
                        make_block(dim_out + dim_in, dim_out, groups=min(32, dim_in // 8)),
                        make_block(dim_out + dim_in, dim_out, groups=min(32, dim_in // 8)),
                        attn,
                        make_up(dim_out, dim_in, is_last),
                    ]
                )
            )

        self.out_dim = cast(int, default(out_dim, channels))
        if ndim == 1 or use_xdconv:
            self.final_block = make_block(
                init_dim * 2, init_dim, ndim=1, groups=min(32, init_dim // 8), use_xdconv=False
            )
            self.final_conv = nn.Conv1d(init_dim, self.out_dim, 1)
        elif ndim == 2:
            self.final_block = make_block(
                init_dim * 2, init_dim, ndim=2, groups=min(32, init_dim // 8), use_xdconv=False
            )
            self.final_conv = conv(init_dim, self.out_dim, 3, padding=1)
        elif ndim == 3:
            self.final_block = make_block(
                init_dim * 2, init_dim, ndim=3, groups=min(32, init_dim // 8), use_xdconv=False
            )
            self.final_conv = nn.Sequential(conv(init_dim, self.out_dim, 1), Rearrange("b c d h w -> b c (d h w)"))

    def forward(
        self,
        x: Tensor | tuple[Tensor, Tensor],
        time: Tensor | None = None,
        x_self_cond: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        if torch.is_tensor(x):
            features = x
            points: Tensor | None = None
        else:
            features, points = x

        if self.self_condition:
            x_self_cond = cast(Tensor, default(x_self_cond, lambda: torch.zeros_like(features)))
            features = torch.cat([features, x_self_cond], dim=1)
        elif x_self_cond is not None:
            raise ValueError("Model does not support self conditioning")

        if self.use_xdconv:
            x_feat: Tensor | None = self.init_conv(features)
            g_feat: Tensor | None = None
            if points is None:
                raise ValueError("XDConv UNet expects point coordinates")
        else:
            x_feat = None
            g_feat = self.init_conv(features)

        if self.use_xdconv:
            r = cast(Tensor, x_feat).clone()
        else:
            r = cast(Tensor, g_feat).clone()

        if self.time_mlp is None:
            t = time
        else:
            if time is None:
                raise ValueError("time must be provided when time conditioning is enabled")
            t = self.time_mlp(time)

        h: list[tuple[Tensor | None, Tensor | None]] = []

        for down_block in self.downs:
            down_layers = cast(nn.ModuleList, down_block)
            block1 = cast(ResnetBlock, down_layers[0])
            block2 = cast(ResnetBlock, down_layers[1])
            attn = cast(nn.Module, down_layers[2])
            downsample = cast(nn.Module, down_layers[3])

            if self.use_xdconv:
                xd_state = ((x_feat, None, g_feat), cast(Tensor, points))
                x_feat, _, g_feat = cast(
                    tuple[Tensor | None, list[Tensor] | None, Tensor | None],
                    block1(xd_state, t, grid_res=self.resolution),
                )
            else:
                x_feat, g_feat = None, cast(Tensor, block1(cast(Tensor, g_feat), t))
            h.append((x_feat, g_feat))

            if self.use_xdconv:
                xd_state = ((x_feat, None, g_feat), cast(Tensor, points))
                x_feat, _, g_feat = cast(tuple[Tensor | None, list[Tensor] | None, Tensor | None], block2(xd_state, t))
            else:
                x_feat, g_feat = None, cast(Tensor, block2(cast(Tensor, g_feat), t))
            if not isinstance(attn, nn.Identity) and g_feat is not None:
                g_feat = cast(Tensor, attn(g_feat)) + g_feat
                if x_feat is not None:
                    x_feat = x_feat + pytorch_sample(g_feat, cast(Tensor, points))
            h.append((x_feat, g_feat))

            if self.use_xdconv:
                if isinstance(downsample, nn.ModuleList):
                    if x_feat is not None:
                        x_feat = cast(nn.Module, downsample[0])(x_feat)
                    if g_feat is not None:
                        g_feat = cast(nn.Module, downsample[1])(g_feat)
                else:
                    if x_feat is not None:
                        x_feat = downsample(x_feat)
                    if g_feat is not None:
                        g_feat = downsample(g_feat)
            else:
                g_feat = cast(Tensor, downsample(cast(Tensor, g_feat)))

        if self.use_xdconv:
            xd_state = ((x_feat, None, g_feat), cast(Tensor, points))
            x_feat, _, g_feat = cast(
                tuple[Tensor | None, list[Tensor] | None, Tensor | None], self.mid_block1(xd_state, t)
            )
        else:
            x_feat, g_feat = None, cast(Tensor, self.mid_block1(cast(Tensor, g_feat), t))
        if self.mid_attn is not None and g_feat is not None:
            g_feat = cast(Tensor, self.mid_attn(g_feat)) + g_feat
            if x_feat is not None:
                x_feat = x_feat + pytorch_sample(g_feat, cast(Tensor, points))
        if self.use_xdconv:
            xd_state = ((x_feat, None, g_feat), cast(Tensor, points))
            x_feat, _, g_feat = cast(
                tuple[Tensor | None, list[Tensor] | None, Tensor | None], self.mid_block2(xd_state, t)
            )
        else:
            x_feat, g_feat = None, cast(Tensor, self.mid_block2(cast(Tensor, g_feat), t))

        for up_block in self.ups:
            up_layers = cast(nn.ModuleList, up_block)
            block1 = cast(ResnetBlock, up_layers[0])
            block2 = cast(ResnetBlock, up_layers[1])
            attn = cast(nn.Module, up_layers[2])
            upsample = cast(nn.Module, up_layers[3])

            skip_x, skip_g = h.pop()
            if x_feat is not None and skip_x is not None:
                x_feat = torch.cat((x_feat, skip_x), dim=1)
            if g_feat is not None and skip_g is not None:
                g_feat = torch.cat((g_feat, skip_g), dim=1)
            if self.use_xdconv:
                xd_state = ((x_feat, None, g_feat), cast(Tensor, points))
                x_feat, _, g_feat = cast(tuple[Tensor | None, list[Tensor] | None, Tensor | None], block1(xd_state, t))
            else:
                x_feat, g_feat = None, cast(Tensor, block1(cast(Tensor, g_feat), t))

            skip_x, skip_g = h.pop()
            if x_feat is not None and skip_x is not None:
                x_feat = torch.cat((x_feat, skip_x), dim=1)
            if g_feat is not None and skip_g is not None:
                g_feat = torch.cat((g_feat, skip_g), dim=1)
            if self.use_xdconv:
                xd_state = ((x_feat, None, g_feat), cast(Tensor, points))
                x_feat, _, g_feat = cast(tuple[Tensor | None, list[Tensor] | None, Tensor | None], block2(xd_state, t))
            else:
                x_feat, g_feat = None, cast(Tensor, block2(cast(Tensor, g_feat), t))
            if not isinstance(attn, nn.Identity) and g_feat is not None:
                g_feat = cast(Tensor, attn(g_feat)) + g_feat
                if x_feat is not None:
                    x_feat = x_feat + pytorch_sample(g_feat, cast(Tensor, points))

            if self.use_xdconv:
                if isinstance(upsample, nn.ModuleList):
                    if x_feat is not None:
                        x_feat = cast(nn.Module, upsample[0])(x_feat)
                    if g_feat is not None:
                        g_feat = cast(nn.Module, upsample[1])(g_feat)
                else:
                    if x_feat is not None:
                        x_feat = upsample(x_feat)
                    if g_feat is not None:
                        g_feat = upsample(g_feat)
            else:
                g_feat = cast(Tensor, upsample(cast(Tensor, g_feat)))

        if self.use_xdconv:
            if g_feat is not None:
                if self.voxel and x_feat is None:
                    assert r.numel() == g_feat.numel(), f"{r.numel()}, {g_feat.numel()}"
                    x_feat = rearrange(g_feat, "b c w h d -> b c (d h w)")
                elif x_feat is not None:
                    x_feat = x_feat + pytorch_sample(g_feat, cast(Tensor, points))
                else:
                    x_feat = pytorch_sample(g_feat, cast(Tensor, points))
        else:
            x_feat = g_feat

        if x_feat is None:
            raise ValueError("UNet forward produced no features")
        x_feat = torch.cat((x_feat, r), dim=1)
        x_feat = cast(Tensor, self.final_block(x_feat, t))
        return self.final_conv(x_feat)


class UNetModel(MultiEvalMixin, PredictMixin, Model):
    def __init__(
        self,
        dim: int = 3,
        resolution: int = 32,
        points: str | None = "single",
        grid: str | None = "single",
        pos_embd: str | None = None,
        use_xdconv: bool = False,
        supervise_inputs: str | None = None,
        voxel: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert not (supervise_inputs and not use_xdconv), "Supervising inputs is only supported with XDConv"

        self.supervise_inputs = supervise_inputs
        self.points_embd = nn.Identity()
        self.feat_embd = nn.Identity()
        if pos_embd and use_xdconv:
            if pos_embd == "shared":
                self.points_embd = self.feat_embd = nn.Linear(dim, 384)
            elif pos_embd == "linear":
                self.feat_embd = nn.Linear(dim, 384)
                self.points_embd = nn.Linear(dim, 384)
            else:
                raise ValueError(f"Unknown positional embedding type: {pos_embd}")

        common_kwargs = set(UNet.get_init_args()) & set(kwargs.keys())
        init_kwargs = {key: value for key, value in kwargs.items() if key in common_kwargs}
        dim_mults = (1, 2) if resolution == 16 else (1, 2, 4) if resolution == 32 else (1, 2, 4, 8)
        self.model = UNet(
            dim=resolution,
            out_dim=1,
            dim_mults=dim_mults,
            channels=384 if pos_embd and use_xdconv else 3 if use_xdconv and not voxel else 1,
            resolution=resolution,
            time_condition=False,
            use_xdconv=use_xdconv,
            points=points,
            grid=grid,
            voxel=voxel,
            **init_kwargs,
        )

    @staticmethod
    def encode(inputs: Tensor, **kwargs) -> Tensor:
        return inputs.float()

    def decode(self, points: Tensor, feature: Tensor, **kwargs) -> dict[str, Tensor]:
        inputs = feature
        if self.model.use_xdconv:
            if self.model.voxel:
                assert inputs.view(inputs.size(0), -1).size(1) == points.size(1)
                inputs = points
                feature = feature.view(feature.size(0), -1, 1)
            else:
                inputs = torch.cat((points, inputs), dim=1)
                feature = torch.cat((self.points_embd(points), self.feat_embd(feature)), dim=1)
        else:
            feature = feature.view(feature.size(0), -1, 1)

        logits = self.model((feature.transpose(1, 2), inputs))
        if self.supervise_inputs and self.training:
            return {"logits": logits.squeeze(1)}
        return {"logits": logits[..., : points.size(1)].squeeze(1)}

    def forward(self, inputs: Tensor, points: Tensor, **kwargs) -> dict[str, Tensor]:
        return self.decode(points, self.encode(inputs, **kwargs), **kwargs)

    def loss(self, data: dict[str, list[str] | Tensor], reduction: str | None = "mean", **kwargs) -> Tensor:
        logits = cast(Tensor | None, data.get("logits"))
        if logits is None:
            logits = cast(Tensor, self(**data, **kwargs)["logits"])
        targets = cast(Tensor, data["points.occ"])

        if self.supervise_inputs and self.training:
            surface_logits = logits[:, targets.size(1) :]
            volume_logits = logits[:, : targets.size(1)]

            if self.supervise_inputs == "bce":
                surface_loss = F.binary_cross_entropy_with_logits(
                    surface_logits, torch.ones_like(surface_logits), reduction="none"
                )
            elif self.supervise_inputs == "mse":
                surface_loss = F.mse_loss(surface_logits, torch.zeros_like(surface_logits), reduction="none")
            elif self.supervise_inputs == "l1":
                surface_loss = surface_logits.abs()
            else:
                raise ValueError(f"Unknown inputs supervision method: {self.supervise_inputs}")

            volume_loss = F.binary_cross_entropy_with_logits(volume_logits, targets, reduction="none")
            return torch.cat([volume_loss, surface_loss], dim=1).mean()

        reduction_value = cast(str, self.reduction if reduction is None else reduction)
        return F.binary_cross_entropy_with_logits(logits, targets, reduction=reduction_value)
