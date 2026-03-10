from typing import Literal, cast

import torch
from torch import Tensor, nn

from ..transformer import DecoderBlock
from .unet import SinusoidalPositionEmbeddings


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, n_embd: int, bias: bool = False):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(n_embd, n_embd * 2))
        self.norm = nn.LayerNorm(n_embd, elementwise_affine=False, bias=bias)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        scale, shift = torch.chunk(self.mlp(t), 2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x


class Layer(nn.Module):
    def __init__(
        self,
        n_layer: int = 24,
        n_embd: int = 512,
        n_head: int = 8,
        bias: bool = True,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        no_self_attn: bool = True,
        activation: Literal["relu", "leaky", "elu", "gelu", "new_gelu", "geglu"] = "geglu",
    ):
        super().__init__()
        self.layer = nn.ModuleList(
            [
                DecoderBlock(
                    n_embd=n_embd,
                    n_head=n_head,
                    bias=bias,
                    dropout=dropout,
                    drop_path=drop_path,
                    no_self_attn=no_self_attn,
                    activation=activation,
                    norm=None,
                )
                for _ in range(n_layer)
            ]
        )
        if not no_self_attn:
            self.self_attn_norm = nn.ModuleList([AdaptiveLayerNorm(n_embd) for _ in range(n_layer)])
        self.cross_attn_norm = nn.ModuleList([AdaptiveLayerNorm(n_embd) for _ in range(n_layer)])
        self.projection_norm = nn.ModuleList([AdaptiveLayerNorm(n_embd) for _ in range(n_layer)])

    def _layer(self, index: int, x: Tensor, t: Tensor, c: Tensor | None = None) -> Tensor:
        block = self.layer[index]
        if block.self_attn is not None:
            self_attn = cast(nn.Module, block.self_attn)
            x = x + cast(Tensor, self_attn(self.self_attn_norm[index](x, t)))
        cross_attn = cast(nn.Module, block.cross_attn)
        x = x + cast(Tensor, cross_attn(self.cross_attn_norm[index](x, t), c))
        projection = cast(nn.Module, block.projection)
        x = x + cast(Tensor, projection(self.projection_norm[index](x, t)))
        return x

    def forward(self, x: Tensor, t: Tensor, c: Tensor | None = None) -> Tensor:
        for i in range(len(self.layer)):
            x = self._layer(i, x, t, c)
        return x


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        n_embd: int = 512,
        n_layer: int = 24,
        n_head: int = 8,
        n_time: int = 256,
        bias: bool = True,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        no_self_attn: bool = True,
        activation: Literal["relu", "leaky", "elu", "gelu", "new_gelu", "geglu"] = "geglu",
    ):
        super().__init__()

        self.time_embd = nn.Sequential(
            SinusoidalPositionEmbeddings(n_time), nn.Linear(n_time, n_embd), nn.SiLU(), nn.Linear(n_embd, n_embd)
        )
        self.layer = Layer(
            n_layer=n_layer,
            n_embd=n_embd,
            n_head=n_head,
            bias=bias,
            dropout=dropout,
            drop_path=drop_path,
            no_self_attn=no_self_attn,
            activation=activation,
        )

    def forward(self, x: Tensor, t: Tensor, c: Tensor | None = None) -> Tensor:
        return self.layer(x, self.time_embd(t).unsqueeze(1), c)


class EDMTransformer(DiffusionTransformer):
    def __init__(
        self,
        n_latent: int | None = None,
        n_layer: int = 24,
        n_embd: int = 512,
        n_head: int = 8,
        n_time: int = 256,
        n_classes: int | None = None,
        bias: bool = True,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        no_self_attn: bool = True,
        activation: Literal["relu", "leaky", "elu", "gelu", "new_gelu", "geglu"] = "geglu",
        sigma_min: float = 0.0,
        sigma_max: float = float("inf"),
        sigma_data: float = 0.5,
    ):
        super().__init__(
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_time=n_time,
            bias=bias,
            dropout=dropout,
            drop_path=drop_path,
            no_self_attn=no_self_attn,
            activation=activation,
        )

        self.n_latent = n_latent
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        self.proj_in = nn.Linear(n_latent or n_embd, n_embd, bias=bias)
        self.proj_out = nn.Linear(n_embd, n_latent or n_embd, bias=bias)
        nn.init.zeros_(self.proj_out.weight)

        if n_classes:
            self.category_emb = nn.Embedding(n_classes, n_embd)

    def forward(self, x: Tensor, sigma: Tensor, cond: Tensor | None = None, **kwargs) -> Tensor:
        cond_emb = cond
        if cond is not None and not torch.is_floating_point(cond):
            cond_emb = self.category_emb(cond).unsqueeze(1)
        return self.proj_out(super().forward(self.proj_in(x), sigma, cond_emb))
