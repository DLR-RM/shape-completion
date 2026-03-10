from collections.abc import Sequence
from functools import cache
from logging import DEBUG
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn

from libs import furthest_point_sample
from utils import DEBUG_LEVEL_2, cosine_anneal, setup_logger

from .mixins import MultiEvalMixin, MultiLossMixin, PredictMixin
from .model import Model
from .transformer import Block, DecoderBlock, EncoderBlock, NeRFEncoding
from .vae import ResidualVQ, VAEModel, VectorQuantize, VQVAEModel

logger = setup_logger(__name__)


def _log_debug_level_2(message: str) -> None:
    if hasattr(logger, "debug_level_2"):
        cast(Any, logger).debug_level_2(message)
    else:
        logger.debug(message)


def _as_module(module: Any) -> nn.Module:
    return cast(nn.Module, module)


def _as_sequential(module: Any) -> nn.Sequential:
    resolved = _as_module(module)
    if isinstance(resolved, nn.Sequential):
        return resolved
    if isinstance(resolved, nn.ModuleList):
        return nn.Sequential(*list(resolved))
    return nn.Sequential(resolved)


try:
    from torch_cluster import fps  # pyright: ignore[reportMissingImports]
except ImportError:
    logger.debug("torch-cluster is not installed. Original 3DShape2VecSet will not be available.")
    fps = None


class Shape3D2VecSet(MultiEvalMixin, MultiLossMixin, PredictMixin, Model):
    def __init__(
        self,
        n_layer: int = 24,
        n_embd: int = 512,
        n_head: int = 8,
        n_queries: int | Sequence[int] = 512,
        activation: Literal["relu", "leaky", "elu", "gelu", "new_gelu", "geglu"] = "geglu",
        learnable_queries: bool = False,
        padding: float = 0.1,
        nerf_freqs: int = 6,
        **kwargs,
    ):
        super().__init__()
        query_levels = [n_queries] if isinstance(n_queries, int) else list(n_queries)
        if len(query_levels) == 0:
            raise ValueError("n_queries sequence cannot be empty")
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_queries = query_levels[-1]
        self.query_levels = query_levels

        self.nerf_enc = NeRFEncoding(out_dim=n_embd, padding=padding, num_frequencies=nerf_freqs, max_freq_exp=nerf_freqs - 1)
        self.inputs_enc: DecoderBlock | nn.ModuleList
        if len(query_levels) == 1:
            self.inputs_enc = DecoderBlock(n_embd=n_embd, n_head=n_head, no_self_attn=True, activation=activation)
        else:
            self.inputs_enc = nn.ModuleList(
                [
                    DecoderBlock(n_embd=n_embd, n_head=n_head, no_self_attn=True, activation=activation)
                    for _ in query_levels
                ]
            )
        self.layer = nn.Sequential(
            *[EncoderBlock(n_embd=n_embd, n_head=n_head, activation=activation) for _ in range(n_layer)]
        )
        self.points_enc = Block(n_embd=n_embd, n_head=n_head, cross_attn=True)
        self.out = nn.Linear(n_embd, 1)

        self.queries = None
        if learnable_queries:
            self.queries = nn.Embedding(self.n_queries, n_embd)  # Hack to make self.queries appear in model summary

    @staticmethod
    @cache
    def indices(n_queries: int, device_type: str) -> Tensor:
        return torch.arange(n_queries, device=device_type)

    def forward(self, inputs: Tensor, points: Tensor, **kwargs) -> dict[str, Tensor]:
        return self.decode(points, self.encode(inputs))

    def encode_inputs(self, inputs: Tensor, **kwargs) -> Tensor:
        if self.queries is None:
            inputs_fps = cast(Tensor, furthest_point_sample(inputs, self.query_levels[0]))
            queries_nerf = self.nerf_enc(inputs_fps)
            inputs_nerf = self.nerf_enc(inputs)
            if len(self.query_levels) == 1:
                return cast(DecoderBlock, self.inputs_enc)(queries_nerf, inputs_nerf)
            else:
                inputs_enc_modules = cast(nn.ModuleList, self.inputs_enc)
                inputs_enc = cast(DecoderBlock, inputs_enc_modules[0])(queries_nerf, inputs_nerf)
                for n_fps, enc in zip(self.query_levels[1:], inputs_enc_modules[1:], strict=False):
                    queries_fps = cast(
                        Tensor, furthest_point_sample(torch.cat([inputs_fps, inputs_enc], dim=-1), n_fps)
                    )
                    inputs_fps = queries_fps[..., :3]
                    queries_enc = queries_fps[..., 3:]
                    inputs_enc = cast(DecoderBlock, enc)(queries_enc, inputs_enc)
                return inputs_enc
        else:
            indices = self.indices(self.n_queries, inputs.device.type)
            queries = self.queries(indices).unsqueeze(0).expand(inputs.size(0), -1, -1)
            if isinstance(self.inputs_enc, nn.ModuleList):
                return cast(DecoderBlock, self.inputs_enc[0])(queries, self.nerf_enc(inputs))
            return self.inputs_enc(queries, self.nerf_enc(inputs))

    def encode(self, inputs: Tensor, n_enc_layer: int | None = None, **kwargs) -> Tensor:
        x = self.encode_inputs(inputs, **kwargs)
        if n_enc_layer:
            return self.layer[:n_enc_layer](x)
        return self.layer(x)

    def decode(self, points: Tensor, feature: Tensor, **kwargs) -> dict[str, Tensor]:
        return dict(logits=self.out(self.points_enc(self.nerf_enc(points), feature)).squeeze(2))

    def loss(
        self,
        data: dict[str, list[str] | Tensor],
        regression: bool = False,
        name: str | None = None,
        reduction: str | None = "mean",
        **kwargs,
    ) -> Tensor:
        return super().loss(data, regression, name, reduction, **kwargs)["occ_loss"]


class Shape3D2VecSetCls(Shape3D2VecSet):
    def __init__(self, n_classes: int, **kwargs):
        super().__init__(**kwargs)
        self.points_enc = None
        self.out = nn.Linear(self.n_embd, n_classes)

    def forward(self, inputs: Tensor, **kwargs) -> dict[str, Tensor]:
        return self.decode(self.encode(inputs))

    def decode(self, feature: Tensor, **kwargs) -> dict[str, Tensor]:
        return dict(cls_logits=self.out(feature.mean(dim=1)))

    def loss(
        self,
        data: dict[str, list[str] | Tensor],
        regression: bool = False,
        name: str | None = None,
        reduction: str | None = "mean",
        **kwargs,
    ) -> Tensor:
        return MultiLossMixin.loss(self, data, regression, name, reduction, **kwargs)["cls_loss"]


class Shape3D2VecSetVAE(VAEModel):
    def __init__(
        self,
        n_embd: int = 512,
        n_layer: int = 24,
        n_head: int = 8,
        n_queries: int = 512,
        activation: Literal["relu", "leaky", "elu", "gelu", "new_gelu", "geglu"] = "geglu",
        learnable_queries: bool = False,
        padding: float = 0.1,
        n_latent: int | None = None,
        n_feat_layer: int | None = None,
        nerf_freqs: int = 6,
        **kwargs,
    ):
        super().__init__(
            ae=Shape3D2VecSet(
                n_embd=n_embd,
                n_layer=n_layer,
                n_head=n_head,
                n_queries=n_queries,
                activation=activation,
                learnable_queries=learnable_queries,
                padding=padding,
                nerf_freqs=nerf_freqs,
            ),
            n_embd=n_embd,
            n_latent=n_latent,
            **kwargs,
        )
        self.n_embd = n_embd
        self.n_queries = n_queries
        self.n_feat_layer = n_feat_layer

        ae_model = cast(Any, self.ae)
        self.latent_to_embd = nn.Sequential(_as_module(self.latent_to_embd), _as_module(ae_model.layer))
        self._latent_to_feat: nn.Module | None = None

    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)
        _ = self.latent_to_feat

    def teardown(self, *args, **kwargs):
        self._latent_to_feat = None
        super().teardown(*args, **kwargs)

    @property
    def latent_to_feat(self) -> nn.Module:
        if self._latent_to_feat is None:
            self._latent_to_feat = _as_module(self.latent_to_embd)
            if self.n_feat_layer:
                ae_layer = _as_sequential(cast(Any, self.ae).layer)
                self._latent_to_feat = nn.Sequential(_as_module(self.latent_to_embd), ae_layer[: self.n_feat_layer])
        return self._latent_to_feat

    def get_mean_logstd(self, inputs: Tensor, clamp: bool = True, **kwargs) -> tuple[Tensor, Tensor]:
        ae_model = cast(Any, self.ae)
        x = cast(Tensor, ae_model.encode_inputs(inputs, **kwargs))
        mean_logstd = self.mean_logstd(x)
        mean, logstd = torch.chunk(mean_logstd, 2, dim=-1)
        if clamp:
            if logstd.abs().max() > 10:
                _log_debug_level_2(f"{self.name}: logstd={logstd.abs().max().item():.2f}")
            logstd = logstd.clamp(min=-10, max=10)
        elif logstd.abs().max() > 10:
            logger.warning(f"{self.name}: logstd={logstd.abs().max().item():.2f}")
        return mean, logstd


class Shape3D2VecSetVQVAE(VQVAEModel):
    def __init__(
        self,
        n_embd: int = 512,
        n_layer: int = 24,
        n_head: int = 8,
        n_queries: int = 512,
        activation: Literal["relu", "leaky", "elu", "gelu", "new_gelu", "geglu"] = "geglu",
        learnable_queries: bool = False,
        padding: float = 0.1,
        n_latent: int = 512,  # 128 in VQDIF
        n_code: int = 4096,  # 4096 in VQDIF
        n_enc_layer: int = 0,
        n_feat_layer: int | None = None,
        nerf_freqs: int = 6,
        **kwargs,
    ):
        super().__init__(
            ae=Shape3D2VecSet(
                n_embd=n_embd,
                n_layer=n_layer,
                n_head=n_head,
                n_queries=n_queries,
                activation=activation,
                learnable_queries=learnable_queries,
                padding=padding,
                nerf_freqs=nerf_freqs,
            ),
            n_latent=n_latent,
            n_hidden=n_embd,
            n_code=n_code,
            **kwargs,
        )
        self.n_embd = n_embd
        self.n_queries = n_queries
        self.n_feat_layer = n_feat_layer

        self.enc_layer = nn.Identity()
        if n_enc_layer:
            ae_model = cast(Any, self.ae)
            ae_layer = _as_sequential(ae_model.layer)
            self.enc_layer = ae_layer[:n_enc_layer]
            ae_model.layer = ae_layer[n_enc_layer:]

        self._latent_to_embd: nn.Module | None = None
        self._latent_to_feat: nn.Module | None = None

    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)
        _ = self.latent_to_embd
        _ = self.latent_to_feat

    def teardown(self, *args, **kwargs):
        ae_model = cast(Any, self.ae)
        if not isinstance(self.enc_layer, nn.Identity):
            ae_model.layer = nn.Sequential(self.enc_layer, _as_module(ae_model.layer))
            self.enc_layer = nn.Identity()
        if self._latent_to_embd:
            latent_to_embd = _as_sequential(self._latent_to_embd)
            cast(Any, self.vq).project_out = latent_to_embd[0]
        self._latent_to_embd = None
        self._latent_to_feat = None
        super().teardown(*args, **kwargs)

    @property
    def latent_to_embd(self) -> nn.Module:
        if self._latent_to_embd is None:
            ae_model = cast(Any, self.ae)
            self._latent_to_embd = nn.Sequential(_as_module(self.vq.project_out), _as_module(ae_model.layer))
            cast(Any, self.vq).project_out = nn.Identity()
        return self._latent_to_embd

    @property
    def latent_to_feat(self) -> nn.Module:
        if self._latent_to_feat is None:
            self._latent_to_feat = self.latent_to_embd
            if self.n_feat_layer:
                ae_layer = _as_sequential(cast(Any, self.ae).layer)
                self._latent_to_feat = ae_layer[: self.n_feat_layer]
        return self._latent_to_feat

    def encode_inputs(self, inputs: Tensor, **kwargs) -> Tensor:
        ae_model = cast(Any, self.ae)
        encoded = cast(Tensor, ae_model.encode_inputs(inputs, **kwargs))
        return cast(Tensor, self.enc_layer(encoded))

    def quantize(self, x: Tensor, **kwargs) -> tuple[Tensor, Tensor, Tensor]:
        return super().quantize(self.encode_inputs(x, **kwargs), **kwargs)

    def requantize(self, x: Tensor, **kwargs) -> tuple[Tensor, Tensor, Tensor]:
        return super().quantize(x, **kwargs)

    def encode(
        self,
        inputs: Tensor,
        return_all: bool = False,
        global_step: int | None = None,
        total_steps: int | None = None,
        **kwargs,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        temperature = 1.0
        beta = 0.0
        q_prob = 0.0
        if global_step is not None and total_steps:
            if isinstance(self.vq, (VectorQuantize, ResidualVQ)):
                if self.sample_codes:
                    temperature = cosine_anneal(1.0, 1 / 16, total_steps // 2, global_step)
                    self.log("temperature", temperature, level=DEBUG_LEVEL_2, train_only=True)
                if isinstance(self.vq, VectorQuantize) and self.vq.codebook_diversity_loss_weight:
                    beta = cosine_anneal(0.0, 1.0, total_steps, global_step)
                    self.log("beta", beta, level=DEBUG_LEVEL_2, train_only=True)
            if self.quantize_soft:
                q_prob = cosine_anneal(0.0, 1.0, total_steps // 2, global_step)
                self.log("q_prob", q_prob, level=DEBUG_LEVEL_2, train_only=True)

        quantized, indices, vq_loss = self.quantize(inputs, temperature=temperature, beta=beta, q_prob=q_prob)

        quantized_hist = quantized.detach().float().cpu().numpy().flatten()
        self.log("quantized_hist", quantized_hist, level=DEBUG_LEVEL_2)
        self.log("mean", quantized_hist.reshape(-1, self.n_latent).mean(0), level=DEBUG, train_only=True, ema=True)
        self.log("std", quantized_hist.reshape(-1, self.n_latent).std(0), level=DEBUG, train_only=True, ema=True)

        feature = self.latent_to_embd(quantized)
        if return_all:
            return feature, indices, vq_loss
        return feature
