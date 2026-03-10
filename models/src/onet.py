from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch import distributions as dist

from .mixins import MultiEvalMixin, MultiLossMixin, PredictMixin
from .model import Model
from .pointnet import ResNetPointNet
from .pvcnn import PVCNN
from .resnet import CBatchNorm1d, CResNetBlockConv1d, ResNet18, ResNetBlockFC, ResNetFC, ResNetGridDecoder
from .utils import get_activation, get_norm
from .vae import IsotropicGaussian, ResNet18VAE, ResNetPointNetVAE, UnconditionalVAE


class PVCNNEncoder(PVCNN):
    def __init__(
        self,
        channels: list[int],
        resolutions: list[int],
        norm: str | None = None,
        activation: str = "relu",
        dropout: float = 0,
        resnet: bool = False,
        **kwargs,
    ):
        super().__init__(
            cast(Any, tuple(channels[:-1])),
            cast(Any, tuple(resolutions)),
            cast(str, norm or "batch"),
            activation,
            dropout,
            resnet=resnet,
            **kwargs,
        )

        self.norm = cast(Any, get_norm)(norm, channels[-2])
        self.activation = cast(Any, get_activation)(activation)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.fc_c = nn.Conv1d(channels[-2], channels[-1], 1)

    def forward(self, inputs: Tensor) -> Tensor:
        net = super().forward(inputs)  # ([B x C x N], [B x C x R x R x R])
        net, _ = net[-1][0].max(dim=2, keepdim=True)  # Max-pool final (-1) point-wise feature (0)

        net = self.norm(net)
        net = self.activation(net)
        net = self.dropout(net)
        feature = self.fc_c(net).squeeze(2)

        return feature  # B x C


class Decoder(nn.Module):
    def __init__(
        self,
        dim: int = 3,
        c_dim: int = 128,
        hidden_dim: int = 128,
        n_blocks: int = 5,
        norm: str | None = None,
        activation: str = "relu",
        dropout: float = 0,
    ):
        super().__init__()
        self.fc_c = None
        if c_dim != hidden_dim:
            self.fc_c = nn.Linear(c_dim, hidden_dim)

        self.fc_p = nn.Conv1d(dim, hidden_dim, 1)
        self.blocks = nn.ModuleList(
            ResNetBlockFC(size_in=hidden_dim, size_out=hidden_dim, norm=norm, activation=activation, dropout=dropout)
            for _ in range(n_blocks)
        )
        self.norm = cast(Any, get_norm)(norm, hidden_dim)
        self.activation = cast(Any, get_activation)(activation)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.fc_o = nn.Conv1d(hidden_dim, 1, 1)

    def forward(self, points: Tensor, feature: Tensor | None = None) -> Tensor:
        if feature is None:
            raise ValueError("feature is required")
        net = self.fc_p(points.transpose(1, 2))  # B x C x N

        net_c = feature
        if self.fc_c is not None:
            net_c = self.fc_c(feature)  # B x C
        if len(net_c.size()) == 2:
            net_c = net_c.unsqueeze(2)
        net = net + net_c

        for block in self.blocks:
            net = block(net)

        net = self.norm(net)
        net = self.activation(net)
        net = self.dropout(net)
        out = self.fc_o(net).squeeze(1)

        return out


class DecoderCBatchNorm(nn.Module):
    def __init__(
        self,
        dim: int = 3,
        c_dim: int = 512,
        hidden_dim: int = 256,
        n_blocks: int = 5,
        norm: str = "batch",
        activation: str = "relu",
        dropout: float = 0,
    ):
        super().__init__()
        self.fc_p = nn.Conv1d(dim, hidden_dim, 1)
        self.blocks = nn.ModuleList(
            CResNetBlockConv1d(
                c_dim=c_dim,
                size_in=hidden_dim,
                norm=norm,
                activation=activation,
                dropout=dropout,
            )
            for _ in range(n_blocks)
        )
        self.norm = CBatchNorm1d(c_dim, hidden_dim, norm)
        self.activation = cast(Any, get_activation)(activation)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.fc_o = nn.Conv1d(hidden_dim, 1, 1)

    def forward(self, points: Tensor, feature: Tensor) -> Tensor:
        net = self.fc_p(points.transpose(1, 2))  # B x N x 3 -> B x C x N

        for block in self.blocks:
            net = block(net, feature)

        net = self.norm(net, feature)
        net = self.activation(net)
        net = self.dropout(net)
        out = self.fc_o(net).squeeze(1)  # B x 1 x N -> B x N

        return out


class OccupancyNetwork(MultiEvalMixin, MultiLossMixin, PredictMixin, Model):
    def __init__(self, encoder: nn.Module | list[nn.Module], decoder: nn.Module):
        super().__init__()
        self.encoder: Any
        self.decoder: Any
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs: Tensor, points: Tensor, **kwargs) -> dict[str, Tensor]:
        feature = self.encode(inputs, **kwargs)
        return self.decode(points, feature)

    def encode(self, inputs: Tensor, sample: bool = False, **kwargs) -> Tensor:
        if isinstance(self.encoder, nn.ModuleList):
            image_feature = self.encoder[0](kwargs["inputs.image"])
            return self.encoder[1](inputs, image_feature=image_feature)
        if isinstance(self.encoder, UnconditionalVAE):
            if self.training:
                mean, logstd = self.encoder(inputs, kwargs["targets"])
                return self.encoder.sample_posterior(mean, logstd)
            return self.encoder.sample_prior(torch.Size((inputs.size(0),)))
        if isinstance(self.encoder, IsotropicGaussian):
            mean, logstd = self.encoder(inputs)
            return self.encoder.sample_posterior(mean, logstd) if self.training or sample else mean
        return self.encoder(inputs)

    def decode(self, points: Tensor, feature: Tensor, **kwargs) -> dict[str, Tensor]:
        x = self.decoder(points, feature, **kwargs)
        if torch.is_tensor(x):
            if x.ndim == 2:
                return dict(logits=x)
            if x.ndim == 3:
                if x.size(1) == 1:
                    return dict(logits=x[..., 0])
                x = x.transpose(1, 2)
                if x.size(2) == 4:
                    return dict(logits=x[..., 0], colors=x[..., 1:])
                return dict(logits=x[..., 0], feature=x[..., 1:])
            raise ValueError(f"Unknown decoder output shape {x.size()}")
        x, feature = x
        return dict(logits=x, feature=feature)

    def compute_elbo(
        self, inputs: Tensor | None = None, points: Tensor | None = None, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        if isinstance(self.encoder, UnconditionalVAE):
            assert points is not None and targets is not None, "points/targets must be provided for unconditional VAE"
            q_z = self.encoder.posterior(points=points, targets=targets)
        else:
            assert inputs is not None, "inputs must be provided for conditional VAE"
            q_z = self.encoder.posterior(inputs=inputs)
        assert points is not None and targets is not None, "points/targets must be provided for ELBO"
        z = q_z.rsample() if self.training else q_z.mean

        logits = self.decoder(points, z)
        p_r = dist.Bernoulli(logits=logits)
        rec_loss = -p_r.log_prob(targets).sum(dim=-1)  # binary cross entropy, i.e. negative log likelihood (NLL)

        kl = dist.kl_divergence(q_z, self.encoder.prior).sum(dim=-1)
        elbo = -rec_loss - kl

        return elbo, rec_loss, kl

    def loss(
        self,
        data: dict[str, list[str] | Tensor],
        regression: bool = False,
        name: str | None = None,
        reduction: str | None = "mean",
        **kwargs,
    ) -> Tensor:
        if isinstance(self.encoder, IsotropicGaussian):
            inputs = cast(Tensor, data["inputs"])
            points = cast(Tensor, data["points"])
            targets = cast(Tensor, data["points.occ"])

            if isinstance(self.encoder, UnconditionalVAE):
                q_z = self.encoder.posterior(points, targets)
            else:
                q_z = self.encoder.posterior(inputs)
            kl_loss = dist.kl_divergence(q_z, self.encoder.prior).sum(dim=-1).mean()

            z = q_z.rsample() if self.training else q_z.mean
            logits = cast(Tensor, data["logits"]) if "logits" in data else self.decoder(points, z)
            if self.training:
                rec_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").sum(dim=-1).mean()
            else:
                rec_loss = -dist.Bernoulli(logits=logits).log_prob(targets).sum(dim=-1).mean()
            elbo = -rec_loss - kl_loss
            loss = -elbo
        else:
            reduction = self.reduction if reduction is None else reduction
            loss = super().loss(data, regression, name, reduction, **kwargs)["occ_loss"]
        return loss

    @torch.no_grad()
    def predict(
        self,
        inputs: Tensor | None = None,
        points: Tensor | None = None,
        feature: Tensor | list[Tensor] | None = None,
        key: str = "auto",
        points_batch_size: int | None = None,
        sample: bool = False,
        **kwargs,
    ) -> Tensor:
        if sample and feature is None:
            if not isinstance(self.encoder, IsotropicGaussian):
                raise ValueError("Sampling not supported for non-VAE models")
            assert inputs is not None, "inputs must be provided when sampling latent feature"
            feature = self.encode(inputs, sample, **kwargs)
        if isinstance(self.encoder, UnconditionalVAE):
            assert points is not None, "points must be provided for unconditional prediction"
            return self.decoder(points, self.encoder.sample_prior(torch.Size((points.size(0),))))
        else:
            return super().predict(inputs, points, feature, key, points_batch_size, **kwargs)


class _NullEncoder(nn.Module):
    def forward(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class ONet(OccupancyNetwork):
    def __init__(
        self,
        arch: str = "onet",
        inputs_type: str | None = None,
        dim: int = 3,
        norm: str | None = None,
        activation: str = "relu",
        dropout: float = 0,
        reduction: str | None = "sum_points",
    ):
        c_dim = 128 if inputs_type is None else 256 if inputs_type == "image" else 512
        encoder_kwargs = {
            "hidden_dim": c_dim,
            "n_blocks": 5,
            "reduce": True,
            "sum_feature": False,
            "norm": norm,
            "activation": activation,
            "dropout": dropout,
        }
        decoder_kwargs = {
            "hidden_dim": 256,
            "n_blocks": 5,
            "norm": "batch",
            "activation": activation,
            "dropout": dropout,
        }

        if inputs_type is None:
            encoder = UnconditionalVAE(dim=dim, c_dim=c_dim, **encoder_kwargs)
        elif inputs_type == "idx" or "idx" in arch:
            assert dim > 9, "dim must be length of dataset for idx input type"
            encoder = nn.Embedding(num_embeddings=dim, embedding_dim=c_dim)
        elif inputs_type in ["image", "rgb", "shading", "normals"] or "render" in inputs_type or "dvr" in arch:
            if "vae" in arch:
                encoder = ResNet18VAE(c_dim)
            else:
                encoder = ResNet18(c_dim=c_dim)
        elif inputs_type in ["rgbd", "rgb+kinect"]:
            encoder = nn.ModuleList(
                [ResNet18(c_dim=c_dim // 2), ResNetPointNet(dim=dim, c_dim=c_dim, **encoder_kwargs)]
            )
            c_dim = 3 * c_dim
        else:
            if arch == "onet" or "pointnet" in arch:
                encoder = ResNetPointNet(dim=dim, c_dim=c_dim, **encoder_kwargs)
            elif "vae" in arch:
                encoder = ResNetPointNetVAE(dim=dim, c_dim=c_dim, **encoder_kwargs)
            elif "fc" in arch:
                encoder = ResNetFC(dim=dim, c_dim=c_dim, **encoder_kwargs)
            elif "pvcnn" in arch:
                encoder = PVCNNEncoder(
                    channels=[3, 32, 64, 128, 256, 256, c_dim],
                    resolutions=[64, 32, 16, 8, 8],
                    resnet=False,
                    **encoder_kwargs,
                )
            else:
                raise ValueError(f"Unknown encoder architecture '{arch}'")

        decoder = DecoderCBatchNorm(dim=dim if dim <= 9 else 3, c_dim=c_dim, **decoder_kwargs)

        if "dvr" in arch:
            condition = "add"
            if "dtu" in arch:
                c_dim = 0
                condition = None
                encoder = _NullEncoder()
            decoder = ResNetGridDecoder(
                dim=dim, c_dim=c_dim, hidden_dim=256, n_blocks=5, condition=condition, sample=False
            )

        self.reduction = reduction
        super().__init__(encoder, decoder)
