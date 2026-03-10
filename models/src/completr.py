from functools import partial
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from pytorch3dunet.unet3d.model import UNet2D, UNet3D
from torch import Tensor, nn

from utils import setup_logger

from .mixins import MultiEvalMixin, MultiLossMixin, PredictMixin
from .model import Model
from .pvcnn import PVCNN
from .resnet import ResNetGridDecoder
from .transformer import Decoder, Encoder
from .utils import grid_sample_3d
from .xdconf import XDCONV_TYPE, DoubleConv, UNetXD, XDConv, pytorch_sample, pytorch_scatter, xdconv_maxpool

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


def _require_planes(value: list[Tensor] | None, name: str) -> list[Tensor]:
    if value is None:
        raise ValueError(f"Missing {name}")
    return value


class CompleTr(MultiEvalMixin, MultiLossMixin, PredictMixin, Model):
    def __init__(
        self,
        dim: int = 3,
        padding: float = 0.1,
        encoder_type: str = "unetxd",
        decoder_type: str = "transformer",
        encoder_kwargs: dict[str, Any] | None = None,
        decoder_kwargs: dict[str, Any] | None = None,
        multires: bool = True,
        custom_grid_sampling: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.encoder: Any
        self.decoder: Any
        self.padding = padding
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.multires = multires
        self.sample = partial(pytorch_sample, custom_grid_sampling=custom_grid_sampling)

        encoder_kwargs = encoder_kwargs or dict()
        decoder_kwargs = decoder_kwargs or dict()
        self.planes = encoder_kwargs.get("planes", ("xy", "xz", "yz"))
        self.attn = decoder_kwargs.get("self_attn", False)
        self.cross_attn = decoder_kwargs.get("cross_attn", True)
        self.channels = (dim, 32, 64, 128, 256)
        self.resolutions = (64, 32, 16, 8)

        if encoder_type == "conv":
            layer_order = encoder_kwargs.get("layer_order", "nca")
            self.encoder = nn.ModuleList(
                [
                    DoubleConv(
                        self.channels[i],
                        self.channels[i + 1],
                        kernel_size=3,
                        dim=3,
                        activation=encoder_kwargs.get("activation", "relu"),
                        norm=encoder_kwargs.get("norm", "batch"),
                        layer_order=layer_order if i > 0 else "ca",
                    )
                    for i in range(len(self.resolutions))
                ]
            )
        elif encoder_type in ["pvconv", "pvcnn"]:
            self.encoder = PVCNN(self.channels, self.resolutions, padding=padding, fuse_voxel_feature=True)
        elif encoder_type == "xdconv":
            layer_order = encoder_kwargs.get("layer_order", "nca")
            self.encoder = nn.ModuleList(
                [
                    XDConv(
                        self.channels[i],
                        self.channels[i + 1],
                        activation=encoder_kwargs.get("activation", "relu"),
                        norm=encoder_kwargs.get("norm", "batch"),
                        padding=padding,
                        points=encoder_kwargs.get("points", "single"),
                        planes=self.planes,
                        grid=encoder_kwargs.get("grid", "double"),
                        fuse_points=encoder_kwargs.get("fuse_points", True),
                        fuse_planes=encoder_kwargs.get("fuse_planes", True),
                        fuse_grid=encoder_kwargs.get("fuse_grid", True),
                        planes_res=res * encoder_kwargs.get("planes_factor", 4),
                        grid_res=res,
                        layer_order=layer_order if i > 0 else "ca",
                        custom_grid_sampling=custom_grid_sampling,
                    )
                    for i, res in enumerate(self.resolutions)
                ]
            )
        elif encoder_type == "unet":
            channels = [*list(self.channels), 2 * self.channels[-1]]
            self.resolutions = [2 * self.resolutions[0], *list(self.resolutions)]
            assert len(channels[1:]) >= max(2, int(np.log2(self.resolutions[0]) - 2))
            self.encoder = nn.Sequential(
                nn.Conv2d(channels[0], channels[1], kernel_size=1),
                cast(Any, UNet2D)(
                    in_channels=channels[1],
                    out_channels=channels[1],
                    f_maps=channels[1:],
                    layer_order=encoder_kwargs.get("layer_order", "gcr"),
                    is_segmentation=False,
                    mode=encoder_kwargs.get("mode", "bilinear"),
                    return_decoder_features=True,
                ),
            )
        elif encoder_type == "unet3d":
            assert len(self.channels[1:]) >= max(2, int(np.log2(self.resolutions[0]) - 2))
            self.encoder = nn.Sequential(
                nn.Conv3d(self.channels[0], self.channels[1], kernel_size=1),
                cast(Any, UNet3D)(
                    in_channels=self.channels[1],
                    out_channels=self.channels[1],
                    f_maps=self.channels[1:],
                    layer_order=encoder_kwargs.get("layer_order", "gcr"),
                    is_segmentation=False,
                    mode=encoder_kwargs.get("mode", "trilinear"),
                    return_decoder_features=True,
                ),
            )
        elif encoder_type == "unetxd":
            points = encoder_kwargs.get("points", "self_attn")
            if points == "single":
                _debug_level_1("Encoder: Point feature w/ single conv layer")
            elif points == "double":
                _debug_level_1("Encoder: Point feature w/ MLP")
            elif points == "self_attn":
                _debug_level_1("Encoder: Point feature w/ Transformer encoder block")
            self.encoder = UNetXD(
                channels=[self.channels[1], *list(self.channels[1:])],
                in_channels=self.channels[0],
                out_channels=self.channels[1],
                layer_order=encoder_kwargs.get("layer_order", "nca"),
                activation=encoder_kwargs.get("activation", "relu"),
                norm=encoder_kwargs.get("norm", "group"),
                padding=padding,
                points=points,
                planes=self.planes,
                grid=encoder_kwargs.get("grid", "double"),
                fuse_points=encoder_kwargs.get("fuse_points", True),
                fuse_planes=encoder_kwargs.get("fuse_planes", True),
                fuse_grid=encoder_kwargs.get("fuse_grid", True),
                planes_res=self.resolutions[0] * encoder_kwargs.get("planes_factor", 4),
                grid_res=self.resolutions[0],
                return_decoder_features=True,
                custom_grid_sampling=custom_grid_sampling,
            )
            if not self.cross_attn:
                cast(Any, self.encoder).conv_out[0] = None
        else:
            raise NotImplementedError(f"Unknown encoder '{encoder_type}'")

        self.n_embd = self.channels[1] if "unet" in self.encoder_type else self.channels[-1]
        if multires:
            self.n_embd = self.channels[-1] if "unet" in self.encoder_type else sum(self.channels[1:])
        _debug_level_1(f"Feature dim: {self.n_embd}")
        self.decoder_type = decoder_type
        n_layer = decoder_kwargs.get("n_layer", 5)
        n_head = decoder_kwargs.get("n_head", 4)
        bias = decoder_kwargs.get("bias", False)
        dropout = decoder_kwargs.get("dropout", 0)
        hidden_layer_multiplier = decoder_kwargs.get("hidden_layer_multiplier", 1)
        if decoder_type == "transformer":
            if self.cross_attn:
                _debug_level_1(f"Decoder: Using {'self- and ' if self.attn else ''}cross-attention")
                self.decoder = Decoder(
                    n_layer,
                    self.n_embd,
                    n_head,
                    bias,
                    dropout,
                    hidden_layer_multiplier=hidden_layer_multiplier,
                    no_self_attn=not self.attn,
                    activation=decoder_kwargs.get("activation", "gelu"),
                    norm=decoder_kwargs.get("norm", "new_layer"),
                )
            else:
                _debug_level_1(f"Decoder: Using {'self-attention' if self.attn else 'ResNet MLP'}")
                self.decoder = Encoder(
                    n_layer,
                    self.n_embd,
                    n_head,
                    bias,
                    dropout,
                    hidden_layer_multiplier=hidden_layer_multiplier,
                    no_self_attn=not self.attn,
                    activation=decoder_kwargs.get("activation", "gelu"),
                    norm=decoder_kwargs.get("norm", "new_layer"),
                )
            self.classifier = nn.Linear(self.n_embd, 1)
        elif decoder_type == "resnet":
            _debug_level_1(f"Decoder: Using Grid ResNet{' w/ cross-attention' if self.cross_attn else ''}")
            self.decoder = ResNetGridDecoder(
                dim=dim if 3 <= dim <= 9 else 3,
                c_dim=self.n_embd,
                hidden_dim=decoder_kwargs.get("hidden_dim", self.n_embd),
                padding=padding,
                n_blocks=n_layer,
                condition="attn" if self.cross_attn else kwargs.get("condition", "add"),
                norm=decoder_kwargs.get("norm", None),
                activation=decoder_kwargs.get("activation", "relu"),
                dropout=dropout,
                n_head=n_head,
                hidden_layer_multiplier=hidden_layer_multiplier,
                no_self_attn=not self.attn,
                grid_sample=grid_sample_3d if custom_grid_sampling else F.grid_sample,
            )
        else:
            raise NotImplementedError(f"Unknown decoder '{decoder_type}'")

    def _encode_conv(self, inputs: Tensor, **kwargs) -> list[tuple[None, None, Tensor]]:
        x = inputs.transpose(1, 2)
        x = pytorch_scatter(x, inputs, res=2 * self.resolutions[0], padding=self.padding)
        feature = list()
        for layer in self.encoder:
            x = layer(x)
            feature.append(x)
            x = F.max_pool3d(x, kernel_size=2)
        return [(None, None, f) for f in feature]

    def _encode_pvconv(self, inputs: Tensor, **kwargs) -> list[tuple[Tensor, None, Tensor]]:
        feature = self.encoder(inputs, **kwargs)
        return [(f[0], None, f[1]) for f in feature]

    def _encode_unet(self, inputs: Tensor, **kwargs) -> list[tuple[None, list[Tensor], None]]:
        feature = list()
        for plane in self.planes:
            x = inputs.transpose(1, 2)
            x = pytorch_scatter(x, inputs, res=self.resolutions[0], padding=self.padding, plane=plane)
            x, dec = self.encoder(x)
            feature.append([*dec, x])
        feature = list(map(list, zip(*feature, strict=False)))  # transpose
        return [(None, f, None) for f in feature]

    def _encode_unet3d(self, inputs: Tensor, **kwargs) -> list[tuple[None, None, Tensor]]:
        x = inputs
        if inputs.ndim == 3:
            x = inputs.transpose(1, 2)
            x = pytorch_scatter(x, inputs, res=self.resolutions[0], padding=self.padding)
        elif inputs.ndim == 4:
            x = x.unsqueeze(1)
        if cast(Any, self.encoder)[1].return_encoder_features:
            x, enc, dec = self.encoder(x, **kwargs)
            return [(None, None, d) for d in [enc[0], *dec, x]]
        x, dec = self.encoder(x, **kwargs)
        return [(None, None, d) for d in [*dec, x]]

    def _encode_xdconv(self, inputs: Tensor, **kwargs) -> list[XDCONV_TYPE]:
        feature = list()
        x = (None, None, None)
        for layer in self.encoder:
            x = layer(x, inputs)
            feature.append(x)
            x = xdconv_maxpool(x)
        return feature

    def _encode_unetxd(self, inputs: Tensor, **kwargs) -> list[XDCONV_TYPE]:
        if self.encoder.return_encoder_features:
            x, enc, dec = self.encoder(inputs, **kwargs)
            return [enc[0], *dec, x]
        x, dec = self.encoder(inputs, **kwargs)
        return [*dec, x]

    def forward(self, inputs: Tensor, points: Tensor, **kwargs) -> dict[str, Tensor]:
        feature = self.encode(inputs, **kwargs)
        if self.cross_attn:
            if self.encoder_type not in ["pvconv", "xdconv", "unetxd"]:
                return self.decode(points, feature, inputs, **kwargs)
        return self.decode(points, feature, **kwargs)

    def encode(self, inputs: Tensor, **kwargs) -> list[XDCONV_TYPE]:
        if self.encoder_type == "conv":
            feature = self._encode_conv(inputs, **kwargs)
        elif self.encoder_type == "pvconv":
            feature = self._encode_pvconv(inputs, **kwargs)
        elif self.encoder_type == "xdconv":
            feature = self._encode_xdconv(inputs, **kwargs)
        elif self.encoder_type == "unet":
            feature = self._encode_unet(inputs, **kwargs)
        elif self.encoder_type == "unet3d":
            feature = self._encode_unet3d(inputs, **kwargs)
        elif self.encoder_type == "unetxd":
            feature = self._encode_unetxd(inputs, **kwargs)
        else:
            raise NotImplementedError(f"Unknown encoder '{self.encoder_type}'")
        return cast(list[XDCONV_TYPE], feature)

    def _decode_transformer(
        self, points: Tensor, feature: list[XDCONV_TYPE], inputs: Tensor | None = None, **kwargs
    ) -> Tensor:
        points_feat: list[Tensor] = []
        grid_feat: list[Tensor] = []
        for feat in feature:
            points_f, planes_f, grid_f = feat
            if points_f is not None:
                points_feat.append(points_f)
            if planes_f is not None:
                if points_f is None and self.cross_attn:
                    if inputs is None:
                        raise ValueError("inputs are required for cross-attention plane sampling")
                    points_samples = [
                        self.sample(pf, inputs, self.padding, p) for p, pf in zip(self.planes, planes_f, strict=False)
                    ]
                    points_f = torch.stack(points_samples, dim=-1).sum(dim=-1)
                    points_feat.append(points_f)
                plane_samples = [
                    self.sample(pf, points, self.padding, p) for p, pf in zip(self.planes, planes_f, strict=False)
                ]
                planes_f = torch.stack(plane_samples, dim=-1).sum(dim=-1)
            if grid_f is not None:
                grid_feat.append(self.sample(grid_f, points, self.padding))
                if points_f is None and self.cross_attn:
                    if inputs is None:
                        raise ValueError("inputs are required for cross-attention grid sampling")
                    points_feat.append(self.sample(grid_f, inputs, self.padding))
                if planes_f is not None:
                    grid_feat[-1] = grid_feat[-1] + planes_f
            elif planes_f is not None:
                grid_feat.append(planes_f)
        points_context: Tensor | None = None
        if self.cross_attn:
            points_context = torch.cat(points_feat, dim=1).transpose(1, 2)
        grid_context = torch.cat(grid_feat, dim=1).transpose(1, 2)

        if self.cross_attn:
            if points_context is None:
                raise ValueError("cross-attention requires non-empty point features")
            query_feat = self.decoder(grid_context, points_context)
        else:
            query_feat = self.decoder(grid_context)

        return self.classifier(query_feat).squeeze(-1)

    def _decode_resnet(self, points: Tensor, feature: list[XDCONV_TYPE], **kwargs) -> Tensor:
        if self.multires:
            feature_dict: dict[str, Tensor | list[Tensor]] = {
                "grid": [_require_tensor(f[2], "grid feature") for f in feature],
            }
        else:
            feature_dict = {"grid": _require_tensor(feature[-1][2], "grid feature")}
        if self.planes is not None:
            if self.multires:
                for plane, feat in zip(self.planes, feature, strict=False):
                    feature_dict[plane] = _require_planes(feat[1], "plane feature")
            else:
                plane_features = _require_planes(feature[-1][1], "plane feature")
                for plane in self.planes:
                    feature_dict[plane] = plane_features
        if self.cross_attn:
            if self.multires:
                point_features = [_require_tensor(f[0], "point feature") for f in feature]
                feature_dict["point_feature"] = torch.cat(point_features, dim=1)
            else:
                feature_dict["point_feature"] = _require_tensor(feature[-1][0], "point feature")
        return self.decoder(points, feature_dict)

    def decode(
        self,
        points: Tensor,
        feature: list[tuple[Tensor | None, list[Tensor] | None, Tensor | None]],
        inputs: Tensor | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        if self.decoder_type == "transformer":
            logits = self._decode_transformer(points, feature, inputs)
        elif self.decoder_type == "resnet":
            logits = self._decode_resnet(points, feature)
        else:
            raise NotImplementedError(f"Unknown decoder '{self.decoder_type}'")
        return {"logits": logits}

    def loss(
        self,
        data: dict[str, list[str] | Tensor],
        regression: bool = False,
        name: str | None = None,
        reduction: str | None = "mean",
        **kwargs,
    ) -> Tensor:
        return super().loss(data, regression, name, reduction, **kwargs)["occ_loss"]
