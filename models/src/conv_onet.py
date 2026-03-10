from functools import partial
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from pytorch3dunet.unet3d.model import UNet3D
from pytorch3dunet.unet3d.utils import number_of_features_per_level
from torch import Tensor, nn

from .grid import GridDecoder
from .mixins import MultiEvalMixin, MultiLossMixin, PredictMixin
from .model import Model
from .pointnet import GridLocalPoolPointNet
from .pvcnn import GridPVCNN
from .resnet import FEAT_DIMS, ResNet18, ResNet50FPN, ResNetGridDecoder
from .utils import get_activation, get_norm, grid_sample_2d, grid_sample_3d


class EmbeddingEncoder(nn.Embedding):
    def forward(self, inputs: Tensor) -> dict[str, Tensor]:
        return {"feature": super().forward(inputs)}


class UNet3DEncoder(nn.Module):
    def __init__(self, c_dim: int, unet3d_kwargs: dict, **kwargs):
        super().__init__()
        self.conv_in = nn.Conv3d(1, c_dim, 1)
        self.unet3d = UNet3D(in_channels=c_dim, out_channels=c_dim, **unet3d_kwargs)

    def forward(self, inputs: Tensor, **kwargs) -> dict[str, Tensor | list[Tensor]]:
        feature = self.conv_in(inputs.permute(0, 3, 2, 1).unsqueeze(1))
        raw_grid_feature = cast(Tensor | tuple[Tensor, list[Tensor]], self.unet3d(feature))
        if isinstance(raw_grid_feature, tuple) and isinstance(raw_grid_feature[1], list):
            grid_feature: Tensor | list[Tensor] = [raw_grid_feature[0]] + raw_grid_feature[1]
        else:
            grid_feature = cast(Tensor, raw_grid_feature)
        return {"grid": grid_feature}


class ResNet18Encoder(ResNet18):
    def __init__(
        self,
        dim: int = 3,
        c_dim: int | None = None,
        weights: str | None = "DEFAULT",
        return_nodes: list[str] | None = None,
        feature_type: str = "uv",
    ):
        super().__init__(dim=dim, c_dim=c_dim, weights=weights, return_nodes=return_nodes)
        self.return_nodes = return_nodes
        self.feature_type = feature_type

    def forward(self, x: Tensor, **kwargs) -> dict[str, Tensor | list[Tensor]]:
        if x.dim() == 3:  # (B, H, W)
            x = x.unsqueeze(1)
        return {self.feature_type: super().forward(x)}


class LinearGridDecoder(GridDecoder):
    def __init__(self, c_dim: int = 32, padding: float = 0.1, **kwargs):
        super().__init__(c_dim=c_dim, padding=padding)

        self.classifier = nn.Conv1d(c_dim, 1, 1)

    def forward(self, points: Tensor, feature_dict: dict[str, Tensor | list[Tensor]]) -> Tensor:
        feature, _ = self.sample_feature(points, feature_dict)
        return self.classifier(feature).squeeze(1)


class SimpleGridDecoder(GridDecoder):
    def __init__(
        self,
        channels: tuple[int, ...],
        norm: str | None = None,
        activation: str = "relu",
        padding: float = 0.1,
        **kwargs,
    ):
        super().__init__(c_dim=channels[0], padding=padding, **kwargs)

        layers = list()
        for i in range(len(channels) - 1):
            layers.append(nn.Conv1d(channels[i], channels[i + 1], 1))
            if i < len(channels) - 2:
                if norm is not None:
                    layers.append(get_norm(cast(Any, norm), channels[i + 1]))
                layers.append(get_activation(cast(Any, activation)))
        self.mlp = nn.Sequential(*layers)

    def forward(self, points: Tensor, feature_dict: dict[str, Tensor | list[Tensor]]) -> Tensor:
        feature, _ = self.sample_feature(points, feature_dict)
        return self.mlp(feature).squeeze(1)


class ConvolutionalOccupancyNetwork(MultiEvalMixin, MultiLossMixin, PredictMixin, Model):
    def __init__(self, encoder: nn.Module | nn.ModuleList, decoder: nn.Module):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, points: Tensor, inputs: Tensor, **kwargs) -> dict[str, Tensor]:
        feature_dict = self.encode(inputs, **kwargs)
        return self.decode(points, feature_dict, **kwargs)

    def encode(self, inputs: Tensor, **kwargs) -> dict[str, Tensor | list[Tensor]]:
        if isinstance(self.encoder, nn.ModuleList):
            image_feature = self.encoder[0](kwargs.get("inputs.image"))
            return self.encoder[1](inputs, image_feature=image_feature)
        feature = cast(nn.Module, self.encoder)(inputs, **kwargs)
        if isinstance(feature, dict):
            return feature
        return {"feature": feature}

    def decode(self, points: Tensor, feature: dict[str, Tensor | list[Tensor]], **kwargs) -> dict[str, Tensor]:
        logits = self.decoder(points, feature, **kwargs)
        # return torch.distributions.Bernoulli(logits=logits).sample()
        return {"logits": logits}

    def loss(
        self,
        data: dict[str, list[str] | Tensor],
        regression: bool = False,
        name: str | None = None,
        reduction: str | None = "mean",
        **kwargs,
    ) -> Tensor:
        reduction = self.reduction if reduction is None else reduction
        losses = super().loss(data, regression, name, reduction, **kwargs)

        zero = next(self.parameters()).new_tensor(0.0)
        occ_loss = losses.get("occ_loss", zero)
        cls_loss = losses.get("cls_loss", zero)
        seg_loss = losses.get("seg_loss", zero)

        occ_term = occ_loss if torch.is_tensor(occ_loss) else zero
        cls_term = cls_loss if torch.is_tensor(cls_loss) else zero
        seg_term = seg_loss if torch.is_tensor(seg_loss) else zero

        cls_weight_raw = data.get("cls_weight", 1.0)
        seg_weight_raw = data.get("seg_weight", 1.0)
        cls_weight = (
            1.0
            if isinstance(cls_weight_raw, list)
            else float(cls_weight_raw.item() if torch.is_tensor(cls_weight_raw) else cls_weight_raw)
        )
        seg_weight = (
            1.0
            if isinstance(seg_weight_raw, list)
            else float(seg_weight_raw.item() if torch.is_tensor(seg_weight_raw) else seg_weight_raw)
        )

        loss = occ_term + cls_weight * cls_term + seg_weight * seg_term

        return loss


class ConvONet(ConvolutionalOccupancyNetwork):
    def __init__(
        self,
        arch: str = "conv_onet_grid",
        dim: int = 3,
        padding: float = 0.1,
        inputs_type: str = "pointcloud",
        custom_grid_sampling: bool = False,
        num_classes: int | None = None,
        activation: str = "relu",
        norm: str | None = None,
        dropout: float = 0,
        **kwargs,
    ):
        common_dim = 32
        grid_sample = F.grid_sample
        layer_order = "gcr"

        if "grid" in arch:
            if custom_grid_sampling:
                grid_sample = grid_sample_3d
            resolution = 2 * common_dim if "big" in arch else common_dim
            encoder_kwargs = {
                "hidden_dim": common_dim,  # ~x15 for ONet
                "feature_type": ("grid",),
                "grid_resolution": resolution,
                "unet3d": True,
                "unet3d_kwargs": {
                    "num_levels": max(2, int(np.log2(resolution) - 2)),
                    "f_maps": common_dim,
                    "is_segmentation": False,
                    "layer_order": layer_order,
                    "mode": "trilinear",
                    "return_encoder_features": True if "pfh" in arch else False,
                    "return_decoder_features": True if "fpn" in arch else False,
                },
                "n_blocks": 5,
                "norm": norm,
                "activation": activation,
                "dropout": dropout,
                "pool_local": True,
            }
        elif "plane" in arch:
            if custom_grid_sampling:
                grid_sample = grid_sample_2d
            resolution = 4 * common_dim if "big" in arch else 2 * common_dim
            feature_type = ["uv"] if "uv" in arch else ["xy"] if "xy" in arch else ["xy", "xz", "yz"]
            encoder_kwargs = {
                "hidden_dim": common_dim,
                "feature_type": feature_type,
                "plane_resolution": resolution,
                "unet": True,
                "unet_kwargs": {
                    "num_levels": max(2, int(np.log2(resolution) - 2)),
                    "f_maps": common_dim,
                    "is_segmentation": False,
                    "layer_order": layer_order,
                    "mode": "bilinear",
                    "return_encoder_features": True if "pfh" in arch else False,
                    "return_decoder_features": True if "fpn" in arch else False,
                },
                "n_blocks": 5,
                "norm": norm,
                "activation": activation,
                "dropout": dropout,
                "pool_local": True,
            }
        elif "multi" in arch:
            assert not custom_grid_sampling, f"Custom grid sampling not supported for arch '{arch}'"
            plane_resolution = 2 * common_dim
            grid_resolution = common_dim
            encoder_kwargs = {
                "hidden_dim": common_dim,
                "feature_type": ("xz", "xy", "yz", "grid"),
                "plane_resolution": plane_resolution,
                "grid_resolution": grid_resolution,
                "unet": True,
                "unet_kwargs": {
                    "num_levels": max(2, int(np.log2(plane_resolution) - 2)),
                    "f_maps": common_dim,
                    "is_segmentation": False,
                    "layer_order": layer_order,
                    "mode": "bilinear",
                },
                "unet3d": True,
                "unet3d_kwargs": {
                    "num_levels": max(2, int(np.log2(grid_resolution) - 2)),
                    "f_maps": common_dim,
                    "is_segmentation": False,
                    "layer_order": layer_order,
                    "mode": "trilinear",
                },
                "n_blocks": 5,
                "norm": norm,
                "activation": activation,
                "dropout": dropout,
                "pool_local": True,
            }
        else:
            raise ValueError(f"Unknown encoder architecture '{arch}'")

        condition = kwargs.get("condition", "add")
        if "cat" in arch:
            condition = "cat"
        elif "attn" in arch:
            condition = "attn"
        elif "batch" in arch:
            condition = "batch"
        elif "global" in arch:
            condition = "global"
        elif "no_cond" in arch:
            condition = None

        decoder_kwargs = {
            "sample_mode": kwargs.get("sample_mode", "bilinear"),
            "padding_mode": kwargs.get("padding_mode", "zeros"),
            "align_corners": kwargs.get("align_corners", True),
            "hidden_dim": 256 if inputs_type == "idx" or "big" in arch else common_dim,  # ~x18 for ONet
            "num_classes": num_classes,
            "n_blocks": 5,
            "sample": inputs_type not in ["idx", None],
            "condition": condition,
            "norm": norm,
            "activation": activation,
            "dropout": dropout,
            "grid_sample": grid_sample,
        }

        c_dim = 512 if inputs_type == "idx" else common_dim
        if inputs_type == "idx":
            assert dim > 9, "dim must be length of dataset for idx input type"
            encoder = EmbeddingEncoder(dim, c_dim)
        elif inputs_type in ["image", "rgb", "shading", "normals"] or ("depth" in inputs_type and dim == 1):
            if "fpn" in arch:
                c_dim = 256
                encoder = ResNet50FPN(c_dim=256, pretrained="imagenet")
            else:
                c_dim = sum(FEAT_DIMS["resnet18"]) if "pfh" in arch else c_dim
                encoder = ResNet18Encoder(
                    dim=dim,
                    c_dim=None if "pfh" in arch else c_dim,
                    weights="DEFAULT" if inputs_type in ["image", "rgb", "shading", "normals"] else None,
                    return_nodes=["layer1", "layer2", "layer3", "layer4"] if "pfh" in arch else None,
                    feature_type="uv" if "uv" in arch else "xy",
                )
        elif inputs_type in ["rgbd", "rgb+depth", "rgb+kinect"]:
            encoder = nn.ModuleList(
                [
                    ResNet18Encoder(dim=dim, c_dim=c_dim, feature_type="uv" if "uv" in arch else "xy"),
                    GridLocalPoolPointNet(dim=dim, c_dim=c_dim, padding=padding, **encoder_kwargs),
                ]
            )
        elif "pvcnn" in arch:
            channels = (3, common_dim, common_dim, common_dim)
            resolutions = (32, 32, 32)
            encoder = GridPVCNN(channels=channels, resolutions=resolutions, padding=padding, **encoder_kwargs)
        elif "unet3d_enc" in arch or "no_enc" in arch:
            encoder = UNet3DEncoder(dim=dim, c_dim=c_dim, padding=padding, **encoder_kwargs)
        else:
            encoder = GridLocalPoolPointNet(dim=dim, c_dim=c_dim, padding=padding, **encoder_kwargs)
            # resize_intrinsic = False: bin features during grid indexing instead during projection to u, v coordinates
            encoder.forward = partial(
                encoder.forward,
                return_point_feature=kwargs.get("return_point_feature", condition in ["global", "attn"]),
                resize_intrinsic=kwargs.get("resize_intrinsic", False),
                show=kwargs.get("encoder_show", False),
            )

        if inputs_type not in ["image", "normals"] and not ("depth" in inputs_type and dim == 1):
            if "fpn" in arch or "pfh" in arch:
                num_levels = max(2, int(np.log2(resolution) - 2))
                c_dim += sum(number_of_features_per_level(common_dim, num_levels - 1))

        if "linear_dec" in arch or "no_dec" in arch:
            decoder = LinearGridDecoder(c_dim=c_dim, padding=padding, **decoder_kwargs)
        elif "simple_dec" in arch:
            decoder = SimpleGridDecoder(c_dim=c_dim, padding=padding, **decoder_kwargs)
        else:
            decoder = ResNetGridDecoder(dim=dim if 3 <= dim <= 9 else 3, c_dim=c_dim, padding=padding, **decoder_kwargs)
            # resize_intrinsic = False allows for bilinear sampling of the feature map
            decoder.forward = partial(
                decoder.forward,
                resize_intrinsic=kwargs.get("resize_intrinsic", False),
                show=kwargs.get("decoder_show", False),
            )

        # encoder.apply(init_weights)
        # decoder.apply(init_weights)

        super().__init__(encoder, decoder)
