from collections.abc import Callable
from typing import Any, Literal, TypeAlias, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision import models
from torchvision.models import ResNet18_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.resnet import conv1x1

from utils import setup_logger

from .grid import GridDecoder
from .transformer import DecoderBlock, MultiHeadAttention
from .utils import get_activation, get_norm

logger = setup_logger(__name__)

ActivationName: TypeAlias = Literal["relu", "leaky", "elu", "gelu", "new_gelu", "geglu", "softplus"]
NormName: TypeAlias = Literal["batch", "instance", "layer", "group"]
FeatureValue: TypeAlias = Tensor | list[Tensor]
FeatureDict: TypeAlias = dict[str, FeatureValue]
GridSampleFn: TypeAlias = Callable[..., Tensor]


def _to_activation_name(name: str) -> ActivationName:
    if name == "leaky_relu":
        return "leaky"
    return cast(ActivationName, name)


def _to_norm_name(name: str) -> NormName:
    return cast(NormName, name)


def _log_debug_level_2(msg: str) -> None:
    debug_level_2 = getattr(logger, "debug_level_2", logger.debug)
    debug_level_2(msg)


class ResNetBackbone(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool_3d)
        self.stage1 = net.layer1
        self.stage2 = net.layer2
        self.stage3 = net.layer3
        self.stage4 = net.layer4

    def forward(self, imgs):
        feats = self.stem(imgs)
        conv1 = self.stage1(feats)  # 18, 34: 64
        conv2 = self.stage2(conv1)
        conv3 = self.stage3(conv2)
        conv4 = self.stage4(conv3)

        return [conv1, conv2, conv3, conv4]


FEAT_DIMS = {
    "resnet18": (64, 128, 256, 512),
    "resnet34": (64, 128, 256, 512),
    "resnet50": (256, 512, 1024, 2048),
    "resnet101": (256, 512, 1024, 2048),
    "resnet152": (256, 512, 1024, 2048),
}


def build_backbone(name, pretrained=True):
    resnets = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
    if name in resnets and name in FEAT_DIMS:
        cnn = getattr(models, name)(pretrained=pretrained)
        backbone = ResNetBackbone(cnn)
        feat_dims = FEAT_DIMS[name]
        return backbone, feat_dims
    else:
        raise ValueError(f'Unrecognized backbone type "{name}"')


class ResNet18(nn.Module):
    def __init__(
        self,
        dim: int = 3,
        c_dim: int | None = None,
        weights: str | ResNet18_Weights | None = "DEFAULT",
        return_nodes: list[str] | None = None,
    ):
        super().__init__()
        self.return_nodes = return_nodes
        resolved_weights = ResNet18_Weights.DEFAULT if weights == "DEFAULT" else weights
        feature = models.resnet18(weights=resolved_weights)
        if dim != 3:
            feature.conv1 = nn.Conv2d(dim, FEAT_DIMS["resnet18"][0], kernel_size=7, stride=2, padding=3, bias=False)
        if c_dim is not None:
            if c_dim != feature.fc.in_features:
                feature.fc = nn.Linear(feature.fc.in_features, c_dim)
            else:
                feature.fc = cast(nn.Linear, nn.Identity())
        elif return_nodes is not None:
            self.feature = create_feature_extractor(feature, return_nodes=return_nodes)
        else:
            self.feature = feature

    def forward(self, x: Tensor, **kwargs) -> Tensor | list[Tensor]:
        if self.return_nodes is None:
            return cast(Tensor, self.feature(x))
        return list(cast(dict[str, Tensor], self.feature(x)).values())


class ResNet50FPN(nn.Module):
    def __init__(
        self,
        c_dim: int = 256,
        config: str = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        pretrained: str = "coco",
    ):
        super().__init__()
        from detectron2 import model_zoo
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.config import get_cfg
        from detectron2.modeling import build_backbone

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(config))
        self.fpn = build_backbone(cfg)
        if pretrained == "coco":
            DetectionCheckpointer(self.fpn).load(model_zoo.get_checkpoint_url(config))
        elif pretrained == "imagenet":
            pass
        else:
            cfg.MODEL.WEIGHTS = ""

        self.fc = None
        if c_dim != 256:
            self.fc = conv1x1(256, c_dim)

    def forward(self, x: Tensor, **kwargs) -> dict[str, Tensor]:
        feature = list(self.fpn(x).values())
        if self.fc is not None:
            # feature = [self.fc(feat) for feat in feature]
            feature = self.fc(feature[-1])
        # return {f"xy_sum{i}": feat for i, feat in enumerate(feature)}
        if isinstance(feature, list):
            feature = feature[-1]
        return {"xy": feature}


class CBatchNorm1d(nn.Module):
    def __init__(self, c_dim: int, f_dim: int, norm: str = "batch"):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.norm = norm

        self.conv_gamma = nn.Conv1d(c_dim, f_dim, 1)
        self.conv_beta = nn.Conv1d(c_dim, f_dim, 1)
        if norm == "instance":
            self.bn = nn.InstanceNorm1d(f_dim, affine=False)
        elif norm == "layer":
            self.bn = nn.LayerNorm(f_dim)
        elif norm == "group":
            self.bn = nn.GroupNorm(num_groups=max(2, f_dim // 4), num_channels=f_dim, affine=False)
        elif norm == "batch":
            self.bn = nn.BatchNorm1d(f_dim, affine=False)
        else:
            raise TypeError(f"Invalid norm type '{norm}'")

        # No conditioning at the start of training
        nn.init.zeros_(self.conv_gamma.weight)
        assert self.conv_gamma.bias is not None
        nn.init.ones_(self.conv_gamma.bias)
        nn.init.zeros_(self.conv_beta.weight)
        assert self.conv_beta.bias is not None
        nn.init.zeros_(self.conv_beta.bias)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        assert x.size(0) == c.size(0), f"Batch sizes are {x.size(0)} and {c.size(0)}"
        assert c.size(1) == self.c_dim, f"Dims are {c.size(1)} and {self.c_dim}"

        # c is assumed to be of size batch_size x c_dim x T
        if c.ndim == 2:
            c = c.unsqueeze(2)

        gamma = self.conv_gamma(c)
        beta = self.conv_beta(c)

        net = self.bn(x)
        out = gamma * net + beta

        return out


class CResNetBlockConv1d(nn.Module):
    def __init__(
        self,
        c_dim: int,
        size_in: int,
        size_out: int | None = None,
        size_h: int | None = None,
        norm: str = "batch",
        activation: str = "relu",
        dropout: float = 0,
    ):
        super().__init__()

        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out

        self.norm1 = CBatchNorm1d(c_dim, size_in, norm)
        self.norm2 = CBatchNorm1d(c_dim, size_h, norm)
        self.conv1 = nn.Conv1d(size_in, size_h, 1)
        self.conv2 = nn.Conv1d(size_h, size_out, 1)
        self.dropout1 = nn.Dropout(dropout) if dropout else nn.Identity()
        self.dropout2 = nn.Dropout(dropout) if dropout else nn.Identity()
        self.activation = get_activation(_to_activation_name(activation))

        self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False) if size_in != size_out else nn.Identity()

        nn.init.zeros_(self.conv2.weight)

    def forward(self, inputs: Tensor, feature: Tensor) -> Tensor:
        net = self.norm1(inputs, feature)
        net = self.activation(net)
        net = self.dropout1(net)
        net = self.conv1(net)

        net = self.norm2(net, feature)
        net = self.activation(net)
        net = self.dropout2(net)
        net = self.conv2(net)

        out = net + self.shortcut(inputs)

        return out


class ResNetBlockFC(nn.Module):
    def __init__(
        self,
        size_in: int,
        size_out: int | None = None,
        size_h: int | None = None,
        norm: str | None = None,
        activation: str = "relu",
        dropout: float = 0,
        attn: bool = False,
    ):
        super().__init__()

        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        bias = False if norm else True

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out

        self.norm1 = get_norm(_to_norm_name(norm), size_in) if norm else nn.Identity()
        self.norm2 = get_norm(_to_norm_name(norm), size_h) if norm else nn.Identity()
        self.conv1 = nn.Conv1d(size_in, size_h, 1, bias=bias)
        self.conv2 = nn.Conv1d(size_h, size_out, 1, bias=bias)
        self.dropout1 = nn.Dropout(dropout) if dropout else nn.Identity()
        self.dropout2 = nn.Dropout(dropout) if dropout else nn.Identity()
        self.activation = get_activation(_to_activation_name(activation))

        self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False) if size_in != size_out else nn.Identity()
        self.attn = MultiHeadAttention(n_embd=size_in, n_head=1) if attn else None
        """
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        if isinstance(self.norm2, (nn.BatchNorm1d, nn.GroupNorm)):
            nn.init.zeros_(self.norm2.weight)
        """
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = x
        if self.attn is not None:
            out = out.transpose(1, 2)
            out = self.attn(out)
            out = out.transpose(1, 2)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout1(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.activation(out)
        out = self.dropout2(out)
        out = self.conv2(out)

        out += self.shortcut(identity)

        return out


class ResNetFC(nn.Module):
    def __init__(
        self,
        dim: int,
        c_dim: int,
        hidden_dim: int,
        n_blocks: int = 5,
        leaky: bool = False,
        batchnorm: bool = False,
        reduce: bool = True,
        sum_feature: bool = False,
    ):
        super().__init__()
        self.fc_pos = nn.Conv1d(dim, hidden_dim, 1)
        self.blocks = nn.ModuleList(
            ResNetBlockFC(hidden_dim, activation="leaky_relu" if leaky else "relu", norm="batch" if batchnorm else None)
            for _ in range(n_blocks)
        )
        self.fc_c = nn.Conv1d(hidden_dim, c_dim, 1)

        self.bn_pos = nn.BatchNorm1d(hidden_dim) if batchnorm else lambda x: x
        self.bn_block = nn.BatchNorm1d(hidden_dim) if batchnorm else lambda x: x

        self.activation = F.relu if not leaky else lambda x: F.leaky_relu(x, negative_slope=0.2)
        self.reduce = reduce
        self.sum_feature = sum_feature

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        if x.size(2) == 1:
            x = x.transpose(1, 2)
        net = self.bn_pos(self.fc_pos(x.transpose(1, 2)))
        for block in self.blocks:
            net = block(net)
        if self.reduce:
            net, _ = net.max(dim=2, keepdim=True)

        net = self.fc_c(self.activation(self.bn_block(net)))
        image_feature = kwargs.get("image_feature")
        if image_feature is not None:
            if self.sum_feature and net.shape[1] == image_feature.shape[1]:
                net += image_feature
            else:
                net = torch.cat([net, image_feature], dim=1)
        return net


class ResNetGridDecoder(GridDecoder):
    def __init__(
        self,
        dim: int = 3,
        c_dim: int = 32,
        hidden_dim: int = 32,
        num_classes: int | None = None,
        n_blocks: int = 5,
        padding: float = 0.1,
        sample: bool = True,
        condition: str | None = "add",
        norm: str | None = None,
        activation: str = "relu",
        dropout: float = 0,
        sample_mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = True,
        grid_sample: GridSampleFn = F.grid_sample,
        **kwargs,
    ):
        super().__init__(
            c_dim=c_dim,
            padding=padding,
            sample_mode=sample_mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
            grid_sample=grid_sample,
            **kwargs,
        )
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.sample = sample
        self.condition = condition
        self.num_classes = num_classes

        self.fc_p = None
        if not c_dim or (dim != hidden_dim and condition not in ["global", "attn", None]):
            self.fc_p = nn.Conv1d(dim, hidden_dim, 1)

        bias = False if norm else True
        self.fc_c = None
        if condition in ["add", "cat", None]:
            if c_dim and c_dim != hidden_dim:
                _log_debug_level_2(f"Adding conditioning projection layer from {c_dim} to {hidden_dim}")
                if self.condition is None:
                    self.fc_c = nn.Conv1d(c_dim, hidden_dim, 1, bias=bias)
                else:
                    self.fc_c = nn.ModuleList(nn.Conv1d(c_dim, hidden_dim, 1, bias=bias) for _ in range(n_blocks))
            self.blocks = nn.ModuleList(
                ResNetBlockFC(
                    size_in=2 * hidden_dim if condition == "cat" else hidden_dim,
                    size_out=hidden_dim,
                    norm=norm,
                    activation=activation,
                    dropout=dropout,
                    attn=kwargs.get("self_attn", False),
                )
                for _ in range(n_blocks)
            )
        elif condition in ["batch", "global"]:
            if condition == "global":
                self.fc_c = nn.ModuleList(nn.Conv1d(32, c_dim, 1, bias=bias) for _ in range(n_blocks))
            self.blocks = nn.ModuleList(
                CResNetBlockConv1d(
                    c_dim=c_dim,
                    size_in=hidden_dim,
                    norm="batch" if norm is None else norm,
                    activation=activation,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            )
        elif condition == "attn":
            if c_dim != hidden_dim:
                _log_debug_level_2(f"Adding conditioning projection layer from {c_dim} to {hidden_dim}")
                self.fc_c = nn.ModuleList(nn.Linear(c_dim, hidden_dim, bias=bias) for _ in range(n_blocks))
            self.blocks = nn.ModuleList(
                DecoderBlock(
                    n_embd=hidden_dim,
                    n_head=kwargs.get("n_head", min(8, max(1, hidden_dim // 8))),
                    bias=False,
                    dropout=dropout,
                    hidden_layer_multiplier=kwargs.get("hidden_layer_multiplier", 1),
                    no_self_attn=kwargs.get("no_self_attn", True),
                    activation=cast(Any, _to_activation_name(activation)),
                    norm=_to_norm_name(norm) if norm else None,
                )
                for _ in range(n_blocks)
            )
        else:
            raise TypeError(f"Invalid condition type '{condition}'")
        _log_debug_level_2(f"Using {condition} conditioning")

        self.norm = get_norm(_to_norm_name(norm), hidden_dim) if norm else nn.Identity()
        self.activation = get_activation(_to_activation_name(activation))
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.fc_o = nn.Conv1d(hidden_dim, num_classes or 1, 1)

    def forward(
        self,
        points: Tensor,
        feature: Tensor | FeatureDict | None = None,
        **kwargs,
    ) -> Tensor | tuple[Tensor, Tensor]:
        net = points.transpose(1, 2)
        feature_tensor: Tensor | None = None

        if self.sample:
            if self.condition in ["global", "attn"]:
                if feature is None or torch.is_tensor(feature):
                    raise ValueError("Sample mode with global/attn conditioning requires a feature dictionary")
                feature_dict = dict(feature)
                point_feature = feature_dict.pop("point_feature", None)
                if point_feature is None:
                    raise KeyError("Missing 'point_feature' in feature dictionary")
                feature_tensor = point_feature[-1] if isinstance(point_feature, list) else point_feature
                net, _ = self.sample_feature(points, feature_dict, **kwargs)
            else:
                if feature is None or torch.is_tensor(feature):
                    raise ValueError("Sample mode requires a feature dictionary")
                feature_tensor, _ = self.sample_feature(points, feature, **kwargs)
        else:
            if feature is not None:
                raw_feature: FeatureValue
                if torch.is_tensor(feature):
                    raw_feature = feature
                else:
                    raw_feature = feature["feature"]
                feature_tensor = raw_feature[-1] if isinstance(raw_feature, list) else raw_feature
                if feature_tensor.ndim == 2:  # [B, C], i.e. global feature vector
                    feature_tensor = feature_tensor.unsqueeze(2)

        if self.fc_p is not None:
            net = self.fc_p(net)
        if feature_tensor is not None:
            if self.condition == "attn":
                net = net.transpose(1, 2)
                feature_tensor = feature_tensor.transpose(1, 2)
            elif self.condition is None:
                if self.fc_c is None:
                    net = feature_tensor
                else:
                    net = cast(nn.Conv1d, self.fc_c)(feature_tensor)

        for index, block in enumerate(self.blocks):
            if self.condition in ["add", "cat"]:
                if feature_tensor is None:
                    raise ValueError("Missing conditioning feature for add/cat conditioning")
                feat = feature_tensor
                if isinstance(self.fc_c, nn.ModuleList):
                    feat = cast(Tensor, self.fc_c[index](feature_tensor))
                if self.condition == "add":
                    net = block(cast(Tensor, net + feat))
                elif self.condition == "cat":
                    net = block(torch.cat([net, feat], dim=1))
            elif self.condition in ["batch", "attn", "global"]:
                if feature_tensor is None:
                    raise ValueError(f"Missing conditioning feature for {self.condition} conditioning")
                feat = feature_tensor
                if isinstance(self.fc_c, nn.ModuleList):
                    feat = cast(Tensor, self.fc_c[index](feature_tensor))
                if self.condition == "global":
                    feat, _ = feat.max(dim=2, keepdim=True)
                net = block(net, feat)
            elif self.condition is None:
                net = block(net)

        if self.condition == "attn":
            net = net.transpose(1, 2)

        point_feat = net
        net = self.norm(net)
        net = self.activation(net)
        net = self.dropout(net)
        out = self.fc_o(net)
        if out.size(1) == 1:
            out = out.squeeze(1)

        if kwargs.get("return_point_feature", False):
            return out, point_feat.transpose(1, 2)
        return out
