from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from eval import eval_cls_seg
from utils import adjust_intrinsic

from .grid import GridEncoder
from .model import Model
from .resnet import ResNetBlockFC
from .utils import get_activation, get_norm, visualize_feature


def _require_tensor(batch: dict[str, str | Tensor], key: str) -> Tensor:
    value = batch[key]
    if not isinstance(value, Tensor):
        raise TypeError(f"Expected tensor at batch['{key}']")
    return value


class GridLocalPoolPointNet(GridEncoder):
    def __init__(
        self,
        dim: int = 3,
        c_dim: int = 32,
        hidden_dim: int = 32,
        scatter_type: str = "mean",
        unet: bool = False,
        unet_kwargs: dict[str, Any] | None = None,
        unet3d: bool = False,
        unet3d_kwargs: dict[str, Any] | None = None,
        plane_resolution: int | None = None,
        grid_resolution: int | None = None,
        feature_type: tuple[str] = ("grid",),
        padding: float = 0.1,
        n_blocks: int = 5,
        norm: str | None = None,
        activation: str = "relu",
        dropout: float = 0,
        pool_local: bool = True,
        **kwargs,
    ):
        super().__init__(
            c_dim,
            unet,
            unet_kwargs,
            unet3d,
            unet3d_kwargs,
            plane_resolution,
            grid_resolution,
            feature_type,
            scatter_type,
            padding,
            **kwargs,
        )
        self.dim = dim
        self.hidden_dim = hidden_dim

        bias = False if norm else True
        self.fc_p = nn.Conv1d(dim, 2 * hidden_dim, 1, bias=bias)
        self.blocks = nn.ModuleList(
            ResNetBlockFC(
                size_in=2 * hidden_dim,
                size_out=hidden_dim,
                norm=norm,
                activation=activation,
                dropout=dropout,
            )
            for _ in range(n_blocks)
        )
        self.norm = get_norm(cast(Any, norm), hidden_dim) if norm else nn.Identity()
        self.activation = get_activation(cast(Any, activation))
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.fc_c = nn.Conv1d(hidden_dim, c_dim, 1)
        self.use_local_pool = pool_local

    def forward(self, x: Tensor, **kwargs) -> dict[str, Tensor | list[Tensor]]:
        coords = x[:, :, :3]
        if "uv" in self.feature_type:
            resize = kwargs.get("resize_intrinsic", False)
            intrinsic = kwargs["inputs.intrinsic"]
            width = kwargs["inputs.width"]
            height = kwargs["inputs.height"]
            width_eq_height = torch.equal(width, height)
            all_eq_size = width.unique().numel() == 1 and height.unique().numel() == 1

            if resize:
                if width_eq_height or all_eq_size:
                    intrinsic = adjust_intrinsic(intrinsic, width, height, size=self.plane_resolution)
                    max_value = self.plane_resolution - 1
                else:
                    raise ValueError("All inputs must have the same size OR be square")
            else:
                if all_eq_size:
                    max_value = max(width[0].item(), height[0].item()) - 1
                elif width_eq_height:
                    max_value = height - 1
                else:
                    raise ValueError("All inputs must have the same size OR be square")

            index_dict = self.get_index_dict(
                coords,
                intrinsic=cast(Tensor | None, intrinsic),
                extrinsic=kwargs.get("inputs.extrinsic"),
                max_value=max_value,
            )
        else:
            index_dict = self.get_index_dict(coords, extrinsic=kwargs.get("inputs.extrinsic"))

        if x.size(2) == self.dim:
            feature = x
        elif x.size(2) - 3 == self.dim:
            feature = x[:, :, 3:]
        else:
            raise ValueError(f"Input points dimension must be in [{self.dim}, {self.dim + 3}] but is {x.size(2)}")

        net = self.fc_p(feature.transpose(1, 2))
        net = self.blocks[0](net)

        for block in self.blocks[1:]:
            if not self.use_local_pool:
                pooled = net.max(dim=2, keepdim=True)[0].expand(net.size())
            else:
                pooled = self.pool_local(net, index_dict)
            net = block(torch.cat([net, pooled], dim=1))  # PointNet concatenates max-pooled features

        net = self.norm(net)
        net = self.activation(net)
        net = self.dropout(net)
        feature = self.fc_c(net)  # [batch_size, c_dim, num_input_points]

        scattered_feature = self.generate_feature(feature, index_dict)

        if kwargs.get("show", False):
            unet = self.unet
            self.unet = None
            images_dict = self.generate_feature(coords.transpose(1, 2), index_dict)
            for key, value in images_dict.items():
                visualize_values = value if isinstance(value, list) else [value]
                for visualize_value in visualize_values:
                    visualize_feature(visualize_value, "encoder " + key, padding=self.padding)
                feat = scattered_feature[key]
                if isinstance(feat, list):
                    feat.extend(visualize_values)
                else:
                    feat = [feat, *visualize_values]
                scattered_feature[key] = feat
            self.unet = unet

        image_feature = kwargs.get("image_feature")
        if image_feature is not None:
            scattered_feature.update(image_feature)
        if kwargs.get("return_point_feature", False):
            scattered_feature["point_feature"] = feature
        return scattered_feature


class ResNetPointNet(nn.Module):
    def __init__(
        self,
        dim: int = 3,
        c_dim: int = 128,
        hidden_dim: int = 128,
        n_blocks: int = 5,
        norm: str | None = None,
        activation: str = "relu",
        dropout: float = 0,
        reduce: bool = True,
        sum_feature: bool = False,
    ):
        super().__init__()

        bias = False if norm else True
        self.fc_p = nn.Conv1d(dim, 2 * hidden_dim, 1, bias=bias)
        self.blocks = nn.ModuleList(
            ResNetBlockFC(
                size_in=2 * hidden_dim,
                size_out=hidden_dim,
                norm=norm,
                activation=activation,
                dropout=dropout,
            )
            for _ in range(n_blocks)
        )
        self.norm = get_norm(cast(Any, norm), hidden_dim) if norm else nn.Identity()
        self.activation = get_activation(cast(Any, activation))
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.fc_c = nn.Conv1d(hidden_dim, c_dim, 1)

        self.reduce = reduce
        self.sum_feature = sum_feature

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        net = self.fc_p(x.transpose(1, 2))
        net = self.blocks[0](net)

        for block in self.blocks[1:]:
            pooled = net.max(dim=2, keepdim=True)[0].expand_as(net)
            net = torch.cat([net, pooled], dim=1)
            net = block(net)

        if self.reduce:
            net = net.max(dim=2, keepdim=True)[0]

        net = self.norm(net)
        net = self.activation(net)
        net = self.dropout(net)
        feature = self.fc_c(net).squeeze(2)

        image_feature = kwargs.get("image_feature")
        if image_feature is not None:
            if self.sum_feature and feature.shape[1] == image_feature.shape[1]:
                feature += image_feature
            else:
                feature = torch.cat([feature, image_feature], dim=1)
        return feature


class SimplePointNet(nn.Module):
    def __init__(
        self,
        dim: int = 3,
        c_dim: int = 128,
        hidden_dim=128,
        n_hidden_layer: int = 5,
        leaky: float = 0,
        batchnorm: bool = False,
        reduce: bool = True,
    ):
        super().__init__()
        self.fc_pos = nn.Conv1d(dim, 2 * hidden_dim, 1)
        self.fc = nn.ModuleList(nn.Conv1d(2 * hidden_dim, hidden_dim, 1) for _ in range(n_hidden_layer))
        self.fc_c = nn.Conv1d(hidden_dim, c_dim, 1)

        self.activation = F.relu if not leaky else lambda x: F.leaky_relu(x, negative_slope=leaky)
        self.batchnorm = batchnorm
        self.reduce = reduce

    def forward(self, x: Tensor) -> Tensor:
        net = self.fc_pos(x.transpose(1, 2))
        net = self.fc[0](self.activation(net))
        for fc in self.fc[1:]:
            pooled = net.max(dim=2, keepdim=True)[0]
            net = torch.cat([net, pooled], dim=1)
            net = fc(self.activation(net))
        if self.reduce:
            net, _ = net.max(dim=2, keepdim=True)
        return self.fc_c(self.activation(net))


class STNkd(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        eye = torch.eye(self.k, dtype=x.dtype, device=x.device)
        x += eye.view(1, self.k**2).expand(x.size())
        return x.view(-1, self.k, self.k)


class PointNetfeat(nn.Module):
    def __init__(
        self,
        channels: tuple[int, ...] | list[int] = (3, 64, 64, 64, 128, 1024),
        k: int = 64,
        global_feat: bool = True,
        feature_transform: bool = True,
    ):
        super().__init__()
        self.k = k
        self.channels = channels
        self.stn = STNkd(k=channels[0])

        self.convs = nn.ModuleList([nn.Conv1d(channels[i - 1], channels[i], 1) for i in range(1, len(channels))])
        self.bns = nn.ModuleList([nn.BatchNorm1d(channels[i]) for i in range(1, len(channels))])

        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=k)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor | None]:
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bns[0](self.convs[0](x)))
        x = F.relu(self.bns[1](self.convs[1](x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bns[2](self.convs[2](x)))
        x = F.relu(self.bns[3](self.convs[3](x)))
        x = self.bns[4](self.convs[4](x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.channels[-1])
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, self.channels[-1], 1).repeat(1, 1, pointfeat.size(2))
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetCls(Model):
    def __init__(self, num_classes: int, dim: int = 3, feature_transform: bool = True):
        super().__init__()
        channels = (dim, 64, 64, 64, 128, 1024)
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(channels, global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.drop1 = nn.Dropout(p=0.3)
        self.drop2 = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, inputs: Tensor, **kwargs) -> tuple[Tensor, Tensor, Tensor | None]:
        feat, trans, trans_feat = self.encode(inputs)
        out = self.decode(feat)
        return out, trans, trans_feat

    def encode(self, inputs: Tensor, **kwargs) -> tuple[Tensor, Tensor, Tensor | None]:
        return self.feat(inputs.transpose(1, 2))

    def decode(self, feature: Tensor, **kwargs) -> Tensor:
        x = self.drop1(F.relu(self.bn1(self.fc1(feature))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        return self.fc3(x)

    def evaluate(
        self, batch: dict[str, str | Tensor], prefix: str = "val/", metrics: list[str] | None = None, **kwargs
    ) -> dict[str, float]:
        net, _, _ = self(**batch)
        labels = _require_tensor(batch, "category.index").long()
        loss = F.cross_entropy(net, labels).cpu().item()
        cls_result = eval_cls_seg(net, labels, metrics, prefix="cls_")
        result: dict[str, float] = {"loss": loss}
        result.update(cast(dict[str, float], cls_result))
        result = {f"{prefix}{k}": v for k, v in result.items()}
        return result

    def predict(self, inputs: Tensor, points: Tensor | None = None, **kwargs) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def feature_transform_regularizer(trans: Tensor) -> Tensor:
        eye = torch.eye(trans.size(1), dtype=trans.dtype, device=trans.device)[None, :, :]
        return torch.mean(torch.norm(torch.bmm(trans, trans.transpose(1, 2)) - eye, dim=(1, 2)))

    def loss(self, batch: dict[str, str | Tensor], **kwargs) -> Tensor:
        net, _trans, trans_feat = self(**batch)
        loss = F.cross_entropy(net, _require_tensor(batch, "category.index").long())
        # if trans is not None:
        #     loss += self.feature_transform_regularizer(trans)
        if trans_feat is not None:
            loss += 0.001 * self.feature_transform_regularizer(trans_feat)
        return loss


class PointNetSeg(Model):
    def __init__(self, num_classes: int, dim: int = 3, feature_transform: bool = True):
        super().__init__()
        channels = (dim, 64, 64, 64, 128, 1024)
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(channels, global_feat=False, feature_transform=feature_transform)
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, num_classes, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, inputs: Tensor, **kwargs) -> tuple[Tensor, Tensor, Tensor | None]:
        feat, trans, trans_feat = self.encode(inputs)
        out = self.decode(feat).transpose(1, 2).contiguous()
        return out, trans, trans_feat

    def encode(self, inputs: Tensor, **kwargs) -> tuple[Tensor, Tensor, Tensor | None]:
        return self.feat(inputs.transpose(1, 2))

    def decode(self, feature: Tensor, **kwargs) -> Tensor:
        x = F.relu(self.bn1(self.conv1(feature)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.conv4(x)

    def evaluate(
        self, batch: dict[str, str | Tensor], prefix: str = "val/", metrics: list[str] | None = None, **kwargs
    ) -> dict[str, float]:
        net, _, _ = self(**batch)
        B, N, C = net.shape
        net = net.view(B * N, C)
        labels = _require_tensor(batch, "inputs.labels").view(-1).long()
        loss = F.cross_entropy(net, labels).cpu().item()
        seg_result = eval_cls_seg(net, labels, metrics, prefix="seg_")
        result: dict[str, float] = {"loss": loss}
        result.update(cast(dict[str, float], seg_result))
        result = {f"{prefix}{k}": v for k, v in result.items()}
        return result

    def predict(self, inputs: Tensor, **kwargs) -> Tensor:
        return self(inputs)[0]

    @staticmethod
    def feature_transform_regularizer(trans: Tensor) -> Tensor:
        eye = torch.eye(trans.size(1), dtype=trans.dtype, device=trans.device)[None, :, :]
        return torch.mean(torch.norm(torch.bmm(trans, trans.transpose(1, 2)) - eye, dim=(1, 2)))

    def loss(self, batch: dict[str, str | Tensor], **kwargs) -> Tensor:
        net, _trans, trans_feat = self(**batch)

        B, N, C = net.shape
        net = net.view(B * N, C)
        loss = F.cross_entropy(net, _require_tensor(batch, "inputs.labels").view(-1))
        # if trans is not None:
        #     loss += self.feature_transform_regularizer(trans)
        if trans_feat is not None:
            loss += 0.001 * self.feature_transform_regularizer(trans_feat)
        return loss
