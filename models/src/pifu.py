from typing import Any, Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .hourglass import HGFilter
from .mixins import MultiEvalMixin, MultiLossMixin, PredictMixin
from .model import Model

Reduction = Literal["none", "mean", "sum"]


def _normalize_reduction(reduction: str | None) -> Reduction:
    if reduction in {"none", "sum", "mean"}:
        return cast(Reduction, reduction)
    return "mean"


def index(feat: Tensor, uv: Tensor) -> Tensor:
    """

    :param feat: [B, C, H, W] image features
    :param uv: [B, 2, N] uv coordinates in the image plane, range [-1, 1]
    :return: [B, C, N] image features at the uv coordinates
    """
    uv = uv.transpose(1, 2)  # [B, N, 2]
    uv = uv.unsqueeze(2)  # [B, N, 1, 2]
    # NOTE: for newer PyTorch, it seems that training results are degraded due to implementation diff in F.grid_sample
    # for old versions, simply remove the aligned_corners argument.
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
    return samples[:, :, :, 0]  # [B, C, N]


def orthogonal(points: Tensor, calibrations: Tensor, transforms: Tensor | None = None) -> Tensor:
    """
    Compute the orthogonal projections of 3D points into the image plane by given projection matrix
    :param points: [B, 3, N] Tensor of 3D points
    :param calibrations: [B, 4, 4] Tensor of projection matrix
    :param transforms: [B, 2, 3] Tensor of image transform matrix
    :return: xyz: [B, 3, N] Tensor of xyz coordinates in the image plane
    """
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    pts = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        pts[:, :2, :] = torch.baddbmm(shift, scale, pts[:, :2, :])
    return pts


def perspective(points: Tensor, calibrations: Tensor, transforms: Tensor | None = None) -> Tensor:
    """
    Compute the perspective projections of 3D points into the image plane by given projection matrix
    :param points: [Bx3xN] Tensor of 3D points
    :param calibrations: [Bx4x4] Tensor of projection matrix
    :param transforms: [Bx2x3] Tensor of image transform matrix
    :return: xy: [Bx2xN] Tensor of xy coordinates in the image plane
    """
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    homo = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    xy = homo[:, :2, :] / homo[:, 2:3, :]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        xy = torch.baddbmm(shift, scale, xy)

    xyz = torch.cat([xy, homo[:, 2:3, :]], 1)
    return xyz


def init_weights(net: nn.Module, init_type: str = "normal", init_gain: float = 0.02) -> None:
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m: nn.Module) -> None:  # define the initialization function
        classname = m.__class__.__name__
        weight = getattr(m, "weight", None)
        bias = getattr(m, "bias", None)
        if isinstance(weight, Tensor) and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                nn.init.normal_(weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(weight.data, gain=init_gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(weight.data, gain=cast(Any, init_gain))
            else:
                raise NotImplementedError(f"initialization method [{init_type}] is not implemented")
            if isinstance(bias, Tensor):
                nn.init.constant_(bias.data, 0.0)
        elif (
            classname.find("BatchNorm2d") != -1
        ):  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            if isinstance(weight, Tensor):
                nn.init.normal_(weight.data, 1.0, init_gain)
            if isinstance(bias, Tensor):
                nn.init.constant_(bias.data, 0.0)

    print(f"initialize network with {init_type}")
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(
    net: nn.Module, init_type: str = "normal", init_gain: float = 0.02, gpu_ids: list[int] | None = None
) -> nn.Module:
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if isinstance(gpu_ids, list) and len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


class SurfaceClassifier(nn.Module):
    def __init__(
        self,
        filter_channels: list[int],
        num_views: int = 1,
        no_residual: bool = True,
        last_op: nn.Module | None = None,
    ):
        super().__init__()

        self.filters: nn.ModuleList = nn.ModuleList()
        self.num_views = num_views
        self.no_residual = no_residual
        self.last_op = last_op

        if self.no_residual:
            for layer_idx in range(len(filter_channels) - 1):
                self.filters.append(nn.Conv1d(filter_channels[layer_idx], filter_channels[layer_idx + 1], 1))
        else:
            for layer_idx in range(len(filter_channels) - 1):
                if layer_idx != 0:
                    self.filters.append(
                        nn.Conv1d(filter_channels[layer_idx] + filter_channels[0], filter_channels[layer_idx + 1], 1)
                    )
                else:
                    self.filters.append(nn.Conv1d(filter_channels[layer_idx], filter_channels[layer_idx + 1], 1))

    def forward(self, feature: Tensor) -> Tensor:
        """
        :param feature: list of [BxC_inxHxW] tensors of image features
        :param xy: [Bx3xN] tensor of (x,y) coodinates in the image plane
        :return: [BxC_outxN] tensor of features extracted at the coordinates
        """

        y = feature
        tmpy = feature
        for i, conv_module in enumerate(self.filters):
            conv = cast(nn.Conv1d, conv_module)
            if self.no_residual:
                y = conv(y)
            else:
                y = conv(y if i == 0 else torch.cat([y, tmpy], 1))
            if i != len(self.filters) - 1:
                y = F.leaky_relu(y)

            if self.num_views > 1 and i == len(self.filters) // 2:
                y = y.view(-1, self.num_views, y.shape[1], y.shape[2]).mean(dim=1)
                tmpy = feature.view(-1, self.num_views, feature.shape[1], feature.shape[2]).mean(dim=1)

        if self.last_op:
            y = self.last_op(y)

        return y


class DepthNormalizer(nn.Module):
    def __init__(self, load_size: int = 512, z_size: float = 200.0):
        super().__init__()

        self.load_size = load_size
        self.z_size = z_size

    def forward(self, z: Tensor):
        """
        Normalize z_feature
        :param z: [B, 1, N] depth value for z in the image coordinate system
        :return:
        """
        z_feat = z * (self.load_size // 2) / self.z_size
        return z_feat


class HGPIFuNet(nn.Module):
    """
    HG PIFu network uses Hourglass stacks as the image filter.
    It does the following:
        1. Compute image feature stacks and store it in self.im_feat_list
            self.im_feat_list[-1] is the last stack (output stack)
        2. Calculate calibration
        3. If training, it index on every intermediate stacks,
            If testing, it index on the last stack.
        4. Classification.
        5. During training, error is calculated on all stacks.
    """

    def __init__(
        self,
        num_views: int = 1,
        mlp_dim: list[int] | None = None,
        no_residual: bool = False,
        skip_hourglass: bool = False,
        projection_mode: str = "perspective",
    ):
        super().__init__()

        self.num_views = num_views
        self.skip_hourglass = skip_hourglass
        if mlp_dim is None:
            mlp_dim = [257, 1024, 512, 256, 128, 1]

        self.image_filter = HGFilter()

        self.surface_classifier = SurfaceClassifier(
            filter_channels=mlp_dim, num_views=num_views, no_residual=no_residual
        )

        self.normalizer = DepthNormalizer()

        # This is a list of [B x Feat_i x H x W] features
        self.im_feat_list: list[Tensor] = []
        self.tmpx: Tensor | None = None
        self.normx: Tensor | None = None

        self.intermediate_preds_list: list[Tensor] = []
        self.projection = perspective if projection_mode == "perspective" else orthogonal

        # init_net(self)

    def filter(self, images: Tensor) -> list[Tensor]:
        """
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        """
        self.im_feat_list, self.tmpx, self.normx = self.image_filter(images)
        # If it is not in training, only produce the last im_feat
        if not self.training:
            self.im_feat_list = [self.im_feat_list[-1]]

        return self.im_feat_list

    def query(
        self, points: Tensor, feature: list[Tensor], calibs: Tensor, transforms: Tensor | None = None
    ) -> list[Tensor]:
        xyz = self.projection(points, calibs, transforms)
        xy = xyz[:, :2, :]
        z = xyz[:, 2:3, :]

        in_img = (xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0)

        z_feat = self.normalizer(z)

        if self.skip_hourglass:
            if self.tmpx is None:
                raise RuntimeError("skip_hourglass=True requires hourglass feature map")
            tmpx_local_feature = index(self.tmpx, xy)

        preds_list = list()
        for im_feat in feature:
            # [B, Feat_i + z, N]
            point_local_feat_list = [index(im_feat, xy), z_feat]

            if self.skip_hourglass:
                point_local_feat_list.append(tmpx_local_feature)

            point_local_feat = torch.cat(point_local_feat_list, 1)

            # out of image plane is always set to 0
            pred = in_img[:, None].float() * self.surface_classifier(point_local_feat)
            preds_list.append(pred)

        return preds_list

    def get_loss(self, preds_list: list[Tensor], labels: Tensor) -> Tensor:
        if not preds_list:
            raise ValueError("preds_list must not be empty")
        loss = torch.zeros((), device=preds_list[0].device, dtype=preds_list[0].dtype)
        for preds in preds_list:
            loss += F.binary_cross_entropy_with_logits(preds, labels)
        return loss / len(preds_list)

    def forward(self, images: Tensor, points: Tensor, calibs: Tensor, transforms: Tensor | None = None) -> list[Tensor]:
        feature = self.filter(images)
        return self.query(points, feature, calibs, transforms)


class PIFu(MultiEvalMixin, MultiLossMixin, PredictMixin, Model):
    def __init__(self):
        super().__init__()

        self.net = HGPIFuNet()

    def encode(self, inputs: Tensor, **kwargs) -> list[Tensor]:
        return self.net.filter(inputs)

    def decode(
        self, points: Tensor, feature: list[Tensor], intrinsic: Tensor, extrinsic: Tensor | None = None, **kwargs
    ) -> dict[str, Tensor]:
        if extrinsic is None:
            extrinsic = torch.eye(4).unsqueeze(0).expand(points.size(0), -1, -1).to(points.device)
        calibration = intrinsic @ extrinsic[:, :3, :]
        return {"logits": self.net.query(points.transpose(1, 2), feature, calibration)[-1]}

    def loss(
        self,
        data: dict[str, list[str] | list[Tensor] | Tensor],
        regression: bool = False,
        name: str | None = None,
        reduction: str | None = "mean",
        **kwargs,
    ) -> Tensor:
        del regression, name, kwargs
        logits = data.get("logits")
        labels = data.get("labels", data.get("points.occ"))
        if logits is None or labels is None:
            raise KeyError("PIFu.loss requires `logits` and `labels`/`points.occ`")
        if not isinstance(labels, Tensor):
            raise TypeError("PIFu.loss expects tensor labels")
        reduction_name = _normalize_reduction(reduction)
        if isinstance(logits, list):
            if not logits:
                raise ValueError("logits list must not be empty")
            if not all(isinstance(logit, Tensor) for logit in logits):
                raise TypeError("PIFu.loss expects tensor logits")
            tensor_logits = cast(list[Tensor], logits)
            losses = [
                F.binary_cross_entropy_with_logits(logit, labels, reduction=reduction_name) for logit in tensor_logits
            ]
            return torch.stack(losses).mean()
        if not isinstance(logits, Tensor):
            raise TypeError("PIFu.loss expects tensor logits")
        return F.binary_cross_entropy_with_logits(logits, labels, reduction=reduction_name)

    def forward(
        self, inputs: Tensor, points: Tensor, intrinsic: Tensor, extrinsic: Tensor | None = None, **kwargs
    ) -> dict[str, Tensor]:
        feature = self.encode(inputs, **kwargs)
        return self.decode(points, feature, intrinsic, extrinsic, **kwargs)
