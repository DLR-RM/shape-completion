import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from functools import wraps
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from einops import rearrange
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from torch import Tensor, nn
from torch.nn import Parameter
from torchvision.ops import focal_loss
from torchvision.ops.boxes import box_area

from libs import furthest_point_sample
from utils import apply_trafo, depth_to_image, depth_to_points, setup_logger

logger = setup_logger(__name__)


def permute(x: np.ndarray | Tensor, dim: int):
    if torch.is_tensor(x):
        perm = torch.randperm(x.size(dim)).to(x).long()
        return x.index_select(dim, perm)
    elif isinstance(x, np.ndarray):
        perm = np.random.permutation(x.shape[dim])
        return x.take(perm, axis=dim)
    else:
        raise TypeError(f"Unsupported type: {type(x)}")


def new_gelu(x: Tensor) -> Tensor:
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


@torch.jit.script
def new_gelu_jit(x: Tensor) -> Tensor:
    return new_gelu(x)


class NewGelu(nn.Module):
    def __init__(self, jit: bool = True):
        super().__init__()
        self.jit = jit

    def forward(self, x: Tensor) -> Tensor:
        if self.jit:
            return new_gelu_jit(x)
        return new_gelu(x)


class GEGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


# From https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"


def classification_loss(
    predictions: Tensor,
    target: Tensor,
    name: str | None = None,
    reduction: Literal["mean", "sum", "none", "sum_points"] | None = "mean",
) -> Tensor:
    if name and ("focal" in name or "dice" in name) and target.size() != predictions.size():
        target = F.one_hot(target, num_classes=predictions.size(-1)).float()

    weight = None
    if name and "weight" in name:
        target_one_hot = target
        if target.size() != predictions.size():
            target_one_hot = F.one_hot(target, num_classes=predictions.size(-1)).float()
        n_total = torch.tensor(
            float(np.prod(predictions.shape[:-1])), device=predictions.device, dtype=predictions.dtype
        )
        n_pos = target_one_hot.view(-1, target_one_hot.size(-1)).sum(dim=0) + 1
        n_neg = n_total - n_pos + 1
        weight = n_neg / n_pos

    loss = torch.zeros((), device=predictions.device, dtype=predictions.dtype)
    if predictions.size() == target.size():
        predictions = reduce_dim(predictions)
        if not name or "bce" in name:
            loss = loss + F.binary_cross_entropy_with_logits(predictions, target, pos_weight=weight, reduction="none")
        if name in ["inv_freq_bce", "weighted_bce", "pifuhd_bce"]:
            loss = loss + F.binary_cross_entropy_with_logits(predictions, target, pos_weight=weight, reduction="none")
            inside = target == 1
            if name == "inv_freq_bce":
                # Compute the number of inside (1) and outside (0) points for each sample in the batch
                n_inside = inside.sum(dim=1).float()
                n_outside = (~inside).sum(dim=1).float()

                # Compute weights for inside and outside for each sample inversely proportional to their frequency
                total = n_inside + n_outside
                # weight_inside = (n_outside / total).unsqueeze(1)
                # weight_outside = (n_inside / total).unsqueeze(1)
                weight_inside = (total / torch.clamp(n_inside, min=1)).unsqueeze(1)
                weight_outside = (total / torch.clamp(n_outside, min=1)).unsqueeze(1)

                weights = target * (weight_inside - weight_outside) + weight_outside
                loss *= weights
            elif name in ["weighted_bce", "pifuhd_bce"]:
                alpha = 1 - target.mean() if name == "pifuhd_bce" else 0.6
                loss = alpha * inside.float() * loss + (1 - alpha) * (~inside).float() * loss
        if name and "focal" in name:
            loss = loss + focal_loss.sigmoid_focal_loss(predictions, target, reduction="none")
        if name and "dice" in name:
            if torch.is_tensor(loss) and loss.ndim > 1:
                loss = loss.flatten(1)
                loss = loss.mean(1) if reduction == "mean" else loss.sum(1)
            loss = loss + dice_loss(predictions, target, reduction="none")
    else:
        loss = loss + F.cross_entropy(
            predictions.view(-1, predictions.size(-1)),
            target.view(-1).long(),
            weight=weight,
            reduction="none" if weight is None else "mean",
        )
    return reduce_loss(loss, reduction)


def regression_loss(
    predictions: Tensor,
    sdf: Tensor,
    tsdf: float | None = None,
    name: str = "l1",
    reduction: str | None = "mean",
) -> Tensor:
    if tsdf:
        predictions = torch.clamp(predictions, -tsdf, tsdf)
    if name == "l1":
        loss = F.l1_loss(predictions, sdf, reduction="none")
    elif name == "smooth_l1":
        loss = F.smooth_l1_loss(predictions, sdf, reduction="none")
    elif name in ["l2", "mse"]:
        loss = F.mse_loss(predictions, sdf, reduction="none")
    elif name == "shape_l1":
        delta1 = 0.1
        delta2 = 0.01
        weight1 = 1
        weight2 = 10
        weight3 = 100

        large_mask = torch.abs(sdf) >= delta1
        small_mask = torch.abs(sdf) < delta2

        occ1 = torch.where(large_mask, sdf, 0)
        occ2 = torch.where(~large_mask & ~small_mask, sdf, 0)
        occ3 = torch.where(small_mask, sdf, 0)

        out1 = torch.where(large_mask, predictions, 0)
        out2 = torch.where(~large_mask & ~small_mask, predictions, 0)
        out3 = torch.where(small_mask, predictions, 0)

        loss1 = weight1 * F.l1_loss(out1, occ1, reduction="none")
        loss2 = weight2 * F.l1_loss(out2, occ2, reduction="none")
        loss3 = weight3 * F.l1_loss(out3, occ3, reduction="none")
        loss = loss1 + loss2 + loss3
    elif name == "inv_weight_l1":
        eps = 1e-10
        weights = 1 / (torch.abs(sdf) + eps)
        loss = weights * F.l1_loss(predictions, sdf, reduction="none")
        loss = torch.clamp(loss / 100, 0, 10)
    elif name == "inv_l1":
        eps = 1e-10
        loss = F.l1_loss(1 / (predictions + eps), 1 / (sdf + eps), reduction="none")
        loss = torch.clamp(loss / 1000, 0, 10)
    elif name == "sign_l1":
        wrong_sign = predictions * sdf < 0
        loss = F.l1_loss(predictions, sdf, reduction="none")
        loss = torch.where(wrong_sign, 10 * loss, loss)
    elif name == "disn":
        m1 = 4
        m2 = 1
        delta = 0.01
        mask = torch.abs(sdf) < delta
        loss = F.l1_loss(predictions, sdf, reduction="none")
        loss = torch.where(mask, m1 * loss, m2 * loss)
    else:
        raise NotImplementedError(f"Loss '{name}' not implemented")

    return reduce_loss(loss, reduction)


def reduce_loss(loss: Tensor, reduction: str | None = "mean") -> Tensor:
    if reduction is None or reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "sum_points":
        assert len(loss.size()) == 2, f"Expected 2-dimensional tensor, got {loss.size()}"
        return loss.sum(1).mean(0)
    else:
        raise NotImplementedError(f"Reduction '{reduction}' not implemented")


def get_loss(
    predictions: Tensor,
    occ_or_sdf: Tensor,
    sdf: str | None = None,
    tsdf: float | None = None,
    reduction: Literal["mean", "sum", "none", "sum_points"] | None = "mean",
) -> Tensor:
    if sdf is not None:
        return regression_loss(predictions, occ_or_sdf, tsdf, name=sdf, reduction=reduction)
    else:
        return classification_loss(predictions, occ_or_sdf, reduction=reduction)


def grid_sample_2d(
    inputs: Tensor, grid: Tensor, mode: str = "bilinear", padding_mode: str = "border", align_corners: bool = True
) -> Tensor:
    assert mode == "bilinear", "Only mode='bilinear' is supported"
    assert padding_mode in ["zeros", "border"], "Only padding_mode='zeros' or 'border' is supported"
    assert align_corners, "Only align_corners=True is supported"

    N, C, IH, IW = inputs.shape
    _, H, W, _ = grid.shape

    ix = grid[..., 0]
    iy = grid[..., 1]

    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)

    with torch.no_grad():
        ix_nw = torch.floor(ix)
        iy_nw = torch.floor(iy)
        ix_ne = ix_nw + 1
        iy_ne = iy_nw
        ix_sw = ix_nw
        iy_sw = iy_nw + 1
        ix_se = ix_nw + 1
        iy_se = iy_nw + 1

    nw = (ix_se - ix) * (iy_se - iy)
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)

    with torch.no_grad():
        if padding_mode == "zeros":
            nw[(ix_nw < 0) | (iy_nw < 0) | (ix_nw >= IW) | (iy_nw >= IH)] = 0
            ne[(ix_ne < 0) | (iy_ne < 0) | (ix_ne >= IW) | (iy_ne >= IH)] = 0
            sw[(ix_sw < 0) | (iy_sw < 0) | (ix_sw >= IW) | (iy_sw >= IH)] = 0
            se[(ix_se < 0) | (iy_se < 0) | (ix_se >= IW) | (iy_se >= IH)] = 0

        ix_nw.clamp_(0, IW - 1)
        iy_nw.clamp_(0, IH - 1)

        ix_ne.clamp_(0, IW - 1)
        iy_ne.clamp_(0, IH - 1)

        ix_sw.clamp_(0, IW - 1)
        iy_sw.clamp_(0, IH - 1)

        ix_se.clamp_(0, IW - 1)
        iy_se.clamp_(0, IH - 1)

        i_nw = iy_nw * IW + ix_nw
        i_ne = iy_ne * IW + ix_ne
        i_sw = iy_sw * IW + ix_sw
        i_se = iy_se * IW + ix_se

    input = inputs.view(N, C, IH * IW)

    nw_val = torch.gather(input, 2, i_nw.long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(input, 2, i_ne.long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(input, 2, i_sw.long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(input, 2, i_se.long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (
        nw_val.view(N, C, H, W) * nw.view(N, 1, H, W)
        + ne_val.view(N, C, H, W) * ne.view(N, 1, H, W)
        + sw_val.view(N, C, H, W) * sw.view(N, 1, H, W)
        + se_val.view(N, C, H, W) * se.view(N, 1, H, W)
    )

    return out_val


def grid_sample_3d(
    inputs: Tensor, grid: Tensor, mode: str = "bilinear", padding_mode: str = "border", align_corners: bool = True
) -> Tensor:
    assert mode == "bilinear", "Only mode='bilinear' is supported"
    assert padding_mode in ["zeros", "border"], "Only padding_mode='zeros' or 'border' is supported"
    assert align_corners, "Only align_corners=True is supported"

    N, C, ID, IH, IW = inputs.shape
    _, D, H, W, _ = grid.shape

    ix = grid[..., 0]
    iy = grid[..., 1]
    iz = grid[..., 2]

    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)
    iz = ((iz + 1) / 2) * (ID - 1)

    with torch.no_grad():
        ix_tnw = torch.floor(ix)
        iy_tnw = torch.floor(iy)
        iz_tnw = torch.floor(iz)

        ix_tne = ix_tnw + 1
        iy_tne = iy_tnw
        iz_tne = iz_tnw

        ix_tsw = ix_tnw
        iy_tsw = iy_tnw + 1
        iz_tsw = iz_tnw

        ix_tse = ix_tnw + 1
        iy_tse = iy_tnw + 1
        iz_tse = iz_tnw

        ix_bnw = ix_tnw
        iy_bnw = iy_tnw
        iz_bnw = iz_tnw + 1

        ix_bne = ix_tnw + 1
        iy_bne = iy_tnw
        iz_bne = iz_tnw + 1

        ix_bsw = ix_tnw
        iy_bsw = iy_tnw + 1
        iz_bsw = iz_tnw + 1

        ix_bse = ix_tnw + 1
        iy_bse = iy_tnw + 1
        iz_bse = iz_tnw + 1

    tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz)
    tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz)
    tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz)
    tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz)
    bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse)
    bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw)
    bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne)
    bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw)

    with torch.no_grad():
        if padding_mode == "zeros":
            tnw[(ix_tnw < 0) | (iy_tnw < 0) | (iz_tnw < 0) | (ix_tnw >= IW) | (iy_tnw >= IH) | (iz_tnw >= ID)] = 0
            tne[(ix_tne < 0) | (iy_tne < 0) | (iz_tne < 0) | (ix_tne >= IW) | (iy_tne >= IH) | (iz_tne >= ID)] = 0
            tsw[(ix_tsw < 0) | (iy_tsw < 0) | (iz_tsw < 0) | (ix_tsw >= IW) | (iy_tsw >= IH) | (iz_tsw >= ID)] = 0
            tse[(ix_tse < 0) | (iy_tse < 0) | (iz_tse < 0) | (ix_tse >= IW) | (iy_tse >= IH) | (iz_tse >= ID)] = 0
            bnw[(ix_bnw < 0) | (iy_bnw < 0) | (iz_bnw < 0) | (ix_bnw >= IW) | (iy_bnw >= IH) | (iz_bnw >= ID)] = 0
            bne[(ix_bne < 0) | (iy_bne < 0) | (iz_bne < 0) | (ix_bne >= IW) | (iy_bne >= IH) | (iz_bne >= ID)] = 0
            bsw[(ix_bsw < 0) | (iy_bsw < 0) | (iz_bsw < 0) | (ix_bsw >= IW) | (iy_bsw >= IH) | (iz_bsw >= ID)] = 0
            bse[(ix_bse < 0) | (iy_bse < 0) | (iz_bse < 0) | (ix_bse >= IW) | (iy_bse >= IH) | (iz_bse >= ID)] = 0

        torch.clamp(ix_tnw, 0, IW - 1, out=ix_tnw)
        torch.clamp(iy_tnw, 0, IH - 1, out=iy_tnw)
        torch.clamp(iz_tnw, 0, ID - 1, out=iz_tnw)

        torch.clamp(ix_tne, 0, IW - 1, out=ix_tne)
        torch.clamp(iy_tne, 0, IH - 1, out=iy_tne)
        torch.clamp(iz_tne, 0, ID - 1, out=iz_tne)

        torch.clamp(ix_tsw, 0, IW - 1, out=ix_tsw)
        torch.clamp(iy_tsw, 0, IH - 1, out=iy_tsw)
        torch.clamp(iz_tsw, 0, ID - 1, out=iz_tsw)

        torch.clamp(ix_tse, 0, IW - 1, out=ix_tse)
        torch.clamp(iy_tse, 0, IH - 1, out=iy_tse)
        torch.clamp(iz_tse, 0, ID - 1, out=iz_tse)

        torch.clamp(ix_bnw, 0, IW - 1, out=ix_bnw)
        torch.clamp(iy_bnw, 0, IH - 1, out=iy_bnw)
        torch.clamp(iz_bnw, 0, ID - 1, out=iz_bnw)

        torch.clamp(ix_bne, 0, IW - 1, out=ix_bne)
        torch.clamp(iy_bne, 0, IH - 1, out=iy_bne)
        torch.clamp(iz_bne, 0, ID - 1, out=iz_bne)

        torch.clamp(ix_bsw, 0, IW - 1, out=ix_bsw)
        torch.clamp(iy_bsw, 0, IH - 1, out=iy_bsw)
        torch.clamp(iz_bsw, 0, ID - 1, out=iz_bsw)

        torch.clamp(ix_bse, 0, IW - 1, out=ix_bse)
        torch.clamp(iy_bse, 0, IH - 1, out=iy_bse)
        torch.clamp(iz_bse, 0, ID - 1, out=iz_bse)

    input = inputs.view(N, C, ID * IH * IW)

    tnw_val = torch.gather(
        input, 2, (iz_tnw * IW * IH + iy_tnw * IW + ix_tnw).long().view(N, 1, D * H * W).repeat(1, C, 1)
    )
    tne_val = torch.gather(
        input, 2, (iz_tne * IW * IH + iy_tne * IW + ix_tne).long().view(N, 1, D * H * W).repeat(1, C, 1)
    )
    tsw_val = torch.gather(
        input, 2, (iz_tsw * IW * IH + iy_tsw * IW + ix_tsw).long().view(N, 1, D * H * W).repeat(1, C, 1)
    )
    tse_val = torch.gather(
        input, 2, (iz_tse * IW * IH + iy_tse * IW + ix_tse).long().view(N, 1, D * H * W).repeat(1, C, 1)
    )
    bnw_val = torch.gather(
        input, 2, (iz_bnw * IW * IH + iy_bnw * IW + ix_bnw).long().view(N, 1, D * H * W).repeat(1, C, 1)
    )
    bne_val = torch.gather(
        input, 2, (iz_bne * IW * IH + iy_bne * IW + ix_bne).long().view(N, 1, D * H * W).repeat(1, C, 1)
    )
    bsw_val = torch.gather(
        input, 2, (iz_bsw * IW * IH + iy_bsw * IW + ix_bsw).long().view(N, 1, D * H * W).repeat(1, C, 1)
    )
    bse_val = torch.gather(
        input, 2, (iz_bse * IW * IH + iy_bse * IW + ix_bse).long().view(N, 1, D * H * W).repeat(1, C, 1)
    )

    out_val = (
        tnw_val.view(N, C, D, H, W) * tnw.view(N, 1, D, H, W)
        + tne_val.view(N, C, D, H, W) * tne.view(N, 1, D, H, W)
        + tsw_val.view(N, C, D, H, W) * tsw.view(N, 1, D, H, W)
        + tse_val.view(N, C, D, H, W) * tse.view(N, 1, D, H, W)
        + bnw_val.view(N, C, D, H, W) * bnw.view(N, 1, D, H, W)
        + bne_val.view(N, C, D, H, W) * bne.view(N, 1, D, H, W)
        + bsw_val.view(N, C, D, H, W) * bsw.view(N, 1, D, H, W)
        + bse_val.view(N, C, D, H, W) * bse.view(N, 1, D, H, W)
    )

    return out_val


def reduce_dim(predictions: Tensor) -> Tensor:
    """Reduces the dimensionality of the given tensor based on its initial dimensions.

    The function supports tensors of dimensions 2, 3, 4, and 5. For tensors of dimensions
    4 and 5, it reshapes them into a 2-dimensional tensor where the first dimension is the batch size
    and the second dimension is the product of the other dimensions. If the tensor has dimensions
    2 or 3, it will return the tensor as is.

    Args:
        predictions (Tensor): A PyTorch Tensor whose dimensions should be reduced.

    Returns:
        Tensor: The input tensor with reduced dimensions.

    Raises:
        ValueError: If the input tensor's dimension is not in [2, 3, 4, 5].

    Example:
        >>> tensor = torch.randn(2, 3, 4, 5)
        >>> result = reduce_dim(tensor)
        >>> print(result.shape)
        torch.Size([2, 60])
    """
    if predictions.dim() in [2, 3]:
        return predictions
    elif predictions.dim() == 4:
        return rearrange(predictions, "b c h w -> b (c h w)")
    elif predictions.dim() == 5:
        return rearrange(predictions, "b c h w d -> b (c h w d)")
    else:
        raise ValueError(f"Unsupported input dimension: {predictions.dim()}")


def get_activation(
    name: Literal["relu", "leaky", "elu", "gelu", "new_gelu", "geglu", "softplus", "silu", "selu"],
    inplace: bool = False,
    **kwargs,
) -> nn.Module:
    if inplace:
        logger.warning("Use of inplace activation functions is discouraged")
    if name == "relu":
        return nn.ReLU(inplace=inplace)
    elif "leaky" in name:
        return nn.LeakyReLU(negative_slope=kwargs.get("negative_slope", 0.01), inplace=inplace)
    elif name == "elu":
        return nn.ELU(alpha=kwargs.get("alpha", 1), inplace=inplace)
    elif name == "gelu":
        return nn.GELU(kwargs.get("approximate", "none"))
    elif name == "new_gelu":
        return NewGelu(kwargs.get("jit", True))
    elif "geglu" in name:
        return GEGLU()
    elif name == "softplus":
        return nn.Softplus(beta=kwargs.get("beta", 1), threshold=kwargs.get("threshold", 20))
    elif name == "silu":
        return nn.SiLU(inplace=inplace)
    elif name == "selu":
        return nn.SELU(inplace=inplace)  # Swish is equivalent to SELU with alpha=1
    else:
        raise NotImplementedError(f"Activation '{name}' not implemented")


def get_norm(
    name: Literal["batch", "instance", "layer", "group"], num_channels: int, dim: int = 1, **kwargs
) -> nn.Module:
    assert dim in [1, 2, 3], f"Invalid dimension '{dim}'"
    if name == "batch":
        if dim == 1:
            return nn.BatchNorm1d(num_channels, **kwargs)
        elif dim == 2:
            return nn.BatchNorm2d(num_channels, **kwargs)
        elif dim == 3:
            return nn.BatchNorm3d(num_channels, **kwargs)
    elif name == "instance":
        if dim == 1:
            return nn.InstanceNorm1d(num_channels, **kwargs)
        elif dim == 2:
            return nn.InstanceNorm2d(num_channels, **kwargs)
        elif dim == 3:
            return nn.InstanceNorm3d(num_channels, **kwargs)
    elif name == "layer":
        assert dim == 1, f"LayerNorm is only supported for 1D tensors, got dim={dim}"
        return nn.LayerNorm(num_channels, **kwargs)
    elif name == "group":
        return nn.GroupNorm(num_channels=num_channels, **kwargs)
    else:
        raise NotImplementedError(f"Normalization '{name}' not implemented")
    raise AssertionError("Unreachable normalization branch")


def probs_from_logits(logits: Tensor) -> Tensor:
    logits = reduce_dim(logits)
    if logits.dim() == 2:
        return torch.sigmoid(logits)
    elif logits.dim() == 3:
        return F.softmax(logits, dim=-1)
    else:
        raise ValueError(f"Unsupported input dimension: {logits.dim()}")


def visualize_feature(
    feature: Tensor,
    name: str | None = None,
    batched: bool = True,
    channels_first: bool = True,
    padding: float = 0.1,
    cmap: str | None = None,
    dim_reduction: Literal["min", "max", "mean", "pca"] = "pca",
):
    if not batched:
        feature = feature.unsqueeze(0)
    if feature.dim() == 4:
        logger.debug("Treating feature as 2D planes")
        for feat in feature:
            if channels_first:
                feat = feat.permute(1, 2, 0)
            if feat.size(2) == 1:
                if feat.min() >= 0 and feat.max() <= 1:
                    logger.debug("Treating feature as probability values")
                    image_feat = feat
                    cmap = "plasma" if cmap is None else cmap
                else:
                    logger.debug("Treating feature as depth values")
                    image_feat = np.asarray(depth_to_image(feat.squeeze(2).detach().cpu().numpy()))
            elif feat.size(2) == 3:
                if -(1 + padding) / 2 <= feat.min() < 0 and feat.max() <= (1 + padding) / 2:
                    logger.debug(f"Treating feature as XYZ values. Range: +-{(1 + padding) / 2}")
                    feat = feat / (1 + padding) + 0.5
                elif 0 <= feat.min() <= 1:
                    logger.debug("Treating feature as normalized RGB values. Range: [0, 1]")
                    pass
                elif feat.min() >= 0 and 1 < feat.max() <= 255:
                    logger.debug("Treating feature as RGB values. Range: [0, 255]")
                    feat = feat / 255
                else:
                    logger.debug("Attempting feature conversion to normalized RGB values. Range: [0, 1]")
                    feat = (feat - feat.min()) / (feat.max() - feat.min())
                image_feat = feat.detach().cpu().numpy()
            else:
                logger.debug(f"Reducing feature dimensionality with '{dim_reduction}' method")
                if dim_reduction == "min":
                    feat = (feat - feat.min()) / (feat.max() - feat.min())
                    image_feat = feat.detach().cpu().numpy().min(axis=2, keepdims=True)
                elif dim_reduction == "max":
                    feat = (feat - feat.min()) / (feat.max() - feat.min())
                    image_feat = feat.detach().cpu().numpy().max(axis=2, keepdims=True)
                elif dim_reduction == "mean":
                    feat = (feat - feat.min()) / (feat.max() - feat.min())
                    image_feat = feat.detach().cpu().numpy().mean(axis=2, keepdims=True)
                elif dim_reduction == "pca":
                    h, w, c = feat.size()
                    feat = feat.detach().cpu().numpy().reshape(-1, c)
                    pca = PCA(n_components=3)
                    pca.fit(feat)
                    pcd_feat = pca.transform(feat)
                    pcd_feat = (pcd_feat - pcd_feat.min()) / (pcd_feat.max() - pcd_feat.min())
                    image_feat = pcd_feat.reshape(h, w, 3)

            if cmap is None:
                plt.imshow(np.asarray(image_feat))
            else:
                image_feat_arr = np.asarray(image_feat)
                image_to_show = (
                    image_feat_arr.mean(axis=2)
                    if image_feat_arr.ndim == 3 and image_feat_arr.shape[2] > 1
                    else image_feat_arr
                )
                plt.imshow(image_to_show, cmap=cmap)
            if name is not None:
                plt.title(name)
            plt.axis("off")
            plt.tight_layout()
            plt.show()
            plt.close()
    elif feature.dim() == 5:
        logger.debug("Treating feature as 3D volumes")
        raise NotImplementedError("3D feature visualization not implemented")
    else:
        raise ValueError(f"Unsupported input dimension: {feature.dim()}")


def patch_attention(model: nn.Module, mode: str | None = None, backend: str | None = None) -> nn.Module:
    if not mode and not backend:
        return model
    for _name, module in model.named_modules():
        if module.__class__.__name__ == "Attention":
            module_any = cast(Any, module)
            if mode is not None:
                module_any.mode = mode
            if backend is not None:
                module_any.backend = backend
    return model


def breakpoint_on_exception(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"Exception in {func.__name__}: {e}")
            breakpoint()

    return wrapper


def check_finite(tensor: Tensor, name: str | None = None, do_raise: bool = True) -> tuple[Tensor, ...] | None:
    finite_mask = torch.isfinite(tensor)
    if finite_mask.all():
        return None
    n_non_finite = (~finite_mask).sum().item()
    non_finite_positions = torch.nonzero(~finite_mask, as_tuple=True)
    shape = ", ".join(map(str, tensor.size()))
    msg = f"Tensor{' ' + name if name else ''} ({shape}) contains {n_non_finite} non-finite value(s)"
    if do_raise:
        raise ValueError(msg)
    else:
        logger.warning(msg)
    return non_finite_positions


@contextmanager
def check_finite_context(
    tensor: Tensor, name: str | None = None, do_raise: bool = True, enabled: bool = True
) -> Iterator[tuple[Tensor, ...] | None]:
    if enabled:
        try:
            yield check_finite(tensor, name, do_raise)
        finally:
            check_finite(tensor, name, do_raise)
    else:
        yield None


def check_precision(expected_dtype: torch.dtype) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if torch.is_tensor(args):
                if args.dtype != expected_dtype:
                    raise ValueError(f"{func.__name__}: Expected dtype {expected_dtype}, but got {args.dtype}")
            elif isinstance(args, Iterable) and not isinstance(args, str):
                for arg in args + tuple(kwargs.values()):
                    if torch.is_tensor(arg):
                        if arg.dtype != expected_dtype:
                            raise ValueError(f"{func.__name__}: Expected dtype {expected_dtype}, but got {arg.dtype}")
                    elif isinstance(arg, Iterable) and not isinstance(arg, str):
                        if isinstance(arg, dict):
                            arg = arg.values()
                        for a in arg:
                            if torch.is_tensor(a) and a.dtype != expected_dtype:
                                raise ValueError(f"{func.__name__}: Expected dtype {expected_dtype}, but got {a.dtype}")

            return func(*args, **kwargs)

        return wrapper

    return decorator


def check_tensor(tensor: Tensor, min_val: float, max_val: float, warning_min: float, warning_max: float, context: str):
    if torch.any(tensor < min_val) or torch.any(tensor > max_val):
        raise ValueError(f"{context} value out of range. Min: {tensor.min().item()}, Max: {tensor.max().item()}")
    if torch.any(tensor < warning_min) or torch.any(tensor > warning_max):
        logger.warning(
            f"{context} value approaching the limits. Min: {tensor.min().item()}, Max: {tensor.max().item()}"
        )


def check_iterable(obj: Any, min_val: float, max_val: float, warning_min: float, warning_max: float, context: str):
    if torch.is_tensor(obj):
        check_tensor(obj, min_val, max_val, warning_min, warning_max, context)
    elif isinstance(obj, Iterable) and not isinstance(obj, str):
        if isinstance(obj, dict):
            for key, value in obj.items():
                check_iterable(value, min_val, max_val, warning_min, warning_max, f"{context}[{key}]")
        else:
            for i, item in enumerate(obj):
                check_iterable(item, min_val, max_val, warning_min, warning_max, f"{context}[{i}]")


def check_range(
    dtype_in: torch.dtype, dtype_out: torch.dtype | None = None, warning_threshold: float = 0.9
) -> Callable:
    min_val_in = min_val_out = torch.finfo(dtype_in).min
    max_val_in = max_val_out = torch.finfo(dtype_in).max
    warning_min_in = warning_min_out = min_val_in * warning_threshold
    warning_max_in = warning_max_out = max_val_in * warning_threshold
    if dtype_out is not None:
        min_val_out = torch.finfo(dtype_out).min
        max_val_out = torch.finfo(dtype_out).max
        warning_min_out = min_val_out * warning_threshold
        warning_max_out = max_val_out * warning_threshold

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            check_iterable(args, min_val_in, max_val_in, warning_min_in, warning_max_in, f"{func.__name__} args")
            check_iterable(kwargs, min_val_in, max_val_in, warning_min_in, warning_max_in, f"{func.__name__} kwargs")
            result = func(*args, **kwargs)
            check_iterable(
                result, min_val_out, max_val_out, warning_min_out, warning_max_out, f"{func.__name__} return"
            )

            return result

        return wrapper

    return decorator


def count_calls(func: Callable) -> Callable:
    count = 0

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal count
        count += 1
        return func(*args, n_calls=count, **kwargs)

    return wrapper


def assign_params_groups(model: nn.Module) -> tuple[list[Parameter], list[Parameter]]:
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = {n: p for n, p in param_dict.items() if p.dim() >= 2}
    nodecay_params = {n: p for n, p in param_dict.items() if p.dim() < 2}

    num_params = sum(p.numel() for p in param_dict.values())
    num_decay_params = sum(p.numel() for p in decay_params.values())
    num_nodecay_params = sum(p.numel() for p in nodecay_params.values())
    assert num_params == num_decay_params + num_nodecay_params, (
        f"Decomposition of parameters failed: {num_params} != {num_decay_params} + {num_nodecay_params}"
    )

    logger.debug(f"Params to decay: {decay_params.keys()}")
    logger.debug(f"Params not to decay: {nodecay_params.keys()}")
    return list(decay_params.values()), list(nodecay_params.values())


def inverse_sigmoid(x: Tensor) -> Tensor:
    x = x.clamp(0, 1)
    x1 = x.clamp(min=1e-6)
    x2 = (1 - x).clamp(min=1e-6)
    return torch.log(x1 / x2)


@torch.jit.script
def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """
    Converts bounding boxes from (center_x, center_y, width, height) to (x_min, y_min, x_max, y_max).

    Args:
        boxes (Tensor): A tensor of shape (N, 4) with boxes in cxcywh format.

    Returns:
        Tensor: A tensor of shape (N, 4) with boxes in xyxy format.
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


@torch.jit.script
def box_xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    """
    Converts bounding boxes from (x_min, y_min, x_max, y_max) to (center_x, center_y, width, height).

    Args:
        boxes (Tensor): A tensor of shape (N, 4) with boxes in xyxy format.

    Returns:
        Tensor: A tensor of shape (N, 4) with boxes in cxcywh format.
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


@torch.jit.script
def box_xywh_to_xyxy(boxes: Tensor) -> Tensor:
    """
    Converts bounding boxes from (x_top_left, y_top_left, width, height) to (x_min, y_min, x_max, y_max).

    Args:
        boxes (Tensor): A tensor of shape (N, 4) with boxes in xywh format.

    Returns:
        Tensor: A tensor of shape (N, 4) with boxes in xyxy format.
    """
    x, y, w, h = boxes.unbind(-1)
    x2 = x + w
    y2 = y + h
    return torch.stack([x, y, x2, y2], dim=-1)


@torch.jit.script
def box_xyxy_to_xywh(boxes: Tensor) -> Tensor:
    """
    Converts bounding boxes from (x_min, y_min, x_max, y_max) to (x_top_left, y_top_left, width, height).

    Args:
        boxes (Tensor): A tensor of shape (N, 4) with boxes in xyxy format.

    Returns:
        Tensor: A tensor of shape (N, 4) with boxes in xywh format.
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    w = x2 - x1
    h = y2 - y1
    return torch.stack([x1, y1, w, h], dim=-1)


@torch.jit.script
def box_cxcywh_to_xywh(boxes: Tensor) -> Tensor:
    """
    Converts bounding boxes from (center_x, center_y, width, height) to (x_top_left, y_top_left, width, height).

    Args:
        boxes (Tensor): A tensor of shape (N, 4) with boxes in cxcywh format.

    Returns:
        Tensor: A tensor of shape (N, 4) with boxes in xywh format.
    """
    cx, cy, w, h = boxes.unbind(-1)
    x = cx - 0.5 * w
    y = cy - 0.5 * h
    return torch.stack([x, y, w, h], dim=-1)


@torch.jit.script
def box_xywh_to_cxcywh(boxes: Tensor) -> Tensor:
    """
    Converts bounding boxes from (x_top_left, y_top_left, width, height) to (center_x, center_y, width, height).

    Args:
        boxes (Tensor): A tensor of shape (N, 4) with boxes in xywh format.

    Returns:
        Tensor: A tensor of shape (N, 4) with boxes in cxcywh format.
    """
    x, y, w, h = boxes.unbind(-1)
    cx = x + 0.5 * w
    cy = y + 0.5 * h
    return torch.stack([cx, cy, w, h], dim=-1)


@torch.jit.script
def box_iou(preds: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
    """
    A robust version of IoU that handles degenerate (invalid) predicted boxes.
    It returns the IoU and the Union area.

    - It does not assert that preds are valid (x2 >= x1).
    - It ensures the area of degenerate pred boxes is 0, not negative.
    - It still asserts that ground truth targets are valid.
    """
    # Ground truth should always be valid
    assert (targets[:, 2:] >= targets[:, :2]).all()

    # --- Robust Area Calculation for Predictions ---
    # Clamp width and height at 0 to avoid negative area
    preds_w = (preds[:, 2] - preds[:, 0]).clamp(min=0)
    preds_h = (preds[:, 3] - preds[:, 1]).clamp(min=0)
    area1 = preds_w * preds_h
    area2 = box_area(targets)

    # --- Standard Intersection Calculation ---
    lt = torch.max(preds[:, None, :2], targets[:, :2])
    rb = torch.min(preds[:, None, 2:], targets[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    # --- Union and IoU Calculation ---
    union = area1[:, None] + area2 - inter
    iou = inter / union.clamp(min=1e-6)

    return iou, union


@torch.jit.script
def generalized_box_iou(preds: Tensor, targets: Tensor) -> Tensor:
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(preds)
    and M = len(targets)
    """
    # Get IoU and Union from the robust IoU function
    iou, union = box_iou(preds, targets)

    # --- GIoU Penalty Term Calculation ---
    lt_enclose = torch.min(preds[:, None, :2], targets[:, :2])
    rb_enclose = torch.max(preds[:, None, 2:], targets[:, 2:])

    wh_enclose = (rb_enclose - lt_enclose).clamp(min=0)
    area_enclose = wh_enclose[:, :, 0] * wh_enclose[:, :, 1]

    giou = iou - (area_enclose - union) / area_enclose.clamp(min=1e-6)
    return giou


@torch.no_grad()
@torch.jit.script
def get_circles_from_masks(masks: Tensor) -> Tensor:
    """
    Computes an enclosing circle for each mask based on its centroid and the farthest foreground pixel.
    Accepts masks in pixel coordinates and returns circles in normalized [0, 1] coordinates.

    Args:
        masks: A tensor of shape (N, H, W) where N is the number of masks,
               and H, W are the dimensions in pixels.

    Returns:
        A tensor of shape (N, 3) containing the normalized circle coordinates (cx, cy, r),
        where cx, cy, and r are all in the [0, 1] range.
    """
    if masks.numel() == 0:
        return torch.empty((0, 3), device=masks.device, dtype=torch.float32)

    N, H, W = masks.shape
    device = masks.device

    # Ensure mask is float for calculations
    binary_masks = (masks > 0).float()

    # Create a normalized coordinate grid for the output
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    # Normalize grid coordinates to [0, 1]
    grid = torch.stack([(x_coords + 0.5) / W, (y_coords + 0.5) / H], dim=-1)

    # Vectorize operations
    grid_flat = grid.view(-1, 2)  # Shape: [H*W, 2]
    masks_flat = binary_masks.view(N, -1)  # Shape: [N, H*W]

    # --- 1. Center of Mass (already in normalized coordinates) ---
    mass = masks_flat.sum(dim=1).clamp(min=1e-6)
    center = (masks_flat @ grid_flat) / mass.unsqueeze(-1)
    cx, cy = center.unbind(-1)

    # --- 2. Radius (calculated in normalized space) ---
    # Calculate squared distance from each grid point to the center for each mask
    dist_sq = ((grid_flat.unsqueeze(0) - center.unsqueeze(1)) ** 2).sum(dim=-1)
    # Apply the mask to only consider foreground pixels
    dist_sq_masked = dist_sq * masks_flat
    # The radius is the sqrt of the max squared distance (farthest point)
    radius = torch.sqrt(dist_sq_masked.max(dim=1).values)

    return torch.stack([cx, cy, radius], dim=-1)


@torch.no_grad()
@torch.jit.script
def get_boxes_from_masks(masks: Tensor) -> Tensor:
    """
    Computes bounding boxes from a batch of masks.
    Accepts masks (logits or binary) and returns boxes in normalized cxcywh format.

    Args:
        masks: A tensor of shape (N, H, W) where N is the number of masks.

    Returns:
        A tensor of shape (N, 4) containing normalized box coordinates (cx, cy, w, h).
    """
    if masks.numel() == 0:
        return torch.empty((0, 4), device=masks.device, dtype=torch.float32)

    N, H, W = masks.shape
    device = masks.device
    binary_masks = masks > 0
    is_empty = ~binary_masks.any(dim=-1).any(dim=-1)

    y_coords = torch.arange(H, dtype=torch.float32, device=device)
    x_coords = torch.arange(W, dtype=torch.float32, device=device)
    y_coords, x_coords = torch.meshgrid(y_coords, x_coords, indexing="ij")

    # Expand coordinates to [N, H, W] for masked ops
    x_coords = x_coords.unsqueeze(0).expand(N, -1, -1).clone()
    y_coords = y_coords.unsqueeze(0).expand(N, -1, -1).clone()

    # Refined: Apply masked_fill directly to the coordinate grids
    x_min = x_coords.masked_fill(~binary_masks, 1e8).flatten(1).min(-1)[0]
    x_max = x_coords.masked_fill(~binary_masks, -1e8).flatten(1).max(-1)[0]  # Use -1e8 for max
    y_min = y_coords.masked_fill(~binary_masks, 1e8).flatten(1).min(-1)[0]
    y_max = y_coords.masked_fill(~binary_masks, -1e8).flatten(1).max(-1)[0]  # Use -1e8 for max

    # For empty masks, overwrite the resulting inf/-inf values
    x_min[is_empty] = 0
    x_max[is_empty] = 0
    y_min[is_empty] = 0
    y_max[is_empty] = 0

    cx = (x_min + x_max) / 2.0 / W
    cy = (y_min + y_max) / 2.0 / H
    w = (x_max - x_min) / W
    h = (y_max - y_min) / H

    return torch.stack([cx, cy, w, h], dim=-1)


@torch.jit.script
def get_circles_from_boxes(boxes: Tensor, size: tuple[int, int] | None = None, method: str = "circumscribed") -> Tensor:
    """
    Computes a circle from bounding boxes.

    Args:
        boxes: A tensor of shape (N, 4) with boxes in xyxy format.
        size: The image size (H, W). If provided, input `boxes` are assumed
              to be in pixel coordinates. If None, input `boxes` are assumed
              to be in normalized coordinates.
        method: The method to use for circle calculation.
                "circumscribed": Smallest circle enclosing the box corners.
                "inscribed": Largest circle that fits inside the box.

    Returns:
        A tensor of shape (N, 3) containing the normalized circle
        coordinates (cx, cy, r).
    """
    if boxes.numel() == 0:
        return torch.empty((0, 3), device=boxes.device, dtype=torch.float32)

    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    box_w = (x2 - x1).clamp(min=0)
    box_h = (y2 - y1).clamp(min=0)

    if method == "circumscribed":
        # The radius of the circle enclosing the box is half the diagonal.
        r = torch.sqrt(box_w**2 + box_h**2) / 2
    elif method == "inscribed":
        # The radius of the circle inscribed in the box is half the minimum side length.
        r = torch.min(box_w, box_h) / 2
    else:
        raise ValueError(f"Unknown method '{method}'. Expected 'circumscribed' or 'inscribed'.")

    if size is not None:
        h, w = float(size[0]), float(size[1])
        cx = cx / w
        cy = cy / h
        r = r / ((h + w) / 2.0)

    return torch.stack([cx, cy, r], dim=-1)


@torch.jit.script
def get_boxes_from_circles(
    circles: Tensor, size: tuple[int, int] | None = None, method: str = "circumscribed"
) -> Tensor:
    """
    Computes the (square) bounding box from a circle.

    Args:
        circles: A tensor of shape (N, 3) containing the normalized
                 circle coordinates (cx, cy, r).
        size: The image size (H, W). If provided, coordinates are denormalized
              and boxes are returned in pixel coordinates. If None, boxes are
              returned in normalized [0, 1] coordinates.
        method: "circumscribed": the smallest square that can enclose the circle.
                "inscribed": the largest square that can fit inside the circle.

    Returns:
        A tensor of shape (N, 4) containing the bounding box coordinates in
        xyxy format [x1, y1, x2, y2]. The coordinates are in pixels if `size`
        is provided, otherwise they are normalized.
    """
    if circles.numel() == 0:
        return torch.empty((0, 4), device=circles.device, dtype=torch.float32)

    # --- 1. Unpack circle coordinates ---
    cx, cy, r = circles.unbind(-1)

    # --- 2. Denormalize if size is provided ---
    if size is not None:
        h, w = size
        # Denormalize center coordinates
        cx = cx * w
        cy = cy * h
        # Denormalize radius using the same factor as in the forward function
        r = r * ((h + w) / 2.0)

    # --- 3. Calculate box dimensions from radius ---
    # We assume the box is a square in the operating coordinate space
    # (pixel space if size is given, normalized space otherwise).
    if method == "circumscribed":
        # For an inscribed circle in a square, radius = side / 2.
        # Therefore, side = 2 * radius.for a circumscribed box.
        box_dim = 2 * r
    elif method == "inscribed":
        # For a circumscribed circle around a square, radius = diagonal / 2.
        # The diagonal is sqrt(2) * side. So, radius = (sqrt(2) * side) / 2.
        # Therefore, side = 2 * radius / sqrt(2) = radius * sqrt(2).for an inscribed box
        sqrt2 = torch.sqrt(torch.tensor(2.0, device=circles.device))
        box_dim = r * sqrt2
    else:
        raise ValueError(f"Unknown method '{method}'. Expected 'inscribed' or 'circumscribed'.")

    # --- 4. Calculate box corners (xyxy) from center and dimensions ---
    half_dim = box_dim / 2
    x1 = cx - half_dim
    y1 = cy - half_dim
    x2 = cx + half_dim
    y2 = cy + half_dim

    # --- 5. Stack and return ---
    return torch.stack([x1, y1, x2, y2], dim=-1)


@torch.jit.script
def generalized_circle_iou(
    preds: Tensor,
    targets: Tensor,
    mode: str = "elementwise",
    return_iou: bool = False,
    eps: float = 1e-6,
) -> Tensor:
    """
    Calculates the Generalized Circle IoU between two sets of circles.

    Args:
        preds: A tensor of shape (N, 3) with circles in normalized cxcyr format
        targets: A tensor of shape (M, 3), for pairwise, or (N, 3) for elementwise,
                 with circles in normalized cxcyr format.
        mode: 'pairwise' for all-pairs comparison, 'elementwise' for one-to-one comparison.

    Returns:
        Tensor: A cost matrix of shape (N, M) for pairwise mode, or a cost
                vector of shape (N,) for elementwise mode.
    """
    if mode == "pairwise":
        x1, y1, r1 = preds.unsqueeze(1).unbind(-1)
        x2, y2, r2 = targets.unsqueeze(0).unbind(-1)
    elif mode == "elementwise":
        if preds.shape != targets.shape:
            raise ValueError(
                f"For 'elementwise' mode, preds and targets must have the same shape, "
                f"but got {preds.shape} and {targets.shape}"
            )
        x1, y1, r1 = preds.unbind(-1)
        x2, y2, r2 = targets.unbind(-1)
    else:
        raise ValueError(f"Unknown mode: {mode}. Expected 'pairwise' or 'elementwise'.")

    # Distance between centers
    d_sq = torch.pow(x1 - x2, 2) + torch.pow(y1 - y2, 2)
    d = torch.sqrt(d_sq.clamp(min=eps))

    # Clamp arguments to acos to the valid range [-1, 1]
    acos_arg1 = (d_sq + r1**2 - r2**2) / (2 * d * r1).clamp(min=eps)
    acos_arg2 = (d_sq + r2**2 - r1**2) / (2 * d * r2).clamp(min=eps)

    # Intersection area calculation
    term1 = r1**2 * torch.acos(acos_arg1.clamp(-1 + eps, 1 - eps))
    term2 = r2**2 * torch.acos(acos_arg2.clamp(-1 + eps, 1 - eps))

    sqrt_term = (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)
    term3 = 0.5 * torch.sqrt(sqrt_term.clamp(min=eps))

    intersection = term1 + term2 - term3
    # Handle containment: if one circle is inside the other, intersection is the area of the smaller circle
    is_contained = d + torch.min(r1, r2) <= torch.max(r1, r2)
    intersection_contained = torch.pi * torch.min(r1, r2) ** 2

    # Set intersection to 0 if circles don't overlap, and handle containment
    intersection = torch.where(d >= r1 + r2, torch.zeros_like(d), intersection)
    intersection = torch.where(is_contained, intersection_contained, intersection)

    # Union area
    union = torch.pi * (r1**2 + r2**2) - intersection
    iou = intersection / union.clamp(min=eps)

    if return_iou:
        return iou

    # Smallest enclosing circle for gCIoU
    # Case 1: One circle is contained within the other. The enclosing circle is the larger circle.
    r_c_contained = torch.max(r1, r2)
    area_c_contained = torch.pi * r_c_contained**2
    # Case 2: Circles are separate.
    r_c_separate = (d + r1 + r2) / 2
    area_c_separate = torch.pi * r_c_separate**2
    # Use the correct area based on whether one circle contains the other
    area_c = torch.where(is_contained, area_c_contained, area_c_separate)

    giou = iou - (area_c - union) / area_c.clamp(min=eps)
    return giou


def tversky_loss(
    logits: Tensor, targets: Tensor, alpha: float = 0.7, beta: float = 0.3, gamma: float = 1.0, eps: float = 1e-6
) -> Tensor:
    p = logits.sigmoid().flatten(1)
    targets = targets.flatten(1).type_as(p)
    tp = (p * targets).sum(1)
    fp = (p * (1 - targets)).sum(1)
    fn = ((1 - p) * targets).sum(1)
    ti = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return (1.0 - ti).mean().clamp_min(1e-6) ** gamma


@torch.jit.script
def calculate_dice_score(preds: Tensor, targets: Tensor) -> Tensor:
    numerator = 2 * (preds * targets).sum(1)
    denominator = preds.sum(-1) + targets.sum(-1)
    return (numerator + 1) / (denominator + 1)


@torch.jit.script
def dice_loss(logits: Tensor, targets: Tensor, reduction: str = "mean") -> Tensor:
    """Compute the Dice loss.

    Args:
        preds: The predicted masks, with shape [N, H, W].
        targets: The ground truth masks, with shape [N, H, W].
        reduction: The reduction method to apply to the loss. Can be "mean", "sum", or "none".

    Returns:
        Tensor: The computed DICE loss, either as a scalar or a tensor of shape [N] depending on reduction.
    """
    if logits.numel() == 0 and targets.numel() == 0:
        if reduction == "none":
            raise ValueError("Both predictions and targets are empty, cannot compute loss.")
        return logits.sum()
    if logits.numel() == 0:
        logits = torch.zeros_like(targets)
    if targets.numel() == 0:
        targets = torch.zeros_like(logits)

    loss = 1 - calculate_dice_score(logits.sigmoid().flatten(1), targets.flatten(1).type_as(logits))

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


@torch.jit.script
def dice_loss_instr(
    logits: Tensor, targets: Tensor, power: float = 0.2, pos_weight: float = 1.0, neg_weight: float = 1.0
) -> Tensor:
    """
    A robust and clean implementation of the DICE loss from the INSTR paper.
    It separates the calculation for positive and negative samples and handles empty inputs.

    If many queries are empty (e.g. Q=100, objects=5), start with neg_weight = 0.25-0.5.
    Increase power (e.g. 0.3-0.5) if negatives are too “soft” (they improve too slowly), or decrease it (<0.2) if they dominate early training.

    Args:
        logits (Tensor): The predicted masks, with shape [N, H, W]. Logits before sigmoid.
        targets (Tensor): The ground truth masks, with shape [N, H, W].
        power (float): The exponent (gamma) for the logarithmic dice loss on negative samples.
        pos_weight (float): The weighting factor for positive (object) samples.
        neg_weight (float): The weighting factor for negative (empty) samples.

    Returns:
        Tensor: The computed DICE loss, either as a scalar or a tensor of shape [N] depending on reduction.
    """

    if logits.numel() == 0 and targets.numel() == 0:
        return logits.sum()
    if logits.numel() == 0:
        logits = torch.zeros_like(targets)
    if targets.numel() == 0:
        targets = torch.zeros_like(logits)

    preds = logits.sigmoid().flatten(1)

    # Find positive (non-empty) and negative (empty) masks
    targets = targets.flatten(1).type_as(preds)
    target_area = targets.sum(-1)
    pos_mask = target_area != 0
    neg_mask = target_area == 0

    # Initialize losses to zero
    loss_pos = torch.tensor(0.0, device=logits.device)
    loss_neg = torch.tensor(0.0, device=logits.device)

    # Calculate loss for positive (object) samples, if any
    if pos_mask.any():
        score_pos = calculate_dice_score(preds[pos_mask], targets[pos_mask])
        loss_pos = (1 - score_pos).mean()

    # Calculate loss for negative (empty) samples, if any
    if neg_mask.any():
        # For negative samples, the loss is on the inverted masks (background)
        score_neg = calculate_dice_score(1 - preds[neg_mask], 1 - targets[neg_mask]).clamp_min(1e-6)
        score_neg_log = -score_neg.log()
        loss_neg = torch.pow(score_neg_log.clamp_min(1e-6), power).mean()

    return pos_weight * loss_pos + neg_weight * loss_neg


@torch.jit.script
def batch_dice_loss(preds: Tensor, targets: Tensor) -> Tensor:
    """
    Compute the DICE loss, similar to generalized IOU for masks.
    This function is adapted from the implementation in matcher_mask_dino.py.

    Args:
        preds: A float tensor of shape [N, H, W]. The predictions for each example.
        targets: A float tensor of shape [M, H, W]. The ground truth for each example.

    Returns:
        A tensor of shape [N, M] representing the DICE loss between each pair of predictions and targets.
    """
    preds = preds.sigmoid().flatten(1)
    targets = targets.flatten(1).type_as(preds)
    numerator = 2 * torch.einsum("nc,mc->nm", preds, targets)
    denominator = preds.sum(-1)[:, None] + targets.sum(-1)[None, :]
    dice_score = (numerator + 1) / (denominator + 1)
    return 1 - dice_score


@torch.jit.script
def batch_binary_ce_or_focal_loss(
    preds: Tensor,
    targets: Tensor,
    focal: bool = False,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tensor:
    """
    Computes a pairwise loss matrix where entry (n, m) is the mean BCE/focal loss
    between preds[n] and targets[m], after flattening per-sample to 1D.

    Args:
        preds: Float logits, shape [N, *]
        targets: Float labels in {0,1}, shape [M, *] (same flattened size as preds)
        focal: If True, use sigmoid focal loss instead of BCE
        alpha: Focal loss alpha (pos-class prior); neg-class uses (1 - alpha)
        gamma: Focal loss gamma

    Returns:
        [N, M] mean loss across the flattened dimension.
    """
    preds = preds.flatten(1)
    targets = targets.flatten(1).to(dtype=preds.dtype)

    if focal:
        pos = focal_loss.sigmoid_focal_loss(preds, torch.ones_like(preds), alpha, gamma)  # [N, C]
        neg = focal_loss.sigmoid_focal_loss(preds, torch.zeros_like(preds), alpha, gamma)  # [N, C]
    else:
        pos = F.binary_cross_entropy_with_logits(preds, torch.ones_like(preds), reduction="none")
        neg = F.binary_cross_entropy_with_logits(preds, torch.zeros_like(preds), reduction="none")

    # Two batched dot products over C -> [N, M]
    loss = pos.matmul(targets.t()) + neg.matmul((1.0 - targets).t())
    return loss / preds.size(1)


@torch.no_grad()
@torch.jit.script
def sample_uncertain(logits: Tensor, num_points: int, importance_sample_ratio: float = 0.75) -> Tensor:
    P, M = logits.shape
    K = min(num_points, M)
    if K >= M or K <= 0:
        return torch.arange(M, device=logits.device).expand(P, M)

    scores = -logits.detach().abs()
    num_imp = int(K * importance_sample_ratio)
    num_rand = K - num_imp
    imp_idx = scores.topk(k=num_imp, dim=1).indices
    if num_rand > 0:
        mask = torch.ones((P, M), dtype=torch.bool, device=logits.device)
        mask.scatter_(1, imp_idx, False)
        probs = mask.to(torch.float32)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1.0)
        rand_idx = torch.multinomial(probs, num_samples=num_rand, replacement=False)
        return torch.cat([imp_idx, rand_idx], dim=1)
    return imp_idx


@torch.no_grad()
def hungarian_matcher(
    batch_size: int,
    mask_logits: Tensor | None = None,
    mask_tgt: list[Tensor] | None = None,
    occ_logits: Tensor | Sequence[Tensor] | None = None,
    occ_tgt: list[Tensor] | Sequence[list[Tensor]] | None = None,
    dets: Tensor | None = None,
    boxes: Tensor | None = None,
    cls_preds: Tensor | None = None,
    mask_weight: float = 1.0,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    occ_weight: float | Sequence[float] = 1.0,
    cls_weight: float = 1.0,
    det_weight: float = 1.0,
    sample: int | Sequence[int | None] | None = None,
    det_type: str = "box",
) -> tuple[list[tuple[Tensor, Tensor]], float]:
    """Performs Hungarian matching.

    Params:
        mask_logits: 2D mask logits [B, N_queries, H, W]
        mask_tgt: List of 2D target masks [B, (N_targets, H, W)]
        cls_preds: Optional classification logits or probabilities [B, N_queries]
        occ_logits: Optional 3D occupancy logits [B, N_queries, N_points]
        occ_tgt: Optional list of 3D target masks [B, (N_targets, N_points)]

    Returns:
        A list of (index_i, index_j) tuples and the average matching cost.
    """
    is_logits = bool(((cls_preds < 0) | (cls_preds > 1)).any()) if cls_preds is not None else False

    occ_logits_levels: list[Tensor] = []
    occ_tgt_levels: list[list[Tensor]] = []
    sample_levels: list[int | None] = []
    occ_weight_levels: list[float] = []

    mask_sample: int | None = None
    if isinstance(sample, int):
        mask_sample = sample
    elif isinstance(sample, Sequence) and len(sample) > 0:
        mask_sample = sample[0]

    if occ_logits is not None:
        if isinstance(occ_logits, Tensor):
            occ_logits_levels = [occ_logits]
        else:
            occ_logits_levels = list(occ_logits)
        if occ_tgt is None:
            raise ValueError("`occ_tgt` must be provided when `occ_logits` is provided")

        if (
            len(occ_logits_levels) == 1
            and isinstance(occ_tgt, list)
            and (len(occ_tgt) == 0 or torch.is_tensor(occ_tgt[0]))
        ):
            occ_tgt_levels = [cast(list[Tensor], occ_tgt)]
        else:
            occ_tgt_levels = list(cast(Sequence[list[Tensor]], occ_tgt))

        if len(occ_tgt_levels) != len(occ_logits_levels):
            raise ValueError("`occ_logits` and `occ_tgt` must have the same number of levels")

        if isinstance(sample, Sequence):
            sample_levels = list(sample)
        else:
            sample_levels = [cast(int | None, sample)] * len(occ_logits_levels)

        if isinstance(occ_weight, Sequence):
            occ_weight_levels = [float(w) for w in occ_weight]
        else:
            occ_weight_levels = [float(occ_weight)] * len(occ_logits_levels)

        if len(sample_levels) != len(occ_logits_levels):
            sample_levels = (sample_levels + [None] * len(occ_logits_levels))[: len(occ_logits_levels)]
        if len(occ_weight_levels) != len(occ_logits_levels):
            occ_weight_levels = (occ_weight_levels + [1.0] * len(occ_logits_levels))[: len(occ_logits_levels)]
        mask_sample = sample_levels[0] if sample_levels else None

    # TODO: Vectorize
    avg_cost = 0.0
    indices: list[tuple[Tensor, Tensor]] = []
    for i in range(batch_size):
        cost_matrix: Tensor | None = None

        def add_cost(cost: Tensor) -> None:
            nonlocal cost_matrix
            cost_matrix = cost if cost_matrix is None else cost_matrix + cost

        # --- 2D Mask Cost ---
        if mask_weight > 0 and mask_logits is not None and mask_tgt is not None:
            pred_masks = mask_logits[i]
            tgt_masks = mask_tgt[i].type_as(pred_masks)

            if mask_sample is not None and mask_sample > 0:
                from detectron2.projects.point_rend.point_features import point_sample

                point_coords = torch.rand(1, int(mask_sample), 2, device=pred_masks.device)
                tgt_masks = point_sample(
                    tgt_masks.unsqueeze(1),
                    point_coords.expand(len(tgt_masks), -1, -1),
                    align_corners=False,
                ).squeeze(1)

                pred_masks = point_sample(
                    pred_masks.unsqueeze(1),
                    point_coords.expand(len(pred_masks), -1, -1),
                    align_corners=False,
                ).squeeze(1)

            cost_dice = dice_weight * batch_dice_loss(pred_masks, tgt_masks)
            cost_mask = bce_weight * batch_binary_ce_or_focal_loss(pred_masks, tgt_masks)
            add_cost(mask_weight * (cost_dice + cost_mask))

        # --- 3D Occupancy Cost ---
        if occ_weight_levels and occ_logits_levels and occ_tgt_levels:
            occ_cost: Tensor | None = None
            for j, (pred_occ, tgt_occ) in enumerate(zip(occ_logits_levels, occ_tgt_levels, strict=False)):
                po, to = pred_occ[i], tgt_occ[i]
                sample_j = sample_levels[j]
                if sample_j is not None and sample_j > 0 and po.numel() > 0:
                    M = po.size(1)
                    K = min(sample_j, M)
                    if K > 0 and K < M:
                        idx_cols = torch.randperm(M, device=po.device)[:K]
                        po = po.index_select(dim=1, index=idx_cols)
                        to = to.index_select(dim=1, index=idx_cols)
                cost_dice_occ = dice_weight * batch_dice_loss(po, to)
                cost_occ = bce_weight * batch_binary_ce_or_focal_loss(po, to)
                weighted_cost = occ_weight_levels[j] * (cost_dice_occ + cost_occ)
                occ_cost = weighted_cost if occ_cost is None else occ_cost + weighted_cost
            if occ_cost is not None:
                add_cost(occ_cost / max(sum(occ_weight_levels), 1.0))

        # --- Detection Cost ---
        if det_weight > 0 and dets is not None:
            pred_dets = dets[i]
            if mask_tgt is None:
                if boxes is None or mask_logits is None:
                    raise ValueError("`boxes` and `mask_logits` must be provided when `mask_tgt` is None")
                h, w = mask_logits.shape[-2:]
                tgt_dets = boxes[i]
                tgt_dets = tgt_dets / torch.tensor([w, h, w, h], device=tgt_dets.device)
                if det_type == "box":
                    tgt_dets = box_xywh_to_cxcywh(tgt_dets)
                elif det_type == "circle":
                    tgt_dets = get_circles_from_boxes(box_xywh_to_xyxy(tgt_dets))
            else:
                if det_type == "box":
                    tgt_dets = get_boxes_from_masks(tgt_masks)
                elif det_type == "circle":
                    tgt_dets = get_circles_from_masks(tgt_masks)

            cost_l1 = torch.cdist(pred_dets, tgt_dets, p=1)
            if det_type == "box":
                pred_dets = box_cxcywh_to_xyxy(pred_dets)
                tgt_dets = box_cxcywh_to_xyxy(tgt_dets)
                cost_giou = -generalized_box_iou(pred_dets, tgt_dets)
            elif det_type == "circle":
                cost_giou = -generalized_circle_iou(pred_dets, tgt_dets, mode="pairwise")
            add_cost(det_weight * (cost_l1 + cost_giou))

        # --- Classification Cost ---
        if cls_weight > 0 and cls_preds is not None:
            pred_cls = cls_preds[i]
            cost_cls = -pred_cls.sigmoid() if is_logits else -pred_cls
            add_cost(cls_weight * cost_cls.unsqueeze(1))

        # Perform matching
        if cost_matrix is None:
            raise ValueError("No valid matching costs were computed")
        row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu())
        row_idx = torch.as_tensor(row_ind, device=cost_matrix.device, dtype=torch.long)
        col_idx = torch.as_tensor(col_ind, device=cost_matrix.device, dtype=torch.long)
        indices.append((row_idx, col_idx))
        avg_cost += torch.nan_to_num(cost_matrix[row_idx, col_idx].mean()).item()

    return indices, avg_cost


@torch.jit.script
def sinkhorn_segmentation_loss(
    preds: Tensor, targets: list[Tensor], temperature: float = 0.05, max_iter: int = 50, reduction: str = "mean"
) -> Tensor:
    """
    Computes a differentiable instance segmentation loss using the Sinkhorn algorithm.

    This function calculates the optimal transport cost between predicted and ground
    truth masks and uses this cost as the loss. It does not return indices.

    Params:
        preds: [batch_size, num_queries, H, W].
        targets: batch_size * [num_target_masks, H, W].
        temperature: Regularization parameter for the Sinkhorn algorithm.
        max_iter: Maximum number of iterations for the Sinkhorn algorithm.
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.

    Returns:
        A scalar tensor representing the loss, or a tensor of shape [batch_size]
        if reduction is 'none'.
    """
    losses = []
    for b in range(preds.size(0)):
        out_mask = preds[b]
        tgt_mask = targets[b]

        num_queries = out_mask.size(0)
        num_targets = tgt_mask.size(0)

        if num_targets == 0:
            # If there are no targets, the loss is the sum of all predictions'
            # confidences (which we can't measure here without logits) or zero.
            # For simplicity, we assign zero cost if there are no targets to match.
            losses.append(torch.tensor(0.0, device=out_mask.device))
            continue

        cost_matrix = batch_dice_loss(out_mask, tgt_mask)

        n = max(num_queries, num_targets)
        padded_cost = torch.full((n, n), 1e9, device=out_mask.device)
        padded_cost[:num_queries, :num_targets] = cost_matrix

        log_P = -padded_cost / temperature

        for _ in range(max_iter):
            log_P = log_P - torch.logsumexp(log_P, dim=1, keepdim=True)
            log_P = log_P - torch.logsumexp(log_P, dim=0, keepdim=True)

        P = log_P.exp()

        loss_b = (P[:num_queries, :num_targets] * cost_matrix).sum()
        losses.append(loss_b)

    final_losses = torch.stack(losses)
    if reduction == "mean":
        return final_losses.mean()
    elif reduction == "sum":
        return final_losses.sum()
    elif reduction == "none":
        return final_losses
    else:
        raise ValueError(f"Invalid reduction type: {reduction}")


@torch.jit.script
def batched_sinkhorn_segmentation_loss(
    preds: Tensor, targets: list[Tensor], temperature: float = 0.05, max_iter: int = 50, reduction: str = "mean"
) -> Tensor:
    """
    Computes a differentiable instance segmentation loss using the Sinkhorn algorithm
    in a fully batched manner, without an explicit Python loop over the batch.

    Params:
        outputs: [batch_size, num_queries, H, W].
        targets: batch_size * [num_target_masks, H, W].
        temperature: Regularization parameter for the Sinkhorn algorithm.
        max_iter: Maximum number of iterations for the Sinkhorn algorithm.
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.

    Returns:
        A scalar tensor representing the loss, or a tensor of shape [batch_size]
        if reduction is 'none'.
    """
    pred_masks = preds
    bs, num_queries, H, W = pred_masks.shape
    device = pred_masks.device

    # 1. Pad ground truth targets to create a single batch tensor
    max_num_targets = max([t.size(0) for t in targets] + [0])

    if max_num_targets == 0:
        # Handle case where there are no targets in the entire batch
        # A proper loss might involve penalizing all predictions, but for simplicity we return 0.
        return torch.tensor(0.0, device=device)

    padded_tgt_masks = torch.zeros(bs, max_num_targets, H, W, device=device)
    tgt_padding_mask = torch.ones(bs, max_num_targets, device=device, dtype=torch.bool)

    for i, t in enumerate(targets):
        num_t = t.size(0)
        if num_t > 0:
            padded_tgt_masks[i, :num_t] = t
            tgt_padding_mask[i, :num_t] = False  # False means it's a real target

    # 2. Compute the Dice cost matrix for the entire batch
    pred_flat = pred_masks.sigmoid().flatten(2)  # Shape: [bs, num_queries, H*W]
    tgt_flat = padded_tgt_masks.flatten(2).float()  # Shape: [bs, max_num_targets, H*W]

    numerator = 2 * torch.einsum("bnx,bmx->bnm", pred_flat, tgt_flat)
    denominator = pred_flat.sum(-1).unsqueeze(2) + tgt_flat.sum(-1).unsqueeze(1)

    dice_score = (numerator + 1) / (denominator + 1)  # Shape: [bs, num_queries, max_num_targets]
    cost_matrices = 1 - dice_score

    # Create a padded version for the Sinkhorn algorithm, but keep the original for the loss calculation
    padded_costs_for_sinkhorn = cost_matrices.clone()
    # Apply padding mask to the cost matrix
    padded_costs_for_sinkhorn.masked_fill_(tgt_padding_mask.unsqueeze(1), 1e9)

    # 3. Pad cost matrices to be square for Sinkhorn
    n = max(num_queries, max_num_targets)
    padded_costs = torch.full((bs, n, n), 1e9, device=device)
    padded_costs[:, :num_queries, :max_num_targets] = padded_costs_for_sinkhorn

    # 4. Run batched Sinkhorn algorithm
    log_P = -padded_costs / temperature

    for _ in range(max_iter):
        log_P = log_P - torch.logsumexp(log_P, dim=2, keepdim=True)
        log_P = log_P - torch.logsumexp(log_P, dim=1, keepdim=True)

    P = log_P.exp()

    # 5. Compute the batched loss using the original, unpadded cost matrix
    batch_loss = (P[:, :num_queries, :max_num_targets] * cost_matrices).sum(dim=[1, 2])

    # 6. Apply reduction
    if reduction == "mean":
        return batch_loss.mean()
    elif reduction == "sum":
        return batch_loss.sum()
    elif reduction == "none":
        return batch_loss
    else:
        raise ValueError(f"Invalid reduction type: {reduction}")


def show_point_cloud_from_depth(
    depth: Tensor,
    intrinsic: Tensor,
    extrinsic: Tensor | None = None,
    normals: Tensor | None = None,
    color: Tensor | None = None,
    frame: bool = True,
):
    if depth.ndim == 3:
        for i in range(len(depth)):
            show_point_cloud_from_depth(
                depth[i],
                intrinsic[i],
                extrinsic[i] if extrinsic is not None else None,
                normals[i] if normals is not None else None,
                color[i] if color is not None else None,
                frame=frame,
            )
        return

    points = depth_to_points(depth, intrinsic)
    if extrinsic is not None:
        points = apply_trafo(points, extrinsic)
    points_np = points.float().cpu().numpy() if torch.is_tensor(points) else np.asarray(points, dtype=np.float32)
    o3d_module = cast(Any, o3d)
    pcd = o3d_module.geometry.PointCloud(o3d_module.utility.Vector3dVector(points_np))
    frame_mesh = o3d_module.geometry.TriangleMesh.create_coordinate_frame(size=0.1) if frame else None
    o3d_module.visualization.draw_geometries([pcd, frame_mesh] if frame_mesh is not None else [pcd])


def show_image_with_masks(
    image: Tensor,
    masks: Tensor | None = None,
    boxes: Tensor | None = None,
    classes: Tensor | None = None,
    scores: Tensor | None = None,
):
    image_np = image.permute(1, 2, 0).cpu().numpy()

    img_min = image_np.min()
    img_max = image_np.max()
    image_np = (image_np - img_min) / (img_max - img_min)
    image_np = (image_np * 255).astype(np.uint8)

    plt.figure(figsize=(8, 8))
    try:
        from detectron2.structures.instances import Instances
        from detectron2.utils.visualizer import Visualizer

        instances = cast(Any, Instances(image_np.shape[:2]))
        if boxes is not None:
            boxes_np = boxes.cpu().numpy()
            boxes_xyxy = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in boxes_np]  # XYWH -> XYXY
            instances.pred_boxes = boxes_xyxy
        if classes is not None:
            instances.pred_classes = classes
        if scores is not None:
            instances.scores = scores
        if masks is not None:
            instances.pred_masks = masks

        v = Visualizer(image_np)
        out = v.draw_instance_predictions(instances.to("cpu"))

        plt.imshow(out.get_image())
    except ImportError:
        plt.imshow(image_np)

        if masks is not None:
            mask_array = masks.cpu().detach().numpy()
            for mask in mask_array:
                color = np.array([*np.random.uniform(size=3), 0.5])

                mask_rgba = np.zeros((*mask.shape, 4))
                mask_rgba[mask > 0] = color

                plt.imshow(mask_rgba)

    plt.axis("off")
    plt.tight_layout()
    plt.show()


class DepthLoss(nn.Module):
    def __init__(self, alpha: float = 0.0, trim_percentage: float = 0.0):
        """Initializes the Scale and Shift Invariant Loss module.

        Args:
            alpha (float): The weight for the gradient matching term.
            trim_percentage (float): The percentage of largest errors to trim for robustness.
        """
        super().__init__()
        self.alpha = alpha
        self.trim_percentage = trim_percentage

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_x: Tensor
        self.sobel_y: Tensor
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, prediction: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
        # If no mask is provided, create one that considers all pixels valid
        if mask is None:
            mask = torch.ones_like(prediction, dtype=torch.bool, device=prediction.device)

        # Ensure mask is boolean
        mask = mask > 0

        # Calculate the robust MAE loss
        l_ssi = self.calculate_robust_mae(prediction, target, mask)

        # Optionally add the gradient matching loss
        if self.alpha > 0:
            l_gm = self.calculate_gradient_matching(prediction, target, mask)
            return (1 - self.alpha) * l_ssi + self.alpha * l_gm
        return l_ssi

    def align_and_scale(self, t: Tensor, mask: Tensor) -> Tensor:
        """
        Aligns the tensor to have zero translation and unit scale.
        """
        # Flatten tensors for batch processing
        t_flat = t.flatten(1)
        mask_flat = mask.flatten(1)

        # Replace masked-out values with 'inf' to ignore them in median calculation
        t_masked = t_flat.clone()
        t_masked[~mask_flat] = float("inf")

        # Calculate median on valid pixels
        t_median, _ = torch.median(t_masked, dim=1, keepdim=True)
        t_median = t_median.view(-1, 1, 1)

        # Align translation to have zero median
        t_aligned = t - t_median

        # Calculate scale using mean absolute deviation on valid pixels
        abs_dev_sum = torch.sum(torch.abs(t_aligned) * mask, dim=(1, 2), keepdim=True)
        valid_pixel_count = torch.sum(mask, dim=(1, 2), keepdim=True)

        t_scale = torch.where(
            valid_pixel_count > 0, abs_dev_sum / valid_pixel_count, torch.tensor(1.0, device=t.device)
        )

        return t_aligned / t_scale.clamp(min=1e-6)

    def calculate_robust_mae(self, prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """
        Calculates the (trimmed) Mean Absolute Error (MAE) between aligned prediction and target.
        """
        pred_aligned = self.align_and_scale(prediction, mask)
        target_aligned = self.align_and_scale(target, mask)

        error = torch.abs(pred_aligned - target_aligned)

        # If trim_percentage is 0, use a fast vectorized path
        if not self.trim_percentage:
            # Calculate MAE only on valid pixels
            masked_error = error * mask
            return masked_error.sum() / mask.sum()

        # Otherwise, use the loop for correct per-image trimming
        total_loss = torch.tensor(0.0, device=prediction.device)
        for i in range(len(prediction)):
            # Select only valid error pixels for the current image
            valid_error = torch.masked_select(error[i], mask[i])

            # Determine how many pixels to keep based on the number of *valid* pixels
            if valid_error.numel() == 0:
                continue  # Skip images with no valid pixels

            num_pixels_to_keep = int(valid_error.shape[0] * (1 - self.trim_percentage))

            # Sort only the valid errors and trim
            trimmed_errors, _ = torch.sort(valid_error)
            trimmed_errors = trimmed_errors[:num_pixels_to_keep]

            total_loss += torch.mean(trimmed_errors)

        return total_loss / len(prediction)

    def calculate_gradient_matching(self, prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """
        Calculates the gradient matching loss to enforce sharp discontinuities.
        """
        pred_aligned = self.align_and_scale(prediction, mask)
        target_aligned = self.align_and_scale(target, mask)

        # Unsqueeze for conv2d
        pred_aligned = pred_aligned.unsqueeze(1)
        target_aligned = target_aligned.unsqueeze(1)
        mask_unsqueezed = mask.unsqueeze(1)

        # Calculate gradients using pre-registered Sobel filters
        sobel_x = cast(Tensor, self.sobel_x)
        sobel_y = cast(Tensor, self.sobel_y)
        pred_grad_x = nn.functional.conv2d(pred_aligned, sobel_x, padding="same")
        pred_grad_y = nn.functional.conv2d(pred_aligned, sobel_y, padding="same")
        target_grad_x = nn.functional.conv2d(target_aligned, sobel_x, padding="same")
        target_grad_y = nn.functional.conv2d(target_aligned, sobel_y, padding="same")

        # Calculate loss on valid pixels
        grad_x_loss = torch.abs(pred_grad_x - target_grad_x) * mask_unsqueezed
        grad_y_loss = torch.abs(pred_grad_y - target_grad_y) * mask_unsqueezed

        return grad_x_loss.mean() + grad_y_loss.mean()


@torch.no_grad()
def labels_from_masks(masks: Tensor, template: Tensor) -> Tensor:
    if masks.numel() == 0:
        return torch.zeros_like(template)
    inst_ids = torch.arange(1, masks.size(0) + 1, device=masks.device, dtype=template.dtype).view(
        masks.size(0), *([1] * (masks.dim() - 1))
    )
    return (masks.to(template.dtype) * inst_ids).amax(dim=0).reshape_as(template)


@torch.no_grad()
@torch.jit.script
def labels_from_logits(logits: Tensor, threshold: float = 0.5) -> Tensor:
    """
    Collapse per-query mask logits into a per-point integer label map.

    Args:
        logits: (B, Q, *S) raw mask logits (no sigmoid).
        threshold: points whose best probs <= threshold become background (0).

    Returns:
        labels: (B, *S) long tensor. 0 = background, 1..Q = winning query indices.
    """
    log_threshold = math.log(threshold / (1.0 - threshold))
    max_vals, argmax_ids = logits.max(dim=1)
    labels = argmax_ids + 1
    labels[max_vals <= log_threshold] = 0
    return labels


@torch.no_grad()
def masks_from_labels(
    labels: torch.Tensor,
    reindex: bool = False,
    drop_background: bool = True,
) -> list[torch.Tensor]:
    """
    Convert integer label maps into per-instance (or per-class) boolean masks.

    Semantics (unchanged from original):
      - reindex=True  -> only ids actually present (after optional background removal), contiguous K = #present.
      - reindex=False -> dense one-hot over [0..max_id], then optionally remove channel 0.

    Args:
        labels: (B, *S) long tensor. 0 = background.
        reindex: See above.
        drop_background: Exclude background channel (id 0) from output.

    Returns:
        List length B; each element (K_b, *S) bool (K_b may be 0).
    """
    B = labels.size(0)
    spatial_shape = labels.shape[1:]

    # Preserve batch shape even for zero-sized spatial dims
    if labels.numel() == 0:
        empty = labels.new_empty((0, *spatial_shape), dtype=torch.bool)
        return [empty for _ in range(B)]

    out: list[torch.Tensor] = []

    for b in range(B):
        labels_b = labels[b]

        if labels_b.numel() == 0:
            out.append(labels_b.new_empty((0, *spatial_shape), dtype=torch.bool))
            continue

        # Fast background-only check
        if not (labels_b > 0).any():
            if drop_background:
                out.append(labels_b.new_empty((0, *spatial_shape), dtype=torch.bool))
            else:
                out.append((labels_b == 0).unsqueeze(0))  # single background mask
            continue

        l_max = int(labels_b.max().item())

        if reindex:
            # Collect only present ids
            present_ids = torch.unique(labels_b)  # sorted
            if drop_background:
                present_ids = present_ids[present_ids > 0]
            if present_ids.numel() == 0:
                out.append(labels_b.new_empty((0, *spatial_shape), dtype=torch.bool))
                continue
            K = present_ids.numel()
            # Broadcast compare: (K, *S)
            masks = (labels_b.unsqueeze(0) == present_ids.view(K, *([1] * labels_b.dim()))).bool()
            out.append(masks)
        else:
            # Dense one-hot over full 0..max range (preserve id->channel mapping)
            flat = labels_b.view(-1)
            one_hot = torch.nn.functional.one_hot(flat, num_classes=l_max + 1).bool().transpose(0, 1)
            masks = one_hot.view(l_max + 1, *spatial_shape)
            if drop_background:
                masks = masks[1:]  # drop channel 0
            out.append(masks)

    return out


@torch.no_grad()
def masks_from_logits(
    logits: Tensor,
    scores: Tensor | None = None,
    threshold: float = 0.5,
    apply_filter: bool = True,
    min_size: int | None = None,
    nms_iou: float | None = None,
) -> list[Tensor]:
    """
    Build an exclusive, non-overlapping set of instance masks per batch item by:
      1) thresholding raw logits to masks,
      2) filtering tiny/duplicate masks (same as AP),
      3) greedily carving exclusive masks in descending score order.

    Args:
        logits: (B, Q, *S) raw mask logits (no sigmoid).
        scores: (B, Q) per-query scores. If None, uses cls_probs_from_logits(logits).
        threshold: threshold to binarize masks.
        min_size: minimum foreground size (in voxels/pixels/points) to keep a mask.
        nms_iou: IoU threshold for duplicate suppression
    Returns:
        List length B. Each element is a bool tensor (K, *S) with exclusive masks.
    """
    log_threshold = math.log(threshold / (1.0 - threshold))
    B = len(logits)
    if scores is None:
        scores = cls_probs_from_logits(logits, threshold=threshold, min_size=min_size)  # (B, Q)

    out: list[Tensor] = []
    for i in range(B):
        # 1) Raw binary instance candidates
        inst_masks = logits[i]  # (Q, *S)
        inst_scores = scores[i]  # (Q,)

        # 2) Filter tiny/duplicate instances like AP
        if apply_filter:
            try:
                inst_masks, inst_scores, _ = filter_instance_masks(
                    masks=inst_masks, scores=inst_scores, min_size=min_size, nms_iou=nms_iou
                )
            except RuntimeError:
                logger.warning("filter_instance_masks failed, returning unfiltered masks")

        # 3) Greedy carve to ensure exclusivity (highest score first)
        if inst_masks.numel() == 0 or len(inst_masks) == 0:
            out.append(inst_masks)  # empty
            continue

        order = inst_scores.argsort(descending=True)
        taken = torch.zeros_like(inst_masks[0], dtype=torch.bool)
        pq_list: list[Tensor] = []
        for idx in order:
            m = (inst_masks[idx] > log_threshold) & ~taken
            if m.any():
                pq_list.append(inst_masks[idx])
                taken |= m

        if pq_list:
            out.append(torch.stack(pq_list, dim=0))
        else:
            out.append(inst_masks[:0])

    return out


@torch.no_grad()
@torch.jit.script
def filter_instance_masks(
    masks: Tensor,
    scores: Tensor,
    min_size: int | None = None,
    max_num: int | None = None,
    nms_iou: float | None = None,
    threshold: float | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    num_masks = len(masks)
    if masks.numel() == 0 or num_masks == 0 or (min_size is None and nms_iou is None):
        return masks, scores, torch.empty(0, dtype=torch.long, device=masks.device)

    # Track original indices
    keep_idx = torch.arange(num_masks, device=masks.device)

    flat = masks.flatten(1)
    if max_num is not None and flat.size(1) > max_num:
        sample_idx = torch.randperm(flat.size(1), device=flat.device)[:max_num]
        flat = flat[:, sample_idx]
        if min_size is not None:
            scaled_min_size = min_size * (max_num / float(flat.size(1)))
            min_size = max(1, int(scaled_min_size))
    if flat.dtype != torch.bool:
        flat = flat.sigmoid()
        if threshold is not None and threshold > 0:
            flat = flat > threshold

    # Area filtering
    areas = flat.sum(dim=1)
    keep = areas > 0
    if min_size is not None:
        keep = keep & (areas >= min_size)
    if int(keep.sum()) != keep.numel():
        masks = masks[keep]
        scores = scores[keep]
        flat = flat[keep]
        areas = areas[keep]
        keep_idx = keep_idx[keep]
        num_masks = len(masks)
        if num_masks == 0:
            return masks, scores, keep_idx  # empty after filtering

    # NMS
    if nms_iou is not None and num_masks > 1:
        order = torch.argsort(scores, descending=True)
        taken = torch.zeros_like(scores, dtype=torch.bool)
        kept_masks = []
        kept_scores = []
        kept_indices: list[Tensor] = []

        for t in range(int(order.numel())):
            idx = int(order[t].item())
            if taken[idx]:
                continue
            ref_flat = flat[idx]
            kept_masks.append(masks[idx : idx + 1])
            kept_scores.append(scores[idx : idx + 1])
            kept_indices.append(keep_idx[idx : idx + 1])

            inter = (ref_flat * flat).sum(1)
            union = areas[idx] + areas - inter
            iou = inter / (union + 1e-6)
            dup = iou >= nms_iou
            taken = taken | dup

        masks = torch.cat(kept_masks, dim=0)
        scores = torch.cat(kept_scores, dim=0)
        keep_idx = torch.cat(kept_indices, dim=0)

    return masks, scores, keep_idx


@torch.no_grad()
def index_from_match(match: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    """
    Flatten a batch-wise matching list into index tensors.

    Args:
        match: list length B of (src_indices, tgt_indices)

    Returns:
        batch_idx: (K,) -> which batch element
        src_idx:   (K,) -> source (prediction) indices
    """
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(match)])
    src_idx = torch.cat([src for (src, _) in match])
    return batch_idx, src_idx


@torch.no_grad()
def cls_probs_from_logits(
    logits: Tensor,
    threshold: float = 0.5,
    method: Literal["mean", "top_k", "focal", "certainty", "adaptive", "ensemble", "peakiness"] = "mean",
    min_size: int | None = None,
    temperature: float = 1.0,
    top_k_ratio: float = 0.3,
    use_expected_size_for_gate: bool = False,
) -> Tensor:
    logits_flat = logits.flatten(2)  # (B, Q, M)
    probs = torch.sigmoid(logits_flat / temperature)
    log_threshold = math.log(threshold / (1.0 - threshold))

    # Foreground masks and sizes
    fg_mask = logits_flat > log_threshold
    fg_count = fg_mask.sum(-1).float()  # thresholded size
    exp_size = probs.sum(-1)  # expected size (soft)

    # Basic scores
    scores: dict[str, Tensor] = {}

    if method in ["mean", "ensemble", "adaptive"]:
        # 1) Foreground-weighted mean (current best)
        if min_size:
            probs_fg = probs * fg_mask
            mean_fg = probs_fg.sum(-1) / fg_count.clamp_min(1.0)
            scores["mean"] = mean_fg
        else:
            scores["mean"] = probs.mean(dim=-1)

    if method in ["certainty", "ensemble", "adaptive"]:
        # 2) Uncertainty (entropy-based)
        entropy = F.binary_cross_entropy_with_logits(logits_flat / temperature, probs, reduction="none")
        scores["certainty"] = 1 - entropy.mean(dim=-1)

    if method in ["focal", "ensemble"]:
        # 3) Focal-inspired confidence (penalize ambiguous points)
        # Use symmetric p*(1 - |2p-1|) damping; higher when p≈0.5 -> reduce confidence
        damp = 1.0 - (4.0 * probs * (1.0 - probs))  # in [0,1], max near p=0.5
        scores["focal"] = (probs * damp).mean(dim=-1)

    if method in ["top_k", "ensemble"]:
        # 4) Top-k stability: prefer confident cores
        # Prefer top-k within predicted foreground; fallback to global if no fg
        k_all = max(1, int(top_k_ratio * probs.size(-1)))
        # Compute per-(B,Q) k within foreground; fallback to k_all if no fg
        # Gather foreground probs with masking; handle raggedness via sorting trick
        p = probs
        # Very large negative for masked-out entries so they never enter topk
        masked = torch.where(fg_mask, p, torch.full_like(p, float("-inf")))
        topk_vals, _ = torch.topk(masked, k=min(k_all, p.size(-1)), dim=-1)
        # If a row had no fg, topk_vals will be -inf; replace with global topk
        no_fg = ~fg_mask.any(dim=-1)  # (B,Q)
        if no_fg.any():
            topk_global, _ = torch.topk(p, k=k_all, dim=-1)
            # broadcast assign back where no_fg
            topk_vals = torch.where(no_fg.unsqueeze(-1), topk_global, topk_vals)
        finite_mask = topk_vals.isfinite()
        topk_vals = torch.where(finite_mask, topk_vals, torch.zeros_like(topk_vals))
        scores["top_k"] = (topk_vals.sum(dim=-1) / finite_mask.sum(dim=-1).clamp_min(1)).nan_to_num(0.0)

    if method in ["peakiness", "ensemble"]:
        # 5) Peakiness: average per-point confidence magnitude; focus on fg if present
        conf_point = 1.0 - (4.0 * probs * (1.0 - probs))  # in [0,1], high near 0 or 1
        if fg_mask.any():
            num = (conf_point * fg_mask).sum(-1)
            den = fg_count.clamp_min(1.0)
            scores["peakiness"] = num / den
        else:
            scores["peakiness"] = conf_point.mean(dim=-1)

    # Combine scores
    if method == "ensemble":
        # Fixed order to avoid relying on dict insertion
        parts = [
            scores["mean"],
            scores["certainty"],
            scores["focal"],
            scores["top_k"],
            scores["peakiness"],
        ]
        w = torch.tensor([0.4, 0.2, 0.2, 0.1, 0.1], device=logits.device, dtype=parts[0].dtype)
        combined = torch.stack([w[i] * parts[i] for i in range(len(parts))], dim=0).sum(dim=0)
        out = combined
    elif method == "adaptive":
        # Blend based on (soft) size ratio
        total_size = logits_flat.size(-1)
        size_ratio = (exp_size / float(total_size)).clamp(0, 1)
        alpha = torch.sigmoid(10 * (size_ratio - 0.1))  # small -> certainty, large -> mean
        out = alpha * scores["mean"] + (1 - alpha) * scores["certainty"]
    else:
        out = scores[method]

    # Uniform min_size gate (applied to any method)
    if min_size:
        size_for_gate = exp_size if use_expected_size_for_gate else fg_count
        out = torch.where(size_for_gate >= float(min_size), out, out.new_zeros(()).expand_as(out))

    return out.clamp(0.0, 1.0)


def queries_from_feat(
    queries: Tensor,
    scores: Tensor | None = None,
    num_queries: int = 100,
    select: Literal["max", "mean", "norm", "fps"] | None = None,
    gather: Literal["hard", "soft", "gumbel"] = "hard",
    ste: bool = False,
    tau: float = 0.5,
    beta: float = 0.75,
) -> Tensor:
    """
    Args:
        queries: (B, S, D) feature vectors to select from
        scores: optional (B, S) scores for each vector
        num_queries: number of queries to keep (clamped to S)
        select: how to score points
        gather: how to gather (hard index, soft weighting, or gumbel sequential sampling)
        ste: straight-through for soft/gumbel
        tau: temperature for softmax / gumbel
        beta: bias term to sharpen soft selection around chosen indices
    """
    x = queries  # (B, S, D)
    B, S, D = x.shape
    if S == 0:
        return x.new_empty((B, 0, D))
    num_queries = min(num_queries, S)

    if select == "fps":
        if gather != "hard":
            raise ValueError("gather='soft' or 'gumbel' not supported when select='fps'")
        with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=False):
            # PCA per batch element -> project to 3D for spatial diversity
            proj = torch.empty((B, S, 3), device=x.device, dtype=torch.float32)
            for b in range(B):
                xb = x[b].float()  # (S, D)
                # q capped for efficiency
                q = min(16, D)
                _, _, V = torch.pca_lowrank(xb, q=q, center=True)
                # V: (D, q); keep first 3 components (pad if q < 3)
                k = min(3, V.size(1))
                comp = xb @ V[:, :k]  # (S, k)
                if k < 3:
                    # zero-pad to 3 dims for FPS kernel expectation
                    pad = comp.new_zeros(S, 3 - k)
                    comp = torch.cat([comp, pad], dim=1)
                proj[b] = comp
            x_idx = cast(Tensor, furthest_point_sample(proj, num_samples=num_queries, return_indices=True))
    else:
        if scores is None:
            if select == "max":
                scores = x.max(dim=-1)[0]
            elif select == "mean":
                scores = x.mean(dim=-1)
            elif select == "norm":
                scores = x.norm(dim=-1)
            else:
                raise ValueError(f"Unknown select method: {select}")

        assert scores is not None
        x_idx = torch.topk(scores, k=num_queries, dim=1).indices  # (B, num_queries)

    if gather == "hard":
        # (B, num_queries, D)
        return torch.gather(x, dim=1, index=x_idx.unsqueeze(-1).expand(-1, -1, D))

    # For soft / gumbel we require scores
    if select == "fps":
        raise RuntimeError("Internal logic error: fps with non-hard gather should have raised earlier.")

    if gather == "soft":
        assert scores is not None
        one_hot = F.one_hot(x_idx, num_classes=S).to(dtype=x.dtype, device=x.device)  # (B, num_queries, S)
        # Turn scores into (B, 1, S) logits; bias chosen indices
        logits = scores.unsqueeze(1) + beta * one_hot
        weights = F.softmax(logits / tau, dim=-1)  # (B, num_queries, S)
        if ste:
            weights = one_hot - weights.detach() + weights
        return torch.bmm(weights, x)  # (B, num_queries, D)

    if gather == "gumbel":
        # Sequential Gumbel sampling without replacement
        assert scores is not None
        logits = scores.clone()  # (B, S)
        rows = []
        mask = torch.zeros_like(logits, dtype=torch.bool)
        for _ in range(num_queries):
            y = F.gumbel_softmax(logits.masked_fill(mask, float("-inf")), tau=tau, hard=ste, dim=-1)  # (B, S)
            rows.append(y.unsqueeze(1))
            picked = y.argmax(dim=-1)  # (B,)
            mask = mask.scatter(1, picked.unsqueeze(1), True)
        weights = torch.cat(rows, dim=1)  # (B, num_queries, S)
        return torch.bmm(weights, x)

    raise ValueError(f"Unknown gather method: {gather}")


@torch.enable_grad()
def get_normals(points: Tensor, logits: Tensor, normalize: bool = True) -> Tensor:
    if not points.requires_grad:
        points.requires_grad = True
    grad = torch.autograd.grad(
        outputs=logits.sum(),
        inputs=points,
        create_graph=logits.requires_grad,  # TODO: same as self.training?
        retain_graph=logits.requires_grad,
        only_inputs=True,
        allow_unused=True,
    )[0]
    if grad is None:
        raise RuntimeError("Failed to compute normals: gradient is None")
    if normalize:
        return -grad / grad.norm(2, dim=-1, keepdim=True)
    return -grad


def get_normal_loss(
    normals_pred: Tensor,
    normals_gt: Tensor | None = None,
    mask: Tensor | None = None,
    eikonal: bool = False,
    mode: Literal["smooth_l1", "l1", "l2", "mse", "cos_sim"] = "cos_sim",
) -> Tensor:
    eikonal_loss = torch.zeros((), device=normals_pred.device, dtype=normals_pred.dtype)
    if eikonal:
        eikonal_loss = (normals_pred.norm(2, dim=-1) - 1) ** 2
        if normals_gt is None:
            return eikonal_loss
        normals_pred = normals_pred / normals_pred.norm(2, dim=-1, keepdim=True)
    if normals_gt is None:
        raise ValueError("`normals_gt` must be provided when `eikonal=False`")

    if normals_gt.min() < 0 and normals_gt.size(1) == 3:
        normals_gt = normals_gt.transpose(1, 2)
        normals_gt = normals_gt * torch.tensor([0.229, 0.224, 0.225]).to(normals_gt)
        normals_gt = normals_gt + torch.tensor([0.485, 0.456, 0.406]).to(normals_gt)
        normals_gt = 2 * normals_gt - 1
    elif normals_gt.min() >= 0 and normals_gt.size(2) == 3:
        if normals_gt.max() > 1:
            normals_gt = normals_gt / 255.0
        normals_gt = 2 * normals_gt - 1
    elif normals_gt.size(2) == 3 and normals_gt.min() >= -1 and normals_gt.max() <= 1:
        pass
    else:
        print(normals_gt.size(), normals_gt.min(), normals_gt.max())
        raise ValueError("Normals must be in range [0, 1], [0, 255] or [-1, 1].")

    if mask is not None:
        normals_gt = normals_gt[mask]
        normals_pred = normals_pred[mask]

    if mode == "smooth_l1":
        loss = F.smooth_l1_loss(normals_pred, normals_gt, reduction="none")
    elif mode == "l2":
        loss = torch.linalg.norm(normals_pred - normals_gt, dim=-1)
    elif mode == "mse":
        loss = F.mse_loss(normals_pred, normals_gt, reduction="none")
    elif mode == "cos_sim":
        loss = 1 - F.cosine_similarity(normals_pred, normals_gt, dim=-1)
    elif mode == "l1":
        loss = F.l1_loss(normals_pred, normals_gt, reduction="none")
    else:
        raise ValueError(f"Unknown normal loss mode: {mode}")
    return loss + eikonal_loss


def entropy_regularization(logits: Tensor) -> Tensor:
    return F.binary_cross_entropy_with_logits(logits, logits.sigmoid().detach())
