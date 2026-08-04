import copy
import math
import random
import time
import warnings
from collections.abc import Sequence
from functools import partial
from logging import DEBUG
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import open3d as o3d
import pynvml
import torch
import torchmetrics.functional as M
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional as F
from torchmetrics.detection import MeanAveragePrecision, PanopticQuality
from torchvision.ops import focal_loss

from eval import MeanAveragePrecision3D, PanopticQuality3D
from libs import furthest_point_sample
from utils import (
    DEBUG_LEVEL_1,
    DEBUG_LEVEL_2,
    PLOTLY_COLORS,
    cosine_anneal,
    depth_to_image,
    filter_dict,
    setup_logger,
    stdout_redirected,
)

from .dpt import DPTHead
from .fpn import FPN, SimpleConvDecoder
from .grid import GridDecoder
from .mixins import EMPTY_EVAL_RESULTS_DICT, MultiEvalMixin, MultiLossMixin, PredictMixin
from .model import Model
from .transformer import MLP, TCNN_EXISTS, Block, Decoder, DecoderBlock, Encoder, HashGridEncoding, NeRFEncoding
from .utils import (
    DepthLoss,
    assign_params_groups,
    batched_sinkhorn_segmentation_loss,
    box_cxcywh_to_xywh,
    box_cxcywh_to_xyxy,
    box_iou,
    box_xywh_to_cxcywh,
    box_xywh_to_xyxy,
    box_xyxy_to_xywh,
    calculate_dice_score,
    cls_probs_from_logits,
    dice_loss,
    dice_loss_instr,
    filter_instance_masks,
    generalized_box_iou,
    generalized_circle_iou,
    get_boxes_from_circles,
    get_boxes_from_masks,
    get_circles_from_boxes,
    get_circles_from_masks,
    hungarian_matcher,
    index_from_match,
    inverse_sigmoid,
    labels_from_logits,
    masks_from_labels,
    masks_from_logits,
    queries_from_feat,
    sample_uncertain,
    show_image_with_masks,
    sinkhorn_segmentation_loss,
    tversky_loss,
)

logger = setup_logger(__name__)

LEGACY_DINO_V2_REPO = "facebookresearch/dinov2:81b2b6419385a321287de91e00282ef7cbd26f94"


def _load_dino_backbone(repo_or_dir: str, backbone: str, **kwargs: Any) -> Any:
    if "verbose" not in kwargs:
        kwargs["verbose"] = False
    return cast(Any, torch.hub.load(repo_or_dir, backbone, **kwargs))


def _require_tensor(data: dict[str, Any], key: str) -> Tensor:
    value = data.get(key)
    if not isinstance(value, Tensor):
        raise TypeError(f"Expected tensor for `{key}`")
    return value


def _checkpoint_model_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if "model" in state_dict:
        state_dict = state_dict["model"]
    return {k.replace("model.", "").replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def _is_legacy_dino3d_state_dict(state_dict: dict[str, Any]) -> bool:
    state_dict = _checkpoint_model_state_dict(state_dict)
    nerf_weight = state_dict.get("nerf_enc.mlp.weight")
    return (
        isinstance(nerf_weight, Tensor)
        and tuple(nerf_weight.shape) == (384, 63)
        and "inputs_enc.ln_1.weight" in state_dict
        and "points_enc.cross_attn.to_q.weight" in state_dict
    )


def dino3d_legacy_config_overrides(state_dict: dict[str, Any] | None) -> dict[str, Any]:
    if state_dict is None or not _is_legacy_dino3d_state_dict(state_dict):
        return {}
    return {
        "repo_or_dir": LEGACY_DINO_V2_REPO,
        "num_queries": 512,
        "cls_token": True,
        "cat_feat": False,
        "nerf_freqs": 10,
        "init_weights": False,
        "normalize_inputs": True,
        "scale_inputs": True,
        "legacy_forward_features": True,
    }


def _remap_legacy_dino3d_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    remapped = dict(state_dict)

    for key, value in list(state_dict.items()):
        if key.startswith("nerf_enc."):
            remapped[f"decoder.0.{key.removeprefix('nerf_enc.')}"] = value
        elif key.startswith("occ_head."):
            remapped[f"decoder.2.{key.removeprefix('occ_head.')}"] = value
        elif key.startswith("points_enc."):
            suffix = key.removeprefix("points_enc.")
            remapped.pop(key, None)
            remapped[f"cross_attn.{suffix}"] = value
            remapped[f"decoder.1.{suffix}"] = value
        elif key.startswith("inputs_enc."):
            suffix = key.removeprefix("inputs_enc.")
            remapped.pop(key, None)
            if suffix == "self_attn.to_qkv.weight":
                remapped["input_enc.self_attn.to_q.weight"] = value[:384]
                remapped["input_enc.self_attn.to_kv.weight"] = value[384:]
            elif suffix == "self_attn.to_qkv.bias":
                remapped["input_enc.self_attn.to_q.bias"] = value[:384]
                remapped["input_enc.self_attn.to_kv.bias"] = value[384:]
            else:
                remapped[f"input_enc.{suffix}"] = value

    return remapped


def _is_legacy_dino_inst_seg_3d_state_dict(state_dict: dict[str, Any]) -> bool:
    state_dict = _checkpoint_model_state_dict(state_dict)
    nerf_weight = state_dict.get("nerf_enc.mlp.weight")
    query_pos = state_dict.get("query_pos.weight")
    return (
        isinstance(nerf_weight, Tensor)
        and tuple(nerf_weight.shape) == (384, 39)
        and isinstance(query_pos, Tensor)
        and tuple(query_pos.shape) == (100, 384)
        and "queries.0.weight" in state_dict
        and "inputs_head.0.weight" in state_dict
        and "cls_quality_head.0.weight" in state_dict
    )


def dino_inst_seg_3d_legacy_config_overrides(state_dict: dict[str, Any] | None) -> dict[str, Any]:
    if state_dict is None or not _is_legacy_dino_inst_seg_3d_state_dict(state_dict):
        return {}
    return {
        "repo_or_dir": LEGACY_DINO_V2_REPO,
        "num_objs": 100,
        "num_queries": 1024,
        "mlp_heads": True,
        "multitask": "head",
        "pred_cls": "quality+objectness",
        "queries_from_feat": "detach",
        "init_weights": False,
        "nerf_freqs": 6,
    }


class DinoCls(MultiEvalMixin, MultiLossMixin, PredictMixin, Model):
    def __init__(
        self,
        n_classes: int,
        freeze: bool = True,
        cls_token: bool = True,
        repo_or_dir: str = "facebookresearch/dinov2",
        backbone: str = "dinov2_vits14",
    ):
        super().__init__()
        warnings.filterwarnings("ignore", message="xFormers is available.*")
        self.encoder = _load_dino_backbone(repo_or_dir, backbone)
        if freeze:
            self.encoder.requires_grad_(False)
            self.encoder.eval()
        self.encoder.mask_token = None
        self.cls_token = cls_token
        if not cls_token:
            self.encoder.cls_token.requires_grad_(False)

        self.n_embd = self.encoder.embed_dim
        self.decoder = nn.Linear(self.n_embd, n_classes)

    def encode(self, inputs, **kwargs) -> Tensor:
        if self.cls_token:
            return self.encoder(inputs)
        return self.encoder.forward_features(inputs)["x_norm_patchtokens"]

    def decode(self, features, **kwargs) -> dict[str, Tensor]:
        return dict(cls_logits=self.decoder(features if self.cls_token else features.mean(dim=1)))

    def forward(self, inputs, **kwargs) -> dict[str, Tensor]:
        return self.decode(self.encode(inputs))

    def loss(
        self,
        data: dict[str, list[str] | Tensor],
        regression: bool = False,
        name: str | None = None,
        reduction: str | None = "mean",
        **kwargs,
    ) -> Tensor:
        return super().loss(data, regression, name, reduction, **kwargs)["cls_loss"]


class Dino3D(MultiEvalMixin, MultiLossMixin, PredictMixin, Model):
    def __init__(
        self,
        num_queries: int = 512,
        repo_or_dir: str = "facebookresearch/dinov2",
        backbone: str = "dinov2_vits14",
        cls_token: bool = False,
        cat_feat: bool = True,
        bias: bool = True,
        dropout: float = 0.05,
        drop_path: float = 0.1,
        init_weights: bool = True,
        loss_name: Literal["dice", "bce", "focal", "dice+bce"] = "dice+bce",
        bce_weight: float = 5.0,
        dice_focal_weight: float = 2.0,
        focal_alpha: float = 0.5,
        focal_gamma: float = 2.0,
        num_classes: int | None = None,
        nerf_enc: Literal["tcnn", "torch"] = "tcnn" if TCNN_EXISTS else "torch",
        nerf_freqs: int = 6,
        padding: float = 0.1,
        normalize_inputs: bool = False,
        scale_inputs: bool = False,
        legacy_forward_features: bool = False,
        sample: int | None = None,
        learn_loss_weights: bool = False,
    ):
        super().__init__()
        warnings.filterwarnings("ignore", message="xFormers is available.*")
        self.encoder = _load_dino_backbone(
            repo_or_dir,
            backbone,
            # mlp_ratio=mlp_ratio,
            qkv_bias=bias,
            ffn_bias=bias,
            proj_bias=bias,
            drop_path_rate=drop_path,
            pretrained=False,
        )
        self.encoder.prepare_tokens_with_masks = lambda x, masks=None: x
        self.encoder.patch_embed = None
        self.encoder.pos_embed = None
        self.encoder.mask_token = None

        out_index = {
            "dinov2_vits14": [2, 5, 8, 11],
            "dinov2_vitb14": [2, 5, 8, 11],
            "dinov2_vitl14": [4, 11, 17, 23],
            "dinov2_vitg14": [9, 19, 29, 39],
        }[backbone.strip("_reg")]
        self.legacy_forward_features = legacy_forward_features
        if legacy_forward_features:
            self.encoder.forward = self.encoder.forward_features
        else:
            self.encoder.forward = partial(
                self.encoder.get_intermediate_layers, n=out_index, reshape=False, return_class_token=True, norm=True
            )

        self.n_embd = self.encoder.embed_dim
        self.n_head = self.encoder.num_heads
        self.num_queries = num_queries
        self.cls_token = cls_token
        self.cat_feat = cat_feat
        self.loss_name = loss_name
        self.learn_loss_weights = learn_loss_weights
        self._bce_focal_weight = bce_weight
        self._dice_weight = dice_focal_weight
        cast(Any, self).sample = sample
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        if learn_loss_weights:
            if "bce" in loss_name or "focal" in loss_name:
                self._bce_focal_weight = nn.Parameter(torch.tensor(-math.log(2.0 * bce_weight)))
            if "dice" in loss_name:
                self._dice_weight = nn.Parameter(torch.tensor(-math.log(2.0 * dice_focal_weight)))

        self.decoder = nn.ModuleList()
        self.nerf_enc = NeRFEncoding(
            out_dim=self.n_embd,
            implementation=nerf_enc,
            num_frequencies=nerf_freqs,
            max_freq_exp=nerf_freqs - 1,
            normalize_inputs=normalize_inputs,
            scale_inputs=scale_inputs,
            padding=padding,
        )

        self.decoder.append(self.nerf_enc)
        self.input_enc = DecoderBlock(n_embd=self.n_embd, n_head=self.n_head, bias=bias, dropout=dropout, chunks=2)

        self.cross_attn = Block(n_embd=self.n_embd, n_head=self.n_head, bias=bias, cross_attn=True, chunks=2)
        self.decoder.append(self.cross_attn)
        self.occ_head = MLP(self.n_embd, n_out=1, bias=bias, dropout=dropout)
        if init_weights:
            nn.init.normal_(self.occ_head.c_proj.weight, mean=0.0, std=1e-3)
            if bias:
                nn.init.zeros_(self.occ_head.c_proj.bias)
        self.decoder.append(self.occ_head)

        self.num_classes = num_classes
        if num_classes:
            self.cls_head = nn.Linear(self.n_embd, num_classes)
            self.decoder.append(self.cls_head)

    def remap_state_dict(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        if _is_legacy_dino3d_state_dict(state_dict):
            logger.info("Remapping legacy Dino3D checkpoint keys")
            return _remap_legacy_dino3d_state_dict(state_dict)
        return super().remap_state_dict(state_dict)

    def load_state_dict(self, state_dict: dict[str, Any], *args: Any, **kwargs: Any):
        if _is_legacy_dino3d_state_dict(state_dict):
            kwargs.setdefault("fuzzy_match", False)
            kwargs.setdefault("shape_suffix_match", False)
        return super().load_state_dict(state_dict, *args, **kwargs)

    @property
    def bce_focal_weight(self) -> float | Tensor:
        if isinstance(self._bce_focal_weight, nn.Parameter):
            return 0.5 * torch.exp(-self._bce_focal_weight)
        return self._bce_focal_weight

    @property
    def dice_weight(self) -> float | Tensor:
        if isinstance(self._dice_weight, nn.Parameter):
            return 0.5 * torch.exp(-self._dice_weight)
        return self._dice_weight

    def apply_feature(self, x: Tensor, feature: tuple[Tensor | None, Tensor]) -> Tensor:
        if self.cross_attn is None:
            raise RuntimeError("Cross-attention is required but not initialized.")
        if feature[0] is None:
            return self.cross_attn(x, feature[1])
        cls_feat = feature[0].unsqueeze(1) if feature[0].ndim == 2 else feature[0]
        return self.cross_attn(x, torch.cat((cls_feat, feature[1]), dim=1))

    def encode(self, inputs: Tensor, **kwargs) -> Tensor:
        inputs_fps = furthest_point_sample(inputs, num_samples=self.num_queries)
        x = self.input_enc(self.nerf_enc(inputs_fps), self.nerf_enc(inputs))
        x = torch.cat((self.encoder.cls_token.expand(len(x), -1, -1), x), dim=1)
        feat = self.encoder(x)
        if self.legacy_forward_features:
            patch_feat = feat["x_norm_patchtokens"]
            if self.cls_token:
                return torch.cat((feat["x_norm_clstoken"].unsqueeze(1), patch_feat), dim=1)
            return patch_feat
        patch_feat = feat[-1][0]
        if self.cat_feat:
            patch_feat = torch.cat([f[0] for f in feat], dim=1)
        if self.cls_token:
            return torch.cat((feat[-1][1].unsqueeze(1), patch_feat), dim=1)
        return patch_feat

    def decode(self, points: Tensor, feature: Tensor, **kwargs) -> dict[str, Tensor]:
        patch_feat = feature
        cls_feat = None
        if self.cls_token:
            patch_feat = feature[:, 1:]
            cls_feat = feature[:, :1]

        x = self.nerf_enc(points)
        if self.cross_attn is None:
            if cls_feat is None:
                raise ValueError("cls_feat is required when cross-attention is disabled.")
            x += cls_feat.unsqueeze(1)
        else:
            x = self.apply_feature(x, (cls_feat, patch_feat))
        logits = self.occ_head(x).squeeze(2)
        if self.num_classes:
            cls_logits = self.cls_head(patch_feat.mean(dim=1)) if cls_feat is None else self.cls_head(cls_feat)
            return dict(logits=logits, cls_logits=cls_logits)
        return dict(logits=logits)

    def forward(self, inputs: Tensor, **kwargs) -> dict[str, Tensor]:
        return self.decode(feature=self.encode(inputs=inputs, **kwargs), **kwargs)

    def loss(
        self,
        data: dict[str, Any],
        regression: bool = False,
        name: str | None = None,
        reduction: str | None = "mean",
        **kwargs,
    ) -> Tensor:
        if not any("logits" in key for key in data.keys()):
            data.update(self(**data, **kwargs))

        name = name or self.loss_name
        reduction_val = reduction or "mean"
        preds = _require_tensor(data, "logits")
        targets = _require_tensor(data, "points.occ")

        if self.sample and self.training and preds.numel() > 0:
            idx = sample_uncertain(preds, num_points=self.sample)
            preds = torch.gather(preds, dim=1, index=idx)
            targets = torch.gather(targets, dim=1, index=idx)

        loss = preds.new_tensor(0.0)
        if "bce" in name:
            bce = F.binary_cross_entropy_with_logits(preds, targets, reduction=reduction_val)
            self.log("bce_loss", cast(Any, bce), level=DEBUG_LEVEL_1, train_only=True)
            comp = self.bce_focal_weight * bce
            if isinstance(self._bce_focal_weight, nn.Parameter):
                comp = comp + 0.5 * self._bce_focal_weight
                self.log("bce_weight", cast(Any, self.bce_focal_weight), level=DEBUG_LEVEL_2, train_only=True)
            loss = loss + comp
        elif "focal" in name:
            focal = focal_loss.sigmoid_focal_loss(
                preds, targets, alpha=self.focal_alpha, gamma=self.focal_gamma, reduction=reduction_val
            )
            self.log("focal_loss", cast(Any, focal), level=DEBUG_LEVEL_1, train_only=True)
            comp = self.bce_focal_weight * focal
            if isinstance(self._bce_focal_weight, nn.Parameter):
                comp = comp + 0.5 * self._bce_focal_weight
                self.log("bce_weight", cast(Any, self.bce_focal_weight), level=DEBUG_LEVEL_2, train_only=True)
            loss = loss + comp

        if "dice" in name:
            d_loss = dice_loss(preds, targets, reduction=reduction_val)
            self.log("dice_loss", cast(Any, d_loss), level=DEBUG_LEVEL_1, train_only=True)
            comp = self.dice_weight * d_loss
            if isinstance(self._dice_weight, nn.Parameter):
                comp = comp + 0.5 * self.dice_weight
                self.log("dice_weight", cast(Any, self.dice_weight), level=DEBUG_LEVEL_2, train_only=True)
            loss = loss + comp

        if "cls_logits" in data or "seg_logits" in data:
            losses = super().loss(filter_dict(data, remove={"logits"}), regression, name, reduction, **kwargs)
            cls_weight_val = data.get("cls_weight", 1.0)
            cls_weight: float | Tensor
            if isinstance(cls_weight_val, list):
                cls_weight = cast(float | Tensor, cls_weight_val[0] if cls_weight_val else 1.0)
            else:
                cls_weight = cast(float | Tensor, cls_weight_val)
            seg_weight = cast(float | Tensor, data.get("seg_weight", 1.0))
            cls_loss = cast(float | Tensor, losses.get("cls_loss", 0.0))
            seg_loss = cast(float | Tensor, losses.get("seg_loss", 0.0))
            loss = loss + cls_weight * cls_loss
            loss = loss + seg_weight * seg_loss

        return loss


class DinoRGB(MultiEvalMixin, MultiLossMixin, PredictMixin, Model):
    def __init__(
        self,
        padding: float = 0.1,
        freeze: bool = True,
        repo_or_dir: str = "facebookresearch/dinov2",
        backbone: str = "dinov2_vits14",
        condition: Literal["cls", "cross-attn", "grid", "grid-gate", "all"] = "all",
        head: Literal["linear", "mlp", "dpt", "multi-scale"] = "mlp",
        mask: Literal["patch", "grid", "cross-attn", "all"] | None = None,
        bias: bool = True,
        init: bool = True,
        nerf_enc: Literal["tcnn", "torch"] = "tcnn" if TCNN_EXISTS else "torch",
        nerf_freqs: int = 6,
    ):
        super().__init__()
        warnings.filterwarnings("ignore", message="xFormers is available.*")
        self.encoder = _load_dino_backbone(repo_or_dir, backbone)
        if freeze:
            self.encoder.requires_grad_(False)
            self.encoder.eval()
        self.encoder.mask_token = None
        self.encoder.forward = self.encoder.forward_features
        self.cls_token = condition == "all" or "cls" in condition
        if not self.cls_token:
            self.encoder.cls_token.requires_grad_(False)

        out_index = {
            "dinov2_vits14": [2, 5, 8, 11],
            "dinov2_vitb14": [2, 5, 8, 11],
            "dinov2_vitl14": [4, 11, 17, 23],
            "dinov2_vitg14": [9, 19, 29, 39],
        }[backbone.strip("_reg")]

        self.mask = mask
        self.freeze = freeze
        self.n_embd = (128 if "vits" in backbone else 256) if head == "dpt" else self.encoder.embed_dim
        self.n_head = 4 if head == "dpt" else self.encoder.num_heads

        self.decoder = nn.ModuleList()
        self.nerf_enc = NeRFEncoding(
            out_dim=self.n_embd,
            padding=padding,
            implementation=nerf_enc,
            num_frequencies=nerf_freqs,
            max_freq_exp=nerf_freqs - 1,
        )
        self.decoder.append(self.nerf_enc)
        self.cross_attn = None
        self.grid_gate: nn.Sequential | None = None
        if condition == "all" or "cross-attn" in condition:
            self.cross_attn = Block(n_embd=self.n_embd, n_head=self.n_head, bias=bias, cross_attn=True, chunks=2)
            self.decoder.append(self.cross_attn)
        self.grid = None
        if condition == "all" or "grid" in condition:
            self.grid = GridDecoder(c_dim=self.n_embd, padding=padding)
            self.decoder.append(self.grid)
            self.grid_gate = None
            if condition == "all" or "gate" in condition:
                self.grid_gate = nn.Sequential(
                    nn.Linear(2 * self.n_embd, self.n_embd),
                    nn.LayerNorm(self.n_embd),
                    nn.ReLU(),
                    nn.Linear(self.n_embd, self.n_embd),
                    nn.Sigmoid(),
                )
                self.decoder.append(self.grid_gate)
            elif "glu" in condition:
                self.grid_gate = nn.Sequential(
                    nn.Linear(2 * self.n_embd, 2 * self.n_embd),
                    nn.GLU(),
                    nn.Linear(self.n_embd, self.n_embd),
                )
                self.decoder.append(self.grid_gate)
        if head == "linear":
            self.head = nn.Linear(self.n_embd, 1, bias=bias)
            if init:
                nn.init.xavier_uniform_(self.head.weight, gain=1e-3)
                if bias:
                    nn.init.constant_(self.head.bias, -2.0)
        elif head in ["mlp", "multi-scale"]:
            self.head = MLP(self.n_embd, n_out=1, bias=bias)
            if init:
                nn.init.xavier_uniform_(self.head.c_proj.weight, gain=1e-3)
                if bias:
                    nn.init.constant_(self.head.c_proj.bias, -2.0)
            if head == "multi-scale":
                self.encoder.forward = partial(
                    self.encoder.get_intermediate_layers,
                    n=4,  # last 4 layers
                    reshape=True,
                    return_class_token=False,
                    norm=True,
                )
        elif head == "dpt":
            channels = [self.encoder.embed_dim // 2 ** (3 - i) for i in range(len(out_index))]

            self.encoder.norm = None
            self.encoder.forward = partial(
                self.encoder.get_intermediate_layers, n=out_index, reshape=True, return_class_token=True, norm=False
            )
            channels_t = tuple(cast(list[int], channels))
            self.head = nn.ModuleList(
                [
                    DPTHead(
                        channels=self.n_embd,
                        embed_dims=self.encoder.embed_dim,
                        post_process_channels=channels_t,
                        resize_type="deconv",
                        conv_type="bottleneck",
                    ),
                    MLP(self.n_embd, n_out=1, bias=bias),
                ]
            )
            if init:
                head_list = cast(nn.ModuleList, self.head)
                mlp_head = cast(MLP, head_list[1])
                nn.init.xavier_uniform_(mlp_head.c_proj.weight, gain=1e-3)
                if bias:
                    nn.init.constant_(mlp_head.c_proj.bias, -2.0)
        self.decoder.append(self.head)

    def apply_grid_feature(self, x: Tensor, points: Tensor, feature: Tensor | list[Tensor], **kwargs) -> Tensor:
        if self.grid is None:
            raise RuntimeError("Grid conditioning is required but not initialized.")
        if torch.is_tensor(feature):
            feature = [feature]

        for feat in feature:
            if feat.ndim == 3:
                p = int(feat.size(1) ** 0.5)
                feat = rearrange(feat, "b (h w) c -> b c h w", h=p, w=p)
            grid_feat = self.grid.sample_feature(points, {"uv": feat}, **kwargs)[0].transpose(1, 2)
            if self.grid_gate is None:
                x = x + grid_feat
            elif isinstance(self.grid_gate[-1], nn.Sigmoid):
                x = x + grid_feat * self.grid_gate(torch.cat((x, grid_feat), dim=2))
            else:
                x = x + self.grid_gate(torch.cat((x, grid_feat), dim=2))
        return x

    def apply_feature(self, x: Tensor, feature: tuple[Tensor | None, Tensor]) -> Tensor:
        if self.cross_attn is None:
            raise RuntimeError("Cross-attention is required but not initialized.")
        if feature[0] is None:
            return self.cross_attn(x, feature[1])
        return self.cross_attn(x, torch.cat((feature[0].unsqueeze(1), feature[1]), dim=1))

    def encode(self, inputs, **kwargs) -> Tensor | list[Tensor]:
        feature = self.encoder(inputs)
        if not isinstance(feature, dict):
            return feature
        cls_feat, _, patch_feat, _, _ = feature.values()
        if self.cls_token:
            if self.grid is None and self.cross_attn is None:
                return cls_feat
            return torch.cat((cls_feat.unsqueeze(1), patch_feat), dim=1)
        return patch_feat

    def decode(self, points: Tensor, feature: Tensor | list[Tensor], **kwargs) -> dict[str, Tensor]:
        head: nn.Module = self.head
        if isinstance(self.head, nn.ModuleList):
            head_list = cast(nn.ModuleList, self.head)
            feat = cast(Tensor, head_list[0](feature))
            patch_feat = rearrange(feat, "b c h w -> b (h w) c")
            cls_feat = None
            head = cast(nn.Module, head_list[1])
        elif self.cls_token:
            if not isinstance(feature, Tensor):
                raise TypeError("Expected tensor feature when cls token decoding is enabled.")
            if feature.ndim == 3:
                cls_feat, patch_feat = feature[:, 0], feature[:, 1:]
            else:
                cls_feat, patch_feat = feature, None
        else:
            if not isinstance(feature, Tensor):
                raise TypeError("Expected tensor feature for decoding.")
            cls_feat, patch_feat = None, feature

        if self.training and self.mask in ["patch", "all"] and random.random() < 0.5:
            if patch_feat is not None:
                mask = torch.bernoulli(torch.ones(patch_feat.shape[:2]) * 0.1).bool().to(points.device)
                patch_feat[mask] = 0
            if cls_feat is not None and random.random() < 0.1:
                cls_feat = torch.zeros_like(cls_feat)

        x = self.nerf_enc(points)
        if self.cls_token and self.cross_attn is None:
            if cls_feat is None:
                raise ValueError("cls_feat is required when cls-token only conditioning is used.")
            x += cls_feat.unsqueeze(1)

        mask_grid = False
        if self.grid is not None and patch_feat is not None:
            feature = patch_feat
            if self.training and self.mask in ["grid", "all"] and random.random() < 0.1:
                feature = torch.zeros_like(patch_feat)
                mask_grid = True
            x = self.apply_grid_feature(x, points, feature, **kwargs)
        if self.cross_attn is not None:
            if patch_feat is None:
                raise ValueError("patch_feat is required when cross-attention conditioning is enabled.")
            if self.training and self.mask in ["cross-attn", "all"] and not mask_grid and random.random() < 0.1:
                patch_feat = torch.zeros_like(patch_feat)
                if cls_feat is not None:
                    cls_feat = torch.zeros_like(cls_feat)
            x = self.apply_feature(x, (cls_feat, patch_feat))
        return dict(logits=cast(Tensor, head(x)).squeeze(2), feature=x)

    def forward(self, inputs, **kwargs) -> dict[str, Tensor]:
        return self.decode(feature=self.encode(inputs=inputs, **kwargs), **kwargs)

    def loss(
        self,
        data: dict[str, list[str] | Tensor],
        regression: bool = False,
        name: str | None = None,
        reduction: str | None = "mean",
        **kwargs,
    ) -> Tensor:
        return super().loss(data, regression, name, reduction, **kwargs)["occ_loss"]

    def optimizer(
        self, lr: float, weight_decay: float = 1e-2, factor: float = 1e-2, fused: bool = False
    ) -> torch.optim.Optimizer | None:
        if self.freeze or not factor:
            return None

        enc_params = list(self.encoder.parameters())
        enc_param_ids = {id(p) for p in enc_params}
        params_decay, params_no_decay = assign_params_groups(self)

        enc_params_decay = [p for p in params_decay if id(p) in enc_param_ids]
        enc_params_no_decay = [p for p in params_no_decay if id(p) in enc_param_ids]
        other_params_decay = [p for p in params_decay if id(p) not in enc_param_ids]
        other_params_no_decay = [p for p in params_no_decay if id(p) not in enc_param_ids]

        param_groups = [
            {"params": enc_params_decay, "lr": lr * factor, "weight_decay": weight_decay * factor},
            {"params": enc_params_no_decay, "lr": lr * factor, "weight_decay": 0.0},
            {"params": other_params_decay, "lr": lr, "weight_decay": weight_decay},
            {"params": other_params_no_decay, "lr": lr, "weight_decay": 0.0},
        ]

        return torch.optim.AdamW(param_groups, fused=fused)


class DinoRGBD(DinoRGB):
    def __init__(
        self,
        n_queries: int | Literal["auto"] = "auto",
        inputs_enc: Literal["", "self-attn", "mlp", "both"] = "both",
        patches_enc: Literal["", "self-attn", "mlp", "both"] = "both",
        condition: Literal["cls", "cross-attn", "grid", "grid-gate", "all"] = "all",
        mask: Literal["mod", "patch", "both"] | None = None,
        padding: float = 0.1,
        freeze: bool = True,
        repo_or_dir: str = "facebookresearch/dinov2",
        backbone: str = "dinov2_vits14",
        bias: bool = True,
    ):
        super().__init__(
            padding=padding,
            freeze=freeze,
            repo_or_dir=repo_or_dir,
            backbone=backbone,
            condition=condition,
            head="mlp",
            bias=bias,
        )
        self.n_queries = n_queries
        self.mask = mask

        inputs_enc_mode: str = "self-attn+mlp" if inputs_enc == "both" else inputs_enc
        self.inputs_enc = Block(
            n_embd=self.n_embd,
            n_head=self.n_head,
            bias=bias,
            self_attn="self-attn" in inputs_enc_mode,
            cross_attn=True,
            projection="mlp" if "mlp" in inputs_enc_mode else None,
            chunks=2,
        )
        self.point_feat = _load_dino_backbone(repo_or_dir, backbone, pretrained=False)
        self.point_feat.prepare_tokens_with_masks = lambda x, masks: x
        self.point_feat.patch_embed = None
        self.point_feat.pos_embed = None
        self.point_feat.mask_token = None
        self.point_feat.forward = self.point_feat.forward_features

        self.self_attn = None
        patches_enc_mode: str = "self-attn+mlp" if patches_enc == "both" else patches_enc
        if patches_enc_mode:
            self.self_attn = Block(
                n_embd=self.n_embd,
                n_head=self.n_head,
                bias=bias,
                self_attn="self-attn" in patches_enc_mode,
                projection="mlp" if "mlp" in patches_enc_mode else None,
            )

        if self.cls_token:
            self.point_feat.cls_token.requires_grad_(False)

    def encode(self, inputs, **kwargs) -> tuple[Tensor | None, ...]:
        n_queries = self.n_queries if isinstance(self.n_queries, int) else 512
        img_cls, img_patches = None, None
        feature = super().encode(kwargs["inputs.image"], **kwargs)
        if isinstance(self.head, nn.ModuleList):
            feat = cast(Tensor, self.head[0](feature))
            img_patches = rearrange(feat, "b c h w -> b (h w) c")
            img_cls = None
        elif self.cls_token:
            if not isinstance(feature, Tensor):
                raise TypeError("Expected tensor features for cls-token conditioning.")
            if feature.ndim == 3:
                img_cls, img_patches = feature[:, 0], feature[:, 1:]
            else:
                img_cls, img_patches = feature, None
        else:
            if not isinstance(feature, Tensor):
                raise TypeError("Expected tensor feature for RGBD encoding.")
            img_cls, img_patches = None, feature
        if self.n_queries == "auto":
            if img_patches is None:
                raise ValueError("Automatic query count requires image patch features.")
            n_queries = img_patches.size(1)

        inputs_fps = furthest_point_sample(inputs, num_samples=n_queries)
        x = self.inputs_enc(self.nerf_enc(inputs_fps), self.nerf_enc(inputs))
        x = torch.cat((self.point_feat.cls_token.expand(x.size(0), -1, -1), x), dim=1)
        point_values = tuple(cast(Any, self.point_feat(x)).values())
        point_cls, _, point_patches, _, _ = cast(tuple[Tensor, Tensor, Tensor, Tensor, Tensor], point_values)

        return img_cls, img_patches, point_cls, point_patches

    def decode(self, points: Tensor, feature: tuple[Tensor | None, ...], **kwargs) -> dict[str, Tensor]:
        img_cls, img_patches, point_cls, point_patches = feature
        x = self.nerf_enc(points)

        if self.training and self.mask:
            if self.mask in ["mod", "both"]:
                # 1. With 10% probability, drop either all image or all point patches
                if random.random() < 0.1:
                    if random.random() < 0.5:  # 50% chance to choose image vs point
                        if img_patches is not None:
                            img_patches = torch.zeros_like(img_patches)
                        if img_cls is not None:
                            img_cls = torch.zeros_like(img_cls)
                    else:
                        if point_patches is not None:
                            point_patches = torch.zeros_like(point_patches)
                        if point_cls is not None:
                            point_cls = torch.zeros_like(point_cls)

            if self.mask in ["patch", "both"]:
                # 2. Additionally, randomly drop 10% of all patches with 50% probability
                if img_patches is not None and random.random() < 0.5:
                    img_patch_mask = torch.bernoulli(torch.ones(img_patches.shape[:2]) * 0.1).bool().to(points.device)
                    img_patches[img_patch_mask] = 0
                    if img_cls is not None and random.random() < 0.1:
                        img_cls = torch.zeros_like(img_cls)

                if point_patches is not None and random.random() < 0.5:
                    point_patch_mask = (
                        torch.bernoulli(torch.ones(point_patches.shape[:2]) * 0.1).bool().to(points.device)
                    )
                    point_patches[point_patch_mask] = 0
                    if point_cls is not None and random.random() < 0.1:
                        point_cls = torch.zeros_like(point_cls)

        if img_patches is not None:
            x = self.apply_grid_feature(x, points, img_patches, **kwargs)
            if img_cls is not None:
                img_patches = torch.cat((img_cls.unsqueeze(1), img_patches), dim=1)

        if point_patches is not None and point_cls is not None:
            point_patches = torch.cat((point_cls.unsqueeze(1), point_patches), dim=1)

        if img_patches is None:
            patches = point_patches
        elif point_patches is None:
            patches = img_patches
        else:
            patches = torch.cat((img_patches, point_patches), dim=1)
            if self.self_attn is not None:
                patches = self.self_attn(patches)

        if patches is None:
            raise ValueError("At least one conditioning patch feature tensor is required.")
        if self.cross_attn is None:
            raise RuntimeError("Cross-attention is required but not initialized.")
        x = self.cross_attn(x, patches)
        return dict(logits=cast(Tensor, self.head(x)).squeeze(2))


class DinoInstSeg(MultiEvalMixin, Model):
    def __init__(
        self,
        num_objs: int = 100,
        mode: Literal["linear", "mlp", "conv", "detr", "instr", "mask"] = "mask",
        freeze: bool = False,
        repo_or_dir: str = "facebookresearch/dinov2",
        backbone: str = "dinov2_vits14",
        bias: bool = True,
        dropout: float = 0.0,
        pred_det: Literal["box", "circle"] | None = None,
        queries_from_feat: Literal["max", "mean", "norm", "fps", "cls", "hard", "soft", "gumbel", "anneal"]
        | None = "cls",
        detach_queries: bool = True,
        straight_through: bool = False,
        combine_feat: Literal["cat", "add", "gate"] | None = None,
        pred_cls: Literal["objectness", "quality"] | None = "objectness",
        num_dec_layers: int | None = None,
        mlp_heads: bool = False,
        aux_depth_loss: bool = False,
        match_cls: bool = True,
        match_det: bool = True,
        global_match: bool = False,
        pad_targets: bool = False,
        sample: int | None = None,
        loss_name: Literal["dice", "focal", "bce", "instr", "dice+bce"] = "dice+bce",
        focal_alpha: float = 0.5,
        focal_gamma: float = 2.0,
        mask_weight: float = 1.0,
        cls_weight: float = 1.0,
        det_weight: float = 1.0,
        depth_weight: float = 1.0,
        aux_weight: float | Literal["auto"] | None = "auto",
        learn_loss_weights: bool = False,
    ):
        super().__init__()

        if not pred_cls:
            if queries_from_feat:
                raise ValueError("queries_from_feat requires classification.")

        if learn_loss_weights and not (pred_cls or pred_det):
            raise ValueError("learn_loss_weights requires instance segmentation and classification or detection.")

        warnings.filterwarnings("ignore", message="xFormers is available.*")
        self.encoder = _load_dino_backbone(repo_or_dir, backbone)
        if freeze:
            self.encoder.requires_grad_(False)
            self.encoder.eval()
        self.encoder.mask_token = None
        if not queries_from_feat:
            self.encoder.norm = None

        # Focal loss parameters
        self.alpha = focal_alpha
        self.gamma = focal_gamma

        self._aux_loss = aux_weight is not None
        self.num_objs = num_objs
        self.det_type = pred_det
        self.queries_from_feat = queries_from_feat
        self.straight_through = straight_through
        self.detach_queries = detach_queries
        self.combine_feat = combine_feat
        self.pred_cls = pred_cls
        self.pred_det = pred_det
        self.match_cls = match_cls
        self.match_det = match_det
        self.global_match = global_match
        self.pad_targets = pad_targets
        self.sample = sample
        self.loss_name = loss_name
        self.mask_weight = mask_weight
        self.cls_weight = cls_weight
        self.det_weight = det_weight
        self.depth_weight = depth_weight
        self.learn_loss_weights = learn_loss_weights
        self.freeze = freeze
        self.mode = mode
        self.n_embd = self.encoder.embed_dim
        self.n_head = self.encoder.num_heads

        self.map = MeanAveragePrecision(iou_type="segm", backend="faster_coco_eval")
        self.pq = PanopticQuality(things={1}, stuffs={0}, return_sq_and_rq=True)

        out_index = {
            "dinov2_vits14": [2, 5, 8, 11],
            "dinov2_vitb14": [2, 5, 8, 11],
            "dinov2_vitl14": [4, 11, 17, 23],
            "dinov2_vitg14": [9, 19, 29, 39],
        }[backbone.strip("_reg")]
        channels = tuple(int(self.n_embd / 2 ** (3 - i)) for i in range(len(out_index)))

        self.encoder.forward = partial(
            self.encoder.get_intermediate_layers, n=out_index, reshape=True, return_class_token=True, norm=False
        )

        head_dim = self.n_embd
        self.dpt_head = None
        if "dpt" in mode or "mask" in mode:
            self.dpt_head = DPTHead(
                channels=self.n_embd,
                embed_dims=self.n_embd,
                post_process_channels=channels,
                resize_type="deconv",
                conv_type="bottleneck",
            )
            head_dim = self.dpt_head.channels

        if pred_cls or "mask" in mode:
            self.head_norm = nn.LayerNorm(head_dim)

        if "conv" in mode:
            if "dpt" in mode:
                self.head = nn.Conv2d(head_dim, num_objs, kernel_size=1)
            else:
                self.head = SimpleConvDecoder(
                    input_dim=self.n_embd,
                    num_masks=num_objs,
                    decoder_channels=list(reversed(channels)),
                    num_groups=self.n_head,
                )
        else:
            self.query_pos = nn.Embedding(num_objs, head_dim)
            if num_dec_layers is None:
                num_dec_layers = {"dinov2_vits14": 3, "dinov2_vitb14": 6, "dinov2_vitl14": 9, "dinov2_vitg14": 12}[
                    backbone.strip("_reg")
                ]
            self.aux_weight = 1 / num_dec_layers if aux_weight == "auto" else aux_weight
            self.query_enc = Decoder(
                n_layer=num_dec_layers,
                n_embd=head_dim,
                n_head=self.n_head if head_dim == self.n_embd else head_dim // 64,
                bias=bias,
                dropout=dropout,
                chunks=3,
                k_dim=head_dim,
                v_dim=head_dim,
            )
            if queries_from_feat:
                query_layers: list[nn.Module] = []
                encoder_norm = cast(Any, self.encoder).norm
                if isinstance(encoder_norm, nn.Module):
                    query_layers.append(encoder_norm)
                query_layers.append(nn.Linear(self.n_embd, head_dim))
                self.queries = nn.Sequential(*query_layers)
            if "mask" in mode:
                self.head = nn.Sequential(self.head_norm, nn.Linear(head_dim, head_dim))
                if mlp_heads:
                    self.head = nn.Sequential(
                        self.head_norm,
                        nn.Linear(head_dim, 2 * head_dim),
                        nn.GELU(),
                        nn.Linear(2 * head_dim, head_dim),
                    )
            else:
                self.block = DecoderBlock(
                    n_embd=head_dim,
                    n_head=self.n_head if head_dim == self.n_embd else head_dim // 64,
                    bias=bias,
                    dropout=dropout,
                    no_projection=True,
                    chunks=3,
                    k_dim=head_dim,
                    v_dim=head_dim,
                )
                if self.block.cross_attn is None:
                    raise RuntimeError("Decoder block requires cross-attention in this mode.")
                cast(Any, self.block.cross_attn).to_out = None
                if "fpn" in mode:
                    self.head = FPN(
                        dim=self.n_embd, fpn_dims=[self.n_embd] * 3, context_dim=self.n_embd, num_groups=self.n_head
                    )
                elif "mlp" in mode:
                    self.head = nn.Sequential(
                        nn.Conv2d(head_dim, head_dim, kernel_size=3, padding=1, bias=bias),
                        nn.GELU(),
                        nn.Conv2d(head_dim, 1, kernel_size=1, bias=bias),
                    )
                else:
                    self.head = nn.Conv2d(head_dim, 1, kernel_size=1, bias=bias)

        if pred_cls:
            self.cls_head = nn.Sequential(self.head_norm, nn.Linear(head_dim, 1))
            if mlp_heads:
                self.cls_head = nn.Sequential(
                    self.head_norm,
                    nn.Linear(head_dim, head_dim // 2),
                    nn.GELU(),
                    nn.Linear(head_dim // 2, 1),
                )
        if pred_det:
            det_dim = 4 if pred_det == "box" else 3
            if "init" in mode:
                self.init_dets = nn.Embedding(num_objs, det_dim)
            self.det_head = nn.Sequential(self.head_norm, nn.Linear(head_dim, det_dim))
            if mlp_heads:
                self.det_head = nn.Sequential(
                    self.head_norm,
                    nn.Linear(head_dim, 2 * head_dim),
                    nn.GELU(),
                    nn.Linear(2 * head_dim, det_dim),
                )

        if learn_loss_weights:
            self.log_var_mask = nn.Parameter(torch.zeros(()))
            if pred_cls:
                self.log_var_cls = nn.Parameter(torch.zeros(()))
            if pred_det:
                self.log_var_det = nn.Parameter(torch.zeros(()))
            if aux_depth_loss:
                self.log_var_depth = nn.Parameter(torch.zeros(()))

        if combine_feat == "gate":
            self.feat_gate = nn.Sequential(
                nn.Linear(2 * self.n_embd, 2 * self.n_embd),
                nn.GLU(),
                nn.Linear(self.n_embd, self.n_embd),
            )

        self.depth_head = None
        if aux_depth_loss:
            self.depth_loss = torch.jit.script(DepthLoss())
            self.depth_head = nn.ModuleList(
                [
                    nn.Conv2d(self.n_embd, self.n_embd // 2, kernel_size=3, stride=1, padding=1),
                    nn.Sequential(
                        nn.Conv2d(self.n_embd // 2, 32, kernel_size=3, stride=1, padding=1),
                        nn.GELU(),
                        nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
                        nn.Softplus(),
                    ),
                ]
            )

    @property
    def aux_loss(self) -> bool:
        return self._aux_loss if self.training else False

    @torch.no_grad()
    def _eval_cls_preds(self, cls_preds: Tensor, cls_targets: Tensor):
        self.log("cls_precision", M.precision(cls_preds, cls_targets, task="binary").item())
        self.log("cls_recall", M.recall(cls_preds, cls_targets, task="binary").item())
        self.log("cls_f1", M.f1_score(cls_preds, cls_targets, task="binary").item())
        cls_auroc = cast(Tensor, M.auroc(cls_preds, cls_targets, task="binary"))
        cls_auprc = cast(Tensor, M.average_precision(cls_preds, cls_targets, task="binary"))
        self.log("cls_auroc", cls_auroc.item())
        self.log("cls_auprc", cls_auprc.item())
        self.log("cls_acc", M.accuracy(cls_preds, cls_targets, task="binary").item())
        self.log("cls_ece", M.calibration_error(cls_preds, cls_targets, task="binary").item())
        self.log("cls_brier", (cls_preds - cls_targets.float()).pow(2).mean().item())

    @staticmethod
    def _apply_box_offsets(boxes: Tensor, offsets: Tensor, method: Literal["detr", "dino"] | None = None) -> Tensor:
        """
        Args:
            boxes: [B, N, 4] tensor of reference boxes in normalized (cx, cy, w, h) format
            offsets: [B, N, 4] tensor of (dx, dy, dw, dh) log-offsets
        """
        if method == "detr":
            new_center = boxes[..., :2] + offsets[..., :2]
            new_dims = boxes[..., 2:] * torch.exp(offsets[..., 2:])
            new_boxes = torch.cat([new_center, new_dims], dim=-1)
        elif method == "dino":
            boxes_logits = inverse_sigmoid(boxes)
            new_boxes = boxes_logits + offsets
        else:
            new_boxes = offsets
        return new_boxes.sigmoid()

    @staticmethod
    def _apply_circle_offsets(
        circles: Tensor, offsets: Tensor, method: Literal["detr", "dino"] | None = None
    ) -> Tensor:
        """
        Args:
            circles: [B, N, 3] tensor of reference circles in normalized (cx, cy, r) format
            offsets: [B, N, 3] tensor of (dx, dy, dr) log-offsets
        """
        if method == "detr":
            new_center = circles[..., :2] + offsets[..., :2]
            new_r = circles[..., 2] * torch.exp(offsets[..., 2])
            new_circles = torch.cat([new_center, new_r.unsqueeze(-1)], dim=-1)
        elif method == "dino":
            circles_logits = inverse_sigmoid(circles)
            new_circles = circles_logits + offsets
        else:
            new_circles = offsets
        return new_circles.sigmoid()

    @staticmethod
    @torch.no_grad()
    def _build_panoptic_tensor(
        instance_masks: list[Tensor], instance_classes: list[Tensor], things_set: set[int] | None = None
    ) -> Tensor:
        """
        Convert per-image exclusive instance masks + per-instance class ids into a batched
        panoptic tensor (B, H, W, 2) containing (category_id, instance_id):
            - category_id in things_set U stuffs_set
            - instance_id > 0 only for 'thing' classes; 0 for 'stuff' (or background)
        """
        if things_set is None:
            things_set = {1}

        assert len(instance_masks) == len(instance_classes)
        B = len(instance_masks)
        panoptic_list = []
        for i in range(B):
            masks_i = instance_masks[i]
            classes_i = instance_classes[i]
            K_i, H, W = masks_i.size()
            pan = torch.zeros(H, W, 2, dtype=torch.long, device=masks_i.device)
            for k in range(K_i):
                cls_id = classes_i[k].item()
                mask = masks_i[k]
                if cls_id in things_set:
                    pan[..., 0][mask] = cls_id
                    pan[..., 1][mask] = k + 1
                else:
                    pan[mask, 0] = cls_id
            panoptic_list.append(pan)
        return torch.stack(panoptic_list, dim=0)  # (B, H, W, 2)

    @torch.no_grad()
    def _update_metrics(
        self,
        logits: Tensor,
        targets: list[Tensor],
        scores: Tensor,
        threshold: float = 0.5,
        method: Literal["argmax", "cls", "greedy"] = "greedy",
        apply_filter: bool = True,
        min_area: int | None = 16,
        nms_iou: float | None = 0.5,
    ):
        if method == "argmax":
            labels = labels_from_logits(logits, threshold)
            pq_masks_list = masks_from_labels(labels, reindex=True)
        elif method == "greedy":
            pq_masks_list = masks_from_logits(
                logits=logits,
                scores=scores,
                threshold=threshold,
                apply_filter=apply_filter,
                min_size=min_area,
                nms_iou=nms_iou,
            )

        preds_for_map: list[dict[str, Tensor]] = []
        target_list: list[dict[str, Tensor]] = []

        pred_panoptic_instances: list[Tensor] = []
        pred_panoptic_classes: list[Tensor] = []
        gt_panoptic_instances: list[Tensor] = []
        gt_panoptic_classes: list[Tensor] = []

        log_threshold = math.log(threshold / (1 - threshold))
        for i in range(len(logits)):
            query_masks = logits[i]
            query_scores = scores[i]

            if apply_filter:
                query_masks, query_scores, _ = filter_instance_masks(
                    masks=query_masks, scores=query_scores, min_size=min_area, nms_iou=nms_iou
                )

            query_masks = query_masks > log_threshold
            num_preds_i = len(query_masks)
            num_gts_i = len(targets[i])

            pred_labels_i = torch.full((num_preds_i,), 1, dtype=torch.long, device=logits.device)
            gt_labels_i = torch.full((num_gts_i,), 1, dtype=torch.long, device=logits.device)

            preds_for_map.append(dict(masks=query_masks, scores=query_scores, labels=pred_labels_i))
            target_list.append(dict(masks=targets[i], labels=gt_labels_i))

            if method == "cls":
                pred_pq_masks_i = query_masks[query_scores > threshold]
            else:
                pred_pq_masks_i = pq_masks_list[i]

            n_pred_i = len(pred_pq_masks_i)
            pred_panoptic_instances.append(pred_pq_masks_i)
            pred_panoptic_classes.append(torch.full((n_pred_i,), 1, dtype=torch.long, device=logits.device))

            gt_panoptic_instances.append(targets[i])
            gt_panoptic_classes.append(torch.full((num_gts_i,), 1, dtype=torch.long, device=logits.device))

        pred_panoptic = self._build_panoptic_tensor(pred_panoptic_instances, pred_panoptic_classes)
        tgt_panoptic = self._build_panoptic_tensor(gt_panoptic_instances, gt_panoptic_classes)
        self.pq.update(pred_panoptic, tgt_panoptic)
        self.map.update(preds_for_map, target_list)

    def _add_pos(self, patch_feat: Tensor, size: tuple[int, int], cls_feat: Tensor | None = None) -> Tensor:
        if cls_feat is None:
            feat = torch.cat((torch.zeros_like(patch_feat[:, 0:1]), patch_feat), dim=1)
        else:
            feat = torch.cat((cls_feat.unsqueeze(1), patch_feat), dim=1)
        feat = feat + self.encoder.interpolate_pos_encoding(feat, *size)
        if cls_feat is None:
            return feat[:, 1:]
        return feat

    def _forward_mask_head(
        self,
        x: Tensor,
        feat: Tensor,
        feature: list[tuple[Tensor, Tensor]],
        size: tuple[int, int],
    ) -> Tensor:
        h = self.n_head
        d = self.n_embd // h
        t = self.num_objs
        s = self.encoder.patch_size
        if feat.ndim == 3:
            b, ji, _c = feat.size()
            j = i = int(ji**0.5)
        else:
            b, _c, j, i = feat.size()

        if "mask" in self.mode:
            x = torch.einsum("b t c, b c j i -> b t j i", self.head(x), feat)
        else:
            query_pos = self.query_pos.weight.unsqueeze(0).expand(b, -1, -1)
            qk = v = self.block.ln_1(x)
            q = k = qk + query_pos
            if self.block.self_attn is None:
                raise RuntimeError("Self-attention is required in this decoding branch.")
            x = x + self.block.dp_1(self.block.self_attn(q, k, v))
            x = self.block.ln_2(x) + query_pos
            if self.block.cross_attn is None:
                raise RuntimeError("Cross-attention is required in this decoding branch.")
            q = self.block.cross_attn.to_q(x) * d**-0.5
            k = self.block.cross_attn.to_k(self._add_pos(feat, (j * s, i * s)))
            v = self.block.cross_attn.to_v(feat)
            if "detr" in self.mode:
                q = rearrange(q, "b t (h d) -> b t h d", h=h)
                k, v = (rearrange(x, "b (j i) (h d) -> b h d j i", h=h, j=j, i=i) for x in (k, v))
                sim = torch.einsum("b t h d, b h d j i -> b t h j i", q, k)
                sim = sim - sim.detach().flatten(3).amax(-1, keepdim=True).unsqueeze(-1)
                attn = F.softmax(sim.flatten(3), dim=-1).view_as(sim)
                # x_exp = torch.cat((_expand(feat, t), attn.flatten(0, 1)), dim=1)
                x_exp = torch.einsum("b t h j i, b h d j i -> b t h d j i", attn, v)
                x_exp = rearrange(x_exp, "b t h d j i -> (b t) (h d) j i")
            elif "instr" in self.mode:
                q, k, v = (rearrange(x, "b t (h d) -> (b h) t d", h=h) for x in (q, k, v))
                sim = torch.einsum("r t d, r l d -> r t l", q, k)
                attn = (sim - sim.detach().amax(dim=-1, keepdim=True)).softmax(dim=-1)
                x_exp = torch.einsum("r t l, r l d -> r t l d", attn, v)
                x_exp = rearrange(x_exp, "(b h) t (j i) d -> (b t) (h d) j i", h=h, j=j, i=i)

            if "fpn" in self.mode:
                interp = partial(F.interpolate, mode="bilinear", align_corners=False)
                feature = [interp(f[0], scale_factor=2 ** (i + 1)) for i, f in enumerate(reversed(feature[:-1]))]
                x = self.head(x_exp, feature)
            else:
                x = self.head(x_exp)

        if self.training and self.sample and not self.pred_det:
            return x
        return F.interpolate(x, size=size, mode="bilinear").view(b, t, *size)

    def forward(self, inputs: Tensor, **kwargs) -> dict[str, Tensor | list[Tensor]]:
        return self.decode(feature=self.encode(inputs, **kwargs), **kwargs)

    def encode(self, inputs, **kwargs) -> list[tuple[Tensor, Tensor]]:
        return self.encoder(kwargs.get("inputs.image", inputs))

    def decode(
        self,
        feature: list[tuple[Tensor, Tensor]],
        global_step: int | None = None,
        total_steps: int | None = None,
        **kwargs,
    ) -> dict[str, Tensor | list[Tensor]]:
        patch_feat, cls_feat = feature[-1]
        t = self.num_objs
        s = self.encoder.patch_size
        b, _c, j, i = patch_feat.size()
        size = j * s, i * s
        out = dict()

        if "conv" in self.mode:
            if self.dpt_head is not None:
                patch_feat = self.dpt_head(feature)
                return dict(logits=F.interpolate(self.head(patch_feat), size=size, mode="bilinear"))
            return dict(logits=self.head(patch_feat, size=size))

        patch_feat = rearrange(patch_feat, "b c j i -> b (j i) c")
        feat_hd = patch_feat
        if self.dpt_head is not None:
            feat_hd = self.dpt_head(feature)

            if self.depth_head is not None:
                d = self.depth_head[0](feat_hd)
                d = F.interpolate(d, size=size, mode="bilinear", align_corners=False)
                d = self.depth_head[1](d)
                out["depth"] = d.squeeze(1)

            if "mask" not in self.mode:
                feat_hd = rearrange(feat_hd, "b c j i -> b (j i) c")

        if self.combine_feat == "gate":
            for f in feature[:-1]:
                patch_feat = self.feat_gate(torch.cat((patch_feat, rearrange(f[0], "b c j i -> b (j i) c")), dim=-1))
        elif self.combine_feat == "add":
            patch_feat += sum(rearrange(f[0], "b c j i -> b (j i) c") for f in feature[:-1])
        elif self.combine_feat == "cat":
            patch_feat_list = [rearrange(f[0], "b c j i -> b (j i) c") for f in feature[:-1]]
            cls_feat_list = [f[1] for f in feature[:-1]]

        query_pos = self.query_pos.weight.unsqueeze(0).expand(b, -1, -1)
        x = torch.zeros_like(query_pos)

        aux_out_list: list[dict[str, Tensor]] = []
        if self.queries_from_feat:
            q_feat = patch_feat
            if self.combine_feat == "cat":
                q_feat = torch.cat((*patch_feat_list, patch_feat), dim=1)

            def score_fn(x: Tensor) -> Tensor:
                return self.cls_head(x).squeeze(2)

            select = (
                "max"
                if "max" in self.queries_from_feat
                else "mean"
                if "mean" in self.queries_from_feat
                else "norm"
                if "norm" in self.queries_from_feat
                else "fps"
                if "fps" in self.queries_from_feat
                else "cls"
            )
            gather = (
                "soft"
                if "soft" in self.queries_from_feat
                else "gumbel"
                if "gumbel" in self.queries_from_feat
                else "hard"
            )
            tau = 0.5
            beta = 0.5 if self.straight_through else 0.75
            if "anneal" in self.queries_from_feat and global_step is not None and total_steps is not None:
                tau = cosine_anneal(1.0, 0.5 if self.straight_through else 0.1, int(0.8 * total_steps), global_step)
            scores = None
            if select == "cls":
                scores = score_fn(q_feat)
                select = None
            x = queries_from_feat(
                queries=q_feat,
                scores=scores,
                num_queries=self.num_objs,
                select=select,
                gather=gather,
                ste=self.straight_through,
                tau=tau,
                beta=beta,
            )

            if self.aux_loss or self.pred_det:
                masks = self._forward_mask_head(x, feat_hd, feature, size)
                if self.pred_det:
                    if "init" in self.mode:
                        dets = self.init_dets.weight.unsqueeze(0).expand(b, -1, -1)
                    if self.det_type == "box":
                        if "init" not in self.mode:
                            dets = get_boxes_from_masks(masks.view(-1, *size)).view(b, t, 4)
                        dets_ref = self._apply_box_offsets(dets, self.det_head(x))
                    elif self.det_type == "circle":
                        if "init" not in self.mode:
                            dets = get_circles_from_masks(masks.view(-1, *size)).view(b, t, 3)
                        dets_ref = self._apply_circle_offsets(dets, self.det_head(x))
                    dets = dets_ref.detach()
                if self.aux_loss:
                    aux_out = dict(logits=masks)
                    if self.pred_cls:
                        aux_out["cls_logits"] = self.cls_head(x).squeeze(-1)
                    if self.pred_det:
                        aux_out["dets"] = dets_ref
                    aux_out_list.append(aux_out)
                    if self.detach_queries:
                        x = x.detach()

        k_cross = self._add_pos(patch_feat, size, cls_feat)
        v_cross = torch.cat((cls_feat.unsqueeze(1), patch_feat), dim=1)
        if self.combine_feat == "cat":
            feat_list = [self._add_pos(pf, size, cf) for pf, cf in zip(patch_feat_list, cls_feat_list, strict=False)]
            k_cross = torch.cat((*feat_list, k_cross), dim=1)
            feat_list = [
                torch.cat((cf.unsqueeze(1), pf), dim=1) for pf, cf in zip(patch_feat_list, cls_feat_list, strict=False)
            ]
            v_cross = torch.cat((*feat_list, v_cross), dim=1)
        for layer_idx, layer in enumerate(cast(Any, self.query_enc).layers):
            layer = cast(Any, layer)
            # self-attention
            qk = v = layer.ln_1(x)
            q = k = qk + query_pos
            x = x + layer.dp_1(layer.self_attn(q, k, v))

            # cross-attention
            q_cross = layer.ln_2(x) + query_pos
            x = x + layer.dp_2(layer.cross_attn(q_cross, k_cross, v_cross))

            # projection
            x = x + layer.dp_3(layer.projection(layer.ln_3(x)))

            if (self.aux_loss or self.pred_det) and layer_idx < len(self.query_enc.layers) - 1:
                if self.pred_det:
                    if self.det_type == "box":
                        dets_ref = self._apply_box_offsets(dets, self.det_head(x))
                    elif self.det_type == "circle":
                        dets_ref = self._apply_circle_offsets(dets, self.det_head(x))
                    dets = dets_ref.detach()
                if self.aux_loss:
                    masks = self._forward_mask_head(x, feat_hd, feature, size)
                    aux_out = dict(logits=masks)
                    if self.pred_cls:
                        aux_out["cls_logits"] = self.cls_head(x).squeeze(-1)
                    if self.pred_det:
                        aux_out["dets"] = dets_ref
                    aux_out_list.append(aux_out)

        out["logits"] = self._forward_mask_head(x, feat_hd, feature, size)
        if self.pred_cls:
            out["cls_logits"] = self.cls_head(x).squeeze(-1)
        if self.pred_det:
            if self.det_type == "box":
                out["dets"] = self._apply_box_offsets(dets, self.det_head(x))
            elif self.det_type == "circle":
                out["dets"] = self._apply_circle_offsets(dets, self.det_head(x))
        if aux_out_list:
            out["aux_out"] = aux_out_list
        return out

    @torch.inference_mode()
    def evaluate(
        self,
        data: dict[str, Any],
        out: dict[str, Any] | None = None,
        show: bool = False,
        **kwargs,
    ) -> dict[str, float]:
        if out is None:
            out = cast(dict[str, Any], self(**{**data, **kwargs}))
        logits = cast(Tensor, out["logits"])
        dets = cast(Tensor | None, out.get("dets"))
        cls_logits = cast(Tensor | None, out.get("cls_logits"))
        masks = cast(list[Tensor], data["inputs.masks"])
        boxes = cast(Tensor | None, data.get("inputs.boxes"))

        cls_preds = cls_probs_from_logits(logits) if cls_logits is None else cls_logits.sigmoid()
        self._update_metrics(logits, masks, cls_preds, method="greedy" if cls_logits is None else "cls")

        match, cost = hungarian_matcher(
            batch_size=len(masks),
            mask_logits=logits,
            mask_tgt=masks,
            dets=dets if self.match_det else None,
            boxes=boxes if self.match_det else None,
            cls_preds=cls_preds if self.match_cls else None,
            mask_weight=self.mask_weight,
            cls_weight=self.cls_weight,
            det_weight=self.det_weight,
            det_type=cast(str, self.det_type or "box"),
        )
        self.log("cost", cost, level=DEBUG_LEVEL_2)

        batch_idx, src_idx = index_from_match(match)
        pred_masks = logits[(batch_idx, src_idx)]
        tgt_masks = torch.cat([masks[i][j] for i, (_, j) in enumerate(match)])

        if show and len(logits) == 1:
            if "depth" in out:
                disp = cast(Tensor, out["depth"])
                depth = 1.0 / (disp + 1e-3)
                depth_to_image(depth[0].float().cpu().numpy().clip(0, 10), cmap="magma").show()
                # show_point_cloud_from_depth(depth, data["inputs.intrinsic"], data["inputs.extrinsic"])

            image = cast(Tensor, data.get("inputs.image", data["inputs"]))
            _b, _c, h, w = image.size()
            pred_dets = None
            if dets is not None:
                pred_dets = dets[cls_preds > 0.5]
                if self.det_type == "box":
                    pred_dets = box_cxcywh_to_xywh(pred_dets)
                elif self.det_type == "circle":
                    pred_dets = box_xyxy_to_xywh(get_boxes_from_circles(pred_dets))
                pred_dets = pred_dets * torch.tensor([w, h, w, h], device=pred_dets.device)

            masks_pred = self.predict(inputs=image)
            show_image_with_masks(image=image[0], masks=masks_pred[0], boxes=pred_dets)

        eval_data = {
            "loss": dice_loss(pred_masks, tgt_masks),
            "logits": pred_masks.flatten(1),
            "points.occ": tgt_masks.flatten(1),
        }

        if not any(len(t) for t in masks):  # If all masks are empty
            val_metrics = EMPTY_EVAL_RESULTS_DICT.copy()
            val_metrics["loss"] = eval_data["loss"].cpu().item()
        else:
            val_metrics = super().evaluate(eval_data, **kwargs)

        if dets is not None:
            pred_dets = dets[batch_idx, src_idx]
            if masks is None:
                if boxes is None:
                    raise ValueError("inputs.boxes is required when masks are unavailable.")
                _b, h, w = tgt_masks.size()
                tgt_dets = torch.cat([boxes[i][j] for i, (_, j) in enumerate(match)])
                tgt_dets = tgt_dets / torch.tensor([w, h, w, h], device=tgt_dets.device)
                if self.det_type == "box":
                    tgt_dets = box_xywh_to_xyxy(tgt_dets)
                elif self.det_type == "circle":
                    tgt_dets = get_circles_from_boxes(box_xywh_to_xyxy(tgt_dets))
            else:
                if self.det_type == "box":
                    tgt_dets = get_boxes_from_masks(tgt_masks)
                    tgt_dets = box_cxcywh_to_xyxy(tgt_dets)
                elif self.det_type == "circle":
                    tgt_dets = get_circles_from_masks(tgt_masks)

            if self.det_type == "box":
                pred_dets = box_cxcywh_to_xyxy(pred_dets)
                det_iou = box_iou(pred_dets, tgt_dets)[0].diag()
            elif self.det_type == "circle":
                det_iou = box_iou(get_boxes_from_circles(pred_dets), get_boxes_from_circles(tgt_dets))[0].diag()
                val_metrics["val/det_ciou"] = generalized_circle_iou(pred_dets, tgt_dets, return_iou=True).mean().item()
            val_metrics["val/det_iou"] = det_iou.mean().item()

        disp_pred = out.get("depth")
        if disp_pred is not None:
            depth_gt = cast(Tensor, data.get("inputs.depth", data["inputs"]))
            disp_gt = depth_gt.clone().clamp(0, 1e3)
            disp_gt[disp_gt == 0] = 1e3
            disp_gt = 1.0 / disp_gt
            val_metrics["val/depth_loss"] = self.depth_loss(disp_pred + 1e-3, disp_gt)

        return val_metrics

    @torch.no_grad()
    def on_validation_epoch_end(self, *args, **kwargs) -> dict[str, float]:
        with stdout_redirected(enabled=not logger.isEnabledFor(DEBUG_LEVEL_2)):
            map_metrics = {k: v.cpu().item() for k, v in self.map.compute().items()}
            pq_metrics = {k: v.cpu().item() for k, v in zip(["pq", "sq", "rq"], self.pq.compute(), strict=False)}
        self.map.reset()
        self.pq.reset()
        return {**map_metrics, **pq_metrics}

    @torch.inference_mode()
    def predict(
        self,
        inputs: Tensor,
        threshold: float = 0.5,
        method: Literal["argmax", "cls", "greedy"] | None = None,
        apply_filter: bool = True,
        min_area: int | None = 16,
        nms_iou: float | None = 0.5,
        **kwargs,
    ) -> list[Tensor]:
        out = self(inputs=inputs, **kwargs)
        logits = out["logits"]
        cls_logits = out.get("cls_logits")
        cls_preds = cls_probs_from_logits(logits) if cls_logits is None else cls_logits.sigmoid()

        if method is None:
            method = "greedy" if cls_logits is None else "cls"

        if method == "argmax":
            labels = labels_from_logits(logits, threshold)
            return masks_from_labels(labels, reindex=True)
        if method == "greedy":
            return masks_from_logits(
                logits=logits,
                scores=cls_preds,
                threshold=threshold,
                apply_filter=apply_filter,
                min_size=min_area,
                nms_iou=nms_iou,
            )

        masks_per_batch: list[Tensor] = []
        for i in range(len(logits)):
            inst_masks = logits[i]
            inst_scores = cls_preds[i]
            if apply_filter:
                inst_masks, inst_scores, _ = filter_instance_masks(
                    masks=inst_masks, scores=inst_scores, min_size=min_area, nms_iou=nms_iou
                )
            masks_per_batch.append(inst_masks[inst_scores > threshold])

        return masks_per_batch

    def _get_loss(
        self,
        mask_logits: Tensor,
        mask_gt: Tensor | list[Tensor],
        cls_logits: Tensor | None = None,
        dets: Tensor | None = None,
        boxes: Tensor | None = None,
        match: list[tuple[Tensor, Tensor]] | None = None,
        name: str = "dice+bce",
        reduction: str | None = "mean",
        log: bool = True,
        return_match: bool = False,
        **kwargs,
    ) -> Tensor | tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        reduction_val = reduction or "mean"
        mask_gt_list = mask_gt if isinstance(mask_gt, list) else [mask_gt[i] for i in range(mask_gt.size(0))]
        cls_preds = cls_probs_from_logits(mask_logits) if cls_logits is None else cls_logits.sigmoid()
        if match is None:
            match, cost = hungarian_matcher(
                batch_size=len(mask_gt_list),
                mask_logits=mask_logits,
                mask_tgt=mask_gt_list,
                cls_preds=cls_preds if self.match_cls else None,
                dets=dets if self.match_det else None,
                boxes=boxes if self.match_det else None,
                mask_weight=self.mask_weight,
                cls_weight=self.cls_weight,
                det_weight=self.det_weight,
                sample=self.sample,
                det_type=cast(str, self.det_type or "box"),
            )
            if log:
                self.log("cost", cost, level=DEBUG_LEVEL_2)

        batch_idx, src_idx = index_from_match(match)
        if self.pad_targets:
            pred_masks = mask_logits
            tgt_masks = torch.zeros_like(pred_masks)
            for i, (si, ti) in enumerate(match):
                tgt_masks[i, si] = mask_gt_list[i][ti].type_as(tgt_masks)
            pred_masks = pred_masks.flatten(0, 1)
            tgt_masks = tgt_masks.flatten(0, 1)
        else:
            pred_masks = mask_logits[(batch_idx, src_idx)]
            tgt_masks = torch.cat([mask_gt_list[i][j] for i, (_, j) in enumerate(match)]).type_as(pred_masks)

        if self.sample:
            from detectron2.projects.point_rend.point_features import (  # pyright: ignore[reportMissingImports]
                get_uncertain_point_coords_with_randomness,
                point_sample,
            )

            with torch.no_grad():
                point_coords = get_uncertain_point_coords_with_randomness(
                    pred_masks.unsqueeze(1),
                    uncertainty_func=lambda x: -x.detach().abs(),
                    num_points=self.sample,
                    oversample_ratio=3.0,
                    importance_sample_ratio=0.75,
                )
                tgt_masks = point_sample(tgt_masks.unsqueeze(1), point_coords, align_corners=False).squeeze(1)
            pred_masks = point_sample(pred_masks.unsqueeze(1), point_coords, align_corners=False).squeeze(1)

        loss_total = mask_logits.new_tensor(0.0)
        loss_mask = mask_logits.new_tensor(0.0)
        if "dice" in name:
            if name and "instr" in name:
                loss_d = dice_loss_instr(pred_masks, tgt_masks)
            else:
                loss_d = dice_loss(pred_masks, tgt_masks, reduction=reduction_val)
            loss_mask += loss_d
        if "focal" in name:
            alpha = self.alpha
            if "auto" in name and tgt_masks.any():
                alpha = float((1 - tgt_masks.sum() / tgt_masks.numel()).item())
            loss_f = focal_loss.sigmoid_focal_loss(pred_masks, tgt_masks, alpha, self.gamma, reduction=reduction_val)
            loss_mask += loss_f
        elif "bce" in name:
            pos_weight = torch.tensor(1.0, device=pred_masks.device, dtype=pred_masks.dtype)
            if "auto" in name and tgt_masks.any():
                sum_m = tgt_masks.sum()
                num_m = tgt_masks.numel()
                pos_weight = ((num_m - sum_m).clamp_min(1.0) / sum_m.clamp_min(1.0)).to(pred_masks)
            loss_b = F.binary_cross_entropy_with_logits(
                pred_masks, tgt_masks, pos_weight=pos_weight, reduction=reduction_val
            )
            loss_mask += loss_b
        elif "ce" in name:
            loss_c = 0
            for i, (si, ti) in enumerate(match):
                preds = mask_logits[i, si]
                _, h, w = preds.size()
                preds_bg = torch.zeros(1, h, w, device=preds.device)
                target = preds_bg[0].long()
                if ti.numel() > 0:
                    target = mask_gt_list[i][ti].float()
                    n = target.size(0)
                    indices = torch.arange(1, n + 1, device=target.device).view(n, 1, 1)
                    target = torch.max(target * indices, dim=0).values.long()
                preds = torch.cat([preds_bg, preds], dim=0)
                loss_c += F.cross_entropy(preds.unsqueeze(0), target.unsqueeze(0), reduction="sum")
            b, n, h, w = mask_logits.size()
            loss_c /= b * h * w
            loss_mask += loss_c

        if self.learn_loss_weights:
            loss_mask = loss_mask * torch.exp(-self.log_var_mask) + self.log_var_mask
            if log:
                self.log("mask_weight", torch.exp(-self.log_var_mask).item(), level=DEBUG_LEVEL_2)
        loss_total += self.mask_weight * loss_mask

        tgt_cls = torch.zeros_like(cls_preds)
        tgt_cls[batch_idx, src_idx] = 1.0
        if self.pred_cls in ["qual", "quality"]:
            with torch.no_grad():
                score = calculate_dice_score(pred_masks.detach().sigmoid().flatten(1), tgt_masks.flatten(1))
                tgt_cls[batch_idx, src_idx] = score.type_as(tgt_cls)

        if cls_logits is not None:
            if "focal" in name:
                alpha = self.alpha
                if "auto" in name and tgt_cls.any():
                    alpha = float((1.0 - (tgt_cls.sum() / tgt_cls.numel())).item())
                loss_cls = focal_loss.sigmoid_focal_loss(
                    cls_logits, tgt_cls, alpha, self.gamma, reduction=reduction_val
                )
            else:
                pos_weight = torch.tensor(1.0, device=cls_logits.device, dtype=cls_logits.dtype)
                if "auto" in name and tgt_cls.any():
                    sum_c = tgt_cls.sum()
                    num_c = tgt_cls.numel()
                    pos_weight = ((num_c - sum_c).clamp_min(1.0) / sum_c.clamp_min(1.0)).to(cls_logits)
                loss_cls = F.binary_cross_entropy_with_logits(
                    cls_logits, tgt_cls, pos_weight=pos_weight, reduction=reduction_val
                )

            if log:
                self.log("cls_loss", cast(Any, loss_cls), level=DEBUG_LEVEL_1)

            if self.learn_loss_weights:
                loss_cls = loss_cls * torch.exp(-self.log_var_cls) + self.log_var_cls
                if log:
                    self.log("cls_weight", torch.exp(-self.log_var_cls).item(), level=DEBUG_LEVEL_2)
            loss_total = loss_total + self.cls_weight * loss_cls

        if dets is not None:
            pred_dets = dets[batch_idx, src_idx]
            if mask_gt is None:
                if boxes is None:
                    raise ValueError("boxes is required when mask targets are unavailable.")
                b, h, w = tgt_masks.size()
                tgt_dets = torch.cat([boxes[i][j] for i, (_, j) in enumerate(match)])
                tgt_dets = tgt_dets / torch.tensor([w, h, w, h], device=tgt_dets.device)
                if self.det_type == "box":
                    tgt_dets = box_xywh_to_cxcywh(tgt_dets)
                elif self.det_type == "circle":
                    tgt_dets = get_circles_from_boxes(box_xywh_to_xyxy(tgt_dets))
            else:
                if self.det_type == "box":
                    tgt_dets = get_boxes_from_masks(tgt_masks)
                elif self.det_type == "circle":
                    tgt_dets = get_circles_from_masks(tgt_masks)

            det_l1_loss = F.l1_loss(pred_dets, tgt_dets)
            giou = torch.ones(pred_dets.size(0), device=pred_dets.device, dtype=pred_dets.dtype)
            if self.det_type == "box":
                pred_dets = box_cxcywh_to_xyxy(pred_dets)
                tgt_dets = box_cxcywh_to_xyxy(tgt_dets)
                giou = generalized_box_iou(pred_dets, tgt_dets).diag()
            elif self.det_type == "circle":
                # giou = generalized_circle_iou(pred_dets, tgt_dets)
                giou = generalized_box_iou(get_boxes_from_circles(pred_dets), get_boxes_from_circles(tgt_dets)).diag()
            det_giou_loss = (1 - giou).mean()
            loss_det = det_l1_loss + det_giou_loss
            if log:
                if logger.isEnabledFor(DEBUG_LEVEL_2):
                    self.log("l1_loss", cast(Any, det_l1_loss), level=DEBUG_LEVEL_2)
                    self.log("giou_loss", cast(Any, det_giou_loss), level=DEBUG_LEVEL_2)
                else:
                    self.log("det_loss", cast(Any, loss_det), level=DEBUG_LEVEL_1)

            if self.learn_loss_weights:
                loss_det = loss_det * torch.exp(-self.log_var_det) + self.log_var_det
                if log:
                    self.log("det_weight", torch.exp(-self.log_var_det).item(), level=DEBUG_LEVEL_2)
            loss_total += self.det_weight * loss_det

        if log:
            if name and ("focal" in name or "bce" in name or "ce" in name) and "dice" in name:
                self.log("dice_loss", loss_d, level=DEBUG_LEVEL_1)
                comp_loss: float | Tensor
                if "focal" in name:
                    comp_loss = loss_f
                elif "bce" in name:
                    comp_loss = loss_b
                else:
                    comp_loss = loss_c
                self.log(
                    f"{'focal' if 'focal' in name else 'bce' if 'bce' in name else 'ce'}_loss",
                    cast(Any, comp_loss),
                    level=DEBUG_LEVEL_1,
                )

            if logger.isEnabledFor(DEBUG_LEVEL_2):
                self._eval_cls_preds(cls_preds, tgt_cls > 0)
                if self.pad_targets or self.sample:
                    pred_masks = mask_logits[(batch_idx, src_idx)]
                    tgt_masks = torch.cat([mask_gt_list[i][j] for i, (_, j) in enumerate(match)]).type_as(pred_masks)
                train_data = {"loss": loss_total, "logits": pred_masks.flatten(1), "points.occ": tgt_masks.flatten(1)}
                train_metrics = super().evaluate(train_data, prefix="train/", **kwargs)
                train_metrics.pop("train/loss")
                if dets is not None:
                    with torch.no_grad():
                        if self.det_type == "box":
                            det_iou = box_iou(pred_dets, tgt_dets)[0].diag()
                        elif self.det_type == "circle":
                            det_iou = box_iou(get_boxes_from_circles(pred_dets), get_boxes_from_circles(tgt_dets))[
                                0
                            ].diag()
                            train_metrics["train/det_ciou"] = (
                                generalized_circle_iou(pred_dets, tgt_dets, return_iou=True).mean().item()
                            )
                        train_metrics["train/det_iou"] = det_iou.mean().item()
                self.log_dict(train_metrics, level=DEBUG_LEVEL_2, train_only=True)

        if return_match:
            return loss_total, match
        return cast(Tensor, loss_total)

    def loss(
        self,
        data: dict[str, Any],
        out: dict[str, Any] | None = None,
        name: str | None = None,
        reduction: str | None = "mean",
        log_freq: int = 10,
        global_step: int | None = None,
        total_steps: int | None = None,
        show: bool = False,
        **kwargs,
    ) -> Tensor:
        if show:
            inputs_show = cast(Tensor, data["inputs"])
            masks_show = cast(list[Tensor], data["inputs.masks"])
            boxes_show = cast(Tensor, data["inputs.boxes"])
            labels_show = cast(Tensor, data["inputs.labels"])
            for i in range(len(cast(list[int], data["index"]))):
                show_image_with_masks(
                    inputs_show[i],
                    masks_show[i],
                    boxes_show[i],
                    labels_show[i],
                )

        if out is None:
            out = cast(dict[str, Any], self(**{**data, **kwargs}, global_step=global_step, total_steps=total_steps))
        logits = cast(Tensor, out["logits"])
        masks = cast(list[Tensor], data["inputs.masks"])
        boxes = cast(Tensor | None, data.get("inputs.boxes"))

        if name and "sinkhorn" in name:
            if "batch" in name:
                return cast(Tensor, batched_sinkhorn_segmentation_loss(logits, masks, reduction=reduction))
            return cast(Tensor, sinkhorn_segmentation_loss(logits, masks, reduction=reduction))

        loss, match = cast(
            tuple[Tensor, list[tuple[Tensor, Tensor]]],
            self._get_loss(
                mask_logits=logits,
                mask_gt=masks,
                cls_logits=cast(Tensor | None, out.get("cls_logits")),
                dets=cast(Tensor | None, out.get("dets")),
                boxes=boxes,
                name=name or self.loss_name,
                reduction=reduction,
                log=True if global_step is None else global_step % log_freq == 0,
                return_match=True,
                **kwargs,
            ),
        )

        aux_out = cast(list[dict[str, Any]] | None, out.get("aux_out"))
        if aux_out:
            aux_loss = logits.new_tensor(0.0)
            for aux in aux_out:
                aux_loss = aux_loss + cast(
                    Tensor,
                    self._get_loss(
                        mask_logits=aux["logits"],
                        mask_gt=masks,
                        cls_logits=cast(Tensor | None, aux.get("cls_logits")),
                        dets=cast(Tensor | None, aux.get("dets")),
                        boxes=boxes,
                        match=match if self.global_match else None,
                        name=name or self.loss_name,
                        reduction=reduction,
                        log=False,
                        **kwargs,
                    ),
                )
            self.log("aux_loss", cast(Any, aux_loss), level=DEBUG_LEVEL_1)
            loss += cast(float, 0.0 if self.aux_weight is None else self.aux_weight) * aux_loss

        disp_pred = out.get("depth")
        if disp_pred is not None:
            depth_gt = cast(Tensor, data.get("inputs.depth", data["inputs"]))
            disp_gt = depth_gt.clone().clamp(0, 1e3)
            disp_gt[disp_gt == 0] = 1e3
            disp_gt = 1.0 / disp_gt
            depth_loss = self.depth_loss(disp_pred + 1e-3, disp_gt)
            self.log("depth_loss", cast(Any, depth_loss), level=DEBUG_LEVEL_1)
            if self.learn_loss_weights:
                depth_loss = depth_loss * torch.exp(-self.log_var_depth) + self.log_var_depth
                self.log("depth_weight", torch.exp(-self.log_var_depth).item(), level=DEBUG_LEVEL_2)
            loss += self.depth_weight * depth_loss

        return loss

    def optimizer(
        self, lr: float, weight_decay: float = 0.01, factor: float = 0.1, **kwargs
    ) -> torch.optim.Optimizer | None:
        if self.freeze or not factor or factor == 1.0:
            return None

        enc_params = list(self.encoder.parameters())
        enc_param_ids = {id(p) for p in enc_params}
        params_decay, params_no_decay = assign_params_groups(self)

        enc_params_decay = [p for p in params_decay if id(p) in enc_param_ids]
        enc_params_no_decay = [p for p in params_no_decay if id(p) in enc_param_ids]
        other_params_decay = [p for p in params_decay if id(p) not in enc_param_ids]
        other_params_no_decay = [p for p in params_no_decay if id(p) not in enc_param_ids]

        param_groups = [
            {"params": enc_params_decay, "lr": lr * factor, "weight_decay": weight_decay * factor},
            {"params": enc_params_no_decay, "lr": lr * factor, "weight_decay": 0.0},
            {"params": other_params_decay, "lr": lr, "weight_decay": weight_decay},
            {"params": other_params_no_decay, "lr": lr, "weight_decay": 0.0},
        ]

        return torch.optim.AdamW(param_groups, **kwargs)


class DinoInstSeg3D(MultiEvalMixin, Model):
    """
    3D instance segmentation (point-wise) and instance-centric shape completion.

    Two usage modes
    - Instance segmentation (points=None):
      Uses `inputs` as the point set and predicts per-query masks over the N input points.
      Output logits have shape (B, Q, N).

    - Shape completion (points provided):
      Builds features from `inputs`, then evaluates per-query occupancy on the supplied
      `points`. Output logits have shape (B, Q, M).

    Inputs
    - inputs: (B, N, D) point cloud used to compute features. D must equal `dim` (default 3).
    - points: Optional (B, M, D) query points for completion. If omitted, M=N and `points=inputs`.

    Outputs
    - logits: (B, Q, M) mask/occupancy logits over points.
    - cls_logits: Optional (B, Q) objectness (or instance score) logits.
    - cls_quality: Optional (B, Q) quality/regression logits (if enabled).
    - in_logits: Optional (B, Q, N) extra branch over `inputs` when multitask=True.
    - queries: Optional (B, Q, C) final query embeddings when queries_from_feat is used.
    - aux_out: Optional list of dicts with the same keys as above for deep supervision.

    Predict and metrics
    - Use `predict(..., method={'argmax','cls','greedy'})` to get per-instance boolean masks.
      For 'cls' and 'greedy', masks are thresholded and filtered (min_num_points, NMS).
    - Use `evaluate(...)` to compute 3D MAP and PQ; matching uses Hungarian assignment
      with configurable loss weights and (optional) classification.

    Notes
    - The point encoding dimension is controlled by `dim`. Extra point channels are not consumed
      unless `dim` is increased or the encoder is adapted.
    - For large M or N, use `points_batch_size` in `decode/predict` to reduce memory.
    """

    def __init__(
        self,
        dim: int = 3,
        num_objs: int = 100,
        num_queries: int = 1024,
        num_enc_layers: int = 1,
        num_query_layers: int | None = None,
        repo_or_dir: str = "facebookresearch/dinov2",
        backbone: str = "dinov2_vits14",
        bias: bool = True,
        dropout: float = 0.05,
        drop_path: float = 0.1,
        mlp_ratio: int = 4,
        cls_token: bool = False,
        cat_feat: bool = True,
        cat_point_feat: bool = False,
        points_dec: Literal["multi", "linear", "mlp", "film"] | None = None,
        pred_cls: Literal["objectness", "quality", "quality+objectness"] | None = "quality+objectness",
        detach_cls: bool = False,
        queries_from_feat: Literal["max", "mean", "norm", "fps", "soft", "gumbel", "anneal", "ste", "detach"]
        | None = "detach",
        cos_sim: bool = False,
        logit_scale: bool = False,
        embd_lvls: bool = False,
        mlp_heads: bool = False,
        mlp_occ_head: bool = False,
        match_cls: bool = True,
        anneal_cls: float | None = None,
        pad_targets: bool = False,
        sample: int | tuple[int, int] | None = None,
        loss_name: Literal["dice", "focal", "bce", "instr", "dice+bce"] = "dice+bce",
        mask_weight: float = 1.0,
        mask_pos_weight: float | Literal["ratio"] | None = None,
        bce_focal_weight: float = 5.0,
        dice_weight: float = 2.0,
        cls_weight: float = 0.5,
        cls_pos_weight: float | Literal["ratio", "dice"] | None = None,
        aux_weight: float | None = None,
        init_weights: bool = False,
        learn_loss_weights: bool | str = False,
        multitask: Literal["head", "dec"] | bool | None = None,
        points_weight: float = 1.0,
        inputs_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        cls_threshold: float = 0.5,
        min_mask_size: int | None = 64,
        apply_filter: bool = True,
        nerf_enc: Literal["tcnn", "torch", "hash"] = "tcnn" if TCNN_EXISTS else "torch",
        nerf_freqs: int = 6,
        depth_descriptor: Literal["local"] | None = None,
        voxel_anchor_resolutions: Sequence[int] | None = None,
        extent_weight: float = 0.0,
        extent_quantile_temperature: float = 0.02,
        allow_branch_warmstart: bool = False,
    ):
        super().__init__()
        warnings.filterwarnings("ignore", message="xFormers is available.*")
        self.encoder = _load_dino_backbone(
            repo_or_dir,
            backbone,
            # mlp_ratio=mlp_ratio,
            qkv_bias=bias,
            ffn_bias=bias,
            proj_bias=bias,
            drop_path_rate=drop_path,
            pretrained=False,
        )
        self.encoder.prepare_tokens_with_masks = lambda x, masks=None: x
        self.encoder.patch_embed = None
        self.encoder.pos_embed = None
        self.encoder.mask_token = None

        out_index = {
            "dinov2_vits14": [2, 5, 8, 11],
            "dinov2_vitb14": [2, 5, 8, 11],
            "dinov2_vitl14": [4, 11, 17, 23],
            "dinov2_vitg14": [9, 19, 29, 39],
        }[backbone.strip("_reg")]
        self.encoder.forward = partial(
            self.encoder.get_intermediate_layers, n=out_index, reshape=False, return_class_token=True, norm=True
        )

        self.n_embd = self.encoder.embed_dim
        self.n_head = self.encoder.num_heads
        self.num_feat_layers = len(out_index)

        self.num_objs = num_objs
        self.num_queries = num_queries
        self.cls_token = cls_token
        self.cat_feat = cat_feat
        self.cat_point_feat = cat_point_feat
        self.cos_sim = cos_sim
        self.pred_cls = "" if pred_cls is None else pred_cls
        self.detach_cls = detach_cls
        self.queries_from_feat = queries_from_feat
        self.match_cls = match_cls
        self.anneal_cls = anneal_cls
        self.pad_targets = pad_targets
        self.sample = sample
        self.loss_name = loss_name
        self.multitask = multitask
        self.cls_threshold = cls_threshold
        self.min_mask_size = min_mask_size
        self.apply_filter = apply_filter
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.depth_descriptor = depth_descriptor
        self.voxel_anchor_resolutions = tuple(int(r) for r in (voxel_anchor_resolutions or ()))
        self.extent_weight = float(extent_weight)
        self.extent_quantile_temperature = float(extent_quantile_temperature)
        self.allow_branch_warmstart = allow_branch_warmstart

        if self.depth_descriptor not in (None, "local"):
            raise ValueError(f"Unsupported depth descriptor: {self.depth_descriptor}")
        if any(r < 2 for r in self.voxel_anchor_resolutions):
            raise ValueError("voxel_anchor_resolutions must contain integers greater than one")
        if self.extent_weight < 0:
            raise ValueError("extent_weight must be non-negative")
        if self.extent_quantile_temperature <= 0:
            raise ValueError("extent_quantile_temperature must be positive")

        self.map = MeanAveragePrecision3D()
        self.pq = PanopticQuality3D()

        if nerf_enc == "hash":
            self.nerf_enc = nn.Sequential(
                HashGridEncoding(
                    in_dim=dim,
                    out_dim=self.n_embd,
                    n_levels=16,  # 10-16
                    n_features_per_level=2,
                    log2_hashmap_size=18,  # 17-22
                    base_resolution=16,
                    per_level_scale=1.5,
                    normalize_inputs=True,
                ),
                nn.LayerNorm(self.n_embd),
            )
        else:
            self.nerf_enc = NeRFEncoding(
                in_dim=dim,
                out_dim=self.n_embd,
                implementation=nerf_enc,
                num_frequencies=nerf_freqs,
                max_freq_exp=nerf_freqs - 1,
                normalize_inputs=False,
                scale_inputs=False,
            )

        self._branch_warmstart_prefixes: tuple[str, ...] = ()
        if self.depth_descriptor == "local":
            self.depth_descriptor_mlp = nn.Sequential(
                nn.Linear(3, self.n_embd),
                nn.GELU(),
                nn.Linear(self.n_embd, self.n_embd),
            )
            nn.init.zeros_(cast(nn.Linear, self.depth_descriptor_mlp[-1]).weight)
            nn.init.zeros_(cast(nn.Linear, self.depth_descriptor_mlp[-1]).bias)
            self._branch_warmstart_prefixes += ("depth_descriptor_mlp.",)

        if self.voxel_anchor_resolutions:
            self.voxel_axis_embeddings = nn.ModuleList(
                [
                    nn.ModuleList([nn.Embedding(resolution, self.n_embd) for _ in range(3)])
                    for resolution in self.voxel_anchor_resolutions
                ]
            )
            local_dim = 10 * len(self.voxel_anchor_resolutions)
            self.voxel_local_mlp = nn.Sequential(
                nn.Linear(local_dim, self.n_embd),
                nn.GELU(),
                nn.Linear(self.n_embd, self.n_embd),
            )
            self.voxel_output = nn.Sequential(nn.LayerNorm(self.n_embd), nn.Linear(self.n_embd, self.n_embd))
            nn.init.zeros_(cast(nn.Linear, self.voxel_output[-1]).weight)
            nn.init.zeros_(cast(nn.Linear, self.voxel_output[-1]).bias)
            self._branch_warmstart_prefixes += (
                "voxel_axis_embeddings.",
                "voxel_local_mlp.",
                "voxel_output.",
            )

        if num_enc_layers == 1:
            self.inputs_enc = DecoderBlock(
                n_embd=self.n_embd,
                n_head=self.n_head,
                bias=bias,
                dropout=dropout,
                hidden_layer_multiplier=mlp_ratio,
                chunks=2,
            )
        else:
            self.inputs_enc = Decoder(
                n_layer=num_enc_layers,
                n_embd=self.n_embd,
                n_head=self.n_head,
                bias=bias,
                dropout=dropout,
                drop_path_rate=drop_path / 2,
                hidden_layer_multiplier=mlp_ratio,
                chunks=2,
            )

        self.query_pos = nn.Embedding(num_objs, self.n_embd)
        if num_query_layers is None:
            num_query_layers = {"dinov2_vits14": 3, "dinov2_vitb14": 6, "dinov2_vitl14": 9, "dinov2_vitg14": 12}[
                backbone.strip("_reg")
            ]
        self._aux_weight = 1 / num_query_layers if aux_weight is None else aux_weight
        self._aux_loss = self._aux_weight > 0
        self.query_enc = Decoder(
            n_layer=num_query_layers,
            n_embd=self.n_embd,
            n_head=self.n_head,
            bias=bias,
            dropout=dropout,
            drop_path_rate=drop_path / 2,
            hidden_layer_multiplier=mlp_ratio,
            chunks=3,
            k_dim=self.n_embd,
            v_dim=self.n_embd,
        )

        if queries_from_feat:
            self.queries = nn.Linear(self.n_embd, self.n_embd)
            if mlp_heads:
                self.queries = nn.Sequential(
                    nn.Linear(self.n_embd, self.n_embd),
                    nn.GELU(),
                    nn.Linear(self.n_embd, self.n_embd),
                )

        if embd_lvls:
            self.lvl_embd = nn.Embedding(self.num_feat_layers, self.n_embd)
            self.lvl_norm = nn.LayerNorm(self.n_embd)

        points_dec_cls = partial(
            Block,
            n_embd=self.n_embd,
            n_head=self.n_head,
            bias=bias,
            dropout=dropout,
            hidden_layer_multiplier=mlp_ratio,
            cross_attn=True,
            chunks=2,
        )
        if points_dec:
            if "linear" in points_dec:
                points_dec_cls = partial(points_dec_cls, projection="linear")
            elif "mlp" in points_dec:
                points_dec_cls = partial(points_dec_cls, projection="mlp", hidden_layer_multiplier=1)
            if cls_token and "film" in points_dec:
                self.film = nn.Linear(self.n_embd, 2 * self.n_embd)
            if cat_feat and "multi" in points_dec:
                self.points_dec = nn.ModuleList(
                    [
                        points_dec_cls(drop_path=(drop_path * i) / self.num_feat_layers)
                        for i in range(self.num_feat_layers)
                    ]
                )
            else:
                self.points_dec = points_dec_cls()
        else:
            self.points_dec = points_dec_cls()

        self.head_norm = nn.LayerNorm(self.n_embd)
        self.points_head = nn.Linear(self.n_embd, self.n_embd)
        if mlp_heads:
            self.points_head = nn.Sequential(
                nn.Linear(self.n_embd, 2 * self.n_embd),
                nn.GELU(),
                nn.Linear(2 * self.n_embd, self.n_embd),
            )

        if multitask:
            if "head" in str(multitask):
                self.inputs_head = copy.deepcopy(self.points_head)
            if "dec" in str(multitask):
                self.inputs_dec = copy.deepcopy(self.points_dec)
            sample_val = cast(Any, self).sample
            if sample_val and not isinstance(sample_val, Sequence):
                cast(Any, self).sample = (sample_val, sample_val)
            self.map_inputs = MeanAveragePrecision3D()
            self.pq_inputs = PanopticQuality3D()

        if pred_cls:
            self.cls_head = nn.Linear(self.n_embd, 1)
            if mlp_heads:
                self.cls_head = nn.Sequential(
                    nn.Linear(self.n_embd, self.n_embd // 2),
                    nn.GELU(),
                    nn.Linear(self.n_embd // 2, 1),
                )
            if "obj" in pred_cls and "qual" in pred_cls:
                self.cls_quality_head = nn.Linear(self.n_embd, 1)
                if mlp_heads:
                    self.cls_quality_head = nn.Sequential(
                        nn.Linear(self.n_embd, self.n_embd // 2),
                        nn.GELU(),
                        nn.Linear(self.n_embd // 2, 1),
                    )

        if mlp_occ_head:
            self.occ_head = MLP(self.n_embd * 2, n_out=1, bias=bias, dropout=dropout)

        if logit_scale:
            self.logit_scale = nn.Parameter(torch.zeros(()))

        if init_weights:
            self._init_weights()

        self.learn_loss_weights = learn_loss_weights
        self.mask_pos_weight = mask_pos_weight
        self.cls_pos_weight = cls_pos_weight
        self._mask_weight = mask_weight
        self._bce_focal_weight = bce_focal_weight
        self._dice_weight = dice_weight
        self._cls_weight = cls_weight
        self._points_weight = points_weight
        self._inputs_weight = inputs_weight
        if learn_loss_weights:
            if not isinstance(learn_loss_weights, str) or "mask" in learn_loss_weights:
                self._mask_weight = nn.Parameter(torch.tensor(-math.log(2.0 * mask_weight)))
            if not isinstance(learn_loss_weights, str) or "bce" in learn_loss_weights or "focal" in learn_loss_weights:
                if "bce" in loss_name or "focal" in loss_name:
                    self._bce_focal_weight = nn.Parameter(torch.tensor(-math.log(2.0 * bce_focal_weight)))
            if not isinstance(learn_loss_weights, str) or "dice" in learn_loss_weights:
                if "dice" in loss_name:
                    self._dice_weight = nn.Parameter(torch.tensor(-math.log(2.0 * dice_weight)))
            if not isinstance(learn_loss_weights, str) or "aux" in learn_loss_weights:
                if self._aux_weight:
                    self._aux_weight = nn.Parameter(torch.tensor(-math.log(2.0 * self._aux_weight)))
            if not isinstance(learn_loss_weights, str) or "cls" in learn_loss_weights:
                if pred_cls:
                    self._cls_weight = nn.Parameter(torch.tensor(-math.log(2.0 * cls_weight)))
            if multitask:
                self._points_weight = nn.Parameter(torch.tensor(-math.log(2.0 * points_weight)))
                self._inputs_weight = nn.Parameter(torch.tensor(-math.log(2.0 * inputs_weight)))

    def load_state_dict(self, state_dict: dict[str, Any], *args, strict: bool = True, **kwargs):
        allow_branch_warmstart = getattr(self, "allow_branch_warmstart", False)
        branch_warmstart_prefixes = getattr(self, "_branch_warmstart_prefixes", ())
        if not strict or not allow_branch_warmstart or not branch_warmstart_prefixes:
            if _is_legacy_dino_inst_seg_3d_state_dict(state_dict):
                kwargs.setdefault("fuzzy_match", False)
                kwargs.setdefault("shape_suffix_match", False)
            return super().load_state_dict(state_dict, *args, strict=strict, **kwargs)

        warmstart_kwargs = dict(kwargs)
        warmstart_kwargs.setdefault("fuzzy_match", False)
        warmstart_kwargs.setdefault("shape_suffix_match", False)
        incompatible = super().load_state_dict(state_dict, *args, strict=False, **warmstart_kwargs)
        invalid_missing = [
            key
            for key in incompatible.missing_keys
            if not any(key.startswith(prefix) for prefix in branch_warmstart_prefixes)
        ]
        if invalid_missing or incompatible.unexpected_keys:
            missing = ", ".join(invalid_missing) or "none"
            unexpected = ", ".join(incompatible.unexpected_keys) or "none"
            raise RuntimeError(
                "Branch warm-start rejected checkpoint incompatibilities: "
                f"missing=[{missing}], unexpected=[{unexpected}]"
            )
        if incompatible.missing_keys:
            logger.warning(
                "Initialized enabled architecture branch parameters absent from the checkpoint: "
                + ", ".join(incompatible.missing_keys)
            )
        return incompatible

    @property
    def mask_weight(self) -> float | Tensor:
        if isinstance(self._mask_weight, nn.Parameter):
            return 0.5 * torch.exp(-self._mask_weight)
        return self._mask_weight

    @property
    def bce_focal_weight(self) -> float | Tensor:
        if isinstance(self._bce_focal_weight, nn.Parameter):
            return 0.5 * torch.exp(-self._bce_focal_weight)
        return self._bce_focal_weight

    @property
    def dice_weight(self) -> float | Tensor:
        if isinstance(self._dice_weight, nn.Parameter):
            return 0.5 * torch.exp(-self._dice_weight)
        return self._dice_weight

    @property
    def cls_weight(self) -> float | Tensor:
        if isinstance(self._cls_weight, nn.Parameter):
            base = 0.5 * torch.exp(-self._cls_weight)
        else:
            base = self._cls_weight
        return base * getattr(self, "cls_weight_scale", 1.0)

    @property
    def aux_weight(self) -> float | Tensor:
        if isinstance(self._aux_weight, nn.Parameter):
            return 0.5 * torch.exp(-self._aux_weight)
        return self._aux_weight

    @property
    def points_weight(self) -> float | Tensor:
        if isinstance(self._points_weight, nn.Parameter):
            return 0.5 * torch.exp(-self._points_weight)
        return self._points_weight

    @property
    def inputs_weight(self) -> float | Tensor:
        if isinstance(self._inputs_weight, nn.Parameter):
            return 0.5 * torch.exp(-self._inputs_weight)
        return self._inputs_weight

    def _init_weights(self):
        # Queries: identity pass-through at init
        if hasattr(self, "queries"):
            if isinstance(self.queries, nn.Linear):
                W = self.queries.weight
                if W.shape[0] == W.shape[1]:
                    nn.init.eye_(W)
                else:
                    nn.init.xavier_uniform_(W)  # fallback if not square
                if self.queries.bias is not None:
                    nn.init.zeros_(self.queries.bias)
            elif isinstance(self.queries, nn.Sequential):
                # Make the block low-perturbation: first small, last identity
                f = self.queries[0]
                last_layer = self.queries[-1]
                if isinstance(f, nn.Linear):
                    nn.init.normal_(f.weight, mean=0.0, std=0.01)
                    if f.bias is not None:
                        nn.init.zeros_(f.bias)
                if isinstance(last_layer, nn.Linear):
                    if last_layer.weight.shape[0] == last_layer.weight.shape[1]:
                        nn.init.eye_(last_layer.weight)
                    else:
                        nn.init.xavier_uniform_(last_layer.weight)
                    if last_layer.bias is not None:
                        nn.init.zeros_(last_layer.bias)

        # FiLM: start as identity (scale=0, shift=0)
        if hasattr(self, "film"):
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

        # 2c) Points head: zero the final linear so initial mask logits ~ 0
        if hasattr(self, "points_head"):
            last_linear = self.points_head[-1] if isinstance(self.points_head, nn.Sequential) else self.points_head
            if isinstance(last_linear, nn.Linear):
                nn.init.normal_(last_linear.weight, mean=0.0, std=1e-3)
                if last_linear.bias is not None:
                    nn.init.zeros_(last_linear.bias)

        if hasattr(self, "inputs_head"):
            inputs_last_linear = (
                self.inputs_head[-1] if isinstance(self.inputs_head, nn.Sequential) else self.inputs_head
            )
            if isinstance(inputs_last_linear, nn.Linear):
                nn.init.normal_(inputs_last_linear.weight, mean=0.0, std=1e-3)
                if inputs_last_linear.bias is not None:
                    nn.init.zeros_(inputs_last_linear.bias)

        # Classification heads with priors
        #   - bias to prior log-odds
        #   - small-normal weights so bias dominates at step 0
        def init_cls_head_with_prior(head: nn.Module, prior_p: float):
            last_layer = head[-1] if isinstance(head, nn.Sequential) else head
            bias = math.log(prior_p / (1 - prior_p))
            # make all Linear weights small and biases zero first
            for mm in head.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.normal_(mm.weight, mean=0.0, std=1e-3)
                    if mm.bias is not None:
                        nn.init.zeros_(mm.bias)
            # Apply the prior to the last layer bias
            if isinstance(last_layer, nn.Linear) and last_layer.bias is not None:
                nn.init.constant_(last_layer.bias, bias)

        if hasattr(self, "cls_head"):
            p_prior = 7 / float(self.num_objs)  # p_prior depends on number of objects
            q_prior = 0.5  # quality prior ~zero
            if "obj" in self.pred_cls:
                init_cls_head_with_prior(self.cls_head, p_prior)
            else:
                init_cls_head_with_prior(self.cls_head, q_prior)

        if hasattr(self, "cls_quality_head"):
            q_prior = 0.5
            init_cls_head_with_prior(self.cls_quality_head, q_prior)

        if hasattr(self, "occ_head"):
            nn.init.normal_(self.occ_head.c_proj.weight, mean=0.0, std=1e-3)
            if self.occ_head.c_proj.bias is not None:
                nn.init.zeros_(self.occ_head.c_proj.bias)

    @property
    def aux_loss(self) -> bool:
        return self._aux_loss if self.training else False

    @staticmethod
    @torch.no_grad()
    @torch.jit.script
    def _align_masks(inputs_labels: Tensor, points_labels: Tensor) -> list[Tensor]:
        out_masks = []
        for i_lbl, p_lbl in zip(inputs_labels, points_labels, strict=False):
            # union of ids, excluding background (0)
            ids = torch.unique(torch.cat([p_lbl, i_lbl], dim=0))
            ids = ids[ids > 0]

            # rebuild with consistent row order
            if ids.numel() == 0:
                im_aligned = torch.zeros((0, i_lbl.numel()), dtype=torch.bool, device=i_lbl.device)
                pm_aligned = torch.zeros((0, p_lbl.numel()), dtype=torch.bool, device=p_lbl.device)
            else:
                im_aligned = i_lbl.unsqueeze(0) == ids.unsqueeze(1)
                pm_aligned = p_lbl.unsqueeze(0) == ids.unsqueeze(1)

            # concat along points axis
            out_masks.append(torch.cat([pm_aligned, im_aligned], dim=1))

        return out_masks

    @staticmethod
    @torch.no_grad()
    @torch.jit.script
    def _align_labels(inputs_labels: Tensor, points_labels: Tensor) -> tuple[Tensor, Tensor]:
        out_inputs, out_points = [], []
        for i_lbl, p_lbl in zip(inputs_labels, points_labels, strict=False):
            ids = torch.unique(torch.cat([p_lbl, i_lbl], dim=0))
            ids = ids[ids > 0]
            if ids.numel() == 0:
                out_inputs.append(torch.zeros_like(i_lbl))
                out_points.append(torch.zeros_like(p_lbl))
                continue

            ids_sorted, _ = torch.sort(ids)
            max_id = int(ids_sorted[-1].item())
            lut = i_lbl.new_zeros(max_id + 1)
            lut[ids_sorted] = torch.arange(1, ids_sorted.numel() + 1, device=i_lbl.device, dtype=i_lbl.dtype)
            out_inputs.append(lut[i_lbl])
            out_points.append(lut[p_lbl])

        return torch.stack(out_inputs, 0), torch.stack(out_points, 0)

    @torch.inference_mode()
    def _diagnose(
        self,
        logits: Tensor,
        scores: Tensor,
        labels: Tensor,
        score_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        apply_filter: bool = True,
        nms_iou: float = 0.5,
    ) -> dict[str, float]:
        B, _Q, _N = logits.shape
        gt_masks_list = masks_from_labels(labels)

        log_thr05 = math.log(0.5 / (1.0 - 0.5))
        covered = []
        bin_masks = (logits > log_thr05).float()
        for b in range(B):
            pm = bin_masks[b]
            gm = gt_masks_list[b].float()
            if pm.numel() == 0 or gm.numel() == 0:
                continue
            inter = pm @ gm.T
            ap = pm.sum(1, keepdim=True)
            ag = gm.sum(1, keepdim=True).T
            iou = inter / (ap + ag - inter + 1e-6)
            covered.append((iou.max(0).values >= iou_threshold).float())
        coverage_at_50 = torch.cat(covered).mean().item() if covered else 1.0

        preds_for_pq, targs_for_pq = [], []
        kept_counts, matched_vals = [], []
        with torch.inference_mode():
            for b in range(B):
                gm = gt_masks_list[b].float()
                sm = logits[b]
                sc = scores[b]
                keep = sc > score_threshold
                sm = sm[keep]
                sc = sc[keep]
                if apply_filter and sm.numel() > 0:
                    sm, sc, _ = filter_instance_masks(
                        masks=sm, scores=sc, min_size=getattr(self, "min_mask_size", 64), nms_iou=nms_iou
                    )
                kept_counts.append(float(len(sm)))
                pm = (sm > log_thr05).float()
                preds_for_pq.append(dict(masks=pm, scores=sc))
                targs_for_pq.append(dict(masks=gm))
                if pm.numel() > 0 and gm.numel() > 0:
                    inter = pm @ gm.T
                    ap = pm.sum(1, keepdim=True)
                    ag = gm.sum(1, keepdim=True).T
                    iou = inter / (ap + ag - inter + 1e-6)
                    work = iou.clone()
                    P, G = work.shape
                    neg = work.new_full((), -1.0)
                    for _ in range(min(P, G)):
                        v, idx = work.view(-1).max(0)
                        if v < iou_threshold:
                            break
                        p = idx // G
                        g = idx % G
                        matched_vals.append(float(v))
                        work[p, :] = neg
                        work[:, g] = neg

        pq_meter = PanopticQuality3D(iou_thresh=iou_threshold, track_bin_iou=False)
        pq_meter.update(preds_for_pq, targs_for_pq)
        pq_stats = pq_meter.compute()

        preds_per_image = sum(kept_counts) / max(1, len(kept_counts))
        if matched_vals:
            t = torch.tensor(matched_vals)
            q25 = t.quantile(0.25).item()
            med = t.median().item()
            q75 = t.quantile(0.75).item()
        else:
            q25 = med = q75 = 0.0

        return dict(
            coverage_at_50=coverage_at_50,
            pq=pq_stats["pq"],
            rq=pq_stats["rq"],
            sq=pq_stats["sq"],
            tp=float(pq_meter.tp.item()),
            fp=float(pq_meter.fp.item()),
            fn=float(pq_meter.fn.item()),
            preds_per_image=preds_per_image,
            matched_iou_q25=q25,
            matched_iou_median=med,
            matched_iou_q75=q75,
        )

    @torch.no_grad()
    def _update_metrics(
        self,
        logits: Tensor,
        targets: list[Tensor],
        scores: Tensor,
        threshold: float = 0.5,
        method: Literal["argmax", "cls", "greedy"] = "cls",
        nms_iou: float | None = 0.5,
        accumulator: Literal["points", "inputs"] = "points",
    ):
        """
        Update 3D MAP + PQ metrics.

        Args:
            logits: (B, Q, M) raw mask logits.
            targets: list length B of (Q, M) bool tensors.
            scores: (B, Q) instance scores (objectness / quality).
            threshold: Threshold for mask binarization.
            method: Method to get instance masks ("argmax", "cls", "greedy").
            nms_iou: IoU threshold for suppression.
        """
        if method == "argmax":
            labels = labels_from_logits(logits, threshold)
            pq_masks_list = masks_from_labels(labels, reindex=True)
        elif method == "greedy":
            pq_masks_list = masks_from_logits(
                logits=logits,
                scores=scores,
                threshold=threshold,
                apply_filter=self.apply_filter,
                min_size=self.min_mask_size,
                nms_iou=nms_iou,
            )

        preds_for_map: list[dict[str, Tensor]] = []
        preds_for_pq: list[dict[str, Tensor]] = []
        target_list: list[dict[str, Tensor]] = []

        log_threshold = math.log(threshold / (1.0 - threshold))
        for i in range(len(logits)):
            inst_masks = logits[i]
            inst_scores = scores[i]
            if self.apply_filter:
                inst_masks, inst_scores, _ = filter_instance_masks(
                    masks=inst_masks, scores=inst_scores, min_size=self.min_mask_size, nms_iou=nms_iou
                )
            inst_masks = inst_masks > log_threshold
            preds_for_map.append(dict(masks=inst_masks, scores=inst_scores))
            if method == "cls":
                preds_for_pq.append(dict(masks=inst_masks[inst_scores > self.cls_threshold]))
            else:
                preds_for_pq.append(dict(masks=pq_masks_list[i] > log_threshold))
            target_list.append(dict(masks=targets[i]))

        if self.multitask and accumulator == "inputs":
            self.map_inputs.update(preds_for_map, target_list)
            self.pq_inputs.update(preds_for_pq, target_list)
        else:
            self.map.update(preds_for_map, target_list)
            self.pq.update(preds_for_pq, target_list)

    @torch.no_grad()
    def _log_cls_metrics(
        self,
        cls_preds: Tensor,
        cls_targets: Tensor,
        cls_quality: Tensor | None = None,
        cls_quality_targets: Tensor | None = None,
        threshold: float = 0.5,
    ):
        self.log("cls/precision", M.precision(cls_preds, cls_targets, task="binary", threshold=threshold).item())
        self.log("cls/recall", M.recall(cls_preds, cls_targets, task="binary", threshold=threshold).item())
        self.log("cls/f1", M.f1_score(cls_preds, cls_targets, task="binary", threshold=threshold).item())
        self.log("cls/acc", M.accuracy(cls_preds, cls_targets, task="binary", threshold=threshold).item())
        self.log("cls/ece", M.calibration_error(cls_preds, cls_targets, task="binary").item())
        self.log("cls/brier", (cls_preds - cls_targets.float()).pow(2).mean().item())
        # probability to rank a randomly chosen positive sample higher than a randomly chosen negative one
        cls_auroc = cast(Tensor, M.auroc(cls_preds, cls_targets, task="binary"))
        self.log("cls/auroc", cls_auroc.item())
        # prioritize true positive samples over false positive ones
        cls_auprc = cast(Tensor, M.average_precision(cls_preds, cls_targets, task="binary"))
        self.log("cls/auprc", cls_auprc.item())

        if cls_quality is not None:
            scores = cls_preds * cls_quality
            self.log(
                "cls_quality/precision", M.precision(scores, cls_targets, task="binary", threshold=threshold).item()
            )
            self.log("cls_quality/recall", M.recall(scores, cls_targets, task="binary", threshold=threshold).item())
            self.log("cls_quality/f1", M.f1_score(scores, cls_targets, task="binary", threshold=threshold).item())
            self.log("cls_quality/acc", M.accuracy(scores, cls_targets, task="binary", threshold=threshold).item())
            self.log("cls_quality/ece", M.calibration_error(scores, cls_targets, task="binary").item())
            self.log("cls_quality/brier", (scores - cls_targets.float()).pow(2).mean().item())
            qual_auroc = cast(Tensor, M.auroc(scores, cls_targets, task="binary"))
            qual_auprc = cast(Tensor, M.average_precision(scores, cls_targets, task="binary"))
            self.log("cls_quality/auroc", qual_auroc.item())
            self.log("cls_quality/auprc", qual_auprc.item())
            if cls_quality_targets is not None:
                pos = cls_targets > 0
                if pos.any():
                    q_pred = cls_quality[pos]
                    q_tgt = cls_quality_targets[pos]
                    self.log("cls_quality/mae_pos", F.l1_loss(q_pred, q_tgt).item())
                    self.log("cls_quality/mse_pos", F.mse_loss(q_pred, q_tgt).item())
                    # ranking: higher-quality masks picked over lower-quality ones
                    self.log("cls_quality/spearman_pos", M.spearman_corrcoef(q_pred, q_tgt).item())
                    # linear correlation between predicted and target quality?
                    if q_pred.std(unbiased=False) > 1e-6 and q_tgt.std(unbiased=False) > 1e-6:
                        self.log("cls_quality/pearson_pos", M.pearson_corrcoef(q_pred.double(), q_tgt.double()).item())

    @torch.no_grad()
    def _log_query_metrics(self, queries: Tensor, queries_logits: Tensor | None = None):
        if queries_logits is not None:
            entropy = F.binary_cross_entropy_with_logits(queries_logits, queries_logits.sigmoid())
            self.log("query/entropy", entropy.item())
            self.log("query/mean_score", queries_logits.sigmoid().mean().item())

        sel_norm = F.normalize(queries, dim=-1)
        sim = sel_norm @ sel_norm.transpose(1, 2)
        b, q, _ = sim.shape
        diversity = torch.tensor(0.0, device=sim.device)
        if q > 1:
            mask = ~torch.eye(q, dtype=torch.bool, device=sim.device)
            off = sim[:, mask].view(b, q, q - 1).flatten()
            diversity = (1 - off).mean()

        self.log("query/diversity", diversity.item())

    def _forward_heads(self, x: Tensor, points_feat: Tensor, inputs_feat: Tensor | None = None) -> dict[str, Any]:
        x = self.head_norm(x)
        q = self.points_head(x)
        p = points_feat
        if hasattr(self, "occ_head"):
            B, Q, D = q.shape
            B2, P, D2 = p.shape
            assert B == B2 and D == D2, f"Shape mismatch: q {q.shape}, p {p.shape}"
            q_exp = q.unsqueeze(2).expand(B, Q, P, D)
            p_exp = p.unsqueeze(1).expand(B, Q, P, D)
            logits = self.occ_head(torch.cat([q_exp, p_exp], dim=-1).reshape(B * Q * P, D * 2))
            logits = logits.reshape(B, Q, P, -1).squeeze(-1)
        else:
            if self.cos_sim:
                q = F.normalize(q, dim=-1)
                p = F.normalize(p, dim=-1)
            logits = q @ p.transpose(1, 2)
        if hasattr(self, "logit_scale"):
            logits = logits * self.logit_scale.exp().clamp(0.1, 10.0)
        out = dict(logits=logits)
        if self.multitask:
            if inputs_feat is None:
                raise ValueError("inputs_feat is required in multitask mode.")
            head_in = getattr(self, "inputs_head", self.points_head)
            q_in = head_in(x)
            p_in = inputs_feat
            if self.cos_sim:
                q_in = F.normalize(q_in, dim=-1)
                p_in = F.normalize(p_in, dim=-1)
            in_logits = q_in @ p_in.transpose(1, 2)
            if hasattr(self, "logit_scale"):
                in_logits = in_logits * self.logit_scale.exp().clamp(0.1, 10.0)
            out["inputs.logits"] = in_logits
        if hasattr(self, "cls_head"):
            if self.detach_cls:
                x = x.detach()
            out["cls_logits"] = self.cls_head(x).squeeze(2)
            if hasattr(self, "cls_quality_head"):
                out["cls_quality"] = self.cls_quality_head(x).squeeze(2)
        return out

    @staticmethod
    def _local_depth_descriptor_maps(depth: Tensor) -> Tensor:
        """Return boundary, curvature, and neighborhood-confidence maps from depth."""
        if depth.ndim == 4 and depth.size(1) == 1:
            depth = depth[:, 0]
        if depth.ndim != 3:
            raise ValueError(f"Expected depth with shape (B, H, W) or (B, 1, H, W), got {depth.shape}")

        z = depth.float().unsqueeze(1)
        valid = z > 0
        padded = F.pad(z, (1, 1, 1, 1), mode="replicate")
        neighbors = torch.cat(
            (
                padded[:, :, :-2, 1:-1],
                padded[:, :, 2:, 1:-1],
                padded[:, :, 1:-1, :-2],
                padded[:, :, 1:-1, 2:],
            ),
            dim=1,
        )
        neighbor_valid = neighbors > 0
        pair_valid = neighbor_valid & valid
        denominator = torch.maximum(neighbors, z).clamp_min(1e-6)
        relative_jump = torch.where(pair_valid, (neighbors - z).abs() / denominator, 0.0)
        boundary = relative_jump.amax(dim=1, keepdim=True).clamp_max(2.0)

        count = neighbor_valid.sum(dim=1, keepdim=True)
        neighbor_mean = torch.where(neighbor_valid, neighbors, 0.0).sum(dim=1, keepdim=True)
        neighbor_mean = neighbor_mean / count.clamp_min(1)
        curvature = torch.where(
            valid & (count > 0),
            (neighbor_mean - z).abs() / z.clamp_min(1e-6),
            0.0,
        ).clamp_max(2.0)
        confidence = (count.float() / 4.0) * valid.float()
        return torch.cat((boundary, curvature, confidence), dim=1)

    def _point_depth_descriptors(self, inputs: Tensor, kwargs: dict[str, Any]) -> Tensor:
        depth = kwargs.get("inputs.depth")
        pixel_coords = kwargs.get("inputs.pixel_coords")
        if not isinstance(depth, Tensor) or not isinstance(pixel_coords, Tensor):
            raise ValueError("depth_descriptor='local' requires tensor inputs.depth and inputs.pixel_coords")
        if pixel_coords.ndim != 3 or pixel_coords.size(0) != inputs.size(0) or pixel_coords.size(1) != inputs.size(1):
            raise ValueError(
                "inputs.pixel_coords must have shape (B, N, 2) aligned with inputs; "
                f"got inputs={inputs.shape}, pixel_coords={pixel_coords.shape}"
            )

        maps = self._local_depth_descriptor_maps(depth)
        height, width = maps.shape[-2:]
        rows = pixel_coords[..., 0].long().clamp(0, height - 1)
        cols = pixel_coords[..., 1].long().clamp(0, width - 1)
        batch = torch.arange(inputs.size(0), device=inputs.device).unsqueeze(1)
        descriptors = maps.permute(0, 2, 3, 1)[batch, rows, cols]
        return descriptors.to(device=inputs.device, dtype=inputs.dtype)

    @staticmethod
    def _voxel_neighbor_counts(cells: Tensor, anchor_cells: Tensor, resolution: int) -> Tensor:
        """Count points in the anchor cell and its six face-adjacent cells."""
        batch_size, num_points, _ = cells.shape
        num_anchors = anchor_cells.size(1)
        resolution_2 = resolution * resolution
        resolution_3 = resolution_2 * resolution
        batch_offsets = torch.arange(batch_size, device=cells.device).view(batch_size, 1) * resolution_3
        linear = cells[..., 0] * resolution_2 + cells[..., 1] * resolution + cells[..., 2]
        counts = torch.bincount(
            (linear + batch_offsets).reshape(-1),
            minlength=batch_size * resolution_3,
        ).float()

        offsets = anchor_cells.new_tensor(
            ((0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
        )
        neighbor_cells = anchor_cells.unsqueeze(2) + offsets.view(1, 1, 7, 3)
        valid = ((neighbor_cells >= 0) & (neighbor_cells < resolution)).all(dim=-1)
        neighbor_cells = neighbor_cells.clamp(0, resolution - 1)
        neighbor_linear = (
            neighbor_cells[..., 0] * resolution_2 + neighbor_cells[..., 1] * resolution + neighbor_cells[..., 2]
        )
        neighbor_linear = neighbor_linear + batch_offsets.view(batch_size, 1, 1)
        gathered = counts[neighbor_linear.reshape(-1)].view(batch_size, num_anchors, 7)
        gathered = torch.where(valid, gathered, 0.0)
        normalizer = math.log1p(max(num_points, 1))
        return torch.log1p(gathered) / normalizer

    def _voxel_anchor_residual(self, inputs: Tensor, fps_idx: Tensor) -> Tensor:
        mins = inputs.amin(dim=1, keepdim=True)
        spans = (inputs.amax(dim=1, keepdim=True) - mins).clamp_min(1e-6)
        normalized = ((inputs - mins) / spans).clamp(0.0, 1.0)
        anchor_idx = fps_idx.unsqueeze(-1).expand(-1, -1, 3)

        identity = inputs.new_zeros((inputs.size(0), fps_idx.size(1), self.n_embd))
        local_features: list[Tensor] = []
        for resolution, axis_embeddings in zip(self.voxel_anchor_resolutions, self.voxel_axis_embeddings, strict=True):
            axis_embeddings = cast(nn.ModuleList, axis_embeddings)
            scaled = normalized * resolution
            cells = scaled.floor().long().clamp(0, resolution - 1)
            anchor_cells = torch.gather(cells, dim=1, index=anchor_idx)
            anchor_scaled = torch.gather(scaled, dim=1, index=anchor_idx)
            offsets = anchor_scaled - anchor_cells.to(anchor_scaled.dtype) - 0.5
            counts = self._voxel_neighbor_counts(cells, anchor_cells, resolution).to(inputs.dtype)
            local_features.extend((offsets, counts))
            for axis, embedding in enumerate(axis_embeddings):
                identity = identity + embedding(anchor_cells[..., axis])

        local = self.voxel_local_mlp(torch.cat(local_features, dim=-1))
        return self.voxel_output(identity + local)

    @staticmethod
    def _soft_support_quantiles(
        points: Tensor, logits: Tensor, quantiles: tuple[float, float], temperature: float
    ) -> Tensor:
        sorted_points, order = points.sort(dim=1)
        probabilities = logits.sigmoid().square().unsqueeze(-1).expand_as(points)
        sorted_probabilities = torch.gather(probabilities, dim=1, index=order)
        total = sorted_probabilities.sum(dim=1, keepdim=True).clamp_min(1e-8)
        cdf_high = sorted_probabilities.cumsum(dim=1) / total
        cdf_low = cdf_high - sorted_probabilities / total

        outputs = []
        for quantile in quantiles:
            log_membership = F.logsigmoid((quantile - cdf_low) / temperature)
            log_membership = log_membership + F.logsigmoid((cdf_high - quantile) / temperature)
            weights = log_membership.softmax(dim=1)
            outputs.append((weights * sorted_points).sum(dim=1))
        return torch.stack(outputs, dim=1)

    def _extent_loss(
        self,
        points: Tensor,
        logits: Tensor,
        targets: Tensor,
        batch_idx: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if logits.numel() == 0:
            zero = logits.sum()
            return zero, zero.detach(), zero.detach()

        matched_points = points[batch_idx]
        target_mask = targets > 0.5
        valid = target_mask.sum(dim=1) >= 2
        if not valid.any():
            zero = logits.sum() * 0.0
            return zero, zero.detach(), zero.detach()

        matched_points = matched_points[valid]
        logits = logits[valid]
        target_mask = target_mask[valid]
        support = self._soft_support_quantiles(
            matched_points,
            logits,
            quantiles=(0.05, 0.95),
            temperature=self.extent_quantile_temperature,
        )
        pred_min, pred_max = support.unbind(dim=1)

        inf = torch.full_like(matched_points, torch.inf)
        gt_min = torch.where(target_mask.unsqueeze(-1), matched_points, inf).amin(dim=1)
        gt_max = torch.where(target_mask.unsqueeze(-1), matched_points, -inf).amax(dim=1)

        pred_center = (pred_min + pred_max) / 2
        gt_center = (gt_min + gt_max) / 2
        pred_extent = ((pred_max - pred_min) / 2).clamp_min(1e-4)
        gt_extent = ((gt_max - gt_min) / 2).clamp_min(1e-4)
        center_error = F.l1_loss(pred_center, gt_center)
        log_extent_error = F.l1_loss(pred_extent.log(), gt_extent.log())
        return center_error + log_extent_error, center_error.detach(), log_extent_error.detach()

    @staticmethod
    def _matched_completion_targets(match_masks: Any, match: list[tuple[Tensor, Tensor]], multitask: bool) -> Tensor:
        point_masks = match_masks[0] if multitask else match_masks
        matched = [point_masks[i][target_idx] for i, (_, target_idx) in enumerate(match) if target_idx.numel()]
        if matched:
            return torch.cat(matched, dim=0)
        width = point_masks[0].size(1) if point_masks else 0
        return torch.empty((0, width), dtype=torch.bool, device=point_masks[0].device if point_masks else None)

    def forward(self, inputs: Tensor, points: Tensor | None = None, **kwargs) -> dict[str, Any]:
        feature = self.encode(inputs=inputs, **kwargs)
        _points = inputs if points is None else points
        return self.decode(points=_points, feature=feature, inputs=inputs, **kwargs)

    def encode(self, inputs: Tensor, **kwargs) -> Tensor:
        need_indices = self.depth_descriptor is not None or bool(self.voxel_anchor_resolutions)
        fps_idx = None
        if need_indices:
            fps_idx = cast(
                Tensor,
                furthest_point_sample(inputs, num_samples=self.num_queries, return_indices=True),
            )
            gather_idx = fps_idx.unsqueeze(-1).expand(-1, -1, inputs.size(-1))
            inputs_fps = torch.gather(inputs, dim=1, index=gather_idx)
        else:
            inputs_fps = cast(Tensor, furthest_point_sample(inputs, num_samples=self.num_queries))

        memory = self.nerf_enc(inputs)
        anchors = self.nerf_enc(inputs_fps)
        if self.depth_descriptor == "local":
            if fps_idx is None:
                raise RuntimeError("Depth descriptors require FPS indices")
            descriptor_residual = self.depth_descriptor_mlp(self._point_depth_descriptors(inputs, kwargs))
            descriptor_idx = fps_idx.unsqueeze(-1).expand(-1, -1, descriptor_residual.size(-1))
            memory = memory + descriptor_residual
            anchors = anchors + torch.gather(descriptor_residual, dim=1, index=descriptor_idx)
            self.log(
                "depth_descriptor/residual_norm",
                cast(Any, descriptor_residual.detach().norm(dim=-1).mean()),
            )
        if self.voxel_anchor_resolutions:
            if fps_idx is None:
                raise RuntimeError("Voxel anchor residuals require FPS indices")
            voxel_residual = self._voxel_anchor_residual(inputs, fps_idx)
            anchors = anchors + voxel_residual
            self.log("voxel_anchor/residual_norm", cast(Any, voxel_residual.detach().norm(dim=-1).mean()))

        x = self.inputs_enc(anchors, memory)
        x = torch.cat((self.encoder.cls_token.expand(len(x), -1, -1), x), dim=1)
        feat = self.encoder(x)
        if hasattr(self, "lvl_embd"):
            feat = [
                (self.lvl_norm(f + e), self.lvl_norm(c + e))
                for (f, c), e in zip(feat, self.lvl_embd.weight, strict=False)
            ]
        patch_feat = feat[-1][0]
        if self.cat_feat:
            patch_feat = torch.cat([f[0] for f in feat], dim=1)
        if self.cls_token:
            return torch.cat((feat[-1][1].unsqueeze(1), patch_feat), dim=1)
        return patch_feat

    def _split_completion_feature(self, feature: Tensor) -> tuple[Tensor, Tensor | None]:
        patch_feat = feature
        cls_feat = None
        if self.cls_token:
            patch_feat = feature[:, 1:]
            cls_feat = feature[:, :1]
        return patch_feat, cls_feat

    def _decode_completion_point_features(
        self,
        points: Tensor,
        patch_feat: Tensor,
        cls_feat: Tensor | None = None,
    ) -> Tensor:
        x = self.nerf_enc(points)
        if self.cat_feat and isinstance(self.points_dec, nn.ModuleList):
            nq = self.num_queries
            feat_list = [patch_feat[:, i * nq : (i + 1) * nq] for i in range(self.num_feat_layers)]
            for layer_idx in range(self.num_feat_layers):
                x = self.points_dec[layer_idx](x, feat_list[layer_idx])
        else:
            x = self.points_dec(x, patch_feat)

        if self.cls_token and hasattr(self, "film"):
            if cls_feat is None:
                raise ValueError("cls_feat is required for FiLM conditioning.")
            film = self.film(cls_feat)
            scale, shift = film.chunk(2, dim=-1)
            x = x * (1 + scale.tanh()) + shift
        return x

    def _decode_completion_queries(
        self,
        feature: Tensor,
        global_step: int | None = None,
        total_steps: int | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        if self.cat_point_feat:
            raise NotImplementedError("Streaming completion does not support cat_point_feat=True")

        patch_feat, _cls_feat = self._split_completion_feature(feature)
        query_pos = self.query_pos.weight.unsqueeze(0).expand(len(feature), -1, -1)
        x = torch.zeros_like(query_pos)

        if self.queries_from_feat:
            queries = self.queries(patch_feat)
            scores = None
            if hasattr(self, "cls_head"):
                z = self.head_norm(queries)
                scores = self.cls_head(z).squeeze(2)
                if hasattr(self, "cls_quality_head"):
                    quality = self.cls_quality_head(z).squeeze(2)
                    scores = F.logsigmoid(scores) + F.logsigmoid(quality)

            select = None
            if "max" in self.queries_from_feat:
                select = "max"
            elif "mean" in self.queries_from_feat:
                select = "mean"
            elif "norm" in self.queries_from_feat:
                select = "norm"
            elif "fps" in self.queries_from_feat:
                select = "fps"

            gather = "hard"
            if "soft" in self.queries_from_feat:
                gather = "soft"
            elif "gumbel" in self.queries_from_feat:
                gather = "gumbel"

            ste = "ste" in self.queries_from_feat
            tau = 0.5
            beta = 0.5 if ste else 0.75
            if "anneal" in self.queries_from_feat and global_step is not None and total_steps is not None:
                tau = cosine_anneal(1.0, 0.3 if ste else 0.2, int(0.5 * total_steps), global_step)
                beta = 1.0 if ste else 1.5
            x = queries_from_feat(
                queries=queries,
                scores=scores,
                num_queries=self.num_objs,
                select=select,
                gather=gather,
                ste=ste,
                tau=tau,
                beta=beta / tau,
            )
            if "detach" in self.queries_from_feat:
                x = x.detach()

        kv = feature
        for layer in cast(Any, self.query_enc).layers:
            layer = cast(Any, layer)
            qk = v = layer.ln_1(x)
            q = k = qk + query_pos
            x = x + layer.dp_1(layer.self_attn(q, k, v))
            x = x + layer.dp_2(layer.cross_attn(layer.ln_2(x) + query_pos, kv, kv))
            x = x + layer.dp_3(layer.projection(layer.ln_3(x)))

        z = self.head_norm(x)
        q = self.points_head(z)
        cls_logits = self.cls_head(z).squeeze(2) if hasattr(self, "cls_head") else None
        cls_quality = self.cls_quality_head(z).squeeze(2) if hasattr(self, "cls_quality_head") else None
        return q, cls_logits, cls_quality

    def decode(
        self,
        points: Tensor,
        feature: Tensor,
        inputs: Tensor | None = None,
        inputs_aux_feat: Tensor | None = None,
        inputs_feat_fusion: nn.Module | None = None,
        query_aux_feat: Tensor | None = None,
        query_feat_fusion: nn.Module | None = None,
        points_batch_size: int | None = None,
        global_step: int | None = None,
        total_steps: int | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        x = self.nerf_enc(points)
        patch_feat = feature
        cls_feat = None
        if self.cls_token:
            patch_feat = feature[:, 1:]
            cls_feat = feature[:, :1]

        if self.cat_feat and isinstance(self.points_dec, nn.ModuleList):
            nq = self.num_queries
            feat_list = [patch_feat[:, i * nq : (i + 1) * nq] for i in range(self.num_feat_layers)]
            for layer_idx in range(self.num_feat_layers):
                x = self.points_dec[layer_idx](x, feat_list[layer_idx])
        else:
            if points_batch_size is None or points.size(1) <= points_batch_size:
                x = self.points_dec(x, patch_feat)
            else:
                splits = torch.split(x, points_batch_size, dim=1)
                x = torch.cat([self.points_dec(s, patch_feat) for s in splits], dim=1)

        if self.cls_token and hasattr(self, "film"):
            if cls_feat is None:
                raise ValueError("cls_feat is required for FiLM conditioning.")
            film = self.film(cls_feat)
            scale, shift = film.chunk(2, dim=-1)
            x = x * (1 + scale.tanh()) + shift
        points_feat = x

        inputs_feat = None
        inputs_idx = None
        if self.multitask:
            if inputs is None:
                raise ValueError("inputs is required in multitask mode.")
            sample_cfg = cast(Any, self).sample
            if self.training and sample_cfg and sample_cfg[1]:
                k = int(3 * sample_cfg[1])
                N = inputs.size(1)
                k = min(k, N)
                if k < N:
                    idx = torch.multinomial(torch.ones(len(inputs), N, device=inputs.device), k, replacement=False)
                    inputs = torch.gather(inputs, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, inputs.size(-1)))
                    if inputs_aux_feat is not None:
                        inputs_aux_feat = torch.gather(
                            inputs_aux_feat,
                            dim=1,
                            index=idx.unsqueeze(-1).expand(-1, -1, inputs_aux_feat.size(-1)),
                        )
                    inputs_idx = idx

            x_in = self.nerf_enc(inputs)
            if points_batch_size is None or inputs.size(1) <= points_batch_size:
                dec = getattr(self, "inputs_dec", self.points_dec)
                x_in = dec(x_in, patch_feat)
            else:
                splits = torch.split(x_in, points_batch_size, dim=1)
                dec = getattr(self, "inputs_dec", self.points_dec)
                x_in = torch.cat([dec(s, patch_feat) for s in splits], dim=1)
            inputs_feat = x_in
            if inputs_feat_fusion is not None:
                if inputs_aux_feat is None:
                    raise ValueError("inputs_aux_feat is required when inputs_feat_fusion is set.")
                inputs_feat = inputs_feat_fusion(inputs_feat, inputs_aux_feat)

        query_pos = self.query_pos.weight.unsqueeze(0).expand(len(feature), -1, -1)
        x = torch.zeros_like(query_pos)

        aux_out: list[dict[str, Tensor]] = []
        if self.queries_from_feat:
            queries = self.queries(patch_feat)
            scores = None
            if hasattr(self, "cls_head"):
                z = self.head_norm(queries)
                scores = self.cls_head(z).squeeze(2)
                if hasattr(self, "cls_quality_head"):
                    quality = self.cls_quality_head(z).squeeze(2)
                    scores = F.logsigmoid(scores) + F.logsigmoid(quality)

            select = None
            if "max" in self.queries_from_feat:
                select = "max"
            elif "mean" in self.queries_from_feat:
                select = "mean"
            elif "norm" in self.queries_from_feat:
                select = "norm"
            elif "fps" in self.queries_from_feat:
                select = "fps"

            gather = "hard"
            if "soft" in self.queries_from_feat:
                gather = "soft"
            elif "gumbel" in self.queries_from_feat:
                gather = "gumbel"

            ste = "ste" in self.queries_from_feat
            tau = 0.5
            beta = 0.5 if ste else 0.75
            if "anneal" in self.queries_from_feat and global_step is not None and total_steps is not None:
                tau = cosine_anneal(1.0, 0.3 if ste else 0.2, int(0.5 * total_steps), global_step)
                beta = 1.0 if ste else 1.5
            x = queries_from_feat(
                queries=queries,
                scores=scores,
                num_queries=self.num_objs,
                select=select,
                gather=gather,
                ste=ste,
                tau=tau,
                beta=beta / tau,
            )
            if self.aux_loss:
                aux_out.append(self._forward_heads(x, points_feat, inputs_feat))
                if "detach" in self.queries_from_feat:
                    x = x.detach()
            queries = x

        kv = feature
        if self.cat_point_feat:
            kv = torch.cat([feature, points_feat], dim=1)
        for i, layer in enumerate(cast(Any, self.query_enc).layers):
            layer = cast(Any, layer)
            qk = v = layer.ln_1(x)
            q = k = qk + query_pos
            x = x + layer.dp_1(layer.self_attn(q, k, v))
            x = x + layer.dp_2(layer.cross_attn(layer.ln_2(x) + query_pos, kv, kv))
            x = x + layer.dp_3(layer.projection(layer.ln_3(x)))
            if self.aux_loss and i < len(self.query_enc.layers) - 1:
                aux_out.append(self._forward_heads(x, points_feat, inputs_feat))

        if query_feat_fusion is not None:
            if query_aux_feat is None:
                raise ValueError("query_aux_feat is required when query_feat_fusion is set.")
            x = query_feat_fusion(x, query_aux_feat)

        out = self._forward_heads(x, points_feat, inputs_feat)
        if aux_out:
            out["aux_out"] = aux_out
        if self.queries_from_feat:
            out["queries"] = queries
        if inputs_idx is not None:
            out["inputs.index"] = inputs_idx
        return out

    @torch.no_grad()
    def _align_predictions_to_gt(
        self,
        logits: Tensor,
        cls_logits: Tensor | None,
        cls_quality: Tensor | None,
        labels_or_masks: Tensor | list[Tensor],
        return_logits: bool = False,
    ) -> tuple[list[Tensor], dict[str, list[Tensor]] | None]:
        """
        Reorder and pad predictions to the GT mask order using Hungarian matching.
        - logits: (B, Kmax, M_total) padded predictions to match against GT.
        - cls_logits/cls_quality: (B, Kmax) classification tensors aligned to logits rows (optional).
        - labels: either a labels Tensor (points-only) or a prebuilt list of GT masks (K_gt, M_total) per batch.
        Returns:
          - masks_per_batch: list len B of (K_gt, M_total) tensors (aligned logits).
          - out_logits: optional dict with aligned per-batch "logits" and cls fields if available.
        """
        if isinstance(labels_or_masks, list):
            gt_masks = labels_or_masks
        else:
            gt_masks = masks_from_labels(labels_or_masks)

        scores_for_match = self._get_scores(logits, cls_logits, cls_quality)
        match = self._match_and_gather(
            logits_for_match=logits,
            masks_for_match=gt_masks,
            scores=scores_for_match,
        )[0]

        batch_size, _, num_total = logits.size()
        neg_fill = -1e4
        masks_per_batch: list[Tensor] = []
        out_logits_dict: dict[str, list[Tensor]] | None = {"logits": []} if return_logits else None

        for bs in range(batch_size):
            K_gt = gt_masks[bs].size(0)
            aligned = logits.new_full((K_gt, num_total), neg_fill)
            src_idx_b, tgt_idx_b = match[bs]
            if src_idx_b.numel() > 0:
                aligned[tgt_idx_b] = logits[bs, src_idx_b]
            masks_per_batch.append(aligned)

            if return_logits and out_logits_dict is not None:
                out_logits_dict["logits"].append(aligned)
                if cls_logits is not None:
                    aligned_cls = logits.new_full((K_gt,), neg_fill)
                    if src_idx_b.numel() > 0:
                        aligned_cls[tgt_idx_b] = cls_logits[bs, src_idx_b]
                    out_logits_dict.setdefault("cls_logits", []).append(aligned_cls)
                if cls_quality is not None:
                    aligned_qual = logits.new_full((K_gt,), neg_fill)
                    if src_idx_b.numel() > 0:
                        aligned_qual[tgt_idx_b] = cls_quality[bs, src_idx_b]
                    out_logits_dict.setdefault("cls_quality", []).append(aligned_qual)

        return masks_per_batch, out_logits_dict

    def stream_instance_logits(
        self,
        inputs: Tensor | None = None,
        points: Tensor | None = None,
        feature: Tensor | None = None,
        data: dict[str, Any] | None = None,
        threshold: float = 0.5,
        method: Literal["cls"] | None = None,
        points_batch_size: int | None = None,
        align_to_gt: bool = False,
        enable_grad: bool = False,
        **kwargs,
    ):
        with torch.set_grad_enabled(enable_grad):
            if align_to_gt:
                raise NotImplementedError("Streaming completion does not support align_to_gt=True")
            if self.apply_filter:
                raise NotImplementedError("Streaming completion requires apply_filter=False")
            if hasattr(self, "occ_head"):
                raise NotImplementedError("Streaming completion does not support mlp_occ_head=True")
            if method not in (None, "cls"):
                raise NotImplementedError(f"Streaming completion only supports method='cls', got {method!r}")
            if points is None:
                if inputs is None:
                    raise ValueError("points or inputs is required for streaming completion.")
                points = inputs
            if points.ndim != 3 or points.size(0) != 1:
                raise NotImplementedError("Streaming completion currently supports batch size 1")
            if feature is None:
                if inputs is None:
                    raise ValueError("inputs is required when feature is not provided.")
                feature = self.encode(inputs=inputs, **kwargs)

            q, cls_logits, cls_quality = self._decode_completion_queries(
                feature=feature,
                global_step=kwargs.get("global_step"),
                total_steps=kwargs.get("total_steps"),
            )
            if cls_logits is None:
                raise NotImplementedError("Streaming completion requires a classification head for query selection")
            scores = self._get_scores(q.new_empty((*q.shape[:2], 0)), cls_logits, cls_quality)
            keep = scores[0] > self.cls_threshold
            instance_indices = torch.nonzero(keep, as_tuple=False).flatten()
            yield {"instance_indices": instance_indices.detach().cpu().tolist(), "num_points": int(points.size(1))}
            if instance_indices.numel() == 0:
                return

            q_keep = q[:, instance_indices]
            patch_feat, cls_feat = self._split_completion_feature(feature)
            points_batch_size = int(points_batch_size or points.size(1))
            for start in range(0, points.size(1), points_batch_size):
                end = min(start + points_batch_size, points.size(1))
                points_feat = self._decode_completion_point_features(points[:, start:end], patch_feat, cls_feat)
                q_chunk = q_keep
                p_chunk = points_feat
                if self.cos_sim:
                    q_chunk = F.normalize(q_chunk, dim=-1)
                    p_chunk = F.normalize(p_chunk, dim=-1)
                logits = q_chunk @ p_chunk.transpose(1, 2)
                if hasattr(self, "logit_scale"):
                    logits = logits * self.logit_scale.exp().clamp(0.1, 10.0)
                yield {
                    "start": start,
                    "end": end,
                    "instance_indices": instance_indices.detach().cpu().tolist(),
                    "logits": logits[0] if enable_grad else logits[0].detach(),
                }

    @torch.inference_mode()
    def predict(
        self,
        inputs: Tensor | None = None,
        points: Tensor | None = None,
        feature: Tensor | list[Tensor] | dict[str, Tensor] | None = None,
        data: dict[str, Any] | None = None,
        threshold: float = 0.5,
        method: Literal["argmax", "cls", "greedy"] | None = None,
        min_num_points: int | None = None,
        nms_iou: float | None = 0.5,
        return_logits: bool = False,
        align_to_gt: bool = False,
        show: bool = False,
        **kwargs,
    ) -> list[Tensor] | tuple[list[Tensor], dict[str, list[Tensor]]]:
        """Produce instance masks from raw model outputs.

        Args:
            inputs: (B, N, 3+C) point cloud
            threshold:
                - For method='argmax': probability threshold for background (converted to logit).
                - For method='cls': probability threshold on per-query scores.
            method: 'argmax' (exclusive labeling) or 'cls' (score-thresholded queries).
            min_num_points: Minimum foreground points per predicted mask (only for method='cls'; 0 disables).
            nms_iou: IoU threshold for duplicate suppression (only if nms=True).
            align_to_gt: If True and GT labels are present in `data`, return predictions
                         reordered to match GT mask order and padded to GT count.
            **kwargs: Forward extras.

        Returns:
            List (len = batch size) of (K, N/M) tensors of mask logits.
            K = num GT if align_to_gt is True and GT present; else varies by method.
        """
        out: dict[str, Any]
        if data is None or not any("logits" in k for k in data.keys()):
            if feature is None:
                if inputs is None:
                    raise ValueError("inputs is required when precomputed outputs are not provided.")
                out = cast(dict[str, Any], self(inputs=inputs, points=points, **kwargs))
            else:
                points_in = points if points is not None else inputs
                if points_in is None:
                    raise ValueError("Either points or inputs is required for decode().")
                out = cast(
                    dict[str, Any],
                    self.decode(points=points_in, feature=cast(Tensor, feature), inputs=inputs, **kwargs),
                )
        else:
            out = data

        logits = cast(Tensor, out["logits"])
        in_logits = cast(Tensor | None, out.get("inputs.logits"))
        cls_logits = cast(Tensor | None, out.get("cls_logits"))
        cls_quality = cast(Tensor | None, out.get("cls_quality"))

        min_num_points = min_num_points or self.min_mask_size
        use_cat = in_logits is not None
        num_points = logits.size(2)
        cat_logits = torch.cat([logits, in_logits], dim=2) if use_cat else logits
        scores = self._get_scores(cat_logits, cls_logits, cls_quality)

        if method is None:
            method = "greedy" if cls_logits is None else "cls"

        masks_per_batch: list[Tensor] = []
        cls_logits_per_batch: list[Tensor] = []
        cls_quality_per_batch: list[Tensor] = []

        if method == "argmax":
            labels = labels_from_logits(cat_logits, threshold)
            masks_per_batch = masks_from_labels(labels, reindex=True)

        elif method == "greedy":
            masks_per_batch = masks_from_logits(
                logits=cat_logits,
                scores=scores,
                threshold=threshold,
                apply_filter=self.apply_filter,
                min_size=min_num_points,
                nms_iou=nms_iou,
            )

        elif method == "cls":
            for i in range(len(cat_logits)):
                inst_masks_cat = cat_logits[i]
                inst_scores = scores[i]

                keep = inst_scores > self.cls_threshold
                inst_masks_cat = inst_masks_cat[keep]
                inst_scores = inst_scores[keep]

                if self.apply_filter and keep.any():
                    inst_masks_cat, inst_scores, idx = filter_instance_masks(
                        masks=inst_masks_cat,
                        scores=inst_scores,
                        min_size=min_num_points,
                        max_num=128**3,
                        nms_iou=nms_iou,
                    )
                    if cls_logits is not None:
                        cls_logits_per_batch.append(cls_logits[i][keep][idx])
                    if cls_quality is not None:
                        cls_quality_per_batch.append(cls_quality[i][keep][idx])
                else:
                    if cls_logits is not None:
                        cls_logits_per_batch.append(cls_logits[i][keep])
                    if cls_quality is not None:
                        cls_quality_per_batch.append(cls_quality[i][keep])

                masks_per_batch.append(inst_masks_cat)
        else:
            raise ValueError(f"Unknown method: {method}")

        aligned_out_logits = None
        if align_to_gt and data is not None:
            if use_cat and ("inputs.labels" in data) and ("points.labels" in data):
                gt_combined = self._align_masks(
                    cast(Tensor, data["inputs.labels"]), cast(Tensor, data["points.labels"])
                )
                if in_logits is None:
                    raise ValueError("inputs.logits is required for aligned multitask predictions.")
                num_total = num_points + in_logits.size(2)
            elif "points.labels" in data:
                gt_combined = masks_from_labels(cast(Tensor, data["points.labels"]))
                num_total = num_points
            else:
                gt_combined = None

            if gt_combined is not None:
                # 1) build coalesced, padded predictions from (post-filter) masks_per_batch
                num_max = max(1, max((m.size(0) for m in masks_per_batch), default=1))
                cat_pred = logits.new_full((len(masks_per_batch), num_max, num_total), -1e4)
                for i, m in enumerate(masks_per_batch):
                    if m.numel() > 0:
                        cat_pred[i, : m.size(0)] = m

                # 2) build padded cls tensors if available (only for method='cls')
                if method == "cls" and cls_logits_per_batch:
                    cls_pad = cat_pred.new_full((len(cls_logits_per_batch), num_max), -1e4)
                    for i, c in enumerate(cls_logits_per_batch):
                        if c.numel() > 0:
                            cls_pad[i, : c.size(0)] = c
                    if cls_quality_per_batch:
                        qual_pad = cat_pred.new_full((len(cls_quality_per_batch), num_max), -1e4)
                        for i, q in enumerate(cls_quality_per_batch):
                            if q.numel() > 0:
                                qual_pad[i, : q.size(0)] = q
                    else:
                        qual_pad = None
                else:
                    cls_pad = None
                    qual_pad = None

                # 3) delegate Hungarian + alignment to helper (classification-aware)
                aligned_cat, aligned_logits_dict = self._align_predictions_to_gt(
                    logits=cat_pred,
                    cls_logits=cls_pad,
                    cls_quality=qual_pad,
                    labels_or_masks=gt_combined,
                    return_logits=return_logits,
                )

                masks_per_batch = aligned_cat

                # 4) optionally split aligned outputs for return
                if return_logits:
                    aligned_out_logits = {"logits": [ac[:, :num_points] for ac in aligned_cat]}
                    if use_cat:
                        aligned_out_logits["inputs.logits"] = [ac[:, num_points:] for ac in aligned_cat]
                    if aligned_logits_dict is not None:
                        if "cls_logits" in aligned_logits_dict:
                            aligned_out_logits["cls_logits"] = aligned_logits_dict["cls_logits"]
                        if "cls_quality" in aligned_logits_dict:
                            aligned_out_logits["cls_quality"] = aligned_logits_dict["cls_quality"]

        combined_masks_per_batch = masks_per_batch
        masks_per_batch = [cm[:, :num_points] for cm in combined_masks_per_batch]
        inputs_masks_per_batch = [cm[:, num_points:] for cm in combined_masks_per_batch] if use_cat else None

        if show and len(logits) == 1:
            geometries = []
            if points is None:
                points = inputs
            else:
                if inputs is None:
                    raise ValueError("inputs is required when showing point clouds.")
                pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(inputs[0].float().cpu().numpy()))
                pcd.paint_uniform_color([0.8, 0.8, 0.8])
                geometries.append(pcd)
            if points is None:
                raise ValueError("points or inputs is required for visualization.")
            log_threshold = math.log(threshold / (1.0 - threshold))

            for i, mask in enumerate(masks_per_batch[0]):
                pts = points[0][mask > log_threshold]
                if pts.numel() > 0:
                    p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.float().cpu().numpy()))
                    p.paint_uniform_color(PLOTLY_COLORS[i])
                    geometries.append(p)

            if inputs is not None and inputs_masks_per_batch is not None:
                for i, mask in enumerate(inputs_masks_per_batch[0]):
                    pts = inputs[0][mask > log_threshold]
                    if pts.numel() > 0:
                        p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.float().cpu().numpy()))
                        p.paint_uniform_color((PLOTLY_COLORS[i] * 0.8).clip(0, 1))
                        geometries.append(p)

            cast(Any, o3d).visualization.draw_geometries(geometries)

        if return_logits:
            if align_to_gt:
                if aligned_out_logits is None:
                    aligned_out_logits = {"logits": masks_per_batch}
                return masks_per_batch, aligned_out_logits
            out_logits = {"logits": masks_per_batch}
            if use_cat and inputs_masks_per_batch is not None:
                out_logits["inputs.logits"] = inputs_masks_per_batch
            if method == "cls" and cls_logits_per_batch:
                out_logits["cls_logits"] = cls_logits_per_batch
                if cls_quality_per_batch:
                    out_logits["cls_quality"] = cls_quality_per_batch
            return masks_per_batch, out_logits

        return masks_per_batch

    @torch.inference_mode()
    def evaluate(
        self,
        data: dict[str, Any],
        threshold: float = 0.5,
        align_to_gt: bool = False,
        show: bool = False,
        prefix: str = "val/",
        **kwargs,
    ) -> dict[str, Tensor]:
        oracle_score = kwargs.pop("oracle_score", None)
        if oracle_score not in (None, False, "points", "inputs", "both"):
            raise ValueError("oracle_score must be one of: points, inputs, both.")
        if not any("logits" in k for k in data.keys()):
            if logger.isEnabledFor(DEBUG):
                torch.cuda.synchronize(self.device)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(self.device)
                m0 = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle).used
                start = time.perf_counter()

            data.update(cast(dict[str, Any], self(**{**data, **kwargs})))

            if logger.isEnabledFor(DEBUG):
                torch.cuda.synchronize(self.device)
                self.log("runtime", time.perf_counter() - start)
                m1 = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle).used
                self.log("vram_delta", cast(Any, int(m1) - int(m0)))
                self.log("vram_peak_allocated", torch.cuda.max_memory_allocated(self.device))
                self.log("vram_peak_reserved", torch.cuda.max_memory_reserved(self.device))
        logits = cast(Tensor, data["logits"])
        cls_logits = cast(Tensor | None, data.get("cls_logits"))
        cls_quality = cast(Tensor | None, data.get("cls_quality"))
        queries = cast(Tensor | None, data.get("queries"))

        # Metrics (MAP/PQ) use single branch by default
        masks_single = masks_from_labels(cast(Tensor, data.get("points.labels", data.get("inputs.labels"))))
        scores = self._get_scores(logits, cls_logits, cls_quality)
        if oracle_score in ("points", "both"):
            scores = self._get_oracle_dice_scores(logits, masks_single)
        self._update_metrics(logits=logits, targets=masks_single, scores=scores, threshold=threshold)
        if logger.isEnabledFor(DEBUG):
            diag = self._diagnose(logits, scores, cast(Tensor, data.get("points.labels", data.get("inputs.labels"))))

        # If multitask, also update metrics for inputs branch (separate accumulators)
        if self.multitask:
            in_logits = cast(Tensor, data["inputs.logits"])
            input_masks = masks_from_labels(cast(Tensor, data["inputs.labels"]))
            input_scores = self._get_scores(in_logits, cls_logits, cls_quality)
            if oracle_score in ("inputs", "both"):
                input_scores = self._get_oracle_dice_scores(in_logits, input_masks)
            self._update_metrics(
                logits=in_logits,
                targets=input_masks,
                scores=input_scores,
                threshold=threshold,
                accumulator="inputs",
            )

        # Build (possibly multitask) matching inputs
        points_logits = logits
        match_logits, match_masks, n_points = self._build_match_views(data, points_logits)

        # Hungarian + gather on coalesced tensors
        cat_logits, _ = self._coalesce_logits_masks(match_logits, match_masks)
        scores = self._get_scores(cat_logits, cls_logits, cls_quality)
        match, cost, batch_idx, src_idx, pred_masks, tgt_masks, _, _ = self._match_and_gather(
            logits_for_match=match_logits,
            masks_for_match=match_masks,
            scores=scores,
        )
        points_preds, points_tgt, inputs_preds, inputs_tgt = self._split_multitask(pred_masks, tgt_masks, n_points)

        if self.extent_weight > 0:
            points = data.get("points")
            if not isinstance(points, Tensor):
                raise ValueError("extent_weight > 0 requires a tensor points field")
            extent_logits = points_logits[(batch_idx, src_idx)]
            extent_targets = self._matched_completion_targets(match_masks, match, bool(self.multitask)).type_as(
                extent_logits
            )
            extent_loss, center_error, log_extent_error = self._extent_loss(
                points=points,
                logits=extent_logits,
                targets=extent_targets,
                batch_idx=batch_idx,
            )
            self.log("extent/loss", cast(Any, extent_loss))
            self.log("extent/center_l1", cast(Any, center_error))
            self.log("extent/log_half_extent_l1", cast(Any, log_extent_error))

        if logger.isEnabledFor(DEBUG_LEVEL_1):
            self.log("cost", cast(Any, cost))
            tgt_cls = torch.zeros_like(scores)
            tgt_cls[batch_idx, src_idx] = 1.0
            tgt_cls_quality = None
            if "qual" in self.pred_cls:
                if self.multitask:
                    if inputs_preds is None or inputs_tgt is None:
                        raise ValueError("inputs predictions and targets are required in multitask mode.")
                    dice_points = calculate_dice_score(points_preds.sigmoid(), points_tgt).type_as(tgt_cls)
                    dice_inputs = calculate_dice_score(inputs_preds.sigmoid(), inputs_tgt).type_as(tgt_cls)
                    w_sum = self.points_weight + self.inputs_weight
                    dice = (self.points_weight * dice_points + self.inputs_weight * dice_inputs) / w_sum
                else:
                    dice = calculate_dice_score(pred_masks.sigmoid(), tgt_masks).type_as(tgt_cls)

                if cls_quality is not None:
                    tgt_cls_quality = tgt_cls.clone()
                    tgt_cls_quality[batch_idx, src_idx] = dice
                else:
                    tgt_cls[batch_idx, src_idx] = dice
            cls_preds_for_log = scores if cls_quality is None else cast(Tensor, cls_logits).sigmoid()
            cls_quality_for_log = None if cls_quality is None else cls_quality.sigmoid()
            self._log_cls_metrics(
                cls_preds=cls_preds_for_log,
                cls_targets=tgt_cls >= 0.5 if self.pred_cls in ["qual", "quality"] else tgt_cls > 0,
                cls_quality=cls_quality_for_log,
                cls_quality_targets=tgt_cls_quality,
                threshold=self.cls_threshold,
            )
            if queries is not None:
                queries_logits = None
                if hasattr(self, "cls_head"):
                    z = self.head_norm(queries)
                    queries_logits = self.cls_head(z).squeeze(2)
                    if hasattr(self, "cls_quality_head"):
                        quality = self.cls_quality_head(z).squeeze(2)
                        queries_logits = (cast(Tensor, queries_logits).sigmoid() * quality.sigmoid()).logit(eps=1e-6)
                self._log_query_metrics(queries, queries_logits)

        # points loss
        loss = F.binary_cross_entropy_with_logits(points_preds, points_tgt) + dice_loss(points_preds, points_tgt)
        val_data = {"loss": loss, "logits": points_preds, "points.occ": points_tgt}
        results = super().evaluate(val_data, threshold=threshold, prefix=prefix, **kwargs)

        # inputs loss (if multitask)
        if self.multitask:
            if inputs_preds is None or inputs_tgt is None:
                raise ValueError("inputs predictions and targets are required in multitask mode.")
            loss_in = F.binary_cross_entropy_with_logits(inputs_preds, inputs_tgt) + dice_loss(inputs_preds, inputs_tgt)
            val_data_in = {"loss": loss_in, "logits": inputs_preds, "points.occ": inputs_tgt}
            results.update(super().evaluate(val_data_in, threshold=threshold, prefix=f"{prefix}inputs/", **kwargs))

        # Per-branch mask metrics with IoU-matrix matching (opt-in)
        if kwargs.get("mask_iou", False):
            mask_metrics = self._compute_mask_iou(points_preds, points_tgt, threshold)
            for k, v in mask_metrics.items():
                results[f"{prefix}mask_{k}"] = v
            if self.multitask and inputs_preds is not None and inputs_tgt is not None:
                in_mask_metrics = self._compute_mask_iou(inputs_preds, inputs_tgt, threshold)
                for k, v in in_mask_metrics.items():
                    results[f"{prefix}inputs/mask_{k}"] = v

        if logger.isEnabledFor(DEBUG):
            results.update({f"{prefix}diagnose/" + k: v for k, v in diag.items()})
            log_items = cast(dict[str, Any], self.get_log())
            results.update(
                {
                    f"{prefix}" + k: v[0]
                    for k, v in log_items.items()
                    if isinstance(v, list) and v and isinstance(v[0], (int, float))
                }
            )

        if (show or align_to_gt) and len(logits) == 1:
            if "inputs.labels" in data and "points.labels" in data:
                new_inputs, new_points = self._align_labels(
                    cast(Tensor, data["inputs.labels"]), cast(Tensor, data["points.labels"])
                )
                data["inputs.labels"], data["points.labels"] = new_inputs, new_points
            pred_out = cast(
                tuple[list[Tensor], dict[str, list[Tensor]]],
                self.predict(
                    inputs=cast(Tensor, data["inputs"]),
                    points=cast(Tensor | None, data.get("points")),
                    data=data,
                    threshold=threshold,
                    method="cls",
                    return_logits=True,
                    align_to_gt=align_to_gt,
                    show=show,
                    **filter_dict(kwargs, remove={"return_logits"}),
                ),
            )
            logits_dict = pred_out[1]
            results[f"{prefix}logits"] = logits_dict["logits"][0]
            if self.multitask:
                results[f"{prefix}inputs/logits"] = logits_dict["inputs.logits"][0]
        return results

    @torch.no_grad()
    def on_validation_epoch_end(self, *args, **kwargs) -> dict[str, float]:
        map_metrics = self.map.compute()
        pq_metrics = self.pq.compute()
        self.map.reset()
        self.pq.reset()

        results = {**map_metrics, **pq_metrics}

        if self.multitask:
            in_map = self.map_inputs.compute()
            in_pq = self.pq_inputs.compute()
            self.map_inputs.reset()
            self.pq_inputs.reset()
            results.update({f"inputs/{k}": v for k, v in {**in_map, **in_pq}.items()})

        return results

    @torch.no_grad()
    def _get_scores(
        self,
        logits: Tensor,
        cls_logits: Tensor | None = None,
        cls_quality: Tensor | None = None,
        method: Literal["mean", "top_k", "focal", "certainty", "adaptive", "ensemble", "peakiness"] = "mean",
        **kwargs,
    ) -> Tensor:
        if cls_logits is None:
            return cls_probs_from_logits(
                logits,
                min_size=self.min_mask_size,
                method=method,
                **filter_dict(kwargs, keep={"threshold", "temperature"}),
            )
        scores = cls_logits.sigmoid()
        if cls_quality is not None:
            scores = scores * cls_quality.sigmoid()
        return scores

    @staticmethod
    @torch.no_grad()
    def _get_oracle_dice_scores(logits: Tensor, targets: list[Tensor]) -> Tensor:
        scores = logits.new_zeros((logits.size(0), logits.size(1)))
        probs = logits.sigmoid()
        eps = 1e-6
        for i, target in enumerate(targets):
            if target.numel() == 0:
                continue
            tgt = target.to(device=logits.device, dtype=probs.dtype)
            numerator = 2 * torch.einsum("qm,gm->qg", probs[i], tgt)
            denominator = probs[i].sum(dim=1, keepdim=True) + tgt.sum(dim=1).unsqueeze(0)
            scores[i] = (numerator / (denominator + eps)).amax(dim=1)
        return scores

    def _coalesce_logits_masks(
        self,
        logits: Any,
        masks: Any,
    ) -> tuple[Tensor, list[Tensor]]:
        """
        Normalize inputs for gather:
            - If (points, inputs) are provided, concat along points-axis (last dim) per batch.
            - Otherwise, pass through.
        Returns:
            cat_logits: (B, Q, M_total)
            aligned_masks: list len B of (Q, M_total) bool tensors
        """
        if isinstance(logits, (tuple, list)):
            logits_seq = cast(Sequence[Tensor], logits)
            cat_logits = torch.cat(list(logits_seq), dim=2)
            assert isinstance(masks, (tuple, list)) and len(masks) == 2
            masks_seq = cast(Sequence[list[Tensor]], masks)
            aligned_masks = [torch.cat([pm, im], dim=1) for pm, im in zip(masks_seq[0], masks_seq[1], strict=False)]
            # Optional guard: ensure concat sizes match across modalities
            assert all(cat_logits.size(2) == am.size(1) for am in aligned_masks), (
                f"Mismatch: logits M={cat_logits.size(2)} vs masks M={[am.size(1) for am in aligned_masks]}"
            )
        else:
            cat_logits = cast(Tensor, logits)
            assert isinstance(masks, list)
            aligned_masks = cast(list[Tensor], masks)
            assert all(cat_logits.size(2) == am.size(1) for am in aligned_masks), (
                f"Mismatch: logits M={cat_logits.size(2)} vs masks M={[am.size(1) for am in aligned_masks]}"
            )
        return cat_logits, aligned_masks

    def _match_and_gather(
        self,
        logits_for_match: Any,
        masks_for_match: Any,
        scores: Tensor | None,
    ) -> tuple[
        list[tuple[Tensor, Tensor]],
        float | Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        list[Tensor],
    ]:
        """
        Runs Hungarian and gathers matched predictions/targets on the coalesced tensors.
        Returns:
            match, cost, batch_idx, src_idx, pred_masks, tgt_masks, cat_logits, aligned_masks
        """
        cat_logits, aligned_masks = self._coalesce_logits_masks(logits_for_match, masks_for_match)

        mw = self.mask_weight
        occ_weight = (mw * self.points_weight, mw * self.inputs_weight) if self.multitask else mw
        match, cost = hungarian_matcher(
            batch_size=len(cat_logits),
            occ_logits=logits_for_match,
            occ_tgt=masks_for_match,
            cls_preds=scores if self.match_cls else None,
            occ_weight=cast(Any, occ_weight),
            bce_weight=cast(float, self.bce_focal_weight),
            dice_weight=cast(float, self.dice_weight),
            cls_weight=cast(float, self.cls_weight),
            sample=cast(Any, self.sample),
        )

        batch_idx, src_idx = index_from_match(match)
        if self.pad_targets and self.training:
            pred_masks = cat_logits
            tgt_masks = torch.zeros_like(pred_masks)
            for i, (src, tgt) in enumerate(match):
                tgt_masks[i, src] = aligned_masks[i][tgt].type_as(tgt_masks)
            pred_masks = pred_masks.flatten(0, 1)
            tgt_masks = tgt_masks.flatten(0, 1)
        else:
            pred_masks = cat_logits[(batch_idx, src_idx)]
            tgt_masks = torch.cat([aligned_masks[i][j] for i, (_, j) in enumerate(match)]).type_as(cat_logits)

        return match, cost, batch_idx, src_idx, pred_masks, tgt_masks, cat_logits, aligned_masks

    def _split_multitask(
        self,
        pred_masks: Tensor,
        tgt_masks: Tensor,
        n_points: int | None,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        if n_points is None:
            return pred_masks, tgt_masks, None, None
        points_preds = pred_masks[:, :n_points]
        points_tgt = tgt_masks[:, :n_points]
        inputs_preds = pred_masks[:, n_points:]
        inputs_tgt = tgt_masks[:, n_points:]
        return points_preds, points_tgt, inputs_preds, inputs_tgt

    @staticmethod
    def _compute_mask_iou(
        preds: Tensor,
        targets: Tensor,
        threshold: float = 0.5,
    ) -> dict[str, float]:
        """Per-scene instance metrics with IoU-matrix Hungarian matching.

        Args:
            preds: [K, M] logits for K instances at M points (single branch).
            targets: [K, M] binary GT masks (same branch, after combined matching).
            threshold: binarization threshold for predictions.

        Returns:
            Dict with mean IoU, precision, and recall of optimally re-matched pairs.
        """
        from scipy.optimize import linear_sum_assignment

        empty = {"iou": 0.0, "precision": 0.0, "recall": 0.0}
        if preds.numel() == 0 or targets.numel() == 0:
            return empty

        pred_bin = preds.sigmoid() >= threshold
        tgt_bin = targets > 0.5

        intersection = (pred_bin.unsqueeze(1) & tgt_bin.unsqueeze(0)).sum(dim=2).float()
        union = (pred_bin.unsqueeze(1) | tgt_bin.unsqueeze(0)).sum(dim=2).float()
        pred_sum = pred_bin.sum(dim=1).float()  # [K_pred]
        tgt_sum = tgt_bin.sum(dim=1).float()  # [K_gt]
        pairwise = intersection / union.clamp_min(1.0)

        row, col = linear_sum_assignment(-pairwise.cpu().numpy())
        if len(row) == 0:
            return empty
        return {
            "iou": pairwise[row, col].mean().item(),
            "precision": (intersection[row, col] / pred_sum[row].clamp_min(1.0)).mean().item(),
            "recall": (intersection[row, col] / tgt_sum[col].clamp_min(1.0)).mean().item(),
        }

    def _maybe_sample_pairs(self, preds: Tensor, tgt: Tensor, k: int | None) -> tuple[Tensor, Tensor]:
        if k and preds.numel() > 0:
            idx = sample_uncertain(preds, num_points=k)
            preds = torch.gather(preds, dim=1, index=idx)
            tgt = torch.gather(tgt, dim=1, index=idx)
        return preds, tgt

    def _build_match_views(
        self,
        data: dict[str, Any],
        points_logits: Tensor,
    ) -> tuple[Any, Any, int | None]:
        """
        Returns:
            - match_logits: Tensor or (points_logits, inputs_logits)
            - match_masks: List[Tensor] or (points_masks, inputs_masks)
            - n_points: points branch width (for post-split), or None if single-task
        """
        masks_single = masks_from_labels(cast(Tensor, data.get("points.labels", data.get("inputs.labels"))))
        if not self.multitask:
            return points_logits, masks_single, None

        inputs_logits = cast(Tensor, data["inputs.logits"])
        inputs_idx = cast(Tensor | None, data.get("inputs.index"))
        i_labels = cast(Tensor, data["inputs.labels"])
        p_labels = cast(Tensor, data["points.labels"])

        if inputs_idx is not None:
            i_labels = torch.stack([lbl[idx] for lbl, idx in zip(i_labels, inputs_idx, strict=False)])

        aligned = self._align_masks(i_labels, p_labels)

        n_points = points_logits.size(2)
        points_masks = [m[:, :n_points] for m in aligned]
        inputs_masks = [m[:, n_points:] for m in aligned]

        assert all(
            (pm.size(1) + im.size(1)) == (pl.size(2) + inputs_logits.size(2))
            for pm, im, pl in zip(points_masks, inputs_masks, [points_logits] * len(points_masks), strict=False)
        )

        return (points_logits, inputs_logits), (points_masks, inputs_masks), n_points

    def _get_mask_loss(
        self,
        mask_logits: Tensor,
        mask_targets: Tensor,
        loss_name: str = "dice+bce",
        log: bool = True,
    ) -> Tensor:
        if mask_logits.numel() == 0:
            if mask_targets.numel() != 0:
                raise ValueError("Empty mask logits require empty mask targets.")
            return mask_logits.sum()

        loss = mask_logits.new_tensor(0.0)
        if "bce" in loss_name:
            pos_weight = None
            if self.mask_pos_weight is not None:
                if self.mask_pos_weight == "ratio":
                    pos = mask_targets.sum(dim=1, keepdim=True)
                    total = mask_targets.size(1)
                    neg = total - pos
                    pos_weight = (neg / pos.clamp_min(1.0)).clamp(0.01, 100.0)
                elif not torch.is_tensor(self.mask_pos_weight):
                    pos_weight = torch.as_tensor(
                        self.mask_pos_weight, device=mask_logits.device, dtype=mask_logits.dtype
                    )

            loss_b = F.binary_cross_entropy_with_logits(mask_logits, mask_targets, pos_weight=pos_weight)
            if log:
                self.log("bce_loss", cast(Any, loss_b), level=DEBUG_LEVEL_1)
            loss_b = self.bce_focal_weight * loss_b
            if isinstance(self._bce_focal_weight, nn.Parameter):
                loss_b = loss_b + 0.5 * self._bce_focal_weight
                if log:
                    self.log("bce_weight", cast(Any, self.bce_focal_weight), level=DEBUG_LEVEL_2)
            loss += loss_b
        elif "focal" in loss_name:
            loss_f = focal_loss.sigmoid_focal_loss(
                mask_logits, mask_targets, self.focal_alpha, self.focal_gamma, reduction="mean"
            )
            if log:
                self.log("focal_loss", cast(Any, loss_f), level=DEBUG_LEVEL_1)
            loss_f = self.bce_focal_weight * loss_f
            if isinstance(self._bce_focal_weight, nn.Parameter):
                loss_f = loss_f + 0.5 * self._bce_focal_weight
                if log:
                    self.log("bce_weight", cast(Any, self.bce_focal_weight), level=DEBUG_LEVEL_2)
            loss += loss_f
        if "tversky" in loss_name:
            # alpha up-weights FP penalty; beta up-weights FN penalty
            loss_tv = tversky_loss(mask_logits, mask_targets, alpha=0.7, beta=0.3)
            if log:
                self.log("tversky_loss", cast(Any, loss_tv), level=DEBUG_LEVEL_1)
            loss_tv = self.dice_weight * loss_tv
            if isinstance(self._dice_weight, nn.Parameter):
                loss_tv = loss_tv + 0.5 * self._dice_weight
                if log:
                    self.log("dice_weight", cast(Any, self.dice_weight), level=DEBUG_LEVEL_2)
            loss += loss_tv
        elif "dice" in loss_name:
            if "instr" in loss_name:
                loss_d = dice_loss_instr(mask_logits, mask_targets, power=0.2, pos_weight=1.0, neg_weight=0.5)
            else:
                loss_d = dice_loss(mask_logits, mask_targets)
            if log:
                self.log("dice_loss", cast(Any, loss_d), level=DEBUG_LEVEL_1)
            loss_d = self.dice_weight * loss_d
            if isinstance(self._dice_weight, nn.Parameter):
                loss_d = loss_d + 0.5 * self._dice_weight
                if log:
                    self.log("dice_weight", cast(Any, self.dice_weight), level=DEBUG_LEVEL_2)
            loss += loss_d
        loss = self.mask_weight * loss
        if isinstance(self._mask_weight, nn.Parameter):
            loss = loss + 0.5 * self._mask_weight
            if log:
                self.log("mask_weight", cast(Any, self.mask_weight), level=DEBUG_LEVEL_2)
        return cast(Tensor, loss)

    def _get_cls_loss(
        self,
        cls_logits: Tensor,
        cls_targets: Tensor,
        cls_quality: Tensor | None = None,
        cls_quality_targets: Tensor | None = None,
        pos_weight: Any = None,
        loss_name: str = "bce",
        log: bool = True,
    ) -> Tensor:
        pos_w = pos_weight
        if "bce" in loss_name:
            if pos_w is not None:
                if pos_w == "ratio":
                    pos = (cls_targets > 0).sum(dim=1, keepdim=True)
                    total = cls_targets.size(1)
                    neg = total - pos
                    pos_w = (neg / pos.clamp_min(1.0)).clamp(0.01, 100.0)
                elif not torch.is_tensor(pos_w):
                    pos_w = torch.as_tensor(pos_w, device=cls_logits.device, dtype=cls_logits.dtype)

            loss_cls = F.binary_cross_entropy_with_logits(cls_logits, cls_targets, pos_weight=cast(Any, pos_w))
        elif "focal" in loss_name:
            loss_cls = focal_loss.sigmoid_focal_loss(
                cls_logits, cls_targets, self.focal_alpha, self.focal_gamma, reduction="mean"
            )
        else:
            raise ValueError(f"Unsupported classification loss: {self.loss_name}")

        if cls_quality is not None and cls_quality_targets is not None:
            if cls_quality.numel() == 0:
                if cls_quality_targets.numel() != 0:
                    raise ValueError("Empty classification quality logits require empty targets.")
                loss_cls += cls_quality.sum()
            else:
                loss_cls += F.binary_cross_entropy_with_logits(cls_quality, cls_quality_targets)

        if log:
            self.log("cls_loss", cast(Any, loss_cls), level=DEBUG_LEVEL_1)
        loss_cls = self.cls_weight * loss_cls
        if isinstance(self._cls_weight, nn.Parameter):
            loss_cls = loss_cls + 0.5 * self._cls_weight
            if log:
                self.log("cls_weight", cast(Any, self.cls_weight), level=DEBUG_LEVEL_2)
        return loss_cls

    def _get_aux_loss(
        self,
        logits: Any,
        masks: Any,
        cls_logits: Tensor | None = None,
        cls_quality: Tensor | None = None,
    ) -> Tensor:
        # 1) scores for matching
        cat_logits, _ = self._coalesce_logits_masks(logits, masks)
        scores = self._get_scores(cat_logits, cls_logits, cls_quality)

        # 2) match + gather on coalesced tensors
        _, _, batch_idx, src_idx, pred_masks, tgt_masks, _, _ = self._match_and_gather(
            logits_for_match=logits,
            masks_for_match=masks,
            scores=scores,
        )

        logits0 = cast(Tensor, logits[0]) if isinstance(logits, (tuple, list)) else cast(Tensor, logits)
        n_points = logits0.size(2) if self.multitask else None
        points_preds, points_tgt, inputs_preds, inputs_tgt = self._split_multitask(pred_masks, tgt_masks, n_points)
        sample_cfg = cast(Any, self).sample
        if self.multitask and sample_cfg:
            if sample_cfg[0]:
                points_preds, points_tgt = self._maybe_sample_pairs(points_preds, points_tgt, sample_cfg[0])
            if sample_cfg[1] and inputs_preds is not None and inputs_tgt is not None:
                inputs_preds, inputs_tgt = self._maybe_sample_pairs(inputs_preds, inputs_tgt, sample_cfg[1])
            if inputs_preds is None or inputs_tgt is None:
                raise ValueError("inputs predictions and targets are required in multitask mode.")
            pred_masks = torch.cat([points_preds, inputs_preds], dim=1)
            tgt_masks = torch.cat([points_tgt, inputs_tgt], dim=1)
        elif sample_cfg and pred_masks.numel() > 0:
            pred_masks, tgt_masks = self._maybe_sample_pairs(pred_masks, tgt_masks, cast(int | None, sample_cfg))

        # 3) main mask loss
        if self.multitask:
            points_loss = self._get_mask_loss(points_preds, points_tgt, loss_name=self.loss_name, log=False)
            if inputs_preds is None or inputs_tgt is None:
                raise ValueError("inputs predictions and targets are required in multitask mode.")
            inputs_loss = self._get_mask_loss(inputs_preds, inputs_tgt, loss_name=self.loss_name, log=False)
            loss = self.points_weight * points_loss + self.inputs_weight * inputs_loss
            if isinstance(self._points_weight, nn.Parameter):
                loss = loss + 0.5 * self._points_weight
            if isinstance(self._inputs_weight, nn.Parameter):
                loss = loss + 0.5 * self._inputs_weight
        else:
            loss = self._get_mask_loss(pred_masks, tgt_masks, loss_name=self.loss_name, log=False)

        # 4) optional classification
        if cls_logits is not None:
            tgt_cls = torch.zeros_like(scores)
            tgt_cls[batch_idx, src_idx] = 1.0

            with torch.no_grad():
                if self.multitask and inputs_preds is not None:
                    dice_points = calculate_dice_score(points_preds.sigmoid(), points_tgt).type_as(tgt_cls)
                    dice_inputs = calculate_dice_score(inputs_preds.sigmoid(), inputs_tgt).type_as(tgt_cls)
                    w_sum = self.points_weight + self.inputs_weight
                    dice = (self.points_weight * dice_points + self.inputs_weight * dice_inputs) / w_sum
                else:
                    dice = calculate_dice_score(pred_masks.sigmoid(), tgt_masks).type_as(tgt_cls)

            q_logits = None
            q_targets = None
            if "qual" in self.pred_cls:
                if cls_quality is not None:
                    q_logits = cls_quality[batch_idx, src_idx]
                    q_targets = dice
                else:
                    tgt_cls[batch_idx, src_idx] = dice

            pos_weight = self.cls_pos_weight
            if "obj" in self.pred_cls:
                pos_weight = torch.ones_like(tgt_cls)
                pos_weight[batch_idx, src_idx] += dice
                if isinstance(self.cls_pos_weight, float):
                    pos_weight[batch_idx, src_idx] *= self.cls_pos_weight

            loss += self._get_cls_loss(
                cls_logits,
                tgt_cls,
                cls_quality=q_logits,
                cls_quality_targets=q_targets,
                pos_weight=pos_weight,
                log=False,
            )
        return loss

    def loss(
        self,
        data: dict[str, Any],
        log_freq: int = 10,
        global_step: int | None = None,
        total_steps: int | None = None,
        **kwargs,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if not any("logits" in k for k in data.keys()):
            data.update(
                cast(dict[str, Any], self(**{**data, **kwargs}, global_step=global_step, total_steps=total_steps))
            )

        log = True
        log_10 = True
        if global_step is not None:
            log = global_step % log_freq == 0
            log_10 = global_step % (10 * log_freq) == 0

        logits = cast(Tensor, data["logits"])
        cls_logits = cast(Tensor | None, data.get("cls_logits"))
        cls_quality = cast(Tensor | None, data.get("cls_quality"))
        queries = cast(Tensor | None, data.get("queries"))

        if self.anneal_cls and global_step is not None and total_steps is not None:
            self.cls_weight_scale = cosine_anneal(
                start=0.01, stop=1.0, steps=int(self.anneal_cls * total_steps), current_step=global_step
            )

        # Build masks + choose matching view (single vs multitask)
        points_logits = logits
        match_logits, match_masks, n_points = self._build_match_views(data, points_logits)

        # Scores for matching
        cat_logits, aligned_masks = self._coalesce_logits_masks(match_logits, match_masks)
        scores = self._get_scores(cat_logits, cls_logits, cls_quality)

        # Hungarian + gather
        match, cost, batch_idx, src_idx, pred_masks, tgt_masks, cat_logits, aligned_masks = self._match_and_gather(
            logits_for_match=match_logits,
            masks_for_match=match_masks,
            scores=scores,
        )

        extent_logits = None
        extent_targets = None
        if self.extent_weight > 0:
            extent_logits = points_logits[(batch_idx, src_idx)]
            extent_targets = self._matched_completion_targets(match_masks, match, bool(self.multitask)).type_as(
                extent_logits
            )

        # Optional per-modality split and stratified sampling
        points_preds, points_tgt, inputs_preds, inputs_tgt = self._split_multitask(pred_masks, tgt_masks, n_points)
        sample_cfg = cast(Any, self).sample
        if self.multitask and sample_cfg:
            if sample_cfg[0]:
                points_preds, points_tgt = self._maybe_sample_pairs(points_preds, points_tgt, sample_cfg[0])
            if sample_cfg[1] and inputs_preds is not None and inputs_tgt is not None:
                inputs_preds, inputs_tgt = self._maybe_sample_pairs(inputs_preds, inputs_tgt, sample_cfg[1])
            if inputs_preds is None or inputs_tgt is None:
                raise ValueError("inputs predictions and targets are required in multitask mode.")
            pred_masks = torch.cat([points_preds, inputs_preds], dim=1)
            tgt_masks = torch.cat([points_tgt, inputs_tgt], dim=1)
        elif sample_cfg and pred_masks.numel() > 0:
            pred_masks, tgt_masks = self._maybe_sample_pairs(pred_masks, tgt_masks, cast(int | None, sample_cfg))

        # Main mask loss
        if self.multitask:
            points_loss = self._get_mask_loss(points_preds, points_tgt, loss_name=self.loss_name, log=log)
            if inputs_preds is None or inputs_tgt is None:
                raise ValueError("inputs predictions and targets are required in multitask mode.")
            inputs_loss = self._get_mask_loss(inputs_preds, inputs_tgt, loss_name=self.loss_name, log=False)
            loss = self.points_weight * points_loss + self.inputs_weight * inputs_loss
            if isinstance(self._points_weight, nn.Parameter):
                loss = loss + 0.5 * self._points_weight
                if log:
                    self.log("points_weight", cast(Any, self.points_weight), level=DEBUG_LEVEL_2)
            if isinstance(self._inputs_weight, nn.Parameter):
                loss = loss + 0.5 * self._inputs_weight
                if log:
                    self.log("inputs_weight", cast(Any, self.inputs_weight), level=DEBUG_LEVEL_2)
        else:
            loss = self._get_mask_loss(pred_masks, tgt_masks, loss_name=self.loss_name, log=log)

        if self.extent_weight > 0:
            points = data.get("points")
            if not isinstance(points, Tensor):
                raise ValueError("extent_weight > 0 requires a tensor points field")
            if extent_logits is None or extent_targets is None:
                raise RuntimeError("Extent tensors were not prepared after matching")
            extent_loss, center_error, log_extent_error = self._extent_loss(
                points=points,
                logits=extent_logits,
                targets=extent_targets,
                batch_idx=batch_idx,
            )
            loss = loss + self.extent_weight * extent_loss
            if log:
                self.log("extent/loss", cast(Any, extent_loss.detach()))
                self.log("extent/center_l1", cast(Any, center_error))
                self.log("extent/log_half_extent_l1", cast(Any, log_extent_error))

        # Classification targets (per-query)
        tgt_cls = torch.zeros_like(scores)
        tgt_cls[batch_idx, src_idx] = 1.0

        with torch.no_grad():
            if self.multitask and inputs_preds is not None:
                dice_points = calculate_dice_score(points_preds.sigmoid(), points_tgt).type_as(tgt_cls)
                dice_inputs = calculate_dice_score(inputs_preds.sigmoid(), inputs_tgt).type_as(tgt_cls)
                w_sum = self.points_weight + self.inputs_weight
                dice = (self.points_weight * dice_points + self.inputs_weight * dice_inputs) / w_sum
            else:
                dice = calculate_dice_score(pred_masks.sigmoid(), tgt_masks).type_as(tgt_cls)

        q_logits = None
        q_targets = None
        tgt_cls_quality = None
        if "qual" in self.pred_cls:
            if cls_quality is not None:
                q_logits = cls_quality[batch_idx, src_idx]
                q_targets = dice
                tgt_cls_quality = tgt_cls.clone()
                tgt_cls_quality[batch_idx, src_idx] = dice
            else:
                tgt_cls[batch_idx, src_idx] = dice

        pos_weight = self.cls_pos_weight
        if "obj" in self.pred_cls:
            pos_weight = torch.ones_like(tgt_cls)
            pos_weight[batch_idx, src_idx] += dice
            if isinstance(self.cls_pos_weight, float):
                pos_weight[batch_idx, src_idx] *= self.cls_pos_weight

        if cls_logits is not None:
            loss += self._get_cls_loss(
                cls_logits, tgt_cls, cls_quality=q_logits, cls_quality_targets=q_targets, pos_weight=pos_weight, log=log
            )

        # Aux heads (tuple-aware)
        aux_out = cast(list[dict[str, Any]] | None, data.get("aux_out"))
        if aux_out:
            aux_loss = logits.new_tensor(0.0)
            for aux in aux_out:
                aux_logits = (aux["logits"], aux["inputs.logits"]) if self.multitask else aux["logits"]
                aux_loss = aux_loss + self._get_aux_loss(
                    logits=aux_logits,
                    masks=match_masks,
                    cls_logits=cast(Tensor | None, aux.get("cls_logits")),
                    cls_quality=cast(Tensor | None, aux.get("cls_quality")),
                )
            self.log("aux_loss", cast(Any, aux_loss), level=DEBUG_LEVEL_1)
            aux_loss = self.aux_weight * aux_loss
            if isinstance(self._aux_weight, nn.Parameter):
                aux_loss = aux_loss + 0.5 * self._aux_weight
                if log:
                    self.log("aux_weight", cast(Any, self.aux_weight), level=DEBUG_LEVEL_2)
            loss += aux_loss

        # Debug/metrics
        with torch.no_grad():
            if logger.isEnabledFor(DEBUG_LEVEL_2):
                if log:
                    self.log("cost", cast(Any, cost))
                    if hasattr(self, "logit_scale"):
                        self.log("logit_scale", self.logit_scale.exp().item())
                if log_10:
                    cls_preds_for_log = scores if cls_quality is None else cast(Tensor, cls_logits).sigmoid()
                    cls_quality_for_log = None if cls_quality is None else cls_quality.sigmoid()
                    self._log_cls_metrics(
                        cls_preds=cls_preds_for_log,
                        cls_targets=tgt_cls >= 0.5 if self.pred_cls in ["qual", "quality"] else tgt_cls > 0,
                        cls_quality=cls_quality_for_log,
                        cls_quality_targets=tgt_cls_quality,
                    )
                    if queries is not None:
                        queries_logits = None
                        if hasattr(self, "cls_head"):
                            z = self.head_norm(queries)
                            queries_logits = self.cls_head(z).squeeze(2)
                            if hasattr(self, "cls_quality_head"):
                                quality = self.cls_quality_head(z).squeeze(2)
                                queries_logits = (cast(Tensor, queries_logits).sigmoid() * quality.sigmoid()).logit(
                                    eps=1e-6
                                )
                        self._log_query_metrics(queries, queries_logits)
                    # ensure per-modality logging mirrors training tensors
                    if self.multitask and points_preds is not None:
                        pred_masks_log = points_preds
                        tgt_masks_log = points_tgt
                    elif self.pad_targets:
                        pred_masks_log = cat_logits[(batch_idx, src_idx)]
                        tgt_masks_log = torch.cat([aligned_masks[i][j] for i, (_, j) in enumerate(match)]).type_as(
                            cat_logits
                        )
                    else:
                        pred_masks_log = pred_masks
                        tgt_masks_log = tgt_masks

                    train_data: dict[str, Any] = {"loss": loss, "logits": pred_masks_log, "points.occ": tgt_masks_log}
                    train_metrics = MultiEvalMixin.evaluate(self, cast(Any, train_data), prefix="train/", **kwargs)
                    train_metrics.pop("train/loss")
                    self.log_dict(train_metrics)

        return loss


class DinoInstSegRGBD(Model):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model_2d = DinoInstSeg(*args, **kwargs)
        self.model_3d = DinoInstSeg3D(*args, **kwargs)
        self.mix_feat = Encoder(n_layer=3, n_embd=self.model_2d.n_embd, n_head=self.model_2d.n_head)
        self.mod_emb = nn.Parameter(torch.zeros(2, self.model_2d.n_embd))
        self.pre_ln_2d = nn.LayerNorm(self.model_2d.n_embd)
        self.pre_ln_3d = nn.LayerNorm(self.model_2d.n_embd)

    def forward(self, inputs: Tensor, points: Tensor | None = None, **kwargs) -> tuple[dict[str, Any], dict[str, Any]]:
        feat_2d, feat_3d = self.encode(inputs=inputs, **kwargs)
        return self.decode(
            points=inputs if points is None else points,
            feature=cast(tuple[Tensor, Tensor], (feat_2d, feat_3d)),
            **kwargs,
        )

    def encode(self, inputs: Tensor, **kwargs) -> tuple[list[tuple[Tensor, Tensor]], Tensor]:
        feat_2d = self.model_2d.encode(inputs=kwargs["inputs.image"], **kwargs)
        feat_3d = self.model_3d.encode(inputs=inputs, **kwargs)
        return feat_2d, feat_3d

    def decode(
        self,
        points: Tensor,
        feature: tuple[Tensor, Tensor],
        **kwargs,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        feat_2d, feat_3d = feature
        b, c, h, w = feat_2d[0][0].size()
        patch_2d = torch.cat([f[0].view(b, c, -1).transpose(1, 2) for f in feat_2d], dim=1)
        cls_2d = [f[1] for f in feat_2d]

        patch_2d = self.pre_ln_2d(patch_2d) + self.mod_emb[0]
        feat_3d = self.pre_ln_3d(feat_3d) + self.mod_emb[1]
        feat = torch.cat([patch_2d, feat_3d], dim=1)
        feat = self.mix_feat(feat)

        patch_2d, feat_3d = feat.split([patch_2d.size(1), feat_3d.size(1)], dim=1)
        patch_2d = patch_2d.chunk(len(feat_2d), dim=1)
        patch_2d = [p.transpose(1, 2).view(b, c, h, w) for p in patch_2d]
        feat_2d = list(zip(patch_2d, cls_2d, strict=False))

        out_2d = self.model_2d.decode(feature=feat_2d, **kwargs)
        out_3d = self.model_3d.decode(points=points, feature=feat_3d, **kwargs)

        return out_2d, out_3d

    def predict(self, *args, **kwargs):
        raise NotImplementedError

    @torch.inference_mode()
    def evaluate(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        out_2d, out_3d = self(**{**data, **kwargs})
        metrics_2d = self.model_2d.evaluate(data, out_2d, **kwargs)
        metrics_3d = self.model_3d.evaluate(data, out_3d, **kwargs)
        metrics_3d = {f"{k}_3d": v for k, v in metrics_3d.items()}

        for key, value in cast(dict[str, Any], self.model_2d.get_log()).items():
            if isinstance(value, list) and len(value) >= 2 and isinstance(value[1], int):
                self.log(key, value[0], level=value[1])
        for key, value in cast(dict[str, Any], self.model_3d.get_log()).items():
            if isinstance(value, list) and len(value) >= 2 and isinstance(value[1], int):
                self.log(f"{key}_3d", value[0], level=value[1])
        self.model_2d.clear_log()
        self.model_3d.clear_log()

        return {**metrics_2d, **metrics_3d}

    @torch.no_grad()
    def on_validation_epoch_end(self, *args, **kwargs) -> dict[str, float]:
        metrics_2d = self.model_2d.on_validation_epoch_end(*args, **kwargs)
        metrics_3d = self.model_3d.on_validation_epoch_end(*args, **kwargs)
        metrics_3d = {f"{k}_3d": v for k, v in metrics_3d.items()}
        return {**metrics_2d, **metrics_3d}

    def loss(self, data: dict[str, Any], **kwargs) -> Tensor:
        out_2d, out_3d = self(**{**data, **kwargs})
        loss_2d = cast(Tensor, self.model_2d.loss(data=data, out=out_2d, **kwargs))
        loss_3d = cast(Tensor, self.model_3d.loss(data=data, out=out_3d, **kwargs))

        for key, value in cast(dict[str, Any], self.model_2d.get_log()).items():
            if isinstance(value, list) and len(value) >= 2 and isinstance(value[1], int):
                self.log(key, value[0], level=value[1])
        for key, value in cast(dict[str, Any], self.model_3d.get_log()).items():
            if isinstance(value, list) and len(value) >= 2 and isinstance(value[1], int):
                self.log(f"{key}_3d", value[0], level=value[1])
        self.model_2d.clear_log()
        self.model_3d.clear_log()

        return loss_2d + loss_3d


class RGBPointFusion(nn.Module):
    def __init__(
        self,
        dim: int,
        rgb_dim: int,
        mode: Literal["add", "zero_add", "gate", "cat_linear"] = "gate",
        gate_init_bias: float = -10.0,
    ):
        super().__init__()
        self.mode = mode
        self.last_gate_stats: dict[str, float] | None = None
        if mode == "zero_add":
            projection = nn.Linear(rgb_dim, dim)
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
            self.rgb_proj = projection
        else:
            self.rgb_proj = nn.Sequential(nn.Linear(rgb_dim, dim), nn.LayerNorm(dim))
        if mode == "gate":
            self.gate = nn.Sequential(nn.Linear(2 * dim, dim), nn.Sigmoid())
            nn.init.constant_(cast(nn.Linear, self.gate[0]).bias, gate_init_bias)
        elif mode == "cat_linear":
            self.fuse = nn.Sequential(nn.Linear(2 * dim, dim), nn.LayerNorm(dim))
        elif mode not in {"add", "zero_add"}:
            raise ValueError(f"Unsupported fusion mode '{mode}'.")

    def forward(self, depth_feat: Tensor, rgb_feat: Tensor) -> Tensor:
        rgb = self.rgb_proj(rgb_feat)
        if depth_feat.shape[:2] != rgb.shape[:2]:
            raise ValueError(f"RGB features must align to depth inputs, got {rgb.shape} and {depth_feat.shape}.")
        if self.mode in {"add", "zero_add"}:
            return depth_feat + rgb
        if self.mode == "gate":
            gate = self.gate(torch.cat((depth_feat, rgb), dim=-1))
            gate_detached = gate.detach()
            self.last_gate_stats = {
                "mean": float(gate_detached.mean().item()),
                "min": float(gate_detached.min().item()),
                "max": float(gate_detached.max().item()),
            }
            return depth_feat + gate * rgb
        self.last_gate_stats = None
        return cast(nn.Module, self.fuse)(torch.cat((depth_feat, rgb), dim=-1))


class RGBMetricAdapter(nn.Module):
    def __init__(self, input_dim: int = 384, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        return F.normalize(self.net(features), dim=-1)


def _norm_bounded_add(
    base: Tensor,
    residual: Tensor,
    max_ratio: float,
) -> tuple[Tensor, dict[str, float]]:
    if base.shape != residual.shape:
        raise ValueError(f"Residual must match base features, got {residual.shape} and {base.shape}.")
    base_norm = base.float().norm(dim=-1, keepdim=True)
    residual_norm = residual.float().norm(dim=-1, keepdim=True)
    precision_margin = 1.0 - 4.0 * torch.finfo(residual.dtype).eps
    allowed_norm = max_ratio * max(precision_margin, 0.95) * base_norm
    scale = torch.minimum(torch.ones_like(residual_norm), allowed_norm / residual_norm.clamp_min(1e-12))
    bounded = residual * scale.to(dtype=residual.dtype)
    fused = base + bounded
    actual_residual_norm = (fused.float() - base.float()).norm(dim=-1)
    ratio = torch.where(base_norm.squeeze(-1) > 0, actual_residual_norm / base_norm.squeeze(-1), 0.0)
    ratio = ratio.detach()
    stats = {
        "mean": float(ratio.mean().item()),
        "p95": float(torch.quantile(ratio, 0.95).item()),
        "max": float(ratio.max().item()),
    }
    return fused, stats


def _fusion_keep_masks(
    random_values: Tensor,
    rgb_residual_dropout: float,
    depth_fusion_dropout: float,
) -> tuple[Tensor, Tensor]:
    if random_values.ndim != 1:
        raise ValueError("Fusion dropout samples must be one-dimensional.")
    if not 0 <= rgb_residual_dropout < 1:
        raise ValueError("rgb_residual_dropout must fall within [0, 1)")
    if not 0 <= depth_fusion_dropout < 1:
        raise ValueError("depth_fusion_dropout must fall within [0, 1)")
    if rgb_residual_dropout + depth_fusion_dropout >= 1:
        raise ValueError("RGB and depth fusion dropout probabilities must sum to less than 1")

    rgb_keep = random_values >= rgb_residual_dropout
    depth_drop = (random_values >= rgb_residual_dropout) & (random_values < rgb_residual_dropout + depth_fusion_dropout)
    depth_keep = ~depth_drop
    return rgb_keep, depth_keep


def _drop_rgb_feature_tokens(
    rgb_features: Tensor,
    dropout_probability: float,
    force_keep_samples: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    if rgb_features.ndim != 3:
        raise ValueError("RGB features must have shape [batch, tokens, channels].")
    if not 0 <= dropout_probability < 1:
        raise ValueError("rgb_token_dropout must fall within [0, 1)")

    keep_mask = (
        torch.rand(
            (*rgb_features.shape[:2], 1),
            device=rgb_features.device,
        )
        >= dropout_probability
    )
    if force_keep_samples is not None:
        if force_keep_samples.ndim != 1 or force_keep_samples.size(0) != rgb_features.size(0):
            raise ValueError("Forced RGB-token keep mask must have shape [batch].")
        keep_mask = keep_mask | force_keep_samples.to(device=rgb_features.device, dtype=torch.bool).view(-1, 1, 1)
    return rgb_features * keep_mask.to(dtype=rgb_features.dtype), keep_mask


class NormBoundedPointFusion(nn.Module):
    def __init__(self, dim: int, rgb_dim: int, max_ratio: float = 0.1):
        super().__init__()
        if not 0 < max_ratio <= 1:
            raise ValueError("max_ratio must fall within (0, 1]")
        self.max_ratio = max_ratio
        self.rgb_proj = nn.Linear(rgb_dim, dim)
        nn.init.zeros_(self.rgb_proj.weight)
        nn.init.zeros_(self.rgb_proj.bias)
        self.last_ratio_stats: dict[str, float] | None = None
        self.sample_mask: Tensor | None = None
        self.depth_sample_mask: Tensor | None = None

    def forward(self, depth_feat: Tensor, rgb_feat: Tensor) -> Tensor:
        residual = self.rgb_proj(rgb_feat)
        if self.sample_mask is not None:
            sample_mask = self.sample_mask.to(device=residual.device, dtype=residual.dtype)
            residual = residual * sample_mask.view(-1, 1, 1)
        fused, self.last_ratio_stats = _norm_bounded_add(depth_feat, residual, self.max_ratio)
        if self.depth_sample_mask is not None:
            depth_mask = self.depth_sample_mask.to(device=depth_feat.device, dtype=depth_feat.dtype)
            bounded_residual = fused - depth_feat
            fused = depth_feat * depth_mask.view(-1, 1, 1) + bounded_residual
        return fused


class NormBoundedQueryFusion(nn.Module):
    def __init__(self, dim: int, rgb_dim: int, num_heads: int, max_ratio: float = 0.1):
        super().__init__()
        if not 0 < max_ratio <= 1:
            raise ValueError("max_ratio must fall within (0, 1]")
        self.max_ratio = max_ratio
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            kdim=rgb_dim,
            vdim=rgb_dim,
            batch_first=True,
        )
        nn.init.zeros_(self.cross_attn.out_proj.weight)
        nn.init.zeros_(self.cross_attn.out_proj.bias)
        self.last_ratio_stats: dict[str, float] | None = None
        self.sample_mask: Tensor | None = None
        self.depth_sample_mask: Tensor | None = None

    def forward(self, queries: Tensor, rgb_memory: Tensor) -> Tensor:
        residual, _ = self.cross_attn(queries, rgb_memory, rgb_memory, need_weights=False)
        if self.sample_mask is not None:
            sample_mask = self.sample_mask.to(device=residual.device, dtype=residual.dtype)
            residual = residual * sample_mask.view(-1, 1, 1)
        fused, self.last_ratio_stats = _norm_bounded_add(queries, residual, self.max_ratio)
        if self.depth_sample_mask is not None:
            depth_mask = self.depth_sample_mask.to(device=queries.device, dtype=queries.dtype)
            bounded_residual = fused - queries
            fused = queries * depth_mask.view(-1, 1, 1) + bounded_residual
        return fused


class DinoPointFeatureProvider(nn.Module):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    mean: Tensor
    std: Tensor

    def __init__(
        self,
        repo_or_dir: str = "facebookresearch/dinov2",
        backbone: str = "dinov2_vits14",
        freeze: bool = True,
        image_normalization: Literal["auto", "imagenet", "rgb"] = "auto",
    ):
        super().__init__()
        self.image_normalization = image_normalization
        self.encoder = _load_dino_backbone(repo_or_dir, backbone, pretrained=True)
        if freeze:
            self.encoder.requires_grad_(False)
            self.encoder.train(False)
        self.encoder.mask_token = None
        self.patch_size = int(getattr(self.encoder, "patch_size", 14))
        self.out_dim = int(self.encoder.embed_dim)
        self.register_buffer("mean", torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1))

    def _prepare_image(self, image: Tensor) -> Tensor:
        if self.image_normalization == "imagenet":
            return image if torch.is_floating_point(image) else image.float()

        if self.image_normalization not in {"auto", "rgb"}:
            raise ValueError(f"Unsupported image_normalization '{self.image_normalization}'.")

        if not torch.is_floating_point(image):
            image = image.float()
        image_min = image.detach().amin()
        image_max = image.detach().amax()
        if image_min >= 0 and image_max > 2:
            image = image / 255.0
        elif self.image_normalization == "auto" and (image_min < 0 or image_max > 1):
            return image
        return (image - self.mean) / self.std

    def _encode_patches(self, image: Tensor) -> tuple[Tensor, int, int]:
        # image: (B, 3, H, W), either unit-range/0-255 RGB or already ImageNet-normalized.
        x = self._prepare_image(image)
        h, w = x.shape[-2:]
        gh = max(1, h // self.patch_size)
        gw = max(1, w // self.patch_size)
        th, tw = gh * self.patch_size, gw * self.patch_size
        if (th, tw) != (h, w):
            x = F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)
        features = self.encoder.forward_features(x)
        if not isinstance(features, dict):
            raise TypeError("DINO provider expects forward_features to return a dict.")
        patches = cast(Tensor, features["x_norm_patchtokens"])
        if patches.shape[1] != gh * gw:
            raise ValueError(f"Expected {gh * gw} patch tokens, got {patches.shape[1]}.")
        return patches, gh, gw

    def forward(self, image: Tensor, pixel_coords: Tensor) -> Tensor:
        h, w = image.shape[-2:]
        patches, gh, gw = self._encode_patches(image)
        b, _, c = patches.shape
        fmap = patches.transpose(1, 2).reshape(b, c, gh, gw)
        coords = pixel_coords.to(device=image.device, dtype=fmap.dtype)
        # pixel_coords are [row(height), col(width)] in original image pixels.
        y = coords[..., 0] / max(h - 1, 1)
        x = coords[..., 1] / max(w - 1, 1)
        grid = torch.stack((2 * x - 1, 2 * y - 1), dim=-1).unsqueeze(2)
        sampled = F.grid_sample(fmap, grid, mode="bilinear", align_corners=True)
        return sampled.squeeze(-1).transpose(1, 2)


class FeatUpPointFeatureProvider(nn.Module):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    mean: Tensor
    std: Tensor

    def __init__(self, repo: str, checkpoint: str):
        super().__init__()
        repo_path = Path(repo)
        checkpoint_path = Path(checkpoint)
        if not repo_path.is_dir():
            raise FileNotFoundError(repo_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

        model: Any = torch.hub.load(
            str(repo_path),
            "dinov2",
            source="local",
            pretrained=False,
            use_norm=True,
        )
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = {
            key: value
            for key, value in checkpoint_data["state_dict"].items()
            if "scale_net" not in key and "downsampler" not in key
        }
        incompatible = model.load_state_dict(state_dict, strict=False)
        # Official FeatUp JBU checkpoints contain only upsampler weights; the
        # constructor above loads the frozen DINOv2 backbone independently.
        if incompatible.unexpected_keys or any(not key.startswith("model.") for key in incompatible.missing_keys):
            raise ValueError(
                "FeatUp checkpoint is incompatible: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        self.model = model.eval().requires_grad_(False)
        self.out_dim = int(model.dim)
        self.register_buffer("mean", torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1))

    def _prepare_image(self, image: Tensor) -> Tensor:
        image = image.float()
        image_min = image.detach().amin()
        image_max = image.detach().amax()
        if image_min >= 0 and image_max > 2:
            image = image / 255.0
        elif image_min < 0 or image_max > 1:
            return image
        return (image - self.mean) / self.std

    @staticmethod
    def _sample(fmap: Tensor, pixel_coords: Tensor, image_shape: tuple[int, int]) -> Tensor:
        height, width = image_shape
        coords = pixel_coords.to(device=fmap.device, dtype=fmap.dtype)
        y = coords[..., 0] / max(height - 1, 1)
        x = coords[..., 1] / max(width - 1, 1)
        grid = torch.stack((2 * x - 1, 2 * y - 1), dim=-1).unsqueeze(2)
        sampled = F.grid_sample(fmap, grid, mode="bilinear", align_corners=True)
        return sampled.squeeze(-1).transpose(1, 2)

    def coarse_and_upsampled(self, image: Tensor, pixel_coords: Tensor) -> tuple[Tensor, Tensor]:
        height, width = image.shape[-2:]
        prepared = self._prepare_image(image)
        target_height = max(14, (height // 14) * 14)
        target_width = max(14, (width // 14) * 14)
        if (target_height, target_width) != (height, width):
            prepared = F.interpolate(
                prepared,
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )
        with torch.no_grad():
            coarse = self.model.model(prepared)
            upsampled = self.model.upsampler(coarse, prepared)
        return (
            self._sample(coarse, pixel_coords, (height, width)),
            self._sample(upsampled, pixel_coords, (height, width)),
        )

    def forward(self, image: Tensor, pixel_coords: Tensor) -> Tensor:
        _, upsampled = self.coarse_and_upsampled(image, pixel_coords)
        return upsampled


class DinoInstSegRGBD3D(Model):
    def __init__(
        self,
        depth_model: DinoInstSeg3D,
        rgb_fusion: Literal[
            "none",
            "raw_point",
            "raw_decode",
            "featup_decode",
            "featup_dual",
            "raw_dual",
            "dino_point",
            "dino_upsampled_point",
            "dino_token_cat",
        ] = "none",
        fusion_mode: Literal["add", "zero_add", "gate", "cat_linear"] = "gate",
        train_mode: Literal["adapter", "task_aligned", "finetune"] = "finetune",
        repo_or_dir: str = "facebookresearch/dinov2",
        backbone: str = "dinov2_vits14",
        freeze_rgb_encoder: bool = True,
        rgb_image_normalization: Literal["auto", "imagenet", "rgb"] = "auto",
        fusion_gate_init_bias: float = -10.0,
        featup_repo: str | None = None,
        featup_checkpoint: str | None = None,
        rgb_metric_checkpoint: str | None = None,
        fusion_max_ratio: float = 0.1,
        query_rgb_tokens: int = 1024,
        rgb_delivery: Literal["point", "query", "dual"] = "dual",
        rgb_residual_dropout: float = 0.0,
        depth_fusion_dropout: float = 0.0,
        rgb_token_dropout: float = 0.0,
    ):
        super().__init__()
        if rgb_fusion is None:
            rgb_fusion = "none"
        if query_rgb_tokens <= 0:
            raise ValueError("query_rgb_tokens must be positive")
        if rgb_delivery not in {"point", "query", "dual"}:
            raise ValueError("rgb_delivery must be one of: point, query, dual")
        if not 0 <= rgb_residual_dropout < 1:
            raise ValueError("rgb_residual_dropout must fall within [0, 1)")
        if not 0 <= depth_fusion_dropout < 1:
            raise ValueError("depth_fusion_dropout must fall within [0, 1)")
        if rgb_residual_dropout + depth_fusion_dropout >= 1:
            raise ValueError("RGB and depth fusion dropout probabilities must sum to less than 1")
        if not 0 <= rgb_token_dropout < 1:
            raise ValueError("rgb_token_dropout must fall within [0, 1)")
        if rgb_token_dropout > 0 and rgb_fusion != "featup_dual":
            raise ValueError("rgb_token_dropout is supported only with featup_dual fusion")
        self.depth_model = depth_model
        self.rgb_fusion = rgb_fusion
        self.fusion_mode = fusion_mode
        self.train_mode = train_mode
        self.freeze_rgb_encoder = freeze_rgb_encoder
        self.rgb_image_normalization = rgb_image_normalization
        self.fusion_gate_init_bias = fusion_gate_init_bias
        self.fusion_max_ratio = fusion_max_ratio
        self.query_rgb_tokens = query_rgb_tokens
        self.rgb_delivery = rgb_delivery
        self.rgb_residual_dropout = rgb_residual_dropout
        self.depth_fusion_dropout = depth_fusion_dropout
        self.rgb_token_dropout = rgb_token_dropout
        self.last_rgb_feature_shape: tuple[int, ...] | None = None
        # Populated by later tasks; declared here so attribute access is always safe.
        self.point_fusion: RGBPointFusion | NormBoundedPointFusion | None = None
        self.query_fusion: NormBoundedQueryFusion | None = None
        self.metric_adapter: RGBMetricAdapter | None = None
        self.rgb_provider: DinoPointFeatureProvider | FeatUpPointFeatureProvider | None = None
        self.rgb_token_proj: nn.Module | None = None
        self.mod_emb: nn.Parameter | None = None

        if rgb_fusion in {"raw_point", "raw_decode"}:
            self.point_fusion = RGBPointFusion(
                self.depth_model.n_embd,
                rgb_dim=3,
                mode=fusion_mode,
                gate_init_bias=fusion_gate_init_bias,
            )

        if rgb_fusion in {"featup_decode", "featup_dual"}:
            if featup_repo is None or featup_checkpoint is None:
                raise ValueError("featup_repo and featup_checkpoint are required for FeatUp RGB fusion")
            featup_repo = str(featup_repo)
            featup_checkpoint = str(featup_checkpoint)

        if rgb_fusion == "featup_decode":
            self.rgb_provider = FeatUpPointFeatureProvider(featup_repo, featup_checkpoint)
            self.point_fusion = RGBPointFusion(
                self.depth_model.n_embd,
                rgb_dim=self.rgb_provider.out_dim,
                mode=fusion_mode,
                gate_init_bias=fusion_gate_init_bias,
            )

        if rgb_fusion in {"featup_dual", "raw_dual"}:
            if rgb_fusion == "featup_dual":
                self.rgb_provider = FeatUpPointFeatureProvider(featup_repo, featup_checkpoint)
                adapter_input_dim = self.rgb_provider.out_dim
            else:
                adapter_input_dim = 3
            self.metric_adapter = RGBMetricAdapter(input_dim=adapter_input_dim)
            if rgb_fusion == "featup_dual" and rgb_metric_checkpoint is not None:
                checkpoint = torch.load(rgb_metric_checkpoint, map_location="cpu", weights_only=False)
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                adapter_state = {
                    str(key).removeprefix("adapter."): value
                    for key, value in state_dict.items()
                    if str(key).startswith("adapter.")
                }
                if not adapter_state:
                    raise ValueError("RGB metric checkpoint contains no adapter.* parameters")
                self.metric_adapter.load_state_dict(adapter_state, strict=True)
            if rgb_delivery in {"point", "dual"}:
                self.point_fusion = NormBoundedPointFusion(
                    self.depth_model.n_embd,
                    rgb_dim=self.metric_adapter.output_dim,
                    max_ratio=fusion_max_ratio,
                )
            if rgb_delivery in {"query", "dual"}:
                self.query_fusion = NormBoundedQueryFusion(
                    self.depth_model.n_embd,
                    rgb_dim=self.metric_adapter.output_dim,
                    num_heads=self.depth_model.n_head,
                    max_ratio=fusion_max_ratio,
                )

        if rgb_fusion == "dino_point":
            self.rgb_provider = DinoPointFeatureProvider(
                repo_or_dir=repo_or_dir,
                backbone=backbone,
                freeze=freeze_rgb_encoder,
                image_normalization=rgb_image_normalization,
            )
            self.point_fusion = RGBPointFusion(
                self.depth_model.n_embd,
                rgb_dim=self.rgb_provider.out_dim,
                mode=fusion_mode,
                gate_init_bias=fusion_gate_init_bias,
            )

        if rgb_fusion == "dino_token_cat":
            if isinstance(self.depth_model.points_dec, nn.ModuleList):
                raise ValueError("dino_token_cat requires a single decoder; point_dec='multi' is unsupported.")
            self.rgb_provider = DinoPointFeatureProvider(
                repo_or_dir=repo_or_dir,
                backbone=backbone,
                freeze=freeze_rgb_encoder,
                image_normalization=rgb_image_normalization,
            )
            self.rgb_token_proj = nn.Sequential(
                nn.Linear(self.rgb_provider.out_dim, self.depth_model.n_embd),
                nn.LayerNorm(self.depth_model.n_embd),
            )
            self.mod_emb = nn.Parameter(torch.zeros(2, self.depth_model.n_embd))

        if train_mode in {"adapter", "task_aligned"}:
            self.depth_model.requires_grad_(False)
            for name, param in self.named_parameters():
                if not name.startswith("depth_model."):
                    param.requires_grad_(True)
        if train_mode == "task_aligned":
            for module in self._task_aligned_depth_modules():
                module.requires_grad_(True)
        if self.rgb_provider is not None and self.freeze_rgb_encoder:
            self.rgb_provider.requires_grad_(False)

    @property
    def map(self):
        return self.depth_model.map

    @property
    def pq(self):
        return self.depth_model.pq

    def _task_aligned_depth_modules(self) -> list[nn.Module]:
        names = (
            "query_enc",
            "queries",
            "query_pos",
            "inputs_dec",
            "inputs_head",
            "head_norm",
            "cls_head",
            "cls_quality_head",
        )
        modules: list[nn.Module] = []
        for name in names:
            module = getattr(self.depth_model, name, None)
            if isinstance(module, nn.Module):
                modules.append(module)
        return modules

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        # A depth-only checkpoint has un-prefixed keys (encoder.*, inputs_enc.*, ...).
        # Route it into depth_model so the trained weights actually land; new RGB
        # modules keep their fresh init. A full-wrapper checkpoint (resume) has
        # depth_model.* keys and loads normally.
        if state_dict and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if state_dict and "model" in state_dict:
            state_dict = state_dict["model"]
        if state_dict and any(str(k).startswith("model.") for k in state_dict):
            state_dict = {str(k).removeprefix("model."): v for k, v in state_dict.items()}
        full_rgbd_prefixes = (
            "depth_model.",
            "rgb_provider.",
            "point_fusion.",
            "query_fusion.",
            "metric_adapter.",
            "rgb_token_proj.",
            "modality_embed",
        )
        if state_dict and not any(str(k).startswith(full_rgbd_prefixes) for k in state_dict):
            return self.depth_model.load_state_dict(state_dict, strict=strict, assign=assign)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep frozen submodules deterministic regardless of the trainer's train() call.
        if self.train_mode == "adapter":
            self.depth_model.train(False)
        elif self.train_mode == "task_aligned":
            self.depth_model.train(False)
            for module in self._task_aligned_depth_modules():
                module.train(mode)
        if self.rgb_provider is not None and self.freeze_rgb_encoder:
            self.rgb_provider.train(False)
        return self

    def _require_tensor(self, data: dict[str, Any], key: str) -> Tensor:
        value = data.get(key)
        if not torch.is_tensor(value):
            raise KeyError(key)
        return cast(Tensor, value)

    def _encode_with_point_features(self, inputs: Tensor, rgb_features: Tensor, **kwargs) -> Tensor:
        if self.point_fusion is None:
            raise RuntimeError("Point fusion module is not initialized.")
        dm = self.depth_model
        inputs_fps = furthest_point_sample(inputs, num_samples=dm.num_queries)
        q = dm.nerf_enc(inputs_fps)
        kv_depth = dm.nerf_enc(inputs)
        kv = self.point_fusion(kv_depth, rgb_features)
        if isinstance(self.point_fusion, RGBPointFusion) and self.point_fusion.last_gate_stats is not None:
            for stat_name, stat_value in self.point_fusion.last_gate_stats.items():
                self.log(f"rgb_gate/{stat_name}", stat_value, level=DEBUG_LEVEL_1)
        self.last_rgb_feature_shape = tuple(kv.shape)
        x = dm.inputs_enc(q, kv)
        x = torch.cat((dm.encoder.cls_token.expand(len(x), -1, -1), x), dim=1)
        feat = dm.encoder(x)
        if hasattr(dm, "lvl_embd"):
            feat = [
                (dm.lvl_norm(f + e), dm.lvl_norm(c + e)) for (f, c), e in zip(feat, dm.lvl_embd.weight, strict=False)
            ]
        patch_feat = feat[-1][0]
        if dm.cat_feat:
            patch_feat = torch.cat([f[0] for f in feat], dim=1)
        if dm.cls_token:
            return torch.cat((feat[-1][1].unsqueeze(1), patch_feat), dim=1)
        return patch_feat

    def _dino_tokens_from_image(self, image: Tensor) -> Tensor:
        if not isinstance(self.rgb_provider, DinoPointFeatureProvider):
            raise RuntimeError("DINO token extraction requires a DINO RGB provider.")
        if self.rgb_token_proj is None:
            raise RuntimeError("RGB token projection is not initialized.")
        patches, _, _ = self.rgb_provider._encode_patches(image)
        return self.rgb_token_proj(patches)

    def encode(self, inputs: Tensor, **kwargs) -> Tensor:
        if self.rgb_fusion in {"none", "raw_decode", "featup_decode", "featup_dual", "raw_dual"}:
            return self.depth_model.encode(inputs=inputs, **kwargs)
        if self.rgb_fusion == "raw_point":
            colors = self._require_tensor(kwargs, "inputs.colors").to(device=inputs.device, dtype=inputs.dtype)
            return self._encode_with_point_features(inputs, colors, **kwargs)
        if self.rgb_fusion == "dino_point":
            if self.rgb_provider is None:
                raise RuntimeError("DINO RGB provider is not initialized.")
            image = self._require_tensor(kwargs, "inputs.image").to(device=inputs.device, dtype=inputs.dtype)
            pixel_coords = self._require_tensor(kwargs, "inputs.pixel_coords").to(device=inputs.device)
            rgb_features = self.rgb_provider(image, pixel_coords).to(dtype=inputs.dtype)
            return self._encode_with_point_features(inputs, rgb_features, **kwargs)
        if self.rgb_fusion == "dino_token_cat":
            if isinstance(self.depth_model.points_dec, nn.ModuleList):
                raise ValueError("dino_token_cat requires a single decoder; point_dec='multi' is unsupported.")
            image = self._require_tensor(kwargs, "inputs.image").to(device=inputs.device, dtype=inputs.dtype)
            depth_tokens = self.depth_model.encode(inputs=inputs, **kwargs)
            rgb_tokens = self._dino_tokens_from_image(image).to(dtype=depth_tokens.dtype)
            self.last_rgb_feature_shape = tuple(rgb_tokens.shape)
            if self.mod_emb is not None:
                rgb_tokens = rgb_tokens + self.mod_emb[1]
            if self.depth_model.cls_token:
                cls_token = depth_tokens[:, :1]
                patch_tokens = depth_tokens[:, 1:]
                if self.mod_emb is not None:
                    patch_tokens = patch_tokens + self.mod_emb[0]
                return torch.cat((cls_token, patch_tokens, rgb_tokens), dim=1)
            if self.mod_emb is not None:
                depth_tokens = depth_tokens + self.mod_emb[0]
            return torch.cat((depth_tokens, rgb_tokens), dim=1)
        raise NotImplementedError(f"RGB fusion '{self.rgb_fusion}' is not implemented yet.")

    def forward(self, inputs: Tensor, points: Tensor | None = None, **kwargs) -> dict[str, Any]:
        feature = self.encode(inputs=inputs, **kwargs)
        _points = inputs if points is None else points
        decode_kwargs: dict[str, Any] = {}
        if self.rgb_fusion in {"raw_decode", "featup_decode", "featup_dual", "raw_dual"}:
            if self.point_fusion is None and self.query_fusion is None:
                raise RuntimeError("Decode-side RGB fusion module is not initialized.")
            if self.rgb_fusion == "raw_decode":
                rgb_features = self._require_tensor(kwargs, "inputs.colors").to(
                    device=inputs.device, dtype=inputs.dtype
                )
            elif self.rgb_fusion == "raw_dual":
                if self.metric_adapter is None:
                    raise RuntimeError("Raw RGB metric adapter is not initialized.")
                colors = self._require_tensor(kwargs, "inputs.colors").to(device=inputs.device)
                rgb_features = self.metric_adapter(colors.float()).to(dtype=inputs.dtype)
            else:
                if not isinstance(self.rgb_provider, FeatUpPointFeatureProvider):
                    raise RuntimeError("FeatUp RGB provider is not initialized.")
                image = self._require_tensor(kwargs, "inputs.image").to(device=inputs.device)
                pixel_coords = self._require_tensor(kwargs, "inputs.pixel_coords").to(device=inputs.device)
                rgb_features = self.rgb_provider(image, pixel_coords).to(dtype=inputs.dtype)
                if self.rgb_fusion == "featup_dual":
                    if self.metric_adapter is None:
                        raise RuntimeError("Task-aligned RGB modules are not initialized.")
                    rgb_features = self.metric_adapter(rgb_features.float()).to(dtype=inputs.dtype)
            rgb_keep_mask = None
            depth_keep_mask = None
            if self.rgb_fusion in {"featup_dual", "raw_dual"}:
                if self.training and self.rgb_residual_dropout + self.depth_fusion_dropout > 0:
                    rgb_keep_mask, depth_keep_mask = _fusion_keep_masks(
                        torch.rand(inputs.size(0), device=inputs.device),
                        self.rgb_residual_dropout,
                        self.depth_fusion_dropout,
                    )
                    self.log(
                        "rgb_residual/keep_fraction",
                        float(rgb_keep_mask.float().mean().item()),
                        level=DEBUG_LEVEL_1,
                    )
                    self.log(
                        "depth_fusion/keep_fraction",
                        float(depth_keep_mask.float().mean().item()),
                        level=DEBUG_LEVEL_1,
                    )
            if self.training and self.rgb_token_dropout > 0:
                rgb_features, token_keep_mask = _drop_rgb_feature_tokens(
                    rgb_features,
                    self.rgb_token_dropout,
                    force_keep_samples=None if depth_keep_mask is None else ~depth_keep_mask,
                )
                self.log(
                    "rgb_token/keep_fraction",
                    float(token_keep_mask.float().mean().item()),
                    level=DEBUG_LEVEL_1,
                )
            self.last_rgb_feature_shape = tuple(rgb_features.shape)
            if self.point_fusion is not None:
                decode_kwargs.update(
                    {
                        "inputs_aux_feat": rgb_features,
                        "inputs_feat_fusion": self.point_fusion,
                    }
                )
            if self.query_fusion is not None:
                num_tokens = min(self.query_rgb_tokens, inputs.size(1))
                indices = cast(
                    Tensor,
                    furthest_point_sample(inputs, num_samples=num_tokens, return_indices=True),
                )
                query_rgb = torch.gather(
                    rgb_features,
                    dim=1,
                    index=indices.unsqueeze(-1).expand(-1, -1, rgb_features.size(-1)),
                )
                decode_kwargs.update(
                    {
                        "query_aux_feat": query_rgb,
                        "query_feat_fusion": self.query_fusion,
                    }
                )
            if self.rgb_fusion in {"featup_dual", "raw_dual"}:
                if self.point_fusion is not None:
                    if not isinstance(self.point_fusion, NormBoundedPointFusion):
                        raise RuntimeError("Task-aligned point fusion module is invalid.")
                    self.point_fusion.sample_mask = rgb_keep_mask
                    self.point_fusion.depth_sample_mask = depth_keep_mask
                if self.query_fusion is not None:
                    self.query_fusion.sample_mask = rgb_keep_mask
                    self.query_fusion.depth_sample_mask = depth_keep_mask
        out = self.depth_model.decode(points=_points, feature=feature, inputs=inputs, **decode_kwargs, **kwargs)
        gate_stats = None if self.point_fusion is None else getattr(self.point_fusion, "last_gate_stats", None)
        if gate_stats is not None:
            for stat_name, stat_value in gate_stats.items():
                self.log(f"rgb_gate/{stat_name}", stat_value, level=DEBUG_LEVEL_1)
        point_ratio_stats = None if self.point_fusion is None else getattr(self.point_fusion, "last_ratio_stats", None)
        if point_ratio_stats is not None:
            for stat_name, stat_value in point_ratio_stats.items():
                self.log(f"rgb_residual/point_{stat_name}", stat_value, level=DEBUG_LEVEL_1)
        if self.query_fusion is not None and self.query_fusion.last_ratio_stats is not None:
            for stat_name, stat_value in self.query_fusion.last_ratio_stats.items():
                self.log(f"rgb_residual/query_{stat_name}", stat_value, level=DEBUG_LEVEL_1)
        return out

    def loss(self, data: dict[str, Any], **kwargs) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        out = self(**{**data, **kwargs})
        return self.depth_model.loss(data={**data, **out}, **kwargs)

    def predict(
        self,
        inputs: Tensor | None = None,
        points: Tensor | None = None,
        feature: Tensor | list[Tensor] | dict[str, Tensor] | None = None,
        data: dict[str, Any] | None = None,
        **kwargs,
    ):
        if data is not None and any("logits" in key for key in data.keys()):
            return self.depth_model.predict(inputs=inputs, points=points, feature=feature, data=data, **kwargs)
        if feature is None:
            if inputs is None:
                if data is None or not torch.is_tensor(data.get("inputs")):
                    raise ValueError("inputs is required when precomputed logits or feature are not provided.")
                inputs = cast(Tensor, data["inputs"])
            feature = self.encode(inputs=inputs, **kwargs)
        return self.depth_model.predict(inputs=inputs, points=points, feature=feature, data=data, **kwargs)

    @torch.inference_mode()
    def evaluate(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        out = self(**{**data, **kwargs})
        return self.depth_model.evaluate({**data, **out}, **kwargs)

    def get_log(self, key: str | None = None):
        if key is not None:
            try:
                return super().get_log(key)
            except KeyError:
                return self.depth_model.get_log(key)
        logs = dict(cast(dict[str, Any], self.depth_model.get_log()))
        logs.update(cast(dict[str, Any], super().get_log()))
        return logs

    def clear_log(self, key: str | None = None, ema: bool = False):
        super().clear_log(key, ema=ema)
        self.depth_model.clear_log(key, ema=ema)

    def optimizer(
        self,
        lr: float,
        weight_decay: float = 0.01,
        factor: float = 0.1,
        fused: bool = False,
        **kwargs,
    ) -> torch.optim.Optimizer:
        depth_param_ids = {id(param) for param in self.depth_model.parameters() if param.requires_grad}
        depth_params: list[nn.Parameter] = []
        new_params: list[nn.Parameter] = []
        for param in self.parameters():
            if not param.requires_grad:
                continue
            (depth_params if id(param) in depth_param_ids else new_params).append(param)

        groups: list[dict[str, Any]] = []
        if depth_params:
            groups.append({"params": depth_params, "lr": lr * factor, "weight_decay": weight_decay})
        if new_params:
            groups.append({"params": new_params, "lr": lr, "weight_decay": weight_decay})
        if not groups:
            raise ValueError("No trainable parameters in RGB-D wrapper.")
        return torch.optim.AdamW(groups, lr=lr, fused=fused)

    @torch.no_grad()
    def on_validation_epoch_end(self, *args, **kwargs) -> dict[str, float]:
        return self.depth_model.on_validation_epoch_end(*args, **kwargs)


class DinoInst3D(DinoInstSeg):
    def __init__(
        self,
        num_objs: int = 100,
        head: Literal["grid", "cross-attn"] = "grid",
        freeze: bool = False,
        repo_or_dir: str = "facebookresearch/dinov2",
        backbone: str = "dinov2_vits14",
        bias: bool = True,
        dropout: float = 0.0,
        n_dec_layers: int | None = None,
        mlp_heads: bool = False,
        match_cls: bool = True,
        match_3d: bool = True,
        separate_match_3d: bool = False,
        pred_2d: bool = True,
        sample: bool = False,
        nerf_enc: Literal["tcnn", "torch"] = "tcnn" if TCNN_EXISTS else "torch",
        nerf_freqs: int = 6,
        mask_weight: float = 1.0,
        occ_weight: float = 1.0,
        cls_weight: float = 1.0,
        aux_weight: float = 1.0,
        loss_weight_2d: float = 1.0,
        loss_weight_3d: float = 1.0,
        learn_loss_weights: bool = False,
    ):
        super().__init__(
            num_objs=num_objs,
            freeze=freeze,
            repo_or_dir=repo_or_dir,
            backbone=backbone,
            bias=bias,
            dropout=dropout,
            num_dec_layers=n_dec_layers,
            mlp_heads=mlp_heads,
            match_cls=match_cls,
            sample=sample,
            mask_weight=mask_weight,
            cls_weight=cls_weight,
            aux_weight=aux_weight,
            learn_loss_weights=learn_loss_weights,
        )

        self.match_3d = match_3d
        self.separate_match_3d = separate_match_3d
        self.occ_weight = occ_weight
        self.loss_weight_2d = loss_weight_2d
        self.loss_weight_3d = loss_weight_3d

        if not pred_2d:
            self.head = None

        self.points_enc = NeRFEncoding(
            out_dim=self.n_embd,
            implementation=nerf_enc,
            num_frequencies=nerf_freqs,
            max_freq_exp=nerf_freqs - 1,
            normalize_inputs=False,
            scale_inputs=False,
        )
        self.grid = None
        if "grid" in head:
            self.grid = GridDecoder(c_dim=self.n_embd, align_corners=False)
        self.cross_attn = None
        if "cross-attn" in head:
            self.cross_attn = DecoderBlock(n_embd=self.n_embd, n_head=self.n_head, bias=bias, dropout=dropout, chunks=2)
        self.head_3d = nn.Sequential(self.head_norm, nn.Linear(self.n_embd, self.n_embd))
        if mlp_heads:
            self.head_3d = nn.Sequential(self.head_norm, MLP(self.n_embd, self.n_embd))
        """
        self.head_3d = ResNetGridDecoder(dim=self.n_embd,
                                         c_dim=self.n_embd,
                                         hidden_dim=self.n_embd,
                                         n_blocks=5,
                                         condition="add",
                                         sample=False)
        self.head_3d.fc_p = None
        """

        if learn_loss_weights:
            self.log_var_occ = nn.Parameter(torch.zeros(()))
            if pred_2d:
                self.log_var_2d = nn.Parameter(torch.zeros(()))
                self.log_var_3d = nn.Parameter(torch.zeros(()))

    @staticmethod
    @torch.no_grad()
    def _masks_from_labels(labels: Tensor, masks: Tensor | list[Tensor]) -> list[Tensor]:
        num_masks_2d = [len(m) for m in masks]
        out: list[Tensor] = []
        for label_tensor, k2d in zip(labels, num_masks_2d, strict=False):
            n = label_tensor.numel()
            if n == 0 or k2d == 0:
                out.append(label_tensor.new_zeros((k2d, n), dtype=torch.bool))
                continue
            # Rows correspond to instance IDs 1..k2d
            ids = torch.arange(1, k2d + 1, device=label_tensor.device).unsqueeze(1)  # (k2d, 1)
            # Compare every point label to each id; background (0) and >k2d become all False
            m3 = label_tensor.unsqueeze(0) == ids  # (k2d, n) boolean
            out.append(m3)
        return out

    def decode(
        self,
        points: Tensor | list[Tensor],
        feature: Any,
        **kwargs,
    ) -> dict[str, Any]:
        patch_feat, cls_feat = feature[-1]
        t = self.num_objs
        s = self.encoder.patch_size
        b, _c, j, i = patch_feat.size()
        size = j * s, i * s

        patch_feat = rearrange(patch_feat, "b c j i -> b (j i) c")
        if self.dpt_head is None:
            raise RuntimeError("DPT head is required for DinoInst3D decoding.")
        feat_hd = self.dpt_head(feature)

        points_feat = 0
        if torch.is_tensor(points) and points.ndim == 3:
            if self.points_enc is not None:
                points_feat += self.points_enc(points)
            if self.grid is not None:
                points_feat += self.grid.sample_feature(points, {"uv": feat_hd}, **kwargs)[0].transpose(1, 2)
            if self.cross_attn is not None:
                points_feat = self.cross_attn(points_feat, torch.cat((cls_feat.unsqueeze(1), patch_feat), dim=1))
            else:
                points_feat += cls_feat.unsqueeze(1)
        else:
            points_feat = points

        query_pos = self.query_pos.weight.unsqueeze(0).expand(b, -1, -1)
        x = torch.zeros_like(query_pos)

        x = self.queries(patch_feat)
        x_select = self.cls_head(x).squeeze(-1)
        x_idx = torch.topk(x_select, k=t, dim=1)[1]
        x = torch.gather(x, dim=1, index=x_idx.unsqueeze(-1).expand_as(query_pos))

        aux_out = dict(cls_logits=self.cls_head(x).squeeze(-1))
        if self.head is not None:
            aux_out["logits"] = self._forward_mask_head(x, feat_hd, patch_feat, size)
        if torch.is_tensor(points_feat):
            aux_out["occ_logits"] = self._forward_head_3d(x, points_feat)
        aux_out_list = [aux_out]
        x = x.detach()

        k_cross = self._add_pos(patch_feat, size, cls_feat)
        v_cross = torch.cat((cls_feat.unsqueeze(1), patch_feat), dim=1)
        for layer_idx, layer in enumerate(cast(Any, self.query_enc).layers):
            layer = cast(Any, layer)
            # self-attention
            qk = v = layer.ln_1(x)
            q = k = qk + query_pos
            x = x + layer.dp_1(layer.self_attn(q, k, v))

            # cross-attention
            q_cross = layer.ln_2(x) + query_pos
            x = x + layer.dp_2(layer.cross_attn(q_cross, k_cross, v_cross))

            # projection
            x = x + layer.dp_3(layer.projection(layer.ln_3(x)))

            if layer_idx < len(self.query_enc.layers) - 1:
                aux_out = dict(cls_logits=self.cls_head(x).squeeze(-1))
                if self.head is not None:
                    aux_out["logits"] = self._forward_mask_head(x, feat_hd, cast(Any, patch_feat), size)
                if torch.is_tensor(points_feat):
                    aux_out["occ_logits"] = self._forward_head_3d(x, points_feat)
                aux_out_list.append(aux_out)

        out = {
            "cls_logits": self.cls_head(x).squeeze(2),
            "queries": x,
        }
        if self.head is not None:
            out["logits"] = self._forward_mask_head(x, feat_hd, cast(Any, patch_feat), size)
        if torch.is_tensor(points_feat) and points_feat.ndim == 3:
            out["occ_logits"] = self._forward_head_3d(x, points_feat)
        if aux_out_list:
            out["aux_out"] = aux_out_list
        return out

    def _forward_head_3d(self, queries: Tensor, points_feat: Tensor) -> Tensor:
        return torch.einsum("b t c, b n c -> b t n", self.head_3d(queries), points_feat)

    def _get_loss_3d(
        self,
        logits_3d: Tensor,
        masks_3d: Any,
        cls_logits: Tensor | None = None,
        match: list[tuple[Tensor, Tensor]] | None = None,
        reduction: str | None = "mean",
        name: str = "dice+bce",
        log: bool = True,
        return_match: bool = False,
        **kwargs,
    ) -> Tensor | tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        reduction_val = reduction or "mean"
        masks_3d_list = masks_3d if isinstance(masks_3d, list) else [masks_3d[i] for i in range(masks_3d.size(0))]
        if match is None:
            cls_preds = cls_probs_from_logits(logits_3d) if cls_logits is None else cls_logits.sigmoid()
            match, cost = hungarian_matcher(
                batch_size=len(masks_3d_list),
                mask_logits=logits_3d,
                mask_tgt=masks_3d_list,
                cls_preds=cls_preds if self.match_cls else None,
                mask_weight=self.mask_weight,
                cls_weight=self.cls_weight,
            )
            if log:
                self.log("cost_3d", cost, level=DEBUG_LEVEL_2)

        pred_masks = logits_3d
        tgt_masks = cast(Tensor, masks_3d)
        batch_idx, src_idx = index_from_match(match)
        if not torch.is_tensor(masks_3d):
            pred_masks = logits_3d[(batch_idx, src_idx)]
            tgt_masks = torch.cat([masks_3d_list[i][j] for i, (_, j) in enumerate(match)], dim=0).type_as(pred_masks)

        loss_total = pred_masks.new_tensor(0.0)
        loss_occ = pred_masks.new_tensor(0.0)
        if "dice" in name:
            loss_d = dice_loss(pred_masks, tgt_masks, reduction=reduction_val)
            if log:
                self.log("dice_loss_3d", cast(Any, loss_d), level=DEBUG_LEVEL_1)
            loss_occ += loss_d
        if "focal" in name:
            loss_f = focal_loss.sigmoid_focal_loss(
                pred_masks, tgt_masks, cast(float, self.alpha), self.gamma, reduction=reduction_val
            )
            if log:
                self.log("focal_loss_3d", cast(Any, loss_f), level=DEBUG_LEVEL_1)
            loss_occ += loss_f
        elif "bce" in name:
            loss_b = F.binary_cross_entropy_with_logits(pred_masks, tgt_masks, reduction=reduction_val)
            if log:
                self.log("bce_loss_3d", cast(Any, loss_b), level=DEBUG_LEVEL_1)
            loss_occ += loss_b

        if self.learn_loss_weights:
            loss_occ = loss_occ * torch.exp(-self.log_var_occ) + self.log_var_occ
            if log:
                self.log("occ_weight", torch.exp(-self.log_var_occ).item(), level=DEBUG_LEVEL_2)
        loss_total += self.occ_weight * loss_occ

        if cls_logits is not None:
            tgt_classes = torch.zeros_like(cls_logits)
            tgt_classes[batch_idx, src_idx] = 1.0

            if "focal" in name:
                alpha = self.alpha
                if "auto" in name and tgt_classes.any():
                    alpha = float((1.0 - (tgt_classes.sum() / tgt_classes.numel())).item())
                loss_cls = focal_loss.sigmoid_focal_loss(
                    cls_logits, tgt_classes, alpha, self.gamma, reduction=reduction_val
                )
            else:
                pos_weight = torch.tensor(1.0, device=cls_logits.device, dtype=cls_logits.dtype)
                if "auto" in name and tgt_classes.any():
                    pos_weight = (tgt_classes.numel() / (tgt_classes.sum().clamp_min(1.0))).to(cls_logits)
                loss_cls = F.binary_cross_entropy_with_logits(
                    cls_logits, tgt_classes, pos_weight=pos_weight, reduction=reduction_val
                )

            if log:
                self.log("cls_loss", cast(Any, loss_cls), level=DEBUG_LEVEL_1)

            if self.learn_loss_weights:
                loss_cls = loss_cls * torch.exp(-self.log_var_cls) + self.log_var_cls
                if log:
                    self.log("cls_weight", torch.exp(-self.log_var_cls).item(), level=DEBUG_LEVEL_2)
            loss_total += self.cls_weight * loss_cls

        if log and logger.isEnabledFor(DEBUG_LEVEL_2):
            train_data = {"loss": loss_total, "logits": pred_masks, "points.occ": tgt_masks}
            train_metrics = MultiEvalMixin.evaluate(self, train_data, prefix="train/", **kwargs)
            train_metrics.pop("train/loss")
            self.log_dict({f"{k}_3d": v for k, v in train_metrics.items()}, level=DEBUG_LEVEL_2, train_only=True)

        if return_match:
            return loss_total, match
        return loss_total

    def on_validation_epoch_end(self, *args, **kwargs) -> dict[str, float] | None:
        if self.head is not None:
            return super().on_validation_epoch_end(*args, **kwargs)

    @torch.inference_mode()
    def predict(
        self,
        inputs: Tensor | None = None,
        points: Tensor | None = None,
        feature: Any | None = None,
        key: str | None = "auto",
        points_batch_size: int | None = None,
        **kwargs,
    ) -> Tensor | list[Tensor]:
        if key is None or key == "auto":
            key = "occ_logits"
        if feature is None:
            if inputs is None:
                raise ValueError("inputs is required when feature is not provided.")
            feature = self.encode(inputs, **kwargs)

        decode = partial(self.decode, feature=feature, **kwargs)

        def pred_fn(p: Tensor) -> dict[str, Tensor]:
            return {k: v for k, v in decode(p).items() if k in [key, "cls_logits"]}

        if points_batch_size is None:
            if points is None:
                raise ValueError("points is required when points_batch_size is None.")
            out = pred_fn(points)
        else:
            if points is None:
                raise ValueError("points is required for batched prediction.")
            out = [pred_fn(p) for p in torch.split(points, points_batch_size, dim=1)]
            out = {key: torch.cat([o[key] for o in out], dim=2), "cls_logits": out[0]["cls_logits"]}

        occ_logits = out[key]
        cls_logits = out.get("cls_logits")
        cls_preds = cls_probs_from_logits(occ_logits) if cls_logits is None else cls_logits.sigmoid()
        mask = cls_preds > 0.5
        if len(occ_logits) == 1:
            return occ_logits[mask]
        return [occ_logits[i][mask[i]] for i in range(len(occ_logits))]

    @torch.inference_mode()
    def evaluate(
        self,
        data: dict[str, Any],
        points_batch_size: int | None = None,
        show: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        feature = self.encode(**{**data, **kwargs})

        points = data.pop("points")
        if not torch.is_tensor(points):
            raise TypeError("Expected tensor at data['points'].")
        decode_keys = ["logits", "cls_logits", "occ_logits", "queries"]
        decode = partial(self.decode, feature=feature, **{**data, **kwargs})

        def pred_fn(p: Tensor) -> dict[str, Tensor]:
            return {k: v for k, v in decode(p).items() if k in decode_keys}

        if not torch.is_tensor(points) or points_batch_size is None or points.size(1) <= points_batch_size:
            out = pred_fn(points)
        else:
            out_list = [pred_fn(p) for p in torch.split(points, points_batch_size, dim=1)]
            out = dict(
                logits=out_list[0].get("logits"),
                cls_logits=out_list[0].get("cls_logits"),
                queries=out_list[0].get("queries"),
            )
            out.update({"occ_logits": torch.cat([o["occ_logits"] for o in out_list], dim=2)})

        logits = cast(Tensor | None, out.get("logits"))
        cls_logits = cast(Tensor | None, out.get("cls_logits"))
        logits_3d = cast(Tensor | None, out.get("occ_logits"))

        masks = cast(list[Tensor], data["inputs.masks"])
        masks_3d_raw = data["points.occ"]
        if torch.is_tensor(masks_3d_raw):
            if masks_3d_raw.ndim != 3:
                raise ValueError("Expected 3D tensor at data['points.occ'].")
            masks_3d: list[Tensor] = [masks_3d_raw[i] for i in range(masks_3d_raw.size(0))]
        else:
            masks_3d = cast(list[Tensor], masks_3d_raw)
        if "points.labels" in data:
            masks_3d = self._masks_from_labels(cast(Tensor, data["points.labels"]), masks)

        val_metrics = dict()
        if logits is None:
            if logits_3d is None:
                raise ValueError("occ_logits are required when logits are unavailable.")
            cls_preds = cls_probs_from_logits(logits_3d) if cls_logits is None else cls_logits.sigmoid()
            match, cost = hungarian_matcher(
                batch_size=len(masks_3d),
                mask_logits=logits_3d,
                mask_tgt=masks_3d,
                cls_preds=cls_preds if self.match_cls else None,
                mask_weight=self.mask_weight,
                cls_weight=self.cls_weight,
            )
            self.log("cost_3d", cost, level=DEBUG_LEVEL_2)

            batch_idx, src_idx = index_from_match(match)
            pred_masks = logits_3d[(batch_idx, src_idx)]
            tgt_masks = torch.cat([masks_3d[i][j] for i, (_, j) in enumerate(match)])
        else:
            cls_preds = cls_probs_from_logits(logits) if cls_logits is None else cls_logits.sigmoid()
            self._update_metrics(logits, masks, cls_preds)
            match, cost = hungarian_matcher(
                batch_size=len(masks),
                mask_logits=logits,
                mask_tgt=masks,
                occ_logits=logits_3d if self.match_3d and not self.separate_match_3d else None,
                occ_tgt=masks_3d if self.match_3d and not self.separate_match_3d else None,
                cls_preds=cls_preds if self.match_cls else None,
                mask_weight=self.mask_weight,
                occ_weight=self.occ_weight,
                cls_weight=self.cls_weight,
            )
            self.log("cost", cost, level=DEBUG_LEVEL_2)

            batch_idx, src_idx = index_from_match(match)
            pred_masks = logits[(batch_idx, src_idx)]
            tgt_masks = torch.cat([masks[i][j] for i, (_, j) in enumerate(match)])

            val_data = {
                "loss": dice_loss(pred_masks, tgt_masks),
                "logits": pred_masks.flatten(1),
                "points.occ": tgt_masks.flatten(1),
            }

            if not any(len(t) for t in masks):  # If all masks are empty
                val_metrics.update(EMPTY_EVAL_RESULTS_DICT)
                val_metrics["loss"] = val_data["loss"].cpu().item()
            else:
                val_metrics.update(MultiEvalMixin.evaluate(self, val_data, **kwargs))

        if logits_3d is None:
            queries = cast(Tensor | None, out.get("queries"))
            if queries is None:
                raise ValueError("queries are required when occ_logits are unavailable.")
            points = torch.cat([points[i][j] for i, (_, j) in enumerate(match)])
            queries = queries[(batch_idx, src_idx)]

            def pred_occ_fn(p: Tensor) -> Tensor:
                return self.head_3d(self.points_enc(p), queries)

            if points_batch_size is None:
                pred_masks_3d = pred_occ_fn(points)
            else:
                pred_masks_3d = torch.cat(
                    [pred_occ_fn(p) for p in torch.split(points, points_batch_size, dim=1)], dim=1
                )
        else:
            if self.match_3d and self.separate_match_3d:
                cls_preds = cls_probs_from_logits(logits_3d) if cls_logits is None else cls_logits.sigmoid()
                match, cost = hungarian_matcher(
                    batch_size=len(masks_3d),
                    mask_logits=logits_3d,
                    mask_tgt=masks_3d,
                    cls_preds=cls_preds if self.head is None else None,
                    mask_weight=self.mask_weight,
                    cls_weight=self.cls_weight,
                )
                self.log("cost_3d", cost, level=DEBUG_LEVEL_2)

                batch_idx, src_idx = index_from_match(match)

            pred_masks_3d = logits_3d[(batch_idx, src_idx)]
        tgt_masks_3d = torch.cat([masks_3d[i][j] for i, (_, j) in enumerate(match)])
        val_data = {"loss": dice_loss(pred_masks_3d, tgt_masks_3d), "logits": pred_masks_3d, "points.occ": tgt_masks_3d}

        if not any(len(t) for t in masks):
            val_metrics.update({f"{k}_3d": v for k, v in EMPTY_EVAL_RESULTS_DICT.items()})
            val_metrics["loss_3d"] = val_data["loss"].cpu().item()
        else:
            val_metrics.update({f"{k}_3d": v for k, v in MultiEvalMixin.evaluate(self, val_data, **kwargs).items()})

        if show and logits is not None and len(logits) == 1:
            points = points if len(points) == len(tgt_masks_3d) else [points[0]] * len(tgt_masks_3d)
            image = data.get("inputs.images", data["inputs"])[0]
            # show_image_with_masks(image=image, masks=logits[cls_preds > 0.5] > 0)
            show_image_with_masks(image=image, masks=pred_masks > 0)
            geometries = []
            for i in range(len(tgt_masks_3d)):
                m_pred = pred_masks_3d[i] > 0
                m_target = tgt_masks_3d[i] > 0
                p = points[i]
                if m_pred.any():
                    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p[m_pred].cpu().numpy()))
                    geometries.append(pcd.paint_uniform_color(np.random.rand(3)))
                if m_target.any():
                    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p[m_target].cpu().numpy()))
                    geometries.append(pcd.paint_uniform_color([0.7, 0.7, 0.7]))
            if geometries:
                cast(Any, o3d).visualization.draw_geometries(geometries)

        return val_metrics

    def loss(
        self,
        data: dict[str, Any],
        name: str | None = None,
        reduction: str | None = "mean",
        points_batch_size: int | None = None,
        log_freq: int = 10,
        global_step: int | None = None,
        **kwargs,
    ) -> Tensor:
        out = self(**{**data, **kwargs})
        logits = cast(Tensor | None, out.get("logits"))
        cls_logits = cast(Tensor | None, out.get("cls_logits"))
        logits_3d = cast(Tensor | None, out.get("occ_logits"))

        masks = cast(list[Tensor], data["inputs.masks"])
        masks_3d_raw = data["points.occ"]
        if torch.is_tensor(masks_3d_raw):
            if masks_3d_raw.ndim != 3:
                raise ValueError("Expected 3D tensor at data['points.occ'].")
            masks_3d: list[Tensor] = [masks_3d_raw[i] for i in range(masks_3d_raw.size(0))]
        else:
            masks_3d = cast(list[Tensor], masks_3d_raw)
        if "points.labels" in data:
            masks_3d = self._masks_from_labels(cast(Tensor, data["points.labels"]), masks)

        loss_2d = (
            logits_3d.new_tensor(0.0)
            if logits_3d is not None
            else torch.tensor(0.0, device=cls_logits.device if cls_logits is not None else None)
        )
        masks_3d_for_loss: Any = masks_3d
        if logits is None:
            if logits_3d is None:
                raise ValueError("Either logits or occ_logits must be available.")
            cls_preds = cls_probs_from_logits(logits_3d) if cls_logits is None else cls_logits.sigmoid()
            match, cost = hungarian_matcher(
                batch_size=len(masks_3d),
                mask_logits=logits_3d,
                mask_tgt=masks_3d,
                cls_preds=cls_preds if self.match_cls else None,
                mask_weight=self.mask_weight,
                cls_weight=self.cls_weight,
            )
            self.log("cost_3d", cost, level=DEBUG_LEVEL_2)
        else:
            cls_preds = cls_probs_from_logits(logits) if cls_logits is None else cls_logits.sigmoid()
            match, cost = hungarian_matcher(
                batch_size=len(masks),
                mask_logits=logits,
                mask_tgt=masks,
                cls_preds=cls_preds if self.match_cls else None,
                occ_logits=logits_3d if self.match_3d and not self.separate_match_3d else None,
                occ_tgt=masks_3d if self.match_3d and not self.separate_match_3d else None,
                mask_weight=self.mask_weight,
                occ_weight=self.occ_weight,
                cls_weight=self.cls_weight,
                sample=112**2 if self.sample else None,
            )
            loss_2d = cast(
                Tensor,
                self._get_loss(
                    mask_logits=logits,
                    mask_gt=masks,
                    cls_logits=cls_logits,
                    match=match,
                    name=name or self.loss_name,
                    reduction=reduction,
                    log=True if global_step is None else global_step % log_freq == 0,
                    **kwargs,
                ),
            )
            self.log("cost", cost, level=DEBUG_LEVEL_2)

        if logits_3d is None:
            batch_idx, src_idx = index_from_match(match)
            queries = cast(Tensor, out["queries"])
            queries = queries[(batch_idx, src_idx)]

            points = data["points"]
            bs = len(points)
            points = torch.cat([points[i][j] for i, (_, j) in enumerate(match)])
            masks_3d_for_loss = torch.cat([masks_3d[i][j] for i, (_, j) in enumerate(match)])

            def pred_fn(p: Tensor, q: Tensor) -> Tensor:
                return self.head_3d(self.points_enc(p), q)

            if points.size(0) <= bs:
                logits_3d = pred_fn(points, queries)
            else:
                logits_3d = torch.cat(
                    [pred_fn(p, q) for p, q in zip(torch.split(points, bs), torch.split(queries, bs), strict=False)]
                )

        loss_3d, match_3d = cast(
            tuple[Tensor, list[tuple[Tensor, Tensor]]],
            self._get_loss_3d(
                logits_3d=logits_3d,
                masks_3d=masks_3d_for_loss,
                cls_logits=cls_logits if self.head is None else None,
                match=None if self.separate_match_3d else match,
                name=name or self.loss_name,
                reduction=reduction,
                log=True if global_step is None else global_step % log_freq == 0,
                return_match=True,
                **kwargs,
            ),
        )

        aux_out = cast(list[dict[str, Any]], out.get("aux_out", []))
        aux_loss = logits_3d.new_tensor(0.0) if logits_3d is not None else torch.tensor(0.0, device=loss_2d.device)
        aux_loss_3d = logits_3d.new_tensor(0.0) if logits_3d is not None else torch.tensor(0.0, device=loss_2d.device)
        for aux in aux_out:
            if not self.global_match:
                match = None
                if "logits" in aux and "occ_logits" in aux:
                    match, cost = hungarian_matcher(
                        batch_size=len(masks),
                        mask_logits=aux["logits"],
                        mask_tgt=masks,
                        cls_preds=cls_preds if self.match_cls else None,
                        occ_logits=aux["occ_logits"] if self.match_3d and not self.separate_match_3d else None,
                        occ_tgt=masks_3d if self.match_3d and not self.separate_match_3d else None,
                        mask_weight=self.mask_weight,
                        occ_weight=self.occ_weight,
                        cls_weight=self.cls_weight,
                        sample=112**2 if self.sample else None,
                    )

            if "logits" in aux:
                aux_loss += cast(
                    Tensor,
                    self._get_loss(
                        mask_logits=aux["logits"],
                        mask_gt=masks,
                        cls_logits=aux.get("cls_logits"),
                        match=match,
                        name=name or self.loss_name,
                        reduction=reduction,
                        log=False,
                        **kwargs,
                    ),
                )
            if "occ_logits" in aux:
                if self.separate_match_3d:
                    match = None
                    if self.global_match:
                        match = match_3d
                aux_loss_3d += cast(
                    Tensor,
                    self._get_loss_3d(
                        logits_3d=aux["occ_logits"],
                        masks_3d=masks_3d,
                        cls_logits=aux.get("cls_logits") if self.head is None else None,
                        match=match,
                        name=name or self.loss_name,
                        reduction=reduction,
                        log=False,
                        **kwargs,
                    ),
                )

        if aux_loss:
            self.log("aux_loss", cast(Any, aux_loss), level=DEBUG_LEVEL_1)
            loss_2d += aux_loss
        if aux_loss_3d:
            self.log("aux_loss_3d", cast(Any, aux_loss_3d), level=DEBUG_LEVEL_1)
            loss_3d += aux_loss_3d

        if self.learn_loss_weights and loss_2d:
            loss_2d = loss_2d * torch.exp(-self.log_var_2d) + self.log_var_2d
            self.log("loss_weight_2d", torch.exp(-self.log_var_2d).item(), level=DEBUG_LEVEL_2)
            loss_3d = loss_3d * torch.exp(-self.log_var_3d) + self.log_var_3d
            self.log("loss_weight_3d", torch.exp(-self.log_var_3d).item(), level=DEBUG_LEVEL_2)

        loss = self.loss_weight_2d * loss_2d + self.loss_weight_3d * loss_3d

        return loss


class InstOccPipeline(Model):
    def __init__(self, inst: DinoInstSeg3D, occ: Dino3D, masks: Literal["inst", "gt"] = "inst"):
        super().__init__()
        mask_mode = masks.lower()
        if mask_mode not in {"inst", "gt"}:
            raise ValueError(f"Invalid mask_source '{mask_mode}'. Expected 'inst' or 'gt'.")

        self.inst = inst
        self.occ = occ
        self.masks = cast(Literal["inst", "gt"], mask_mode)
        self.inst.eval()
        self.occ.eval()

        if mask_mode == "inst":
            self.inst.multitask = True
            self.inst.map_inputs = MeanAveragePrecision3D()
            self.inst.pq_inputs = PanopticQuality3D()

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass

        for obj in (self.inst, self.occ):
            try:
                return getattr(obj, name)
            except AttributeError:
                continue

        raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

    def forward(
        self,
        inputs: Tensor,
        points: Tensor,
        masks: list[Tensor] | None = None,
        threshold: float = 0.5,
        to_cpu: bool = False,
        **kwargs,
    ) -> dict[str, Tensor]:
        in_logits = None
        inst_scores_cls: list[Tensor] | None = None
        inst_scores_qual: list[Tensor] | None = None
        if masks is None:
            pred_out = self.inst.predict(
                inputs=inputs,
                method="cls",
                threshold=threshold,
                return_logits=True,
                align_to_gt=False,
                show=False,
                **filter_dict(kwargs, remove={"inputs", "method", "threshold", "return_logits", "align_to_gt", "show"}),
            )
            if not isinstance(pred_out, tuple):
                raise TypeError("Expected tuple output from instance predictor when return_logits=True.")
            masks, inst_logits = pred_out
            if isinstance(inst_logits, dict):
                in_logits = inst_logits.get("logits")
                inst_scores_cls = inst_logits.get("cls_logits")
                inst_scores_qual = inst_logits.get("cls_quality")
        if masks is None:
            raise ValueError("Instance masks are required.")

        B = len(masks)
        Q = int(self.inst.num_objs)
        M = points.size(1)

        neg_inf = -1e4
        log_threshold = math.log(threshold / (1.0 - threshold))
        logits_all: list[Tensor] = []
        in_logits_all: list[Tensor] = []
        cls_logits_all: list[Tensor] = []
        cls_quality_all: list[Tensor] = []

        has_logits = in_logits is not None
        has_cls = inst_scores_cls is not None
        has_qual = inst_scores_qual is not None

        for b in range(B):
            inst_masks = masks[b]

            logits_b = torch.full((Q, M), neg_inf, device=points.device, dtype=points.dtype)

            in_logits_b = None
            src_in: Tensor | None = None
            if has_logits and in_logits is not None:
                src_in = in_logits[b]
                in_logits_b = torch.full((Q, src_in.size(1)), neg_inf, device=src_in.device, dtype=src_in.dtype)

            cls_logits_b = torch.full((Q,), neg_inf, device=points.device, dtype=points.dtype) if has_cls else None
            cls_quality_b = torch.full((Q,), neg_inf, device=points.device, dtype=points.dtype) if has_qual else None

            if inst_masks.numel() == 0:
                logits_all.append(logits_b.float().cpu() if to_cpu else logits_b)
                if has_logits and in_logits_b is not None:
                    in_logits_all.append(in_logits_b.float().cpu() if to_cpu else in_logits_b)
                if has_cls and cls_logits_b is not None:
                    cls_logits_all.append(cls_logits_b.float().cpu() if to_cpu else cls_logits_b)
                if has_qual and cls_quality_b is not None:
                    cls_quality_all.append(cls_quality_b.float().cpu() if to_cpu else cls_quality_b)
                continue

            # Ensure iterable over instance masks (K_b, N)
            masks_iter = inst_masks if inst_masks.ndim > 1 else inst_masks.unsqueeze(0)

            k_out = 0
            for i, mask in enumerate(masks_iter):
                if k_out >= Q:
                    break

                if mask.dtype == torch.bool or log_threshold is None:
                    idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
                else:
                    idx = torch.nonzero(mask > log_threshold, as_tuple=False).squeeze(1)
                if idx.numel() == 0:
                    k_out += 1
                    continue

                # Select instance inputs (Nk, D) and batch query points (M, D)
                pts_sel = inputs[b][idx]
                pts_query = points[b]

                # Center (median) and robust scale (quantile 0.02-0.98, cube)
                c = pts_sel.median(dim=0).values
                q_low = torch.quantile(pts_sel, 0.02, dim=0)
                q_high = torch.quantile(pts_sel, 0.98, dim=0)
                s = (q_high - q_low).amax().clamp_min(1e-6)

                # Normalize
                pts_sel_norm = (pts_sel - c) / s
                pts_query_norm = (pts_query - c) / s

                # Cube crop with padding=1.0 -> bound = 1.0
                bound = 1.0
                m_inputs = (pts_sel_norm.abs() <= bound).all(dim=-1)
                m_points = (pts_query_norm.abs() <= bound).all(dim=-1)

                if not m_inputs.any() or not m_points.any():
                    k_out += 1
                    continue

                pts_k = pts_sel_norm[m_inputs].unsqueeze(0)
                pts_q = pts_query_norm[m_points].unsqueeze(0)

                # Encode instance inputs once, then decode query points in chunks
                # to avoid OOM on large per-instance grids (Pipeline loads 2 models).
                occ_kwargs = filter_dict(kwargs, remove={"feature"})
                pbs = occ_kwargs.pop("points_batch_size", None)
                feature_k = self.occ.encode(inputs=pts_k, **occ_kwargs)
                if pbs is not None and pts_q.size(1) > pbs:
                    chunks = torch.split(pts_q, pbs, dim=1)
                    logits_chunks = [
                        self.occ.decode(points=c, feature=feature_k, **occ_kwargs)["logits"] for c in chunks
                    ]
                    out_k: dict[str, Tensor] = dict(logits=torch.cat(logits_chunks, dim=1))
                else:
                    out_k = self.occ.decode(points=pts_q, feature=feature_k, **occ_kwargs)
                logits_k: Tensor = out_k["logits"]

                # Normalize shape to (Mk,)
                if logits_k.ndim == 3:
                    logits_k = logits_k[:, 0, :]
                if logits_k.ndim == 2:
                    logits_k = logits_k.squeeze(0)

                # Scatter occupancy
                logits_b[k_out, m_points] = logits_k.to(logits_b.dtype)

                # Scatter per-instance scores aligned with i (order from predict after filtering/NMS)
                if has_logits and src_in is not None and in_logits_b is not None and i < len(src_in):
                    in_logits_b[k_out] = src_in[i].float().cpu() if to_cpu else src_in[i].to(in_logits_b.dtype)
                if (
                    has_cls
                    and inst_scores_cls is not None
                    and cls_logits_b is not None
                    and b < len(inst_scores_cls)
                    and i < len(inst_scores_cls[b])
                ):
                    cls_logits_b[k_out] = (
                        inst_scores_cls[b][i].float().cpu() if to_cpu else inst_scores_cls[b][i].to(cls_logits_b.dtype)
                    )
                if (
                    has_qual
                    and inst_scores_qual is not None
                    and cls_quality_b is not None
                    and b < len(inst_scores_qual)
                    and i < len(inst_scores_qual[b])
                ):
                    cls_quality_b[k_out] = (
                        inst_scores_qual[b][i].float().cpu()
                        if to_cpu
                        else inst_scores_qual[b][i].to(cls_quality_b.dtype)
                    )
                k_out += 1

            logits_all.append(logits_b.float().cpu() if to_cpu else logits_b)
            if has_logits and in_logits_b is not None:
                in_logits_all.append(in_logits_b.float().cpu() if to_cpu else in_logits_b)
            if has_cls and cls_logits_b is not None:
                cls_logits_all.append(cls_logits_b.float().cpu() if to_cpu else cls_logits_b)
            if has_qual and cls_quality_b is not None:
                cls_quality_all.append(cls_quality_b.float().cpu() if to_cpu else cls_quality_b)

        out: dict[str, Tensor] = dict(logits=torch.stack(logits_all, dim=0))
        if has_logits:
            out["inputs.logits"] = torch.stack(in_logits_all, dim=0)
        if has_cls:
            out["cls_logits"] = torch.stack(cls_logits_all, dim=0)
        if has_qual:
            out["cls_quality"] = torch.stack(cls_quality_all, dim=0)
        return out

    def stream_instance_logits(
        self,
        inputs: Tensor,
        points: Tensor,
        data: dict[str, Tensor] | None = None,
        threshold: float = 0.5,
        points_batch_size: int | None = None,
        align_to_gt: bool = False,
        enable_grad: bool = False,
        **kwargs,
    ):
        with torch.set_grad_enabled(enable_grad):
            if align_to_gt:
                raise NotImplementedError("Streaming pipeline completion does not support align_to_gt=True")
            if points.ndim != 3 or points.size(0) != 1:
                raise NotImplementedError("Streaming pipeline completion currently supports batch size 1")

            masks = None
            if self.masks == "gt":
                label_source = kwargs.get("inputs.labels")
                if label_source is None and data is not None:
                    label_source = data.get("inputs.labels")
                if label_source is None:
                    raise ValueError("inputs.labels is required for oracle streaming pipeline completion.")
                masks = masks_from_labels(label_source)

            if masks is None:
                pred_out = self.inst.predict(
                    inputs=inputs,
                    method="cls",
                    threshold=threshold,
                    return_logits=True,
                    align_to_gt=False,
                    show=False,
                    **filter_dict(
                        kwargs,
                        remove={"inputs", "method", "threshold", "return_logits", "align_to_gt", "show"},
                    ),
                )
                if not isinstance(pred_out, tuple):
                    raise TypeError("Expected tuple output from instance predictor when return_logits=True.")
                masks = pred_out[0]

            log_threshold = math.log(threshold / (1.0 - threshold))
            inst_masks = masks[0]
            masks_iter = inst_masks if inst_masks.ndim > 1 else inst_masks.unsqueeze(0)
            occ_kwargs = filter_dict(kwargs, remove={"feature", "inputs", "points", "data", "masks"})

            instances: list[tuple[int, Tensor, Tensor, Tensor]] = []
            k_out = 0
            for mask in masks_iter:
                if k_out >= int(self.inst.num_objs):
                    break
                if mask.dtype == torch.bool:
                    idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
                else:
                    idx = torch.nonzero(mask > log_threshold, as_tuple=False).squeeze(1)
                if idx.numel() == 0:
                    k_out += 1
                    continue

                pts_sel = inputs[0][idx]
                c = pts_sel.median(dim=0).values
                q_low = torch.quantile(pts_sel, 0.02, dim=0)
                q_high = torch.quantile(pts_sel, 0.98, dim=0)
                s = (q_high - q_low).amax().clamp_min(1e-6)
                pts_sel_norm = (pts_sel - c) / s
                m_inputs = (pts_sel_norm.abs() <= 1.0).all(dim=-1)
                if not m_inputs.any():
                    k_out += 1
                    continue
                feature_k = self.occ.encode(inputs=pts_sel_norm[m_inputs].unsqueeze(0), **occ_kwargs)
                instances.append((k_out, c, s, feature_k))
                k_out += 1

            instance_indices = [inst[0] for inst in instances]
            yield {"instance_indices": instance_indices, "num_points": int(points.size(1))}
            if not instances:
                return

            points_batch_size = int(points_batch_size or points.size(1))
            neg_inf = -1e4
            for start in range(0, points.size(1), points_batch_size):
                end = min(start + points_batch_size, points.size(1))
                pts_query = points[0, start:end]
                logits_chunk = torch.full(
                    (len(instances), end - start),
                    neg_inf,
                    device=points.device,
                    dtype=points.dtype,
                )
                for row, (_slot, c, s, feature_k) in enumerate(instances):
                    pts_query_norm = (pts_query - c) / s
                    m_points = (pts_query_norm.abs() <= 1.0).all(dim=-1)
                    if not m_points.any():
                        continue
                    out_k = self.occ.decode(
                        points=pts_query_norm[m_points].unsqueeze(0), feature=feature_k, **occ_kwargs
                    )
                    logits_k: Tensor = out_k["logits"]
                    if logits_k.ndim == 3:
                        logits_k = logits_k[:, 0, :]
                    if logits_k.ndim == 2:
                        logits_k = logits_k.squeeze(0)
                    logits_chunk[row, m_points] = logits_k.to(logits_chunk.dtype)
                yield {
                    "start": start,
                    "end": end,
                    "instance_indices": instance_indices,
                    "logits": logits_chunk if enable_grad else logits_chunk.detach(),
                }

    def predict(self, inputs: Tensor, points: Tensor, data: dict[str, Tensor] | None = None, **kwargs) -> list[Tensor]:
        masks = None
        if self.masks == "gt":
            masks = masks_from_labels(kwargs["inputs.labels"])
        if data is None:
            data = dict()
        data.update(self(inputs=inputs, points=points, masks=masks, to_cpu=False, **kwargs))
        pred_out = self.inst.predict(inputs=inputs, points=points, data=data, **kwargs)
        if isinstance(pred_out, tuple):
            return pred_out[0]
        return pred_out

    def evaluate(self, data: dict[str, Tensor], **kwargs) -> dict[str, Any]:
        masks = None
        if self.masks == "gt":
            masks = masks_from_labels(data["inputs.labels"])

        if logger.isEnabledFor(DEBUG):
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
            m0 = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle).used
            start = time.perf_counter()

        data.update(self(inputs=data["inputs"], points=data["points"], masks=masks, **kwargs))

        if logger.isEnabledFor(DEBUG):
            torch.cuda.synchronize(self.device)
            self.inst.log("runtime", time.perf_counter() - start)
            m1 = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle).used
            self.inst.log("vram_delta", cast(int, m1) - cast(int, m0))
            self.inst.log("vram_peak_allocated", torch.cuda.max_memory_allocated(self.device))
            self.inst.log("vram_peak_reserved", torch.cuda.max_memory_reserved(self.device))

        return self.inst.evaluate(data, **kwargs)

    def loss(self, data: dict[str, Tensor], **kwargs) -> Tensor:
        raise NotImplementedError

    def get_log(self, key: str | None = None):
        return self.inst.get_log(key)

    def clear_log(self, key: str | None = None, ema: bool = False):
        return self.inst.clear_log(key, ema=ema)

    def on_validation_epoch_end(self, *args, **kwargs):
        return self.inst.on_validation_epoch_end(*args, **kwargs)
