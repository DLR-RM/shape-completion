from typing import Any, Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from eval import eval_cls_seg
from libs import furthest_point_sample
from utils import coordinates_to_index, points_to_coordinates, setup_logger

from .mixins import MultiEvalMixin, MultiLossMixin, PredictMixin
from .model import Model
from .transformer import Decoder, Encoder, MixerBlock, NeRFEncoding, PositionalEncoding, PyTorchDecoder, PyTorchEncoder
from .utils import classification_loss, reduce_loss

logger = setup_logger(__name__)

Reduction = Literal["none", "mean", "sum"]


def _debug_level_1(message: str) -> None:
    debug_level_1 = getattr(cast(Any, logger), "debug_level_1", None)
    if callable(debug_level_1):
        debug_level_1(message)
        return
    logger.debug(message)


def _require_tensor(value: Tensor | list[str] | None, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"Expected tensor for `{name}`")
    return value


def _normalize_reduction(reduction: str | None) -> Reduction:
    if reduction in {"none", "sum", "mean"}:
        return cast(Reduction, reduction)
    return "mean"


def get_enc_dec(
    n_embd: int = 512,
    n_layer: int = 8,
    n_head: int = 8,
    dropout: float = 0,
    bias: bool = False,
    enc_type: str = "encoder",
    dec_type: str = "decoder",
    implementation: str = "karpathy",
    occupancy: bool = True,
    use_linear_attn: bool = False,
) -> tuple[nn.Module, nn.Module]:
    decoder: nn.Module = nn.Identity()
    if implementation == "torch":
        if enc_type is None:
            encoder = nn.Identity()
        elif "enc" in enc_type:
            encoder = PyTorchEncoder(n_layer, n_embd, n_head, bias, dropout)
        else:
            raise NotImplementedError(f"Unknown encoder type: {enc_type}")
        if occupancy:
            if dec_type is None:
                decoder = nn.Identity()
            elif "enc" in dec_type:
                decoder = PyTorchEncoder(n_layer, n_embd, n_head, bias, dropout)
            elif "dec" in dec_type:
                decoder = PyTorchDecoder(n_layer, n_embd, n_head, bias, dropout)
            elif dec_type == "cross_attn":
                raise NotImplementedError("Cross-Attention decoder not supported with PyTorch")
            else:
                raise NotImplementedError(f"Unknown decoder type: {dec_type}")
    else:
        if enc_type is None:
            encoder = nn.Identity()
        elif "enc" in enc_type:
            encoder = Encoder(n_layer, n_embd, n_head, bias, dropout, linear=use_linear_attn)
        else:
            raise NotImplementedError(f"Unknown encoder type: {enc_type}")
        if occupancy:
            if dec_type is None:
                decoder = nn.Identity()
            elif "enc" in dec_type:
                decoder = Encoder(
                    n_layer,
                    n_embd,
                    n_head,
                    bias,
                    dropout,
                    # no_self_attn=False,
                    linear=use_linear_attn,
                )
            elif "dec" in dec_type:
                decoder = Decoder(n_layer, n_embd, n_head, bias, dropout, linear=use_linear_attn)
            elif dec_type == "mixer":
                decoder = nn.Sequential(*[MixerBlock(n_embd, 4096) for _ in range(n_layer)])
            else:
                raise NotImplementedError(f"Unknown decoder type: {dec_type}")
    return encoder, decoder


def get_pos_enc(
    dim: int = 3, n_embd: int = 512, dropout: float = 0, shared_pos_enc: bool = True, pos_enc_type: str | None = None
) -> nn.Module | None:
    if pos_enc_type is None:
        return None
    else:
        if shared_pos_enc:
            _debug_level_1(f"Using shared {pos_enc_type} positional encoding")
            if pos_enc_type == "linear":
                pos_enc = nn.Linear(dim, n_embd)
            elif pos_enc_type == "embedding":
                pos_enc = nn.Embedding(32**3, 1)  # Limits res. to 32^3
            elif pos_enc_type == "vanilla":
                _ = dropout
                pos_enc = PositionalEncoding(n_embd, max_len=64**3)  # Limits res. to 64^3
            else:
                raise NotImplementedError(f"Unknown positional encoding type: {pos_enc_type}")
        else:
            _debug_level_1("Using separate positional embedding")
            pos_enc = nn.ModuleList([nn.Linear(dim, n_embd), nn.Linear(dim, n_embd)])
    return pos_enc


def get_pcd_enc(
    dim: int = 3,
    n_embd: int = 512,
    pcd_enc_type: str | None = "nerf",
    pos_enc_type: str | None = None,
    padding: float = 0.1,
) -> nn.Module:
    if pcd_enc_type == "linear":
        enc = nn.Linear(dim, n_embd - 1 if pos_enc_type == "embedding" else n_embd)
    elif pcd_enc_type == "nerf":
        enc = NeRFEncoding(in_dim=dim, out_dim=n_embd - 1 if pos_enc_type == "embedding" else n_embd, padding=padding)
    else:
        raise NotImplementedError(f"Unknown point cloud encoding type: {pcd_enc_type}")
    return enc


class PointTransformer(MultiEvalMixin, MultiLossMixin, PredictMixin, Model):
    def __init__(
        self,
        dim: int = 3,
        n_embd: int = 512,
        n_layer: int = 8,
        n_head: int = 8,
        dropout: float = 0,
        bias: bool = False,
        enc_type: str = "encoder",
        dec_type: str = "decoder",
        pcd_enc_type: str | None = "nerf",
        shared_pcd_enc: bool = False,
        pos_enc_type: str | None = None,
        shared_pos_enc: bool = True,
        implementation: str = "karpathy",
        padding: float = 0.1,
        occupancy: bool = True,
        aux_loss: bool = False,
        supervise_inputs: str | None = None,
        cls_num_classes: int | None = None,
        seg_num_classes: int | None = None,
        use_linear_attn: bool = False,
        n_fps: int | None = None,
    ):
        super().__init__()
        assert enc_type in [None, "encoder"] or "enc" in enc_type
        assert dec_type in [None, "cross_attn", "encoder", "decoder", "mixer"] or "enc" in dec_type or "dec" in dec_type
        assert pos_enc_type in [None, "linear", "embedding", "vanilla"] or (
            pos_enc_type is not None and "emb" in pos_enc_type
        )
        assert pcd_enc_type in [None, "linear", "nerf", "cross_attn"]
        assert not (not shared_pos_enc and pos_enc_type is not None and "embd" in pos_enc_type)
        assert implementation in ["torch", "xformers", "karpathy", "nanogpt"]

        if use_linear_attn and implementation in ["torch", "xformers"]:
            logger.warning(f"Linear attention not supported with the {implementation} implementation. Switching.")
            implementation = "karpathy"

        self.dim = dim
        self.padding = padding
        self.occupancy = occupancy
        self.implementation = implementation
        self.enc_type = enc_type
        self.dec_type = dec_type
        self.supervise_inputs = supervise_inputs
        self.n_fps = 1024 if pcd_enc_type == "cross_attn" and n_fps is None else n_fps

        self.encoder, self.decoder = get_enc_dec(
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            dropout=dropout,
            bias=bias,
            enc_type=enc_type,
            dec_type=dec_type,
            implementation=implementation,
            occupancy=occupancy,
            use_linear_attn=use_linear_attn,
        )

        self.pos_enc_type = pos_enc_type
        self.shared_pos_enc = shared_pos_enc
        self.pos_enc = get_pos_enc(
            dim=dim, n_embd=n_embd, dropout=dropout, shared_pos_enc=shared_pos_enc, pos_enc_type=pos_enc_type
        )

        self.in_enc = get_pcd_enc(
            dim=dim, n_embd=n_embd, pcd_enc_type=pcd_enc_type, pos_enc_type=pos_enc_type, padding=padding
        )

        self.seg_head = None
        self.aux_head = None
        if occupancy:
            if shared_pcd_enc:
                _debug_level_1("Using shared point cloud encoding")
                self.query_enc = self.in_enc
            else:
                _debug_level_1("Using separate point cloud encoding")
                self.query_enc = get_pcd_enc(
                    dim=dim, n_embd=n_embd, pcd_enc_type=pcd_enc_type, pos_enc_type=pos_enc_type, padding=padding
                )

            if seg_num_classes is None:
                self.occ_head = nn.Linear(n_embd, 1)
            else:
                self.seg_head = nn.Linear(n_embd, seg_num_classes + 1)
                if aux_loss:
                    self.aux_head = nn.Linear(n_embd, seg_num_classes)
        elif seg_num_classes is not None:
            self.seg_head = nn.Linear(n_embd, seg_num_classes)

        self.cls_head = None
        if cls_num_classes is not None:
            self.cls_head = nn.Linear(n_embd, cls_num_classes)

    def forward(self, inputs: Tensor, points: Tensor | None = None, **kwargs) -> dict[str, Tensor]:
        inputs_feat = self.encode(inputs, **kwargs)
        if "enc" not in self.dec_type and self.supervise_inputs:
            if points is None:
                raise ValueError("`points` are required when supervise_inputs is enabled")
            points = torch.cat((points, inputs), dim=1)
        return self.decode(points, inputs_feat)

    def encode(self, inputs: Tensor, **kwargs) -> Tensor:
        inputs_embd = self.in_enc(furthest_point_sample(inputs, self.n_fps)) if self.n_fps else self.in_enc(inputs)

        if self.pos_enc is not None:
            if isinstance(self.pos_enc, (nn.Embedding, PositionalEncoding)):
                coords = points_to_coordinates(inputs, max_value=1 + self.padding)
                if not isinstance(coords, Tensor):
                    raise TypeError("points_to_coordinates(inputs) must return a tensor")
                if isinstance(self.pos_enc, nn.Embedding):
                    index = coordinates_to_index(coords, 32)
                    inputs_embd = torch.cat((inputs_embd, self.pos_enc(index)), dim=2)
                else:
                    index = coordinates_to_index(coords, 64)
                    inputs_embd = self.pos_enc(inputs_embd, index)
            else:
                if self.shared_pos_enc:
                    pos_enc = cast(nn.Module, self.pos_enc)
                    inputs_embd += cast(Tensor, pos_enc(inputs))
                else:
                    pos_enc = cast(nn.ModuleList, self.pos_enc)
                    inputs_embd += cast(Tensor, pos_enc[0](inputs))

        features = self.encoder(inputs_embd)
        return features

    def get_points_feat(self, points: Tensor, feature: Tensor) -> Tensor:
        points_embd = self.query_enc(points)
        if self.pos_enc is not None:
            if isinstance(self.pos_enc, (nn.Embedding, PositionalEncoding)):
                coords = points_to_coordinates(points, max_value=1 + self.padding)
                if not isinstance(coords, Tensor):
                    raise TypeError("points_to_coordinates(points) must return a tensor")
                if isinstance(self.pos_enc, nn.Embedding):
                    index = coordinates_to_index(coords, 32)
                    points_embd = torch.cat((points_embd, self.pos_enc(index)), dim=2)
                else:
                    index = coordinates_to_index(coords, 64)
                    points_embd = self.pos_enc(points_embd, index)
            else:
                if self.shared_pos_enc:
                    pos_enc = cast(nn.Module, self.pos_enc)
                    points_embd += cast(Tensor, pos_enc(points))
                else:
                    pos_enc = cast(nn.ModuleList, self.pos_enc)
                    points_embd += cast(Tensor, pos_enc[1](points))

        if "enc" in self.dec_type or self.dec_type == "mixer":
            points_feat = self.decoder(torch.cat((points_embd, feature), dim=1))
        else:
            points_feat = self.decoder(points_embd, feature)

        if self.training:
            return points_feat
        return points_feat[:, : points.size(1)]

    def decode(self, points: Tensor | None, feature: Tensor, **kwargs) -> dict[str, Tensor]:
        result: dict[str, Tensor] = dict()
        if self.occupancy:
            if points is None:
                raise ValueError("`points` are required for occupancy decoding")
            points_feat = self.get_points_feat(points, feature)
            if not self.supervise_inputs:
                points_feat = points_feat[:, : points.size(1)]
            if self.cls_head is not None:
                result["cls_logits"] = self.cls_head(points_feat.mean(dim=1))
            if self.seg_head is None:
                logits = self.occ_head(points_feat).squeeze(-1)
                result["logits"] = logits
            else:
                result["seg_logits"] = self.seg_head(points_feat)
                if self.aux_head is not None:
                    result["aux_logits"] = self.aux_head(feature)
        else:
            if self.cls_head is not None:
                result["cls_logits"] = self.cls_head(feature.mean(dim=1))
            if self.seg_head is not None:
                result["seg_logits"] = self.seg_head(feature)
        return result

    @torch.no_grad()
    def evaluate(
        self,
        data: dict[str, list[str] | Tensor],
        threshold: float = 0.5,
        regression: bool = False,
        reduction: str | None = "mean",
        prefix: str = "val/",
        metrics: list[str] | None = None,
        points_batch_size: int | None = None,
        **kwargs,
    ) -> dict[str, float]:
        result = super().evaluate(data, threshold, regression, reduction, prefix, metrics, points_batch_size, **kwargs)

        aux_logits = data.get("aux_logits")
        if isinstance(aux_logits, Tensor):
            B, N, C = aux_logits.shape
            seg_labels = _require_tensor(data.get("inputs.labels"), "inputs.labels")
            seg_metrics = None if metrics is None else [m for m in metrics if m.startswith("seg_")]
            aux_result = eval_cls_seg(aux_logits.view(B * N, C), seg_labels.view(-1), seg_metrics, prefix="seg_")
            aux_result = {k.replace("seg_", "aux_"): v for k, v in aux_result.items()}
            result.update(cast(dict[str, float], aux_result))

        return result

    def loss(
        self,
        data: dict[str, list[str] | Tensor],
        regression: bool = False,
        name: str | None = None,
        reduction: str | None = "mean",
        **kwargs,
    ) -> Tensor:
        reduction_name = _normalize_reduction(self.reduction if reduction is None else reduction)

        if self.supervise_inputs:
            logits = data.get("logits")
            if logits is None:
                logits = self(**data, **kwargs)["logits"]
            logits = _require_tensor(logits, "logits")
            targets = _require_tensor(data.get("points.occ"), "points.occ")

            volume_logits = logits[:, : targets.size(1)]
            loss = F.binary_cross_entropy_with_logits(volume_logits, targets, reduction="none")

            if self.training:
                surface_logits = logits[:, targets.size(1) :]
                if self.supervise_inputs == "bce":
                    surface_loss = F.binary_cross_entropy_with_logits(
                        surface_logits, torch.ones_like(surface_logits), reduction="none"
                    )
                elif self.supervise_inputs == "mse":
                    surface_loss = F.mse_loss(surface_logits, torch.zeros_like(surface_logits), reduction="none")
                elif self.supervise_inputs == "l1":
                    surface_loss = surface_logits.abs()
                else:
                    raise ValueError(f"Unknown inputs supervision method: {self.supervise_inputs}")

                loss = torch.cat([loss, surface_loss], dim=1)
            return reduce_loss(loss, reduction_name)

        out = super().loss(data, regression, name, reduction_name)

        occ_loss = out.get("occ_loss")
        if occ_loss is None:
            loss = torch.zeros((), device=next(self.parameters()).device)
        else:
            loss = occ_loss

        cls_weight_raw = data.get("cls_weight", 1.0)
        cls_weight = cls_weight_raw if isinstance(cls_weight_raw, (float, int)) else 1.0
        seg_default = 0.5 if "aux_logits" in out else 1.0
        seg_weight_raw = data.get("seg_weight", seg_default)
        seg_weight = seg_weight_raw if isinstance(seg_weight_raw, (float, int)) else seg_default

        cls_loss = out.get("cls_loss")
        if cls_loss is not None:
            loss = loss + float(cls_weight) * cls_loss
        seg_loss = out.get("seg_loss")
        if seg_loss is not None:
            loss = loss + float(seg_weight) * seg_loss

        if "aux_logits" in out:
            seg_logits = out["aux_logits"]
            B, N, C = seg_logits.shape
            seg_labels = _require_tensor(data.get("inputs.labels"), "inputs.labels")
            aux_loss = classification_loss(seg_logits.view(B * N, C), seg_labels.view(-1), name, reduction_name)
            loss += 0.5 * aux_loss

        return loss
