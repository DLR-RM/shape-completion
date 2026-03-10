from typing import Any, Literal, cast, overload

import torch
from torch import Tensor, nn

from eval import eval_cls_seg, eval_occupancy
from utils import binary_from_multi_class

from .utils import classification_loss, probs_from_logits, regression_loss

EMPTY_EVAL_RESULTS_DICT = {
    "iou": 0.0,
    "f1": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "auprc": 0.0,
    "ece": 0.0,
    "npv": 0.0,
    "conf_pos": 0.0,
    "conf_neg": 0.0,
}

BatchValue = list[str] | Tensor
BatchDict = dict[str, BatchValue]
Reduction = Literal["mean", "sum", "none"]


def _require_tensor(data: BatchDict, key: str) -> Tensor:
    value = data.get(key)
    if not isinstance(value, Tensor):
        raise TypeError(f"Expected tensor at data['{key}']")
    return value


def _require_output_dict(output: object) -> dict[str, Tensor]:
    if not isinstance(output, dict):
        raise TypeError("Expected model output to be a dict[str, Tensor]")
    return cast(dict[str, Tensor], output)


def _normalize_reduction(reduction: str | None) -> Reduction:
    if reduction in {"mean", "sum", "none"}:
        return cast(Reduction, reduction)
    return "mean"


class MultiEvalMixin(nn.Module):
    @torch.no_grad()
    def evaluate(
        self,
        data: BatchDict,
        threshold: float = 0.5,
        regression: bool = False,
        reduction: str | None = "mean",
        prefix: str = "val/",
        metrics: list[str] | None = None,
        points_batch_size: int | None = None,
        return_logits: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        import torchmetrics.functional as M

        self_any = cast(Any, self)
        if "train" not in prefix and self.training:
            raise RuntimeError("Model should be in eval mode during evaluation")

        if not any("logits" in key for key in data.keys()):
            points = data.get("points")
            if points is not None:
                if not isinstance(points, Tensor):
                    raise TypeError("Expected tensor at data['points']")
                assert points.ndim == 3 and points.size(2) == 3, "Points must be of shape (B, N, 3)"
                if points_batch_size is None or points_batch_size >= points.size(1):
                    data.update(_require_output_dict(self_any(**data, **kwargs)))
                elif hasattr(self, "predict"):
                    data.update(
                        _require_output_dict(
                            self_any.predict(**data, points_batch_size=points_batch_size, key=None, **kwargs)
                        )
                    )
                else:
                    feature = self_any.encode(**data, **kwargs)
                    idx = torch.randperm(points.size(1))
                    p_split = torch.split(points[:, idx], points_batch_size, dim=1)
                    _out = [
                        _require_output_dict(self_any.decode(points=pi, feature=feature, **kwargs)) for pi in p_split
                    ]
                    data.update(
                        {k: torch.cat([o[k] for o in _out], dim=1)[:, torch.argsort(idx)] for k in _out[0].keys()}
                    )
            else:
                data.update(_require_output_dict(self_any(**data, **kwargs)))

        if "loss" in data:
            result: dict[str, Any] = {"loss": _require_tensor(data, "loss").mean().cpu().item()}
        else:
            result = {"loss": self_any.loss(data, regression=regression, reduction=reduction, **kwargs).cpu().item()}

        if "logits" in data or ("seg_logits" in data and "points.labels" in data):
            if "logits" in data:
                logits = _require_tensor(data, "logits").float()
                occ = _require_tensor(data, "points.occ").long() == 1
            else:
                seg_logits = _require_tensor(data, "seg_logits").float()
                free = seg_logits.size(-1) - 1
                labels = _require_tensor(data, "points.labels").long()
                occ = labels != free
                logits = binary_from_multi_class(seg_logits)

            if logits.numel() == 0:
                result.update(EMPTY_EVAL_RESULTS_DICT)
            else:
                probs = probs_from_logits(logits)
                pred_occ = probs >= threshold
                occ_result = eval_occupancy(probs, occ, threshold)
                occ_result = {
                    "iou": occ_result["iou"],
                    "f1": occ_result["f1"],
                    "precision": occ_result["precision"],  # occ[pred_occ], pos. predictive value
                    "recall": occ_result["recall"],
                    "auprc": cast(Tensor, M.average_precision(probs, occ, task="binary")).cpu().item(),
                    "ece": cast(Tensor, M.calibration_error(probs, occ, task="binary")).cpu().item(),
                    "brier": (probs - occ.float()).pow(2).mean().cpu().item(),
                    "npv": torch.nan_to_num(torch.nanmean((~occ[~pred_occ]).float()))
                    .cpu()
                    .item(),  # neg. pred. val. (neg. prec.)
                    "conf_pos": torch.nan_to_num(torch.nanmean(probs[pred_occ])).cpu().item(),
                    "conf_neg": torch.nan_to_num(torch.nanmean(1 - probs[~pred_occ])).cpu().item(),
                }
                result.update(occ_result)
            if return_logits:
                result["logits"] = logits

        if "cls_logits" in data:
            cls_logits = _require_tensor(data, "cls_logits").float()
            cls_labels = _require_tensor(data, "category.index")
            cls_metrics = None if metrics is None else [m for m in metrics if m.startswith("cls_")]
            cls_result = eval_cls_seg(cls_logits, cls_labels, cls_metrics, prefix="cls_")
            result.update(cls_result)
            if return_logits:
                result["cls_logits"] = cls_logits

        if "seg_logits" in data:
            seg_logits = _require_tensor(data, "seg_logits").float()
            B, N, C = seg_logits.shape
            seg_labels = (
                _require_tensor(data, "points.labels")
                if "points.labels" in data
                else _require_tensor(data, "inputs.labels")
            )
            seg_metrics = None if metrics is None else [m for m in metrics if m.startswith("seg_")]
            seg_result = eval_cls_seg(seg_logits.view(B * N, C), seg_labels.view(-1), seg_metrics, prefix="seg_")
            result.update(seg_result)
            if return_logits:
                result["seg_logits"] = seg_logits

        result = {f"{prefix}{k}": v for k, v in result.items()}
        return result


class MultiLossMixin(nn.Module):
    def loss(
        self,
        data: BatchDict,
        regression: bool = False,
        name: str | None = None,
        reduction: str | None = "mean",
        **kwargs,
    ) -> dict[str, Tensor]:
        self_any = cast(Any, self)
        reduction_value = _normalize_reduction(reduction)
        if not any("logits" in key for key in data.keys()):
            data.update(_require_output_dict(self_any(**data, **kwargs)))

        out: dict[str, Tensor] = dict()

        if "logits" in data:
            if regression:
                name = "l1" if name is None else name
                out["occ_loss"] = regression_loss(
                    _require_tensor(data, "logits"),
                    _require_tensor(data, "points.occ"),
                    tsdf=1.0 if "tsdf" in name else None,
                    name=name.replace("tsdf_", ""),
                    reduction=reduction_value,
                )
            else:
                out["occ_loss"] = classification_loss(
                    _require_tensor(data, "logits"), _require_tensor(data, "points.occ"), name, reduction_value
                )

        if "cls_logits" in data:
            out["cls_loss"] = classification_loss(
                _require_tensor(data, "cls_logits"), _require_tensor(data, "category.index"), name, reduction_value
            )

        if "seg_logits" in data:
            seg_logits = _require_tensor(data, "seg_logits")
            B, N, C = seg_logits.shape
            seg_labels = (
                _require_tensor(data, "points.labels")
                if "points.labels" in data
                else _require_tensor(data, "inputs.labels")
            )
            out["seg_loss"] = classification_loss(
                seg_logits.view(B * N, C), seg_labels.view(-1), reduction=reduction_value
            )

        return out


class PredictMixin(nn.Module):
    @torch.no_grad()
    @overload
    def predict(
        self,
        inputs: Tensor | None = None,
        points: Tensor | None = None,
        feature: Tensor | list[Tensor] | dict[str, Tensor] | None = None,
        key: None = None,
        points_batch_size: int | None = None,
        **kwargs,
    ) -> dict[str, Tensor]: ...

    @overload
    def predict(
        self,
        inputs: Tensor | None = None,
        points: Tensor | None = None,
        feature: Tensor | list[Tensor] | dict[str, Tensor] | None = None,
        key: str = "auto",
        points_batch_size: int | None = None,
        **kwargs,
    ) -> Tensor: ...

    def predict(
        self,
        inputs: Tensor | None = None,
        points: Tensor | None = None,
        feature: Tensor | list[Tensor] | dict[str, Tensor] | None = None,
        key: str | None = "auto",
        points_batch_size: int | None = None,
        **kwargs,
    ) -> Tensor | dict[str, Tensor]:
        assert not self.training, "Model should be in eval mode when predicting"
        assert inputs is not None or points is not None, "Either inputs or points must be provided"
        assert points is None or (points.ndim == 3 and points.size(2) == 3), "Points must be of shape (B, N, 3)"

        self_any = cast(Any, self)
        out: dict[str, Tensor]
        if feature is None and hasattr(self, "encoder") and isinstance(self.encoder, nn.Module):
            if points is None:
                out = _require_output_dict(self_any(inputs=inputs, **kwargs))
            else:
                if points_batch_size is None or points_batch_size >= points.size(1):
                    out = _require_output_dict(self_any(inputs=inputs, points=points, **kwargs))
                else:
                    feature = self_any.encode(inputs=inputs, **kwargs)
                    return self.predict(
                        inputs=inputs,
                        points=points,
                        feature=feature,
                        key=key,
                        points_batch_size=points_batch_size,
                        **kwargs,
                    )
        else:
            if points is None:
                raise ValueError("Points must be provided when using decoder path")
            if points_batch_size is None or points_batch_size >= points.size(1):
                out = _require_output_dict(self_any.decode(points=points, feature=feature, inputs=inputs, **kwargs))
            else:
                idx = torch.randperm(points.size(1))
                p_split = torch.split(points[:, idx], points_batch_size, dim=1)
                out_list = [
                    _require_output_dict(self_any.decode(points=pi, feature=feature, inputs=inputs, **kwargs))
                    for pi in p_split
                ]
                out = {
                    k: torch.cat([o[k] for o in out_list], dim=1)[:, torch.argsort(idx)]
                    for k in list(set(("logits", "seg_logits")) & set(out_list[0].keys()))
                }
                if "cls_logits" in out_list[0].keys():
                    out["cls_logits"] = out_list[0]["cls_logits"]
        if key is None:
            return out
        if key == "auto":
            key = "logits" if "logits" in out else "cls_logits" if "cls_logits" in out else "seg_logits"
        value = out.get(key)
        if value is None:
            raise KeyError(f"Prediction key '{key}' not found in output")
        return value.float()
