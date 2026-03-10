from __future__ import annotations

import torch

from ..src.point_transformer import PointTransformer


def test_loss():
    model = PointTransformer(occupancy=True, seg_num_classes=50, pcd_enc_type="linear")
    assert hasattr(model, "seg_head")
    assert model.seg_head is not None
    assert model.seg_head.weight.shape[0] == 51
    data: dict[str, torch.Tensor | list[str]] = {
        "inputs": torch.randn(2, 50, 3),
        "points": torch.randn(2, 100, 3),
        "points.labels": torch.randint(0, 51, (2, 100)).float(),
    }
    loss = model.loss(data)
    assert loss >= 0.0


def test_evaluate():
    model = PointTransformer(pcd_enc_type="linear").eval()
    data: dict[str, torch.Tensor | list[str]] = {
        "inputs": torch.randn(2, 100, 3),
        "points": torch.randn(2, 100, 3),
        "points.occ": torch.randint(0, 2, (2, 100)).float(),
    }
    result = model.evaluate(data)
    assert result["val/loss"] >= 0.0
    assert 0 <= result["val/f1"] <= 1.0

    model = PointTransformer(occupancy=False, cls_num_classes=57, pcd_enc_type="linear").eval()
    data = {"inputs": torch.randn(10, 100, 3), "category.index": torch.randint(0, 57, (10,)).float()}
    result = model.evaluate(data, metrics=["cls_acc_micro"])
    assert result["val/loss"] >= 0.0
    assert 0 <= result["val/cls_acc_micro"] <= 1.0

    model = PointTransformer(occupancy=True, seg_num_classes=50, pcd_enc_type="linear").eval()
    data = {
        "inputs": torch.randn(2, 50, 3),
        "points": torch.randn(2, 100, 3),
        "points.labels": torch.randint(0, 51, (2, 100)).float(),
    }
    result = model.evaluate(data, metrics=["seg_iou_weighted"])
    assert result["val/loss"] >= 0.0
    assert 0 <= result["val/iou"] <= 1.0

    model = PointTransformer(occupancy=False, seg_num_classes=50, pcd_enc_type="linear").eval()
    data = {"inputs": torch.randn(2, 100, 3), "inputs.labels": torch.randint(0, 50, (2, 100)).float()}
    result = model.evaluate(data, metrics=["seg_iou_weighted"])
    assert result["val/loss"] >= 0.0
    assert 0 <= result["val/seg_iou_weighted"] <= 1.0
