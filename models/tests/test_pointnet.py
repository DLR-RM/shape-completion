import torch

from ..src.pointnet import PointNetCls, PointNetSeg


class TestPointNetCls:
    def test_init(self):
        PointNetCls(num_classes=57)

    def test_forward(self):
        model = PointNetCls(num_classes=57)
        x = torch.rand(32, 2048, 3)
        y = model(x)[0]
        assert y.shape == (32, 57)

    def test_loss(self):
        model = PointNetCls(num_classes=57)
        x = torch.rand(32, 2048, 3)
        y = torch.randint(0, 57, (32,))
        loss = model.loss({"inputs": x, "category.index": y})
        assert loss.shape == ()

    def test_evaluate(self):
        model = PointNetCls(num_classes=57)
        x = torch.rand(32, 2048, 3)
        y = torch.randint(0, 57, (32,))
        metrics = model.evaluate({"inputs": x, "category.index": y}, metrics=["acc_micro"])
        assert metrics["val/loss"] >= 0.0
        assert 0 <= metrics["val/cls_acc_micro"] <= 1.0


class TestPointNetSeg:
    def test_init(self):
        PointNetSeg(num_classes=50)

    def test_forward(self):
        model = PointNetSeg(num_classes=50)
        x = torch.rand(32, 2048, 3)
        y = model(x)[0]
        assert y.shape == (32, 2048, 50)

    def test_loss(self):
        model = PointNetSeg(num_classes=50)
        x = torch.rand(32, 2048, 3)
        y = torch.randint(0, 50, (32, 2048))
        loss = model.loss({"inputs": x, "inputs.labels": y})
        assert loss.shape == ()

    def test_evaluate(self):
        model = PointNetSeg(num_classes=50)
        x = torch.rand(32, 2048, 3)
        y = torch.randint(0, 50, (32, 2048))
        metrics = model.evaluate({"inputs": x, "inputs.labels": y}, metrics=["iou_weighted"])
        assert metrics["val/loss"] >= 0.0
        assert 0 <= metrics["val/seg_iou_weighted"] <= 1.0
