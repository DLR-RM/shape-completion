import torch

from ..src.onet import ONet


def test_forward_smoke():
    model = ONet(inputs_type="pointcloud").eval()
    inputs = torch.randn(2, 128, 3)
    points = torch.randn(2, 64, 3)
    out = model(inputs=inputs, points=points)
    assert "logits" in out
    assert out["logits"].shape == (2, 64)


def test_loss_smoke():
    model = ONet(inputs_type="pointcloud")
    data: dict[str, torch.Tensor | list[str]] = {
        "inputs": torch.randn(2, 128, 3),
        "points": torch.randn(2, 64, 3),
        "points.occ": torch.randint(0, 2, (2, 64)).float(),
    }
    loss = model.loss(data)
    assert torch.is_tensor(loss)
    assert loss.ndim == 0
