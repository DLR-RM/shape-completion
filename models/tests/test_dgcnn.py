import torch

from ..src.dgcnn import DGCNN_cls, DGCNN_partseg, get_graph_feature


def test_get_graph_feature_cpu_shape_and_device():
    x = torch.randn(2, 3, 32)
    out = get_graph_feature(x, k=8)

    assert out.shape == (2, 6, 32, 8)
    assert out.device == x.device


def test_dgcnn_partseg_forward_grid_smoke():
    model = DGCNN_partseg(c_dim=16, hidden_dim=8, grid_resolution=4, feature_type=("grid",), k=8, scatter_type="mean")
    points = torch.randn(2, 32, 3)

    out = model(points)

    assert "grid" in out
    assert out["grid"].shape == (2, 16, 4, 4, 4)


def test_dgcnn_cls_forward_grid_smoke():
    model = DGCNN_cls(c_dim=32, hidden_dim=None, grid_resolution=4, feature_type=("grid",), k=8, scatter_type="mean")
    points = torch.randn(2, 3, 32)

    out = model(points)

    assert "grid" in out
    assert out["grid"].shape == (2, 32)
